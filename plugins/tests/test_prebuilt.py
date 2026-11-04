"""What a pip install relies on, on the CPU (docs/dev/v1-pip.md D5, D-H): an overlay's prebuilt modules are found
before the cache and never rebuilt, a torch they do not fit is refused with the fix, every Triton user checks for a C
toolchain, and a prebuild of several SMs builds each in its own process."""
import pathlib
import re
import subprocess
import sysconfig

import pytest
import torch
from holon.api import HostError
from holon_plugins import host

PLUGINS = pathlib.Path(host.__file__).parent
EXTENSIONS = ("ggml_cuda", "expert_io", "promote", "state_copy", "cpu_task")


def _ext(name):
    return __import__(f"holon_plugins.kernels.{name}.extension", fromlist=["x"])


def test_the_c_toolchain_check_names_what_is_missing_and_the_fix(monkeypatch):
    check, include = host.require_c_toolchain.__wrapped__, sysconfig.get_paths()["include"]     # uncached
    check()                                                   # this host has gcc and Python's headers
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.setattr(host.shutil, "which", lambda _: None)
    with pytest.raises(HostError, match=r"no C compiler.*apt install build-essential python3.12-dev"):
        check()
    monkeypatch.setattr(host.shutil, "which", lambda c: "/usr/bin/gcc" if c == "gcc" else None)
    monkeypatch.setattr(host.sysconfig, "get_paths", lambda scheme: {"include": "/nonexistent"})
    with pytest.raises(HostError, match=r"no Python.h in /nonexistent.*uv tool install"):
        check()
    monkeypatch.setattr(host.sysconfig, "get_paths", lambda scheme: {"include": include})
    monkeypatch.setenv("CC", "/nonexistent/cc")                           # a $CC that is not there: refused, no retry
    with pytest.raises(HostError, match=r"\(/nonexistent/cc\) that cannot build a Python extension: error: .*nonexist"):
        check()
    monkeypatch.setenv("CC", "ccache gcc")                                # one program, as Triton runs it
    with pytest.raises(HostError, match=r"\(ccache gcc\) that cannot build"):
        check()


def test_only_a_cuda_13_nvcc_counts_as_a_toolkit(tmp_path, monkeypatch):
    for v in ("CUDA_HOME", "CUDA_PATH"):
        monkeypatch.delenv(v, raising=False)
    nvcc = tmp_path / "nvcc"
    monkeypatch.setattr(host.shutil, "which", lambda name: str(nvcc) if name == "nvcc" else None)
    monkeypatch.setattr(host.pathlib.Path, "is_file", lambda p: str(p) == str(nvcc) and p.exists())
    for release, ok in (("12.4", False), (torch.version.cuda, True)):
        nvcc.write_text(f"#!/bin/sh\necho 'Cuda compilation tools, release {release}, V{release}.1'\n")
        nvcc.chmod(0o755)
        host.toolkit.cache_clear()
        assert host.toolkit() is ok, release
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "runtime-only"))      # set: the only place torch and FlashInfer look
    host.toolkit.cache_clear()
    assert host.toolkit() is False
    host.toolkit.cache_clear()


def test_every_triton_user_checks_the_c_toolchain_at_load():
    """The plugins that import Triton (found by grep, not by family) call the check in `register` (rev 2.1, A-R2)."""
    users = {p.parent.relative_to(PLUGINS).parts[:2] for p in PLUGINS.rglob("*.py")
             if "vendor" not in p.parts and re.search(r"^\s*(import triton|from triton)", p.read_text(), re.M)}
    assert users
    for parts in users:
        text = (PLUGINS.joinpath(*parts) / "plugin.py").read_text()
        register = re.search(r"\n    def register\(.*?(?=\n    def |\Z)", text, re.S)
        assert register and "host.require_c_toolchain()" in register.group(0), parts


@pytest.mark.parametrize("name", EXTENSIONS)
def test_a_shipped_module_is_found_first_and_never_rebuilt(name, tmp_path, monkeypatch):
    ext = _ext(name)
    monkeypatch.setattr(ext, "SHIPPED", tmp_path)
    assert not ext.module_path(86).is_relative_to(tmp_path)                 # nothing shipped: the user cache
    shipped = tmp_path / ext.key(86) / f"{ext.NAME}.so"
    shipped.parent.mkdir()
    shipped.write_bytes(b"")
    assert ext.module_path(86) == shipped
    assert (ext.build(86, 1) if name == "ggml_cuda" else ext.build(86)) == shipped   # no compiler run


def test_a_torch_the_shipped_modules_do_not_fit_is_refused_with_the_fix(tmp_path, monkeypatch):
    ext = _ext("expert_io")
    monkeypatch.setattr(ext, "SHIPPED", tmp_path)
    want = ext.key(120)
    assert "torch" + torch.__version__ in want
    (tmp_path / want.replace(f"torch{torch.__version__}", "torch2.11.0+cu130x")).mkdir()       # another torch build
    with pytest.raises(HostError, match=r"torch is .*install --force-reinstall --no-deps \"torch==2.11.0\+cu130\""):
        ext.load(120)
    for d in tmp_path.iterdir():
        d.rmdir()
    (tmp_path / (want[:-16] + "0" * 16)).mkdir()                            # this torch, other sources: another version
    with pytest.raises(HostError, match=r"another Holon version.*force-reinstall holonlm=="):
        ext.load(120)
    (tmp_path / (want[:-16] + "0" * 16)).rmdir()
    (tmp_path / ext.key(86)).mkdir()                                        # another SM only: the add-on, always,
    monkeypatch.setattr(host, "toolkit", lambda: True)                      # and with a CUDA 13 toolkit the prebuild
    with pytest.raises(HostError, match=r"\(sm 120\) are not installed: `pip install holonlm-cuda-sm120==.*source "
                                        r"install builds them with: python -m holon_plugins.kernels.expert_io"):
        ext.load(120)
    monkeypatch.setattr(host, "toolkit", lambda: False)
    with pytest.raises(HostError, match=r"\(sm 120\) are not installed: `pip install holonlm-cuda-sm120==[^;]*$"):
        ext.load(120)
    with pytest.raises(HostError, match=r"this GPU is sm 80, and Holon's prebuilt modules cover sm 86, 89, 120"):
        ext.load(80)


@pytest.mark.parametrize("name", EXTENSIONS)
def test_a_prebuild_of_several_sms_builds_each_in_its_own_process(name, monkeypatch):
    """torch's loader renames a module built twice in one process (`<name>_v1`), never found by `load` (v1-pip F8)."""
    prebuild = __import__(f"holon_plugins.kernels.{name}.prebuild", fromlist=["x"])
    runs = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: runs.append(cmd))
    monkeypatch.setattr(prebuild.extension, "build", lambda *a: pytest.fail("built in this process"))
    prebuild.main(["--sm", "86", "120"])
    assert [cmd[cmd.index("--sm") + 1:cmd.index("--sm") + 2] for cmd in runs] == [["86"], ["120"]]


def test_flashinfer_reads_holon_s_modules_for_one_sm(tmp_path, monkeypatch, capsys):
    aot = pytest.importorskip("holon_plugins.kernels.flashinfer.aot")
    monkeypatch.setattr(aot, "PREBUILT", tmp_path / "prebuilt")
    monkeypatch.setattr(host, "toolkit", lambda: True)
    monkeypatch.setattr(aot.env, "FLASHINFER_JIT_DIR", tmp_path / "jit")
    monkeypatch.setattr(aot.env, "FLASHINFER_AOT_DIR", pathlib.Path("/elsewhere"))
    needed, listed = ["rope", "silu_and_mul"], []                           # the modules Holon needs, by name
    monkeypatch.setattr(aot.prebuild, "specs", lambda sm: listed.append(sm) or [type("S", (), {"name": n})
                                                                               for n in needed])
    capability = {"cuda:0": (8, 6), "cuda:1": (8, 9)}
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d: capability[d])

    def bind(device):                                                       # a fresh process's first bind
        aot.directory.cache_clear(), aot._bind.cache_clear()
        aot.bind(device)
    bind("cuda:0")                           # set even with no modules shipped: never a foreign jit-cache's
    aot.bind("cuda:0")                       # every op's constructor binds: checked once a process and SM
    assert listed == [86] and "compiles each at first use, once" in capsys.readouterr().err
    assert aot.env.FLASHINFER_AOT_DIR == aot.directory(86) and aot.directory(86).name.endswith(
        f"-cu{torch.version.cuda}-sm86") and "-tvmffi" in aot.directory(86).name
    with pytest.raises(HostError, match="one GPU architecture a process"):
        aot.bind("cuda:1")
    monkeypatch.setattr(aot.env, "FLASHINFER_AOT_DIR", pathlib.Path("/elsewhere"))
    foreign = aot.directory(89).name.replace(f"-cu{torch.version.cuda}-", "-cu12.8-")    # another torch's add-on
    (tmp_path / "prebuilt" / foreign).mkdir(parents=True)
    with pytest.raises(HostError, match="prebuilt modules are for CUDA 13.0"):
        bind("cuda:1")
    for n in needed:                                                        # the user built every one Holon needs:
        (tmp_path / "jit" / n).mkdir(parents=True)                          # theirs serve, as the extensions' cache
        (tmp_path / "jit" / n / f"{n}.so").write_bytes(b"")
    bind("cuda:1")
    monkeypatch.setattr(aot.env, "FLASHINFER_AOT_DIR", pathlib.Path("/elsewhere"))
    (tmp_path / "jit" / "rope" / "rope.so").unlink()                        # one missing: not theirs, so refused
    with pytest.raises(HostError, match="prebuilt modules are for CUDA 13.0"):
        bind("cuda:1")
    for d in (tmp_path / "prebuilt").glob("*-cu12.8-*"):
        d.rmdir()
    monkeypatch.setattr(host, "toolkit", lambda: False)                     # no add-on module, no toolkit: refused
    with pytest.raises(HostError, match=r"\(sm 89\) are not installed: `pip install holonlm-cuda-sm89=="):
        bind("cuda:1")
    aot.directory.cache_clear(), aot._bind.cache_clear()
