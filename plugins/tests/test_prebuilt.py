"""What a pip install relies on, on the CPU (docs/dev/v1-pip.md D5, D-H): an overlay's prebuilt modules are found
before the cache and never rebuilt, a torch they do not fit is refused with the fix, every Triton user checks for a C
toolchain, and a prebuild of several SMs builds each in its own process."""
import pathlib
import re
import subprocess

import pytest
import torch
from holon.api import HostError
from holon_plugins import host

PLUGINS = pathlib.Path(host.__file__).parent
EXTENSIONS = ("ggml_cuda", "expert_io", "promote", "state_copy", "cpu_task")


def _ext(name):
    return __import__(f"holon_plugins.kernels.{name}.extension", fromlist=["x"])


def test_the_c_toolchain_check_names_what_is_missing_and_the_fix(monkeypatch):
    check = host.require_c_toolchain.__wrapped__              # uncached
    check()                                                   # this host has gcc and Python's headers
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.setattr(host.shutil, "which", lambda _: None)
    with pytest.raises(HostError, match=r"no C compiler.*apt install build-essential python3.12-dev"):
        check()
    monkeypatch.setattr(host.shutil, "which", lambda c: "/usr/bin/gcc" if c == "gcc" else None)
    monkeypatch.setattr(host.sysconfig, "get_paths", lambda scheme: {"include": "/nonexistent"})
    with pytest.raises(HostError, match=r"no Python.h in /nonexistent.*uv tool install"):
        check()


def test_every_triton_user_checks_the_c_toolchain_at_load():
    """The plugins that import Triton (found by grep, not by family) call the check in `register` (rev 2.1, A-R2)."""
    users = {p.parent.relative_to(PLUGINS).parts[:2] for p in PLUGINS.rglob("*.py")
             if "vendor" not in p.parts and re.search(r"^\s*(import triton|from triton)", p.read_text(), re.M)}
    assert len(users) == 10
    for parts in users:
        plugin = (PLUGINS.joinpath(*parts) / "plugin.py").read_text()
        call = r"def register\(self, regs[^)]*\) -> None:\n        host\.require_c_toolchain\(\)"
        assert re.search(call, plugin), parts


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
    with pytest.raises(HostError, match=r"torch is .*pip install --force-reinstall torch==2.11.0 --index-url"):
        ext.load(120)
    for d in tmp_path.iterdir():
        d.rmdir()
    (tmp_path / (want[:-16] + "0" * 16)).mkdir()                            # this torch, other sources: another version
    with pytest.raises(ext.NotBuilt, match=r"another Holon version.*force-reinstall holonlm=="):
        ext.load(120)
    (tmp_path / (want[:-16] + "0" * 16)).rmdir()
    (tmp_path / ext.key(86)).mkdir()                                        # another SM only: the usual message
    with pytest.raises(ext.NotBuilt, match=r"no prebuilt module for sm 120.*CUDA 13's nvcc"):
        ext.load(120)


@pytest.mark.parametrize("name", EXTENSIONS)
def test_a_prebuild_of_several_sms_builds_each_in_its_own_process(name, monkeypatch):
    """torch's loader renames a module built twice in one process (`<name>_v1`), never found by `load` (v1-pip F8)."""
    prebuild = __import__(f"holon_plugins.kernels.{name}.prebuild", fromlist=["x"])
    runs = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: runs.append(cmd))
    monkeypatch.setattr(prebuild.extension, "build", lambda *a: pytest.fail("built in this process"))
    prebuild.main(["--sm", "86", "120"])
    assert [cmd[cmd.index("--sm") + 1:cmd.index("--sm") + 2] for cmd in runs] == [["86"], ["120"]]


def test_flashinfer_reads_holon_s_modules_for_one_sm(tmp_path, monkeypatch):
    aot = pytest.importorskip("holon_plugins.kernels.flashinfer.aot")
    monkeypatch.setattr(aot, "PREBUILT", tmp_path)
    monkeypatch.setattr(aot.env, "FLASHINFER_AOT_DIR", pathlib.Path("/elsewhere"))
    capability = {"cuda:0": (8, 6), "cuda:1": (8, 9)}
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d: capability[d])
    aot.bind("cuda:0")                       # set even with no modules shipped: never a foreign jit-cache's
    assert aot.env.FLASHINFER_AOT_DIR == aot.directory(86) and aot.directory(86).name.endswith(
        f"-cu{torch.version.cuda}-sm86") and "-tvmffi" in aot.directory(86).name
    with pytest.raises(RuntimeError, match="one SM a process"):
        aot.bind("cuda:1")
    monkeypatch.setattr(aot.env, "FLASHINFER_AOT_DIR", pathlib.Path("/elsewhere"))
    (tmp_path / aot.directory(89).name.replace(f"-cu{torch.version.cuda}-", "-cu12.8-")).mkdir()
    with pytest.raises(HostError, match="prebuilt modules are for CUDA 13.0"):
        aot.bind("cuda:1")
