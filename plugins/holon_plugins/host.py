"""What the plugins need from the host, checked where they load so a missing piece fails the first start at once
(`HostError`: no retry) with the fix, rather than deep in a start (docs/dev/v1-pip.md D-H, D5).

A library of the plugins distribution, outside every plugin directory, so no unit's `src` hashes it. The rule for
code outside a unit: it must not shape outputs, or the units that use it pin it. This module only checks and words
errors; it chooses nothing a plan records.
"""
from __future__ import annotations

import functools
import importlib.metadata
import os
import pathlib
import re
import shutil
import subprocess
import sysconfig
import tempfile

import torch
from holon.api import HostError

TOOLCHAIN_FIX = ("install a C compiler and Python 3.12's headers (Ubuntu: `sudo apt install build-essential "
                 "python3.12-dev`), or use a uv-managed Python, which has its headers (`uv tool install --python 3.12 "
                 "holonlm`)")
# The exact local label matches only the cu130 build (PyPI's files, v1-pip F2), whatever extra index pip.conf adds.
TORCH_FIX = ('`pip install --force-reinstall "torch==2.11.0+cu130" --index-url '
             'https://download.pytorch.org/whl/cu130`, in a venv without extra indexes')
SHIPPED_SMS = (86, 89, 120)          # the add-ons published, holonlm-cuda-sm<N> (docs/dev/v1-pip.md D1, D3)
_KEY = re.compile(r"^(?P<head>.*-torch(?P<torch>.+))-[0-9a-f]{16}$")    # the extensions' cache keys


@functools.cache
def require_c_toolchain() -> None:
    """Triton compiles a small C launcher per kernel at first use, with `$CC`, gcc or clang against Python's headers
    (`triton.runtime.build`): compile one the same way, once a process, or raise `HostError` naming the fix."""
    cc = os.environ.get("CC") or shutil.which("gcc") or shutil.which("clang")
    scheme = sysconfig.get_default_scheme()
    include = sysconfig.get_paths(scheme="posix_prefix" if scheme == "posix_local" else scheme)["include"]
    if cc is None:
        missing = "no C compiler ($CC, gcc or clang)"
    elif not pathlib.Path(include, "Python.h").is_file():
        missing = f"no Python.h in {include}"
    else:
        with tempfile.TemporaryDirectory() as d:
            src = pathlib.Path(d, "probe.c")
            src.write_text("#include <Python.h>\nint holon_probe(void) { return 0; }\n")
            try:
                done = subprocess.run([*cc.split(), str(src), "-O3", "-shared", "-fPIC", "-o",
                                       str(src.with_suffix(".so")), f"-I{include}"], capture_output=True, text=True)
            except OSError as e:                      # $CC names a program that is not there
                done = subprocess.CompletedProcess(cc, 1, "", f"error: {e}")
        if done.returncode == 0:
            return
        lines = done.stderr.strip().splitlines()
        missing = f"a C compiler ({cc}) that cannot build a Python extension: " + next(
            (line for line in lines if "error" in line), lines[-1] if lines else f"exit {done.returncode}")
    raise HostError(f"Triton compiles a small C launcher at first use, and this host has {missing}: {TOOLCHAIN_FIX}")


def toolkit() -> bool:
    """Whether a CUDA toolkit could build a missing module here, found as torch's and FlashInfer's builds find it."""
    return bool(os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or shutil.which("nvcc")
                or pathlib.Path("/usr/local/cuda/bin/nvcc").is_file())


def no_add_on(what: str, sm: int) -> HostError:
    """No prebuilt module for `sm` is installed and no toolkit could build one: the add-on to install, or that this GPU
    is not among those shipped (docs/dev/v1-pip.md D5)."""
    v = importlib.metadata.version("holonlm-plugins")
    if sm in SHIPPED_SMS:
        return HostError(f"{what}: this GPU's prebuilt modules (sm {sm}) are not installed: `pip install "
                         f"holonlm-cuda-sm{sm}=={v}` (or `pip install holonlm=={v}`, which brings them)")
    return HostError(f"{what}: this GPU is sm {sm}, and Holon's prebuilt modules cover sm "
                     f"{', '.join(map(str, SHIPPED_SMS))} (RTX 30, 40 and 50 series, RTX PRO 6000 Blackwell); another "
                     f"GPU needs a source install with CUDA 13's nvcc (the README's source install)")


def missing(what: str, sm: int, want: str, shipped: pathlib.Path, build: str, not_built: type) -> Exception:
    """The error for a module of cache key `want` that neither an add-on (in `shipped`) nor the user cache holds: a
    `HostError` when nothing at runtime can fix it (another torch build; no add-on and no toolkit to build one), else
    the plugin's own `not_built` (another Holon version's add-on, or `build`'s prebuild command with a toolkit)."""
    if reason := shipped_mismatch(what, want, shipped):         # raises for another torch build
        return not_built(reason)
    if not toolkit():
        return no_add_on(what, sm)
    return not_built(f"{what}: no prebuilt module for sm {sm} (not in {shipped} nor the user cache); build it with: "
                     f"{build} (CUDA 13's nvcc)")


def shipped_mismatch(what: str, want: str, shipped: pathlib.Path) -> str | None:
    """Why the prebuilt modules an overlay installed in `shipped` (one directory per cache key) hold no `want`: a
    `HostError` raised when they are for another torch build (a `+cu128` torch from another index); the reason when
    they are another Holon version's (the same key but its sources' hash); None when nothing is shipped or the key
    differs otherwise (another SM or Python), which the caller words."""
    keys = [m for p in shipped.iterdir() if p.is_dir() and (m := _KEY.match(p.name))] if shipped.is_dir() else []
    if not keys or not (want_key := _KEY.match(want)):
        return None
    if want_key["torch"] not in {k["torch"] for k in keys}:
        raise HostError(f"{what}: this install's torch is {torch.__version__}, and its prebuilt modules are for torch "
                        f"{' / '.join(sorted({k['torch'] for k in keys}))} (PyPI's, CUDA 13.0): {TORCH_FIX}")
    if want_key["head"] in {k["head"] for k in keys}:
        return (f"{what}: the prebuilt modules installed are from another Holon version than this plugin's code: "
                f"`pip install --force-reinstall holonlm=={importlib.metadata.version('holonlm-plugins')}` (every "
                f"holonlm distribution at one version)")
    return None
