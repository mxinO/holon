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
TORCH_FIX = ("`pip install --force-reinstall torch==2.11.0 --index-url https://pypi.org/simple`, in a venv without "
             "extra indexes")
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
            done = subprocess.run([cc, str(src), "-O3", "-shared", "-fPIC", "-o", str(src.with_suffix(".so")),
                                   f"-I{include}"], capture_output=True, text=True)
        if done.returncode == 0:
            return
        missing = f"{cc} cannot build a Python extension: {done.stderr.strip().splitlines()[-1:] or done.returncode}"
    raise HostError(f"Triton compiles a small C launcher at first use, and this host has {missing}: {TOOLCHAIN_FIX}")


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
