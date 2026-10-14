"""The native extension: vendored llama.cpp kernels + shim + C API as a static library (llama.cpp's own nvcc flags),
linked into a thin torch module. Built ahead of time by `prebuild` into the user cache, keyed by everything that
changes the binary; the engine only loads it and never compiles (M0.1 design §4.4)."""
from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import os
import pathlib
import re
import shutil
import subprocess
import sys
from types import ModuleType

import torch

HERE = pathlib.Path(__file__).resolve().parent
VENDOR, CSRC = HERE / "vendor", HERE / "csrc"
CUDA_DIR = VENDOR / "ggml" / "src" / "ggml-cuda"
NAME = "holon_ggml_cuda"
PREBUILD = "python -m holon_plugins.kernels.ggml_cuda.prebuild"
# The user cache (XDG), which an image pins to a path of its own (Dockerfile).
CACHE = pathlib.Path(os.environ.get("XDG_CACHE_HOME") or pathlib.Path.home() / ".cache") / "holon" / "ggml_cuda"

ARCHS = {86: "86", 89: "89", 120: "120a"}     # 120a: the fp4 mmq instances need the arch-specific target
NVCC_FLAGS = ("-std=c++17", "-O3", "-DNDEBUG", "-use_fast_math", "-extended-lambda", "-Xcompiler", "-fPIC")
INCLUDES = (VENDOR / "ggml" / "include", VENDOR / "ggml" / "src", CUDA_DIR, CSRC)


class NotBuilt(RuntimeError):
    pass


def sources() -> list[pathlib.Path]:
    return sorted(CUDA_DIR.glob("*.cu")) + sorted(CUDA_DIR.glob("template-instances/*.cu")) + sorted(CSRC.glob("*.cu"))


def gencode(sm: int) -> str:
    return f"-gencode=arch=compute_{ARCHS[sm]},code=sm_{ARCHS[sm]}"


def cuda_major() -> int:
    return int(torch.version.cuda.split(".")[0])


def key(sm: int) -> str:
    """The cache key: vendored + own sources, flags, target SM, CUDA major, the Python ABI and torch (the module
    links libtorch)."""
    h = hashlib.sha256(" ".join((*NVCC_FLAGS, gencode(sm))).encode())
    for f in sorted(p for d in (VENDOR, CSRC) for p in d.rglob("*") if p.is_file()):
        h.update(str(f.relative_to(HERE)).encode() + b"\0" + f.read_bytes())
    return f"sm{sm}-cu{cuda_major()}-{sys.implementation.cache_tag}-torch{torch.__version__}-{h.hexdigest()[:16]}"


def module_path(sm: int) -> pathlib.Path:
    return CACHE / key(sm) / "ext" / f"{NAME}.so"


def load(sm: int) -> ModuleType:
    """The prebuilt module for `sm`; NotBuilt (naming the prebuild command) if there is none."""
    if sm not in ARCHS:
        raise NotBuilt(f"kernels.ggml_cuda supports sm {sorted(ARCHS)}, not sm {sm}")
    path = module_path(sm)
    if not path.is_file():
        raise NotBuilt(f"kernels.ggml_cuda: no prebuilt extension for sm {sm}, CUDA {cuda_major()}, torch "
                       f"{torch.__version__} (looked for {path}); build it with: {PREBUILD} --sm {sm}")
    spec = importlib.util.spec_from_file_location(NAME, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(sm: int, jobs: int, verbose: bool = False) -> pathlib.Path:
    """Compile the static library with ninja, then link the torch module; returns the module path. Concurrent
    builds of one key take turns (a lock file in its directory)."""
    from torch.utils import cpp_extension

    nvcc = shutil.which("nvcc") or (cpp_extension.CUDA_HOME and str(pathlib.Path(cpp_extension.CUDA_HOME, "bin/nvcc")))
    if not nvcc or not pathlib.Path(nvcc).is_file():
        raise RuntimeError("kernels.ggml_cuda: nvcc not found")
    version = subprocess.run([nvcc, "--version"], capture_output=True, text=True, check=True).stdout
    major = int(re.search(r"release (\d+)\.", version).group(1))
    if major != cuda_major():
        raise RuntimeError(f"kernels.ggml_cuda: nvcc is CUDA {major}, torch is built for CUDA {cuda_major()}")
    out = CACHE / key(sm)
    (out / "obj").mkdir(parents=True, exist_ok=True)
    with open(out / "build.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _build(sm, nvcc, out, jobs, verbose)
    return module_path(sm)


def _build(sm: int, nvcc: str, out: pathlib.Path, jobs: int, verbose: bool) -> None:
    from torch.utils import cpp_extension

    flags = " ".join((*NVCC_FLAGS, gencode(sm), *(f"-I{d}" for d in INCLUDES)))
    lines = [f"rule cu\n  command = {nvcc} {flags} -MD -MF $out.d -c $in -o $out\n  depfile = $out.d\n  deps = gcc\n",
             "rule ar\n  command = rm -f $out && ar rcs $out $in\n"]
    objs = []
    for src in sources():
        objs.append(f"obj/{src.stem}.o")
        lines.append(f"build {objs[-1]}: cu {src}\n")
    lines.append(f"build libggml_cuda.a: ar {' '.join(objs)}\n")
    (out / "build.ninja").write_text("".join(lines))
    subprocess.run(["ninja", "-C", str(out), "-j", str(jobs)] + ([] if verbose else ["--quiet"]), check=True)
    (out / "ext").mkdir(exist_ok=True)
    cpp_extension.load(name=NAME, sources=[str(CSRC / "torch_ext.cpp")], extra_include_paths=[str(CSRC)],
                       extra_cflags=["-O2"], extra_ldflags=[str(out / "libggml_cuda.a")],
                       build_directory=str(out / "ext"), with_cuda=True, verbose=verbose)
