"""The native module of one CPU tier: the vendored ggml-cpu subset compiled as C with the tier's flags, and
csrc/shim.c, into a static library, linked into a thin torch module (csrc/cpu_experts.cpp). Only the vendored
quantizers and dot products take the tier's flags: what runs when the module loads, and the experts' own f32
arithmetic (SwiGLU), is baseline x86-64, so a module loads on any host (a build host may lack the tier) and its
SwiGLU rounds alike on every tier. Built ahead of time by `prebuild` into the user cache, keyed by everything that
changes the binary; the engine only loads it and never compiles. A tier is a fixed flag set
(`holon.core.platform.CPU_TIERS`), never `-march=native`."""
from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import os
import pathlib
import shutil
import subprocess
import sys
from types import ModuleType

import torch

HERE = pathlib.Path(__file__).resolve().parent
VENDOR, CSRC = HERE / "vendor", HERE / "csrc"
GGML = VENDOR / "ggml"
NAME = "holon_ggml_cpu"
PREBUILD = "python -m holon_plugins.kernels.ggml_cpu.prebuild"
# The user cache (XDG), which an image pins to a path of its own (Dockerfile).
CACHE = pathlib.Path(os.environ.get("XDG_CACHE_HOME") or pathlib.Path.home() / ".cache") / "holon" / "ggml_cpu"
AVX2 = ("-mavx2", "-mfma", "-mf16c", "-mbmi2")
TIERS = {"x86-avx2": AVX2, "x86-avx512": (*AVX2, "-mavx512f", "-mavx512bw", "-mavx512dq", "-mavx512vl")}
C_FLAGS = ("-std=c11", "-O3", "-DNDEBUG", "-D_GNU_SOURCE", "-fPIC")
C_SOURCES = (GGML / "src" / "ggml-quants.c", GGML / "src" / "ggml-cpu" / "quants.c",
             GGML / "src" / "ggml-cpu" / "arch" / "x86" / "quants.c", CSRC / "shim.c")   # the last: baseline
INCLUDES = (GGML / "include", GGML / "src", GGML / "src" / "ggml-cpu", CSRC)


class NotBuilt(RuntimeError):
    pass


def key(tier: str) -> str:
    """The cache key: vendored and own sources, the build recipe (this file), the tier's flags, the Python ABI and torch
    (the module links libtorch)."""
    h = hashlib.sha256(" ".join((*C_FLAGS, *TIERS[tier])).encode())
    for f in sorted(p for d in (VENDOR, CSRC) for p in d.rglob("*") if p.is_file()) + [pathlib.Path(__file__)]:
        h.update(str(f.relative_to(HERE)).encode() + b"\0" + f.read_bytes())
    return f"{tier}-{sys.implementation.cache_tag}-torch{torch.__version__}-{h.hexdigest()[:16]}"


def module_path(tier: str) -> pathlib.Path:
    return CACHE / key(tier) / "ext" / f"{NAME}.so"


def load(tier: str) -> ModuleType:
    """The prebuilt module for CPU tier `tier`; NotBuilt (naming the prebuild command) if there is none."""
    if tier not in TIERS:
        raise NotBuilt(f"kernels.ggml_cpu supports CPU tiers {sorted(TIERS)}, not {tier!r}")
    path = module_path(tier)
    if not path.is_file():
        raise NotBuilt(f"kernels.ggml_cpu: no prebuilt module for {tier}, torch {torch.__version__} (looked for "
                       f"{path}); build it with: {PREBUILD}")
    spec = importlib.util.spec_from_file_location(NAME, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(tier: str, verbose: bool = False) -> pathlib.Path:
    """Compile the static library, then link the torch module; returns the module path. Concurrent builds of one key
    take turns (a lock file in its directory)."""
    from torch.utils import cpp_extension

    cc = os.environ.get("CC") or shutil.which("gcc") or shutil.which("cc")
    if not cc:
        raise RuntimeError("kernels.ggml_cpu: no C compiler (gcc or $CC)")
    out = CACHE / key(tier)
    (out / "obj").mkdir(parents=True, exist_ok=True)
    with open(out / "build.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        objs = []
        for src in C_SOURCES:
            objs.append(str(out / "obj" / f"{src.parent.name}-{src.stem}.o"))
            isa = TIERS[tier] if src.parent != CSRC else ()
            subprocess.run([cc, *C_FLAGS, *isa, *(f"-I{d}" for d in INCLUDES), "-c", str(src), "-o", objs[-1]],
                           check=True, capture_output=not verbose)
        lib = out / "libggml_cpu.a"
        lib.unlink(missing_ok=True)
        subprocess.run(["ar", "rcs", str(lib), *objs], check=True)
        (out / "ext").mkdir(exist_ok=True)
        cpp_extension.load(name=NAME, sources=[str(CSRC / "cpu_experts.cpp")],
                           extra_include_paths=[str(d) for d in INCLUDES], extra_cflags=["-O3"],
                           extra_ldflags=[str(lib)], build_directory=str(out / "ext"), with_cuda=False,
                           verbose=verbose)
    return module_path(tier)
