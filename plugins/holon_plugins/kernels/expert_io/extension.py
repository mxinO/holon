"""The native module (csrc/expert_io.cu), built ahead of time by `prebuild` into the user cache, keyed by everything
that changes the binary; the engine only loads it and never compiles (as kernels.ggml_cuda's)."""
from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import os
import pathlib
import sys
from types import ModuleType

import torch

from holon_plugins import host

HERE = pathlib.Path(__file__).resolve().parent
SOURCE = HERE / "csrc" / "expert_io.cu"
NAME = "holon_expert_io"
PREBUILD = "python -m holon_plugins.kernels.expert_io.prebuild"
# The user cache (XDG), which an image pins to a path of its own (Dockerfile).
SHIPPED = HERE / "prebuilt"   # an overlay's modules by key (holonlm-cuda-sm*, docs/dev/v1-pip.md D5), before the cache
CACHE = pathlib.Path(os.environ.get("XDG_CACHE_HOME") or pathlib.Path.home() / ".cache") / "holon" / "expert_io"
ARCHS = (86, 89, 120)
FLAGS = ("-O3", "-DNDEBUG", "-std=c++17")


def key(sm: int) -> str:
    """The source, flags, the build recipe (this file), target SM, CUDA, the Python ABI and torch (the module links
    libtorch)."""
    h = hashlib.sha256(" ".join(FLAGS).encode() + b"\0" + SOURCE.read_bytes() + b"\0"
                       + pathlib.Path(__file__).read_bytes())
    return f"sm{sm}-cu{torch.version.cuda}-{sys.implementation.cache_tag}-torch{torch.__version__}-{h.hexdigest()[:16]}"


def module_path(sm: int, k: str | None = None) -> pathlib.Path:
    """The module an overlay ships for `sm`, else the user cache's (`k`: its key, if the caller has it)."""
    k = k or key(sm)
    shipped = SHIPPED / k / f"{NAME}.so"
    return shipped if shipped.is_file() else CACHE / k / f"{NAME}.so"


def load(sm: int) -> ModuleType:
    """The prebuilt module for `sm`; a `HostError` naming the fix (`host.missing`) if there is none."""
    path = module_path(sm, k := key(sm))
    if sm not in ARCHS or not path.is_file():
        raise host.missing("kernels.expert_io", sm, k, SHIPPED, f"{PREBUILD} --sm {sm}")
    spec = importlib.util.spec_from_file_location(NAME, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(sm: int, verbose: bool = False) -> pathlib.Path:
    """Compile the module for `sm` (concurrent builds of one key take turns); returns its path."""
    from torch.utils import cpp_extension

    if (shipped := SHIPPED / (k := key(sm)) / f"{NAME}.so").is_file():   # an overlay ships it: nothing to build
        return shipped
    out = CACHE / k
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "build.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{sm // 10}.{sm % 10}"     # this process builds one target
        cpp_extension.load(name=NAME, sources=[str(SOURCE)], extra_cflags=["-O2"], extra_cuda_cflags=list(FLAGS),
                           build_directory=str(out), with_cuda=True, verbose=verbose)
    return module_path(sm, k)
