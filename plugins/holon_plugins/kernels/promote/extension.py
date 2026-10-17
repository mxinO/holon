"""The native module (csrc/promote.cu), built ahead of time by `prebuild` into the user cache, keyed by everything
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

HERE = pathlib.Path(__file__).resolve().parent
SOURCE = HERE / "csrc" / "promote.cu"
NAME = "holon_promote"
PREBUILD = "python -m holon_plugins.kernels.promote.prebuild"
# The user cache (XDG), which an image pins to a path of its own (Dockerfile).
CACHE = pathlib.Path(os.environ.get("XDG_CACHE_HOME") or pathlib.Path.home() / ".cache") / "holon" / "promote"
ARCHS = (86, 89, 120)
FLAGS = ("-O3", "-DNDEBUG", "-std=c++17")


class NotBuilt(RuntimeError):
    pass


def key(sm: int) -> str:
    """The source, flags, target SM, CUDA, the Python ABI and torch (the module links libtorch)."""
    h = hashlib.sha256(" ".join(FLAGS).encode() + b"\0" + SOURCE.read_bytes())
    return f"sm{sm}-cu{torch.version.cuda}-{sys.implementation.cache_tag}-torch{torch.__version__}-{h.hexdigest()[:16]}"


def module_path(sm: int) -> pathlib.Path:
    return CACHE / key(sm) / f"{NAME}.so"


def load(sm: int) -> ModuleType:
    """The prebuilt module for `sm`; NotBuilt (naming the prebuild command) if there is none."""
    path = module_path(sm)
    if sm not in ARCHS or not path.is_file():
        raise NotBuilt(f"kernels.promote: no prebuilt module for sm {sm} (looked for {path}); build it with: "
                       f"{PREBUILD} --sm {sm}")
    spec = importlib.util.spec_from_file_location(NAME, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(sm: int, verbose: bool = False) -> pathlib.Path:
    """Compile the module for `sm` (concurrent builds of one key take turns); returns its path."""
    from torch.utils import cpp_extension

    out = CACHE / key(sm)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "build.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{sm // 10}.{sm % 10}"     # this process builds one target
        cpp_extension.load(name=NAME, sources=[str(SOURCE)], extra_cflags=["-O2"], extra_cuda_cflags=list(FLAGS),
                           build_directory=str(out), with_cuda=True, verbose=verbose)
    return module_path(sm)
