"""Holon's own FlashInfer modules as FlashInfer's AOT directory (docs/dev/v1-pip.md D5): an overlay (holonlm-cuda-sm*)
installs them under `prebuilt/`, one directory per everything they depend on, and FlashInfer loads a module found
there without ninja or nvcc. Without them (a source install) FlashInfer falls to its JIT cache, where `holon prebuild`
put them.

The directory is FlashInfer's module global `jit.env.FLASHINFER_AOT_DIR`, which its `jit/env.py` keeps overridable for
this; a kernel plugin may set third-party state (invariant 3 is about Holon's own). It is always set, so an installed
`flashinfer-jit-cache` never supplies modules Holon did not build.
"""
from __future__ import annotations

import functools
import importlib.metadata
import pathlib

import flashinfer
import flashinfer.norm
import torch
from flashinfer.jit import env
from holon.api import HostError

from holon_plugins import host

PREBUILT = pathlib.Path(__file__).resolve().parent / "prebuilt"


@functools.cache
def directory(sm: int) -> pathlib.Path:
    """The modules for `sm`: named by FlashInfer, TVM FFI (they link it), CUDA and the SM, as the extensions' keys. (A
    foreign torch is told from these names here and from the parsed keys in `host.shipped_mismatch`: keep the two
    refusals' words in step.)"""
    tvm_ffi = importlib.metadata.version("apache-tvm-ffi")
    return PREBUILT / f"{flashinfer.__version__}-tvmffi{tvm_ffi}-cu{torch.version.cuda}-sm{sm}"


def bind(device: torch.device) -> None:
    """FlashInfer reads Holon's modules for `device`'s SM, refused at once (`HostError`, no retry) where it cannot:
    - no add-on module for this SM and none the user built (FlashInfer's JIT cache), with another torch build's
      add-ons, or with no CUDA toolkit to build them (the add-on to install, or that this SM is not shipped);
    - a second SM in one process: one major's binaries run on its newer minors, so the first SM's modules would run;
    - FlashInfer's CUDA norm in place of its CuTe DSL one (nvidia-cutlass-dsl not importing, or
      FLASHINFER_USE_CUDA_NORM=1): other numerics than the locks'."""
    major, minor = torch.cuda.get_device_capability(device)
    sm = 10 * major + minor
    want = directory(sm)
    if not want.is_dir() and not any(env.FLASHINFER_JIT_DIR.glob("*/*.so")):
        shipped = [p.name for p in PREBUILT.iterdir() if p.is_dir()] if PREBUILT.is_dir() else []
        if shipped and not any(f"-cu{torch.version.cuda}-" in name for name in shipped):
            raise HostError(f"kernels.flashinfer: this install's torch is {torch.__version__} (CUDA "
                            f"{torch.version.cuda}), and its prebuilt modules are for CUDA 13.0: {host.TORCH_FIX}")
        if not host.toolkit():
            raise host.no_add_on("kernels.flashinfer", sm)
    if env.FLASHINFER_AOT_DIR != want:
        if env.FLASHINFER_AOT_DIR.parent == PREBUILT:
            raise HostError(f"kernels.flashinfer: one GPU architecture a process: it serves "
                            f"{env.FLASHINFER_AOT_DIR.name}, not {want.name}")
        env.FLASHINFER_AOT_DIR = want
    if flashinfer.norm._USE_CUDA_NORM:
        raise HostError("kernels.flashinfer: FlashInfer would run its CUDA norm (FLASHINFER_USE_CUDA_NORM=1, or "
                        "nvidia-cutlass-dsl not importing), other numerics than the locks': unset the variable, or "
                        "reinstall nvidia-cutlass-dsl")
