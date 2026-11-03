"""Holon's own FlashInfer modules as FlashInfer's AOT directory (docs/dev/v1-pip.md D5): an overlay (holonlm-cuda-sm*)
installs them under `prebuilt/`, one directory per everything they depend on, and FlashInfer loads a module found
there without ninja or nvcc. Without them (a source install) FlashInfer falls to its JIT cache, where `holon prebuild`
put them.

The directory is FlashInfer's module global `jit.env.FLASHINFER_AOT_DIR`, which its `jit/env.py` keeps overridable for
this; a kernel plugin may set third-party state (invariant 3 is about Holon's own). It is always set, so an installed
`flashinfer-jit-cache` never supplies modules Holon did not build.
"""
from __future__ import annotations

import importlib.metadata
import pathlib

import flashinfer
import flashinfer.norm
import torch
from flashinfer.jit import env
from holon.api import HostError

from holon_plugins import host

PREBUILT = pathlib.Path(__file__).resolve().parent / "prebuilt"


def directory(sm: int) -> pathlib.Path:
    """The modules for `sm`: named by FlashInfer, TVM FFI (they link it), CUDA and the SM, as the extensions' keys."""
    tvm_ffi = importlib.metadata.version("apache-tvm-ffi")
    return PREBUILT / f"{flashinfer.__version__}-tvmffi{tvm_ffi}-cu{torch.version.cuda}-sm{sm}"


def bind(device: torch.device) -> None:
    """FlashInfer reads Holon's modules for `device`'s SM. One SM a process: a second SM's modules could load and run
    the first's code (one major's binaries run on its newer minors), so that is refused. The norms must be FlashInfer's
    CuTe DSL ones: without nvidia-cutlass-dsl it silently takes its CUDA norm, other numerics than the locks'."""
    major, minor = torch.cuda.get_device_capability(device)
    want = directory(10 * major + minor)
    shipped = [p.name for p in PREBUILT.iterdir() if p.is_dir()] if PREBUILT.is_dir() else []
    if shipped and not want.is_dir() and not any(f"-cu{torch.version.cuda}-" in name for name in shipped):
        raise HostError(f"kernels.flashinfer: this install's torch is {torch.__version__} (CUDA "
                             f"{torch.version.cuda}), and its prebuilt modules are for CUDA 13.0: {host.TORCH_FIX}")
    if env.FLASHINFER_AOT_DIR != want:
        if env.FLASHINFER_AOT_DIR.parent == PREBUILT:
            raise RuntimeError(f"kernels.flashinfer: one SM a process; it serves {env.FLASHINFER_AOT_DIR.name}, not "
                               f"{want.name}")
        env.FLASHINFER_AOT_DIR = want
    if flashinfer.norm._USE_CUDA_NORM:
        raise RuntimeError("kernels.flashinfer: FlashInfer's CuTe DSL norms do not import (nvidia-cutlass-dsl), so it "
                           "would run its CUDA norm, other numerics than the locks'")
