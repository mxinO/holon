"""Device abstraction (design §8). The only place in the core allowed to touch `torch.cuda` (invariant 6)."""
from __future__ import annotations

import re
from collections.abc import Callable

import torch

from holon.api import Fingerprint


def fingerprint(device: torch.device) -> Fingerprint:
    """Static facts of `device`; the GPU class is the product name without vendor words, e.g. "rtx-5090"."""
    toolchain = f"cu{(torch.version.cuda or 'none').replace('.', '')}-torch{'.'.join(torch.__version__.split('.')[:2])}"
    if device.type != "cuda":
        return Fingerprint(gpu="cpu", sm=0, vram_gib=0, toolchain=toolchain)
    p = torch.cuda.get_device_properties(device)
    words = [w for w in re.split(r"[\s_]+", p.name.lower()) if w not in ("nvidia", "geforce")]
    return Fingerprint(gpu="-".join(words), sm=10 * p.major + p.minor, vram_gib=round(p.total_memory / 2**30),
                       toolchain=toolchain, num_sms=p.multi_processor_count)


def _cuda(device: torch.device, what: str) -> None:
    if device.type != "cuda":
        raise ValueError(f"{what} is only measured on CUDA devices (device {device}); configure it explicitly")


def free_memory(device: torch.device) -> int:
    """Bytes free on `device` after returning cached allocator blocks to the driver."""
    _cuda(device, "free memory")
    torch.cuda.empty_cache()
    return torch.cuda.mem_get_info(device)[0]


def peak_memory(device: torch.device, fn: Callable[[], object]) -> int:
    """Peak bytes `fn()` allocates on `device` beyond what was allocated before it."""
    _cuda(device, "peak memory")
    synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    before = torch.cuda.memory_allocated(device)
    fn()
    synchronize(device)
    return torch.cuda.max_memory_allocated(device) - before


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def pinned_empty(n: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Host staging memory for `device`: page-locked for CUDA (async copies need it), plain memory otherwise."""
    return torch.empty(n, dtype=dtype, pin_memory=device.type == "cuda")


class Event:
    """Orders host work after device work: `record()` in stream order, `synchronize()` waits for it on the host."""

    def __init__(self, device: torch.device) -> None:
        self._event = torch.cuda.Event() if device.type == "cuda" else None

    def record(self) -> None:
        if self._event is not None:
            self._event.record()

    def synchronize(self) -> None:
        if self._event is not None:
            self._event.synchronize()


class no_host_sync:  # noqa: N801 — used as a context manager
    """Make any implicit device→host synchronization an error inside the block (CUDA only; a test tripwire)."""

    def __init__(self, device: torch.device) -> None:
        self.cuda = device.type == "cuda"

    def __enter__(self) -> None:
        if self.cuda:
            torch.cuda.set_sync_debug_mode("error")

    def __exit__(self, *exc: object) -> None:
        if self.cuda:
            torch.cuda.set_sync_debug_mode("default")
