"""Device abstraction (design §8). The only place in the core allowed to touch `torch.cuda` (invariant 6)."""
from __future__ import annotations

from collections.abc import Callable

import torch


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
