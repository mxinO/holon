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


def visible_sms() -> list[int]:
    """The SMs of the visible CUDA devices, each once, e.g. [86, 120] (none without CUDA)."""
    if not torch.cuda.is_available():
        return []
    return sorted({10 * a + b for a, b in map(torch.cuda.get_device_capability, range(torch.cuda.device_count()))})


def _cuda(device: torch.device, what: str) -> None:
    if device.type != "cuda":
        raise ValueError(f"{what} is only measured on CUDA devices (device {device}); configure it explicitly")


def configure_allocator(device: torch.device) -> None:
    """Expandable segments: freed memory is reused across allocation sizes, so long-running mixes of large prefill
    activations and graph pools do not fragment the budget the ledger computed. Launchers should set it before CUDA
    starts (PYTORCH_CUDA_ALLOC_CONF); this is the in-process fallback, checked by `check_expandable`."""
    if device.type == "cuda":
        torch._C._accelerator_setAllocatorSettings("expandable_segments:True")


def check_expandable(t: torch.Tensor) -> None:
    """Fail loudly unless `t` (allocated after `configure_allocator`) lives in an expandable segment — torch offers no
    getter for the setting, and its API for it is private, so the effect is checked instead."""
    if t.device.type != "cuda":
        return
    address = t.data_ptr()
    segment = next(s for s in torch.cuda.memory_snapshot() if s["address"] <= address < s["address"] + s["total_size"])
    if not segment.get("is_expandable"):
        raise RuntimeError("could not enable expandable CUDA allocator segments, which the engine requires (the "
                           "ledger's budget assumes no fragmentation); set PYTORCH_CUDA_ALLOC_CONF="
                           "expandable_segments:True before starting, on a platform with CUDA virtual memory support")


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


def reserved_growth(device: torch.device, fn: Callable[[], object]) -> int:
    """Bytes the allocator still reserves after `fn()` beyond before it (e.g. a CUDA-graph pool `fn` creates)."""
    _cuda(device, "reserved memory")
    synchronize(device)
    torch.cuda.empty_cache()
    before = torch.cuda.memory_reserved(device)
    fn()
    synchronize(device)
    torch.cuda.empty_cache()                 # only what `fn` keeps (live graph pools are not released)
    return torch.cuda.memory_reserved(device) - before


def release_cache(device: torch.device) -> None:
    """Return cached, unused allocator blocks to the driver."""
    if device.type == "cuda":
        torch.cuda.empty_cache()


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


class Graphs:
    """Captured CUDA graphs by key, sharing one memory pool. The first time a key is seen its function runs eagerly
    (a warm-up: kernel compilation, allocator growth); the second time it is captured; afterwards it is replayed.
    Outputs live in the shared pool, so read them before the next replay."""

    def __init__(self, device: torch.device) -> None:
        self.enabled = device.type == "cuda"
        self._graphs: dict[object, tuple[torch.cuda.CUDAGraph, object]] = {}
        self._seen: set[object] = set()
        self._pool = torch.cuda.graph_pool_handle() if self.enabled else None

    def __len__(self) -> int:
        return len(self._graphs)

    def run(self, key: object, fn: Callable[[], object]) -> object:
        entry = self._graphs.get(key)
        if entry is None:
            if not self.enabled or key not in self._seen:
                self._seen.add(key)
                return fn()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._pool):
                out = fn()
            entry = self._graphs[key] = (graph, out)
        entry[0].replay()
        return entry[1]
