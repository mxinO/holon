"""Device abstraction (design §8). The only place in the core allowed to touch `torch.cuda` (invariant 6)."""
from __future__ import annotations

import mmap
import os
import pathlib
import re
from collections.abc import Callable, Sequence

import torch

from holon.api import Fingerprint

# CPU tiers by machine, best first: prebuilt compiler flag sets (never `-march=native`); a host's is the best whose
# flags its CPU lists.
CPU_TIERS = {"x86_64": (("x86-avx512", frozenset({"avx512f", "avx512bw", "avx512dq", "avx512vl", "avx2", "fma",
                                                    "f16c"})),
                        ("x86-avx2", frozenset({"avx2", "fma", "f16c"})),
                        ("x86", frozenset())),
             "aarch64": (("arm64", frozenset()),)}


def fingerprint(device: torch.device) -> Fingerprint:
    """Static facts of `device`; the GPU class is the product name without vendor words, e.g. "rtx-5090"."""
    toolchain = f"cu{(torch.version.cuda or 'none').replace('.', '')}-torch{'.'.join(torch.__version__.split('.')[:2])}"
    if device.type != "cuda":
        return Fingerprint(gpu="cpu", sm=0, vram_gib=0, toolchain=toolchain, cpu_isa=cpu_tier())
    p = torch.cuda.get_device_properties(device)
    words = [w for w in re.split(r"[\s_]+", p.name.lower()) if w not in ("nvidia", "geforce")]
    return Fingerprint(gpu="-".join(words), sm=10 * p.major + p.minor, vram_gib=round(p.total_memory / 2**30),
                       toolchain=toolchain, num_sms=p.multi_processor_count, cpu_isa=cpu_tier())


def cpu_tier(machine: str | None = None, flags: set[str] | None = None) -> str:
    """The best of `CPU_TIERS` this host's CPU runs (the kernel's flag list); "" for an unknown machine."""
    if flags is None:
        try:
            with open("/proc/cpuinfo") as f:          # x86's "flags", arm's "Features"
                line = next((ln for ln in f if ln.startswith(("flags", "Features"))), "")
            flags = set(line.partition(":")[2].split())
        except OSError:
            flags = set()
    return next((name for name, need in CPU_TIERS.get(machine or os.uname().machine, ()) if need <= flags), "")


def pin_loop(core: int) -> None:
    """Run the calling thread, the engine loop, on CPU `core` only (e.g. one no CPU worker spins on); threads it starts
    later (torch's CPU pool, if it starts one) inherit that."""
    os.sched_setaffinity(0, {core})


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


def peak_memory(device: torch.device, fn: Callable[[], object], out: dict | None = None) -> int:
    """Peak bytes `fn()` takes on `device` beyond what it held before: its allocations' peak, or what the caching
    allocator maps for them where that is more — its segments' rounding and the holes its blocks leave (allocations
    alone left an RTX 5090 at --memory-fraction 0.99 with 3 MiB free after 32K-token steps). It returns cached,
    unused blocks to the driver first, as `reserved_growth` does, so the growth is measured from a clean cache. With
    `out`, `out["overshoot"]`: what the allocator reserved beyond the peak returned (none: the peak counts it)."""
    _cuda(device, "peak memory")
    synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    allocated, reserved = torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device)
    fn()
    synchronize(device)
    grown = torch.cuda.max_memory_reserved(device) - reserved
    peak = max(torch.cuda.max_memory_allocated(device) - allocated, grown)
    if out is not None:
        out["overshoot"] = max(0, grown - peak)
    return peak


def gpu_shared(device: torch.device) -> str | None:
    """Why other clients may grow on `device` after the start ("" if none seen; None: unknown, without NVML as on
    arm64): it drives a display, or a process visible in this PID namespace, not this engine nor its parent, holds
    memory on it."""
    try:
        import pynvml
        pynvml.nvmlInit()
        gpu = pynvml.nvmlDeviceGetHandleByUUID(f"GPU-{torch.cuda.get_device_properties(device).uuid}")
        if pynvml.nvmlDeviceGetDisplayActive(gpu):
            return "drives a display"
        pids = {p.pid for f in (pynvml.nvmlDeviceGetComputeRunningProcesses,
                                pynvml.nvmlDeviceGetGraphicsRunningProcesses) for p in f(gpu)}
    except Exception:  # noqa: BLE001 — no NVML or no answer
        return None
    others = sorted(p for p in pids - {os.getpid(), os.getppid()} if os.path.exists(f"/proc/{p}"))
    return f"processes {others} use it" if others else ""


def host_ram(proc: str = "/proc", cgroup: str = "/sys/fs/cgroup") -> tuple[int, int, str]:
    """(available, total, what limits it): the lesser of `MemAvailable` and, for this process's cgroup (v2) and each
    ancestor with a limit, the limit less what it holds besides clean file pages; total the lesser of `MemTotal` and
    the lowest limit (docs/dev/v1-auto-config.md D1). Under cgroup v1, `MemAvailable` alone."""
    read = lambda f: pathlib.Path(f).read_text() if os.path.exists(f) else ""      # noqa: E731
    info = {k: int(v.split()[0]) << 10 for k, v in (ln.split(":", 1) for ln in read(f"{proc}/meminfo").splitlines())}
    available, total, why, groups = info["MemAvailable"], info["MemTotal"], "meminfo", read(f"{proc}/self/cgroup")
    if ":memory:" in groups:
        return available, total, "meminfo; cgroup v1 limits not read: --host-memory sets them"
    root, found = pathlib.Path(cgroup), (available, total, why)
    path = root / next((ln[3:].lstrip("/") for ln in groups.splitlines() if ln.startswith("0::")), "")
    try:
        for d in (path if path.is_dir() else root, *path.parents):   # a container's own cgroup is its mount's root
            if limits := [int(v) for f in ("max", "high") if (v := read(d / f"memory.{f}").strip()) not in ("", "max")]:
                stat = {k: int(v) for k, v in (ln.split() for ln in read(d / "memory.stat").splitlines())}
                free = min(limits) - int(read(d / "memory.current")) + sum(stat.get(k, 0) * s for k, s in (
                    ("active_file", 1), ("inactive_file", 1), ("file_dirty", -1), ("file_writeback", -1)))
                binds = free < found[0]
                found = (max(0, min(found[0], free)), min(found[1], *limits),
                         f"cgroup /{d.relative_to(root)}" if binds else found[2])
            if d == root:
                break
    except (OSError, ValueError) as e:             # unreadable: what meminfo says, and why
        return available, total, f"meminfo; cgroup not read ({e}): --host-memory sets a limit"
    return found


def host_headroom(total: int) -> int:
    """Host memory kept free of pinned pools: what grows after the start (worker processes, the frontend, a desktop)."""
    return max(6 << 30, total // 20)


def reserved_growth(device: torch.device, fn: Callable[[], object], copies: int = 1) -> int:
    """Bytes the allocator still reserves after `fn()` beyond before it (e.g. a CUDA-graph pool `fn` creates), and
    `copies` times what the device keeps for it outside the allocator (e.g. its graphs' instantiated execs)."""
    _cuda(device, "reserved memory")
    synchronize(device)
    torch.cuda.empty_cache()
    before, free = torch.cuda.memory_reserved(device), torch.cuda.mem_get_info(device)[0]
    fn()
    synchronize(device)
    torch.cuda.empty_cache()                 # only what `fn` keeps (live graph pools are not released)
    reserved = torch.cuda.memory_reserved(device) - before
    return reserved + copies * max(free - torch.cuda.mem_get_info(device)[0] - reserved, 0)


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


def host_memory(nbytes: int, device: torch.device, mapped: bool) -> torch.Tensor:
    """Exactly `nbytes` of host memory in whole pages (an anonymous map), page-locked for CUDA by registering it — not
    torch's pinned allocator, which rounds up to a power of two —, with `mapped` also addressable by kernels
    (`device_alias`). Plain memory off CUDA. It is unregistered and unmapped with the last tensor viewing it. The map is
    private (never shared with a child process). On CUDA it asks for transparent huge pages (`MADV_HUGEPAGE`): the
    link reads them faster, kernels' zero-copy reads ≈ 4% and copy-engine DMA ≈ 13% on an RTX 5090, and they page-lock
    4× faster (0.06 s per GiB there, 0.26 with 4 KiB pages; FN §23.6). Where the kernel cannot find huge pages it falls
    back to small ones."""
    region = _Pinned(-1, nbytes, flags=mmap.MAP_PRIVATE)                # anonymous: fileno -1
    out = torch.frombuffer(region, dtype=torch.uint8)                    # its views keep the map alive
    if device.type == "cuda":
        region.madvise(mmap.MADV_HUGEPAGE)                               # before registering faults the pages in
        _check(torch.cuda.cudart().cudaHostRegister(out.data_ptr(), nbytes, 1 | 2 * mapped),  # portable, mapped
               "cudaHostRegister")
        region.registered = out.data_ptr()
    return out


class _Pinned(mmap.mmap):
    """An anonymous map that unregisters itself before it is unmapped."""

    registered: int | None = None

    def __del__(self) -> None:
        if self.registered is None:
            return
        try:
            err = int(torch.cuda.cudart().cudaHostUnregister(self.registered))
        except Exception:  # noqa: BLE001 — the interpreter is shutting down: the process's memory goes with it
            return
        if err not in _CUDA_GONE:
            _check(err, "cudaHostUnregister")


_CUDA_GONE = (0, 4, 709)     # success, cudaErrorCudartUnloading, cudaErrorContextIsDestroyed (CUDA shut down first)


def device_alias(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """`host_memory(…, mapped=True)` as a tensor on `device` over the same bytes (unified addressing: registered
    memory has its host address on the device); `t` itself off CUDA."""
    if device.type != "cuda":
        return t
    view = type("Mapped", (), {"__cuda_array_interface__": {
        "shape": (t.numel(),), "typestr": "|u1", "data": (t.data_ptr(), False), "version": 3}, "host": t})
    alias = torch.as_tensor(view(), device=device)              # keeps `view`, which keeps the host memory
    if alias.data_ptr() != t.data_ptr():          # not the current device: torch copied instead
        raise RuntimeError(f"no alias of host memory on {device} (make it the current device)")
    return alias


def _check(err: object, what: str) -> None:
    if int(err):
        raise RuntimeError(f"{what}: {torch.cuda.cudart().cudaGetErrorString(err)}")


class Event:
    """Orders host work after device work: `record()` in stream order, `synchronize()` waits for it on the host,
    `wait()` makes work enqueued next on the current stream wait for it on the device."""

    def __init__(self, device: torch.device) -> None:
        self._event = torch.cuda.Event() if device.type == "cuda" else None

    def record(self) -> None:
        if self._event is not None:
            self._event.record()

    def synchronize(self) -> None:
        if self._event is not None:
            self._event.synchronize()

    def wait(self) -> None:
        if self._event is not None:
            torch.cuda.current_stream().wait_event(self._event)


class Stream:
    """A stream beside the step's (the core's copies); off CUDA, work runs at once, in order."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._stream = torch.cuda.Stream(device) if device.type == "cuda" else None

    def run(self, fn: Callable[[], object], after: Sequence[Event] = ()) -> Event:
        """Enqueue `fn()`'s device work on this stream, after `after`; the event recorded once it is done."""
        done = Event(self.device)
        if self._stream is None:
            fn()
            return done
        with torch.cuda.stream(self._stream):
            for event in after:
                event.wait()
            fn()
            done.record()
        return done


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
