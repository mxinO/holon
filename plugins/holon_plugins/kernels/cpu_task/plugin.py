"""`cpu_task@1` (std.moe.kernels; docs/dev/v1-offload.md D6, spike S6): CUDA graphs hand CPU work to a persistent native
worker — a signal op, a wait op, a word per slot in mapped pinned memory, fail-open by contract (csrc/cpu_task.cu).

Each kind of work has its own worker (the fact `pool`): `compute` (CPU experts, F5b) and `io` (F9's SSD reads), so a
blocking read never stalls a layer's CPU experts. A worker is the engine's one shared object of its kind
(`cpu_task.<pool>`): tasks are bound while kernels are bound, its words are a mapped host pool requested before the
first materialize, its threads start after it (pinned, the first one coordinating) and stop after the drain.

Options (EngineConfig.plugin_options), from tune for a lock: `cpu_task.threads` (compute threads, the coordinator
included; default 4: S6 and M3 saturate DRAM with 2–4 for Q4_K experts), `cpu_task.io_threads` (default 2: M7's two
`pread` workers), `cpu_task.cores` / `cpu_task.io_cores` (CPU lists such as "2-5,8", the first `threads` of them
taken; default: one per physical core of
the process's CPUs, past the first — the OS's and the engine loop's —, compute first, then I/O), `cpu_task.wait`
(`spin`: 1-thread kernels, 2–3 µs per handshake; `memop`: stream memory operations, no SM held, for display GPUs;
default spin), `cpu_task.timeout` (seconds a wait waits before it fails open; default 60).

The native module is built ahead of time (`python -m holon_plugins.kernels.cpu_task.prebuild`); without it an
instance cannot be created and the error names that command."""
from __future__ import annotations

import ctypes
import functools
import os
import pathlib

import torch
from holon.api import Constraint, KernelSpec

from . import extension

OP = "cpu_task"
POOLS = {"compute": ("threads", "cores", 4), "io": ("io_threads", "io_cores", 2)}
TIMEOUT = 60.0                 # seconds (Maya bounds its wait at about 60 s)
LINE = 64                      # bytes per slot word (one cache line each)


class Service:
    """The engine's worker of one kind (a shared object): its slots' tasks, words and threads."""

    def __init__(self, ctx, pool: str) -> None:
        opts = ctx.options.get("plugins", {})
        threads, cores, default = POOLS[pool]
        self.lib, self.name = extension.load(ctx.fingerprint.sm), f"cpu_task.{pool}"
        n = int(opts.get(f"cpu_task.{threads}", default))
        self.cores = parse_cores(str(opts[f"cpu_task.{cores}"]))[:n] if f"cpu_task.{cores}" in opts else \
            default_cores(pool, n, int(opts.get("cpu_task.threads", POOLS["compute"][2])))
        if len(self.cores) != n:
            raise ValueError(f"{self.name}: {n} threads on cores {self.cores}")
        self.memop = {"spin": False, "memop": True}[str(opts.get("cpu_task.wait", "spin"))]
        self.timeout_ns = int(float(opts.get("cpu_task.timeout", TIMEOUT)) * 1e9)
        self.tasks, self.native, self.words, self.left = [], None, None, False

    def bind(self, task) -> int:
        if self.native is not None:
            raise RuntimeError(f"{self.name}: a task bound after the worker started")
        self.tasks.append(task)
        return len(self.tasks) - 1

    def before_materialize(self, ctx) -> None:
        if self.tasks:
            n = len(self.tasks) + 1                                   # the slots' words, then the flag
            self.words = ctx.pool(f"{self.name}.words", unit_bytes=LINE, min_units=n, pref_units=n, device="host",
                                  mapped=True)

    def after_materialize(self, ctx) -> None:
        if self.native is None and self.tasks:              # host pools are allocated once: start once
            n = (len(self.tasks) + 1) * LINE
            host, device = self.words.view(torch.uint8, (n,)), self.words.device_view(torch.uint8, (n,))
            self.views = host, device                      # the words stay mapped while this object lives
            self.native = self.lib.Worker(host.data_ptr(), device.data_ptr(), [t.native for t in self.tasks],
                                          self.cores, self.timeout_ns, self.memop)

    def shutdown(self) -> None:                     # (idempotent: a store may stop its I/O worker first)
        if self.left or self.native is None:
            return
        self.left = self.native.stop()
        if self.left:
            # A coordinator stuck in a task past the time bound was left behind: it may still write the words and its
            # tasks' buffers, so this object (and through its tasks, theirs) is leaked on purpose, never freed.
            ctypes.pythonapi.Py_IncRef(ctypes.py_object(self))


class CpuTask:
    """`cpu_task@1`'s instance over the engine's worker of kind `pool`."""

    def __init__(self, ctx, *, pool: str) -> None:
        self.service = ctx.shared(f"cpu_task.{pool}", functools.partial(Service, pool=pool))
        self.errors = ctx.static(f"cpu_task.{pool}.errors", (1,), torch.int32)

    def bind(self, task) -> int:
        return self.service.bind(task)

    def signal(self, slot: int) -> None:
        self.service.native.signal(slot, self.errors)

    def wait(self, slot: int) -> None:
        self.service.native.wait(slot, self.errors)

    def copy(self, roles) -> CopyTask:
        """The generic copy task (F9) over `roles`: per role (descriptors, units, requests), host tensors."""
        return CopyTask(self.service.lib.CopyTask(list(roles)))


class CopyTask:
    """A `cpu_task@1` task: callable (runs now, on one thread), `native` its ABI handle."""

    def __init__(self, task) -> None:
        self._task, self.native = task, task.native()

    def __call__(self) -> None:
        if self._task.run(1):
            raise RuntimeError("cpu_task@1's copy task failed (a bad request or a short read)")


def parse_cores(text: str) -> list[int]:
    """"2-5,8" -> [2, 3, 4, 5, 8]."""
    out = []
    for part in filter(None, (p.strip() for p in text.split(","))):
        a, _, b = part.partition("-")
        out += range(int(a), int(b or a) + 1)
    return out


def default_cores(pool: str, n: int, compute: int) -> list[int]:
    """One CPU per physical core of the process's, past the first: the compute threads first, then the I/O ones
    (wrapping round when there are too few). A thread pinned to one CPU (an engine loop's `loop_core`, before a second
    engine of the process is built) leaves the choice to the machine's online CPUs."""
    allowed, seen, cores = sorted(os.sched_getaffinity(0)), set(), []
    if len(allowed) == 1:
        allowed = parse_cores(pathlib.Path("/sys/devices/system/cpu/online").read_text().strip())
    for cpu in allowed:
        siblings = pathlib.Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
        core = siblings.read_text().strip() if siblings.is_file() else str(cpu)
        if core not in seen:
            seen.add(core)
            cores.append(cpu)
    rest = cores[1:] or cores
    start = 0 if pool == "compute" else compute
    return [rest[(start + i) % len(rest)] for i in range(n)]


class Plugin:
    def register(self, regs) -> None:
        regs.kernels.add(OP, KernelSpec(
            op=OP, provider="kernels.cpu_task", impl=CpuTask, requires=Constraint(sms=frozenset(extension.ARCHS)),
            supports=lambda f: f["pool"] in POOLS, props=frozenset({"graph_safe", "deterministic"}), priority=10))
