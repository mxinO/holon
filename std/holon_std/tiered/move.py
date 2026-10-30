"""D8x (docs/dev/v1-glm.md): a short prompt step parks its lent experts in the idle read-ahead rows; at its end each is
paired with RAM's coldest expert of its classes (strictly colder), whose units it reserves; one host thread then copies
the rows into those units, chunk by chunk, hottest first, and the store maps chunk i at launch (the prompt step + LAG +
i) — placement depends on the step count only. This module holds a move's bookkeeping and the thread; the device side
is the store's (`TieredStore._move_start`, `_move_due`)."""
from __future__ import annotations

import os
import queue
import threading
import time

import numpy as np

CHUNK = 50                  # experts a chunk (C: about one decode token's copy at the deployment's sizes)
LAG = 2                     # chunk i is mapped at launch (the prompt step + LAG + i): with overlap, launch + 1 is made
#                             before the prompt step's device work ends, so no copy can have begun by then
NICE = 10                   # the thread's niceness: it yields to the engine loop and the workers


class Move:
    """One prompt step's move: its table on the device [room, 2 + R] (item, row, its reserved unit per role; hottest
    first) and that table and its count copied into pinned host memory, valid once `event` has passed (None: a CPU
    device, at once). `entries` is the host's copy (`Mover.entries`); the thread sets `copied` (chunks), the store
    `mapped` (chunks, each one's launch in `mapped_at`)."""

    def __init__(self, step: int, chunk: int, table, host, count, event) -> None:
        self.step, self.chunk, self.table, self.host, self.count, self.event = step, chunk, table, host, count, event
        self.entries: np.ndarray | None = None
        self.copied, self.mapped, self.mapped_at = 0, 0, []
        self.error: BaseException | None = None

    @property
    def chunks(self) -> int:
        """Its chunks (its entries known: `entries`)."""
        return -(-len(self.entries) // self.chunk)


class Mover:
    """The move thread: one named daemon thread at low priority, off the engine loop's and the I/O worker's cores
    (`cores`), copying each submitted move's chunks with `copy(move, entries)` (numpy copies of pinned views: the GIL
    released per piece); an error is kept on the move (the store raises it at its next commit or join), never lost."""

    def __init__(self, name: str, copy, cores: set[int]) -> None:
        self._copy, self._cores = copy, cores
        self._jobs: queue.SimpleQueue = queue.SimpleQueue()
        self._cond = threading.Condition()
        self.thread = threading.Thread(target=self._run, name=f"{name}.move", daemon=True)
        self.thread.start()

    def submit(self, move: Move) -> None:
        self._jobs.put(move)

    def entries(self, move: Move) -> np.ndarray:
        """`move`'s entries, read from its host table once its event has passed (by whichever of the thread and the
        caller comes first)."""
        if move.entries is None:
            if move.event is not None:
                move.event.synchronize()
            entries = move.host[:int(move.count[0])].numpy().copy()
            with self._cond:
                if move.entries is None:
                    move.entries = entries
                self._cond.notify_all()
        return move.entries

    def wait(self, move: Move, chunks: int | None = None) -> int:
        """Until `move`'s entries are known and its first `chunks` chunks copied (None: all of them); the ns this
        waited (0: none). A thread error raises."""
        def ready() -> bool:
            if move.error is not None:
                return True
            if move.entries is None:
                return False
            return move.copied >= (move.chunks if chunks is None else min(chunks, move.chunks))
        with self._cond:                     # (the lock orders the thread's copies before what the caller does next)
            if ready():
                waited = 0
            else:
                t = time.perf_counter_ns()
                self._cond.wait_for(ready)
                waited = max(time.perf_counter_ns() - t, 1)
        if move.error is not None:
            raise RuntimeError(f"the D8x move thread failed: {move.error!r}") from move.error
        return waited

    def stop(self) -> None:
        """After the engine's drain: the thread finishes its moves and ends."""
        self._jobs.put(None)
        self.thread.join()

    def _run(self) -> None:
        try:                                 # (Linux: both apply to the calling thread)
            if self._cores:
                os.sched_setaffinity(0, self._cores)
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), NICE)
        except (AttributeError, OSError):
            pass
        while (move := self._jobs.get()) is not None:
            try:
                entries = self.entries(move)
                for i in range(move.chunks):
                    self._copy(move, entries[i * move.chunk:(i + 1) * move.chunk])
                    with self._cond:         # (release: the chunk's copies before its flag)
                        move.copied = i + 1
                        self._cond.notify_all()
            except BaseException as e:      # kept, raised by the store
                with self._cond:
                    move.error = e
                    self._cond.notify_all()


def mover_cores(io_cores: list[int] | tuple[int, ...]) -> set[int]:
    """The move thread's CPUs: the process's (the machine's online ones where the engine loop pinned itself to one),
    less the loop's and the I/O worker's — all of them where nothing is left."""
    allowed = os.sched_getaffinity(0)
    every = set(range(os.cpu_count() or 1)) if len(allowed) == 1 else set(allowed)
    rest = every - (allowed if len(allowed) == 1 else set()) - set(io_cores)
    return rest or every
