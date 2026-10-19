"""std.state_tier's I/O threads (docs/v1-kv-tiers.md D7's I/O order, D11's read priority).

Jobs run in the order the tier decides them, on `n` threads, with one exception: while a read is queued, no write
starts except one a queued read depends on. A job submitted to the read queue takes the queued writes it depends on
(its dependencies' tickets, and theirs) ahead of it, so a job never waits for one queued behind it; a job first waits
for its dependencies (earlier jobs' tickets, device fences, on the host).

The threads release the GIL in `os.preadv` / `os.pwritev` and in `zlib.crc32` of large buffers. A thread that inherits
an engine loop pinned to one CPU (`loop_core`) moves to the machine's other online CPUs."""
from __future__ import annotations

import collections
import os
import pathlib
import threading
from collections.abc import Callable, Iterable


class Ticket:
    """The end of one I/O job: `done()` without waiting, `wait()` on the host (re-raising the job's error)."""

    def __init__(self, job: Job | None = None) -> None:
        self._event, self.error, self.job = threading.Event(), None, job

    def done(self) -> bool:
        return self._event.is_set()

    def wait(self) -> None:
        self._event.wait()
        if self.error is not None:
            raise RuntimeError("state_tier: an I/O job failed") from self.error

    synchronize = wait


class Job:
    """`deps`: what it waits for first, on its thread: fences or tickets (`synchronize`), or callables."""

    def __init__(self, fn: Callable[[], object], deps: Iterable[object], read: bool) -> None:
        self.fn, self.deps, self.read, self.ticket = fn, list(deps), read, Ticket(self)
        self.then: list[Callable[[], object]] | None = []          # run once it is done (None: done)


def _cpus() -> set[int]:
    """The CPUs an I/O thread may run on: the process's, or every online one but the loop's if it is pinned to one."""
    allowed = os.sched_getaffinity(0)
    if len(allowed) > 1:
        return allowed
    online = set()
    for part in pathlib.Path("/sys/devices/system/cpu/online").read_text().strip().split(","):
        a, _, b = part.partition("-")
        online |= set(range(int(a), int(b or a) + 1))
    return (online - allowed) or allowed


class IoThreads:
    def __init__(self, n: int, name: str = "state_tier.io") -> None:
        self.cv = threading.Condition()
        self.reads: collections.deque[Job] = collections.deque()     # reads, and the writes they depend on
        self.writes: collections.deque[Job] = collections.deque()    # the other writes: while no read is queued
        self.closed, self.cpus = False, _cpus()
        self.threads = [threading.Thread(target=self._loop, name=f"{name}.{i}", daemon=True) for i in range(n)]
        for t in self.threads:
            t.start()

    def submit(self, fn: Callable[[], object], deps: Iterable[object] = (), read: bool = False) -> Job:
        job = Job(fn, deps, read)
        with self.cv:
            if read:
                self._ahead(job)
            (self.reads if read else self.writes).append(job)
            self.cv.notify()
        return job

    def urgent(self, job: Job) -> None:
        """A queued write a read depends on: it moves to the read queue, after the writes it depends on."""
        with self.cv:
            if job in self.writes:
                self.writes.remove(job)
                self._ahead(job)
                self.reads.append(job)
                self.cv.notify()

    def _ahead(self, job: Job) -> None:
        """Move the queued writes `job` depends on (and theirs) to the read queue, dependencies first."""
        for dep in job.deps:
            other = getattr(dep, "job", None)
            if isinstance(other, Job) and other in self.writes:
                self.writes.remove(other)
                self._ahead(other)
                self.reads.append(other)

    def then(self, job: Job | None, fn: Callable[[], object]) -> None:
        """Run `fn` once `job` has run: at once if it has (or None), else on its thread right after it, before its
        ticket is done (so a job waiting for that ticket runs after `fn`)."""
        with self.cv:
            if job is not None and job.then is not None:
                job.then.append(fn)
                return
        fn()

    def drain(self) -> None:
        """Wait for every job submitted so far."""
        with self.cv:
            jobs = [*self.reads, *self.writes]
        for job in jobs:
            job.ticket._event.wait()

    def close(self) -> None:
        self.drain()
        with self.cv:
            self.closed = True
            self.cv.notify_all()
        for t in self.threads:
            t.join()

    def _take(self) -> Job | None:
        with self.cv:
            while not (self.reads or self.writes or self.closed):
                self.cv.wait()
            if self.reads:
                return self.reads.popleft()
            return self.writes.popleft() if self.writes else None

    def _loop(self) -> None:
        os.sched_setaffinity(0, self.cpus)
        while (job := self._take()) is not None:
            try:
                for dep in job.deps:
                    (dep.synchronize if hasattr(dep, "synchronize") else dep)()
                job.fn()
            except BaseException as e:  # noqa: BLE001 — kept on the ticket, raised where it is waited for
                job.ticket.error = e
            with self.cv:
                then, job.then = job.then, None
            for fn in then:                          # before its ticket: whatever waits for it sees them done
                fn()
            job.ticket._event.set()
