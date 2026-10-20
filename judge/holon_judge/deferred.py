"""A deferred copy stream for the CPU, for the judge's tier checks and core tests (docs/v1-kv-tiers.md, panel A11): on
the CPU the core's streams run their work at once, so a copy nobody waits for still lands in time and a missing wait
passes by luck. Here work enqueued on a stream runs only when a fence after it is waited for (`wait`, `synchronize`) or
when `pump` (one per launch) finds it due — a seeded 0 … `most` launches after it was enqueued, in order on its stream.
A step's own work still runs at once, as on the CPU, so a launch that forgets to wait reads what was there before the
copy.

    deferred = Deferred(seed, most=2)
    monkeypatch.setattr(platform, "Stream", lambda device: deferred.stream())
"""
from __future__ import annotations

import random
import threading
from collections.abc import Callable, Sequence


class Job:
    """A fence: the end of one enqueued function."""

    def __init__(self, stream: DeferredStream, fn: Callable[[], object], after: Sequence, at: int) -> None:
        self.stream, self.fn, self.after, self.at, self.ran, self.claimed = stream, fn, list(after), at, False, False

    def wait(self) -> None:
        self.stream.until(self)

    synchronize = wait

    def done(self) -> bool:
        return self.ran


class DeferredStream:
    def __init__(self, owner: Deferred) -> None:
        self.owner, self.queue = owner, []

    def run(self, fn: Callable[[], object], after: Sequence = ()) -> Job:
        with self.owner.cv:
            job = Job(self, fn, after, self.owner.launches + self.owner.rng.randint(0, self.owner.most))
            self.queue.append(job)
        return job

    def until(self, job: Job) -> None:
        """Run this stream's jobs up to `job` (each after its `after` fences). Threads may call it (a tier's I/O
        threads wait for copies too): one runs each job, the others wait for it; no lock is held while a job waits."""
        cv = self.owner.cv
        while True:
            with cv:
                cv.wait_for(lambda: job.ran or not self.queue[0].claimed)
                if job.ran:
                    return
                head = self.queue[0]
                head.claimed = True
            for fence in head.after:
                fence.wait()
            head.fn()
            with cv:
                self.queue.pop(0)
                head.ran = True
                cv.notify_all()


class Deferred:
    def __init__(self, seed: int = 0, most: int = 2) -> None:
        self.rng, self.most, self.launches, self.streams = random.Random(seed), most, 0, []
        self.cv = threading.Condition()

    def stream(self) -> DeferredStream:
        self.streams.append(DeferredStream(self))
        return self.streams[-1]

    def pump(self) -> None:
        """A launch went by: what is due by now lands, in order on each stream."""
        with self.cv:
            self.launches += 1
        for stream in self.streams:
            while (head := self._head(stream)) is not None and head.at <= self.launches:
                stream.until(head)

    def _head(self, stream: DeferredStream) -> Job | None:
        with self.cv:
            return next((j for j in stream.queue if not j.ran), None)

    def drain(self) -> None:
        for stream in self.streams:
            with self.cv:
                last = stream.queue[-1] if stream.queue else None
            if last is not None:
                stream.until(last)
