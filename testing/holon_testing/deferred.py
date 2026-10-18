"""A deferred copy stream for CPU tests (docs/v1-kv-tiers.md, panel A11): on the CPU the core's streams run their work
at once, so a copy nobody waits for still lands in time and a missing wait passes by luck. Here work enqueued on a
stream runs only when a fence after it is waited for (`wait`, `synchronize`) or when `pump` (one per launch) finds it
due — a seeded 0 … `most` launches after it was enqueued, in order on its stream. A step's own work still runs at
once, as on the CPU, so a launch that forgets to wait reads what was there before the copy.

    deferred = Deferred(seed, most=2)
    monkeypatch.setattr(platform, "Stream", lambda device: deferred.stream())
"""
from __future__ import annotations

import random
from collections.abc import Callable, Sequence


class Job:
    """A fence: the end of one enqueued function."""

    def __init__(self, stream: DeferredStream, fn: Callable[[], object], after: Sequence, at: int) -> None:
        self.stream, self.fn, self.after, self.at, self.ran = stream, fn, list(after), at, False

    def wait(self) -> None:
        self.stream.until(self)

    synchronize = wait

    def done(self) -> bool:
        return self.ran


class DeferredStream:
    def __init__(self, owner: Deferred) -> None:
        self.owner, self.queue = owner, []

    def run(self, fn: Callable[[], object], after: Sequence = ()) -> Job:
        job = Job(self, fn, after, self.owner.launches + self.owner.rng.randint(0, self.owner.most))
        self.queue.append(job)
        return job

    def until(self, job: Job) -> None:
        """Run this stream's jobs up to `job` (each after its `after` fences)."""
        while not job.ran:
            head = self.queue.pop(0)
            for fence in head.after:
                fence.wait()
            head.fn()
            head.ran = True


class Deferred:
    def __init__(self, seed: int = 0, most: int = 2) -> None:
        self.rng, self.most, self.launches, self.streams = random.Random(seed), most, 0, []

    def stream(self) -> DeferredStream:
        self.streams.append(DeferredStream(self))
        return self.streams[-1]

    def pump(self) -> None:
        """A launch went by: what is due by now lands, in order on each stream."""
        self.launches += 1
        for stream in self.streams:
            while stream.queue and stream.queue[0].at <= self.launches:
                stream.until(stream.queue[0])

    def drain(self) -> None:
        for stream in self.streams:
            if stream.queue:
                stream.until(stream.queue[-1])
