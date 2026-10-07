"""Scheduling: which requests run in the next step and how many tokens each (design §7.1).

v0 policy (FCFS, no phase mixing): if waiting requests can be admitted the step prefills them, in chunks bounded by
the token budget; otherwise it decodes every running request.

Requests are ranked running (oldest first), then waiting (queue order). When a request cannot get the state it
needs, and releasing the admitted requests below it would make room, the lowest-ranked of them is preempted by
recompute, repeatedly; otherwise a running request preempts itself and a waiting one ends the step's prefill.

A waiting request is first admitted (a slot, plus its longest cached prefix), then reserves state for its chunk.

Liveness: the pools always hold one request of the engine's maximum length, and every unit is held by some request
or only by the prefix cache, which evicts such units before anyone is preempted. So the top-ranked request fits once
everything below it is preempted, and every step makes progress.
"""
from __future__ import annotations

import collections

import numpy as np

from holon.api import BatchPlan, Seq, Status

from .batching import host_plan
from .state import StateManager


class Scheduler:
    def __init__(self, state: StateManager, max_step_tokens: int) -> None:
        self.state, self.max_step_tokens = state, max_step_tokens
        self.waiting: collections.deque[Seq] = collections.deque()
        self.running: list[Seq] = []
        self.preemptions = 0
        self._step = 0

    def add(self, seq: Seq, front: bool = False) -> None:
        seq.status = Status.WAITING
        if front:
            self.waiting.appendleft(seq)
        else:
            self.waiting.append(seq)

    def schedule(self) -> tuple[BatchPlan, list[Seq]] | None:
        rows = self._prefill() or self._decode()
        if not rows:
            return None
        seqs = [s for s, _ in rows]
        n_query = np.array([n for _, n in rows], dtype=np.int32)
        starts = np.array([s.num_computed for s in seqs], dtype=np.int32)
        n_logits = (starts + n_query == np.array([len(s.tokens) for s in seqs])).astype(np.int32)
        tokens = np.concatenate([s.tokens[st:st + n] for s, st, n in zip(seqs, starts, n_query, strict=True)])
        slots = np.array([s.slot for s in seqs], dtype=np.int32)
        self._step += 1
        return host_plan(self._step, slots, starts, n_query, n_logits, tokens.astype(np.int64)), seqs

    def _prefill(self) -> list[tuple[Seq, int]]:
        rows, budget = [], self.max_step_tokens
        for seq in list(self.waiting):
            if budget == 0 or (seq.slot < 0 and not self._fit(seq, None)):
                break
            n = min(len(seq.tokens) - seq.num_computed, budget)       # after admission: a prefix hit counts
            if not self._fit(seq, n):
                break
            rows.append((seq, n))
            budget -= n
        return rows

    def _decode(self) -> list[tuple[Seq, int]]:
        rows = []
        for seq in list(self.running):
            if seq.status is not Status.RUNNING:  # preempted to make room for a higher-ranked request
                continue
            if self._fit(seq, 1):
                rows.append((seq, 1))
            else:
                self._preempt(seq)
        return rows

    def _fit(self, seq: Seq, n: int | None) -> bool:
        """Admit `seq` (n None) or reserve state for `n` more of its tokens, preempting lower-ranked admitted
        requests, lowest first, while releasing them can make room."""
        while not (self.state.admit(seq) if n is None else self.state.reserve(seq, n)):
            ranked = self.running + list(self.waiting)
            below = [s for s in ranked[ranked.index(seq) + 1:] if s.slot >= 0]
            if not below or not self.state.could_fit(seq, n or 0, below):
                return False
            self._preempt(below[-1])
        return True

    def _preempt(self, seq: Seq) -> None:
        self.state.release(seq)
        seq.num_computed = 0
        self.preemptions += 1
        if seq.status is Status.RUNNING:  # back to the head of the queue, ahead of requests never run
            self.running.remove(seq)
            self.waiting.appendleft(seq)
            seq.status = Status.PREEMPTED

    def finished_prefill(self, seq: Seq) -> None:
        self.waiting.remove(seq)
        seq.status = Status.RUNNING
        self.running.append(seq)

    def finish(self, seq: Seq, reason: str) -> None:
        (self.running if seq.status is Status.RUNNING else self.waiting).remove(seq)
        self.state.release(seq)
        seq.status, seq.finish_reason = Status.FINISHED, reason
