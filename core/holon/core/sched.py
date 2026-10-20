"""Scheduling: which requests run in the next step and how many tokens each (design §7.1).

Policy (FCFS): if waiting requests can be admitted the step prefills them, in chunks bounded by the token budget;
otherwise it decodes every running request — unless steps mix (below).

Requests are ranked running (oldest first), then waiting (queue order). When a request cannot get the state it
needs, and releasing the admitted requests below it would make room, the lowest-ranked of them is preempted by
recompute, repeatedly; otherwise a running request preempts itself and a waiting one ends the step's prefill.

A waiting request is first admitted (a slot, plus its longest cached prefix), then reserves state for its chunk.

With overlap, the step being scheduled runs while the previous one is still on the device: a request's tokens are
its committed ones plus `pending` sampled ones not yet committed. A decode row then reads its input token from the
device (TokenSrc.DEVICE_LAST); a waiting request with a pending token (preempted in flight) waits for its commit; a
request that reaches its length limit with its pending tokens is not scheduled again — its commit finishes it.

Speculative decoding: the draft policy picks k per step (`draft_k`, from the running requests); every row of a
decode step is then a verify row of 1 + k tokens that all produce logits, and how many of its drafts are real is
on the device (`SeqTable.n_draft`). A row reserves state for its verify tokens plus what the drafter writes ahead
of them (`lookahead(k)`), and a prompt-ending prefill chunk for its tokens plus that lookahead — the drafter drafts
after it too; reservations stop at `max_model_len` (positions past it compute nowhere real). A request with an
uncommitted verify row knows its length only on the device; steps whose decode kernel plans from host lengths
(`host_lens_only`) leave it out until that row commits.

Constrained requests (a logits processor masks their rows from their committed tokens) run at depth 0: one is left
out while it has a row in flight (the engine may commit that row first), and while BLOCKED (its processor is not
ready), without holding up the requests queued behind it.

Mixed steps (`chunk_with_decode` C > 0; docs/m07-perf.md §3.1): without a drafter, a step whose running requests
decode also carries prefill chunks — its decode rows first (`BatchPlan.n_decode` of them, each reserved before any
chunk), then chunks of up to C tokens (the engine checks `max_step_tokens` >= max_seqs + C, so a step always has room
for them); a step with nothing to decode prefills up to `max_step_tokens`, as today. With a drafter rows never mix: a
prefill step capped at C alternates with a decode (verify) step while both kinds can run, else the step takes the
kind that can.

Load-gated requests (a prefix hit the tier loads, docs/v1-kv-tiers.md D4): admitted, they wait until the tier's clock
reaches their loads' due time (`clock`, `StateManager.due`) while the requests behind them run; when nothing else can
run, the earliest-due one runs at once and its step waits for its copies on the device.

Liveness: the pools always hold one request of the engine's maximum length, and every unit is held by some request
or only by the prefix cache, which evicts such units before anyone is preempted. So the top-ranked request fits once
everything below it is preempted, and every step makes progress.
"""
from __future__ import annotations

import collections
from collections.abc import Callable

import numpy as np

from holon.api import BatchPlan, RowKind, Seq, Status, TokenSrc

from .batching import host_plan
from .state import StateManager


def _host_tokens(seq: Seq, start: int, n: int, on_device: bool) -> list[int]:
    """A row's input tokens as far as the host knows them; the rest (the last sampled token while it is pending,
    a verify row's drafts) are placeholders the device fills in."""
    known = [] if on_device else seq.tokens[start:start + n]
    return known + [0] * (n - len(known))


class Scheduler:
    def __init__(self, state: StateManager, max_step_tokens: int, max_model_len: int,
                 draft_k: Callable[[list[Seq]], int] | None = None, lookahead: Callable[[int], int] | None = None,
                 host_lens_only: bool = False, chunk_with_decode: int = 0,
                 clock: Callable[[], float] = lambda: 0.0) -> None:
        self.state, self.max_step_tokens, self.max_model_len = state, max_step_tokens, max_model_len
        self.clock = clock
        self.chunk_with_decode = chunk_with_decode
        self._prefilled = False                   # with a drafter and C > 0: the last step prefilled (alternation)
        self.draft_k, self.host_lens_only = draft_k, host_lens_only
        self.lookahead = lookahead or (lambda k: 0)
        self._k = 0                               # the step being scheduled: its plan's draft_k
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
        rows, n_decode, decoding = self._rows()
        while not rows and self.state.due:          # only load-gated requests left: the earliest-due runs now
            del self.state.due[min(self.state.due, key=self.state.due.get)]
            rows, n_decode, decoding = self._rows()
        if not rows:
            return None
        seqs, n_query, n_logits = (list(c) for c in zip(*rows, strict=True))
        starts = [s.num_computed for s in seqs]
        device = [s.pending > 0 for s in seqs]                             # decode input still on the device
        if all(n == 1 for n in n_query):          # the common decode step: one input per row
            tokens = [0 if dev else s.tokens[st] for s, st, dev in zip(seqs, starts, device, strict=True)]
        else:
            tokens = [t for s, st, n, dev in zip(seqs, starts, n_query, device, strict=True)
                      for t in _host_tokens(s, st, n, dev)]
        src = np.where(device, TokenSrc.DEVICE_LAST, TokenSrc.HOST).astype(np.int8)
        self._step += 1
        plan = host_plan(self._step, np.array([s.slot for s in seqs], dtype=np.int32), np.array(starts, np.int32),
                         np.array(n_query, np.int32), np.array(n_logits, np.int32), np.array(tokens, np.int64), src)
        plan.draft_k, plan.n_decode = self._k, n_decode
        kind = np.where(plan.n_query > 1, RowKind.VERIFY, RowKind.DECODE)            # the leading decoding rows
        plan.row_kind = np.where(np.arange(len(seqs)) < decoding, kind, RowKind.PREFILL).astype(np.int8)
        return plan, seqs

    def _rows(self) -> tuple[list[tuple[Seq, int, int]], int, int]:
        """The step's rows (seq, query tokens, logits rows), how many leading ones are mixed-in decode rows, and how
        many leading ones decode or verify (`BatchPlan.row_kind`)."""
        cap = self.chunk_with_decode
        if not cap:
            rows = self._prefill(self.max_step_tokens)
            return (rows, 0, 0) if rows else self._decoding(self._decode())
        if self.draft_k is None:                  # mixed: decode rows first, then a capped chunk
            decode = self._decode()
            if not decode:
                return self._prefill(self.max_step_tokens), 0, 0
            chunks = self._prefill(min(cap, self.max_step_tokens - len(decode)))
            return decode + chunks, len(decode) if chunks else 0, len(decode)
        ready, cap = bool(self._ready()), min(cap, self.max_step_tokens)    # a drafter: alternate while both can run
        if self._prefilled and ready:
            rows = self._decode()
            self._prefilled = not rows
            return self._decoding(rows) if rows else (self._prefill(cap), 0, 0)
        rows = self._prefill(cap if ready else self.max_step_tokens)
        self._prefilled = bool(rows)
        return (rows, 0, 0) if rows else self._decoding(self._decode())

    @staticmethod
    def _decoding(rows: list[tuple[Seq, int, int]]) -> tuple[list[tuple[Seq, int, int]], int, int]:
        return rows, 0, len(rows)

    def _prefill(self, budget: int) -> list[tuple[Seq, int, int]]:
        rows, ahead, now = [], None, self.clock()
        self._k = 0
        for seq in list(self.waiting):
            if seq.pending or seq.status is Status.BLOCKED:      # preempted while its decode was in flight: wait for
                continue                                         # the commit; or its logits processor is not ready
            if budget == 0 or (seq.slot < 0 and not self._fit(seq, None)):
                break
            if self.state.due.get(seq.slot, now) > now:          # load-gated
                continue
            n = min(len(seq.tokens) - seq.num_computed, budget)       # after admission: a prefix hit counts
            n = self.state.cut(seq, n)                                 # end where a state kind publishes
            last = seq.num_computed + n == len(seq.tokens)
            if last and ahead is None:            # the step's k, asked once: the drafter drafts after this chunk
                k = self.draft_k(self.running + [s for s, _, _ in rows] + [seq]) if self.draft_k is not None else 0
                ahead = self.lookahead(k) if k else 0
            if not self._fit(seq, min(n + ahead, self.max_model_len - seq.num_computed) if last else n):
                break
            rows.append((seq, n, int(last)))
            if last:                              # a row the drafter drafts after: the step speculates at k
                self._k = k
            budget -= n
        return rows

    def _ready(self) -> list[Seq]:
        """The running requests a decode step may take."""
        return [s for s in self.running if not self._at_limit(s) and not (self.host_lens_only and s.inexact)
                and not (s.constrained and s.pending)]

    def _decode(self) -> list[tuple[Seq, int, int]]:
        ready = self._ready()
        self._k = k = self.draft_k(ready) if self.draft_k is not None and ready else 0
        need = 1 + k + (self.lookahead(k) if k else 0)
        rows, capacity, limit = [], self.state.capacity, self.max_model_len
        for seq in ready:
            if seq.status is not Status.RUNNING:                       # preempted by a row above it
                continue
            n = min(need, limit - seq.num_computed)
            # StateManager.reserve's own fast path, inline: no call for the many rows whose units already fit.
            if seq.num_computed + n <= capacity.get(seq.slot, -1) or self._fit(seq, n):
                rows.append((seq, 1 + k, 1 + k if k else 1))
            else:
                self._preempt(seq)
        return rows

    def _at_limit(self, seq: Seq) -> bool:
        length = len(seq.tokens) + seq.pending
        return length >= self.max_model_len or (
            seq.params.max_tokens is not None and length - seq.prompt_len >= seq.params.max_tokens)

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
