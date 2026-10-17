"""Scheduler liveness and preemption order, against a fake state manager counting tokens."""
import random

import numpy as np
import pytest
import torch
from holon.api import RowKind, SamplingParams, Seq, Status
from holon.core.memory import Ledger, OutOfUnits
from holon.core.sched import Scheduler


class FakeState:
    """`capacity` tokens and `slots` request slots; a request holds one unit per token."""

    def __init__(self, capacity: int, slots: int) -> None:
        self.free, self.slots, self.held = capacity, list(range(slots)), {}
        self.capacity: dict[int, int] = {}       # no fast path: every reservation is checked

    def cut(self, seq: Seq, n: int) -> int:
        return n

    def admit(self, seq: Seq) -> bool:
        if not self.slots:
            return False
        seq.slot = self.slots.pop()
        self.held[seq.slot] = 0
        return True

    def reserve(self, seq: Seq, n: int) -> bool:
        if n > self.free:
            return False
        self.free -= n
        self.held[seq.slot] += n
        return True

    def could_fit(self, seq: Seq, n: int, releasing: list[Seq]) -> bool:
        slot_ok = seq.slot >= 0 or bool(self.slots) or any(s.slot >= 0 for s in releasing)
        return slot_ok and n <= self.free + sum(self.held[s.slot] for s in releasing if s.slot >= 0)

    def release(self, seq: Seq) -> None:
        if seq.slot >= 0:
            self.free += self.held.pop(seq.slot)
            self.slots.append(seq.slot)
            seq.slot = -1


def _run(sched: Scheduler, seqs: list[Seq], max_steps: int = 10_000) -> int:
    """Drive the scheduler like the engine (token = 0); returns the steps taken. Checks each step's shape: a mixed
    step (`n_decode` > 0) is its decode rows, then capped chunks; with a drafter, no step mixes and a capped prefill
    step is followed by a decode step whenever a running request could decode."""
    for seq in seqs:
        sched.add(seq)
    cap, drafting, prefilled = sched.chunk_with_decode, sched.draft_k is not None, False
    for step in range(max_steps):
        if not (sched.waiting or sched.running):
            return step
        could_decode = bool(sched._ready())
        plan, rows = sched.schedule()
        decoding = [seq.status is Status.RUNNING for seq in rows]
        n, chunk = plan.n_decode, int(plan.n_query[plan.n_decode:].sum())
        assert decoding == [True] * n + [False] * (len(rows) - n) or (not n and all(decoding)), step
        kinds = [RowKind.PREFILL if not d else RowKind.VERIFY if q > 1 else RowKind.DECODE
                 for d, q in zip(decoding, plan.n_query, strict=True)]
        assert plan.row_kind.tolist() == kinds, step                        # every step, mixed or not
        assert int(plan.n_query.sum()) <= max(sched.max_step_tokens, len(rows) if all(decoding) else 0)
        assert (n > 0) == (0 < n < len(rows)) and (not n or (cap and not drafting))
        if not cap:
            assert all(decoding) or not any(decoding)                       # today: never mixed
        elif n:
            assert (plan.n_query[:n] == 1).all() and chunk <= cap
        elif drafting and not any(decoding):
            assert not (prefilled and could_decode) and chunk <= (cap if could_decode else sched.max_step_tokens)
        prefilled = not any(decoding)
        for seq, n, sampled in zip(rows, plan.n_query, plan.n_logits, strict=True):
            assert seq.num_computed == plan.start_upper[list(rows).index(seq)]
            seq.num_computed += int(n)
            if sampled:
                seq.tokens.append(0)
                if seq.status is not Status.RUNNING:
                    sched.finished_prefill(seq)
                if len(seq.tokens) - seq.prompt_len >= seq.params.max_tokens:
                    sched.finish(seq, "length")
    raise AssertionError("requests did not finish")


def _seq(rid: int, prompt: int, gen: int) -> Seq:
    return Seq(str(rid), [1] * prompt, SamplingParams(max_tokens=gen), prompt_len=prompt)


@pytest.mark.parametrize("mode", ["today", "mixed", "alternating"])
def test_every_request_finishes_when_one_max_request_fits(mode):
    """Liveness over 40 random workloads, with mixed steps (`chunk_with_decode` > 0) and, with a drafter, alternating
    steps too."""
    for seed in range(40):
        rng = random.Random(seed)
        max_len = 40
        seqs = [_seq(i, p := rng.randint(1, max_len - 1), rng.randint(1, max_len - p))
                for i in range(rng.randint(1, 12))]
        state = FakeState(capacity=rng.randint(max_len, 3 * max_len), slots=(slots := rng.randint(1, 4)))
        cap = rng.randint(1, 20) if mode != "today" else 0      # the engine's check: max_step_tokens >= slots + cap
        _run(Scheduler(state, max_step_tokens=rng.randint(slots + cap if mode == "mixed" else 1, 50),
                       max_model_len=10**9, draft_k=(lambda seqs: 0) if mode == "alternating" else None,
                       chunk_with_decode=cap), seqs)
        assert all(s.status is Status.FINISHED for s in seqs), seed
        assert state.held == {} and len(state.slots) > 0, seed


def _commit(plan, rows, sched):
    for seq, n, sampled in zip(rows, plan.n_query, plan.n_logits, strict=True):
        seq.num_computed += int(n)
        if sampled:
            seq.tokens.append(0)
            if seq.status is not Status.RUNNING:
                sched.finished_prefill(seq)


def test_waiting_head_preempts_partial_prefill_behind_it():
    # B is mid-prefill holding most of the memory when preempted A returns to the head: B must yield, not deadlock.
    state = FakeState(capacity=8, slots=3)
    sched = Scheduler(state, max_step_tokens=8, max_model_len=10**9)
    a, b = _seq(0, 2, 8), _seq(1, 9, 1)
    sched.add(a)
    sched.add(b)
    _commit(*sched.schedule(), sched)     # A prefills 2 and samples, B prefills 6 of 9 → all 8 held
    state.release(a)                      # A preempted by recompute: back to the head of the queue
    sched.running.remove(a)
    a.num_computed, a.status = 0, Status.PREEMPTED
    sched.waiting.appendleft(a)
    plan, rows = sched.schedule()         # A needs 3 tokens, 2 are free: B is preempted and restarts after A
    assert rows == [a, b] and list(plan.start_upper) == [0, 0] and sched.preemptions == 1


def test_decode_preempts_lowest_ranked_first():
    state = FakeState(capacity=6, slots=3)
    sched = Scheduler(state, max_step_tokens=100, max_model_len=10**9)
    seqs = [_seq(i, 2, 5) for i in range(3)]
    for s in seqs:
        sched.add(s)
    _commit(*sched.schedule(), sched)     # all three prefilled: 6 of 6 held
    plan, rows = sched.schedule()         # decode: the newest is preempted to make room
    assert rows == seqs[:2] and list(sched.waiting) == seqs[2:]
    _commit(plan, rows, sched)
    plan, rows = sched.schedule()         # the preempted one cannot come back yet; the next newest yields
    assert rows == seqs[:1] and list(sched.waiting) == seqs[1:]
    assert all(s.status is Status.PREEMPTED and s.num_computed == 0 for s in seqs[1:])


def test_no_progress_returns_none():
    sched = Scheduler(FakeState(capacity=3, slots=1), max_step_tokens=8, max_model_len=10**9)
    sched.add(_seq(0, 5, 1))
    assert sched.schedule() is None


def test_ledger_splits_fill_pools_by_tokens():
    ledger = Ledger(device=torch.device("cpu"))
    slot = ledger.request("slot", unit_bytes=100, min_units=4, pref_units=4)
    kv = ledger.request("kv", unit_bytes=64, segments=4, min_units=2, tokens_per_unit=16)
    idx = ledger.request("idx", unit_bytes=8, min_units=1, tokens_per_unit=64)
    ledger.materialize(400 + 64 * 32 + 8 * 8)
    assert (slot.capacity, kv.capacity, idx.capacity) == (4, 32, 8)  # 512 tokens each
    with pytest.raises(OutOfUnits, match="kv"):
        ledger.materialize(400 + 64)


def test_pool_refcounts_and_copy():
    ledger = Ledger(device=torch.device("cpu"))
    pool = ledger.request("p", unit_bytes=8, segments=2, min_units=4, tokens_per_unit=1)
    ledger.materialize(None)
    units = pool.alloc(2)
    pool.incref(units[:1])
    pool.decref(np.concatenate([units, units[:1]]))   # duplicate ids in one call
    assert pool.num_free == 4
    with pytest.raises(RuntimeError, match="twice"):
        pool.decref(units[:1])
    data = pool.view(torch.uint8, (2, 4, 4))           # [segments, units, bytes per segment]
    data[:, 1] = 7
    pool.copy(torch.tensor([1]), torch.tensor([3]))
    assert (data[:, 3] == 7).all()


def test_hopeless_preemption_is_skipped():
    # The waiting head needs more than the waiting requests below it could free: they keep their work.
    state = FakeState(capacity=10, slots=4)
    sched = Scheduler(state, max_step_tokens=100, max_model_len=10**9)
    running, partial = _seq(0, 6, 4), _seq(1, 3, 1)
    sched.add(running)
    _commit(*sched.schedule(), sched)     # running holds 6
    head = _seq(2, 8, 1)
    sched.add(partial)
    assert state.admit(partial) and state.reserve(partial, 2)   # a partial prefill holding 2; a head needing 8 jumps it
    partial.num_computed = 2
    sched.waiting.appendleft(head)
    plan, rows = sched.schedule()         # 2 free + 2 below < 8: nothing is preempted, decode runs
    assert rows == [running] and partial.num_computed == 2 and sched.preemptions == 0
