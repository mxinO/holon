"""Liveness with the real StateManager, PrefixIndex and Scheduler: shared prefixes, a tight pool, random traffic.

With a tier (the toy tier on a deferred stream: refused page stores, its own drops of interior blocks, loads due up to
three launches out) and optionally an eviction policy, the same traffic plus aborts is also the property test of
docs/v1-kv-tiers.md D6: after every step resident entries are prefix-closed, every entry's parent exists, the child
and resident-child counts match a recount and the index's mask matches its resident units; and, at random points,
`could_fit` says yes exactly when releasing the requests it was given and evicting frees the units (loads in flight,
evicted loading entries and refused eviction stores included). The tier holds no copy of a block the index dropped,
and every load is reported to it once no request runs."""
import random

import numpy as np
import pytest
import torch
from holon.api import Paged, SamplingParams, Seq, StateSpec, Status
from holon.core.memory import Ledger
from holon.core.prefix import PrefixIndex
from holon.core.sched import Scheduler
from holon.core.state import BoundSpec, StateManager, _View
from holon_testing.deferred import Deferred
from holon_testing.toy_tier.plugin import bind

PAGE, MAX_LEN = 4, 40


class _PagedKind:
    def __init__(self, spec):
        self.spec = spec

    def units_needed(self, seq, new_tokens):
        return max(0, -(-(seq.num_computed + new_tokens) // PAGE) - len(seq.units(self.spec.name)))


class _Done:
    """A launched step's end (the work runs at once here)."""

    def wait(self) -> None:
        pass

    synchronize = wait


class _ShallowFirst:
    """An eviction policy keeping the first two blocks of every chain while others can go (a root pin)."""

    def victim(self, candidates):
        first = None
        for e in candidates:
            if e.depth > 1:
                return e
            first = first or e
        return first


def _state(units: int, slots: int, tier: dict | None = None, rng: random.Random | None = None) -> StateManager:
    ledger = Ledger(torch.device("cpu"), host_memory=1 << 20)
    pool = ledger.request("kv", unit_bytes=8, min_units=units, tokens_per_unit=PAGE)
    ledger.materialize(None)
    spec = StateSpec("kv", "paged@1", Paged(8, PAGE), (0,), "test")
    index = PrefixIndex(1, PAGE, {"kv": pool}, ledger.pools)
    if tier is not None:
        bind(index, ledger, tier, Deferred(rng.randrange(1 << 30), most=2).stream)
        index.policy = _ShallowFirst() if rng.random() < 0.5 else None
    return StateManager([BoundSpec(spec, pool, _PagedKind(spec))], slots, MAX_LEN, torch.device("cpu"), index=index)


def _check(state: StateManager) -> None:
    """D6's invariants after a step."""
    idx, pool = state.index, state.bound[0].pool
    every = idx.entries | idx.lower
    assert not idx.entries.keys() & idx.lower.keys()
    for e in every.values():
        assert e.parent is None or e.parent in every
        assert e.children == {c.key for c in every.values() if c.parent == e.key}
        assert e.kids == sum(c in idx.entries for c in e.children)
    for e in idx.entries.values():
        assert e.parent is None or e.parent in idx.entries                     # resident entries are prefix-closed
    for e in idx.lower.values():
        assert e.lower and not e.units and e.loading is None
    assert {k for k, _ in idx.tier.copies} <= every.keys()               # every block the index drops goes
    held = [u for e in idx.entries.values() for u in e.units.get("kv", ())]
    assert sorted(np.flatnonzero(idx.cached("kv"))) == sorted(held) and (pool.refcount(np.array(held)) >= 1).all()


def _could_fit_agrees(state: StateManager, sched: Scheduler, rng: random.Random) -> None:
    """Ask `could_fit` for a random admitted request and the others below it, then do it: release them and evict."""
    admitted = [s for s in sched.running + list(sched.waiting) if s.slot >= 0]
    if len(admitted) < 2:
        return
    seq, releasing = admitted[0], admitted[1:]
    pool = state.bound[0].pool
    n = rng.randrange(1, 3 * MAX_LEN)
    need = state.bound[0].kind.units_needed(_View(seq, state.held[seq.slot], {}), n)
    said = state.could_fit(seq, n, releasing)
    for s in releasing:
        sched._preempt(s)
    state.index.evict("kv", need)
    assert said == (pool.num_free >= need)


def _run(state: StateManager, sched: Scheduler, seqs: list[Seq], rng: random.Random, aborts: bool) -> None:
    pending, tier = list(seqs), state.index.tier
    for _ in range(20_000):
        while pending and rng.random() < 0.5:                 # arrivals interleave with steps
            sched.add(pending.pop(0))
        if not (sched.waiting or sched.running):
            if not pending:
                return
            continue
        if tier is not None:
            state.index.poll()
        if aborts and rng.random() < 0.05:                    # an abort, gated or loading requests included
            live = [s for s in sched.running + list(sched.waiting) if not s.pending]
            if live:
                sched.finish(rng.choice(live), "abort")
                continue
        if tier is not None and rng.random() < 0.1:
            _could_fit_agrees(state, sched, rng)
        scheduled = sched.schedule()
        assert scheduled is not None, "no progress"
        plan, rows = scheduled
        loads = state.launching(rows)
        if tier is not None:
            tier.before_launch(plan)
        state.launched(loads, _Done())
        for seq, n, sampled in zip(rows, plan.n_query, plan.n_logits, strict=True):
            seq.num_computed += int(n)
            state.publish(seq)
            if sampled:
                seq.tokens.append(rng.randrange(3))           # few distinct tokens: generations share blocks too
                if seq.status is not Status.RUNNING:
                    sched.finished_prefill(seq)
                if len(seq.tokens) - seq.prompt_len >= seq.params.max_tokens:
                    sched.finish(seq, "length")
        if tier is not None:
            _check(state)
    raise AssertionError("requests did not finish")


@pytest.mark.parametrize("tiered", [False, True])
@pytest.mark.parametrize("seed", range(30))
def test_shared_prefix_traffic_always_finishes(seed, tiered):
    rng = random.Random(seed)
    prefixes = [[rng.randrange(3) for _ in range(rng.randrange(4, 20))] for _ in range(3)]
    seqs = []
    for i in range(rng.randint(2, 12)):
        prompt = (rng.choice(prefixes) + [rng.randrange(3) for _ in range(rng.randrange(1, 12))])[: MAX_LEN - 2]
        seqs.append(Seq(str(i), prompt, SamplingParams(max_tokens=rng.randint(1, MAX_LEN - len(prompt)))))
    tier = {"toy_tier.refuse": 0.2, "toy_tier.drops": 0.2, "toy_tier.delay": rng.randrange(4), "toy_tier.scale": 1,
            "toy_tier.seed": seed} if tiered else None
    state = _state(units=rng.randint(MAX_LEN // PAGE, 3 * MAX_LEN // PAGE), slots=rng.randint(1, 4), tier=tier,
                   rng=rng)
    sched = Scheduler(state, max_step_tokens=rng.randint(4, 30), max_model_len=MAX_LEN,
                      clock=state.index.tier.clock if tiered else lambda: 0.0)
    _run(state, sched, seqs, rng, aborts=tiered)
    assert all(s.status is Status.FINISHED for s in seqs) and state.held == {}
    pool = state.bound[0].pool
    cached = state.index.cached("kv")
    assert pool.num_free + cached.sum() == pool.capacity and (pool.refcount(np.flatnonzero(cached)) == 1).all()
    if tiered:
        state.launched([], _Done())                                       # a launch tells the tier of unread loads
        assert not state.waits and not state.due and not state.index.tier.untold()
