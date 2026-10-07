"""Liveness with the real StateManager, PrefixIndex and Scheduler: shared prefixes, a tight pool, random traffic."""
import random

import numpy as np
import pytest
import torch
from holon.api import Paged, SamplingParams, Seq, StateSpec, Status
from holon.core.memory import Ledger
from holon.core.prefix import PrefixIndex
from holon.core.sched import Scheduler
from holon.core.state import BoundSpec, StateManager

PAGE, MAX_LEN = 4, 40


class _PagedKind:
    def __init__(self, spec):
        self.spec = spec

    def units_needed(self, seq, new_tokens):
        return max(0, -(-(seq.num_computed + new_tokens) // PAGE) - len(seq.units(self.spec.name)))


def _state(units: int, slots: int) -> StateManager:
    ledger = Ledger(torch.device("cpu"))
    pool = ledger.request("kv", unit_bytes=8, min_units=units, tokens_per_unit=PAGE)
    ledger.materialize(None)
    spec = StateSpec("kv", "paged@1", Paged(8, PAGE), (0,), "test")
    index = PrefixIndex(1, PAGE, {"kv": pool})
    return StateManager([BoundSpec(spec, pool, _PagedKind(spec))], slots, MAX_LEN, torch.device("cpu"), index=index)


def _run(state: StateManager, sched: Scheduler, seqs: list[Seq], rng: random.Random) -> None:
    pending = list(seqs)
    for _ in range(20_000):
        while pending and rng.random() < 0.5:                 # arrivals interleave with steps
            sched.add(pending.pop(0))
        if not (sched.waiting or sched.running):
            if not pending:
                return
            continue
        scheduled = sched.schedule()
        assert scheduled is not None, "no progress"
        plan, rows = scheduled
        for seq, n, sampled in zip(rows, plan.n_query, plan.n_logits, strict=True):
            seq.num_computed += int(n)
            state.publish(seq)
            if sampled:
                seq.tokens.append(rng.randrange(3))           # few distinct tokens: generations share blocks too
                if seq.status is not Status.RUNNING:
                    sched.finished_prefill(seq)
                if len(seq.tokens) - seq.prompt_len >= seq.params.max_tokens:
                    sched.finish(seq, "length")
    raise AssertionError("requests did not finish")


@pytest.mark.parametrize("seed", range(30))
def test_shared_prefix_traffic_always_finishes(seed):
    rng = random.Random(seed)
    prefixes = [[rng.randrange(3) for _ in range(rng.randrange(4, 20))] for _ in range(3)]
    seqs = []
    for i in range(rng.randint(2, 12)):
        prompt = (rng.choice(prefixes) + [rng.randrange(3) for _ in range(rng.randrange(1, 12))])[: MAX_LEN - 2]
        seqs.append(Seq(str(i), prompt, SamplingParams(max_tokens=rng.randint(1, MAX_LEN - len(prompt)))))
    state = _state(units=rng.randint(MAX_LEN // PAGE, 3 * MAX_LEN // PAGE), slots=rng.randint(1, 4))
    _run(state, Scheduler(state, max_step_tokens=rng.randint(4, 30), max_model_len=MAX_LEN), seqs, rng)
    assert all(s.status is Status.FINISHED for s in seqs) and state.held == {}
    pool = state.bound[0].pool
    cached = state.index.cached("kv")
    assert pool.num_free + cached.sum() == pool.capacity and (pool.refcount(np.flatnonzero(cached)) == 1).all()
