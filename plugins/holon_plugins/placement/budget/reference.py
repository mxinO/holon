"""Oracles of placement.budget (FN step 7): F11 (placement.lfu's decayed-count admission) with the CPU's share admitted
as any miss and the admissions held back, as Strata's adaptive tier holds its swaps (its `--adapt-min-gain` and
`--adapt-swaps`, src/program/generate.cpp, the `adapt` round). `moe_resolve.budget@1` is `moe_resolve.lfu@1` with:
- the CPU's experts candidates as any miss (the call `defers` their copies: `later`, `n_later`, fetched after it while
  the CPU serves their entries), so the cache learns from every miss whoever computed it;
- a candidate admitted only while its score passes its victim's by `gain` (f32; F11: 0, strictly greater);
- at most `budget` admissions a call, the empty slots it takes included (−1: no cap).
`replay` is F11's with the same terms (the trace's CPU experts are candidates); `lend` and `lend_tail` are F11's."""
from __future__ import annotations

from ..lfu.reference import lend, lend_tail, moe_resolve_lfu
from ..lfu.reference import replay as lfu_replay

__all__ = ["lend", "lend_tail", "moe_resolve_budget", "replay"]


def moe_resolve_budget(*args, budget: int = -1, gain: float = 0.5, **kw) -> dict:
    """One call (`moe_resolve_lfu`'s arguments): its outputs, `later` and `n_later` included."""
    return moe_resolve_lfu(*args, **kw, defer=True, budget=None if budget < 0 else budget, gain=gain)


def replay(*args, budget: int = -1, gain: float = 0.5, **kw) -> list[dict]:
    """The L1 parity oracle (F11's `replay` with this policy's terms)."""
    return lfu_replay(*args, **kw, budget=None if budget < 0 else budget, gain=gain)
