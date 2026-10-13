"""`paged` state kind: state that grows with tokens in fixed-size pages (e.g. a KV cache).

It only does page accounting; the unit that owns the spec decides what a page holds and how the pool is laid out.
"""
from __future__ import annotations

import math


class Paged:
    def __init__(self, spec, pool, ctx) -> None:
        self.spec, self.pool = spec, pool
        self.page = spec.shape.tokens_per_unit

    def units_needed(self, seq, new_tokens: int) -> int:
        return max(0, math.ceil((seq.num_computed + new_tokens) / self.page) - len(seq.units(self.spec.name)))

    def capacity(self, seq) -> int:
        return len(seq.units(self.spec.name)) * self.page

    def reclaimable(self, seq) -> list[int]:
        return []
