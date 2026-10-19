"""Qwen3.8-Flash-Next's PLE ops in Triton (op contracts: models.qwen4_exp; docs/v1-flash-next.md D4, D13′): the n-gram
table rows with their history, and the block, float32 throughout."""
from __future__ import annotations

from holon.api import Constraint, KernelSpec

from .kernel import ple_block, ple_rows

ROWS, BLOCK = "qwen4_exp.ple_rows", "qwen4_exp.ple_block"


class PleRows:
    def __init__(self, ctx, *, ngram: int, heads_per_ngram: int, eos: int) -> None:
        self.facts = ngram, heads_per_ngram, eos

    def __call__(self, ids, history, slots, query_start, has_init, multipliers, sizes, offsets):
        return ple_rows(ids, history, slots, query_start, has_init, multipliers, sizes, offsets, *self.facts)


class PleBlock:
    def __init__(self, ctx, *, streams: int, hidden: int, kernel: int, dilation: int, eps: float, dtype: str) -> None:
        self.streams, self.dilation, self.eps = streams, dilation, eps

    def __call__(self, key, value, r, w_key, w_query, w_conv, conv, state, slots, query_start, has_init):
        ple_block(key, value, r, w_key, w_query, w_conv, conv, state, slots, query_start, has_init, self.streams,
                  self.dilation, self.eps)


class Plugin:
    def register(self, regs) -> None:
        for op, impl, supports in ((ROWS, PleRows, lambda f: f["ngram"] in (2, 3)),
                                   (BLOCK, PleBlock, lambda f: f["kernel"] <= 8)):
            regs.kernels.add(op, KernelSpec(op=op, provider="kernels.ple_triton", impl=impl,
                                            requires=Constraint(sm_min=80), supports=supports,
                                            props=frozenset({"graph_safe", "deterministic"})))
