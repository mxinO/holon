"""models.qwen4_exp: Qwen3.8-Flash-Next. For now its PLE ops (contracts in `kernels.py`, oracles in `reference.py`) and
their Triton kernels (`kernel.py`); the model's recipe and weight rules land with FN's core seams (step 3)."""
from __future__ import annotations

from holon.api import Constraint, KernelSpec

from . import kernels
from . import reference as ref

ROWS, BLOCK = kernels.ROWS.split("@")[0], kernels.BLOCK.split("@")[0]


class PleRows:
    def __init__(self, ctx, *, ngram: int, heads_per_ngram: int, eos: int) -> None:
        self.facts = ngram, heads_per_ngram, eos

    def __call__(self, ids, history, slots, query_start, has_init, multipliers, sizes, offsets):
        from .kernel import ple_rows

        return ple_rows(ids, history, slots, query_start, has_init, multipliers, sizes, offsets, *self.facts)


class PleBlock:
    def __init__(self, ctx, *, streams: int, hidden: int, kernel: int, dilation: int, eps: float, dtype: str) -> None:
        self.streams, self.dilation, self.eps = streams, dilation, eps

    def __call__(self, key, value, r, w_key, w_query, w_conv, conv, state, slots, query_start, has_init):
        from .kernel import ple_block

        ple_block(key, value, r, w_key, w_query, w_conv, conv, state, slots, query_start, has_init, self.streams,
                  self.dilation, self.eps)


class Plugin:
    def register(self, regs) -> None:
        regs.ops.add(ROWS, ref.ple_rows)
        regs.ops.add(BLOCK, ref.ple_block)
        for op, impl, supports in ((ROWS, PleRows, lambda f: f["ngram"] in (2, 3)),
                                   (BLOCK, PleBlock, lambda f: f["kernel"] <= 8)):
            regs.kernels.add(op, KernelSpec(op=op, provider="models.qwen4_exp", impl=impl,
                                            requires=Constraint(sm_min=80), supports=supports,
                                            props=frozenset({"graph_safe", "deterministic"})))
