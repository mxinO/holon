"""QSA's kernels in Triton (op contracts: std.mixers.qsa; docs/dev/v1-flash-next.md D6, D10): the index write and its
block pooling, the block selection (float32 IEEE scores, an exact top-k with lower-block ties) and the sparse
attention over gqa's KV layouts, q8_0 and fp8 read in the kernel."""
from __future__ import annotations

from holon.api import Constraint, KernelSpec

from .kernel import qsa_attend, qsa_index, qsa_select

INDEX, SELECT, ATTEND = "qsa_index", "qsa_select", "qsa_attend"
SUB_BATCH = 512
LAYOUTS = ("kv.hnd", "kv.hnd.fp8", "kv.hnd.q8_0")


class Index:
    def __init__(self, ctx, *, head_dim: int, ratio: int, page: int, rotary_dim: int, max_positions: int,
                 eps: float, dtype: str) -> None:
        self.ratio, self.eps = ratio, eps

    def __call__(self, k, raw, pooled, table, seq, positions, weight, cos_sin):
        qsa_index(k, raw, pooled, table, seq, positions, weight, cos_sin, self.ratio, self.eps)


class Select:
    def __init__(self, ctx, *, heads: int, head_dim: int, ratio: int, budget: int, page: int) -> None:
        self.ratio, self.budget = ratio, budget

    def __call__(self, q, pooled, table, seq, positions):
        return qsa_select(q, pooled, table, seq, positions, self.ratio, self.budget, SUB_BATCH)


class Attend:
    def __init__(self, ctx, *, heads: int, kv_heads: int, head_dim: int, ratio: int, width: int, page: int,
                 layout: str, q_dtype: str, kv_dtype: str) -> None:
        self.ratio = ratio

    def __call__(self, q, kv, blocks, table, seq, positions, gate=None):
        return qsa_attend(q, kv, blocks, table, seq, positions, gate, self.ratio)


def _pow2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


class Plugin:
    def register(self, regs) -> None:
        checks = {INDEX: lambda f: _pow2(f["head_dim"]) and f["page"] % f["ratio"] == 0
                  and f["rotary_dim"] % 2 == 0 and f["rotary_dim"] <= f["head_dim"],
                  SELECT: lambda f: _pow2(f["head_dim"]) and f["head_dim"] >= 16 and f["heads"] <= 16,
                  ATTEND: lambda f: _pow2(f["head_dim"]) and f["head_dim"] % 32 == 0 and f["layout"] in LAYOUTS
                  and f["q_dtype"] in ("bfloat16", "float16")}
        fp8 = lambda f: f["layout"] == "kv.hnd.fp8"                              # noqa: E731
        anywhere, fp8_sms = Constraint(sm_min=80), Constraint(sm_min=89, sm_max=129)   # Triton's e4m3 from sm_89
        for op, impl, sms, check in ((INDEX, Index, anywhere, checks[INDEX]),
                                     (SELECT, Select, anywhere, checks[SELECT]),
                                     (ATTEND, Attend, anywhere, lambda f: checks[ATTEND](f) and not fp8(f)),
                                     (ATTEND, Attend, fp8_sms, lambda f: checks[ATTEND](f) and fp8(f))):
            regs.kernels.add(op, KernelSpec(op=op, provider="kernels.qsa_triton", impl=impl, requires=sms,
                                            supports=check, props=frozenset({"graph_safe", "deterministic"})))
