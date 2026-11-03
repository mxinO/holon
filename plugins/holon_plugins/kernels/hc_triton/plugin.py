"""`hc_gr_mix@1` and `hc_gr_combine@1` (op contracts: std.hyper; docs/dev/v1-flash-next.md D3, D10), `hc_mhc_mix@1` and
`hc_mhc_writeback@1` (docs/dev/v1-glm.md D3, D12) in Triton, float32 products with IEEE sums."""
from __future__ import annotations

import torch
from holon.api import Constraint, KernelSpec

from holon_plugins import host

from . import mhc
from .kernel import BK, hc_gr_combine, hc_gr_mix

MIX, COMBINE, MHC_MIX, MHC_WRITEBACK = "hc_gr_mix", "hc_gr_combine", "hc_mhc_mix", "hc_mhc_writeback"


class Mix:
    def __init__(self, ctx, *, streams: int, hidden: int, rank: int, eps: float, dtype: str, inject: bool) -> None:
        self.streams, self.eps, self.dtype = streams, eps, getattr(torch, dtype)
        # prefill's projections: float32 SGEMMs, or bf16x3 (`hc.prefill`; bf16 weights: three tensor-core products
        # of the activation's exact split, the lead's D10 call for FN step 7)
        self.prefill = str(ctx.options.get("plugins", {}).get("hc.prefill", "sgemm"))
        if self.prefill not in ("sgemm", "bf16x3"):
            raise ValueError(f"kernels.hc_triton: hc.prefill {self.prefill!r}: sgemm or bf16x3")

    def __call__(self, r, norm, down, up, inject, y_prev=None, w_prev=None):
        return hc_gr_mix(r, norm, down, up, inject, y_prev, w_prev, self.streams, self.eps, self.dtype, self.prefill)


class Combine:
    def __init__(self, ctx, *, streams: int, hidden: int, dtype: str) -> None:
        pass

    def __call__(self, r, y, w):
        hc_gr_combine(r, y, w)


class MhcMix:
    def __init__(self, ctx, *, streams: int, hidden: int, eps: float, hc_eps: float, iters: int, dtype: str) -> None:
        self.hidden, self.eps, self.hc_eps, self.iters, self.dtype = hidden, eps, hc_eps, iters, getattr(torch, dtype)

    def __call__(self, r, fn, base, scale, y_prev=None, post_prev=None, comb_prev=None):
        return mhc.hc_mhc_mix(r, fn, base, scale, y_prev, post_prev, comb_prev, self.hidden, self.eps, self.hc_eps,
                              self.iters, self.dtype)


class MhcWriteback:
    def __init__(self, ctx, *, streams: int, hidden: int, dtype: str) -> None:
        pass

    def __call__(self, r, y, post, comb):
        mhc.hc_mhc_writeback(r, y, post, comb)


def _mhc(f: dict) -> bool:
    """n = 4 streams; whole `_collapse` blocks, a power-of-two number of `_dots` chunks; at least one iteration."""
    chunks = f["hidden"] // mhc.BK
    return (f["streams"] == mhc.N_STREAMS and f["hidden"] % mhc.BH == 0 and chunks & (chunks - 1) == 0
            and f.get("iters", 1) >= 1)


class Plugin:
    def register(self, regs) -> None:
        host.require_c_toolchain()            # Triton's launchers need a C compiler (docs/dev/v1-pip.md D-H)
        for op, impl, supports in ((MIX, Mix, lambda f: f["hidden"] % BK == 0),
                                   (COMBINE, Combine, lambda f: f["hidden"] % BK == 0),
                                   (MHC_MIX, MhcMix, _mhc), (MHC_WRITEBACK, MhcWriteback, _mhc)):
            regs.kernels.add(op, KernelSpec(op=op, provider="kernels.hc_triton", impl=impl,
                                            requires=Constraint(sm_min=80), supports=supports,
                                            props=frozenset({"graph_safe", "deterministic"})))
