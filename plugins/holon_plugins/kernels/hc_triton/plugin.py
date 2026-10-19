"""`hc_gr_mix@1` and `hc_gr_combine@1` (op contracts: std.hyper; docs/v1-flash-next.md D3, D10) in Triton, float32
products with IEEE sums."""
from __future__ import annotations

import torch
from holon.api import Constraint, KernelSpec

from .kernel import BK, hc_gr_combine, hc_gr_mix

MIX, COMBINE = "hc_gr_mix", "hc_gr_combine"


class Mix:
    def __init__(self, ctx, *, streams: int, hidden: int, rank: int, eps: float, dtype: str, inject: bool) -> None:
        self.streams, self.eps, self.dtype = streams, eps, getattr(torch, dtype)

    def __call__(self, r, norm, down, up, inject, y=None, w=None):
        return hc_gr_mix(r, norm, down, up, inject, y, w, self.streams, self.eps, self.dtype)


class Combine:
    def __init__(self, ctx, *, streams: int, hidden: int, dtype: str) -> None:
        pass

    def __call__(self, r, y, w):
        hc_gr_combine(r, y, w)


class Plugin:
    def register(self, regs) -> None:
        for op, impl in ((MIX, Mix), (COMBINE, Combine)):
            regs.kernels.add(op, KernelSpec(op=op, provider="kernels.hc_triton", impl=impl,
                                            requires=Constraint(sm_min=80), supports=lambda f: f["hidden"] % BK == 0,
                                            props=frozenset({"graph_safe", "deterministic"})))
