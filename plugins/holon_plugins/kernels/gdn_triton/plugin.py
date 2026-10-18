"""Our Triton kernels as `causal_conv1d@1` and `gated_rmsnorm@1` (op contracts: std.mixers.gdn; M0.3 design D3).
Correct and graph-safe; not tuned."""
from __future__ import annotations

from holon.api import Constraint, KernelSpec

from .kernel import causal_conv1d, gated_rmsnorm

CONV, NORM = "causal_conv1d", "gated_rmsnorm"


class CausalConv1d:
    def __init__(self, ctx, *, channels: int, kernel_size: int, dtype: str) -> None:
        pass

    def __call__(self, x, weight, state, slots, query_start, has_init):
        return causal_conv1d(x, weight, state, slots, query_start, has_init)


class GatedRMSNorm:
    def __init__(self, ctx, *, dim: int, eps: float, dtype: str, out_gate: str = "silu") -> None:
        self.eps, self.sigmoid = eps, out_gate == "sigmoid"

    def __call__(self, o, z, weight):
        return gated_rmsnorm(o, z, weight, self.eps, self.sigmoid)


def _half(f: dict) -> bool:
    return f["dtype"] in ("bfloat16", "float16")


class Plugin:
    def register(self, regs) -> None:
        for op, impl, supports in ((CONV, CausalConv1d, lambda f: _half(f) and 2 <= f["kernel_size"] <= 8),
                                   (NORM, GatedRMSNorm, lambda f: _half(f) and f["dim"] in (64, 128, 256)
                                    and f.get("out_gate", "silu") in ("silu", "sigmoid"))):
            regs.kernels.add(op, KernelSpec(op=op, provider="kernels.gdn_triton", impl=impl,
                                            requires=Constraint(sm_min=80), supports=supports,
                                            props=frozenset({"graph_safe", "deterministic"})))
