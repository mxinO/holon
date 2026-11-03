"""Our Triton kernels as `causal_conv1d@1` and `gated_rmsnorm@1` (op contracts: std.mixers.gdn; M0.3 design D3).
Correct and graph-safe; not tuned."""
from __future__ import annotations

from holon.api import Constraint, KernelSpec

from holon_plugins import host

from .kernel import causal_conv1d, gated_rmsnorm

CONV, NORM = "causal_conv1d", "gated_rmsnorm"


class CausalConv1d:
    def __init__(self, ctx, *, channels: int, kernel_size: int, dtype: str, weight_dtype: str | None = None) -> None:
        pass                                       # (weights read in their own dtype: f32 for KDA's convs)

    def __call__(self, x, weight, state, slots, query_start, has_init):
        return causal_conv1d(x, weight, state, slots, query_start, has_init)


class GatedRMSNorm:
    def __init__(self, ctx, *, dim: int, eps: float, dtype: str, out_gate: str = "silu", round: str = "model") -> None:
        self.eps, self.sigmoid, self.unrounded = eps, out_gate == "sigmoid", round == "none"

    def __call__(self, o, z, weight):
        return gated_rmsnorm(o, z, weight, self.eps, self.sigmoid, self.unrounded)


def _half(f: dict) -> bool:
    return f["dtype"] in ("bfloat16", "float16")


class Plugin:
    def register(self, regs) -> None:
        host.require_c_toolchain()            # Triton's launchers need a C compiler (docs/dev/v1-pip.md D-H)
        for op, impl, supports in ((CONV, CausalConv1d, lambda f: _half(f) and 2 <= f["kernel_size"] <= 8
                                    and f.get("weight_dtype", "float32") == "float32"),   # (KDA's f32 weights)
                                   (NORM, GatedRMSNorm, lambda f: _half(f) and f["dim"] in (64, 128, 256)
                                    and f.get("out_gate", "silu") in ("silu", "sigmoid")
                                    and f.get("round", "model") in ("model", "none"))):   # (none: KDA's)
            regs.kernels.add(op, KernelSpec(op=op, provider="kernels.gdn_triton", impl=impl,
                                            requires=Constraint(sm_min=80), supports=supports,
                                            props=frozenset({"graph_safe", "deterministic"})))
