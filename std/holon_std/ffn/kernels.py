"""Op contract std.ffn owns, and its reference kernel (lowest priority, any device and dtype).

- `silu_mul@1`: instance `(gate_up [T, 2*inter]) -> silu(gate) * up [T, inter]`, gate first. Facts: inter, dtype.
"""
from __future__ import annotations

from . import reference as ref

SILU_MUL = "silu_mul@1"


class SiluMul:
    def __init__(self, ctx, *, inter: int, dtype: str) -> None:
        self.inter = inter

    def __call__(self, gate_up):
        return ref.silu_mul(gate_up[..., : self.inter], gate_up[..., self.inter:])
