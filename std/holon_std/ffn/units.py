from __future__ import annotations

import torch
from holon.api import Ref
from torch import nn

from . import kernels
from . import reference as ref

LINEAR = "std.ops.linear@1"


class SwiGLU(nn.Module):
    def __init__(self, ctx, *, hidden: int, inter: int, layer: int = 0) -> None:
        super().__init__()
        self.gate_proj = ctx.unit(Ref(LINEAR, in_features=hidden, out_features=inter))
        self.up_proj = ctx.unit(Ref(LINEAR, in_features=hidden, out_features=inter))
        self.down_proj = ctx.unit(Ref(LINEAR, in_features=inter, out_features=hidden))
        facts = {"inter": inter, "dtype": str(ctx.dtype).removeprefix("torch.")}
        spec = ctx.select_kernel(kernels.SILU_MUL, site=f"std.ffn.swiglu[inter={inter}]", **facts)
        self._act = spec.impl(ctx, **facts)
        self._gate_up: torch.Tensor | None = None

    def finalize_weights(self, ctx) -> None:
        merge = getattr(type(self.gate_proj), "merge", None)       # one GEMM for gate and up where possible
        self._gate_up = merge(self.gate_proj, self.up_proj) if merge else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._gate_up is not None:
            return self.down_proj(self._act(torch.nn.functional.linear(x, self._gate_up)))
        return self.down_proj(ref.silu_mul(self.gate_proj(x), self.up_proj(x)))
