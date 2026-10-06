from __future__ import annotations

import torch
from holon.api import Ref
from torch import nn

from . import reference as ref

LINEAR = "std.ops.linear@1"


class SwiGLU(nn.Module):
    def __init__(self, ctx, *, hidden: int, inter: int, layer: int = 0) -> None:
        super().__init__()
        self.gate_proj = ctx.unit(Ref(LINEAR, in_features=hidden, out_features=inter))
        self.up_proj = ctx.unit(Ref(LINEAR, in_features=hidden, out_features=inter))
        self.down_proj = ctx.unit(Ref(LINEAR, in_features=inter, out_features=hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(ref.silu_mul(self.gate_proj(x), self.up_proj(x)))
