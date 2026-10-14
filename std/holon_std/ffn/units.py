from __future__ import annotations

import torch
from holon.api import Ref
from torch import nn

from . import kernels

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
        self._gate_up: nn.Module | None = None

    def finalize_weights(self, ctx) -> None:
        merge = getattr(type(self.gate_proj), "merge", None)       # one matmul for gate and up where possible
        self._gate_up = merge(self.gate_proj, self.up_proj) if merge else None
        if self._gate_up is not None and hasattr(self._gate_up, "use_glu"):   # and the activation fused into it
            self._gate_up.use_glu("swiglu")                                    # where its kernel can (D6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._gate_up is not None:
            fused = self._gate_up.glu(x) if hasattr(self._gate_up, "glu") else None
            return self.down_proj(self._act(self._gate_up(x)) if fused is None else fused)
        h = getattr(self.gate_proj, "prepare_input", lambda x: x)(x)    # one input, two projections (D6)
        return self.down_proj(self._act(torch.cat([self.gate_proj(h), self.up_proj(h)], dim=-1)))
