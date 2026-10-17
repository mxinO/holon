from __future__ import annotations

import torch
from holon.api import Ref
from torch import nn


class PreNorm(nn.Module):
    """Pre-norm residual block; the mixer, ffn and norm are injected units. An ffn unit may define `add(x, residual)`
    (`residual + ffn(x)`, bitwise, its add fused into its last kernel: std.moe's)."""

    def __init__(self, ctx, *, layer: int, hidden: int, mixer: Ref, ffn: Ref, norm: Ref) -> None:
        super().__init__()
        self.input_norm = ctx.unit(norm, dim=hidden)
        self.mixer = ctx.unit(mixer, layer=layer, hidden=hidden)
        self.post_norm = ctx.unit(norm, dim=hidden)
        self.ffn = ctx.unit(ffn, layer=layer, hidden=hidden)
        self._ffn_add = getattr(self.ffn, "add", None)

    def forward(self, h: torch.Tensor, ctx) -> torch.Tensor:
        x, h = self.post_norm.add(self.mixer(self.input_norm(h), ctx), h)   # fused residual add + norm
        return h + self.ffn(x) if self._ffn_add is None else self._ffn_add(x, h)
