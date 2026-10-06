from __future__ import annotations

import torch
from holon.api import Ref
from torch import nn


class PreNorm(nn.Module):
    """Pre-norm residual block; the mixer, ffn and norm are injected units."""

    def __init__(self, ctx, *, layer: int, hidden: int, mixer: Ref, ffn: Ref, norm: Ref) -> None:
        super().__init__()
        self.input_norm = ctx.unit(norm, dim=hidden)
        self.mixer = ctx.unit(mixer, layer=layer, hidden=hidden)
        self.post_norm = ctx.unit(norm, dim=hidden)
        self.ffn = ctx.unit(ffn, layer=layer, hidden=hidden)

    def forward(self, h: torch.Tensor, ctx) -> torch.Tensor:
        h = h + self.mixer(self.input_norm(h), ctx)
        return h + self.ffn(self.post_norm(h))
