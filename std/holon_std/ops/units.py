"""std.ops units: small modules that own their weights and call the bound kernel (initially the reference)."""
from __future__ import annotations

import torch
from torch import nn

from . import reference as ref


class RMSNorm(nn.Module):
    def __init__(self, ctx, *, dim: int, eps: float, offset: float = 0.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, dtype=ctx.dtype), requires_grad=False)
        self.eps, self.offset = eps, offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ref.rmsnorm(x, self.weight, self.eps, self.offset)


class Rope(nn.Module):
    def __init__(self, ctx, *, theta: float, style: str = "neox", rotary_dim: int | None = None) -> None:
        super().__init__()
        if style != "neox":
            raise ValueError(f"unsupported rope style {style!r}")
        self.theta, self.rotary_dim = theta, rotary_dim

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return ref.rope_neox(x, positions, self.theta, self.rotary_dim)


class Linear(nn.Module):
    def __init__(self, ctx, *, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=ctx.dtype), requires_grad=False)
        self.bias = nn.Parameter(torch.empty(out_features, dtype=ctx.dtype), requires_grad=False) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ref.linear(x, self.weight, self.bias)


class Embedding(nn.Module):
    def __init__(self, ctx, *, vocab: int, hidden: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab, hidden, dtype=ctx.dtype), requires_grad=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return nn.functional.embedding(ids, self.weight)


class LMHead(nn.Module):
    """Output projection; `tied=True` reuses the embedding table (passed in as `embed`)."""

    def __init__(self, ctx, *, vocab: int, hidden: int, tied: bool, embed: Embedding | None = None) -> None:
        super().__init__()
        if tied:
            if embed is None:
                raise ValueError("tied lm_head needs the embedding unit")
            # Held in a list, not as a submodule: `to_empty` would otherwise un-tie the shared parameter.
            self._embed = [embed]
            self.weight = None
        else:
            self._embed = []
            self.weight = nn.Parameter(torch.empty(vocab, hidden, dtype=ctx.dtype), requires_grad=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        weight = self._embed[0].weight if self._embed else self.weight
        return ref.linear(h, weight).float()
