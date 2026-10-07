"""std.ops units: small modules that own their weights and call the bound kernel (initially the reference)."""
from __future__ import annotations

import torch
from torch import nn

from . import kernels
from . import reference as ref


def _kernel(ctx, op: str, **facts):
    """This op's kernel for these facts (chosen by the core; the reference unless a faster one is installed)."""
    return ctx.select_kernel(op, site=f"std.ops/{op}", **facts).impl(ctx, **facts)


class RMSNorm(nn.Module):
    def __init__(self, ctx, *, dim: int, eps: float, offset: float = 0.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, dtype=ctx.dtype), requires_grad=False)
        facts = {"dim": dim, "eps": eps, "offset": offset, "dtype": str(ctx.dtype).removeprefix("torch.")}
        self._norm, self._add_norm = _kernel(ctx, kernels.RMSNORM, **facts), _kernel(ctx, kernels.ADD_RMSNORM, **facts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x, self.weight)

    def add(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(norm(residual + x), residual + x), fused where the kernel allows; `x` and `residual` may be reused."""
        return self._add_norm(x, residual, self.weight)


class Rope(nn.Module):
    """Rotates queries and keys together (neox layout)."""

    def __init__(self, ctx, *, theta: float, style: str = "neox", rotary_dim: int | None = None,
                 head_dim: int | None = None) -> None:
        super().__init__()
        if style != "neox":
            raise ValueError(f"unsupported rope style {style!r}")
        if head_dim is None:
            raise ValueError("rope needs the head_dim of the site that uses it")
        self._rope = _kernel(ctx, kernels.ROPE, head_dim=head_dim, rotary_dim=rotary_dim or head_dim, theta=theta,
                             max_positions=ctx.options.get("max_model_len", 8192),
                             dtype=str(ctx.dtype).removeprefix("torch."))

    def forward(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._rope(q, k, positions)


class Linear(nn.Module):
    def __init__(self, ctx, *, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=ctx.dtype), requires_grad=False)
        self.bias = nn.Parameter(torch.empty(out_features, dtype=ctx.dtype), requires_grad=False) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ref.linear(x, self.weight, self.bias)

    @staticmethod
    def merge(*units: Linear) -> torch.Tensor | None:
        """One weight for several projections of the same input (one GEMM instead of several): the units' weights
        become views into it. None if they cannot share one (biases, mixed dtypes)."""
        if any(u.bias is not None for u in units) or len({u.weight.dtype for u in units}) > 1:
            return None
        merged = torch.cat([u.weight.data for u in units])
        start = 0
        for u in units:
            n = u.weight.shape[0]
            u.weight.data = merged[start:start + n]
            start += n
        return merged


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
