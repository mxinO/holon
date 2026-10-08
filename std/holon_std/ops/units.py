"""std.ops units: small modules that own their weights and call the kernel the core selected for them."""
from __future__ import annotations

import torch
from torch import nn

from . import kernels


def _kernel(ctx, op: str, site: str, **facts):
    """This op's kernel for these facts, chosen by the core among the installed kernel plugins."""
    return ctx.select_kernel(op, site=site, **facts).impl(ctx, **facts)


class RMSNorm(nn.Module):
    def __init__(self, ctx, *, dim: int, eps: float, offset: float = 0.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, dtype=ctx.dtype), requires_grad=False)
        facts = {"dim": dim, "eps": eps, "offset": offset, "dtype": str(ctx.dtype).removeprefix("torch.")}
        site = f"std.ops.rmsnorm[dim={dim}]"
        self._norm = _kernel(ctx, kernels.RMSNORM, site, **facts)
        self._add_norm = _kernel(ctx, kernels.ADD_RMSNORM, site, **facts)

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
        self._rope = _kernel(ctx, kernels.ROPE, f"std.ops.rope[head_dim={head_dim}]", head_dim=head_dim,
                             rotary_dim=rotary_dim or head_dim, theta=theta,
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
        return nn.functional.linear(x, self.weight, self.bias)

    @staticmethod
    def merge(*units: Linear) -> torch.Tensor | None:
        """One weight for several projections of the same input (one GEMM instead of several): the units' weights
        become views into it. None if they cannot share one (biases, mixed dtypes).

        Optional capability of every `unit.std.ops.linear@1` provider, used duck-typed by composites (attention q/k/v,
        gate/up) in `finalize_weights`: return a dense weight `[sum(out), in]` whose rows are the units' weights in
        order and rebind each unit to its slice, or return None (e.g. quantized weights a plain GEMM cannot read)."""
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
        return nn.functional.linear(h, weight).float()
