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


class _Quantized:
    """A unit's quantized weight (adopted at load, design §5.5): its layout and the op that reads it. The kernel is
    selected on first use — after `finalize_weights` — so a unit merged away never binds one or leaves a site in the
    lock."""

    def __init__(self, ctx, layout, op: str, **facts) -> None:
        self.ctx, self.layout, self.op = ctx, layout, op
        self.facts = {"layout": f"{layout.scheme.elem}/{layout.name}", **facts}
        self._kernel = None

    def __call__(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if self._kernel is None:
            site = f"std.ops.{self.op.split('@')[0]}[" + ",".join(f"{k}={v}" for k, v in self.facts.items()) + "]"
            self._kernel = self.ctx.select_kernel(self.op, site=site, **self.facts).impl(self.ctx, **self.facts)
        return self._kernel(x, weight)


def _adopt(param: nn.Parameter, blocks: torch.Tensor, device: torch.device) -> None:
    """The meta parameter becomes the stored blocks on the device (the Parameter object kept)."""
    torch.utils.swap_tensors(param, nn.Parameter(blocks.to(device), requires_grad=False))


def _dtype(t: torch.dtype) -> str:
    return str(t).removeprefix("torch.")


class Linear(nn.Module):
    def __init__(self, ctx, *, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__()
        self.in_features, self.out_features, self._ctx = in_features, out_features, ctx
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=ctx.dtype), requires_grad=False)
        self.bias = nn.Parameter(torch.empty(out_features, dtype=ctx.dtype), requires_grad=False) if bias else None
        self._q: _Quantized | None = None

    def _quantized(self, layout) -> _Quantized:
        dtype = _dtype(self._ctx.dtype)
        return _Quantized(self._ctx, layout, kernels.QLINEAR, **{"in": self.in_features, "out": self.out_features},
                          act_dtype=dtype, out_dtype=dtype)

    def adopt(self, ctx, name: str, q) -> bool:
        if name != "weight" or tuple(q.logical_shape) != (self.out_features, self.in_features):
            return False
        self._q = self._quantized(q.layout)
        _adopt(self.weight, q.parts["blocks"], ctx.device)
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._q is None:
            return nn.functional.linear(x, self.weight, self.bias)
        y = self._q(x, self.weight)
        return y if self.bias is None else y + self.bias

    @staticmethod
    def merge(*units: Linear) -> Linear | None:
        """One unit for several projections of the same input — one GEMM, or one quantized matmul, instead of
        several — whose output rows are the units' outputs in order; the units' weights become views into it. None if
        they cannot share one: biases, or different dtypes, layouts or input sizes (ggml rows are independent block
        runs, so quantized weights of one type concatenate by rows).

        Optional capability of every `unit.std.ops.linear@1` provider, used duck-typed by composites (attention q/k/v,
        gate/up) in `finalize_weights`."""
        first = units[0]
        kinds = {(u.weight.dtype, u._q and u._q.facts["layout"], u.in_features) for u in units}
        if any(u.bias is not None for u in units) or len(kinds) > 1:
            return None
        weight = torch.cat([u.weight.data for u in units])
        with torch.device("meta"):
            merged = Linear(first._ctx, in_features=first.in_features, out_features=sum(u.out_features for u in units))
        torch.utils.swap_tensors(merged.weight, nn.Parameter(weight, requires_grad=False))
        if first._q is not None:
            merged._q = merged._quantized(first._q.layout)
        start = 0
        for u in units:
            n = u.weight.shape[0]
            u.weight.data = weight[start:start + n]
            start += n
        return merged


class Embedding(nn.Module):
    def __init__(self, ctx, *, vocab: int, hidden: int) -> None:
        super().__init__()
        self.vocab, self.hidden, self.dtype = vocab, hidden, ctx.dtype
        self.weight = nn.Parameter(torch.empty(vocab, hidden, dtype=ctx.dtype), requires_grad=False)
        self._q: _Quantized | None = None

    def adopt(self, ctx, name: str, q) -> bool:
        if name != "weight" or tuple(q.logical_shape) != (self.vocab, self.hidden):
            return False
        self._q = _Quantized(ctx, q.layout, kernels.QEMBEDDING, vocab=self.vocab, hidden=self.hidden,
                             dtype=_dtype(self.dtype))
        _adopt(self.weight, q.parts["blocks"], ctx.device)
        return True

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return nn.functional.embedding(ids, self.weight) if self._q is None else self._q(ids, self.weight)


class LMHead(nn.Module):
    """Output projection; `tied=True` reuses the embedding table (passed in as `embed`). Logits are float32."""

    def __init__(self, ctx, *, vocab: int, hidden: int, tied: bool, embed: Embedding | None = None) -> None:
        super().__init__()
        self.vocab, self.hidden, self.dtype = vocab, hidden, ctx.dtype
        if tied:
            if embed is None:
                raise ValueError("tied lm_head needs the embedding unit")
            # Held in a list, not as a submodule: the shared parameter must keep a single owner.
            self._embed = [embed]
            self.weight = None
        else:
            self._embed = []
            self.weight = nn.Parameter(torch.empty(vocab, hidden, dtype=ctx.dtype), requires_grad=False)
        self._q: _Quantized | None = None

    def _quantized(self, ctx, layout) -> _Quantized:
        return _Quantized(ctx, layout, kernels.QLINEAR, **{"in": self.hidden, "out": self.vocab},
                          act_dtype=_dtype(self.dtype), out_dtype="float32")

    def adopt(self, ctx, name: str, q) -> bool:
        if self._embed or name != "weight" or tuple(q.logical_shape) != (self.vocab, self.hidden):
            return False
        self._q = self._quantized(ctx, q.layout)
        _adopt(self.weight, q.parts["blocks"], ctx.device)
        return True

    def finalize_weights(self, ctx) -> None:
        if self._embed and self._embed[0]._q is not None:          # tied to a quantized table: read its blocks
            self._q = self._quantized(ctx, self._embed[0]._q.layout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        weight = self._embed[0].weight if self._embed else self.weight
        return nn.functional.linear(h, weight).float() if self._q is None else self._q(h, weight)
