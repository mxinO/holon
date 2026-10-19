"""The gated-residual hyper-connection units (docs/v1-flash-next.md D3, D4; op contracts in `kernels.py`).

The trunk's residual R is [T, n·H] float32 (`ModelSpec.residual`): `expand` repeats the inner embedding n times,
every block (`gr_block`) takes and returns it complete, and the model's `norm` is the output mixer (`gr_mix`), which
reads it to [T, H] for `lm_head`.

A block runs `pre` (at the layers `pre_at`: a unit updating R in place, e.g. a model's PLE), then the attention half —
mix, mixer — and the FFN half, whose mix applies the attention half's combine as it reads R (`hc_gr_mix@1`'s pending
combine: one launch less), then the FFN and its own combine, applied before the block returns. So R is complete at
every block boundary: what taps, the drafter and the next block see (`std.hyper`'s contract). The block forwards
`pre`'s state hooks (`BuildCtx.unit_class`): its state is declared by the block (C2, C3)."""
from __future__ import annotations

from typing import Any

import torch
from holon.api import Ref
from torch import nn

from . import kernels


class _HC(nn.Module):
    """One gated-residual mixer's weights and kernel instances: `norm` (float32, zero-centred), `down`, `up`, and with
    `inject` the per-stream combine weights."""

    def __init__(self, ctx: Any, *, n: int, hidden: int, rank: int, eps: float, inject: bool, site: str) -> None:
        super().__init__()
        width, dtype = n * hidden, str(ctx.dtype).removeprefix("torch.")
        self.norm = nn.Parameter(torch.empty(width, dtype=torch.float32), requires_grad=False)
        self.down = nn.Parameter(torch.empty(rank, width, dtype=ctx.dtype), requires_grad=False)
        self.up = nn.Parameter(torch.empty(width, rank, dtype=ctx.dtype), requires_grad=False)
        self.inject = nn.Parameter(torch.empty(n, width, dtype=ctx.dtype), requires_grad=False) if inject else None
        facts = {"streams": n, "hidden": hidden, "rank": rank, "eps": eps, "dtype": dtype, "inject": inject}
        self._mix = ctx.select_kernel(kernels.MIX, site=site, **facts).impl(ctx, **facts)
        if inject:
            facts = {"streams": n, "hidden": hidden, "dtype": dtype}
            self._combine = ctx.select_kernel(kernels.COMBINE, site=site, **facts).impl(ctx, **facts)

    def mix(self, r: torch.Tensor, y_prev: torch.Tensor | None = None, w_prev: torch.Tensor | None = None):
        return self._mix(r, self.norm, self.down, self.up, self.inject, y_prev, w_prev)

    def combine(self, r: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> None:
        self._combine(r, y, w)


class Expand(nn.Module):
    """The embed: `inner`'s [T, H] repeated into n float32 streams, [T, n·H]."""

    def __init__(self, ctx: Any, *, vocab: int, hidden: int, inner: Ref, n: int) -> None:
        super().__init__()
        self.inner, self.n = ctx.unit(inner, vocab=vocab, hidden=hidden), n

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.inner(ids).float().repeat(1, self.n)


class GRMix(nn.Module):
    """The output mixer, as the model's `norm`: R [T, n·H] → x [T, H] in the model's dtype."""

    def __init__(self, ctx: Any, *, dim: int, n: int, rank: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.hc = _HC(ctx, n=n, hidden=dim, rank=rank, eps=eps, inject=False, site="std.hyper.gr_mix")

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.hc.mix(r)[0]


class GRBlock(nn.Module):
    """One layer: `pre` (at `pre_at`), then the attention and FFN halves, each a gated-residual mix and combine around
    its unit. Ignores the `norm` site parameter (the core passes the model's norm: here the output mixer)."""

    @classmethod
    def state_specs(cls, params: dict, layers: tuple[int, ...], ctx: Any) -> list:
        pre, at = params.get("pre"), tuple(i for i in layers if i in params.get("pre_at", ()))
        hooks = ctx.unit_class(pre) if pre is not None and at else None
        return hooks.state_specs(dict(pre.params), at, ctx) if hasattr(hooks, "state_specs") else []

    @classmethod
    def planner(cls, params: dict, specs: list, ctx: Any) -> Any:
        pre = params["pre"]
        hooks = ctx.unit_class(pre)
        if not hasattr(hooks, "planner"):
            raise ValueError(f"std.hyper.gr_block: its pre unit {pre.target} declares state but no planner")
        return hooks.planner(dict(pre.params), specs, ctx)

    def __init__(self, ctx: Any, *, layer: int, hidden: int, mixer: Ref, ffn: Ref, norm: Ref, n: int, rank: int,
                 eps: float = 1e-6, pre: Ref | None = None, pre_at: tuple[int, ...] = ()) -> None:
        super().__init__()
        site = f"std.hyper.gr_block[layer={layer}]"
        self.pre = ctx.unit(pre, layer=layer, hidden=hidden) if pre is not None and layer in pre_at else None
        self.attn_hc = _HC(ctx, n=n, hidden=hidden, rank=rank, eps=eps, inject=True, site=site)
        self.mixer = ctx.unit(mixer, layer=layer, hidden=hidden)
        self.ffn_hc = _HC(ctx, n=n, hidden=hidden, rank=rank, eps=eps, inject=True, site=site)
        self.ffn = ctx.unit(ffn, layer=layer, hidden=hidden)

    def forward(self, r: torch.Tensor, ctx: Any) -> torch.Tensor:
        if self.pre is not None:
            r = self.pre(r, ctx)
        x, w = self.attn_hc.mix(r)
        y = self.mixer(x, ctx)
        x, w_ffn = self.ffn_hc.mix(r, y, w)                  # the attention half's combine, as it reads R
        self.ffn_hc.combine(r, self.ffn(x), w_ffn)           # R complete when the block returns
        return r
