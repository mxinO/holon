"""The DFlash / DFlash2 draft model as units (docs/m04-spec.md D4; the published forward: SGLang `models/dflash.py`,
llama.cpp `src/models/dflash.cpp`, studied, not copied): a context projection of the target's tapped residual streams,
pre-norm layers whose attention and MLP a grouped dynamic convolution wraps, and DFlash2's candidate selector. The
attention is std's GQA (bidirectional within a block, a sliding window), the norms, linears, embeddings and MLP std's
— all injected by Ref."""
from __future__ import annotations

import torch
from holon.api import Ref
from torch import nn

LINEAR, EMBEDDING = "std.ops.linear@1", "std.ops.embedding@1"


class Conv(nn.Module):
    """The grouped dynamic depthwise convolution across a block (DFlash2): each channel group mixes a token with the
    `taps − 1` before it in its block, with coefficients `base_kernel[side, tap]` plus a per-group term projected from
    the sublayer's input. `prepare` convolves that input (side 0) and returns the projection's side-1 terms, with which
    `finish` convolves the sublayer's output. Rows are whole blocks: row i is block position i mod `block`."""

    def __init__(self, ctx, *, hidden: int, block: int, taps: int, group: int) -> None:
        super().__init__()
        self.block, self.taps, self.groups = block, taps, hidden // group
        self.base_kernel = nn.Parameter(torch.empty(2, taps, hidden, dtype=ctx.dtype), requires_grad=False)
        self.kernel_projection = ctx.unit(Ref(LINEAR, in_features=hidden, out_features=2 * taps * self.groups))

    def _convolve(self, x: torch.Tensor, delta: torch.Tensor, side: int) -> torch.Tensor:
        blocks = x.unflatten(-1, (self.groups, -1))                                   # [n, groups, group]
        coef = self.base_kernel[side].view(1, self.taps, self.groups, -1) + delta[..., None]
        out = coef[:, 0] * blocks
        position = torch.arange(len(x), device=x.device) % self.block
        for tap in range(1, self.taps):
            before = nn.functional.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
            out = out + coef[:, tap] * torch.where((position >= tap)[:, None, None], before, 0)
        return out.flatten(-2)

    def prepare(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        delta = self.kernel_projection(x).view(len(x), 2, self.taps, self.groups)
        return self._convolve(x, delta[:, 0], 0), delta[:, 1]

    def finish(self, y: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return self._convolve(y, delta, 1)


class Layer(nn.Module):
    """A pre-norm layer: attention and MLP each wrapped by a `Conv` (DFlash2; none for plain DFlash)."""

    def __init__(self, ctx, *, layer: int, hidden: int, mixer: Ref, ffn: Ref, norm: Ref, conv: dict | None) -> None:
        super().__init__()
        self.input_norm = ctx.unit(norm, dim=hidden)
        self.mixer = ctx.unit(mixer, layer=layer, hidden=hidden)
        self.post_norm = ctx.unit(norm, dim=hidden)
        self.ffn = ctx.unit(ffn, layer=layer, hidden=hidden)
        self.attention_conv = Conv(ctx, hidden=hidden, **conv) if conv else None
        self.mlp_conv = Conv(ctx, hidden=hidden, **conv) if conv else None

    @staticmethod
    def _wrapped(conv: Conv | None, sublayer, x: torch.Tensor, *args) -> torch.Tensor:
        if conv is None:
            return sublayer(x, *args)
        x, delta = conv.prepare(x)
        return conv.finish(sublayer(x, *args), delta)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None, ctx) -> tuple[torch.Tensor, torch.Tensor]:
        """(this layer's MLP output, the residual stream before it); `x` is the previous layer's MLP output (the first
        layer's: the embeddings, with no residual yet)."""
        h, residual = (self.input_norm(x), x) if residual is None else self.input_norm.add(x, residual)
        h, residual = self.post_norm.add(self._wrapped(self.attention_conv, self.mixer, h, ctx), residual)
        return self._wrapped(self.mlp_conv, self.ffn, h), residual


class Selector(nn.Module):
    """DFlash2's candidate selector: scores the top-k candidates of adjacent block slots pairwise with bilinear
    codebooks, score[e, p, c] = unary[e, c] + Σ_r A[pred(e, p), r] · (W h_e)[r] · B[cand(e, c), r], the predecessor of
    slot 0 being the block's anchor token (fp32)."""

    def __init__(self, ctx, *, hidden: int, vocab: int, rank: int) -> None:
        super().__init__()
        self.predecessor = ctx.unit(Ref(EMBEDDING), vocab=vocab, hidden=rank)
        self.successor = ctx.unit(Ref(EMBEDDING), vocab=vocab, hidden=rank)
        self.hidden_projection = ctx.unit(Ref(LINEAR, in_features=hidden, out_features=rank))

    @staticmethod
    def _rows(table: nn.Module, ids: torch.Tensor) -> torch.Tensor:
        return table(ids.reshape(-1)).float().view(*ids.shape, -1)

    def forward(self, ids: torch.Tensor, unary: torch.Tensor, hidden: torch.Tensor, anchor: torch.Tensor
                ) -> torch.Tensor:
        """ids, unary [R, E, K] (candidates per slot, their logits), hidden [R, E, H], anchor [R] → [R, E, K, K]."""
        k = ids.shape[-1]
        projected = self.hidden_projection(hidden.reshape(-1, hidden.shape[-1])).float().view(*ids.shape[:2], -1)
        before = torch.cat([anchor[:, None, None].expand(-1, 1, k), ids[:, :-1]], 1)     # each slot's predecessors
        edges = torch.einsum("bepr,becr->bepc", self._rows(self.predecessor, before) * projected[:, :, None],
                             self._rows(self.successor, ids))
        return unary[:, :, None, :] + edges


class Draft(nn.Module):
    """The draft model, built as an auxiliary model (`BuildCtx.aux_model`). `context` turns the target's tapped
    features into every layer's context keys and values (no attention runs on them); `block` runs a block of rows
    through the layers and the final norm."""

    def __init__(self, ctx, *, layers, hidden: int, features: int, norm: Ref, conv: dict | None = None,
                 vocab: int = 0, rank: int = 0) -> None:
        super().__init__()
        self.fc = ctx.unit(Ref(LINEAR, in_features=features * hidden, out_features=hidden))
        self.hidden_norm = ctx.unit(norm, dim=hidden)
        self.layers = nn.ModuleList(Layer(ctx, layer=i, hidden=hidden, mixer=d.mixer, ffn=d.ffn, norm=norm, conv=conv)
                                    for i, d in enumerate(layers))
        self.norm = ctx.unit(norm, dim=hidden)
        self.selector = Selector(ctx, hidden=hidden, vocab=vocab, rank=rank) if rank else None

    def context(self, features: torch.Tensor, ctx) -> None:
        c = self.hidden_norm(self.fc(features))
        for layer in self.layers:
            layer.mixer.write_kv(c, ctx)

    def block(self, x: torch.Tensor, ctx) -> torch.Tensor:
        residual = None
        for layer in self.layers:
            x, residual = layer(x, residual, ctx)
        return self.norm.add(x, residual)[0]
