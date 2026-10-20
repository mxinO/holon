"""A multi-token-prediction draft head (docs/m04-spec.md D2): the shape the Qwen3.5/3.8, DeepSeek-V3, GLM-4.5 and MiMo
heads share. For the token at position p and the target's final-norm hidden state of position p − 1:

    x = fc(concat(enorm(embed(token)), hnorm(h)))      (`order` "embed_first"; "hidden_first" swaps the halves)
    y = layers(x)                                       decoder layers, their state under the aux model's namespace
    out = norm(y)                                       logits = the target's lm_head(out); `out` is also the hidden
                                                        the next chained step takes

The embedding and the head are the target's (`TargetView.shared`); the drafter owns the chaining.
"""
from __future__ import annotations

import torch
from holon.api import Ref
from torch import nn

LINEAR = "std.ops.linear@1"
ORDERS = ("embed_first", "hidden_first")


class MTPHead(nn.Module):
    def __init__(self, ctx, *, layers, hidden: int, block: Ref, norm: Ref, order: str = "embed_first") -> None:
        super().__init__()
        if order not in ORDERS:
            raise ValueError(f"order {order!r}: one of {ORDERS}")
        self.order = order
        self.enorm = ctx.unit(norm, dim=hidden)
        self.hnorm = ctx.unit(norm, dim=hidden)
        self.fc = ctx.unit(Ref(LINEAR, in_features=2 * hidden, out_features=hidden))
        self.layers = nn.ModuleList(ctx.unit(block, layer=i, hidden=hidden, mixer=d.mixer, ffn=d.ffn, norm=norm)
                                    for i, d in enumerate(layers))
        self.norm = ctx.unit(norm, dim=hidden)

    def forward(self, embedded: torch.Tensor, h: torch.Tensor, ctx) -> torch.Tensor:
        """`embedded` [T, H] (the target's embedding of each token), `h` [T, H] (the target's final-norm hidden of
        each token's predecessor) → [T, H], the head's output before the target's lm_head."""
        e, g = self.enorm(embedded), self.hnorm(h)
        x = self.fc(torch.cat([e, g] if self.order == "embed_first" else [g, e], dim=-1))
        for layer in self.layers:
            x = layer(x, ctx)
        return self.norm(x)


class MTPHCHead(nn.Module):
    """A multi-token-prediction head over a hyper-connection residual (Qwen3.8-Flash-Next's, FN D9): it takes and
    returns the trunk's residual R [T, n·H] float32 (`ModelSpec.residual`) instead of a final-norm hidden. For the
    token at position p and R of position p − 1:

        R_s = fc_embedding(enorm(e)) + fc_hidden(hnorm(R))_s    hnorm pooled: one RMS over all n·H features (SGLang's
                                                                and Strata's form), fc_hidden per stream s
        R = layers(R)                                           gated-residual blocks (`block`), state in the aux
                                                                model's namespace
        logits = the target's lm_head(out(R))                   `out`: the head's output mixer; R is also what the
                                                                next chained step takes (`carries`)

    `e` is the target's embedding: the first H columns of what its embed returns (`std.hyper.expand@1` repeats e)."""

    carries = "residual"

    def __init__(self, ctx, *, layers, hidden: int, n: int, block: Ref, norm: Ref, out: Ref) -> None:
        super().__init__()
        self.hidden, self.n, self.dtype = hidden, n, ctx.dtype
        self.enorm = ctx.unit(norm, dim=hidden)
        self.hnorm = ctx.unit(norm, dim=n * hidden)
        self.fc_embedding = ctx.unit(Ref(LINEAR, in_features=hidden, out_features=hidden))
        self.fc_hidden = ctx.unit(Ref(LINEAR, in_features=hidden, out_features=hidden))
        self.layers = nn.ModuleList(ctx.unit(block, layer=i, hidden=hidden, mixer=d.mixer, ffn=d.ffn, norm=out)
                                    for i, d in enumerate(layers))
        self.out = ctx.unit(out, dim=hidden)

    def forward(self, embedded: torch.Tensor, r: torch.Tensor, ctx) -> torch.Tensor:
        """`embedded` [T, ≥ H] (the target's embed of each token), `r` [T, n·H] (R of each token's predecessor) →
        the head's R [T, n·H] float32, before its output mixer."""
        t, h = r.shape[0], self.hidden
        e = self.fc_embedding(self.enorm(embedded[:, :h].to(self.dtype)))
        g = self.fc_hidden(self.hnorm(r.to(self.dtype)).view(t * self.n, h)).view(t, self.n, h)
        x = (g.float() + e.float()[:, None]).view(t, self.n * h)
        for layer in self.layers:
            x = layer(x, ctx)
        return x
