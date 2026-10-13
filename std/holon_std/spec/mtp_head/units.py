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
