from __future__ import annotations

import torch
from holon.api import Ref
from torch import nn

from . import reference as ref

LINEAR = "std.ops.linear@1"


class GQA(nn.Module):
    """Grouped-query attention with optional per-head q/k norm. Rope and norms are injected units.

    Stateless reference path for now (whole sequences, no KV cache); state specs and the paged-KV planner land with
    the engine runtime (plan C3/C6), on the kernels chosen by spike S1.
    """

    def __init__(self, ctx, *, layer: int, hidden: int, heads: int, kv_heads: int, head_dim: int, rope: Ref,
                 qk_norm: Ref | None = None) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.q_proj = ctx.unit(Ref(LINEAR, in_features=hidden, out_features=heads * head_dim))
        self.k_proj = ctx.unit(Ref(LINEAR, in_features=hidden, out_features=kv_heads * head_dim))
        self.v_proj = ctx.unit(Ref(LINEAR, in_features=hidden, out_features=kv_heads * head_dim))
        self.o_proj = ctx.unit(Ref(LINEAR, in_features=heads * head_dim, out_features=hidden))
        self.q_norm = ctx.unit(qk_norm, dim=head_dim) if qk_norm else None
        self.k_norm = ctx.unit(qk_norm, dim=head_dim) if qk_norm else None
        self.rope = ctx.unit(rope)

    def forward(self, x: torch.Tensor, ctx) -> torch.Tensor:
        dev = ctx.batch.dev
        t = x.shape[0]
        q = self.q_proj(x).view(t, self.heads, self.head_dim)
        k = self.k_proj(x).view(t, self.kv_heads, self.head_dim)
        v = self.v_proj(x).view(t, self.kv_heads, self.head_dim)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = self.rope(q, dev.positions), self.rope(k, dev.positions)
        o = ref.causal_attention(q, k, v, dev.query_start)
        return self.o_proj(o.reshape(t, self.heads * self.head_dim))
