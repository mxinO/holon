from __future__ import annotations

import torch
from holon.api import Paged, Ref, StateSpec
from torch import nn

from . import reference as ref

LINEAR = "std.ops.linear@1"


class GQA(nn.Module):
    """Grouped-query attention with optional per-head q/k norm. Rope and norms are injected units.

    With engine state bound, keys and values live in a paged pool shared by all layers of this mixer, one pool
    segment per layer — [layers, units, 2 (k/v), page, kv_heads, head_dim] — so each layer's slice is contiguous
    for kernels. In a model-only build (no state) it runs the stateless whole-sequence reference path.
    """

    @classmethod
    def state_specs(cls, params: dict, layers: tuple[int, ...], ctx) -> list[StateSpec]:
        page = ctx.options.get("page_size", 16)
        per_layer = 2 * page * params["kv_heads"] * params["head_dim"] * ctx.dtype.itemsize
        return [StateSpec("kv", "paged@1", Paged(per_layer * len(layers), page), layers, "kv.nhd",
                          segments=len(layers))]

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
        bound = ctx.state("kv", layer)
        self._kv = None if bound is None else (bound[1], bound[0].layers.index(layer), len(bound[0].layers),
                                                bound[0].shape.tokens_per_unit, ctx.dtype)

    def _kv_layer(self) -> torch.Tensor:
        pool, index, n_layers, page, dtype = self._kv
        shape = (n_layers, pool.capacity, 2, page, self.kv_heads, self.head_dim)
        return pool.view(dtype, shape)[index]

    def forward(self, x: torch.Tensor, ctx) -> torch.Tensor:
        dev = ctx.batch.dev
        t = x.shape[0]
        q = self.q_proj(x).view(t, self.heads, self.head_dim)
        k = self.k_proj(x).view(t, self.kv_heads, self.head_dim)
        v = self.v_proj(x).view(t, self.kv_heads, self.head_dim)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = self.rope(q, dev.positions), self.rope(k, dev.positions)
        if self._kv is None:
            o = ref.causal_attention(q, k, v, dev.query_start)
        else:
            pages, kv = ctx.meta("kv"), self._kv_layer()
            unit, off = pages.write_index // pages.page, pages.write_index % pages.page
            kv[unit, 0, off], kv[unit, 1, off] = k, v
            o = ref.paged_attention(q, kv, pages.table, dev.slots, pages.lens, dev.query_start, dev.positions)
        return self.o_proj(o.reshape(t, self.heads * self.head_dim))
