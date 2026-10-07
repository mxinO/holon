from __future__ import annotations

import dataclasses
from typing import Any

import torch
from holon.api import Paged, Ref, StateSpec
from torch import nn

from . import kernels
from . import reference as ref

LINEAR = "std.ops.linear@1"


@dataclasses.dataclass(frozen=True)
class AttnMeta:
    """One step's attention metadata for every layer of the group: the kernel this step uses and its plan."""

    kernel: Any
    handle: Any
    pages: Any

    @property
    def graph_key(self) -> Any:
        """Hashable if this step's attention may be replayed from a captured graph, else None."""
        key = getattr(self.kernel, "graph_key", None)
        return None if key is None else key(self.handle)


class Planner:
    """Owns this group's kernel instances; each step is all prefill chunks, all decode rows or all verify rows of one
    width (v0 scheduler)."""

    def __init__(self, prefill: Any, decode: Any, verify: Any, needs_host_lens: bool) -> None:
        self.prefill, self.decode, self.verify = prefill, decode, verify
        # Decode steps plan from host lengths (e.g. FlashInfer), so a request whose length is only known on the
        # device (an uncommitted verify row) must sit them out; the core reads this (design §7.3).
        self.needs_host_lens = needs_host_lens

    def plan_bounds(self, batch, pages) -> AttnMeta:
        n_query, n_logits = batch.plan.n_query, batch.plan.n_logits
        if (n_query == 1).all():
            kernel = self.decode
        elif (n_logits == n_query).all() and (n_query == n_query[0]).all():
            kernel = self.verify
        else:
            kernel = self.prefill
        return AttnMeta(kernel, kernel.plan(batch, pages), pages)

    def prepare_device(self, meta: AttnMeta, ctx) -> None:
        prepare = getattr(meta.kernel, "prepare", None)          # in-graph part of a kernel's plan (optional)
        if prepare is not None:
            prepare(meta.handle)


class GQA(nn.Module):
    """Grouped-query attention with optional per-head q/k norm. Rope and norms are injected units.

    With engine state bound, keys and values live in a paged pool shared by all layers of this mixer, one pool
    segment per layer — [layers, units, 2 (k/v), kv_heads, page, head_dim] (HND, the layout every selected
    attention kernel must read) — so each layer's slice is contiguous for kernels. In a model-only build (no
    state) it runs the stateless whole-sequence reference path.
    """

    @classmethod
    def state_specs(cls, params: dict, layers: tuple[int, ...], ctx) -> list[StateSpec]:
        page = ctx.options.get("page_size", 16)
        per_layer = 2 * page * params["kv_heads"] * params["head_dim"] * ctx.dtype.itemsize
        return [StateSpec("kv", "paged@1", Paged(per_layer * len(layers), page), layers, kernels.LAYOUT,
                          segments=len(layers))]

    @classmethod
    def planner(cls, params: dict, specs: list[StateSpec], ctx) -> Planner:
        dtype = str(ctx.dtype).removeprefix("torch.")
        facts = {"heads": params["heads"], "kv_heads": params["kv_heads"], "head_dim": params["head_dim"],
                 "page": specs[0].shape.tokens_per_unit, "q_dtype": dtype, "kv_dtype": dtype,
                 "layout": kernels.LAYOUT, "max_seqs": ctx.options["max_seqs"]}
        ops = (kernels.PREFILL, kernels.DECODE, kernels.VERIFY)
        chosen = [ctx.select_kernel(op, site=specs[0].group, **facts) for op in ops]
        decode_props = chosen[1].props
        return Planner(*(spec.impl(ctx, **facts) for spec in chosen), needs_host_lens="needs_host_lens" in decode_props)

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
        self.rope = ctx.unit(rope, head_dim=head_dim)
        self._qkv: torch.Tensor | None = None
        bound = ctx.state("kv", layer)
        self._kv = None if bound is None else (bound[1], bound[0].layers.index(layer), len(bound[0].layers),
                                                bound[0].shape.tokens_per_unit, ctx.dtype)

    def _kv_layer(self) -> torch.Tensor:
        pool, index, n_layers, page, dtype = self._kv
        shape = (n_layers, pool.capacity, 2, self.kv_heads, page, self.head_dim)
        return pool.view(dtype, shape)[index]

    def finalize_weights(self, ctx) -> None:
        merge = getattr(type(self.q_proj), "merge", None)          # one GEMM for q, k, v where the linears allow
        self._qkv = merge(self.q_proj, self.k_proj, self.v_proj) if merge else None

    def forward(self, x: torch.Tensor, ctx) -> torch.Tensor:
        dev = ctx.batch.dev
        t = x.shape[0]
        if self._qkv is not None:
            q, k, v = torch.nn.functional.linear(x, self._qkv).split(
                [self.heads * self.head_dim, self.kv_heads * self.head_dim, self.kv_heads * self.head_dim], dim=-1)
        else:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = q.reshape(t, self.heads, self.head_dim)
        k = k.reshape(t, self.kv_heads, self.head_dim)
        v = v.reshape(t, self.kv_heads, self.head_dim)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = self.rope(q, k, dev.positions)
        if self._kv is None:
            o = ref.causal_attention(q, k, v, dev.query_start)
        else:
            meta, kv = ctx.meta("kv"), self._kv_layer()
            unit, off = meta.pages.write_index // meta.pages.page, meta.pages.write_index % meta.pages.page
            kv[unit, 0, :, off], kv[unit, 1, :, off] = k, v       # advanced indices around a slice: [T, Hkv, D]
            o = meta.kernel.run(q, kv, meta.handle)
        return self.o_proj(o.reshape(t, self.heads * self.head_dim))
