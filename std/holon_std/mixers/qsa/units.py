"""`std.mixers.qsa@1`: Qwen3.8-Flash-Next's indexed sparse attention (docs/v1-flash-next.md D6; op contracts in
`kernels.py`).

The attention is gqa's, composed by Ref (`attn`: `std.mixers.gqa@1` with the same parameters, so its projections,
q / k norms, rotation, KV layouts and kernels are gqa's; its class's state hooks forwarded, `BuildCtx.unit_class`):
spec `"kv"` is gqa's. The indexer adds spec `"idx"` in its own page group: per page, the raw index keys and the pooled
keys. Every step writes them (`qsa_index@1`); a step whose rows all end within `dense_limit` tokens (host-known upper
bounds: a verify row by its width) attends densely through gqa, any other step selects each query's blocks
(`qsa_select@1`) and attends over them (`qsa_attend@1`, the output gate applied at its store), so a short request's
bits depend on whether a long one shares its step (D6; the snapshot fixes the schedule). A model-only build attends
over each whole sequence the same way, its pages made for the call. The indexer's q and k projections are two
linears as llama.cpp's converter splits them, or with `fused_index_qk` one (`index_qk_proj`, q's rows then k's), as
transformers stores them; split linears that can share one matmul are merged into `index_qk_proj` once loaded (one
launch less per call)."""
from __future__ import annotations

import dataclasses
from typing import Any

import torch
from holon.api import Paged, Ref, StateSpec
from torch import nn

from . import kernels
from .reference import cos_sin

GQA = "std.mixers.gqa@1"
LINEAR = "std.ops.linear@1"


@dataclasses.dataclass(eq=False)
class IndexMeta:
    """One step's indexer metadata for every QSA layer: the idx pages, whether the step attends densely, and (from
    `prepare_device`) each token's table row."""

    pages: Any
    dense: bool
    tokens: int
    graph_key: Any
    seq: torch.Tensor | None = None


class IndexPlanner:
    mixes_row_kinds = True

    def __init__(self, limit: int) -> None:
        self.limit = limit

    def plan_bounds(self, batch, pages) -> IndexMeta:
        plan = batch.plan
        lens = plan.n_query.tolist()
        dense = bool(((plan.start_upper + plan.n_query) <= self.limit).all())
        uniform = bool(lens) and not plan.n_decode and len(set(lens)) == 1 and (
            lens[0] == 1 or bool((plan.n_logits == plan.n_query).all()))          # decode, or verify of one width
        return IndexMeta(pages, dense, int(sum(lens)), ("qsa", len(lens), lens[0], dense) if uniform else None)

    def prepare_device(self, meta: IndexMeta, ctx) -> None:
        dev = ctx.batch.dev
        rows = torch.repeat_interleave(torch.arange(len(dev.slots), device=dev.slots.device), dev.query_start.diff(),
                                       output_size=meta.tokens)
        meta.seq = dev.slots[rows].to(torch.int32)


def _gqa(params: dict) -> Ref:
    keys = ("heads", "kv_heads", "head_dim", "rope", "qk_norm", "output_gate", "causal", "window")
    return Ref(GQA, **{k: params[k] for k in keys if k in params})


class QSA(nn.Module):
    @classmethod
    def state_specs(cls, params: dict, layers: tuple[int, ...], ctx) -> list[StateSpec]:
        attn = _gqa(params)
        page, ratio, dk = ctx.options.get("page_size", 16), params["ratio"], params["index_head_dim"]
        if page % ratio:
            raise ValueError(f"std.mixers.qsa: page_size {page} must be a multiple of the ratio {ratio}")
        per_layer = page * dk * ctx.dtype.itemsize + page // ratio * dk * 4             # raw keys, pooled f32
        return [*ctx.unit_class(attn).state_specs(dict(attn.params), layers, ctx),
                StateSpec("idx", "paged@1", Paged(per_layer * len(layers), page), layers, "qsa.idx",
                          segments=len(layers))]

    @classmethod
    def planner(cls, params: dict, specs: list[StateSpec], ctx) -> Any:
        if specs[0].layout == "qsa.idx":                 # (an aux model's spec names are namespaced)
            return IndexPlanner(kernels.dense_limit(params["budget"], params["ratio"]))
        attn = _gqa(params)
        return ctx.unit_class(attn).planner(dict(attn.params), specs, ctx)

    def __init__(self, ctx, *, layer: int, hidden: int, heads: int, kv_heads: int, head_dim: int, rope: Ref,
                 index_heads: int, index_head_dim: int, ratio: int, budget: int, qk_norm: Ref | None = None,
                 output_gate: str = "none", eps: float = 1e-6, causal: bool = True, window: int = 0,
                 fused_index_qk: bool = False) -> None:
        super().__init__()
        params = {k: v for k, v in locals().items() if k not in ("self", "ctx", "layer", "hidden", "__class__")}
        self.attn = ctx.unit(_gqa(params), layer=layer, hidden=hidden)
        self.hi, self.dk, self.ratio, self.budget = index_heads, index_head_dim, ratio, budget
        self.limit = kernels.dense_limit(budget, ratio)
        if fused_index_qk:
            self.index_qk_proj = ctx.unit(Ref(LINEAR, in_features=hidden,
                                              out_features=(index_heads + 1) * index_head_dim))
        else:
            self.index_q_proj = ctx.unit(Ref(LINEAR, in_features=hidden, out_features=index_heads * index_head_dim))
            self.index_k_proj = ctx.unit(Ref(LINEAR, in_features=hidden, out_features=index_head_dim))
        self.index_q_norm = ctx.unit(qk_norm, dim=index_head_dim) if qk_norm else None
        self.index_k_norm = nn.Parameter(torch.empty(index_head_dim, dtype=ctx.dtype), requires_grad=False)
        self.index_rope = ctx.unit(rope, head_dim=index_head_dim)
        rotary, theta = rope.param("rotary_dim") or index_head_dim, rope.param("theta")
        positions = ctx.options.get("max_model_len", 8192)
        self.cos_sin = ctx.static(f"std.mixers.qsa.cos_sin[{rotary},{theta}]", (positions, rotary), torch.float32)
        with torch.device("cpu"):                                     # (a build runs under the meta device)
            self.cos_sin.copy_(cos_sin(positions, rotary, theta))     # the pooled keys' rotation (rope@1's table)
        dtype, site = str(ctx.dtype).removeprefix("torch."), f"std.mixers.qsa[layer={layer}]"
        kv, idx = ctx.state("kv", layer), ctx.state("idx", layer)
        page = kv[0].shape.tokens_per_unit if kv is not None else 16
        layout = kv[0].layout if kv is not None else "kv.hnd"
        self._idx = None if idx is None else (idx[1], idx[0].layers.index(layer), len(idx[0].layers), page)
        facts = {"head_dim": index_head_dim, "ratio": ratio, "page": page, "rotary_dim": rotary,
                 "max_positions": positions, "eps": eps, "dtype": dtype}
        self._index = ctx.select_kernel(kernels.INDEX, site=site, **facts).impl(ctx, **facts)
        facts = {"heads": index_heads, "head_dim": index_head_dim, "ratio": ratio, "budget": budget, "page": page}
        self._select = ctx.select_kernel(kernels.SELECT, site=site, **facts).impl(ctx, **facts)
        kv_dtype = {"kv.hnd.q8_0": "q8_0", "kv.hnd.fp8": "float8_e4m3fn"}.get(layout, dtype)
        facts = {"heads": heads, "kv_heads": kv_heads, "head_dim": head_dim, "ratio": ratio,
                 "width": kernels.width(budget, ratio), "page": page, "layout": layout, "q_dtype": dtype,
                 "kv_dtype": kv_dtype}
        self._attend = ctx.select_kernel(kernels.ATTEND, site=site, **facts).impl(ctx, **facts)
        self.page = page

    def finalize_weights(self, ctx) -> None:
        if not hasattr(self, "index_qk_proj") and (merge := getattr(type(self.index_q_proj), "merge", None)):
            merged = merge(self.index_q_proj, self.index_k_proj)        # q's rows then k's: the fused layout
            if merged is not None:
                self.index_qk_proj = merged

    def _idx_layer(self) -> tuple[torch.Tensor, torch.Tensor]:
        """This layer's raw [units, page, Dk] (model dtype) and pooled [units, page / ratio, Dk] (f32) keys."""
        pool, index, n, page = self._idx
        units = pool.view(torch.uint8, (n, pool.capacity, pool.unit_bytes // n))[index]
        raw_bytes = page * self.dk * self.index_k_norm.element_size()
        return (units[:, :raw_bytes].view(self.index_k_norm.dtype).view(-1, page, self.dk),
                units[:, raw_bytes:].view(torch.float32).view(-1, page // self.ratio, self.dk))

    def _indexer(self, x: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The indexer's queries [T, Hi, Dk] (normed and rotated at their positions) and raw keys [T, Dk] (fused: a
        strided view of the projection, which `qsa_index@1` reads in place)."""
        t = x.shape[0]
        if hasattr(self, "index_qk_proj"):
            q, k = self.index_qk_proj(x).split([self.hi * self.dk, self.dk], dim=1)
        else:
            q, k = self.index_q_proj(x), self.index_k_proj(x)
        q = q.reshape(t, self.hi, self.dk)
        if self.index_q_norm is not None:
            q = self.index_q_norm(q)
        q, _ = self.index_rope(q, q.new_empty(t, 0, self.dk), positions)       # q only (rope@1 with no key heads)
        return q, k

    def forward(self, x: torch.Tensor, ctx) -> torch.Tensor:
        if self._idx is None:
            return self._whole_sequences(x, ctx)
        dev, meta = ctx.batch.dev, ctx.meta("idx")
        iq, ik = self._indexer(x, dev.positions)
        raw, pooled = self._idx_layer()
        table = meta.pages.table
        self._index(ik, raw, pooled, table, meta.seq, dev.positions, self.index_k_norm, self.cos_sin)
        if meta.dense:
            return self.attn(x, ctx)
        kv = self.attn.kv_layer()
        q, gate = self.attn.project_and_write(x, ctx, kv)
        blocks = self._select(iq, pooled, table, meta.seq, dev.positions)
        o = self._attend(q, kv, blocks, ctx.meta("kv").pages.table, meta.seq, dev.positions, gate)
        return self.attn.o_proj(o.reshape(x.shape[0], -1))

    def write_kv(self, x: torch.Tensor, ctx) -> None:
        """The step's index keys and its attention's keys and values written at its positions, without attending
        (gqa's `write_kv`): what a later step of this layer reads of these tokens."""
        dev, meta = ctx.batch.dev, ctx.meta("idx")
        _, ik = self._indexer(x, dev.positions)
        raw, pooled = self._idx_layer()
        self._index(ik, raw, pooled, meta.pages.table, meta.seq, dev.positions, self.index_k_norm, self.cos_sin)
        self.attn.write_kv(x, ctx)

    def _whole_sequences(self, x: torch.Tensor, ctx) -> torch.Tensor:
        """A model-only build: dense through gqa when every sequence fits `dense_limit`, else sparse over pages made
        for this call (one table row per sequence)."""
        dev = ctx.batch.dev
        lens = dev.query_start.diff().tolist()
        if max(lens) <= self.limit:
            return self.attn(x, ctx)
        page, t, rows = self.page, x.shape[0], len(lens)
        pages = -(-max(lens) // page)
        table = torch.arange(rows * pages, dtype=torch.int32, device=x.device).view(rows, pages)
        seq = torch.repeat_interleave(torch.arange(rows, device=x.device), dev.query_start.diff()).to(torch.int32)
        unit, off = table[seq.long(), dev.positions // page].long(), dev.positions % page
        q, gate, k, v = self.attn.project(x, ctx)
        kv = x.new_zeros(rows * pages, 2, self.attn.kv_heads, page, self.attn.head_dim)
        kv[unit, 0, :, off], kv[unit, 1, :, off] = k, v
        raw = x.new_zeros(rows * pages, page, self.dk)
        pooled = torch.zeros(rows * pages, page // self.ratio, self.dk, device=x.device)
        iq, ik = self._indexer(x, dev.positions)
        self._index(ik, raw, pooled, table, seq, dev.positions, self.index_k_norm, self.cos_sin)
        blocks = self._select(iq, pooled, table, seq, dev.positions)
        o = self._attend(q.contiguous(), kv, blocks, table, seq, dev.positions, gate)
        return self.attn.o_proj(o.reshape(t, -1))
