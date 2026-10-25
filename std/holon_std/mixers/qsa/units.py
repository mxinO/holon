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
from .reference import all_blocks, cos_sin

GQA = "std.mixers.gqa@1"
LINEAR = "std.ops.linear@1"



def _call_pages(query_start: torch.Tensor, positions: torch.Tensor, page: int):
    """Pages made for one call (a model-only build: one table row per sequence, each its whole length): (units, the
    table [rows, pages] int32, each token's sequence [T] int32, its unit and offset)."""
    lens = query_start.diff()
    rows, pages = len(lens), -(-int(lens.max()) // page)
    table = torch.arange(rows * pages, dtype=torch.int32, device=positions.device).view(rows, pages)
    seq = torch.repeat_interleave(torch.arange(rows, device=positions.device), lens).to(torch.int32)
    return rows * pages, table, seq, table[seq.long(), positions // page].long(), positions % page

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
        if max(dev.query_start.diff().tolist()) <= self.limit:
            return self.attn(x, ctx)
        page, t = self.page, x.shape[0]
        units, table, seq, unit, off = _call_pages(dev.query_start, dev.positions, page)
        q, gate, k, v = self.attn.project(x, ctx)
        kv = x.new_zeros(units, 2, self.attn.kv_heads, page, self.attn.head_dim)
        kv[unit, 0, :, off], kv[unit, 1, :, off] = k, v
        raw = x.new_zeros(units, page, self.dk)
        pooled = torch.zeros(units, page // self.ratio, self.dk, device=x.device)
        iq, ik = self._indexer(x, dev.positions)
        self._index(ik, raw, pooled, table, seq, dev.positions, self.index_k_norm, self.cos_sin)
        blocks = self._select(iq, pooled, table, seq, dev.positions)
        o = self._attend(q.contiguous(), kv, blocks, table, seq, dev.positions, gate)
        return self.attn.o_proj(o.reshape(t, -1))


@dataclasses.dataclass(eq=False)
class LatentMeta:
    """One step's latent-cache metadata for every DSA layer: its pages and each token's (unit, offset) to write."""

    pages: Any
    graph_key: Any
    write_unit: torch.Tensor | None = None
    write_off: torch.Tensor | None = None


class LatentPlanner:
    """The DSA latent group's: the step's pages and write positions (the attention reads the selected tokens
    itself)."""

    mixes_row_kinds = True

    def plan_bounds(self, batch, pages) -> LatentMeta:
        plan = batch.plan
        lens = plan.n_query.tolist()
        uniform = bool(lens) and not plan.n_decode and len(set(lens)) == 1 and (
            lens[0] == 1 or bool((plan.n_logits == plan.n_query).all()))
        return LatentMeta(pages, ("dsa", len(lens), lens[0]) if uniform else None)

    def prepare_device(self, meta: LatentMeta, ctx) -> None:
        pages = meta.pages
        meta.write_unit, meta.write_off = pages.write_index // pages.page, pages.write_index % pages.page


class DSAMLA(nn.Module):
    """`std.mixers.dsa_mla@1`: GLM-5.3-Flash's DSA attention (docs/v1-glm.md D5; HF's `Glm5NextTextAttention` and
    `Glm5NextTextIndexer`), NoPE MLA absorbed over its latent, the indexer reading MLA's query residual `q_r`.

    Per step: `q_r = norm(q_a(x))`, the latent `c = norm(kv_a(x))` written at the tokens' positions (spec `"kv"`); the
    indexer's keys `LayerNorm(wk(x))` (its weight and bias, eps `index_eps`) and gates `W_g(x)` stored and pooled
    (`dsa_index@1`, spec `"idx"`), its queries `wq_b(q_r)` and signed head weights `weights_proj(x) / √Hi` choosing each
    query's blocks (`qsa_select@1` with head weights); a step whose rows all end within `dense_limit` attends over every
    block it sees through the same kernel (the planner's `dense` only skips scoring, D5), so a row's bits do not depend
    on its step's other rows. Then `q = q_b(q_r)` per head, absorbed `q̃_h = W_k,hᵀ q_h` (`k_b`), `mla_attend@1` over
    the latent, `o_h = W_v,h ctx_h` (`v_b`), `o_proj`. With `rows` > 0 the query path (`q_b`, `q̃`, the attention, `v_b`,
    `o_proj`) runs in sub-batches of `rows` queries, every key in place (D9). `k_b` / `v_b` are held in the model dtype
    (D5: the release's BF16)."""

    @classmethod
    def state_specs(cls, params: dict, layers: tuple[int, ...], ctx) -> list[StateSpec]:
        page, ratio, dk = ctx.options.get("page_size", 16), params["ratio"], params["index_head_dim"]
        if page % ratio:
            raise ValueError(f"std.mixers.dsa_mla: page_size {page} must be a multiple of the ratio {ratio}")
        latent = page * params["kv_lora"] * ctx.dtype.itemsize
        idx = 2 * page * dk * ctx.dtype.itemsize + page // ratio * dk * 4           # raw keys and gates, pooled f32
        n = len(layers)
        return [StateSpec("kv", "paged@1", Paged(latent * n, page), layers, "dsa.kv", segments=n),
                StateSpec("idx", "paged@1", Paged(idx * n, page), layers, "dsa.idx", segments=n)]

    @classmethod
    def planner(cls, params: dict, specs: list[StateSpec], ctx) -> Any:
        if specs[0].layout.endswith("idx"):
            return IndexPlanner(kernels.dense_limit(params["budget"], params["ratio"]))
        return LatentPlanner()

    def __init__(self, ctx, *, layer: int, hidden: int, heads: int, q_lora: int, kv_lora: int, nope: int, v: int,
                 index_heads: int, index_head_dim: int, ratio: int, budget: int, norm: Ref, index_eps: float = 1e-6,
                 rows: int = 0) -> None:
        super().__init__()
        lin = lambda i, o: ctx.unit(Ref(LINEAR, in_features=i, out_features=o))       # noqa: E731
        self.h, self.dc, self.nope, self.v, self.rows = heads, kv_lora, nope, v, rows
        self.hi, self.dk, self.ratio, self.budget = index_heads, index_head_dim, ratio, budget
        self.limit, self.index_eps = kernels.dense_limit(budget, ratio), index_eps
        self.q_a, self.q_a_norm = lin(hidden, q_lora), ctx.unit(norm, dim=q_lora)
        self.q_b = lin(q_lora, heads * nope)
        self.kv_a, self.kv_a_norm = lin(hidden, kv_lora), ctx.unit(norm, dim=kv_lora)
        self.k_b = nn.Parameter(torch.empty(heads, kv_lora, nope, dtype=ctx.dtype), requires_grad=False)
        self.v_b = nn.Parameter(torch.empty(heads, v, kv_lora, dtype=ctx.dtype), requires_grad=False)
        self.o_proj = lin(heads * v, hidden)
        self.wq_b, self.wk = lin(q_lora, index_heads * index_head_dim), lin(hidden, index_head_dim)
        self.weights_proj, self.compress_gate = lin(hidden, index_heads), lin(hidden, index_head_dim)
        f32 = lambda *s: nn.Parameter(torch.empty(*s, dtype=torch.float32), requires_grad=False)   # noqa: E731
        self.k_norm_weight, self.k_norm_bias = f32(index_head_dim), f32(index_head_dim)
        self.ape = f32(ratio, index_head_dim)
        self.scale = nope ** -0.5                                         # 1 / √(the NoPE query-key head dim)
        dtype, site = str(ctx.dtype).removeprefix("torch."), f"std.mixers.dsa_mla[layer={layer}]"
        kv, idx = ctx.state("kv", layer), ctx.state("idx", layer)
        page = kv[0].shape.tokens_per_unit if kv is not None else ctx.options.get("page_size", 16)
        self._kv = None if kv is None else (kv[1], kv[0].layers.index(layer), len(kv[0].layers))
        self._idx = None if idx is None else (idx[1], idx[0].layers.index(layer), len(idx[0].layers))
        self.page, self.width = page, kernels.width(budget, ratio)
        facts = {"head_dim": index_head_dim, "ratio": ratio, "page": page, "dtype": dtype}
        self._index = ctx.select_kernel(kernels.DSA_INDEX, site=site, **facts).impl(ctx, **facts)
        facts = {"heads": index_heads, "head_dim": index_head_dim, "ratio": ratio, "budget": budget, "page": page,
                 "head_weights": True}
        self._select = ctx.select_kernel(kernels.SELECT, site=site, **facts).impl(ctx, **facts)
        facts = {"heads": heads, "latent": kv_lora, "ratio": ratio, "width": self.width, "page": page,
                 "scale": self.scale, "q_dtype": "float32", "kv_dtype": dtype}       # (q̃ f32, D5)
        self._attend = ctx.select_kernel(kernels.MLA_ATTEND, site=site, **facts).impl(ctx, **facts)
        self._dtype = ctx.dtype

    def adopt_weights(self, ctx, name: str, q, index: int | None = None) -> bool:
        """A quantized `k_b` / `v_b` (the release's Q6_K) dequantized at load into the model dtype (D5: a recorded
        rounding, as Maya's pack makes; gguf-py's dequantization, the GGUF oracle). Here, not as a value transform:
        the core refuses transforms on quantized tensors, so this is the core-free path (B3 nit)."""
        if name not in ("k_b", "v_b") or index is not None or not q.layout.scheme.elem.startswith("ggml."):
            return False
        import gguf

        qtype = gguf.GGMLQuantizationType[q.layout.scheme.elem.removeprefix("ggml.").upper()]
        x = gguf.quants.dequantize(q.parts["blocks"].cpu().numpy(), qtype)
        shape = tuple(getattr(self, name).shape)
        if x.size != torch.Size(shape).numel():
            return False
        dense = torch.from_numpy(x.reshape(shape).copy()).to(device=ctx.device, dtype=self._dtype)
        setattr(self, name, nn.Parameter(dense, requires_grad=False))
        return True

    def _latent(self, pool=None) -> torch.Tensor:
        """This layer's latent pages [units, page, Dc] (model dtype)."""
        pool, index, n = self._kv if pool is None else (pool, self._kv[1], self._kv[2])
        return pool.view(self._dtype, (n, pool.capacity, self.page, self.dc))[index]

    def _idx_layer(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """This layer's raw keys and gates [units, page, Dk] (model dtype) and pooled keys [units, page / ratio, Dk]
        f32."""
        pool, index, n = self._idx
        units = pool.view(torch.uint8, (n, pool.capacity, pool.unit_bytes // n))[index]
        half = self.page * self.dk * self._dtype.itemsize
        view = lambda a, b: units[:, a:b].view(self._dtype).view(-1, self.page, self.dk)     # noqa: E731
        pooled = units[:, 2 * half:].view(torch.float32).view(-1, self.page // self.ratio, self.dk)
        return view(0, half), view(half, 2 * half), pooled

    def _indexer(self, x: torch.Tensor, q_r: torch.Tensor):
        """The indexer's queries [T, Hi, Dk], signed head weights [T, Hi] f32 (÷ √Hi), keys and gates [T, Dk]."""
        k = torch.nn.functional.layer_norm(self.wk(x).float(), (self.dk,), self.k_norm_weight, self.k_norm_bias,
                                           self.index_eps).to(self._dtype)
        w = self.weights_proj(x).float() * self.hi ** -0.5
        return self.wq_b(q_r).reshape(len(x), self.hi, self.dk), w, k, self.compress_gate(x)

    def _attention(self, q_r: torch.Tensor, kv: torch.Tensor, blocks: torch.Tensor, table: torch.Tensor,
                   seq: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """The query path for its rows: q, absorbed q̃, the latent attention, v_b, o_proj."""
        t = len(q_r)
        q = self.q_b(q_r).reshape(t, self.h, self.nope)
        q_abs = torch.einsum("thn,hcn->thc", q.float(), self.k_b.float())             # q̃ in f32 (D5: decode's)
        ctx_ = self._attend(q_abs, kv, blocks, table, seq, positions)
        o = torch.einsum("thc,hvc->thv", ctx_.float(), self.v_b.float()).to(self._dtype)
        return self.o_proj(o.reshape(t, self.h * self.v))

    def forward(self, x: torch.Tensor, ctx) -> torch.Tensor:
        if self._kv is None:
            return self._whole_sequences(x, ctx)
        dev, meta, latent_meta = ctx.batch.dev, ctx.meta("idx"), ctx.meta("kv")
        q_r = self.q_a_norm(self.q_a(x))
        kv = self._latent()
        kv[latent_meta.write_unit, latent_meta.write_off] = self.kv_a_norm(self.kv_a(x)).to(kv.dtype)
        iq, w, k, gate = self._indexer(x, q_r)
        raw, gates, pooled = self._idx_layer()
        self._index(k, gate, raw, gates, pooled, self.ape, meta.pages.table, meta.seq, dev.positions)
        blocks = (all_blocks(dev.positions, self.ratio, self.width) if meta.dense else
                  self._select(iq, pooled, meta.pages.table, meta.seq, dev.positions, w))
        table, step = latent_meta.pages.table, self.rows or len(x)
        if step >= len(x):
            return self._attention(q_r, kv, blocks, table, meta.seq, dev.positions)
        out = x.new_empty(x.shape)                          # into one output: a piece's at a time (C-m5)
        for a in range(0, len(x), step):
            out[a:a + step] = self._attention(q_r[a:a + step], kv, blocks[a:a + step], table, meta.seq[a:a + step],
                                              dev.positions[a:a + step])
        return out

    def write_kv(self, x: torch.Tensor, ctx) -> None:
        """The step's latent and index keys written at its positions, without attending."""
        dev, meta, latent_meta = ctx.batch.dev, ctx.meta("idx"), ctx.meta("kv")
        q_r = self.q_a_norm(self.q_a(x))
        kv = self._latent()
        kv[latent_meta.write_unit, latent_meta.write_off] = self.kv_a_norm(self.kv_a(x)).to(kv.dtype)
        _, _, k, gate = self._indexer(x, q_r)
        raw, gates, pooled = self._idx_layer()
        self._index(k, gate, raw, gates, pooled, self.ape, meta.pages.table, meta.seq, dev.positions)

    def _whole_sequences(self, x: torch.Tensor, ctx) -> torch.Tensor:
        """A model-only build: each sequence over pages made for this call."""
        dev, page = ctx.batch.dev, self.page
        units, table, seq, unit, off = _call_pages(dev.query_start, dev.positions, page)
        kv = x.new_zeros(units, page, self.dc)
        q_r = self.q_a_norm(self.q_a(x))
        kv[unit, off] = self.kv_a_norm(self.kv_a(x)).to(kv.dtype)
        iq, w, k, gate = self._indexer(x, q_r)
        raw, gates = x.new_zeros(units, page, self.dk), x.new_zeros(units, page, self.dk)
        pooled = torch.zeros(units, page // self.ratio, self.dk, device=x.device)
        self._index(k, gate, raw, gates, pooled, self.ape, table, seq, dev.positions)
        dense = int(dev.query_start.diff().max()) <= self.limit
        blocks = (all_blocks(dev.positions, self.ratio, self.width) if dense else
                  self._select(iq, pooled, table, seq, dev.positions, w))
        return self._attention(q_r, kv, blocks, table, seq, dev.positions)
