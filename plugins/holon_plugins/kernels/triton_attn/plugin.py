"""Our Triton paged attention as `attention.paged_decode@1` and `attention.paged_verify@1` kernels, a paged prefill
for q8_0 KV, `kv_write@1` for every KV layout, and `attention.qk_rope_kv@1` (q/k norm, rotation and the KV write in
one launch, docs/m07-perf.md D3) for the model-dtype and q8_0 layouts (op contracts: std.mixers.gqa)."""
from __future__ import annotations

import dataclasses
import functools

import torch
from holon.api import Constraint, KernelSpec

from .kernel import (
    FP8,
    MODEL,
    Q8_0,
    SHARED,
    Config,
    block_m,
    fitted,
    paged_attention,
    paged_prefill,
    qk_rope_kv,
    splits_for,
    write_model,
    write_q8_0,
)


@dataclasses.dataclass(frozen=True)
class _Handle:
    table: torch.Tensor
    rows: torch.Tensor
    lens: torch.Tensor
    splits: int
    source_lens: torch.Tensor


class Decode:
    """One query row per sequence. The block table and lengths are read on the device from buffers this instance
    owns (stable addresses for graph capture); the only host decision is the split count, taken from the batch size
    and the table's capacity, so one graph per batch bucket serves every length (splits past a sequence's end are
    empty).

    The launch config is the class's pinned one for the head_dim, `fitted` at build to the device's shared memory
    for each row width the deployment declares (one; verify: 1 + k for every k up to `max_draft_tokens`) — a pure
    function of the class, the facts and the widths, so the lock pins it through this plugin's `src` and its
    `[engine]`. A width not declared (a drafter's block) is fitted by the same function when it first runs."""

    def __init__(self, ctx, *, heads: int, kv_heads: int, head_dim: int, page: int, q_dtype: str, kv_dtype: str,
                 layout: str, max_seqs: int, causal: bool = True, window: int = 0) -> None:
        fp = ctx.fingerprint
        self.kv_heads, self.num_sms = kv_heads, fp.num_sms
        self.causal, self.window = causal, window          # the mask facts (one decode row sees all n either way)
        self._fit = functools.partial(
            fitted, Config.pinned(fp.gpu, fp.sm, head_dim), fp.sm,
            SHARED.get(fp.sm) or torch.cuda.get_device_properties(ctx.device).shared_memory_per_block_optin,
            g=heads // kv_heads,
            d=head_dim, page=page, fmt=FORMAT[layout], causal=causal, window=window, dtype=q_dtype)
        self.g, self.cfgs = heads // kv_heads, {}
        for width in sorted(self.widths(ctx), reverse=True):     # each BLOCK_M fitted at its widest declared width
            self.cfg(width)
        self.rows = torch.zeros(max_seqs, dtype=torch.int32, device=ctx.device)
        self.lens = torch.zeros(max_seqs, dtype=torch.int32, device=ctx.device)

    def plan(self, batch, pages) -> _Handle:
        b = len(batch.plan.slots)
        self.rows[:b].copy_(batch.dev.slots, non_blocking=True)
        self.lens[b:].zero_()                    # rows the instance holds beyond this batch: no stale lengths
        span = pages.table.shape[1] * pages.page                     # keys a row may see, at most
        if self.window:
            span = min(span, self.window + int(batch.plan.n_query.max(initial=1)) - 1)
        splits = splits_for(b, self.kv_heads, span, self.num_sms)
        return _Handle(pages.table, self.rows[:b], self.lens[:b], splits, pages.lens)

    def prepare(self, h: _Handle) -> None:
        h.lens.copy_(h.source_lens)             # exact device lengths of this step

    def graph_key(self, h: _Handle) -> tuple:
        return ("triton_attn.decode", len(h.rows), h.splits)

    def run(self, q: torch.Tensor, kv: torch.Tensor, h: _Handle) -> torch.Tensor:
        nq = len(q) // len(h.rows)
        return paged_attention(q, kv, h.table, h.rows, h.lens, nq=nq, splits=h.splits, cfg=self.cfg(nq),
                               causal=self.causal, window=self.window)

    @staticmethod
    def widths(ctx) -> tuple[int, ...]:
        return (1,)

    def cfg(self, nq: int) -> Config:
        bm = block_m(nq, self.g)
        if bm not in self.cfgs:
            self.cfgs[bm] = self._fit(nq=nq)
        return self.cfgs[bm]


class Verify(Decode):
    """1 + k query rows per sequence (the same k for all), causal among themselves or (`causal` False) not, within a
    `window` if one is given; lengths from the device."""

    def graph_key(self, h: _Handle) -> tuple:
        return ("triton_attn.verify", len(h.rows), h.splits)

    @staticmethod
    def widths(ctx) -> tuple[int, ...]:
        return tuple(range(1, 2 + ctx.options.get("max_draft_tokens", 0)))


class Prefill:
    """Ragged chunk rows, each continuing its sequence (bottom-right causal), for KV formats FlashInfer cannot read.
    Planned from the host's row counts alone; slots, query starts and lengths are read on the device in `run`. One
    program walks a row's whole KV (no split), so a mixed step's decode rows go to the decode kernel instead (no
    `mixes_row_kinds`; docs/m07-perf.md §15)."""

    def __init__(self, ctx, **facts) -> None:
        pass

    def plan(self, batch, pages) -> tuple:
        return (pages.table, batch.dev.slots, pages.lens, batch.dev.query_start), int(batch.plan.n_query.max(initial=0))

    def run(self, q: torch.Tensor, kv, h: tuple) -> torch.Tensor:
        (table, rows, lens, query_start), max_rows = h
        return paged_prefill(q, kv, table, rows, lens, query_start, max_rows=max_rows)


# KV layout -> its kv_dtype fact (None: the model dtype, q_dtype), and the stored format the kernel reads
KV_DTYPE = {"kv.hnd": None, "kv.hnd.fp8": "float8_e4m3fn", "kv.hnd.q8_0": "q8_0"}
FORMAT = {"kv.hnd": MODEL.value, "kv.hnd.fp8": FP8.value, "kv.hnd.q8_0": Q8_0.value}


def _supports(layout: str, masks: bool = True):
    """`masks`: the kernel implements the `causal` / `window` facts (decode and verify; the prefill does not)."""
    def supports(f: dict) -> bool:
        return (f["layout"] == layout and f["kv_dtype"] == (KV_DTYPE[layout] or f["q_dtype"])
                and f["q_dtype"] in ("bfloat16", "float16") and f["head_dim"] in (64, 128, 256)
                and f["heads"] % f["kv_heads"] == 0 and (masks or not {"causal", "window"} & f.keys()))
    return supports


class KvWrite:
    """`kv_write@1` for the model-dtype layout (`kv.hnd`): this step's k/v copied into their pages, one Triton launch
    for both (docs/m07-perf.md D3; it was two index_put kernels)."""

    def __init__(self, ctx, **facts) -> None:
        pass

    def __call__(self, k, v, kv, unit, off) -> None:
        write_model(k, v, kv, unit, off)


class KvWriteFp8(KvWrite):
    """`kv.hnd.fp8`: e4m3 with saturation (out-of-range values clamp to ±448; a plain cast would give NaN)."""

    def __call__(self, k, v, kv, unit, off) -> None:
        kv[unit, :, :, off] = torch.stack((k, v), 1).clamp_(-448.0, 448.0).to(kv.dtype)   # [T, 2, Hkv, D]


class KvWriteQ8(KvWrite):
    """`kv.hnd.q8_0`: GGUF q8_0 per 32 values, one Triton launch for k and v."""

    def __call__(self, k, v, kv, unit, off) -> None:
        write_q8_0(k, v, kv, unit, off)


class QkRopeKv:
    """`attention.qk_rope_kv@1`: the rope kernel's cos|sin table (its static buffer, by name) and the norm's eps and
    offset; one launch per call."""

    def __init__(self, ctx, *, rotary_dim: int, max_positions: int, cos_sin: str, eps: float, offset: float,
                 **facts) -> None:
        self.table = ctx.static(cos_sin, (max_positions, rotary_dim), torch.float32)
        self.args = {"eps": eps, "offset": offset, "rotary_dim": rotary_dim}

    def __call__(self, q, k, v, positions, q_weight, k_weight, kv, unit, off) -> torch.Tensor:
        return qk_rope_kv(q, k, v, positions, q_weight, k_weight, self.table, kv, unit, off, **self.args)


def _write_supports(layout: str):
    def supports(f: dict) -> bool:
        return f["layout"] == layout and (layout != "kv.hnd.q8_0" or f["head_dim"] in (64, 128, 256))
    return supports


def _fused_supports(layout: str):
    """Every fact set its KV writer's spec promises (`config["fused"]`): the writer's, the reference ops the units
    name, an even rotary_dim within head_dim, a half or f32 model dtype."""
    writes = _write_supports(layout)

    def supports(f: dict) -> bool:
        return (writes(f) and f["norm"] == "rmsnorm@1" and f["rope"] == "rope@1" and f["rotary_dim"] % 2 == 0
                and 0 < f["rotary_dim"] <= f["head_dim"] and f["dtype"] in ("bfloat16", "float16", "float32"))
    return supports


class Plugin:
    def register(self, regs) -> None:
        anywhere, fp8_sms = Constraint(sm_min=80, sm_max=129), Constraint(sm_min=89, sm_max=129)
        props = frozenset({"graph_safe", "deterministic"})
        for layout, impl in (("kv.hnd", KvWrite), ("kv.hnd.fp8", KvWriteFp8), ("kv.hnd.q8_0", KvWriteQ8)):
            fused = layout != "kv.hnd.fp8"                     # D3 covers the model dtype and q8_0
            regs.kernels.add("kv_write", KernelSpec(
                op="kv_write", provider="kernels.triton_attn", impl=impl, requires=anywhere,
                supports=_write_supports(layout), props=props, priority=10,
                config={"fused": ["attention.qk_rope_kv@1"]} if fused else {}))
            if fused:
                regs.kernels.add("attention.qk_rope_kv", KernelSpec(
                    op="attention.qk_rope_kv", provider="kernels.triton_attn", impl=QkRopeKv, requires=anywhere,
                    supports=_fused_supports(layout), props=props, priority=10))
        # Decode and verify read every layout; fp8 is offered where FlashInfer serves its prefill (sm_89 on: no fast
        # e4m3 before), and decode stays ours there too, as for the model dtype: graph-safe, where FlashInfer's is not.
        for op, impl, layout, requires in (
                ("attention.paged_decode", Decode, "kv.hnd", anywhere),
                ("attention.paged_decode", Decode, "kv.hnd.q8_0", anywhere),
                ("attention.paged_decode", Decode, "kv.hnd.fp8", fp8_sms),
                ("attention.paged_verify", Verify, "kv.hnd", anywhere),
                ("attention.paged_verify", Verify, "kv.hnd.q8_0", anywhere),
                ("attention.paged_verify", Verify, "kv.hnd.fp8", fp8_sms),
                ("attention.paged_prefill", Prefill, "kv.hnd.q8_0", anywhere)):
            regs.kernels.add(op, KernelSpec(
                op=op, provider="kernels.triton_attn", impl=impl, in_layouts={"kv": layout}, requires=requires,
                supports=_supports(layout, impl is not Prefill),
                props=props if impl is not Prefill else frozenset({"deterministic"}),
                priority=10))
