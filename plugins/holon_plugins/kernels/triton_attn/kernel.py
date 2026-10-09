"""Paged attention for decode (and, from M0.4, verify) rows, reading the block table and lengths on the device; a
paged prefill for KV formats FlashInfer cannot read; and the KV write of the quantized formats.

q:      [B * NQ, Hq, D]                  NQ query rows per sequence (1 = decode), uniform per launch
kv:     [units, 2, Hkv, page, D]         one layer of HND paged KV; this step's tokens are already written. In the
                                         model dtype, float8_e4m3fn, or q8_0's pair (int8 values of that shape, fp16
                                         scales [units, 2, Hkv, page, D / 32]); std.mixers.gqa.kernels has the formats
table:  [slots, max_pages] int32         persistent block table, one row per request slot
rows:   [B] int32                        the slot (table row) of each batch row
lens:   [B] int32                        KV length per row including this step's tokens

Row i of sequence b sits at position lens[b] - NQ + i and attends causally. One program handles all NQ * group
query rows of one (sequence, kv head, split), so K/V are read once per kv head. The split count is a launch
constant chosen on the host from an upper bound on the lengths, never from the lengths themselves; split
boundaries are computed on the device and partials are merged in a fixed order, so the output is run-to-run
deterministic.

Stored formats are decoded in registers right after the load (`_load_kv`): q8_0 as values × scale, rounded once to
fp16 (q joins it there: `_compute_q`; so |K|, |V| must stay below fp16's 65504, which normed keys do), fp8 by moving
its bits into an fp16 (exact, and on every SM: Triton rejects e4m3 on sm_86; a NaN k/v, which the saturating write
keeps, reads back as ±480).
"""
from __future__ import annotations

import dataclasses
import math

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

MODEL, FP8, Q8_0 = tl.constexpr(0), tl.constexpr(1), tl.constexpr(2)     # stored KV formats
Q8_BLOCK = tl.constexpr(32)                                                 # values per q8_0 scale


@triton.jit
def _compute_q(q, qk_scale, FMT: tl.constexpr):
    """q in the dtype K and V are decoded to, and the score scale left to apply. q8_0 decodes to fp16, whose 11-bit
    significand holds int8 × scale far closer than bf16's 8; q is pre-scaled into it (the scale keeps it in range)."""
    if FMT == Q8_0:
        q, qk_scale = (q.to(tl.float32) * qk_scale).to(tl.float16), 1.0
    return q, qk_scale


@triton.jit
def _load_kv(base, scales, off, s_off, ok, FMT: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
             OUT: tl.constexpr):
    """A [BLOCK_N, D] tile of K or V in the compute dtype OUT, read in its stored format."""
    x = tl.load(base + off, mask=ok[:, None], other=0)
    if FMT == Q8_0:
        s = tl.load(scales + s_off, mask=ok[:, None], other=0.0).to(tl.float32)
        x = tl.reshape(tl.reshape(x.to(tl.float32), [BLOCK_N, D // Q8_BLOCK, Q8_BLOCK]) * s[:, :, None],
                       [BLOCK_N, D])
    elif FMT == FP8:
        # e4m3's bits (s eeee mmm) placed as an fp16's (s 0eeee mmm0000000) are the value × 2^-8, subnormals included
        b = x.to(tl.uint16)
        x = (((b & 0x80) << 8) | ((b & 0x7F) << 7)).to(tl.float16, bitcast=True).to(tl.float32) * 256.0
    return x.to(OUT)


@triton.jit
def _paged_attn(Q, K, V, KS, VS, TABLE, ROWS, LENS, OUT, WS_ML, WS_ACC,
                stride_qt, stride_qh, stride_ku, stride_kh, stride_kt, stride_su, stride_sh, stride_st, stride_table,
                qk_scale,
                NQ: tl.constexpr, G: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, SPLITS: tl.constexpr, FMT: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    hq = tl.num_programs(1) * G

    n = tl.load(LENS + b)
    slot = tl.load(ROWS + b)
    m = tl.arange(0, BLOCK_M)
    qi = m // G
    head = h * G + m % G
    row_ok = m < NQ * G
    d = tl.arange(0, D)
    q = tl.load(Q + (b * NQ + qi)[:, None] * stride_qt + head[:, None] * stride_qh + d[None, :],
                mask=row_ok[:, None], other=0.0)
    q, qk_scale = _compute_q(q, qk_scale, FMT)
    last_visible = n - NQ + qi                                    # causal bound per row

    per_split = tl.cdiv(tl.cdiv(n, SPLITS), BLOCK_N) * BLOCK_N
    lo = s * per_split
    hi = tl.minimum(lo + per_split, n)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    k_head, v_head = K + h * stride_kh, V + h * stride_kh
    ks_head, vs_head = KS + h * stride_sh, VS + h * stride_sh
    for start in range(lo, hi, BLOCK_N):
        tok = start + tl.arange(0, BLOCK_N)
        tok_ok = tok < hi
        unit = tl.load(TABLE + slot * stride_table + tok // PAGE, mask=tok_ok, other=0).to(tl.int64)
        off = unit[:, None] * stride_ku + (tok % PAGE)[:, None] * stride_kt + d[None, :]
        s_off = unit[:, None] * stride_su + (tok % PAGE)[:, None] * stride_st + tl.arange(0, D // Q8_BLOCK)[None, :]
        k = _load_kv(k_head, ks_head, off, s_off, tok_ok, FMT, BLOCK_N, D, q.dtype)
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        visible = (tok[None, :] <= last_visible[:, None]) & tok_ok[None, :]
        qk = tl.where(visible, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp2(qk - m_safe[:, None])
        alpha = tl.exp2(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = _load_kv(v_head, vs_head, off, s_off, tok_ok, FMT, BLOCK_N, D, q.dtype)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    row = (b * NQ + qi) * hq + head                               # flat (token, head) index
    if SPLITS == 1:
        out = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]
        tl.store(OUT + row[:, None] * D + d[None, :], out.to(OUT.dtype.element_ty), mask=row_ok[:, None])
    else:
        tl.store(WS_ML + (row * SPLITS + s) * 2, m_i, mask=row_ok)
        tl.store(WS_ML + (row * SPLITS + s) * 2 + 1, l_i, mask=row_ok)
        tl.store(WS_ACC + (row * SPLITS + s)[:, None] * D + d[None, :], acc, mask=row_ok[:, None])


@triton.jit
def _merge_splits(WS_ML, WS_ACC, OUT, D: tl.constexpr, SPLITS: tl.constexpr, SPLITS_P2: tl.constexpr):
    row = tl.program_id(0)
    s = tl.arange(0, SPLITS_P2)
    s_ok = s < SPLITS
    m = tl.load(WS_ML + (row * SPLITS + s) * 2, mask=s_ok, other=float("-inf"))
    l = tl.load(WS_ML + (row * SPLITS + s) * 2 + 1, mask=s_ok, other=0.0)  # noqa: E741
    m_max = tl.max(m, 0)
    w = tl.where(m == float("-inf"), 0.0, tl.exp2(m - m_max))
    d = tl.arange(0, D)
    acc = tl.load(WS_ACC + (row * SPLITS + s)[:, None] * D + d[None, :], mask=s_ok[:, None], other=0.0)
    den = tl.sum(l * w, 0)
    out = tl.sum(acc * w[:, None], 0) / tl.where(den == 0.0, 1.0, den)   # length-0 (padding) rows give zeros
    tl.store(OUT + row * D + d, out.to(OUT.dtype.element_ty))


# Pinned launch configs (spike S1 sweeps): the most robust config per (sm major, head_dim), overridden per
# fingerprint class where the consumer-card sweeps found a better one; anything unswept gets DEFAULT, which is
# correct everywhere and merely untuned. No autotuning at run time.
DEFAULT = (64, 4, 1)
CONFIGS = {
    (8, 128): (64, 4, 1), (8, 256): (32, 4, 3),
    (12, 128): (64, 4, 1), (12, 256): (64, 4, 2),
}
CLASS_CONFIGS = {
    ("rtx-4090", 128): (64, 8, 1), ("rtx-4090", 256): (32, 4, 3),
    ("rtx-5080", 128): (32, 4, 3), ("rtx-5080", 256): (64, 8, 3),
    ("rtx-5090", 128): (64, 4, 3), ("rtx-5090", 256): (64, 4, 3),
}


@dataclasses.dataclass(frozen=True)
class Config:
    block_n: int
    num_warps: int
    num_stages: int

    @classmethod
    def pinned(cls, gpu_class: str | None, sm: int, head_dim: int) -> Config:
        return cls(*CLASS_CONFIGS.get((gpu_class, head_dim)) or CONFIGS.get((sm // 10, head_dim), DEFAULT))


def splits_for(batch: int, kv_heads: int, len_bound: int, num_sms: int, min_tokens: int = 64) -> int:
    """KV splits from the batch size and an upper bound on lengths only: about two programs per SM."""
    want = -(-2 * num_sms // (batch * kv_heads))
    return max(1, min(want, -(-len_bound // min_tokens)))


def _operands(kv) -> tuple:
    """K, V, their q8_0 scales (else K and V again, never read) and the stored format of one layer's pages."""
    if isinstance(kv, tuple):
        values, scales = kv
        return values[:, 0], values[:, 1], scales[:, 0], scales[:, 1], Q8_0.value
    fmt = MODEL
    if kv.dtype == torch.float8_e4m3fn:
        kv, fmt = kv.view(torch.uint8), FP8                         # decoded from its bits (`_load_kv`)
    return kv[:, 0], kv[:, 1], kv[:, 0], kv[:, 1], fmt.value


def paged_attention(q: torch.Tensor, kv, table: torch.Tensor, rows: torch.Tensor,
                    lens: torch.Tensor, *, nq: int, splits: int, cfg: Config) -> torch.Tensor:
    """See the module docstring; returns [B * nq, Hq, D]."""
    t, hq, d = q.shape
    k, v, ks, vs, fmt = _operands(kv)
    hkv, page = k.shape[1], k.shape[2]
    b, g = rows.numel(), hq // hkv
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and ks.stride(-1) == 1
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)     # the kernel writes it densely
    ws_ml = torch.empty(t * hq * splits * 2 if splits > 1 else 1, dtype=torch.float32, device=q.device)
    ws_acc = torch.empty(t * hq * splits * d if splits > 1 else 1, dtype=torch.float32, device=q.device)
    _paged_attn[(b, hkv, splits)](
        q, k, v, ks, vs, table, rows, lens, out, ws_ml, ws_acc,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), k.stride(2), ks.stride(0), ks.stride(1), ks.stride(2),
        table.stride(0), 1.0 / math.sqrt(d) * math.log2(math.e),
        NQ=nq, G=g, D=d, PAGE=page, BLOCK_M=max(16, triton.next_power_of_2(nq * g)), BLOCK_N=cfg.block_n,
        SPLITS=splits, FMT=fmt, num_warps=cfg.num_warps, num_stages=cfg.num_stages)
    if splits > 1:
        _merge_splits[(t * hq,)](ws_ml, ws_acc, out, D=d, SPLITS=splits,
                                 SPLITS_P2=triton.next_power_of_2(splits), num_warps=4)
    return out


# Prefill: ragged chunk rows, each continuing its sequence's earlier context (bottom-right causal). One program
# handles ROWS query rows of one batch row times the G query heads of one kv head (BLOCK_M = ROWS * G, padded), and
# walks the KV from the start to its last row's position in one pass: no split, so it is deterministic. The grid
# comes from the host's row counts (an upper bound per batch row); query starts and lengths are read on the device.
@triton.jit
def _paged_prefill(Q, K, V, KS, VS, TABLE, ROWS, LENS, QSTART, OUT,
                   stride_qt, stride_qh, stride_ku, stride_kh, stride_kt, stride_su, stride_sh, stride_st,
                   stride_table, qk_scale,
                   G: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr, ROWS_PER_TILE: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, FMT: tl.constexpr):
    tile = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    hq = tl.num_programs(2) * G

    q0 = tl.load(QSTART + b)
    nq = tl.load(QSTART + b + 1) - q0
    n = tl.load(LENS + b)
    slot = tl.load(ROWS + b)
    first = tile * ROWS_PER_TILE
    m = tl.arange(0, BLOCK_M)
    r = first + m // G                                            # row within this batch row's chunk
    head = h * G + m % G
    row_ok = (m < ROWS_PER_TILE * G) & (r < nq)
    d = tl.arange(0, D)
    q = tl.load(Q + (q0 + r)[:, None] * stride_qt + head[:, None] * stride_qh + d[None, :],
                mask=row_ok[:, None], other=0.0)
    q, qk_scale = _compute_q(q, qk_scale, FMT)
    last_visible = n - nq + r                                     # bottom-right causal: the chunk ends at n
    hi = tl.where(first < nq, n - nq + tl.minimum(first + ROWS_PER_TILE, nq), 0)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    k_head, v_head = K + h * stride_kh, V + h * stride_kh
    ks_head, vs_head = KS + h * stride_sh, VS + h * stride_sh
    for start in range(0, hi, BLOCK_N):
        tok = start + tl.arange(0, BLOCK_N)
        tok_ok = tok < hi
        unit = tl.load(TABLE + slot * stride_table + tok // PAGE, mask=tok_ok, other=0).to(tl.int64)
        off = unit[:, None] * stride_ku + (tok % PAGE)[:, None] * stride_kt + d[None, :]
        s_off = unit[:, None] * stride_su + (tok % PAGE)[:, None] * stride_st + tl.arange(0, D // Q8_BLOCK)[None, :]
        k = _load_kv(k_head, ks_head, off, s_off, tok_ok, FMT, BLOCK_N, D, q.dtype)
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        visible = (tok[None, :] <= last_visible[:, None]) & tok_ok[None, :]
        qk = tl.where(visible, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp2(qk - m_safe[:, None])
        alpha = tl.exp2(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = _load_kv(v_head, vs_head, off, s_off, tok_ok, FMT, BLOCK_N, D, q.dtype)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    out = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]          # length-0 (padding) rows give zeros
    tl.store(OUT + ((q0 + r) * hq + head)[:, None] * D + d[None, :], out.to(OUT.dtype.element_ty),
             mask=row_ok[:, None])


# Prefill launch configs (block_m, block_n, num_warps, num_stages) per head_dim, from a sweep on sm_86 (not re-tuned
# per class).
PREFILL_CONFIGS = {64: (64, 64, 4, 2), 128: (64, 64, 4, 2), 256: (64, 64, 8, 2)}


def paged_prefill(q: torch.Tensor, kv, table: torch.Tensor, rows: torch.Tensor, lens: torch.Tensor,
                  query_start: torch.Tensor, *, max_rows: int) -> torch.Tensor:
    """Rows query_start[b]:query_start[b+1] of q [T, Hq, D] are the last ones of batch row b's lens[b] tokens;
    `max_rows` bounds the rows of any batch row. Returns [T, Hq, D]."""
    t, hq, d = q.shape
    k, v, ks, vs, fmt = _operands(kv)
    hkv, page = k.shape[1], k.shape[2]
    g = hq // hkv
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and ks.stride(-1) == 1
    block_m, block_n, num_warps, num_stages = PREFILL_CONFIGS[d]
    block_m = max(block_m, triton.next_power_of_2(g))
    per_tile = block_m // g
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    _paged_prefill[(max(1, -(-max_rows // per_tile)), rows.numel(), hkv)](
        q, k, v, ks, vs, table, rows, lens, query_start, out,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), k.stride(2), ks.stride(0), ks.stride(1), ks.stride(2),
        table.stride(0), 1.0 / math.sqrt(d) * math.log2(math.e),
        G=g, D=d, PAGE=page, ROWS_PER_TILE=per_tile, BLOCK_M=block_m, BLOCK_N=block_n, FMT=fmt,
        num_warps=num_warps, num_stages=num_stages)
    return out


@triton.jit
def _quantize_q8_0(X, VALUES, SCALES, NB: tl.constexpr):
    """One row of NB · 32 values as GGUF q8_0, bitwise as std.mixers.gqa.reference.quantize_q8_0: d = amax / 127 and
    its inverse in fp32, q = floor(|x · (1/d)| + 1/2) with x's sign, d stored fp16 — every step rounded on its own."""
    i = tl.arange(0, NB)[:, None] * Q8_BLOCK + tl.arange(0, Q8_BLOCK)[None, :]
    x = tl.load(X + i).to(tl.float32)
    amax = tl.max(tl.abs(x), 1)
    d = tl.math.div_rn(amax, tl.full(amax.shape, 127.0, tl.float32))
    inv = tl.where(d == 0.0, 0.0, tl.math.div_rn(tl.full(d.shape, 1.0, tl.float32), d))
    y = libdevice.mul_rn(x, inv[:, None])          # never contracted into an fma with the + 0.5: ties must stay
    r = tl.minimum(tl.floor(libdevice.add_rn(tl.abs(y), 0.5)), 127.0)
    tl.store(VALUES + i, tl.where(y < 0.0, -r, r).to(tl.int8))
    tl.store(SCALES + tl.arange(0, NB), d.to(tl.float16))


@triton.jit
def _write_q8_0(K, V, VALUES, SCALES, UNIT, OFF,
                stride_kt, stride_kh, stride_vt, stride_vh,
                stride_xu, stride_xi, stride_xh, stride_xt, stride_su, stride_si, stride_sh, stride_st,
                NB: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    unit = tl.load(UNIT + t).to(tl.int64)
    off = tl.load(OFF + t)
    values = VALUES + unit * stride_xu + h * stride_xh + off * stride_xt
    scales = SCALES + unit * stride_su + h * stride_sh + off * stride_st
    _quantize_q8_0(K + t * stride_kt + h * stride_kh, values, scales, NB)
    _quantize_q8_0(V + t * stride_vt + h * stride_vh, values + stride_xi, scales + stride_si, NB)


def write_q8_0(k: torch.Tensor, v: torch.Tensor, kv: tuple[torch.Tensor, torch.Tensor], unit: torch.Tensor,
               off: torch.Tensor) -> None:
    """k, v [T, Hkv, D] into q8_0 pages (values, scales) at (unit[t], off[t]): one program per token and kv head."""
    (values, scales), (t, hkv, d) = kv, k.shape
    assert k.stride(-1) == 1 and v.stride(-1) == 1 and values.stride(-1) == 1 and scales.stride(-1) == 1
    if t == 0:
        return
    _write_q8_0[(t, hkv)](k, v, values, scales, unit, off, k.stride(0), k.stride(1), v.stride(0), v.stride(1),
                          *values.stride()[:4], *scales.stride()[:4], NB=d // Q8_BLOCK.value, num_warps=1)
