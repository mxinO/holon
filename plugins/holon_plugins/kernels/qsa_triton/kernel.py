"""Triton kernels for QSA (`qsa_index@1`, `qsa_select@1`, `qsa_attend@1`; std.mixers.qsa): one launch for the index,
two for the selection, one for the attention (two when its shape splits the tokens), the output gate included.

`_attend` and `_merge` adapted from FreeToken 21d9578 (python/freetoken/kernel/triton/qsa/attend.py, itself adapted
from vLLM's qwen4_exp QSA kernel), Apache-2.0. They follow that kernel's structure: one program per (query, kv head,
split of the selected tokens), the query heads of the group stacked as a tile, the tokens gathered through the page
table tile by tile with an online softmax, the splits (by shape only: many for decode, none for prefill) merged by
their log-sum-exp in a second launch. (Merging in the last split to finish saves that launch but cost 24 µs of one
program's serial reads at decode on sm_86: measured, rejected.) The selection is in blocks and the KV cache in gqa's HND
layouts, q8_0 (decoded to fp16, as triton_attn does) and fp8 dequantized as they are loaded; the output gate is applied
where the output is stored.

The index and the selection are ours. `_index` stores a token's raw key and pools the block it completes. `_score`:
one program per (query, tile of 64 blocks), the same for every shape (shape invariance), each key chunk loaded once
for four indexer heads and its products summed within a thread (IEEE float32 products, no TF32). `_topk`: the k-th
largest score found exactly from its float bits, a bit a round with an early exit, over keys held in registers up to
8,192 blocks; then one scan writes the chosen blocks in index order, which gives D2's lower-block ties exactly.
Correct and graph-safe; tuned further in FN's step 7.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

TOPK_CHUNK = 8192                  # a row of up to 8,192 blocks (32K tokens) selects from registers
SCORE_BLOCKS, SCORE_CHUNK = 64, 8  # fixed for every shape (shape invariance); one warp per program


@triton.jit
def _index(k_ptr, raw_ptr, pooled_ptr, table_ptr, seq_ptr, pos_ptr, w_ptr, cs_ptr, eps, stride_k, stride_table,
           PAGE: tl.constexpr, D: tl.constexpr, RATIO: tl.constexpr, ROT: tl.constexpr):
    """One token: store its raw key; when it completes a block, pool the block: mean (f32, rounded to raw's dtype),
    norm, rope. The block's keys are what the cache holds after this call: a position this call stores is read from
    k in raw's dtype (the value it stores; its row is t − (p − position): each sequence's tokens are consecutive rows
    in position order, the contract's precondition), any other from `raw`."""
    t = tl.program_id(0).to(tl.int64)
    s = tl.load(seq_ptr + t)
    p = tl.load(pos_ptr + t)
    d = tl.arange(0, D)
    if s >= 0:
        unit = tl.load(table_ptr + s.to(tl.int64) * stride_table + p // PAGE).to(tl.int64)
        tl.store(raw_ptr + (unit * PAGE + p % PAGE) * D + d, tl.load(k_ptr + t * stride_k + d))
        if p % RATIO == RATIO - 1:
            start = p - RATIO + 1                   # page % RATIO == 0: the block is in one page
            unit = tl.load(table_ptr + s.to(tl.int64) * stride_table + start // PAGE).to(tl.int64)
            base = raw_ptr + (unit * PAGE + start % PAGE) * D
            half: tl.constexpr = ROT // 2
            partner = tl.where(d < half, d + half, tl.where(d < ROT, d - half, d))
            x = tl.zeros((D,), tl.float32)
            y = tl.zeros((D,), tl.float32)
            for i in tl.static_range(RATIO):
                j = tl.maximum(t - (RATIO - 1 - i), 0)
                mine = (t >= RATIO - 1 - i) & (tl.load(seq_ptr + j) == s) & (tl.load(pos_ptr + j) == start + i)
                kx = tl.load(k_ptr + j * stride_k + d, mask=(d < D) & mine, other=0.0).to(raw_ptr.dtype.element_ty)
                ky = tl.load(k_ptr + j * stride_k + partner, mask=(d < D) & mine,
                             other=0.0).to(raw_ptr.dtype.element_ty)            # as stored: in raw's dtype
                rx = tl.load(base + i * D + d, mask=(d < D) & ~mine, other=0.0)
                ry = tl.load(base + i * D + partner, mask=(d < D) & ~mine, other=0.0)
                x += tl.where(mine, kx, rx).to(tl.float32)
                y += tl.where(mine, ky, ry).to(tl.float32)
            x = (x / RATIO).to(raw_ptr.dtype.element_ty).to(tl.float32)
            y = (y / RATIO).to(raw_ptr.dtype.element_ty).to(tl.float32)
            rstd = tl.rsqrt(tl.sum(x * x) / D + eps)
            x = x * rstd * (1.0 + tl.load(w_ptr + d).to(tl.float32))
            y = y * rstd * (1.0 + tl.load(w_ptr + partner).to(tl.float32))
            c = tl.where(d < half, d, d - half)
            cos = tl.load(cs_ptr + start * ROT + c, mask=d < ROT, other=1.0)
            sin = tl.load(cs_ptr + start * ROT + half + c, mask=d < ROT, other=0.0)
            out = tl.where(d < half, x * cos - y * sin, tl.where(d < ROT, x * cos + y * sin, x))
            tl.store(pooled_ptr + (unit * (PAGE // RATIO) + (start % PAGE) // RATIO) * D + d, out)


@triton.jit
def _relu_dots(rowp, q, h0, ok, H: tl.constexpr, D: tl.constexpr, DC: tl.constexpr):
    """Σ relu(⟨k_b, q_h⟩) over heads h0 … h0 + 3 (< H) for the tile's blocks (rows at `rowp`): each key chunk loaded
    once for the four, DC dims at a time so the products sum within a thread."""
    c = tl.arange(0, DC)
    a0 = tl.zeros(ok.shape, tl.float32)
    a1 = tl.zeros(ok.shape, tl.float32)
    a2 = tl.zeros(ok.shape, tl.float32)
    a3 = tl.zeros(ok.shape, tl.float32)
    for d0 in tl.static_range(0, D, DC):
        k = tl.load(rowp[:, None] + d0 + c[None, :], mask=ok[:, None], other=0.0)
        a0 += tl.sum(k * tl.load(q + h0 * D + d0 + c).to(tl.float32)[None, :], 1)
        a1 += tl.sum(k * tl.load(q + (h0 + 1) * D + d0 + c, mask=h0 + 1 < H, other=0.0).to(tl.float32)[None, :], 1)
        a2 += tl.sum(k * tl.load(q + (h0 + 2) * D + d0 + c, mask=h0 + 2 < H, other=0.0).to(tl.float32)[None, :], 1)
        a3 += tl.sum(k * tl.load(q + (h0 + 3) * D + d0 + c, mask=h0 + 3 < H, other=0.0).to(tl.float32)[None, :], 1)
    return tl.maximum(a0, 0.0) + tl.maximum(a1, 0.0) + tl.maximum(a2, 0.0) + tl.maximum(a3, 0.0)


@triton.jit
def _score(q_ptr, pooled_ptr, table_ptr, seq_ptr, pos_ptr, ws_ptr, stride_q, stride_table, stride_ws, scale,
           H: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr, RATIO: tl.constexpr, BB: tl.constexpr,
           DC: tl.constexpr):
    """Scores of BB of a query's complete blocks: Σ_h relu(⟨q_h, k_b⟩) / √D, float32 products and sums."""
    t = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1) * BB + tl.arange(0, BB)
    s = tl.load(seq_ptr + t)
    n = (tl.load(pos_ptr + t) + 1) // RATIO
    ok = (b < n) & (s >= 0)
    if tl.program_id(1) * BB < n:
        tok = b.to(tl.int64) * RATIO
        unit = tl.load(table_ptr + tl.maximum(s, 0).to(tl.int64) * stride_table + tok // PAGE, mask=ok,
                       other=0).to(tl.int64)
        rowp = pooled_ptr + (unit * (PAGE // RATIO) + (tok % PAGE) // RATIO) * D
        score = tl.zeros((BB,), tl.float32)
        for h0 in tl.static_range(0, H, 4):
            score += _relu_dots(rowp, q_ptr + t * stride_q, h0, ok, H, D, DC)
        tl.store(ws_ptr + t * stride_ws + b, score * scale, mask=ok)


@triton.jit
def _keys(base, i, n):
    """Scores as int32 keys: non-negative floats order as their bits; past n, −1 (below every key)."""
    return tl.load(base + i, mask=i < n, other=-1.0).to(tl.int32, bitcast=True)


@triton.jit
def _emit(out, key, i, kth, need, above, equal):
    """Write a tile's chosen blocks at their ranks: every key above the k-th, and the keys equal to it while fewer
    than `need` came before. One scan carries both counts (above in the low 16 bits, equal in the high); `above` and
    `equal` are the counts before the tile. Returns them after it."""
    greater = (key > kth).to(tl.int32)
    eq = (key == kth).to(tl.int32)
    packed = greater | (eq << 16)
    before = tl.cumsum(packed, 0) - packed
    eq_before = equal + (before >> 16)
    pick = (greater == 1) | ((eq == 1) & (eq_before < need))
    tl.store(out + above + (before & 0xFFFF) + tl.minimum(eq_before, need), i, mask=pick)
    total = tl.sum(packed, 0)
    return above + (total & 0xFFFF), equal + (total >> 16)


@triton.jit
def _topk(ws_ptr, seq_ptr, pos_ptr, out_ptr, stride_ws, stride_out, RATIO: tl.constexpr, TOP: tl.constexpr,
          CHUNK: tl.constexpr, PAD: tl.constexpr):
    """A query's top min(TOP, n) blocks, ties to the lower block, ascending, then its open block. The k-th largest
    key is found exactly, a bit a round from the top (the largest v with count(key ≥ v) ≥ k), over keys held in
    registers when the row fits CHUNK (≤ 65,535), else re-read per round; then one pass writes the chosen blocks in
    index order."""
    row = tl.program_id(0).to(tl.int64)
    out = out_ptr + row * stride_out
    pad = tl.arange(0, PAD)
    if tl.load(seq_ptr + row) < 0:                  # a padding token: all −1
        tl.store(out + pad, -1, mask=pad <= TOP)
        return
    p = tl.load(pos_ptr + row)
    n = ((p + 1) // RATIO).to(tl.int32)
    k = tl.minimum(n, TOP)
    tl.store(out + pad, -1, mask=(pad >= k + ((p + 1) % RATIO != 0)) & (pad <= TOP))   # past what is written below
    base = ws_ptr + row * stride_ws
    i = tl.arange(0, CHUNK)
    if n <= TOP:
        for c0 in range(0, n, CHUNK):
            tl.store(out + c0 + i, c0 + i, mask=c0 + i < n)
    elif n <= CHUNK:
        key = _keys(base, i, n)
        kth = n * 0
        settled = n * 0
        for b in tl.static_range(31):
            if settled == 0:                        # until exactly k keys are at or above kth
                count = tl.sum((key >= kth + (1 << (30 - b))).to(tl.int32), 0)
                if count >= k:
                    kth += 1 << (30 - b)
                    settled = (count == k).to(tl.int32)
        need = k                                    # settled: every key at or above kth, exactly k of them
        if settled == 0:
            need = k - tl.sum((key > kth).to(tl.int32), 0)
        _emit(out, key, i, kth, need, 0, 0)
    else:
        kth = n * 0
        for b in tl.static_range(31):
            count = n * 0
            for c0 in range(0, n, CHUNK):
                count += tl.sum((_keys(base, c0 + i, n) >= kth + (1 << (30 - b))).to(tl.int32), 0)
            kth += (count >= k).to(tl.int32) << (30 - b)
        greater = n * 0
        for c0 in range(0, n, CHUNK):
            greater += tl.sum((_keys(base, c0 + i, n) > kth).to(tl.int32), 0)
        above = n * 0
        equal = n * 0
        for c0 in range(0, n, CHUNK):
            above, equal = _emit(out, _keys(base, c0 + i, n), c0 + i, kth, k - greater, above, equal)
    if (p + 1) % RATIO:
        tl.store(out + k, n)


@triton.jit
def _attend(q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, blocks_ptr, table_ptr, seq_ptr, pos_ptr, gate_ptr, out_ptr, part_ptr,
            lse_ptr, stride_q, stride_q_head, stride_gate, stride_gate_head, stride_blocks, stride_table, stride_out,
            stride_out_head, unit_stride,
            kv_head_stride, scale_unit_stride, scale_kv_head_stride, scale_log2, rows,
            HKV: tl.constexpr, G: tl.constexpr, GP: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
            RATIO: tl.constexpr, WIDTH: tl.constexpr, BN: tl.constexpr, SPLITS: tl.constexpr, Q8: tl.constexpr,
            GATED: tl.constexpr):
    """One query, one kv head, one split of the selected tokens: its G query heads, an online softmax. One split
    writes the output (times σ(gate) with GATED); more write each split's normalized partial and log2-sum-exp for
    `_merge`."""
    t = tl.program_id(0).to(tl.int64)
    kvh = tl.program_id(1)
    split = tl.program_id(2)
    h = tl.arange(0, GP)
    d = tl.arange(0, D)
    s = tl.load(seq_ptr + t)
    heads = kvh * G + h
    if (s < 0) & (SPLITS == 1):
        out = out_ptr + t * stride_out + heads[:, None] * stride_out_head + d[None, :]
        tl.store(out, tl.zeros((GP, D), out_ptr.dtype.element_ty), mask=(h < G)[:, None])
        return
    p = tl.load(pos_ptr + t)
    q = tl.load(q_ptr + t * stride_q + heads[:, None] * stride_q_head + d[None, :], mask=(h < G)[:, None],
                other=0.0)
    # As triton_attn: q8_0 decodes to fp16 (values × scale, rounded once), whose 11-bit significand holds int8 × scale
    # far closer than bf16's 8; q joins it there, pre-scaled by the score scale, so |K|, |V| must stay below 65504.
    if Q8:
        q = (q.to(tl.float32) * scale_log2).to(tl.float16)
        scale_log2 = 1.0
    m = tl.full((GP,), -1.0e30, tl.float32)
    den = tl.zeros((GP,), tl.float32)
    acc = tl.zeros((GP, D), tl.float32)
    j = tl.arange(0, BN)
    per: tl.constexpr = triton.cdiv(triton.cdiv(WIDTH * RATIO, BN), SPLITS) * BN
    for tile in range(split * per, tl.minimum(split * per + per, WIDTH * RATIO), BN):
        # the selection is a prefix of the row (then −1): a tile starting at −1 has nothing
        if (s >= 0) & (tl.load(blocks_ptr + t * stride_blocks + tile // RATIO) >= 0):
            blk = tl.load(blocks_ptr + t * stride_blocks + (tile + j) // RATIO, mask=tile + j < WIDTH * RATIO,
                          other=-1)
            tok = blk.to(tl.int64) * RATIO + (tile + j) % RATIO
            ok = (blk >= 0) & (tok <= p)
            unit = tl.load(table_ptr + s.to(tl.int64) * stride_table + tok // PAGE, mask=ok, other=0).to(tl.int64)
            row = unit * unit_stride + kvh * kv_head_stride + (tok % PAGE) * D
            k = tl.load(k_ptr + row[None, :] + d[:, None], mask=ok[None, :], other=0.0)
            v = tl.load(v_ptr + row[:, None] + d[None, :], mask=ok[:, None], other=0.0)
            if Q8:
                srow = unit * scale_unit_stride + kvh * scale_kv_head_stride + (tok % PAGE) * (D // 32)
                k = k.to(tl.float32) * tl.load(ks_ptr + srow[None, :] + (d // 32)[:, None], mask=ok[None, :],
                                               other=0.0).to(tl.float32)
                v = v.to(tl.float32) * tl.load(vs_ptr + srow[:, None] + (d // 32)[None, :], mask=ok[:, None],
                                               other=0.0).to(tl.float32)
            scores = tl.dot(q, k.to(q.dtype)) * scale_log2
            scores = tl.where(ok[None, :], scores, -1.0e30)
            m_new = tl.maximum(m, tl.max(scores, 1))
            alpha = tl.math.exp2(m - m_new)
            prob = tl.where(ok[None, :], tl.math.exp2(scores - m_new[:, None]), 0.0)
            acc = tl.dot(prob.to(q.dtype), v.to(q.dtype), acc * alpha[:, None])
            den = den * alpha + tl.sum(prob, 1)
            m = m_new
    o = acc / tl.where(den > 0, den, 1.0)[:, None]
    if SPLITS == 1:
        if GATED:
            o *= tl.sigmoid(tl.load(gate_ptr + t * stride_gate + heads[:, None] * stride_gate_head + d[None, :],
                                    mask=(h < G)[:, None], other=0.0).to(tl.float32))
        out = out_ptr + t * stride_out + heads[:, None] * stride_out_head + d[None, :]
        tl.store(out, o.to(out_ptr.dtype.element_ty), mask=(h < G)[:, None])
    else:
        at = (split * rows + t) * HKV * G + heads                     # partials [SPLITS, T, Hq]
        tl.store(part_ptr + at[:, None] * D + d[None, :], o, mask=(h < G)[:, None])
        tl.store(lse_ptr + at, tl.where(den > 0, m + tl.math.log2(den), float("-inf")), mask=h < G)


@triton.jit
def _merge(part_ptr, lse_ptr, gate_ptr, out_ptr, stride_gate, stride_gate_head, stride_out, stride_out_head, rows,
           HQ: tl.constexpr, D: tl.constexpr, SPLITS: tl.constexpr, GATED: tl.constexpr):
    """The splits of one query head, weighted by their sums (a padding token's, all empty: 0), times σ(gate) with
    GATED."""
    t = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    sp = tl.arange(0, SPLITS)
    d = tl.arange(0, D)
    at = (sp * rows + t) * HQ + head
    lse = tl.load(lse_ptr + at)
    top = tl.max(lse, 0)
    w = tl.where(lse > float("-inf"), tl.math.exp2(lse - tl.where(top > float("-inf"), top, 0.0)), 0.0)
    o = tl.sum(tl.load(part_ptr + at[:, None] * D + d[None, :]) * w[:, None], 0)
    total = tl.sum(w, 0)
    o = o / tl.where(total > 0, total, 1.0)
    if GATED:                                   # a padding row stays 0 whatever its gate holds (NaN included)
        g = tl.load(gate_ptr + t * stride_gate + head * stride_gate_head + d).to(tl.float32)
        o = tl.where(total > 0, o * tl.sigmoid(g), 0.0)
    tl.store(out_ptr + t * stride_out + head * stride_out_head + d, o.to(out_ptr.dtype.element_ty))


def qsa_index(k, raw, pooled, table, seq, positions, weight, cos_sin, ratio: int, eps: float) -> None:
    t, d = k.shape
    if t:
        _index[(t,)](k, raw, pooled, table, seq, positions, weight, cos_sin, eps, k.stride(0), table.stride(0),
                     raw.shape[1], d, ratio, cos_sin.shape[1])


def scores(q, pooled, table, seq, positions, ratio: int, ws: torch.Tensor) -> None:
    """Every query's block scores into `ws` [T, blocks]: one program per (query, tile of SCORE_BLOCKS), the same tile,
    chunk and warps for every shape, so a query's bits never depend on the call's other rows or its row count."""
    t, heads, d = q.shape
    page = pooled.shape[1] * ratio
    _score[(t, triton.cdiv(ws.shape[1], SCORE_BLOCKS))](
        q, pooled, table, seq, positions, ws, q.stride(0), table.stride(0), ws.stride(0), 1.0 / math.sqrt(d), heads,
        d, page, ratio, SCORE_BLOCKS, SCORE_CHUNK, num_warps=1)


def qsa_select(q, pooled, table, seq, positions, ratio: int, budget: int, sub_batch: int) -> torch.Tensor:
    t, heads, d = q.shape
    top = budget // ratio
    out = torch.empty(t, top + 1, dtype=torch.int32, device=q.device)
    blocks = table.shape[1] * pooled.shape[1]                  # the most complete blocks a query can see
    q = q.contiguous()
    for a in range(0, t, sub_batch):
        rows = min(sub_batch, t - a)
        ws = torch.empty(rows, blocks, dtype=torch.float32, device=q.device)
        scores(q[a:a + rows], pooled, table, seq[a:], positions[a:], ratio, ws)
        _topk[(rows,)](ws, seq[a:], positions[a:], out[a:], ws.stride(0), out.stride(0), ratio, top, TOPK_CHUNK,
                       triton.next_power_of_2(top + 1), num_warps=8)
    return out


def qsa_attend(q, kv, blocks, table, seq, positions, gate, ratio: int) -> torch.Tensor:
    t, hq, d = q.shape
    out = torch.empty_like(q)
    if t == 0:
        return out
    q8 = isinstance(kv, tuple)
    values, scales = kv if q8 else (kv, None)
    units, _, hkv, page, _ = values.shape
    g = hq // hkv
    k, v = values[:, 0], values[:, 1]
    ks, vs = (scales[:, 0], scales[:, 1]) if q8 else (k, v)
    gated = gate is not None
    if gated and gate.stride(2) != 1:
        raise ValueError("qsa_attend: the gate's last dim must be contiguous")
    gate = gate if gated else q
    programs = t * hkv          # split the tokens across programs while there are few (decode, verify): by shape only
    tokens, splits = ((32, 32) if programs <= 8 else (32, 8) if programs <= 32 else (64, 4) if programs <= 256
                      else (64, 1))
    part = torch.empty(splits, t, hq, d, dtype=torch.float32, device=q.device) if splits > 1 else out
    lse = torch.empty(splits, t, hq, dtype=torch.float32, device=q.device) if splits > 1 else out
    _attend[(t, hkv, splits)](
        q, k, v, ks, vs, blocks, table, seq, positions, gate, out, part, lse, q.stride(0), q.stride(1),
        gate.stride(0), gate.stride(1), blocks.stride(0), table.stride(0), out.stride(0), out.stride(1),
        values.stride(0), values.stride(2), scales.stride(0) if q8 else 0, scales.stride(2) if q8 else 0,
        1.4426950408889634 / math.sqrt(d), t, hkv, g, max(16, triton.next_power_of_2(g)), d, page, ratio,
        blocks.shape[1], tokens, splits, q8, gated, num_warps=4)
    if splits > 1:
        _merge[(t, hq)](part, lse, gate, out, gate.stride(0), gate.stride(1), out.stride(0), out.stride(1), t, hq, d,
                        splits, gated)
    return out
