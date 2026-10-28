"""Triton kernels for GLM-5.3-Flash's DSA (`dsa_index@1`, `mla_attend@1`; std.mixers.qsa; docs/dev/v1-glm.md D5, D12).
Ours, written for this plugin (not adapted).

- `_index`: one program per token. It stores the token's key and gate, and when the token completes a block it pools
  the block: per channel, the softmax over the block's tokens of gate + APE, then the keys' weighted sum, in float32.
  The block's keys and gates are what the cache holds after the call (as `qsa_index@1`'s): a position this call
  stores is read from the call's rows (in the cache's dtype), any other from the cache.
- `_attend`: a row's selected tokens (its blocks' tokens up to its position) in chunks of CHUNK, each chunk an online
  softmax in float32 from its first tile, the chunks combined in order: one program per (query, group of 16 heads)
  walking every chunk, or, for a call of few rows (decode, verify), a program per chunk and `_merge` combining them —
  the same operations either way (one tile step, one combine, one warp count), so a row's bits never depend on the
  call's other rows, their number or the path (the dense path is this kernel over every block). A tile's latent
  rows are loaded once through the page table for both products, on tensor cores against the bfloat16 latent: q̃
  as one bfloat16 part, the softmax weights as two (hi + lo: exact products, f32 sums); relative L2 ≈ 1e-3 against
  the float32 oracle (two parts of q̃: 3e-6, at 4 dots a tile; docs/dev/v1-glm.md §13, step 7: the real model's KL).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

HEADS = 16          # a program's query heads: tl.dot's smallest M
TOKENS = 32         # selected tokens a tile
CHUNK = 256         # selected tokens a chunk (its online softmax from scratch; the chunks combined in order)
WARPS = 8           # every `_attend` and `_merge` program's (a row's bits depend on the warps: the reductions' order)
SPLIT_BELOW = 64    # a call of fewer (query, head group) programs than this: a program a chunk, then `_merge`


@triton.jit
def _index(k_ptr, g_ptr, raw_ptr, gates_ptr, pooled_ptr, ape_ptr, table_ptr, seq_ptr, pos_ptr, stride_k, stride_g,
           stride_table, raw_unit, gates_unit, pooled_unit, PAGE: tl.constexpr, D: tl.constexpr,
           RATIO: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    s = tl.load(seq_ptr + t)
    p = tl.load(pos_ptr + t)
    d = tl.arange(0, D)
    if s >= 0:
        unit = tl.load(table_ptr + s.to(tl.int64) * stride_table + p // PAGE).to(tl.int64)
        off = (p % PAGE) * D
        tl.store(raw_ptr + unit * raw_unit + off + d, tl.load(k_ptr + t * stride_k + d))
        tl.store(gates_ptr + unit * gates_unit + off + d, tl.load(g_ptr + t * stride_g + d))
        if p % RATIO == RATIO - 1:
            start = p - RATIO + 1                   # page % RATIO == 0: the block is in one page
            unit = tl.load(table_ptr + s.to(tl.int64) * stride_table + start // PAGE).to(tl.int64)
            keys = raw_ptr + unit * raw_unit + (start % PAGE) * D
            gates = gates_ptr + unit * gates_unit + (start % PAGE) * D
            m = tl.full((D,), float("-inf"), tl.float32)
            for i in tl.static_range(RATIO):        # the logits' maximum per channel
                j = tl.maximum(t - (RATIO - 1 - i), 0)
                mine = (t >= RATIO - 1 - i) & (tl.load(seq_ptr + j) == s) & (tl.load(pos_ptr + j) == start + i)
                gx = tl.where(mine, tl.load(g_ptr + j * stride_g + d).to(gates_ptr.dtype.element_ty),
                              tl.load(gates + i * D + d, mask=~mine, other=0.0))
                m = tl.maximum(m, gx.to(tl.float32) + tl.load(ape_ptr + i * D + d))
            total = tl.zeros((D,), tl.float32)
            acc = tl.zeros((D,), tl.float32)
            for i in tl.static_range(RATIO):
                j = tl.maximum(t - (RATIO - 1 - i), 0)
                mine = (t >= RATIO - 1 - i) & (tl.load(seq_ptr + j) == s) & (tl.load(pos_ptr + j) == start + i)
                gx = tl.where(mine, tl.load(g_ptr + j * stride_g + d).to(gates_ptr.dtype.element_ty),
                              tl.load(gates + i * D + d, mask=~mine, other=0.0))
                kx = tl.where(mine, tl.load(k_ptr + j * stride_k + d).to(raw_ptr.dtype.element_ty),
                              tl.load(keys + i * D + d, mask=~mine, other=0.0))
                e = tl.exp(gx.to(tl.float32) + tl.load(ape_ptr + i * D + d) - m)
                total += e
                acc += e * kx.to(tl.float32)
            tl.store(pooled_ptr + unit * pooled_unit + ((start % PAGE) // RATIO) * D + d, acc / total)


@triton.jit
def _split2(x):
    """f32 x as two bfloat16 parts whose sum carries 16 of its bits."""
    hi = x.to(tl.bfloat16)
    return hi, (x - hi.to(tl.float32)).to(tl.bfloat16)


@triton.jit
def _tile(q, kv_ptr, blocks_ptr, trow, t, p, count, j0, m, lsum, acc, stride_blocks, kv_unit, scale,
          DC_: tl.constexpr, PAGE: tl.constexpr, RATIO: tl.constexpr, BN: tl.constexpr):
    """The online softmax's step over the row's selected tokens j0 … j0 + BN − 1."""
    idx = j0 + tl.arange(0, BN)                     # the selected blocks lead the row (−1 after them): their tokens
    block = tl.load(blocks_ptr + t * stride_blocks + idx // RATIO, mask=idx < count * RATIO, other=-1)
    tok = block.to(tl.int64) * RATIO + idx % RATIO
    ok = (block >= 0) & (tok <= p)
    unit = tl.load(trow + tok // PAGE, mask=ok, other=0).to(tl.int64)
    cv = tl.load(kv_ptr + unit[:, None] * kv_unit + (tok % PAGE)[:, None] * DC_ + tl.arange(0, DC_)[None, :],
                 mask=ok[:, None], other=0.0)                                                    # [BN, DC] bf16
    scores = tl.where(ok[None, :], tl.dot(q, tl.trans(cv)) * scale, float("-inf"))
    m_new = tl.maximum(m, tl.max(scores, 1))
    alpha = tl.exp(m - m_new)
    pw = tl.exp(scores - m_new[:, None])
    p_hi, p_lo = _split2(pw)
    return m_new, lsum * alpha + tl.sum(pw, 1), acc * alpha[:, None] + tl.dot(p_hi, cv) + tl.dot(p_lo, cv)


@triton.jit
def _combine(m, lsum, acc, m_c, l_c, acc_c):
    """The chunks so far and the next one's (maximum, sum, unnormalized output), as one."""
    m_new = tl.maximum(m, m_c)
    a = tl.exp(m - m_new)
    b = tl.exp(m_c - m_new)
    return m_new, lsum * a + l_c * b, acc * a[:, None] + acc_c * b[:, None]


@triton.jit
def _row(seq_ptr, pos_ptr, blocks_ptr, t, stride_blocks, WIDTH: tl.constexpr, WPAD: tl.constexpr):
    """A row's sequence, position and number of selected blocks."""
    w = tl.arange(0, WPAD)
    count = tl.sum((tl.load(blocks_ptr + t * stride_blocks + w, mask=w < WIDTH, other=-1) >= 0).to(tl.int32), 0)
    return tl.load(seq_ptr + t), tl.load(pos_ptr + t), count


@triton.jit
def _attend(q_ptr, kv_ptr, blocks_ptr, table_ptr, seq_ptr, pos_ptr, out_ptr, part_ptr, stats_ptr, stride_q, stride_qh,
            stride_blocks, stride_table, stride_out, stride_oh, kv_unit, scale, H: tl.constexpr, DC_: tl.constexpr,
            PAGE: tl.constexpr, RATIO: tl.constexpr, WIDTH: tl.constexpr, WPAD: tl.constexpr, HB: tl.constexpr,
            BN: tl.constexpr, CHUNK: tl.constexpr, CHUNKS: tl.constexpr, SPLIT: tl.constexpr):
    """One program per (query, head group) over every chunk; with SPLIT one per (query, head group, chunk), storing
    the chunk's output and (maximum, sum) for `_merge`."""
    t = tl.program_id(0).to(tl.int64)
    g = tl.program_id(1)
    i = tl.arange(0, HB)
    h = g * HB + i
    hok = h < H
    c = tl.arange(0, DC_)
    s, p, count = _row(seq_ptr, pos_ptr, blocks_ptr, t, stride_blocks, WIDTH, WPAD)
    trow = table_ptr + tl.maximum(s, 0).to(tl.int64) * stride_table
    q = tl.load(q_ptr + t * stride_q + h[:, None] * stride_qh + c[None, :], mask=hok[:, None], other=0.0)
    q = q.to(tl.bfloat16)
    m = tl.full((HB,), float("-inf"), tl.float32)
    lsum = tl.zeros((HB,), tl.float32)
    acc = tl.zeros((HB, DC_), tl.float32)
    if SPLIT:
        k = tl.program_id(2)
        if (s >= 0) & (k * CHUNK < count * RATIO):
            for j0 in range(k * CHUNK, tl.minimum(k * CHUNK + CHUNK, count * RATIO), BN):
                m, lsum, acc = _tile(q, kv_ptr, blocks_ptr, trow, t, p, count, j0, m, lsum, acc, stride_blocks,
                                     kv_unit, scale, DC_, PAGE, RATIO, BN)
            at = (t * CHUNKS + k) * tl.cdiv(H, HB) + g                          # [T, CHUNKS, groups] partials
            tl.store(part_ptr + at * HB * DC_ + i[:, None] * DC_ + c[None, :], acc)
            tl.store(stats_ptr + at * 2 * HB + i, m)
            tl.store(stats_ptr + at * 2 * HB + HB + i, lsum)
        return
    out = out_ptr + t * stride_out + h[:, None] * stride_oh + c[None, :]
    if s < 0:                                       # a padding token: 0
        tl.store(out, tl.zeros((HB, DC_), tl.float32), mask=hok[:, None])
        return
    m_all = tl.full((HB,), float("-inf"), tl.float32)  # the chunks before this one (one loop over the tiles,
    l_all = tl.zeros((HB,), tl.float32)                # pipelined as one; a chunk folded in at the next one's start)
    a_all = tl.zeros((HB, DC_), tl.float32)
    for j0 in range(0, count * RATIO, BN):
        if (j0 % CHUNK == 0) & (j0 > 0):
            if j0 == CHUNK:
                m_all, l_all, a_all = m, lsum, acc
            else:
                m_all, l_all, a_all = _combine(m_all, l_all, a_all, m, lsum, acc)
            m = tl.full((HB,), float("-inf"), tl.float32)
            lsum = tl.zeros((HB,), tl.float32)
            acc = tl.zeros((HB, DC_), tl.float32)
        m, lsum, acc = _tile(q, kv_ptr, blocks_ptr, trow, t, p, count, j0, m, lsum, acc, stride_blocks, kv_unit,
                             scale, DC_, PAGE, RATIO, BN)
    if count * RATIO > CHUNK:
        m, lsum, acc = _combine(m_all, l_all, a_all, m, lsum, acc)
    tl.store(out, (acc / lsum[:, None]).to(out_ptr.dtype.element_ty), mask=hok[:, None])


@triton.jit
def _merge(blocks_ptr, seq_ptr, pos_ptr, out_ptr, part_ptr, stats_ptr, stride_blocks, stride_out, stride_oh,
           H: tl.constexpr, DC_: tl.constexpr, RATIO: tl.constexpr, WIDTH: tl.constexpr, WPAD: tl.constexpr,
           HB: tl.constexpr, CHUNK: tl.constexpr, CHUNKS: tl.constexpr):
    """A split row's chunks combined in order, as `_attend` unsplit combines them, normalized into the output."""
    t = tl.program_id(0).to(tl.int64)
    g = tl.program_id(1)
    i = tl.arange(0, HB)
    h = g * HB + i
    hok = h < H
    c = tl.arange(0, DC_)
    out = out_ptr + t * stride_out + h[:, None] * stride_oh + c[None, :]
    s, _, count = _row(seq_ptr, pos_ptr, blocks_ptr, t, stride_blocks, WIDTH, WPAD)
    if s < 0:
        tl.store(out, tl.zeros((HB, DC_), tl.float32), mask=hok[:, None])
        return
    groups = tl.cdiv(H, HB)
    at = t * CHUNKS * groups + g
    m = tl.load(stats_ptr + at * 2 * HB + i)
    lsum = tl.load(stats_ptr + at * 2 * HB + HB + i)
    acc = tl.load(part_ptr + at * HB * DC_ + i[:, None] * DC_ + c[None, :])
    for k in range(1, tl.cdiv(count * RATIO, CHUNK)):
        at = (t * CHUNKS + k) * groups + g
        m, lsum, acc = _combine(m, lsum, acc, tl.load(stats_ptr + at * 2 * HB + i),
                                tl.load(stats_ptr + at * 2 * HB + HB + i),
                                tl.load(part_ptr + at * HB * DC_ + i[:, None] * DC_ + c[None, :]))
    tl.store(out, (acc / lsum[:, None]).to(out_ptr.dtype.element_ty), mask=hok[:, None])


def _dense_rows(*views) -> None:
    for v in views:
        if v.stride(-1) != 1 or v.stride(-2) != v.shape[-1]:
            raise ValueError(f"dsa: a unit's rows must be dense, not strides {v.stride()} for {tuple(v.shape)}")


def dsa_index(k, gate, raw, gates, pooled, ape, table, seq, positions, ratio: int) -> None:
    _dense_rows(raw, gates, pooled)
    t, d = k.shape
    if t:
        _index[(t,)](k, gate, raw, gates, pooled, ape, table, seq, positions, k.stride(0), gate.stride(0),
                     table.stride(0), raw.stride(0), gates.stride(0), pooled.stride(0), raw.shape[1], d, ratio)


def mla_attend(q, kv, blocks, table, seq, positions, ratio: int, scale: float) -> torch.Tensor:
    """q [T, H, Dc] (any float dtype; its heads strided, its dims contiguous), kv [units, page, Dc] bfloat16 (units
    strided, rows dense) → [T, H, Dc] of q's dtype."""
    _dense_rows(kv)
    t, heads, dc = q.shape
    out = torch.empty_like(q)
    if t:
        if q.stride(2) != 1 or blocks.stride(1) != 1:
            raise ValueError("mla_attend: q's dims and the blocks must be contiguous")
        width, wpad = blocks.shape[1], triton.next_power_of_2(blocks.shape[1])
        groups, chunks = triton.cdiv(heads, HEADS), triton.cdiv(blocks.shape[1] * ratio, CHUNK)
        split = t * groups < SPLIT_BELOW                # by shape: the bits are the row's either way
        part = torch.empty(t * chunks * groups * HEADS * dc if split else 1, dtype=torch.float32, device=q.device)
        stats = torch.empty(t * chunks * groups * 2 * HEADS if split else 1, dtype=torch.float32, device=q.device)
        _attend[(t, groups, chunks if split else 1)](
            q, kv, blocks, table, seq, positions, out, part, stats, q.stride(0), q.stride(1), blocks.stride(0),
            table.stride(0), out.stride(0), out.stride(1), kv.stride(0), scale, heads, dc, kv.shape[1], ratio, width,
            wpad, HEADS, TOKENS, CHUNK, chunks, split, num_warps=WARPS, num_stages=2)
        if split:
            _merge[(t, groups)](blocks, seq, positions, out, part, stats, blocks.stride(0), out.stride(0),
                                out.stride(1), heads, dc, ratio, width, wpad, HEADS, CHUNK, chunks, num_warps=WARPS)
    return out
