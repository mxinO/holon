"""Triton kernels for GLM-5.3-Flash's DSA (`dsa_index@1`, `mla_attend@1`; std.mixers.qsa; docs/v1-glm.md D5, D12).
Ours, written for this plugin (not adapted).

- `_index`: one program per token. It stores the token's key and gate, and when the token completes a block it pools
  the block: per channel, the softmax over the block's tokens of gate + APE, then the keys' weighted sum, in float32.
  The block's keys and gates are what the cache holds after the call (as `qsa_index@1`'s): a position this call
  stores is read from the call's rows (in the cache's dtype), any other from the cache.
- `_attend`: one program per (query, group of 16 heads), the same for every shape, so a row's bits never depend on
  the call's other rows or their number (the dense path is this kernel over every block). It walks the query's
  selected tokens (its blocks' tokens up to its position) a tile at a time through the page table, each tile's
  latent rows loaded once for both products, with an online softmax in float32. Both products run on tensor cores
  against the bfloat16 latent, q̃ and the softmax weights each split into two bfloat16 parts (hi + lo: 16 bits;
  exact products, f32 sums): one precision for every row, relative L2 ≈ 3e-6 against the float32 oracle (D5's
  budget 2e-4; a third part of q̃ measured no closer, one part 9e-4).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

HEADS = 16          # a program's query heads: tl.dot's smallest M
TOKENS = 32         # selected tokens a tile


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
def _attend(q_ptr, kv_ptr, blocks_ptr, table_ptr, seq_ptr, pos_ptr, out_ptr, stride_q, stride_qh, stride_blocks,
            stride_table, stride_out, stride_oh, kv_unit, scale, H: tl.constexpr, DC_: tl.constexpr,
            PAGE: tl.constexpr, RATIO: tl.constexpr, WIDTH: tl.constexpr, WPAD: tl.constexpr, HB: tl.constexpr,
            BN: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1) * HB + tl.arange(0, HB)
    hok = h < H
    c = tl.arange(0, DC_)
    out = out_ptr + t * stride_out + h[:, None] * stride_oh + c[None, :]
    s = tl.load(seq_ptr + t)
    if s < 0:                                       # a padding token: 0
        tl.store(out, tl.zeros((HB, DC_), tl.float32), mask=hok[:, None])
        return
    p = tl.load(pos_ptr + t)
    w = tl.arange(0, WPAD)
    count = tl.sum((tl.load(blocks_ptr + t * stride_blocks + w, mask=w < WIDTH, other=-1) >= 0).to(tl.int32), 0)
    trow = table_ptr + s.to(tl.int64) * stride_table
    q_hi, q_lo = _split2(tl.load(q_ptr + t * stride_q + h[:, None] * stride_qh + c[None, :], mask=hok[:, None],
                                 other=0.0).to(tl.float32))
    m = tl.full((HB,), float("-inf"), tl.float32)
    lsum = tl.zeros((HB,), tl.float32)
    acc = tl.zeros((HB, DC_), tl.float32)
    j = tl.arange(0, BN)
    for j0 in range(0, count * RATIO, BN):          # the selected blocks lead the row (−1 after them): their tokens
        idx = j0 + j
        block = tl.load(blocks_ptr + t * stride_blocks + idx // RATIO, mask=idx < count * RATIO, other=-1)
        tok = block.to(tl.int64) * RATIO + idx % RATIO
        ok = (block >= 0) & (tok <= p)
        unit = tl.load(trow + tok // PAGE, mask=ok, other=0).to(tl.int64)
        cv = tl.load(kv_ptr + unit[:, None] * kv_unit + (tok % PAGE)[:, None] * DC_ + c[None, :], mask=ok[:, None],
                     other=0.0)                                                                  # [BN, DC] bf16
        ct = tl.trans(cv)
        scores = tl.dot(q_hi, ct) + tl.dot(q_lo, ct)
        scores = tl.where(ok[None, :], scores * scale, float("-inf"))
        m_new = tl.maximum(m, tl.max(scores, 1))
        alpha = tl.exp(m - m_new)
        pw = tl.exp(scores - m_new[:, None])
        lsum = lsum * alpha + tl.sum(pw, 1)
        p_hi, p_lo = _split2(pw)
        acc = acc * alpha[:, None] + tl.dot(p_hi, cv) + tl.dot(p_lo, cv)
        m = m_new
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
        width = blocks.shape[1]
        _attend[(t, triton.cdiv(heads, HEADS))](
            q, kv, blocks, table, seq, positions, out, q.stride(0), q.stride(1), blocks.stride(0), table.stride(0),
            out.stride(0), out.stride(1), kv.stride(0), scale, heads, dc, kv.shape[1], ratio, width,
            triton.next_power_of_2(width), HEADS, TOKENS, num_warps=4, num_stages=2)
    return out
