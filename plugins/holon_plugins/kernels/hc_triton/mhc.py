"""Our kernels for GLM-5.3-Flash's manifold-constrained hyper-connections (`hc_mhc_mix@1`, `hc_mhc_writeback@1`;
std.hyper; docs/dev/v1-glm.md D3, D12), written for Holon. Every operation in float32 with IEEE products and sums.

A row's bits depend on neither T nor its neighbours (the contract): every step size takes the same arithmetic.
- `_dots`: per (tile of rows, chunk of KPP · BK columns h): all four streams' columns h of those rows, so a pending
  write-back (`_writeback_tile`) is applied in place by the program that owns column h of every stream (D3: work split
  over h, never over streams). Then the chunk's (2 + n)·n products (`tl.dot`, IEEE: FMAs whose order is set by BK and
  KPP, not by the tile's rows or warps) and its sums of squares, as partials [T, chunks, ·]. Decode's split-K over h
  is its grid's second axis (32 chunks at H = 4,096); decode takes 16-row tiles, prefill 128 (the same bits).
- `_weights`: per row, the partials summed over the chunks in a fixed order, the RMS (m = rsqrt(·) · (fn · R), within
  L0 of the oracle's fn · (R · rsqrt(·))), the affine maps and the Sinkhorn in float32 (1 softmax, then `iters` column
  and `iters` − 1 row normalisations, HF's order): pre (scratch), post and comb.
- `_collapse`: x = Σ_s pre_s · R_s, rounded once.
- `_writeback`: R'_d = Σ_s comb[s, d] · R_s + post_d · y through the same elementwise `_writeback_tile` as the mix's
  pending write-back, so the two leave the same bits.
Explicit FMAs (`tl.fma`) fix the order of each elementwise sum."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

N_STREAMS = 4                    # the streams the kernels serve (GLM's n); (2 + n)·n = 24 outputs, padded to BW
# `_dots`: BM rows × (KPP sub-chunks of BK columns h) of every stream a program (32 chunks at H = 4,096); its bits
# do not depend on BM or the warps (sequential FMAs), so decode takes (16 rows, 2 warps), prefill (128, 4) (an RTX
# A5000 sweep). `_collapse`: BMC rows × BH columns; `_writeback`: BMW rows × BH.
SMALL, BM_SMALL, BM, BK, KPP, BW, STAGES = 64, (16, 2), (128, 4), 16, 8, 32, 2
BMC, BMW, BH = 4, 4, 512
WARPS = 4


@triton.jit
def _writeback_tile(r_ptr, y_ptr, post_ptr, comb_ptr, rows, h, m, H: tl.constexpr):
    """A tile's write-back, in place: R'_d = Σ_s comb[s, d] · R_s + post_d · y over its rows [·, 1] and columns h of
    all four streams, elementwise (explicit FMAs, s in order: the same bits for any tile). Returns the new streams."""
    at = r_ptr + rows.to(tl.int64) * (4 * H) + h
    r0 = tl.load(at, mask=m, other=0.0)
    r1 = tl.load(at + H, mask=m, other=0.0)
    r2 = tl.load(at + 2 * H, mask=m, other=0.0)
    r3 = tl.load(at + 3 * H, mask=m, other=0.0)
    y = tl.load(y_ptr + rows.to(tl.int64) * H + h, mask=m, other=0.0).to(tl.float32)
    cp, pp = comb_ptr + rows * 16, post_ptr + rows * 4
    outs = ()
    for d in tl.static_range(4):
        acc = tl.load(cp + d, mask=m, other=0.0) * r0
        acc = tl.fma(tl.load(cp + 4 + d, mask=m, other=0.0), r1, acc)
        acc = tl.fma(tl.load(cp + 8 + d, mask=m, other=0.0), r2, acc)
        acc = tl.fma(tl.load(cp + 12 + d, mask=m, other=0.0), r3, acc)
        out = tl.fma(tl.load(pp + d, mask=m, other=0.0), y, acc)
        tl.store(at + d * H, out, mask=m)
        outs = outs + (out,)
    return outs


@triton.jit
def _dots(r_ptr, fn_ptr, y_ptr, post_ptr, comb_ptr, part_ptr, sq_ptr, T, H: tl.constexpr, W: tl.constexpr,
          PENDING: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr, KPP: tl.constexpr, BW: tl.constexpr):
    """Partials of one chunk (KPP · BK columns h of every stream) for BM rows: the (2 + n)·n products [BM, BW] and
    the sums of squares, after the pending write-back (this program owns columns h of every stream: D3)."""
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    chunk = tl.program_id(1)
    rm = rows < T
    m = rm[:, None]
    j = tl.arange(0, BW)
    wm = (j < W)[None, :]
    acc = tl.zeros([BM, BW], tl.float32)
    q = tl.zeros([BM, BK], tl.float32)
    for kk in range(KPP):
        h = (chunk * KPP + kk) * BK + tl.arange(0, BK)
        if PENDING:
            r0, r1, r2, r3 = _writeback_tile(r_ptr, y_ptr, post_ptr, comb_ptr, rows[:, None], h[None, :], m, H)
        else:
            at = r_ptr + rows[:, None].to(tl.int64) * (4 * H) + h[None, :]
            r0 = tl.load(at, mask=m, other=0.0)
            r1 = tl.load(at + H, mask=m, other=0.0)
            r2 = tl.load(at + 2 * H, mask=m, other=0.0)
            r3 = tl.load(at + 3 * H, mask=m, other=0.0)
        wp = fn_ptr + j[None, :].to(tl.int64) * (4 * H) + h[:, None]
        acc = tl.dot(r0, tl.load(wp, mask=wm, other=0.0), acc, input_precision="ieee")
        acc = tl.dot(r1, tl.load(wp + H, mask=wm, other=0.0), acc, input_precision="ieee")
        acc = tl.dot(r2, tl.load(wp + 2 * H, mask=wm, other=0.0), acc, input_precision="ieee")
        acc = tl.dot(r3, tl.load(wp + 3 * H, mask=wm, other=0.0), acc, input_precision="ieee")
        q = tl.fma(r0, r0, q)
        q = tl.fma(r1, r1, q)
        q = tl.fma(r2, r2, q)
        q = tl.fma(r3, r3, q)
    # q's row sums as a product with ones: a reduction tree would follow the tile's layout (so BM and the warps),
    # where the dot's FMAs do not (BK and KPP are fixed: they set the order)
    ones = tl.full([BK, 16], 1.0, tl.float32)
    sq = tl.sum(tl.where(tl.arange(0, 16)[None, :] == 0, tl.dot(q, ones, input_precision="ieee"), 0.0), 1)
    chunks = tl.num_programs(1)
    tl.store(part_ptr + (rows[:, None].to(tl.int64) * chunks + chunk) * BW + j[None, :], acc, mask=m)
    tl.store(sq_ptr + rows.to(tl.int64) * chunks + chunk, sq, mask=rm)


@triton.jit
def _sigmoid(x):
    return 1.0 / (1.0 + tl.exp(-x))


@triton.jit
def _weights(part_ptr, sq_ptr, base_ptr, scale_ptr, pre_ptr, post_ptr, comb_ptr, eps, hc_eps, H: tl.constexpr,
             CH: tl.constexpr, ITERS: tl.constexpr, BW: tl.constexpr):
    """One row's partials summed over the chunks in a fixed order, the RMS, the affine maps and the Sinkhorn
    (float32): pre (scratch), post and comb."""
    t = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, CH)
    i = tl.arange(0, 4)
    rs = tl.rsqrt(tl.sum(tl.load(sq_ptr + t * CH + c), 0) / (4 * H) + eps)
    part = part_ptr + (t * CH + c[:, None]) * BW
    pre = _sigmoid(tl.sum(tl.load(part + i[None, :]), 0) * rs * tl.load(scale_ptr) + tl.load(base_ptr + i)) + hc_eps
    post = 2.0 * _sigmoid(tl.sum(tl.load(part + 4 + i[None, :]), 0) * rs * tl.load(scale_ptr + 1)
                          + tl.load(base_ptr + 4 + i))
    ii = i[:, None] * 4 + i[None, :]                                   # [s, d]
    logits = (tl.sum(tl.load(part[:, :, None] + 8 + ii[None, :, :]), 0) * rs * tl.load(scale_ptr + 2)
              + tl.load(base_ptr + 8 + ii))
    e = tl.exp(logits - tl.max(logits, 1)[:, None])
    comb = e / tl.sum(e, 1)[:, None] + hc_eps
    comb = comb / (tl.sum(comb, 0)[None, :] + hc_eps)
    for _ in tl.static_range(ITERS - 1):
        comb = comb / (tl.sum(comb, 1)[:, None] + hc_eps)
        comb = comb / (tl.sum(comb, 0)[None, :] + hc_eps)
    tl.store(pre_ptr + t * 4 + i, pre)
    tl.store(post_ptr + t * 4 + i, post)
    tl.store(comb_ptr + t * 16 + ii, comb)


@triton.jit
def _collapse(r_ptr, pre_ptr, x_ptr, T, H: tl.constexpr, BMC: tl.constexpr, BH: tl.constexpr):
    """x = Σ_s pre_s · R_s (s in order), rounded once."""
    rows = tl.program_id(0) * BMC + tl.arange(0, BMC)
    h = tl.program_id(1) * BH + tl.arange(0, BH)
    m = (rows < T)[:, None]
    at = r_ptr + rows[:, None].to(tl.int64) * (4 * H) + h[None, :]
    pp = pre_ptr + rows[:, None] * 4
    x = tl.load(pp, mask=m, other=0.0) * tl.load(at, mask=m, other=0.0)
    x = tl.fma(tl.load(pp + 1, mask=m, other=0.0), tl.load(at + H, mask=m, other=0.0), x)
    x = tl.fma(tl.load(pp + 2, mask=m, other=0.0), tl.load(at + 2 * H, mask=m, other=0.0), x)
    x = tl.fma(tl.load(pp + 3, mask=m, other=0.0), tl.load(at + 3 * H, mask=m, other=0.0), x)
    tl.store(x_ptr + rows[:, None].to(tl.int64) * H + h[None, :], x.to(x_ptr.dtype.element_ty), mask=m)


@triton.jit
def _writeback(r_ptr, y_ptr, post_ptr, comb_ptr, T, H: tl.constexpr, BMW: tl.constexpr, BH: tl.constexpr):
    rows = tl.program_id(0) * BMW + tl.arange(0, BMW)
    h = tl.program_id(1) * BH + tl.arange(0, BH)
    _writeback_tile(r_ptr, y_ptr, post_ptr, comb_ptr, rows[:, None], h[None, :], (rows < T)[:, None], H)


def hc_mhc_mix(r, fn, base, scale, y_prev, post_prev, comb_prev, hidden: int, eps: float, hc_eps: float, iters: int,
               dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t, dev = r.shape[0], r.device
    x = torch.empty(t, hidden, dtype=dtype, device=dev)
    post = torch.empty(t, 4, dtype=torch.float32, device=dev)
    comb = torch.empty(t, 4, 4, dtype=torch.float32, device=dev)
    if t == 0:
        return x, post, comb
    chunks = hidden // (BK * KPP)
    part = torch.empty(t, chunks, BW, dtype=torch.float32, device=dev)
    sq = torch.empty(t, chunks, dtype=torch.float32, device=dev)
    pre = torch.empty(t, 4, dtype=torch.float32, device=dev)
    pending = y_prev is not None
    bm, warps = BM_SMALL if t <= SMALL else BM
    _dots[(triton.cdiv(t, bm), chunks)](r, fn, y_prev if pending else r, post_prev if pending else r,
                                        comb_prev if pending else r, part, sq, t, hidden, fn.shape[0], pending, bm,
                                        BK, KPP, BW, num_warps=warps, num_stages=STAGES)
    _weights[(t,)](part, sq, base, scale, pre, post, comb, eps, hc_eps, hidden, chunks, iters, BW, num_warps=1)
    _collapse[(triton.cdiv(t, BMC), hidden // BH)](r, pre, x, t, hidden, BMC, BH, num_warps=WARPS)
    return x, post, comb


def hc_mhc_writeback(r, y, post, comb) -> None:
    t, hidden = y.shape
    if t:
        _writeback[(triton.cdiv(t, BMW), hidden // BH)](r, y, post, comb, t, hidden, BMW, BH, num_warps=WARPS)
