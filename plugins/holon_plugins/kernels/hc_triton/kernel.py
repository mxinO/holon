"""Our Triton kernels for the gated-residual hyper-connections (`hc_gr_mix@1`, `hc_gr_combine@1`; std.hyper). Written
for Holon after studying FreeToken's `kernel/triton/hc.py` (its separate norm, SiLU and mix kernels around cuBLAS
projections): here the norm (its sums of squares too) is folded into the down projection and the mix into the up
projection's epilogue, and a pending combine rides along, so a mix is two launches; every product in float32 with IEEE
sums (D10), each weight read in float32. Correct and graph-safe; tuned in FN's step 7."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

BM, BN, BK = 16, 64, 64          # token, output and reduction tiles


@triton.jit
def _down(r_ptr, y_ptr, wp_ptr, norm_ptr, down_ptr, inj_ptr, a_ptr, inj_out_ptr, rstd_ptr, T, H, RANK, eps,
          S: tl.constexpr, PENDING: tl.constexpr, INJECT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
          BK: tl.constexpr):
    """[silu(Rn · downᵀ / n) | 2 σ(Rn · injectᵀ / n)] for this program's output columns (down's rank, then inject's n),
    Rn = R · rstd · (1 + norm). Per stream, the products of R · (1 + norm) and the sum of squares share one pass over
    R (with a pending combine, R + w · y as it is loaded); the stream's partial is scaled by its rstd at the end. The
    first column tile also stores each (token, stream)'s rstd for `_up_mix`."""
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    rows = m < T
    is_down = n < RANK
    is_inj = (n >= RANK) & (n < RANK + S) & INJECT
    row = m[:, None].to(tl.int64)
    acc = tl.zeros((BM, BN), tl.float32)
    for s in tl.static_range(S):
        part = tl.zeros((BM, BN), tl.float32)
        sq = tl.zeros((BM,), tl.float32)
        if PENDING:
            wv = tl.load(wp_ptr + m.to(tl.int64) * S + s, mask=rows, other=0.0)
        for k0 in range(0, H, BK):
            k = k0 + tl.arange(0, BK)
            col = s * H + k
            x = tl.load(r_ptr + row * S * H + col[None, :], mask=rows[:, None], other=0.0)
            if PENDING:
                yv = tl.load(y_ptr + row * H + k[None, :], mask=rows[:, None], other=0.0).to(tl.float32)
                x = x + wv[:, None] * yv
            sq += tl.sum(x * x, 1)
            x = x * (1.0 + tl.load(norm_ptr + col).to(tl.float32))[None, :]
            w = tl.where(is_down[None, :],
                         tl.load(down_ptr + n[None, :].to(tl.int64) * S * H + col[:, None], mask=is_down[None, :],
                                 other=0.0).to(tl.float32),
                         tl.load(inj_ptr + (n - RANK)[None, :].to(tl.int64) * S * H + col[:, None],
                                 mask=is_inj[None, :], other=0.0).to(tl.float32))
            part = tl.dot(x, w, part, input_precision="ieee")
        rstd = tl.rsqrt(sq / H + eps)
        acc += part * rstd[:, None]
        if tl.program_id(1) == 0:
            tl.store(rstd_ptr + m.to(tl.int64) * S + s, rstd, mask=rows)
    acc = acc / S
    tl.store(a_ptr + row * RANK + n[None, :], acc * tl.sigmoid(acc), mask=rows[:, None] & is_down[None, :])
    tl.store(inj_out_ptr + row * S + (n - RANK)[None, :], 2.0 * tl.sigmoid(acc), mask=rows[:, None] & is_inj[None, :])


@triton.jit
def _up_mix(a_ptr, up_ptr, r_ptr, y_ptr, wp_ptr, rstd_ptr, norm_ptr, x_ptr, T, H, RANK,
            S: tl.constexpr, PENDING: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """x[:, j] = mean_s(σ(a · up[sH + j]ᵀ) · Rn[:, sH + j]) for this program's columns j of every stream. With a
    pending combine, the program also stores R + w · y for exactly these elements (no other program touches them)."""
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    j = tl.program_id(1) * BN + tl.arange(0, BN)
    rows = m < T
    row = m[:, None].to(tl.int64)
    here = rows[:, None] & (j < H)[None, :]
    out = tl.zeros((BM, BN), tl.float32)
    for s in tl.static_range(S):
        acc = tl.zeros((BM, BN), tl.float32)
        for k0 in range(0, RANK, BK):
            k = k0 + tl.arange(0, BK)
            a = tl.load(a_ptr + row * RANK + k[None, :], mask=rows[:, None] & (k < RANK)[None, :], other=0.0)
            w = tl.load(up_ptr + (s * H + j)[None, :].to(tl.int64) * RANK + k[:, None],
                        mask=(j < H)[None, :] & (k < RANK)[:, None], other=0.0)
            acc = tl.dot(a, w.to(tl.float32), acc, input_precision="ieee")
        col = s * H + j
        rn = tl.load(r_ptr + row * S * H + col[None, :], mask=here, other=0.0)
        if PENDING:
            wv = tl.load(wp_ptr + m.to(tl.int64) * S + s, mask=rows, other=0.0)
            rn = rn + wv[:, None] * tl.load(y_ptr + row * H + j[None, :], mask=here, other=0.0).to(tl.float32)
            tl.store(r_ptr + row * S * H + col[None, :], rn, mask=here)
        rstd = tl.load(rstd_ptr + m.to(tl.int64) * S + s, mask=rows, other=0.0)
        rn = rn * rstd[:, None] * (1.0 + tl.load(norm_ptr + col, mask=j < H, other=0.0).to(tl.float32))[None, :]
        out += tl.sigmoid(acc) * rn
    tl.store(x_ptr + row * H + j[None, :], (out / S).to(x_ptr.dtype.element_ty), mask=here)


@triton.jit
def _combine(r_ptr, y_ptr, w_ptr, H, S: tl.constexpr, BLOCK: tl.constexpr):
    row, s = tl.program_id(0).to(tl.int64), tl.program_id(1)
    c = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    w = tl.load(w_ptr + row * S + s)
    y = tl.load(y_ptr + row * H + c, mask=c < H, other=0.0).to(tl.float32)
    p = r_ptr + row * S * H + s * H + c
    tl.store(p, tl.load(p, mask=c < H, other=0.0) + w * y, mask=c < H)


def hc_gr_mix(r, norm, down, up, inject, y, w_prev, streams: int, eps: float, dtype: torch.dtype):
    """Two launches: `_down` (its rstd folded in) and `_up_mix`; a pending combine (y, w_prev) rides along."""
    t, width = r.shape
    hidden, rank = width // streams, down.shape[0]
    if hidden % BK or r.stride(1) != 1 or r.stride(0) != width:
        raise ValueError(f"hc_gr_mix: hidden {hidden} must be a multiple of {BK}, r contiguous")
    x = torch.empty(t, hidden, dtype=dtype, device=r.device)
    w = None if inject is None else torch.empty(t, streams, dtype=torch.float32, device=r.device)
    if t == 0:
        return x, w
    pending = y is not None
    rstd = torch.empty(t, streams, dtype=torch.float32, device=r.device)
    a = torch.empty(t, rank, dtype=torch.float32, device=r.device)
    sink = w if w is not None else rstd                           # never written without inject
    y, w_prev = (y.contiguous(), w_prev.contiguous()) if pending else (rstd, rstd)
    down = down.contiguous()
    inj = inject.contiguous() if inject is not None else down
    _down[(triton.cdiv(t, BM), triton.cdiv(rank + (streams if inject is not None else 0), BN))](
        r, y, w_prev, norm, down, inj, a, sink, rstd, t, hidden, rank, eps, streams, pending, inject is not None,
        BM, BN, BK, num_warps=4)
    _up_mix[(triton.cdiv(t, BM), triton.cdiv(hidden, BN))](
        a, up.contiguous(), r, y, w_prev, rstd, norm, x, t, hidden, rank, streams, pending, BM, BN, 32, num_warps=4)
    return x, w


def hc_gr_combine(r, y, w) -> None:
    t, width = r.shape
    streams = w.shape[1]
    if t:
        _combine[(t, streams, triton.cdiv(width // streams, 1024))](r, y.contiguous(), w.contiguous(), width // streams,
                                                                     streams, 1024)
