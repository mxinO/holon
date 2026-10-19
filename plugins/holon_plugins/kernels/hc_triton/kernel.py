"""Our Triton kernels for the gated-residual hyper-connections (`hc_gr_mix@1`, `hc_gr_combine@1`; std.hyper). Written
for Holon after studying FreeToken's `kernel/triton/hc.py` (its separate norm, SiLU and mix kernels around cuBLAS
projections): here the norm is folded into the projections' loads and the mix into the up projection's epilogue, every
product in float32 with IEEE sums (D10). Correct and graph-safe; tuned in FN's step 7."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

BM, BN, BK = 16, 64, 64          # token, output and reduction tiles


@triton.jit
def _rstd(r_ptr, out_ptr, H, eps, S: tl.constexpr, BLOCK: tl.constexpr):
    """1 / rms of each (token, stream) of R."""
    row, s = tl.program_id(0).to(tl.int64), tl.program_id(1)
    acc = tl.zeros((BLOCK,), tl.float32)
    for c in range(0, H, BLOCK):
        x = tl.load(r_ptr + row * S * H + s * H + c + tl.arange(0, BLOCK), mask=c + tl.arange(0, BLOCK) < H, other=0.0)
        acc += x * x
    tl.store(out_ptr + row * S + s, tl.rsqrt(tl.sum(acc) / H + eps))


@triton.jit
def _down(r_ptr, rstd_ptr, norm_ptr, w_ptr, a_ptr, inj_out_ptr, T, H, RANK, OUT,
          S: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """[silu(Rn · downᵀ / n) | 2 σ(Rn · injectᵀ / n)] with Rn = R · rstd · (1 + norm) formed as it is loaded; `w` is
    down's rows followed by inject's (OUT = RANK [+ S])."""
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, S * H, BK):
        k = k0 + tl.arange(0, BK)
        s = k0 // H                                           # one stream per tile: H % BK == 0
        x = tl.load(r_ptr + m[:, None].to(tl.int64) * S * H + k[None, :], mask=(m < T)[:, None], other=0.0)
        rstd = tl.load(rstd_ptr + m.to(tl.int64) * S + s, mask=m < T, other=0.0)
        g = 1.0 + tl.load(norm_ptr + k).to(tl.float32)
        x = x * rstd[:, None] * g[None, :]
        w = tl.load(w_ptr + n[None, :].to(tl.int64) * S * H + k[:, None], mask=(n < OUT)[None, :], other=0.0)
        acc = tl.dot(x, w.to(tl.float32), acc, input_precision="ieee")
    acc = acc / S
    row = m[:, None].to(tl.int64)
    tl.store(a_ptr + row * RANK + n[None, :], acc * tl.sigmoid(acc), mask=(m < T)[:, None] & (n < RANK)[None, :])
    tl.store(inj_out_ptr + row * S + (n - RANK)[None, :], 2.0 * tl.sigmoid(acc),
             mask=(m < T)[:, None] & (n >= RANK)[None, :] & (n < OUT)[None, :])


@triton.jit
def _up_mix(a_ptr, up_ptr, r_ptr, rstd_ptr, norm_ptr, x_ptr, T, H, RANK,
            S: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """x[:, j] = mean_s(σ(a · up[sH + j]ᵀ) · Rn[:, sH + j]) for this program's columns j."""
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    j = tl.program_id(1) * BN + tl.arange(0, BN)
    out = tl.zeros((BM, BN), tl.float32)
    for s in range(S):
        acc = tl.zeros((BM, BN), tl.float32)
        for k0 in range(0, RANK, BK):
            k = k0 + tl.arange(0, BK)
            a = tl.load(a_ptr + m[:, None].to(tl.int64) * RANK + k[None, :],
                        mask=(m < T)[:, None] & (k < RANK)[None, :], other=0.0)
            w = tl.load(up_ptr + (s * H + j)[None, :].to(tl.int64) * RANK + k[:, None],
                        mask=(j < H)[None, :] & (k < RANK)[:, None], other=0.0)
            acc = tl.dot(a, w.to(tl.float32), acc, input_precision="ieee")
        col = s * H + j
        rn = tl.load(r_ptr + m[:, None].to(tl.int64) * S * H + col[None, :], mask=(m < T)[:, None] & (j < H)[None, :],
                     other=0.0)
        rstd = tl.load(rstd_ptr + m.to(tl.int64) * S + s, mask=m < T, other=0.0)
        rn = rn * rstd[:, None] * (1.0 + tl.load(norm_ptr + col, mask=j < H, other=0.0).to(tl.float32))[None, :]
        out += tl.sigmoid(acc) * rn
    tl.store(x_ptr + m[:, None].to(tl.int64) * H + j[None, :], (out / S).to(x_ptr.dtype.element_ty),
             mask=(m < T)[:, None] & (j < H)[None, :])


@triton.jit
def _combine(r_ptr, y_ptr, w_ptr, H, S: tl.constexpr, BLOCK: tl.constexpr):
    row, s = tl.program_id(0).to(tl.int64), tl.program_id(1)
    c = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    w = tl.load(w_ptr + row * S + s)
    y = tl.load(y_ptr + row * H + c, mask=c < H, other=0.0).to(tl.float32)
    p = r_ptr + row * S * H + s * H + c
    tl.store(p, tl.load(p, mask=c < H, other=0.0) + w * y, mask=c < H)


def hc_gr_mix(r, norm, down, up, inject, streams: int, eps: float, dtype: torch.dtype):
    t, width = r.shape
    hidden, rank = width // streams, down.shape[0]
    if hidden % BK or r.stride(1) != 1 or r.stride(0) != width:
        raise ValueError(f"hc_gr_mix: hidden {hidden} must be a multiple of {BK}, r contiguous")
    x = torch.empty(t, hidden, dtype=dtype, device=r.device)
    w = None if inject is None else torch.empty(t, streams, dtype=torch.float32, device=r.device)
    if t == 0:
        return x, w
    rstd = torch.empty(t, streams, dtype=torch.float32, device=r.device)
    _rstd[(t, streams)](r, rstd, hidden, eps, streams, 1024)
    weights = down.contiguous() if inject is None else torch.cat((down, inject.to(down.dtype))).contiguous()
    a = torch.empty(t, rank, dtype=torch.float32, device=r.device)
    sink = w if w is not None else torch.empty(1, device=r.device)
    _down[(triton.cdiv(t, BM), triton.cdiv(len(weights), BN))](
        r, rstd, norm, weights, a, sink, t, hidden, rank, len(weights), streams, BM, BN, BK, num_warps=4)
    _up_mix[(triton.cdiv(t, BM), triton.cdiv(hidden, BN))](
        a, up.contiguous(), r, rstd, norm, x, t, hidden, rank, streams, BM, BN, 32, num_warps=4)
    return x, w


def hc_gr_combine(r, y, w) -> None:
    t, width = r.shape
    streams = w.shape[1]
    if t:
        _combine[(t, streams, triton.cdiv(width // streams, 1024))](r, y.contiguous(), w.contiguous(), width // streams,
                                                                     streams, 1024)
