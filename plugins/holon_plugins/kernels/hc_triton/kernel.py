"""Our Triton kernels for the gated-residual hyper-connections (`hc_gr_mix@1`, `hc_gr_combine@1`; std.hyper). Written
for Holon after studying FreeToken's `kernel/triton/hc.py` (its separate norm, SiLU and mix kernels around cuBLAS
projections): here the norm (its sums of squares too) is folded into the down projection and the mix into the up
projection's epilogue, and a pending combine (the same block's previous half) rides along, so a mix is two launches;
every product in float32 with IEEE sums (D10), each weight read in float32. Correct and graph-safe; tuned in FN's step
7."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

BM, BN, BK = 16, 64, 64          # token, output and reduction tiles
# Decode and verify steps (at most GEMV_ROWS tokens) run GEMV-shaped kernels: each program holds a weight tile once and
# loops over the tokens, `_down_gemv` over one (stream, K chunk) with its partials summed in a fixed order by
# `_down_sum`. FN step 5 measured the tiled `_down` at 442 µs per decode call on the RTX PRO 6000 (six programs for a
# 10,240-long reduction): 78% of the decode step.
GEMV_ROWS, KCHUNK, BNV, BJ = 16, 512, 16, 16


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
def _down_gemv(r_ptr, y_ptr, wp_ptr, norm_ptr, down_ptr, inj_ptr, part_ptr, sq_ptr, T, H, RANK, NOUT,
               S: tl.constexpr, PENDING: tl.constexpr, INJECT: tl.constexpr, BNV: tl.constexpr, CH: tl.constexpr):
    """`_down`'s products over one (stream, K chunk of CH) for BNV outputs, every token: `part` [S·KS, T, NOUT];
    the first output tile also writes the chunk's sums of squares `sq` [S·KS, T]. Products summed within a thread,
    float32."""
    n = tl.program_id(0) * BNV + tl.arange(0, BNV)
    z = tl.program_id(1)
    ks = H // CH
    s = z // ks
    k = (z % ks) * CH + tl.arange(0, CH)
    col = s * H + k
    is_down = n < RANK
    is_inj = (n >= RANK) & (n < NOUT) & INJECT
    w = tl.where(is_down[:, None],
                 tl.load(down_ptr + n[:, None].to(tl.int64) * S * H + col[None, :], mask=is_down[:, None],
                         other=0.0).to(tl.float32),
                 tl.load(inj_ptr + (n - RANK)[:, None].to(tl.int64) * S * H + col[None, :], mask=is_inj[:, None],
                         other=0.0).to(tl.float32))
    scale = 1.0 + tl.load(norm_ptr + col).to(tl.float32)
    for t in range(T):
        x = tl.load(r_ptr + t * S * H + col)
        if PENDING:
            x = x + tl.load(wp_ptr + t * S + s) * tl.load(y_ptr + t * H + k).to(tl.float32)
        if tl.program_id(0) == 0:
            tl.store(sq_ptr + z * T + t, tl.sum(x * x, 0))
        tl.store(part_ptr + (z * T + t) * NOUT + n, tl.sum(w * (x * scale)[None, :], 1), mask=n < NOUT)


@triton.jit
def _down_sum(part_ptr, sq_ptr, a_ptr, inj_out_ptr, rstd_ptr, T, H, RANK, NOUT, eps, S: tl.constexpr,
              KS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    """`_down`'s epilogue over the partials, chunks summed in order: per stream its rstd and partial, then the mean
    over streams and the activations."""
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    rows = m < T
    row = m[:, None].to(tl.int64)
    ok = rows[:, None] & (n < NOUT)[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for s in tl.static_range(S):
        sq = tl.zeros((BM,), tl.float32)
        part = tl.zeros((BM, BN), tl.float32)
        for c in tl.static_range(KS):
            z = s * KS + c
            sq += tl.load(sq_ptr + z * T + m, mask=rows, other=0.0)
            part += tl.load(part_ptr + (z * T + row) * NOUT + n[None, :], mask=ok, other=0.0)
        rstd = tl.rsqrt(sq / H + eps)
        acc += part * rstd[:, None]
        if tl.program_id(1) == 0:
            tl.store(rstd_ptr + m.to(tl.int64) * S + s, rstd, mask=rows)
    acc = acc / S
    tl.store(a_ptr + row * RANK + n[None, :], acc * tl.sigmoid(acc), mask=rows[:, None] & (n < RANK)[None, :])
    tl.store(inj_out_ptr + row * S + (n - RANK)[None, :], 2.0 * tl.sigmoid(acc),
             mask=rows[:, None] & ((n >= RANK) & (n < NOUT))[None, :])


@triton.jit
def _up_mix_gemv(a_ptr, up_ptr, r_ptr, rstd_ptr, norm_ptr, x_ptr, T, H, RANK, S: tl.constexpr, BJ: tl.constexpr,
                 RK: tl.constexpr):
    """`_up_mix` for BJ columns of every stream, every token, each up tile held once (RK ≥ RANK, masked); R as the
    caller left it (a pending combine already applied: storing it here raced with Triton's re-run loads on sm_86)."""
    j = tl.program_id(0) * BJ + tl.arange(0, BJ)
    kk = tl.arange(0, RK)
    jm = j < H
    for t in range(T):
        a = tl.load(a_ptr + t * RANK + kk, mask=kk < RANK, other=0.0)
        out = tl.zeros((BJ,), tl.float32)
        for s in tl.static_range(S):
            w = tl.load(up_ptr + (s * H + j)[:, None].to(tl.int64) * RANK + kk[None, :],
                        mask=jm[:, None] & (kk < RANK)[None, :], other=0.0).to(tl.float32)
            acc = tl.sum(w * a[None, :], 1)
            col = s * H + j
            r_s = tl.load(r_ptr + t * S * H + col, mask=jm, other=0.0)
            scale = tl.load(rstd_ptr + t * S + s) * (1.0 + tl.load(norm_ptr + col, mask=jm, other=0.0).to(tl.float32))
            out += tl.sigmoid(acc) * (r_s * scale)
        tl.store(x_ptr + t * H + j, (out / S).to(x_ptr.dtype.element_ty), mask=jm)


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


def hc_gr_mix(r, norm, down, up, inject, y_prev, w_prev, streams: int, eps: float, dtype: torch.dtype):
    """Two launches: `_down` (its rstd folded in) and `_up_mix`, a pending combine (y_prev, w_prev: the same block's
    previous half) riding along; at most GEMV_ROWS tokens, three (`_down_gemv`, `_down_sum`, `_up_mix_gemv`), after
    `_combine` for a pending combine. The returned w is a new tensor, never w_prev."""
    t, width = r.shape
    hidden, rank = width // streams, down.shape[0]
    if hidden % BK or r.stride(1) != 1 or r.stride(0) != width:
        raise ValueError(f"hc_gr_mix: hidden {hidden} must be a multiple of {BK}, r contiguous")
    x = torch.empty(t, hidden, dtype=dtype, device=r.device)
    w = None if inject is None else torch.empty(t, streams, dtype=torch.float32, device=r.device)
    if t == 0:
        return x, w
    pending = y_prev is not None
    rstd = torch.empty(t, streams, dtype=torch.float32, device=r.device)
    a = torch.empty(t, rank, dtype=torch.float32, device=r.device)
    sink = w if w is not None else rstd                           # never written without inject
    y, w_prev = (y_prev.contiguous(), w_prev.contiguous()) if pending else (rstd, rstd)
    down = down.contiguous()
    inj = inject.contiguous() if inject is not None else down
    nout = rank + (streams if inject is not None else 0)
    if t <= GEMV_ROWS and hidden % KCHUNK == 0:                 # decode and verify steps
        ks = hidden // KCHUNK
        part = torch.empty(streams * ks, t, nout, dtype=torch.float32, device=r.device)
        sq = torch.empty(streams * ks, t, dtype=torch.float32, device=r.device)
        if pending:                                             # R + w · y first: `_combine`'s same expression
            _combine[(t, streams, triton.cdiv(hidden, 1024))](r, y, w_prev, hidden, streams, 1024)
        _down_gemv[(triton.cdiv(nout, BNV), streams * ks)](
            r, y, w_prev, norm, down, inj, part, sq, t, hidden, rank, nout, streams, False, inject is not None,
            BNV, KCHUNK, num_warps=4)
        _down_sum[(triton.cdiv(t, BM), triton.cdiv(nout, BN))](
            part, sq, a, sink, rstd, t, hidden, rank, nout, eps, streams, ks, BM, BN, num_warps=4)
        _up_mix_gemv[(triton.cdiv(hidden, BJ),)](
            a, up.contiguous(), r, rstd, norm, x, t, hidden, rank, streams, BJ, triton.next_power_of_2(rank),
            num_warps=4)
        return x, w
    _down[(triton.cdiv(t, BM), triton.cdiv(nout, BN))](
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
