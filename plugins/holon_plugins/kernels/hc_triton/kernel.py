"""Our kernels for the gated-residual hyper-connections (`hc_gr_mix@1`, `hc_gr_combine@1`; std.hyper). Written for
Holon after studying FreeToken's `kernel/triton/hc.py` (its separate norm, SiLU and mix kernels around cuBLAS
projections). Decode and verify (GEMV-shaped, Triton): the norm (its sums of squares too) folded into the down
projection, the mix into the up projection, a pending combine (the same block's previous half) riding along. Prefill:
the two projections as cuBLAS SGEMMs (FN step 7: the Triton tiles' float32 dots reached 9% of the card's float32 peak,
cuBLAS's 50%) between Triton prologue and epilogues. Every product in float32 with IEEE sums (D10), each weight read in
float32. Correct and graph-safe."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

BM, BN, BK = 16, 64, 64          # token, output and reduction tiles (`_down_sum`'s, and the hidden size's multiple)
PREFILL_ROWS = 4096               # prefill: rows per pass (its float32 scratch: 2 × rows × S·H) — under bf16x3
PREFILL_ROWS_X3 = 1024           # 1,024 (`_mm3`'s products are per row tile and K order: whole tiles, the same bits)
# Decode and verify steps (at most GEMV_ROWS tokens) run GEMV-shaped kernels, `_down_gemv` over one (stream, K chunk)
# with its weight tile held while the rows loop. A one-token step takes two launches: every program of `_mix_gemv`
# sums the partials in a fixed order and a pending combine rides along (`rn`). At 2-16 tokens `_down_sum` sums once
# and `_mix_rows` holds each up tile while the rows loop (FN step 7, lever B: 69 -> 49 µs per 4-token mix, 117 -> 78
# at 8 on an RTX A5000; per-row reductions keep it above 1.5x the one-token time). FN step 5 measured the tiled
# `_down` at 442 µs per decode call on the RTX PRO 6000 (six programs for a 10,240-long reduction): 78% of the step.
GEMV_ROWS, KCHUNK, BNV, BJ = 16, 512, 16, 16
# 2-16 tokens: `_down_gemv`'s outputs per program, chunk and warps, `_mix_rows`' columns and warps (RTX A5000 sweep)
DOWN_ROWS, MIX_ROWS = (8, 512, 1), (8, 4)
WARPS = 4                        # every launch's; L0's stress case runs others


@triton.jit
def _down_gemv(r_ptr, y_ptr, wp_ptr, norm_ptr, down_ptr, inj_ptr, part_ptr, sq_ptr, rn_ptr, T, H, RANK, NOUT,
               S: tl.constexpr, PENDING: tl.constexpr, INJECT: tl.constexpr, BNV: tl.constexpr, CH: tl.constexpr):
    """`_down`'s products over one (stream, K chunk of CH) for BNV outputs, every token: `part` [S·KS, T, NOUT];
    the first output tile also writes the chunk's sums of squares `sq` [S·KS, T] and, with a pending combine, the
    combined R + w · y into the scratch `rn` (R itself is written by `_mix_gemv`, from `rn`: no kernel stores into a
    buffer it reads, which raced on sm_86 when one did). Products summed within a thread, float32."""
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
            if PENDING:
                tl.store(rn_ptr + t * S * H + col, x)
        tl.store(part_ptr + (z * T + t) * NOUT + n, tl.sum(w * (x * scale)[None, :], 1), mask=n < NOUT)


@triton.jit
def _mix_rows(a_ptr, rstd_ptr, up_ptr, r_ptr, rn_ptr, norm_ptr, x_ptr, T, H, RANK, S: tl.constexpr,
              PENDING: tl.constexpr, BJ: tl.constexpr, RK: tl.constexpr, TP: tl.constexpr):
    """`_mix_gemv` for 2 to TP tokens (`_down_sum`'s `a` and `rstd`): per stream the up tile [BJ, RK] is read once
    and held while the rows loop over it, each row's contribution added into its row of a [TP, BJ] accumulator; a
    pending combine's R from `rn`, written to R for exactly this program's columns."""
    j = tl.program_id(0) * BJ + tl.arange(0, BJ)
    kk = tl.arange(0, RK)
    tt = tl.arange(0, TP)
    jm, km = j < H, kk < RANK
    out = tl.zeros((TP, BJ), tl.float32)
    for s in tl.static_range(S):
        w = tl.load(up_ptr + (s * H + j)[:, None].to(tl.int64) * RANK + kk[None, :],
                    mask=jm[:, None] & km[None, :], other=0.0).to(tl.float32)
        col = s * H + j
        scale = 1.0 + tl.load(norm_ptr + col, mask=jm, other=0.0).to(tl.float32)
        for t in range(T):
            a = tl.load(a_ptr + t * RANK + kk, mask=km, other=0.0)
            up = tl.sum(w * a[None, :], 1)
            if PENDING:
                r_s = tl.load(rn_ptr + t * S * H + col, mask=jm, other=0.0)
                tl.store(r_ptr + t * S * H + col, r_s, mask=jm)
            else:
                r_s = tl.load(r_ptr + t * S * H + col, mask=jm, other=0.0)
            c = tl.sigmoid(up) * (r_s * (tl.load(rstd_ptr + t * S + s) * scale))
            out = tl.where((tt == t)[:, None], out + c[None, :], out)
    tl.store(x_ptr + tt[:, None] * H + j[None, :], (out / S).to(x_ptr.dtype.element_ty),
             mask=(tt < T)[:, None] & jm[None, :])


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
def _mix_gemv(part_ptr, sq_ptr, a_ptr, rstd_ptr, up_ptr, r_ptr, rn_ptr, norm_ptr, x_ptr, inj_out_ptr, T, H, RANK,
              NOUT, eps, S: tl.constexpr, KS: tl.constexpr, SUMMED: tl.constexpr, PENDING: tl.constexpr,
              BJ: tl.constexpr, RK: tl.constexpr, SP: tl.constexpr):
    """`_up_mix` for BJ columns of every stream, every token, each up tile held once (RK ≥ RANK, masked), and unless
    `_down_sum` ran (SUMMED: its activations `a` and `rstd`) `_down`'s epilogue too: every program then sums the
    partials itself, chunks in order (per stream its rstd and partial, then the mean over streams and the activations:
    `_down_sum`'s expressions in its order in every program, so the same bits), and the first stores the combine
    weights (inject). With a pending combine the mixed R is `rn` (R + w · y, `_down_gemv`'s), which this program also
    writes to R for exactly its columns (no other program touches them)."""
    j = tl.program_id(0) * BJ + tl.arange(0, BJ)
    kk = tl.arange(0, RK)
    ii = RANK + tl.arange(0, SP)
    jm, km, im = j < H, kk < RANK, ii < NOUT
    for t in range(T):
        if SUMMED:
            a = tl.load(a_ptr + t * RANK + kk, mask=km, other=0.0)
        else:
            a = _activations(part_ptr, sq_ptr, inj_out_ptr, t, T, H, RANK, NOUT, eps, kk, ii, km, im, S, KS, RK, SP)
        out = tl.zeros((BJ,), tl.float32)
        for s in tl.static_range(S):
            if SUMMED:
                rstd = tl.load(rstd_ptr + t * S + s)
            else:
                sq = 0.0
                for c in tl.static_range(KS):
                    sq += tl.load(sq_ptr + (s * KS + c) * T + t)
                rstd = tl.rsqrt(sq / H + eps)
            w = tl.load(up_ptr + (s * H + j)[:, None].to(tl.int64) * RANK + kk[None, :],
                        mask=jm[:, None] & km[None, :], other=0.0).to(tl.float32)
            up = tl.sum(w * a[None, :], 1)
            col = s * H + j
            if PENDING:
                r_s = tl.load(rn_ptr + t * S * H + col, mask=jm, other=0.0)
                tl.store(r_ptr + t * S * H + col, r_s, mask=jm)
            else:
                r_s = tl.load(r_ptr + t * S * H + col, mask=jm, other=0.0)
            scale = rstd * (1.0 + tl.load(norm_ptr + col, mask=jm, other=0.0).to(tl.float32))
            out += tl.sigmoid(up) * (r_s * scale)
        tl.store(x_ptr + t * H + j, (out / S).to(x_ptr.dtype.element_ty), mask=jm)


@triton.jit
def _activations(part_ptr, sq_ptr, inj_out_ptr, t, T, H, RANK, NOUT, eps, kk, ii, km, im, S: tl.constexpr,
                 KS: tl.constexpr, RK: tl.constexpr, SP: tl.constexpr):
    """`_down_sum` for token t, in registers: the activations over kk (RANK, masked); program 0 stores the combine
    weights."""
    acc = tl.zeros((RK,), tl.float32)
    inj = tl.zeros((SP,), tl.float32)
    for s in tl.static_range(S):
        sq = 0.0
        part = tl.zeros((RK,), tl.float32)
        part_i = tl.zeros((SP,), tl.float32)
        for c in tl.static_range(KS):
            z = s * KS + c
            sq += tl.load(sq_ptr + z * T + t)
            part += tl.load(part_ptr + (z * T + t) * NOUT + kk, mask=km, other=0.0)
            part_i += tl.load(part_ptr + (z * T + t) * NOUT + ii, mask=im, other=0.0)
        rstd = tl.rsqrt(sq / H + eps)
        acc += part * rstd
        inj += part_i * rstd
    acc = acc / S
    if tl.program_id(0) == 0:
        tl.store(inj_out_ptr + t * S + (ii - RANK), 2.0 * tl.sigmoid(inj / S), mask=im)
    return tl.where(km, acc * tl.sigmoid(acc), 0.0)


@triton.jit
def _prep(r_ptr, y_ptr, wp_ptr, norm_ptr, xn_ptr, sq_ptr, T, H, S: tl.constexpr, PENDING: tl.constexpr,
          BM: tl.constexpr, BH: tl.constexpr):
    """Prefill: Xn = (R + w · y) · (1 + norm) [T, S·H] (a pending combine applied as R is read; R unchanged) and each
    (token, stream)'s sum of squares of R + w · y, for this program's rows of one stream."""
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    s = tl.program_id(1)
    rows = m < T
    row = m[:, None].to(tl.int64)
    if PENDING:
        wv = tl.load(wp_ptr + m.to(tl.int64) * S + s, mask=rows, other=0.0)
    sq = tl.zeros((BM,), tl.float32)
    for h0 in range(0, H, BH):
        k = h0 + tl.arange(0, BH)
        col = s * H + k
        x = tl.load(r_ptr + row * S * H + col[None, :], mask=rows[:, None], other=0.0)
        if PENDING:
            x = x + wv[:, None] * tl.load(y_ptr + row * H + k[None, :], mask=rows[:, None], other=0.0).to(tl.float32)
        sq += tl.sum(x * x, 1)
        tl.store(xn_ptr + row * S * H + col[None, :], x * (1.0 + tl.load(norm_ptr + col).to(tl.float32))[None, :],
                 mask=rows[:, None])
    tl.store(sq_ptr + m.to(tl.int64) * S + s, sq, mask=rows)


@triton.jit
def _act(p_ptr, sq_ptr, a_ptr, inj_out_ptr, rstd_ptr, T, H, RANK, NOUT, eps, S: tl.constexpr, BM: tl.constexpr,
         BN: tl.constexpr):
    """Prefill: the down projection's epilogue over the per-stream products P [S, T, NOUT] (cuBLAS's): per stream its
    rstd, the mean over streams of P · rstd, then silu (a) and 2σ (the combine weights)."""
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    rows = m < T
    row = m[:, None].to(tl.int64)
    ok = rows[:, None] & (n < NOUT)[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for s in tl.static_range(S):
        rstd = tl.rsqrt(tl.load(sq_ptr + m.to(tl.int64) * S + s, mask=rows, other=0.0) / H + eps)
        acc += tl.load(p_ptr + (s * T + row) * NOUT + n[None, :], mask=ok, other=0.0) * rstd[:, None]
        if tl.program_id(1) == 0:
            tl.store(rstd_ptr + m.to(tl.int64) * S + s, rstd, mask=rows)
    acc = acc / S
    tl.store(a_ptr + row * RANK + n[None, :], acc * tl.sigmoid(acc), mask=rows[:, None] & (n < RANK)[None, :])
    tl.store(inj_out_ptr + row * S + (n - RANK)[None, :], 2.0 * tl.sigmoid(acc),
             mask=rows[:, None] & ((n >= RANK) & (n < NOUT))[None, :])


@triton.jit
def _mix_out(u_ptr, r_ptr, y_ptr, wp_ptr, rstd_ptr, norm_ptr, x_ptr, T, H, S: tl.constexpr, PENDING: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr):
    """Prefill: x[:, j] = mean_s(σ(U[:, sH + j]) · Rn[:, sH + j]) for this program's columns j, U = a · upᵀ (cuBLAS's),
    Rn = (R + w · y) · rstd · (1 + norm) from R as left; with a pending combine, R + w · y stored for exactly these
    elements — R read once, written in the same layout after the value is used."""
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    j = tl.program_id(1) * BN + tl.arange(0, BN)
    rows = m < T
    row = m[:, None].to(tl.int64)
    here = rows[:, None] & (j < H)[None, :]
    out = tl.zeros((BM, BN), tl.float32)
    for s in tl.static_range(S):
        col = s * H + j
        p = r_ptr + row * S * H + col[None, :]
        rn = tl.load(p, mask=here, other=0.0)
        if PENDING:
            wv = tl.load(wp_ptr + m.to(tl.int64) * S + s, mask=rows, other=0.0)
            rn = rn + wv[:, None] * tl.load(y_ptr + row * H + j[None, :], mask=here, other=0.0).to(tl.float32)
        u = tl.load(u_ptr + row * S * H + col[None, :], mask=here, other=0.0)
        rstd = tl.load(rstd_ptr + m.to(tl.int64) * S + s, mask=rows, other=0.0)
        scale = (1.0 + tl.load(norm_ptr + col, mask=j < H, other=0.0).to(tl.float32))[None, :]
        out += tl.sigmoid(u) * (rn * rstd[:, None] * scale)
        if PENDING:
            tl.store(p, rn, mask=here)           # after its use, in the layout it was read in
    tl.store(x_ptr + row * H + j[None, :], (out / S).to(x_ptr.dtype.element_ty), mask=here)


@triton.jit
def _combine(r_ptr, y_ptr, w_ptr, H, S: tl.constexpr, BLOCK: tl.constexpr):
    row, s = tl.program_id(0).to(tl.int64), tl.program_id(1)
    c = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    w = tl.load(w_ptr + row * S + s)
    y = tl.load(y_ptr + row * H + c, mask=c < H, other=0.0).to(tl.float32)
    p = r_ptr + row * S * H + s * H + c
    tl.store(p, tl.load(p, mask=c < H, other=0.0) + w * y, mask=c < H)


@triton.jit
def _mm3(a_ptr, b_ptr, c_ptr, M, N, K, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
         BM: tl.constexpr, BN: tl.constexpr, BK3: tl.constexpr):
    """C[b] = A[b] (float32) · B[b] (bf16): A's exact split into three bf16 terms (hi, mid, lo: 24 bits of mantissa),
    each times B on tensor cores into one float32 accumulator — float32 products of B's exact values, sums in the
    tensor cores' order (FN step 7 item 7, the lead's D10 call)."""
    pid_m, pid_n, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rm, rn, rk = pid_m * BM + tl.arange(0, BM), pid_n * BN + tl.arange(0, BN), tl.arange(0, BK3)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK3):
        a = tl.load(a_ptr + b * sab + rm[:, None] * sam + (k0 + rk)[None, :] * sak,
                    mask=(rm[:, None] < M) & ((k0 + rk)[None, :] < K), other=0.0)
        w = tl.load(b_ptr + b * sbb + (k0 + rk)[:, None] * sbk + rn[None, :] * sbn,
                    mask=((k0 + rk)[:, None] < K) & (rn[None, :] < N), other=0.0)
        hi = a.to(tl.bfloat16)
        rest = a - hi.to(tl.float32)
        mid = rest.to(tl.bfloat16)
        lo = (rest - mid.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(hi, w, acc)
        acc = tl.dot(mid, w, acc)
        acc = tl.dot(lo, w, acc)
    tl.store(c_ptr + b * scb + rm[:, None] * scm + rn[None, :] * scn, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


MM3 = (128, 64, 32)                  # its tile (rows, columns, depth): FN's shapes on an RTX A5000 and RTX 5080


def bf16x3_bmm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """[B, M, K] float32 · [B, K, N] bf16 → [B, M, N] float32 by `_mm3` (any strides)."""
    b, m, k = x.shape
    n = w.shape[2]
    c = torch.empty(b, m, n, dtype=torch.float32, device=x.device)
    bm, bn, bk = MM3
    _mm3[(triton.cdiv(m, bm), triton.cdiv(n, bn), b)](x, w, c, m, n, k, *x.stride(), *w.stride(), *c.stride(),
                                                      bm, bn, bk, num_warps=4, num_stages=3)
    return c


def hc_gr_mix(r, norm, down, up, inject, y_prev, w_prev, streams: int, eps: float, dtype: torch.dtype,
              prefill: str = "sgemm"):
    """At most GEMV_ROWS tokens (decode, verify): `_down_gemv`, then `_mix_gemv` for one token, `_down_sum` and
    `_mix_rows` for more (a pending combine through `rn`). Prefill, in passes of PREFILL_ROWS rows: `_prep`
    (Xn and the sums of squares, a pending combine applied as R is read), the down projection as one cuBLAS batched
    SGEMM per stream (TF32 off: float32 products, IEEE sums, D10), `_act`, the up projection as one SGEMM, and
    `_mix_out` (which applies a pending combine to R). The returned w is a new tensor, never w_prev."""
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
        bnv, kchunk, warps = (BNV, KCHUNK, WARPS) if t == 1 else DOWN_ROWS
        ks = hidden // kchunk
        part = torch.empty(streams * ks, t, nout, dtype=torch.float32, device=r.device)
        sq = torch.empty(streams * ks, t, dtype=torch.float32, device=r.device)
        rn = torch.empty_like(r) if pending else r              # R + w · y, written by `_down_gemv`
        _down_gemv[(triton.cdiv(nout, bnv), streams * ks)](
            r, y, w_prev, norm, down, inj, part, sq, rn, t, hidden, rank, nout, streams, pending, inject is not None,
            bnv, kchunk, num_warps=warps)
        if t == 1:                       # two launches: every `_mix_gemv` program sums the partials itself
            _mix_gemv[(triton.cdiv(hidden, BJ),)](
                part, sq, a, rstd, up.contiguous(), r, rn, norm, x, sink, t, hidden, rank, nout, eps, streams, ks,
                False, pending, BJ, triton.next_power_of_2(rank), triton.next_power_of_2(max(nout - rank, 1)),
                num_warps=WARPS)
            return x, w
        _down_sum[(triton.cdiv(t, BM), triton.cdiv(nout, BN))](
            part, sq, a, sink, rstd, t, hidden, rank, nout, eps, streams, ks, BM, BN, num_warps=WARPS)
        bj, warps = MIX_ROWS
        _mix_rows[(triton.cdiv(hidden, bj),)](
            a, rstd, up.contiguous(), r, rn, norm, x, t, hidden, rank, streams, pending, bj,
            triton.next_power_of_2(rank), GEMV_ROWS, num_warps=warps)
        return x, w
    if torch.backends.cuda.matmul.allow_tf32:                  # D10: float32 products, IEEE sums
        raise RuntimeError("hc_gr_mix: TF32 matmuls are enabled (torch.backends.cuda.matmul.allow_tf32)")
    x3 = prefill == "bf16x3" and down.dtype == up.dtype == inj.dtype == torch.bfloat16
    cast = (lambda t: t) if x3 else (lambda t: t.float())       # bf16x3 reads the bf16 weights as they are
    w_down = torch.cat([cast(down), cast(inj)]) if inject is not None else cast(down)   # [NOUT, S·H]
    w_down = w_down.view(nout, streams, hidden).permute(1, 2, 0)                   # [S, H, NOUT]
    w_up = cast(up).t()                                                            # [rank, S·H]
    rows = PREFILL_ROWS_X3 if x3 else PREFILL_ROWS
    for c0 in range(0, t, rows):                               # the scratch bounded at `rows` rows
        c1 = min(t, c0 + rows)
        n = c1 - c0
        rc, xc = r[c0:c1], x[c0:c1]
        yc, wc = (y[c0:c1], w_prev[c0:c1]) if pending else (y, w_prev)
        xn = torch.empty(n, width, dtype=torch.float32, device=r.device)
        sq = torch.empty(n, streams, dtype=torch.float32, device=r.device)
        _prep[(triton.cdiv(n, 16), streams)](rc, yc, wc, norm, xn, sq, n, hidden, streams, pending, 16, BK,
                                             num_warps=WARPS)
        xs = xn.view(n, streams, hidden).transpose(0, 1)
        p = bf16x3_bmm(xs, w_down) if x3 else torch.bmm(xs, w_down)                # [S, n, NOUT]
        ac, rs = a[c0:c1], rstd[c0:c1]
        _act[(triton.cdiv(n, BM), triton.cdiv(nout, BN))](
            p, sq, ac, sink[c0:c1], rs, n, hidden, rank, nout, eps, streams, BM, BN, num_warps=WARPS)
        u = bf16x3_bmm(ac[None], w_up[None])[0] if x3 else ac @ w_up              # [n, S·H]
        del xn
        _mix_out[(triton.cdiv(n, 16), triton.cdiv(hidden, 128))](
            u, rc, yc, wc, rs, norm, xc, n, hidden, streams, pending, 16, 128, num_warps=WARPS)
    return x, w


def hc_gr_combine(r, y, w) -> None:
    t, width = r.shape
    streams = w.shape[1]
    if t:
        _combine[(t, streams, triton.cdiv(width // streams, 1024))](r, y.contiguous(), w.contiguous(), width // streams,
                                                                     streams, 1024)
