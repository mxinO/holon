"""Our Triton kernels for Qwen3.8-Flash-Next's PLE ops (`qwen4_exp.ple_rows@1`, `qwen4_exp.ple_block@1`; contracts:
models.qwen4_exp). The block is float32 throughout with every reduction per token in a fixed tile order, so a row split
into chunks is bitwise one shot. Correct and graph-safe; tuned in FN's step 7."""
from __future__ import annotations

import triton
import triton.language as tl

TOKENS, CHANNELS = 64, 512


@triton.jit
def _ple_rows(ids_ptr, hist_ptr, slots_ptr, qs_ptr, init_ptr, mult_ptr, size_ptr, off_ptr, out_ptr,
              HEADS_PER: tl.constexpr, NGRAM: tl.constexpr, EOS: tl.constexpr, BT: tl.constexpr):
    r = tl.program_id(0)
    s, e = tl.load(qs_ptr + r), tl.load(qs_ptr + r + 1)
    i = s + tl.program_id(1) * BT + tl.arange(0, BT)
    ok = i < e
    slot = tl.load(slots_ptr + r).to(tl.int64)
    init = tl.load(init_ptr + r) != 0
    h0 = tl.where(init, tl.load(hist_ptr + slot * 2), EOS)          # the id two before the row
    h1 = tl.where(init, tl.load(hist_ptr + slot * 2 + 1), EOS)      # the id just before it
    x = tl.load(ids_ptr + i, mask=ok, other=EOS)
    p1 = tl.where(i - 1 >= s, tl.load(ids_ptr + i - 1, mask=ok & (i - 1 >= s), other=EOS), h1)
    p2 = tl.where(i - 2 >= s, tl.load(ids_ptr + i - 2, mask=ok & (i - 2 >= s), other=EOS), tl.where(i - 1 >= s, h1, h0))
    p2 = tl.where(p1 == EOS, EOS, p2)                                 # the window is cut at EOS
    mixed = x * tl.load(mult_ptr) ^ p1 * tl.load(mult_ptr + 1)
    for n in tl.static_range(2, NGRAM + 1):
        if n == 3:
            mixed = mixed ^ p2 * tl.load(mult_ptr + 2)
        for h in tl.static_range(HEADS_PER):
            head = (n - 2) * HEADS_PER + h
            size = tl.load(size_ptr + head)
            rem = mixed % size
            rem = tl.where(rem < 0, rem + size, rem)
            tl.store(out_ptr + i.to(tl.int64) * (NGRAM - 1) * HEADS_PER + head, rem + tl.load(off_ptr + head),
                     mask=ok)


@triton.jit
def _ple_history(ids_ptr, hist_ptr, slots_ptr, qs_ptr, init_ptr, EOS: tl.constexpr):
    r = tl.program_id(0)
    s, e = tl.load(qs_ptr + r), tl.load(qs_ptr + r + 1)
    if e > s:
        slot = tl.load(slots_ptr + r).to(tl.int64)
        before = tl.where(tl.load(init_ptr + r) != 0, tl.load(hist_ptr + slot * 2 + 1), EOS)
        older = tl.where(e - s >= 2, tl.load(ids_ptr + tl.maximum(e - 2, s)), before)
        tl.store(hist_ptr + slot * 2, older)
        tl.store(hist_ptr + slot * 2 + 1, tl.load(ids_ptr + e - 1))


@triton.jit
def _ple_gate(key_ptr, value_ptr, r_ptr, wk_ptr, wq_ptr, wc_ptr, g_out_ptr, c_out_ptr, H, eps, sk, sv,
              S: tl.constexpr, BC: tl.constexpr):
    """Per token: G = σ(signed √u) · value per stream and C = norm(G) · (1 + w_conv), into the scratch rows."""
    t = tl.program_id(0).to(tl.int64)
    vv = tl.zeros((BC,), tl.float32)
    for c0 in range(0, H, BC):
        c = c0 + tl.arange(0, BC)
        v = tl.load(value_ptr + t * sv + c, mask=c < H, other=0.0).to(tl.float32)
        vv += v * v
    v_sq = tl.sum(vv)
    for s in tl.static_range(S):
        kk = tl.zeros((BC,), tl.float32)
        qq = tl.zeros((BC,), tl.float32)
        kq = tl.zeros((BC,), tl.float32)
        for c0 in range(0, H, BC):
            c = c0 + tl.arange(0, BC)
            col = s * H + c
            k = tl.load(key_ptr + t * sk + col, mask=c < H, other=0.0).to(tl.float32)
            q = tl.load(r_ptr + t * S * H + col, mask=c < H, other=0.0)
            wk = 1.0 + tl.load(wk_ptr + col, mask=c < H, other=0.0).to(tl.float32)
            wq = 1.0 + tl.load(wq_ptr + col, mask=c < H, other=0.0).to(tl.float32)
            kk += k * k
            qq += q * q
            kq += (k * wk) * (q * wq)
        rk = tl.rsqrt(tl.sum(kk) / H + eps)
        rq = tl.rsqrt(tl.sum(qq) / H + eps)
        u = rk * rq * tl.sum(kq) / tl.sqrt(H.to(tl.float32))
        root = tl.sqrt(tl.maximum(tl.abs(u), 1e-6))
        g = tl.sigmoid(tl.where(u > 0, root, tl.where(u < 0, -root, 0.0)))
        rc = tl.rsqrt(g * g * v_sq / H + eps)
        for c0 in range(0, H, BC):
            c = c0 + tl.arange(0, BC)
            col = s * H + c
            gated = g * tl.load(value_ptr + t * sv + c, mask=c < H, other=0.0).to(tl.float32)
            wc = 1.0 + tl.load(wc_ptr + col, mask=c < H, other=0.0).to(tl.float32)
            tl.store(g_out_ptr + t * S * H + col, gated, mask=c < H)
            tl.store(c_out_ptr + t * S * H + col, gated * rc * wc, mask=c < H)


@triton.jit
def _ple_conv(r_ptr, g_ptr, c_ptr, w_ptr, state_ptr, slots_ptr, qs_ptr, init_ptr, C,
              K: tl.constexpr, DIL: tl.constexpr, WIN: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr):
    """r += G + silu(Σ_k w_k · C[t − (K − 1 − k)·DIL]), earlier rows from the state (zeros without has_init)."""
    row = tl.program_id(0)
    s, e = tl.load(qs_ptr + row), tl.load(qs_ptr + row + 1)
    i = s + tl.program_id(1) * BT + tl.arange(0, BT)
    c = tl.program_id(2) * BC + tl.arange(0, BC)
    ok = (i < e)[:, None] & (c < C)[None, :]
    slot = tl.load(slots_ptr + row).to(tl.int64)
    init = tl.load(init_ptr + row) != 0
    acc = tl.zeros((BT, BC), tl.float32)
    for k in tl.static_range(K):
        j = i - (K - 1 - k) * DIL                                     # the tap's token
        here = ok & (j >= s)[:, None]
        past = ok & (j < s)[:, None] & init
        x = tl.load(c_ptr + j.to(tl.int64)[:, None] * C + c[None, :], mask=here, other=0.0)
        x += tl.load(state_ptr + (slot * WIN + WIN + (j - s))[:, None] * C + c[None, :], mask=past, other=0.0)
        acc += tl.load(w_ptr + c * K + k, mask=c < C, other=0.0).to(tl.float32)[None, :] * x
    at = i.to(tl.int64)[:, None] * C + c[None, :]
    rv = tl.load(r_ptr + at, mask=ok, other=0.0)
    gv = tl.load(g_ptr + at, mask=ok, other=0.0)
    tl.store(r_ptr + at, rv + gv + acc * tl.sigmoid(acc), mask=ok)


@triton.jit
def _ple_state(c_ptr, state_ptr, slots_ptr, qs_ptr, init_ptr, C, WIN: tl.constexpr, WP: tl.constexpr,
               BC: tl.constexpr):
    """state[slot] ← the last WIN rows of (state ‖ the row's C)."""
    row = tl.program_id(0)
    s, e = tl.load(qs_ptr + row), tl.load(qs_ptr + row + 1)
    n = e - s
    if n > 0:
        c = tl.program_id(1) * BC + tl.arange(0, BC)
        w = tl.arange(0, WP)
        slot = tl.load(slots_ptr + row).to(tl.int64)
        init = tl.load(init_ptr + row) != 0
        src = n + w                                                   # in (state ‖ row): WIN + n rows
        fresh = (src >= WIN) & (w < WIN)
        old = (src < WIN) & (w < WIN) & init
        x = tl.load(c_ptr + (s + src - WIN).to(tl.int64)[:, None] * C + c[None, :],
                    mask=fresh[:, None] & (c < C)[None, :], other=0.0)
        x += tl.load(state_ptr + (slot * WIN + tl.minimum(src, WIN - 1))[:, None] * C + c[None, :],
                     mask=old[:, None] & (c < C)[None, :], other=0.0)
        tl.debug_barrier()
        tl.store(state_ptr + (slot * WIN + w)[:, None] * C + c[None, :], x, mask=(w < WIN)[:, None] & (c < C)[None, :])


def ple_rows(ids, history, slots, query_start, has_init, multipliers, sizes, offsets, ngram: int, heads_per: int,
             eos: int):
    import torch

    t, rows = len(ids), len(slots)
    out = torch.empty(t, (ngram - 1) * heads_per, dtype=torch.int64, device=ids.device)
    if t and rows:
        _ple_rows[(rows, triton.cdiv(t, TOKENS))](ids, history, slots, query_start, has_init, multipliers, sizes,
                                                  offsets, out, heads_per, ngram, eos, TOKENS)
        _ple_history[(rows,)](ids, history, slots, query_start, has_init, eos)
    return out


def ple_block(key, value, r, w_key, w_query, w_conv, conv, state, slots, query_start, has_init, streams: int,
              dilation: int, eps: float):
    """Updates `r` and `state` in place; returns C [T, n·H] f32, the conv's inputs."""
    import torch

    t, width = r.shape
    rows, taps, window = len(slots), conv.shape[1], state.shape[1]
    if t == 0 or rows == 0:
        return torch.zeros(t, width, dtype=torch.float32, device=r.device)
    gated = torch.empty(t, width, dtype=torch.float32, device=r.device)
    normed = torch.empty_like(gated)
    _ple_gate[(t,)](key, value, r, w_key, w_query, w_conv, gated, normed, width // streams, eps, key.stride(0),
                    value.stride(0), streams, CHANNELS)
    _ple_conv[(rows, triton.cdiv(t, TOKENS), triton.cdiv(width, CHANNELS // 4))](
        r, gated, normed, conv.contiguous(), state, slots, query_start, has_init, width, taps, dilation, window,
        TOKENS, CHANNELS // 4)
    _ple_state[(rows, triton.cdiv(width, CHANNELS))](normed, state, slots, query_start, has_init, width, window,
                                                     triton.next_power_of_2(window), CHANNELS)
    return normed
