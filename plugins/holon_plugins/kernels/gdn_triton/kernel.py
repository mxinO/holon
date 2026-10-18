"""Our Triton kernels for the Gated DeltaNet's per-token ops (M0.3 design D3): the varlen causal depthwise convolution
with its K-1 state carry, and the gated RMSNorm. Grids follow from the shapes alone and nothing is read back to the
host, so every call is capture-safe; configurations are pinned (no autotuning)."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

BT, BC, BR = 8, 256, 64         # conv: token block, channel block, rows searched per step
NORM_ROWS = 16                  # gated norm: (token, head) rows per program


@triton.jit
def _taps(x_ptr, w_ptr, state_ptr, t, c, start, unit, init, ok, c_ok, stride_x, stride_su, stride_sc,
          K: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr, STATE: tl.constexpr):
    """Σ_j w_j · input(t − (K−1) + j), in tap order; before the row's start the input is the row's state unit (if
    has_init, else zero), read only with STATE (a block whose tokens are all K−1 past their row's start skips it)."""
    acc = tl.zeros([BLOCK_T, BLOCK_C], dtype=tl.float32)
    for j in tl.static_range(K):
        p = t - (K - 1) + j                                   # the tap's token
        in_row = (p >= start)[:, None]
        v = tl.load(x_ptr + p[:, None].to(tl.int64) * stride_x + c[None, :], mask=ok & in_row,
                    other=0.0).to(tl.float32)
        if STATE:                                             # before `start`: state column K - 1 + p - start
            col = (K - 1 + p - start)[:, None]
            v += tl.load(state_ptr + unit[:, None] * stride_su + c[None, :].to(tl.int64) * stride_sc + col,
                         mask=ok & ~in_row & init[:, None], other=0.0).to(tl.float32)
        acc += v * tl.load(w_ptr + c * K + j, mask=c_ok, other=0.0).to(tl.float32)[None, :]
    return acc


@triton.jit
def _conv(x_ptr, w_ptr, state_ptr, slots_ptr, qs_ptr, init_ptr, y_ptr, T, C, R, stride_x, stride_y,
          stride_su, stride_sc, K: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr,
          BLOCK_R: tl.constexpr):
    """y for a block of tokens × channels. Each token finds its row in query_start (rows whose end ≤ t precede it;
    zero-token rows included)."""
    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    t_ok, c_ok = t < T, c < C
    row = tl.zeros([BLOCK_T], dtype=tl.int32)
    for r0 in range(0, R, BLOCK_R):
        r = r0 + tl.arange(0, BLOCK_R)
        ends = tl.load(qs_ptr + 1 + r, mask=r < R, other=2147483647)
        row += tl.sum((ends[None, :] <= t[:, None]).to(tl.int32), 1)
    row = tl.where(t_ok, row, 0)
    start = tl.load(qs_ptr + row)
    unit = tl.load(slots_ptr + row).to(tl.int64)
    init = tl.load(init_ptr + row) != 0
    ok = t_ok[:, None] & c_ok[None, :]
    if tl.sum(((t - start < K - 1) & t_ok).to(tl.int32), 0) > 0:
        acc = _taps(x_ptr, w_ptr, state_ptr, t, c, start, unit, init, ok, c_ok, stride_x, stride_su, stride_sc, K,
                    BLOCK_T, BLOCK_C, True)
    else:
        acc = _taps(x_ptr, w_ptr, state_ptr, t, c, start, unit, init, ok, c_ok, stride_x, stride_su, stride_sc, K,
                    BLOCK_T, BLOCK_C, False)
    y = acc / (1.0 + tl.exp(-acc))                            # SiLU in f32
    tl.store(y_ptr + t[:, None].to(tl.int64) * stride_y + c[None, :], y.to(y_ptr.dtype.element_ty), mask=ok)


@triton.jit
def _conv_state(x_ptr, state_ptr, slots_ptr, qs_ptr, init_ptr, C, stride_x, stride_su, stride_sc,
                K: tl.constexpr, KP: tl.constexpr, BLOCK_C: tl.constexpr):
    """state[slot] ← the last K-1 inputs of (previous state, or zeros without has_init ‖ the row's tokens); a row
    without tokens leaves its unit untouched. Runs after `_conv`, which reads the previous state."""
    r = tl.program_id(0)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    start, end = tl.load(qs_ptr + r), tl.load(qs_ptr + r + 1)
    n = end - start
    if n == 0:
        return
    unit = tl.load(slots_ptr + r).to(tl.int64)
    init = tl.load(init_ptr + r) != 0
    i = tl.arange(0, KP)
    src = n + i                                               # position in (previous K-1 ‖ row)
    ok = c[:, None] < C
    col_ok = (i < K - 1)[None, :]
    from_x = (src >= K - 1)[None, :]
    p_state = state_ptr + unit * stride_su + c[:, None].to(tl.int64) * stride_sc
    xv = tl.load(x_ptr + (start + src - (K - 1))[None, :].to(tl.int64) * stride_x + c[:, None],
                 mask=ok & col_ok & from_x, other=0.0)
    sv = tl.load(p_state + src[None, :], mask=ok & col_ok & ~from_x & init, other=0.0)
    new = tl.where(from_x, xv.to(state_ptr.dtype.element_ty), sv)
    tl.debug_barrier()                                        # every old column is read before any is overwritten
    tl.store(p_state + i[None, :], new, mask=ok & col_ok)


def causal_conv1d(x: torch.Tensor, weight: torch.Tensor, state: torch.Tensor, slots: torch.Tensor,
                  query_start: torch.Tensor, has_init: torch.Tensor) -> torch.Tensor:
    """`y [T, C]` (x's dtype) for `x [T, C]` (rows contiguous) and `weight [C, K]`; updates `state [units, C, K-1]`
    (channel rows contiguous) at the rows' slots."""
    t, c = x.shape
    k = weight.shape[1]
    r = slots.shape[0]
    if x.stride(1) != 1 or state.stride(2) != 1:
        raise ValueError(f"x and state need contiguous last dims (strides {x.stride()}, {state.stride()})")
    weight = weight.contiguous()
    y = torch.empty(t, c, dtype=x.dtype, device=x.device)
    if t == 0 or r == 0:
        return y
    _conv[(triton.cdiv(t, BT), triton.cdiv(c, BC))](
        x, weight, state, slots, query_start, has_init, y, t, c, r, x.stride(0), y.stride(0), state.stride(0),
        state.stride(1), k, BT, BC, BR, num_warps=4)
    _conv_state[(r, triton.cdiv(c, BC))](
        x, state, slots, query_start, has_init, c, x.stride(0), state.stride(0), state.stride(1), k,
        triton.next_power_of_2(k - 1), BC, num_warps=4)
    return y


@triton.jit
def _gated_rmsnorm(o_ptr, z_ptr, w_ptr, out_ptr, rows, H, D, eps, so_t, so_h, sz_t, sz_h,
                   ROWS: tl.constexpr, BLOCK_D: tl.constexpr, W_IN_F32: tl.constexpr, SIGMOID: tl.constexpr):
    """HF's RMSNormGated per (token, head) row: the norm in f32 rounded to o's dtype, times the weight (in the
    product's dtype: o's, or f32 for an f32 weight), times silu(z) (or sigmoid(z): SIGMOID) in f32."""
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    d = tl.arange(0, BLOCK_D)
    ok = (row < rows)[:, None] & (d < D)[None, :]
    tok, head = (row // H).to(tl.int64), (row % H).to(tl.int64)
    x = tl.load(o_ptr + (tok * so_t + head * so_h)[:, None] + d[None, :], mask=ok, other=0.0).to(tl.float32)
    z = tl.load(z_ptr + (tok * sz_t + head * sz_h)[:, None] + d[None, :], mask=ok, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + d, mask=d < D, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, 1) / D + eps)
    normed = (x * rstd[:, None]).to(out_ptr.dtype.element_ty).to(tl.float32)
    gated = w[None, :] * normed
    if not W_IN_F32:
        gated = gated.to(out_ptr.dtype.element_ty).to(tl.float32)
    act = 1.0 / (1.0 + tl.exp(-z))
    y = gated * (act if SIGMOID else z * act)
    tl.store(out_ptr + row.to(tl.int64)[:, None] * D + d[None, :], y.to(out_ptr.dtype.element_ty), mask=ok)


def gated_rmsnorm(o: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float, sigmoid: bool = False
                  ) -> torch.Tensor:
    """`[T, H, D]` (o's dtype) for `o, z [T, H, D]` (last dims contiguous) and `weight [D]`."""
    t, h, d = o.shape
    if o.stride(2) != 1 or z.stride(2) != 1:
        raise ValueError(f"o and z need contiguous last dims (strides {o.stride()}, {z.stride()})")
    out = torch.empty(t, h, d, dtype=o.dtype, device=o.device)
    if t == 0:
        return out
    weight = weight.contiguous()
    w_f32 = torch.promote_types(weight.dtype, o.dtype) == torch.float32 and o.dtype != torch.float32
    _gated_rmsnorm[(triton.cdiv(t * h, NORM_ROWS),)](
        o, z, weight, out, t * h, h, d, eps, o.stride(0), o.stride(1), z.stride(0), z.stride(1), NORM_ROWS,
        triton.next_power_of_2(d), w_f32, sigmoid, num_warps=4)
    return out
