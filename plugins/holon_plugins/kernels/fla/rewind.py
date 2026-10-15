"""Our Triton kernel for `gdn_rewind@1` (op contract: std.mixers.gdn; docs/m07-perf.md D5): one launch per verify step
for every GDN layer of a group — the saved gates masked past each row's accepted tokens, and the conv state at the
accepted end gathered from window ‖ saved x straight into the pool. Data movement only (the oracle's values exactly);
nothing is read back to the host, so it captures into the verify step's graph."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

MASKED = tl.constexpr(-1e4)     # the contract's gates past the accepted tokens
BLOCK_C = 256                   # channels per program


@triton.jit
def _rewind(state_ptr, window_ptr, x_ptr, a_ptr, b_ptr, slots_ptr, accepted_ptr, W, C, H, stride_sl, stride_su,
            stride_wl, stride_xl, stride_al, K1: tl.constexpr, KP: tl.constexpr, BLOCK_C: tl.constexpr,
            BLOCK_W: tl.constexpr, BLOCK_H: tl.constexpr):
    """Program (layer, row, channel block): the row's K-1 conv inputs at its accepted end; the first channel block
    also masks the row's gates."""
    layer, r, cb = tl.program_id(0).to(tl.int64), tl.program_id(1), tl.program_id(2)
    unit = tl.load(slots_ptr + r).to(tl.int64)
    keep = 1 + tl.load(accepted_ptr + unit)                   # the row's accepted tokens (its input token included)
    c = cb * BLOCK_C + tl.arange(0, BLOCK_C)
    i = tl.arange(0, KP)
    src = keep + i                                            # position in (window K-1 ‖ the row's W tokens)
    ok = (c < C)[:, None] & (i < K1)[None, :]
    from_x = (src >= K1)[None, :]
    win = tl.load(window_ptr + layer * stride_wl + r * C * K1 + c[:, None] * K1 + src[None, :], mask=ok & ~from_x)
    tok = (r * W + src - K1).to(tl.int64)
    xs = tl.load(x_ptr + layer * stride_xl + tok[None, :] * C + c[:, None], mask=ok & from_x)
    tl.store(state_ptr + layer * stride_sl + unit * stride_su + c[:, None] * K1 + i[None, :], tl.where(from_x, xs, win),
             mask=ok)
    if cb == 0:
        w = tl.arange(0, BLOCK_W)
        h = tl.arange(0, BLOCK_H)
        past = ((w >= keep) & (w < W))[:, None] & (h < H)[None, :]
        at = layer * stride_al + (r * W + w)[:, None] * H + h[None, :]
        tl.store(a_ptr + at, tl.full([BLOCK_W, BLOCK_H], MASKED, tl.float32).to(a_ptr.dtype.element_ty), mask=past)
        tl.store(b_ptr + at, tl.full([BLOCK_W, BLOCK_H], MASKED, tl.float32).to(b_ptr.dtype.element_ty), mask=past)


class GDNRewind:
    def __init__(self, ctx, *, channels: int, kernel_size: int, heads_v: int, dtype: str) -> None:
        pass

    @torch.no_grad()
    def __call__(self, state, window, x, a, b, slots, accepted, width: int) -> None:
        layers, _, c, k1 = state.shape
        rows = len(slots)
        if not (state.stride(-1) == 1 and state.stride(-2) == k1 and state.stride(1) == c * k1
                and window.is_contiguous() and x.is_contiguous() and a.is_contiguous() and b.is_contiguous()):
            raise ValueError("gdn_rewind takes contiguous buffers and units")
        if rows:
            _rewind[(layers, rows, triton.cdiv(c, BLOCK_C))](
                state, window, x, a, b, slots, accepted, width, c, a.shape[2], state.stride(0), state.stride(1),
                window.stride(0), x.stride(0), a.stride(0), K1=k1, KP=triton.next_power_of_2(k1), BLOCK_C=BLOCK_C,
                BLOCK_W=triton.next_power_of_2(width), BLOCK_H=triton.next_power_of_2(a.shape[2]))


def supports(f: dict) -> bool:
    return f["dtype"] == "bfloat16" and 2 <= f["kernel_size"] <= 8          # the L0's dtype
