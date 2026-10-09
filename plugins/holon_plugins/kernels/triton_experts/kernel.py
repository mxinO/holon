"""A grouped GEMM over dense expert stacks (docs/m02-moe.md D8): rows sorted by expert on the device, each expert's
rows padded to whole row blocks, one program per (row block, column block) over a grid fixed by the shapes alone —
so a call is capture-safe and replays for any routing. Operands are the weight dtype (bf16 × bf16, f32
accumulation: TF32 never enters), configurations are pinned (no autotuning: the summation order is fixed)."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

BM, BN, BK = 32, 64, 64                     # pinned: row block, column block, reduction block


def routing(ids: torch.Tensor, experts: int) -> tuple[torch.Tensor, torch.Tensor]:
    """For the padded layout, on the device and without host syncs: each row block's expert [NB], and each padded
    position's (token, slot) row [NB·BM], −1 for padding. NB = ceil((T·k + E·(BM − 1)) / BM) covers any routing."""
    flat = ids.reshape(-1).long()
    rows = flat.numel()
    nb = (rows + experts * (BM - 1) + BM - 1) // BM
    order = torch.argsort(flat, stable=True)
    bounds = torch.searchsorted(flat[order], torch.arange(experts + 1, device=ids.device))   # rows per expert
    counts = bounds[1:] - bounds[:-1]
    padded = torch.zeros(experts + 1, dtype=torch.long, device=ids.device)
    padded[1:] = torch.cumsum((counts + BM - 1) // BM * BM, 0)
    pos = torch.arange(nb * BM, device=ids.device)
    expert = torch.searchsorted(padded[1:], pos, right=True).clamp(max=experts - 1)
    local = pos - padded[expert]
    valid = (local < counts[expert]) & (pos < padded[-1])
    src = (bounds[expert] + local).clamp(max=max(rows - 1, 0))
    row = torch.where(valid, order[src] if rows else src, -1).to(torch.int32)
    return expert[::BM].to(torch.int32).contiguous(), row.contiguous()


@triton.jit
def _grouped(x_ptr, w_ptr, y_ptr, rows_ptr, expert_ptr, k_slots: tl.constexpr, rows_per_token: tl.constexpr,
             n_in, n_out_y, stride_xr, stride_we, stride_wo, stride_yr, swiglu: tl.constexpr,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    b, n = tl.program_id(0), tl.program_id(1)
    rows = tl.load(rows_ptr + b * BLOCK_M + tl.arange(0, BLOCK_M))
    live = rows >= 0
    if tl.max(live.to(tl.int32), 0) == 0:                    # a block past the used ones, or all padding
        return
    expert = tl.load(expert_ptr + b)
    x_row = tl.where(live, rows // k_slots if rows_per_token == 1 else rows, 0)
    cols = n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_ok = cols < n_out_y
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w = w_ptr + expert.to(tl.int64) * stride_we
    for k0 in range(0, n_in, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        k_ok = ks < n_in
        a = tl.load(x_ptr + x_row[:, None].to(tl.int64) * stride_xr + ks[None, :],
                    mask=live[:, None] & k_ok[None, :], other=0.0)
        g = tl.load(w + cols[None, :].to(tl.int64) * stride_wo + ks[:, None], mask=col_ok[None, :] & k_ok[:, None],
                    other=0.0)
        acc = tl.dot(a, g, acc)
        if swiglu:                                           # the up rows sit n_out_y rows after the gate rows
            u = tl.load(w + (cols[None, :] + n_out_y).to(tl.int64) * stride_wo + ks[:, None],
                        mask=col_ok[None, :] & k_ok[:, None], other=0.0)
            acc_up = tl.dot(a, u, acc_up)
    if swiglu:
        acc = acc / (1.0 + tl.exp(-acc)) * acc_up
    tl.store(y_ptr + rows[:, None].to(tl.int64) * stride_yr + cols[None, :], acc,
             mask=live[:, None] & col_ok[None, :])


def grouped(x: torch.Tensor, weight: torch.Tensor, ids: torch.Tensor, glu: str) -> torch.Tensor:
    """`y [T, k, out_y]` f32 for `x [T, R, in]` (rounded to the weight dtype here: the dense contract) over the stack
    `weight [E, out, in]`."""
    if weight.stride(2) != 1:
        raise ValueError(f"the expert stack's rows must be contiguous (strides {weight.stride()})")
    t, k = ids.shape
    out_y = weight.shape[1] // 2 if glu == "swiglu" else weight.shape[1]
    y = torch.empty(t, k, out_y, dtype=torch.float32, device=x.device)
    if t == 0:
        return y
    xs = x.reshape(-1, x.shape[-1]).to(weight.dtype).contiguous()
    expert, rows = routing(ids, weight.shape[0])
    grid = (expert.numel(), triton.cdiv(out_y, BN))
    _grouped[grid](xs, weight, y, rows, expert, k, x.shape[1], weight.shape[2], out_y, xs.stride(0),
                   weight.stride(0), weight.stride(1), out_y, glu == "swiglu", BM, BN, BK)
    return y
