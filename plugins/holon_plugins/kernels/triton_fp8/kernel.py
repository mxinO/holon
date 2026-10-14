"""FP8 (e4m3, 128 × 128 block scales) matmuls in Triton (docs/m06-lowp.md D12, §4.6): `y = x @ W.T` for `W` stored as
e4m3 bits `[out, in]` uint8 and one f32 multiplier per 128 × 128 block, `scale [out / 128, in / 128]`.

- `decode` (W8A16, split-K, M ≤ 16): the weight bytes are decoded to bf16 exactly by bit operations (no fp8 type, so it
  also runs below sm_89) and multiplied with the bf16 activations, f32 accumulation, each 128-wide K step's product
  scaled by its block's multiplier. One program per (column block, K split) streams its weights once: one row as a
  vector product (products in f32, summed per K step), 2–16 rows as one 16-row `tl.dot` tile. With more than one split
  the f32 partials are summed by a second kernel in split order (deterministic, no atomics).

Configurations are pinned per GPU class and shape (no autotuning): the summation order is fixed for a shape, a class
and a row kind (one row, or 2–16).
"""
from __future__ import annotations

import dataclasses

import torch
import triton
import triton.language as tl

BLOCK = 128                      # the scales' block (both dimensions); also every kernel's K step
DECODE_ROWS = 16                 # decode's largest M: tl.dot's minimum tile, so 2–16 rows are one tile


@dataclasses.dataclass(frozen=True)
class Decode:
    """A decode launch: output columns per program, K splits, warps, pipeline stages."""
    block_n: int = 64
    splits: int = 1
    num_warps: int = 4
    num_stages: int = 4


# Pinned per (GPU class, out, in, rows: 1 or 16) from `bench/fp8_decode_gate.py sweep` (the fastest at M = 1 and at
# M = 16); other shapes and classes take `Decode()` with four splits where K has them.
DECODE_CONFIGS: dict[tuple[str, int, int, int], Decode] = {}


def decode_config(gpu: str, n: int, k: int, rows: int) -> Decode:
    return DECODE_CONFIGS.get((gpu, n, k, rows)) or Decode(splits=4 if k // BLOCK % 4 == 0 else 1)


@triton.jit
def e4m3_to_bf16(w):
    """e4m3 bits (uint8) → bf16, exactly: a normal's exponent rebiased (7 → 127) and mantissa widened in place; a
    subnormal (exponent field 0: m · 2^-9) through f32, where it is exact; the sign bit carried over."""
    w = w.to(tl.int32)
    mag = w & 0x7F
    normal = (mag << 4) + (120 << 7)
    sub = (mag.to(tl.float32) * 0.001953125).to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32)
    bits = tl.where(mag < 8, sub, normal) | ((w & 0x80) << 8)
    return bits.to(tl.int16).to(tl.bfloat16, bitcast=True)


@triton.jit
def _decode(x_ptr, w_ptr, s_ptr, y_ptr, m, n, k, stride_x, blocks_per_split,
            BLOCK_N: tl.constexpr, ROWS: tl.constexpr, K_STEP: tl.constexpr, NUM_STAGES: tl.constexpr):
    pid_n, split = tl.program_id(0), tl.program_id(1)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.arange(0, K_STEP)
    s_row = s_ptr + (cols // K_STEP) * (k // K_STEP)               # each column's row of block multipliers
    first = split * blocks_per_split
    if ROWS == 1:
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for kb in tl.range(first, first + blocks_per_split, num_stages=NUM_STAGES):
            a = tl.load(x_ptr + kb * K_STEP + ks).to(tl.float32)
            w = e4m3_to_bf16(tl.load(w_ptr + cols[:, None].to(tl.int64) * k + (kb * K_STEP + ks)[None, :]))
            acc += tl.sum(w.to(tl.float32) * a[None, :], axis=1) * tl.load(s_row + kb)
        tl.store(y_ptr + split * n + cols, acc.to(y_ptr.dtype.element_ty))
    else:
        rows = tl.arange(0, ROWS)
        row_ok = rows < m
        acc = tl.zeros((ROWS, BLOCK_N), dtype=tl.float32)
        for kb in tl.range(first, first + blocks_per_split, num_stages=NUM_STAGES):
            a = tl.load(x_ptr + rows[:, None] * stride_x + (kb * K_STEP + ks)[None, :], mask=row_ok[:, None],
                        other=0.0)
            w = e4m3_to_bf16(tl.load(w_ptr + cols[None, :].to(tl.int64) * k + (kb * K_STEP + ks)[:, None]))
            acc += tl.dot(a, w) * tl.load(s_row + kb)[None, :]
        out = y_ptr + split * m * n + rows[:, None] * n + cols[None, :]
        tl.store(out, acc.to(y_ptr.dtype.element_ty), mask=row_ok[:, None])


@triton.jit
def _sum_splits(p_ptr, y_ptr, size, splits: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ok = i < size
    acc = tl.load(p_ptr + i, mask=ok, other=0.0)
    for s in tl.static_range(1, splits):
        acc += tl.load(p_ptr + s * size + i, mask=ok, other=0.0)
    tl.store(y_ptr + i, acc.to(y_ptr.dtype.element_ty), mask=ok)


def decode(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, out_dtype: torch.dtype,
           cfg: Decode) -> torch.Tensor:
    """`y [M, out]` of `out_dtype` for `x [M ≤ 16, in]` bf16 (unit stride along `in`)."""
    m, k = x.shape
    n = weight.shape[0]
    y = torch.empty(m, n, dtype=out_dtype, device=x.device)
    if m == 0:
        return y
    partial = y if cfg.splits == 1 else torch.empty(cfg.splits, m, n, dtype=torch.float32, device=x.device)
    _decode[(n // cfg.block_n, cfg.splits)](x, weight, scale, partial, m, n, k, x.stride(0), k // BLOCK // cfg.splits,
                                            cfg.block_n, 1 if m == 1 else DECODE_ROWS, BLOCK, cfg.num_stages,
                                            num_warps=cfg.num_warps)
    if cfg.splits > 1:
        _sum_splits[(triton.cdiv(m * n, 1024),)](partial, y, m * n, cfg.splits, 1024)
    return y
