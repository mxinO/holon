"""`linear_rows@1` (op contract: std.ops): `x @ weight.T` for a dense bf16 / fp16 weight and 2 to 16 rows, float32
sums in a fixed order — each program owns `BN` output columns of every row and walks `in` in `BK` steps —, rounded
once. cuBLAS runs such a GEMM (a GDN layer's a/b projection at a verify pass's 4 rows) as split-K in ≈ 15 µs; this
reads the weight once across 32-64 programs (FN §23.8)."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

BK = 512
OUT_MIN, OUT_MAX = 16, 256      # narrow outputs: cuBLAS's few-row GEMMs are fine past them, its GEMV below them


@triton.jit
def _rows(x, w, y, M, N, SW, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    m = tl.arange(0, BM)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        xb = tl.load(x + m[:, None] * K + k[None, :], mask=(m < M)[:, None], other=0).to(tl.float32)
        wb = tl.load(w + n[:, None] * SW + k[None, :], mask=(n < N)[:, None], other=0).to(tl.float32)
        acc += tl.sum(xb[:, None, :] * wb[None, :, :], axis=2)
    tl.store(y + m[:, None] * N + n[None, :], acc.to(y.dtype.element_ty), mask=(m < M)[:, None] & (n < N)[None, :])


def linear_rows(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """`w` [out, in] read with its row stride (e.g. rows of a wider merged weight); its elements must be dense."""
    m, k = x.shape
    n = w.shape[0]
    if w.shape[1] != k or w.stride(1) != 1:
        raise ValueError(f"linear_rows: weight {tuple(w.shape)} strides {w.stride()}: [out, {k}] with dense rows")
    y = torch.empty(m, n, dtype=x.dtype, device=x.device)
    bn = 2
    _rows[(triton.cdiv(n, bn),)](x.contiguous(), w, y, m, n, w.stride(0), k, triton.next_power_of_2(m), bn, BK,
                                 num_warps=4)
    return y
