"""L0 op oracle helper: a unit's output vs its `reference.py` on the same inputs, under a budget held by the judge.

Error model for an output y reduced over K terms with an fp32 accumulator and stored in `dtype` (u = unit roundoff
of `dtype`, u32 = that of fp32):

- rounding to `dtype`: the candidate's last two roundings (an unfused epilogue such as silu then mul) and the
  reference's cast back to `dtype`:                                              ≤ 3u · |y|, a hard bound
- inputs and intermediates rounded to `dtype` where the reference keeps fp32 add K independent errors of size
  u · |term|, which grow like √K · |term| ≈ rms(y):                              ~ u · rms(y)
- another summation order (tiling, split-K) is a K-step random walk over fp32 partial sums of size ≈ rms(y):
                                                                                  ~ u32 · √K · rms(y)

The last two are statistical, so with scale s = u + u32 · √K the budget has two parts:

- per element: |cand − ref| ≤ 3u · |ref| + 6s · rms(ref). The rms term covers outputs near zero (cancellation),
  where a relative bound means nothing; 6s keeps the maximum over ~10⁸ elements of Gaussian-like noise inside.
  Catches local errors (a wrong row, mask or index).
- on average: mean |cand − ref| ≤ s · rms(ref). Legitimate ops sit at 0.2–0.45 of it (matmul, softmax, silu·mul,
  rmsnorm; CPU and sm_86, cuBLAS reduced-precision reductions included); systematic errors below one ulp per
  element, such as a 1% scale error in bf16 (usage 2.0) or 0.1% in fp16 (1.6), exceed it.
"""
from __future__ import annotations

import dataclasses
import math

import torch

U32 = torch.finfo(torch.float32).eps / 2


@dataclasses.dataclass(frozen=True)
class Budget:
    rtol: float        # per element, × |ref|
    atol: float        # per element, × rms(ref)
    mean: float        # mean absolute error, × rms(ref)


@dataclasses.dataclass(frozen=True)
class Report:
    max_abs: float
    max_rel: float     # max |cand − ref| / |ref| over elements with ref ≠ 0 (informational: unbounded near zero)
    mean_abs: float
    usage: float       # the larger of (max element error / its allowance) and (mean error / its allowance)
    budget: Budget
    passed: bool


def budget(dtype: torch.dtype, reduction_len: int) -> Budget:
    """Tolerance for an output reduced over `reduction_len` terms (1 for elementwise ops) and stored in `dtype`."""
    if not dtype.is_floating_point or reduction_len < 1:
        raise ValueError(f"budget needs a float dtype and reduction_len >= 1, got {dtype}, {reduction_len}")
    u = torch.finfo(dtype).eps / 2
    s = u + U32 * math.sqrt(reduction_len)
    return Budget(rtol=3 * u, atol=6 * s, mean=s)


def check(candidate: torch.Tensor, reference: torch.Tensor, dtype: torch.dtype, reduction_len: int) -> Report:
    """Compare `candidate` with `reference` (same shape, any dtypes) under `budget(dtype, reduction_len)`.
    Non-finite values must match exactly, in place and value."""
    if candidate.shape != reference.shape:
        raise ValueError(f"shape {tuple(candidate.shape)} != reference {tuple(reference.shape)}")
    b = budget(dtype, reduction_len)
    c, r = candidate.detach().cpu().double(), reference.detach().cpu().double()
    finite = r.isfinite()
    nonfinite_ok = torch.equal(c.isfinite(), finite) and torch.equal(c[~finite].nan_to_num(), r[~finite].nan_to_num())
    c, r = c[finite], r[finite]
    if not len(r):
        return Report(0.0, 0.0, 0.0, 0.0, b, nonfinite_ok)
    err = (c - r).abs()
    rms = r.square().mean().sqrt()
    if rms == 0:
        usage = math.inf if err.any() else 0.0
    else:
        usage = max((err / (b.rtol * r.abs() + b.atol * rms)).max().item(), (err.mean() / (b.mean * rms)).item())
    nz = r != 0
    return Report(max_abs=err.max().item(), max_rel=(err[nz] / r[nz].abs()).max().item() if nz.any() else 0.0,
                  mean_abs=err.mean().item(), usage=usage, budget=b, passed=nonfinite_ok and usage <= 1)
