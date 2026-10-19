"""The op contracts of the gated-residual hyper-connections (docs/v1-flash-next.md D3, D13′: named for this form; GLM's
mHC, with a stream-mixing matrix, will add siblings). Their oracles are in `reference.py`; kernels come from kernel
plugins. Kernel instances are stateless across steps and safe to call inside CUDA graphs.

The trunk carries the residual R [T, n·H] in float32: n streams of the hidden size H, stream s at columns s·H … s·H + H.
Every projection is computed in float32 with IEEE products and sums (no TF32; D10), from weights of any float dtype,
each read in float32.

- `hc_gr_mix@1`: instance `(r [T, n·H] f32, norm [n·H], down [rank, n·H], up [n·H, rank], inject [n, n·H] | None,
  y [T, H] | None, w [T, n] f32 | None) -> (x [T, H] of `dtype`, w [T, n] f32 | None)`:
  - given `y` and `w` (the previous block's output and its weights: a pending `hc_gr_combine`), first `R_s += w_s · y`
    in place, exactly as `hc_gr_combine@1` does, so a layer's combine needs no launch of its own;
  - `Rn` = R normed per stream (`x · rsqrt(mean_H(x²) + eps)`) times (1 + norm) (a zero-centred weight);
  - `x = mean_s(sigmoid(up · silu(down · Rn / n))_s ⊙ Rn_s)`, rounded once to `dtype`;
  - with `inject` (a block's mixer; None for the output mixer): `w = 2 · sigmoid(inject · Rn / n)`.
  Any T ≥ 0; `r`'s rows contiguous. Facts: streams, hidden, rank, eps, dtype, inject (bool).
- `hc_gr_combine@1`: instance `(r [T, n·H] f32, y [T, H], w [T, n] f32) -> None`: in place, `R_s += w_s · y` (y read in
  float32). Facts: streams, hidden, dtype (y's).
"""
from __future__ import annotations

MIX, COMBINE = "hc_gr_mix@1", "hc_gr_combine@1"
