"""The op contracts of the gated-residual hyper-connections (docs/v1-flash-next.md D3, D13′: named for this form; GLM's
mHC, with a stream-mixing matrix, will add siblings). Their oracles are in `reference.py`; kernels come from kernel
plugins. Kernel instances are stateless across steps and safe to call inside CUDA graphs.

The trunk carries the residual R [T, n·H] in float32: n streams of the hidden size H, stream s at columns s·H … s·H + H.
**R is complete at every block boundary:** a block (`std.hyper.gr_block@1`) applies its own last combine before it
returns, so the trunk, taps, C1's consumers, drafters and the output mixer only ever see a complete R. A pending combine
may ride into the next mix only within one block (the attention half's into the FFN half's mix), never across blocks.
Every projection is computed in float32 with IEEE products and sums (no TF32; D10), from weights of any float dtype,
each read in float32.

- `hc_gr_mix@1`: instance `(r [T, n·H] f32, norm [n·H], down [rank, n·H], up [n·H, rank], inject [n, n·H] | None,
  y_prev [T, H] | None, w_prev [T, n] f32 | None) -> (x [T, H] of `dtype`, w [T, n] f32 | None)`:
  - given `y_prev` and `w_prev` (the same block's previous half: its output and the weights its mix returned, a
    pending `hc_gr_combine`), first `R_s += w_prev_s · y_prev` in place, exactly as `hc_gr_combine@1` does, so that
    combine needs no launch of its own. `w_prev` is only read; the returned `w` is always a new tensor, so the two
    never alias (passing a mix's returned `w` back as the next mix's `w_prev` is the intended use);
  - `Rn` = R normed per stream (`x · rsqrt(mean_H(x²) + eps)`) times (1 + norm) (a zero-centred weight);
  - `x = mean_s(sigmoid(up · silu(down · Rn / n))_s ⊙ Rn_s)`, rounded once to `dtype`;
  - with `inject` (a block's mixer; None for the output mixer): `w = 2 · sigmoid(inject · Rn / n)`.
  Any T ≥ 0; `r`'s rows contiguous. Facts: streams, hidden, rank, eps, dtype, inject (bool).
- `hc_gr_combine@1`: instance `(r [T, n·H] f32, y [T, H], w [T, n] f32) -> None`: in place, `R_s += w_s · y` (y read in
  float32). Facts: streams, hidden, dtype (y's).

GLM-5.3-Flash's manifold-constrained hyper-connections (mHC; docs/v1-glm.md D3, HF's `Glm5NextTextHyperConnection`),
siblings of the above on the same residual, every operation in float32 with IEEE products and sums:

- `hc_mhc_mix@1`: instance `(r [T, n·H] f32, fn [(2 + n)·n, n·H], base [(2 + n)·n] f32, scale [3] f32, y_prev [T, H]
  | None, post_prev [T, n] f32 | None, comb_prev [T, n, n] f32 | None) -> (x [T, H] of `dtype`, post [T, n] f32,
  comb [T, n, n] f32)`:
  - given `y_prev`, `post_prev` and `comb_prev` (the same block's attention half: its mixer's output and the weights
    its mix returned, a pending `hc_mhc_writeback`), first that write-back in place, exactly as `hc_mhc_writeback@1`
    does — only within one block, so R is complete at every block boundary;
  - `m = fn · (R · rsqrt(mean_{n·H}(R²) + eps))` (the unweighted RMS over all n streams; the (2 + n)·n-wide product
    summed in an order the provider fixes, deterministic and independent of T and of the other rows);
  - `pre = σ(scale₀ · m[0:n] + base[0:n]) + hc_eps`; `post = 2 · σ(scale₁ · m[n:2n] + base[n:2n])`;
  - `comb = softmax_d(scale₂ · m[2n:] + base[2n:])` as [n (source s), n (destination d)], `+ hc_eps`, then the
    Sinkhorn normalisation exactly as HF runs it: divide by the column sums (over s) + hc_eps, then `iters` − 1 times
    by the row sums (over d) + hc_eps, then by the column sums + hc_eps;
  - `x = Σ_s pre_s · R_s`, rounded once to `dtype`.
  Any T ≥ 0; `r`'s rows contiguous. Facts: streams, hidden, eps, hc_eps, iters, dtype.
- `hc_mhc_writeback@1`: instance `(r [T, n·H] f32, y [T, H], post [T, n] f32, comb [T, n, n] f32) -> None`: in place,
  `R'_d = post_d · y + Σ_s comb[s, d] · R_s` (y read in f32; the sum over s in order, then the post term added). **The
  write-back is cross-stream**: every output column h reads all n streams' old values at h, so R is written in place
  only by a thread or CTA that owns column h of all n streams (work split over h, never over streams); any other split
  writes out of place. Facts: streams, hidden, dtype (y's).
"""
from __future__ import annotations

MIX, COMBINE = "hc_gr_mix@1", "hc_gr_combine@1"
MHC_MIX, MHC_WRITEBACK = "hc_mhc_mix@1", "hc_mhc_writeback@1"
