"""The op contracts of the gated-residual hyper-connections (docs/dev/v1-flash-next.md D3, D13′: named for this form;
GLM's
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
"""
from __future__ import annotations

MIX, COMBINE = "hc_gr_mix@1", "hc_gr_combine@1"
