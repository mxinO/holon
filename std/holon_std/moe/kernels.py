"""Op contracts std.moe owns (docs/m02-moe.md D2). Their oracles are in `reference.py`; kernels come from kernel
plugins. Kernel instances are stateless across steps and safe to call inside CUDA graphs; any `T` ≥ 0.

- `moe_topk@1`: instance `(logits [T, E] f32) -> (weights [T, k] f32, ids [T, k] int32)`, both contiguous: softmax
  over E in f32, the k largest (ties to the lower index; NaN as the lowest value), divided by their sum (clamped at
  6.1e-5) when `renormalize`. Ids are unique and in range in every row. Facts: experts, top_k, renormalize.
- `linear_experts@1`: instance `(x [T, R, in], parts, ids [T, k]) -> y [T, k, out_y]` of `out_dtype`: row j of token
  t applies expert `ids[t, j]` to `x[t, 0]` (R = 1) or `x[t, j]` (R = k). `parts` holds the stack's tensors by role,
  as `QTensor.parts` (`{"blocks": [E·out, row_bytes]}` for ggml layouts; `{"weight": [E, out, in]}` for
  `dense/<dtype>`). With `glu = "swiglu"` the stack is a merged gate/up stack (per expert `out/2` gate rows, then
  `out/2` up rows) and `y = silu(gate) · up` in f32 (out_y = out/2); with `glu = "none"`, out_y = out. A dense
  layout's activation operand is rounded to the weight dtype before the product, with f32 accumulation. Facts:
  layout, experts, in, out, top_k, rows_per_token, glu, act_dtype, out_dtype.
- `moe_combine@1`: instance `(y [T, k, H] of y_dtype, weights [T, k] f32, out [T, H])`: writes `Σ_j w[t, j] · y[t,
  j]`, accumulated in f32 in slot order, into `out` (of `dtype`). Facts: hidden, top_k, dtype, y_dtype.
"""
from __future__ import annotations

MOE_TOPK, LINEAR_EXPERTS, MOE_COMBINE = "moe_topk@1", "linear_experts@1", "moe_combine@1"
