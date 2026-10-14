"""Op contracts std.ops owns. Their oracles are in `reference.py`; kernels come from kernel plugins.

- `rmsnorm@1`: instance `(x [..., dim], weight) -> y`. Facts: dim, eps, offset, dtype.
- `add_rmsnorm@1`: instance `(x, residual, weight) -> (normed, residual + x)`; may update `x` and `residual` in
  place (callers pass tensors they no longer need). Facts as rmsnorm.
- `rope@1`: instance `(q [T, Hq, D], k [T, Hk, D], positions [T]) -> (q, k)`, neox (half-split) rotation of the
  first `rotary_dim` dims; may rotate in place. Facts: head_dim, rotary_dim, theta, max_positions, dtype.
- `qlinear@2`: instance `(x [T, in], parts) -> y [T, out]` with `y = x @ W.T`, `W` the dequantized weight of the
  layout and `parts` its tensors by role, as `QTensor.parts` (`{"blocks": [out, row_bytes] uint8}` for ggml layouts);
  any `T` ≥ 0; `x` of `act_dtype`, `y` of `out_dtype`. Facts: layout (`"<scheme elem>/<layout name>"` as adopted,
  e.g. `"ggml.q4_k/gguf.native"`), in, out, act_dtype, out_dtype, act_quant (`ACT_QUANT`: the lowest activation
  precision the kernel may use, the deployment's `quant.act` plugin option, default `a8`; docs/m06-lowp.md D3).
  An instance may define `prepare(q: QTensor) -> QTensor`, called once by the unit after binding: the weight in the
  layout the instance reads, losslessly, its first part the primary one, without modifying `q` (a tied head reads the
  embedding's parts); the unit keeps only what it returns. Without it the instance reads the parts as adopted.
  A spec may name the activation mode each of its paths computes in, `config["act"]` (path -> mode); with the
  deployment option `chunk_with_decode` > 0 and no drafter (mixed steps: one call's rows take one path) the unit
  refuses, at bind, a spec whose modes differ (docs/m07-perf.md §3.1).
- `qembedding@1`: instance `(ids [T] int64, blocks [vocab, row_bytes] uint8) -> [T, hidden]`: rows of the
  dequantized table, of `dtype`. Facts: layout, vocab, hidden, dtype.
Kernel instances are stateless across steps and safe to call inside CUDA graphs.
"""
from __future__ import annotations

RMSNORM, ADD_RMSNORM, ROPE = "rmsnorm@1", "add_rmsnorm@1", "rope@1"
QLINEAR, QEMBEDDING = "qlinear@2", "qembedding@1"
ACT_QUANT = ("a4", "a8")            # `act_quant` of quantized weights (`a16` is not offered for them)
