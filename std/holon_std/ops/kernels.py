"""Op contracts std.ops owns. Their oracles are in `reference.py`; kernels come from kernel plugins.

- `rmsnorm@1`: instance `(x [..., dim], weight) -> y`. Facts: dim, eps, offset, dtype.
- `add_rmsnorm@1`: instance `(x, residual, weight) -> (normed, residual + x)`; may update `x` and `residual` in
  place (callers pass tensors they no longer need). Facts as rmsnorm.
- `rope@1`: instance `(q [T, Hq, D], k [T, Hk, D], positions [T]) -> (q, k)`, neox (half-split) rotation of the
  first `rotary_dim` dims; may rotate in place. Facts: head_dim, rotary_dim, theta, max_positions, dtype.
- `qlinear@1`: instance `(x [T, in], blocks [out, row_bytes] uint8) -> y [T, out]` with `y = x @ W.T`, `W` the
  dequantized weight of the layout; any `T` ≥ 0; `x` of `act_dtype`, `y` of `out_dtype`. Facts: layout
  (`"<scheme elem>/<layout name>"`, e.g. `"ggml.q4_k/gguf.native"`), in, out, act_dtype, out_dtype.
- `qembedding@1`: instance `(ids [T] int64, blocks [vocab, row_bytes] uint8) -> [T, hidden]`: rows of the
  dequantized table, of `dtype`. Facts: layout, vocab, hidden, dtype.
Kernel instances are stateless across steps and safe to call inside CUDA graphs.
"""
from __future__ import annotations

RMSNORM, ADD_RMSNORM, ROPE = "rmsnorm@1", "add_rmsnorm@1", "rope@1"
QLINEAR, QEMBEDDING = "qlinear@1", "qembedding@1"
