"""Op contracts std.ops owns. Their oracles are in `reference.py`; kernels come from kernel plugins.

- `rmsnorm@1`: instance `(x [..., dim], weight) -> y`. Facts: dim, eps, offset, dtype.
- `add_rmsnorm@1`: instance `(x, residual, weight) -> (normed, residual + x)`; may update `x` and `residual` in
  place (callers pass tensors they no longer need). Facts as rmsnorm.
- `rope@1`: instance `(q [T, Hq, D], k [T, Hk, D], positions [T]) -> (q, k)`, neox (half-split) rotation of the
  first `rotary_dim` dims; may rotate in place. Facts: head_dim, rotary_dim, theta, max_positions, dtype.
Kernel instances are stateless across steps and safe to call inside CUDA graphs.
"""
from __future__ import annotations

RMSNORM, ADD_RMSNORM, ROPE = "rmsnorm@1", "add_rmsnorm@1", "rope@1"
