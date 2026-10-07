"""Op contracts std.ops owns, and their reference kernels (lowest priority, any device and dtype).

- `rmsnorm@1`: instance `(x [..., dim], weight) -> y`. Facts: dim, eps, offset, dtype.
- `add_rmsnorm@1`: instance `(x, residual, weight) -> (normed, residual + x)`; may update `x` and `residual` in
  place (callers pass tensors they no longer need). Facts as rmsnorm.
- `rope@1`: instance `(q [T, Hq, D], k [T, Hk, D], positions [T]) -> (q, k)`, neox (half-split) rotation of the
  first `rotary_dim` dims; may rotate in place. Facts: head_dim, rotary_dim, theta, max_positions, dtype.
Kernel instances are stateless across steps and safe to call inside CUDA graphs.
"""
from __future__ import annotations

from . import reference as ref

RMSNORM, ADD_RMSNORM, ROPE = "rmsnorm@1", "add_rmsnorm@1", "rope@1"


class RMSNorm:
    def __init__(self, ctx, *, dim: int, eps: float, offset: float, dtype: str) -> None:
        self.eps, self.offset = eps, offset

    def __call__(self, x, weight):
        return ref.rmsnorm(x, weight, self.eps, self.offset)


class AddRMSNorm(RMSNorm):
    def __call__(self, x, residual, weight):
        residual = residual + x
        return ref.rmsnorm(residual, weight, self.eps, self.offset), residual


class Rope:
    def __init__(self, ctx, *, head_dim: int, rotary_dim: int, theta: float, max_positions: int, dtype: str) -> None:
        self.theta, self.rotary_dim = theta, rotary_dim

    def __call__(self, q, k, positions):
        return (ref.rope_neox(q, positions, self.theta, self.rotary_dim),
                ref.rope_neox(k, positions, self.theta, self.rotary_dim))
