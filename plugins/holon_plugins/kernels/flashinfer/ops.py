"""FlashInfer kernels for the small per-token ops: rmsnorm, add_rmsnorm, rope (std.ops) and silu_mul (std.ffn; its
clamp, `glu_limit`, a torch clamp before FlashInfer's kernel).

Op contracts are documented next to their references (std.ops.kernels, std.ffn.kernels). FlashInfer's norms take
2-D input, so leading dimensions are flattened. RoPE uses a cos/sin cache computed with the reference's formula, so
the angles match the oracle; it is a named static buffer, so every layer with the same rope shares one copy and it
is allocated (and accounted) at build time. The rope instance names it (`cos_sin`, std.ops' rope@1 contract), so a
fused kernel rotates with the same table (docs/m07-perf.md D3).
"""
from __future__ import annotations

import flashinfer
import torch


class RMSNorm:
    def __init__(self, ctx, *, dim: int, eps: float, offset: float, dtype: str) -> None:
        self.eps, self.dim = eps, dim
        self._norm = flashinfer.gemma_rmsnorm if offset == 1.0 else flashinfer.rmsnorm

    def __call__(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return self._norm(x.reshape(-1, self.dim), weight, self.eps).view(x.shape)


class AddRMSNorm:
    def __init__(self, ctx, *, dim: int, eps: float, offset: float, dtype: str) -> None:
        self.eps = eps
        self._add_norm = flashinfer.gemma_fused_add_rmsnorm if offset == 1.0 else flashinfer.fused_add_rmsnorm

    def __call__(self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor):
        self._add_norm(x, residual, weight, self.eps)       # in place: residual += x; x = norm(residual)
        return x, residual


class Rope:
    def __init__(self, ctx, *, head_dim: int, rotary_dim: int, theta: float, max_positions: int, dtype: str) -> None:
        self.head_dim, self.cos_sin = head_dim, f"rope.cos_sin[{theta},{rotary_dim},{max_positions}]"
        self.cache = ctx.static(self.cos_sin, (max_positions, rotary_dim), torch.float32)
        dims = torch.arange(0, rotary_dim, 2, device=ctx.device, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (dims / rotary_dim))
        angles = torch.arange(max_positions, device=ctx.device, dtype=torch.float32)[:, None] * inv_freq[None, :]
        self.cache.copy_(torch.cat([angles.cos(), angles.sin()], dim=-1))

    def __call__(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor):
        q, k = q.contiguous(), k.contiguous()
        flashinfer.apply_rope_with_cos_sin_cache_inplace(positions, q.view(len(q), -1), k.view(len(k), -1),
                                                         self.head_dim, self.cache, is_neox=True)
        return q, k


class SiluMul:
    """`silu_mul@1`; with `glu_limit` L (docs/v1-glm.md D6) gate and up first clamped, in one torch clamp against
    per-column bounds (gate's at L, up's at ±L: a named static, L exact in the dtype, `supports_glu_limit`), so
    FlashInfer's silu_and_mul then computes silu(min(gate, L)) · clamp(up, −L, L) from the clamped values exactly (a
    NaN kept, as the reference's clamp)."""

    def __init__(self, ctx, *, inter: int, dtype: str, glu_limit: float = 0.0) -> None:
        self.bounds = None
        if glu_limit:
            self.bounds = ctx.static(f"flashinfer.glu_bounds[{inter},{glu_limit},{dtype}]", (2, 2 * inter),
                                     getattr(torch, dtype))
            self.bounds[0, :inter], self.bounds[0, inter:], self.bounds[1] = -float("inf"), -glu_limit, glu_limit

    def __call__(self, gate_up: torch.Tensor) -> torch.Tensor:
        if self.bounds is not None:
            gate_up = torch.clamp(gate_up, self.bounds[0], self.bounds[1])
        return flashinfer.silu_and_mul(gate_up)


def supports_glu_limit(f: dict) -> bool:
    """No clamp, or one whose limit the dtype holds exactly (the clamp runs in the activations' dtype)."""
    limit = f.get("glu_limit", 0.0)
    return not limit or torch.tensor(limit, dtype=getattr(torch, f["dtype"])).item() == limit
