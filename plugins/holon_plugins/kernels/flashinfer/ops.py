"""FlashInfer kernels for the small per-token ops: rmsnorm, add_rmsnorm, rope (std.ops) and silu_mul (std.ffn).

Op contracts are documented next to their references (std.ops.kernels, std.ffn.kernels). FlashInfer's norms take
2-D input, so leading dimensions are flattened. RoPE uses a cos/sin cache computed with the reference's formula, so
the angles match the oracle; it is built once per instance on first use (outside any graph capture, because every
graph key runs eagerly once before it is captured).
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
        self.head_dim, self.rotary_dim, self.theta, self.max_positions = head_dim, rotary_dim, theta, max_positions
        self._cache: torch.Tensor | None = None

    def _cos_sin(self, device: torch.device) -> torch.Tensor:
        if self._cache is None:
            rd = self.rotary_dim
            inv_freq = 1.0 / (self.theta ** (torch.arange(0, rd, 2, device=device, dtype=torch.float32) / rd))
            angles = torch.arange(self.max_positions, device=device, dtype=torch.float32)[:, None] * inv_freq[None, :]
            self._cache = torch.cat([angles.cos(), angles.sin()], dim=-1)
        return self._cache

    def __call__(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor):
        q, k = q.contiguous(), k.contiguous()
        flashinfer.apply_rope_with_cos_sin_cache_inplace(positions, q.view(len(q), -1), k.view(len(k), -1),
                                                         self.head_dim, self._cos_sin(q.device), is_neox=True)
        return q, k


class SiluMul:
    def __init__(self, ctx, *, inter: int, dtype: str) -> None:
        pass

    def __call__(self, gate_up: torch.Tensor) -> torch.Tensor:
        return flashinfer.silu_and_mul(gate_up)
