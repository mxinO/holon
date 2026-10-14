"""References for fused kernels, composed from the oracles of the units they replace (docs/m07-perf.md D3): the std
unit that selects a fused op registers no oracle for it, so each unit's reference stays its own (unit isolation).

- `attention.qk_rope_kv@1` (std.mixers.gqa): `rmsnorm@1`, then `rope@1`, on q's and k's heads; `kv_write@1` of the
  results' k and of v.
"""
from __future__ import annotations

import torch
from holon_std.mixers.gqa import reference as gqa
from holon_std.ops import reference as ops


def qk_rope_kv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, positions: torch.Tensor, q_weight: torch.Tensor,
               k_weight: torch.Tensor, kv, unit: torch.Tensor, off: torch.Tensor, *, eps: float, offset: float,
               theta: float, rotary_dim: int) -> torch.Tensor:
    """The `attention.qk_rope_kv@1` reference: writes k and v into `kv` (a tensor or q8_0's pair, as `kv_write@1`)
    and returns q' [T, Hq, D] (q's first D values per head)."""
    d = k.shape[-1]
    q, k = (ops.rope_neox(ops.rmsnorm(x, w, eps, offset), positions, theta, rotary_dim)
            for x, w in ((q[..., :d], q_weight), (k, k_weight)))
    gqa.kv_write(k, v, kv, unit, off)
    return q.contiguous()
