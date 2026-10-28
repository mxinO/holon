"""Reference for std.ffn: SwiGLU = down(silu(gate(x)) * up(x)); the projections are std.ops.linear units."""
from __future__ import annotations

import torch


def silu_mul(gate: torch.Tensor, up: torch.Tensor, limit: float = 0.0) -> torch.Tensor:
    """silu(gate) · up; with `limit` L > 0 (`glu_limit`) silu(min(gate, L)) · clamp(up, −L, L) (docs/dev/v1-glm.md
    D6)."""
    if limit:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    return torch.nn.functional.silu(gate.float()).to(gate.dtype) * up
