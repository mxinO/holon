"""Reference for std.ffn: SwiGLU = down(silu(gate(x)) * up(x)); the projections are std.ops.linear units."""
from __future__ import annotations

import torch


def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(gate.float()).to(gate.dtype) * up
