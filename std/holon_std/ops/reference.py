"""Pure-torch reference implementations (oracles) for the std ops. fp32 internally, cast back to the input dtype."""
from __future__ import annotations

import torch


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float, offset: float = 0.0) -> torch.Tensor:
    """`x / rms(x) * (weight + offset)`; offset=1.0 gives zero-centred norms."""
    x32 = x.float()
    y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (y * (weight.float() + offset)).to(x.dtype)


def rope_neox(x: torch.Tensor, positions: torch.Tensor, theta: float, rotary_dim: int | None = None) -> torch.Tensor:
    """Rotate the first `rotary_dim` dims of `x` [tokens, heads, head_dim] in neox (half-split) layout."""
    rd = rotary_dim or x.shape[-1]
    inv_freq = 1.0 / (theta ** (torch.arange(0, rd, 2, device=x.device, dtype=torch.float32) / rd))
    angles = positions.float()[:, None] * inv_freq[None, :]
    cos, sin = angles.cos()[:, None, :], angles.sin()[:, None, :]
    rot, rest = x[..., :rd].float(), x[..., rd:]
    a, b = rot[..., : rd // 2], rot[..., rd // 2 :]
    out = torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1).to(x.dtype)
    return torch.cat((out, rest), dim=-1)


def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    return torch.nn.functional.linear(x, weight, bias)

