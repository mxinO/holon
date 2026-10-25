"""Oracles of the gated-residual hyper-connection ops (`kernels.py`): float32 throughout, HF's `GatedResidual`
(transformers 5.19.0, `models/qwen4_exp/modeling_qwen4_exp.py:995-1023`) on a flattened residual."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def stream_norm(r: torch.Tensor, weight: torch.Tensor, streams: int, eps: float) -> torch.Tensor:
    """R [T, n·H] normed per stream, times (1 + weight)."""
    x = r.float().reshape(len(r), streams, r.shape[1] // streams)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.reshape(r.shape) * (1.0 + weight.float())


def hc_gr_mix(r: torch.Tensor, norm: torch.Tensor, down: torch.Tensor, up: torch.Tensor,
              inject: torch.Tensor | None, y_prev: torch.Tensor | None, w_prev: torch.Tensor | None, streams: int,
              eps: float, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor | None]:
    if y_prev is not None:
        hc_gr_combine(r, y_prev, w_prev)
    rn = stream_norm(r, norm, streams, eps)
    u = torch.sigmoid(F.silu(rn @ down.float().T / streams) @ up.float().T)
    x = (u * rn).reshape(len(r), streams, r.shape[1] // streams).mean(1).to(dtype)
    w = None if inject is None else 2 * torch.sigmoid(rn @ inject.float().T / streams)
    return x, w


def hc_gr_combine(r: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> None:
    streams = w.shape[1]
    r.view(len(r), streams, r.shape[1] // streams).add_(w[:, :, None] * y.float()[:, None, :])


def hc_mhc_mix(r: torch.Tensor, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor,
               y_prev: torch.Tensor | None, post_prev: torch.Tensor | None, comb_prev: torch.Tensor | None,
               streams: int, eps: float, hc_eps: float, iters: int, dtype: torch.dtype
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """HF's `Glm5NextTextHyperConnection` (transformers 5.19.0, `models/glm5_next/modeling_glm5_next.py:286-330`) on a
    flattened residual, after the pending write-back."""
    if y_prev is not None:
        hc_mhc_writeback(r, y_prev, post_prev, comb_prev)
    n, t = streams, len(r)
    x = r.float()
    m = F.linear(x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps), fn.float())
    b, s = base.float(), scale.float()
    pre = torch.sigmoid(m[:, :n] * s[0] + b[:n]) + hc_eps
    post = 2 * torch.sigmoid(m[:, n:2 * n] * s[1] + b[n:2 * n])
    comb = torch.softmax(m[:, 2 * n:].view(t, n, n) * s[2] + b[2 * n:].view(n, n), dim=-1) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    out = (pre[:, :, None] * x.view(t, n, r.shape[1] // n)).sum(dim=1).to(dtype)
    return out, post, comb


def hc_mhc_writeback(r: torch.Tensor, y: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> None:
    """In place: R'_d = post_d · y + Σ_s comb[s, d] · R_s (HF's `post · y + combᵀ @ R`)."""
    n = post.shape[1]
    rs = r.view(len(r), n, r.shape[1] // n)
    rs.copy_(post[:, :, None] * y.float()[:, None, :] + torch.matmul(comb.transpose(-1, -2), rs))
