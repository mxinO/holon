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
              inject: torch.Tensor | None, streams: int, eps: float, dtype: torch.dtype
              ) -> tuple[torch.Tensor, torch.Tensor | None]:
    rn = stream_norm(r, norm, streams, eps)
    u = torch.sigmoid(F.silu(rn @ down.float().T / streams) @ up.float().T)
    x = (u * rn).reshape(len(r), streams, r.shape[1] // streams).mean(1).to(dtype)
    w = None if inject is None else 2 * torch.sigmoid(rn @ inject.float().T / streams)
    return x, w


def hc_gr_combine(r: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> None:
    streams = w.shape[1]
    r.view(len(r), streams, r.shape[1] // streams).add_(w[:, :, None] * y.float()[:, None, :])
