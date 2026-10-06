"""Reference grouped-query attention: causal softmax(q k^T / sqrt(d)) v per sequence, fp32 math."""
from __future__ import annotations

import torch


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, query_start: torch.Tensor) -> torch.Tensor:
    """q [T, Hq, D], k/v [T, Hkv, D]; sequences are the row ranges given by `query_start` [B+1]."""
    rep = q.shape[1] // k.shape[1]
    out = torch.empty_like(q)
    for s, e in zip(query_start[:-1].tolist(), query_start[1:].tolist(), strict=True):
        qs, ks, vs = q[s:e].float(), k[s:e].float(), v[s:e].float()
        ks, vs = ks.repeat_interleave(rep, dim=1), vs.repeat_interleave(rep, dim=1)
        scores = torch.einsum("qhd,khd->hqk", qs, ks) / q.shape[-1] ** 0.5
        mask = torch.ones(e - s, e - s, dtype=torch.bool, device=q.device).triu(1)
        probs = scores.masked_fill(mask, float("-inf")).softmax(-1)
        out[s:e] = torch.einsum("hqk,khd->qhd", probs, vs).to(q.dtype)
    return out
