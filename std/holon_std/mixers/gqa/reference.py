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


def paged_attention(q: torch.Tensor, kv: torch.Tensor, block_table: torch.Tensor, rows: torch.Tensor,
                    lens: torch.Tensor, query_start: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Attention of new rows over paged KV.

    q [T, Hq, D]; kv [units, 2, page, Hkv, D] (one layer); `block_table` [slots, max_pages] unit ids; batch row b
    uses table row `rows[b]` and holds `lens[b]` tokens, including this step's. Each query attends to keys at
    positions ≤ its own.
    """
    rep, page = q.shape[1] // kv.shape[3], kv.shape[2]
    out = torch.empty_like(q)
    bounds = zip(query_start[:-1].tolist(), query_start[1:].tolist(), rows.tolist(), lens.tolist(), strict=True)
    for s, e, row, n in bounds:
        units = block_table[row, : -(-n // page)].long()
        k = kv[units, 0].reshape(-1, *kv.shape[3:])[:n].float().repeat_interleave(rep, dim=1)
        v = kv[units, 1].reshape(-1, *kv.shape[3:])[:n].float().repeat_interleave(rep, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q[s:e].float(), k) / q.shape[-1] ** 0.5
        future = torch.arange(n, device=q.device)[None, :] > positions[s:e, None]
        probs = scores.masked_fill(future, float("-inf")).softmax(-1)
        out[s:e] = torch.einsum("hqk,khd->qhd", probs, v).to(q.dtype)
    return out
