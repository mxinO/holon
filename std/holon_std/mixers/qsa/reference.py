"""Oracles of the QSA ops (`kernels.py`): HF's indexer and sparse attention (transformers 5.19.0,
`models/qwen4_exp/modeling_qwen4_exp.py:665-893`) on paged state, written out per token in float32 (float64 sums for
the scores), with D2's tie rule."""
from __future__ import annotations

import math

import torch


def cos_sin(max_positions: int, rotary_dim: int, theta: float) -> torch.Tensor:
    """`rope@1`'s table: per position, cos(p·θ_i) for i < rotary_dim / 2, then sin(p·θ_i),
    θ_i = theta^(−2i/rotary_dim)."""
    inv = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    angles = torch.arange(max_positions, dtype=torch.float32)[:, None] * inv[None, :]
    return torch.cat((angles.cos(), angles.sin()), dim=-1)


def rope(x: torch.Tensor, position: int, table: torch.Tensor) -> torch.Tensor:
    """x [..., D] f32 with its first `rotary_dim` dims rotated (neox) at `position`."""
    half = table.shape[1] // 2
    cos, sin = table[position, :half].to(x.device), table[position, half:].to(x.device)
    a, b = x[..., :half], x[..., half:2 * half]
    return torch.cat((a * cos - b * sin, b * cos + a * sin, x[..., 2 * half:]), dim=-1)


def stable_topk(x: torch.Tensor, k: int, dim: int = -1, largest: bool = True, sorted: bool = True):
    """D2's tie rule: a stable sort, so equal scores keep their order (the lower block first). `torch.topk`'s
    signature, so the judge's scoped wrapper of HF's indexer (`holon_judge.qwen4exp.wrap_indexers`) uses it too."""
    values, indices = torch.sort(x, dim=dim, descending=largest, stable=True)
    return torch.return_types.topk((values.narrow(dim, 0, k), indices.narrow(dim, 0, k)))


def _where(table: torch.Tensor, seq: int, position: int, page: int) -> tuple[int, int]:
    return int(table[seq, position // page]), position % page


def qsa_index(k: torch.Tensor, raw: torch.Tensor, pooled: torch.Tensor, table: torch.Tensor, seq: torch.Tensor,
              positions: torch.Tensor, weight: torch.Tensor, cos_sin: torch.Tensor, ratio: int, eps: float) -> None:
    page = raw.shape[1]
    for t in range(len(k)):
        if seq[t] >= 0:
            unit, off = _where(table, int(seq[t]), int(positions[t]), page)
            raw[unit, off] = k[t]
    for t in range(len(k)):
        p = int(positions[t])
        if seq[t] < 0 or p % ratio != ratio - 1:
            continue
        start = p - ratio + 1
        keys = torch.stack([raw[_where(table, int(seq[t]), start + i, page)] for i in range(ratio)])
        x = keys.float().mean(0).to(raw.dtype).float()
        x = x * torch.rsqrt(x.pow(2).mean() + eps) * (1.0 + weight.float())
        unit, off = _where(table, int(seq[t]), start, page)
        pooled[unit, off // ratio] = rope(x, start, cos_sin)


def qsa_select(q: torch.Tensor, pooled: torch.Tensor, table: torch.Tensor, seq: torch.Tensor, positions: torch.Tensor,
               ratio: int, budget: int) -> torch.Tensor:
    page = pooled.shape[1] * ratio
    top = budget // ratio
    out = torch.full((len(q), top + 1), -1, dtype=torch.int32)
    for t in range(len(q)):
        if seq[t] < 0:
            continue
        p = int(positions[t])
        n = (p + 1) // ratio
        if n:
            first = torch.arange(n) * ratio                          # each block's first position
            keys = pooled[table[int(seq[t]), first // page].long(), (first % page) // ratio].double()
            scores = (keys @ q[t].double().T).relu().sum(-1) / math.sqrt(q.shape[-1])
            scores = scores.float()                                  # the contract's float32 scores
            chosen = stable_topk(scores, min(top, n)).indices.sort().values
            out[t, :len(chosen)] = chosen.int()
        if (p + 1) % ratio:
            out[t, min(top, n)] = n
    return out


def _kv(kv) -> torch.Tensor:
    """The cache in float32, read through its format (gqa's layouts: model dtype, fp8, or q8_0's pair of int8 values
    and an fp16 scale per 32)."""
    if isinstance(kv, tuple):
        values, scales = kv
        return (values.float().unflatten(-1, (-1, 32)) * scales.float()[..., None]).flatten(-2)
    return kv.float()


def qsa_attend(q: torch.Tensor, kv, blocks: torch.Tensor, table: torch.Tensor, seq: torch.Tensor,
               positions: torch.Tensor, gate: torch.Tensor | None, ratio: int) -> torch.Tensor:
    cache = _kv(kv)                                                  # [units, 2, Hkv, page, D]
    page, rep = cache.shape[3], q.shape[1] // cache.shape[2]
    out = torch.zeros_like(q)
    for t in range(len(q)):
        if seq[t] < 0:
            continue
        p = int(positions[t])
        tokens = torch.tensor([ratio * int(b) + i for b in blocks[t] if b >= 0 for i in range(ratio)
                               if ratio * int(b) + i <= p])
        units, offsets = table[int(seq[t]), tokens // page].long(), tokens % page
        k = cache[units, 0, :, offsets].repeat_interleave(rep, dim=1)                # [n, Hq, D]
        v = cache[units, 1, :, offsets].repeat_interleave(rep, dim=1)
        scores = torch.einsum("hd,nhd->hn", q[t].float(), k) / math.sqrt(q.shape[-1])
        o = torch.einsum("hn,nhd->hd", scores.softmax(-1), v)
        out[t] = (o if gate is None else o * torch.sigmoid(gate[t].float())).to(q.dtype)   # padding rows stay 0
    return out
