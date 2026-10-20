"""Oracles of Qwen3.8-Flash-Next's PLE ops (`kernels.py`), written out per row from HF's module
(transformers 5.19.0, `models/qwen4_exp/modeling_qwen4_exp.py:1072-1247`)."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def ple_rows(ids: torch.Tensor, history: torch.Tensor, slots: torch.Tensor, query_start: torch.Tensor,
             has_init: torch.Tensor, multipliers: torch.Tensor, sizes: torch.Tensor, offsets: torch.Tensor,
             ngram: int, heads_per_ngram: int, eos: int) -> torch.Tensor:
    out = torch.empty(len(ids), (ngram - 1) * heads_per_ngram, dtype=torch.long)
    for r, (s, e) in enumerate(zip(query_start[:-1].tolist(), query_start[1:].tolist(), strict=True)):
        before = history[slots[r]].tolist() if has_init[r] else [eos, eos]
        seq = before + ids[s:e].tolist()
        for i in range(2, len(seq)):
            pred = []
            for shift in range(ngram):
                j = i - shift
                pred.append(seq[j] if j >= 0 and eos not in seq[j:i] else eos)
            heads = []
            for n in range(2, ngram + 1):
                mixed = 0
                for shift in range(n):
                    mixed ^= _wrap(pred[shift] * int(multipliers[shift]))
                start = (n - 2) * heads_per_ngram
                heads += [mixed % int(sizes[h]) + int(offsets[h]) for h in range(start, start + heads_per_ngram)]
            out[s + i - 2] = torch.tensor(heads)
        if e > s:
            history[slots[r]] = torch.tensor(seq[-2:])
    return out


def _wrap(x: int) -> int:
    """x as a wrapping int64 (torch's and the kernels' product)."""
    x &= (1 << 64) - 1
    return x - (1 << 64) if x >> 63 else x


def _norm(x: torch.Tensor, weight: torch.Tensor, streams: int, eps: float) -> torch.Tensor:
    y = x.reshape(len(x), streams, x.shape[1] // streams)
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    return y.reshape(x.shape) * (1.0 + weight.float())


def _gate(u: torch.Tensor) -> torch.Tensor:
    """The signed-√ gate per (token, stream)."""
    return torch.sigmoid(u.sign() * u.abs().clamp_min(1e-6).sqrt())


def ple_block(key: torch.Tensor, value: torch.Tensor, r: torch.Tensor, w_key: torch.Tensor, w_query: torch.Tensor,
              w_conv: torch.Tensor, conv: torch.Tensor, state: torch.Tensor, slots: torch.Tensor,
              query_start: torch.Tensor, has_init: torch.Tensor, streams: int, dilation: int, eps: float
              ) -> torch.Tensor:
    hidden = value.shape[1]
    k = _norm(key.float(), w_key, streams, eps).reshape(len(key), streams, hidden)
    q = _norm(r.float(), w_query, streams, eps).reshape(len(r), streams, hidden)
    g = _gate((k * q).sum(-1) / math.sqrt(hidden))
    gated = (g[:, :, None] * value.float()[:, None, :]).reshape(r.shape)
    c = _norm(gated, w_conv, streams, eps)
    taps, window = conv.shape[1], state.shape[1]
    for i, (s, e) in enumerate(zip(query_start[:-1].tolist(), query_start[1:].tolist(), strict=True)):
        if e == s:
            continue
        past = state[slots[i]] if has_init[i] else torch.zeros_like(state[slots[i]])
        seq = torch.cat((past, c[s:e]))                                 # [window + n, C]
        y = sum(conv[:, k].float() * seq[window - (taps - 1 - k) * dilation:][: e - s] for k in range(taps))
        r[s:e] += gated[s:e] + F.silu(y)
        state[slots[i]] = seq[-window:]
    return c
