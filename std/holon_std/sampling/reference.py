"""Sampling oracle: the exact distribution the default sampler draws from, per row, in float64.

Semantics: logits / temperature → top-k (keep the k largest) → top-p (keep the smallest prefix of the remaining,
renormalized distribution whose mass reaches p; the most likely token always stays) → min-p (drop tokens whose
probability is below min_p × the largest). Temperature 0 is greedy.
"""
from __future__ import annotations

import torch
from holon.api import SamplingParams


def probs(logits: torch.Tensor, p: SamplingParams) -> torch.Tensor:
    """Distribution over the vocabulary for one row of logits [V]."""
    x = logits.double()
    if p.temperature == 0:
        return torch.nn.functional.one_hot(x.argmax(), x.numel()).double()
    x = x / p.temperature
    keep = torch.ones_like(x, dtype=torch.bool)
    order = x.argsort(descending=True)
    if p.top_k > 0:
        keep[order[p.top_k:]] = False
    q = torch.where(keep, x, float("-inf")).softmax(-1)
    if p.top_p < 1:
        sorted_q = q[order]
        drop = sorted_q.cumsum(-1) - sorted_q > p.top_p
        keep[order[drop]] = False
    if p.min_p > 0:
        keep &= q >= p.min_p * q.max()
    return torch.where(keep, x, float("-inf")).softmax(-1)
