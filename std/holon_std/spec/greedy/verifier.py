"""Exact-match verification: accept the longest prefix of drafts that equals the token the sampler draws at each
column, then emit the sampler's token after it. Every emitted token is the sampler's own draw (its counter-based
uniform at that position), so speculative decoding emits exactly what plain decoding would given the same logits —
greedy or sampled (verify rows' logits may differ from decode rows' in the last bits on GPU: the law is exact, the
tokens agree up to rounding). Greedy rows accept whatever matches the argmax; sampled rows accept less often than
`std.spec.rejection` does."""
from __future__ import annotations

import torch


def verify(logits: torch.Tensor, ctx):
    dev, table, sampler = ctx.batch.dev, ctx.seq_table, ctx.sampler
    width = int(ctx.batch.plan.n_logits.max())                       # 1 + k, the same for every verify row
    tokens = sampler.sample(logits, ctx).tokens                      # what plain decoding draws at each column
    target = tokens.view(-1, width)
    slots = dev.slots[dev.logits_rows].view(-1, width)[:, 0]
    drafts = table.draft_tokens[slots, : width - 1]
    real = torch.arange(width - 1, device=logits.device) < table.n_draft[slots, None]     # padding is never accepted
    n_accept = ((drafts == target[:, :-1]) & real).long().cumprod(-1).sum(-1)
    return sampler.logprobs(logits, tokens, ctx), n_accept


class Plugin:
    """The `verifier` slot's plugin object is the verifier."""

    def register(self, regs) -> None:
        pass

    verify = staticmethod(verify)
