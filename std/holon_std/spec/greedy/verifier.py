"""Greedy-exact verification: accept the longest prefix of drafts that equals the target model's argmax, then emit
the argmax after it — so speculative greedy decoding produces exactly the tokens plain greedy decoding would."""
from __future__ import annotations

import torch
from holon.api import SampleOut


def verify(logits: torch.Tensor, ctx) -> tuple[SampleOut, torch.Tensor]:
    dev, table = ctx.batch.dev, ctx.seq_table
    width = int(ctx.batch.plan.n_logits.max())                       # 1 + k, the same for every verify row
    target = logits.argmax(-1).view(-1, width)
    slots = dev.slots[dev.logits_rows].view(-1, width)[:, 0]
    drafts = table.draft_tokens[slots, : width - 1]
    real = torch.arange(width - 1, device=logits.device) < table.n_draft[slots, None]     # padding is never accepted
    n_accept = ((drafts == target[:, :-1]) & real).long().cumprod(-1).sum(-1)
    return SampleOut(target.reshape(-1)), n_accept


class Plugin:
    """The `verifier` slot's plugin object is the verifier."""

    def register(self, regs) -> None:
        pass

    verify = staticmethod(verify)
