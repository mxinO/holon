"""Speculative sampling (docs/m04-spec.md D5): lossless verification of deterministic drafts. With p the distribution
the sampler would draw the token after j accepted drafts from (`Sampler.distribution`, its temperature and filters),
draft d at column j is accepted with probability p(d) (an independent coin, sampler stream 1); at the first rejection
the token is drawn from p with d removed and renormalized (stream 2); after the row's last real draft it is drawn
from p as plain decoding would (stream 0, the same draw). Each emitted token then has exactly the target's
distribution. Greedy rows reduce to greedy-exact (p is one-hot)."""
from __future__ import annotations

import torch


def draw(q: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Inverse CDF over each row of `q` (unnormalized): the first index whose cumulative mass exceeds u × total."""
    c = q.cumsum(-1)
    return torch.searchsorted(c, (u * c[:, -1])[:, None], right=True)[:, 0].clamp_(max=q.shape[-1] - 1)


def verify(logits: torch.Tensor, ctx):
    dev, table, sampler = ctx.batch.dev, ctx.seq_table, ctx.sampler
    width = int(ctx.batch.plan.n_logits.max())
    rows = dev.slots[dev.logits_rows]
    slots = rows.view(-1, width)[:, 0]
    p = sampler.distribution(logits, ctx)                             # [R·w, V]
    k = width - 1
    drafts = table.draft_tokens[slots, :k]
    real = torch.arange(k, device=logits.device) < table.n_draft[slots, None]
    pd = p.view(-1, width, p.shape[-1])[:, :k].gather(-1, drafts[..., None])[..., 0]          # p_j(d_j)
    residual = p.clone()
    padded = torch.cat([drafts, drafts.new_zeros(len(drafts), 1)], 1).reshape(-1)          # column j's draft
    residual.scatter_(-1, padded[:, None], 0.0)
    sure = (residual.sum(-1) == 0).view(-1, width)[:, :k]             # p(d) = 1 up to fp32: nothing to draw instead
    coins = sampler.uniform(ctx, stream=1).view(-1, width)[:, :k]
    n_accept = (((coins < pd) | sure) & real).long().cumprod(-1).sum(-1)
    plain = sampler.sample(logits, ctx).tokens                        # the token after the row's last real draft
    rejected = draw(residual, sampler.uniform(ctx, stream=2))
    j = torch.arange(width, device=logits.device)
    n_real = table.n_draft[slots].clamp(max=k)
    final = torch.where((j[None, :] < n_real[:, None]), rejected.view(-1, width), plain.view(-1, width))
    accepted = torch.cat([drafts, drafts.new_zeros(len(drafts), 1)], 1)
    tokens = torch.where(j[None, :] < n_accept[:, None], accepted, final).reshape(-1)
    return sampler.logprobs(logits, tokens, ctx), n_accept


class Plugin:
    """The `verifier` slot's plugin object is the verifier."""

    def register(self, regs) -> None:
        pass

    verify = staticmethod(verify)
