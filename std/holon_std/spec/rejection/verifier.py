"""Speculative sampling (docs/m04-spec.md D5): lossless verification of deterministic or sampled drafts. With p the
distribution the sampler would draw the token after j accepted drafts from (`Sampler.distribution`, its temperature
and filters), and q the law draft d at column j was drawn from, d is accepted with probability min(1, p(d) / q(d))
(an independent coin u, sampler stream 1: accepted iff u · q(d) < p(d)); at the first rejection the token is drawn
from the residual max(p − q, 0), renormalized (stream 2); after the row's last real draft it is drawn from p as plain
decoding would (stream 0, the same draw). Each emitted token then has exactly the target's distribution: P(emit t) =
q(t) min(1, p(t)/q(t)) + (1 − Σ min(p, q)) · max(p(t) − q(t), 0) / Σ max(p − q, 0) = min(p, q)(t) + max(p − q, 0)(t)
= p(t), as Σ max(p − q, 0) = 1 − Σ min(p, q).

Deterministic drafts (no `SeqTable.draft_q`): q is one-hot at d, so d is accepted with probability p(d) and the
residual is p with d removed. Sampled drafts (`draft_q_ids` / `draft_q`, Kq entries per draft): q is zero off its
support, so the residual is p there and max(p − q, 0) on it. Greedy rows reduce to greedy-exact (p is one-hot). An
empty residual (p ≤ q everywhere, so p = q up to rounding) accepts the draft, whatever the coin."""
from __future__ import annotations

import torch


def draw(q: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Inverse CDF over each row of `q` (unnormalized): the first index whose cumulative mass exceeds u × total."""
    c = q.cumsum(-1)
    return torch.searchsorted(c, (u * c[:, -1])[:, None], right=True)[:, 0].clamp_(max=q.shape[-1] - 1)


def _sampled(p: torch.Tensor, table, slots: torch.Tensor, drafts: torch.Tensor, real: torch.Tensor):
    """p_j(d_j), q_j(d_j) and the residual of every column with the drafter's laws [R, k, Kq] (rows past n_draft read
    as an empty law at token 0); ids repeated within a law add up (a drafter may pad its support with zero mass)."""
    width = drafts.shape[1] + 1
    ids = table.draft_q_ids[slots, : width - 1].where(real[..., None], 0)
    q = table.draft_q[slots, : width - 1].where(real[..., None], 0.0)
    q_at = (q[..., None, :] * (ids[..., :, None] == ids[..., None, :])).sum(-1)       # q at each entry's id
    pv = p.view(-1, width, p.shape[-1])
    residual = pv.clone()
    on = residual[:, :-1]                                                                # the drafts' columns
    on.scatter_(-1, ids, (on.gather(-1, ids) - q_at).clamp_(min=0.0))                  # max(p − q, 0) on the support
    pd = pv[:, :-1].gather(-1, drafts[..., None])[..., 0]
    qd = (q * (ids == drafts[..., None])).sum(-1)
    return pd, qd, residual.view_as(p)


def verify(logits: torch.Tensor, ctx):
    dev, table, sampler = ctx.batch.dev, ctx.seq_table, ctx.sampler
    width = int(ctx.batch.plan.n_logits.max())
    rows = dev.slots[dev.logits_rows]
    slots = rows.view(-1, width)[:, 0]
    p = sampler.distribution(logits, ctx)                             # [R·w, V]
    k = width - 1
    real = torch.arange(k, device=logits.device) < table.n_draft[slots, None]
    drafts = table.draft_tokens[slots, :k].where(real, 0)             # past n_draft: never read (any id may be there)
    if table.draft_q is None:                                         # deterministic drafts: q one-hot at d
        pd = p.view(-1, width, p.shape[-1])[:, :k].gather(-1, drafts[..., None])[..., 0]          # p_j(d_j)
        residual = p.clone()
        padded = torch.cat([drafts, drafts.new_zeros(len(drafts), 1)], 1).reshape(-1)          # column j's draft
        residual.scatter_(-1, padded[:, None], 0.0)
        qd = None
    else:
        pd, qd, residual = _sampled(p, table, slots, drafts, real)
    sure = (residual.sum(-1) == 0).view(-1, width)[:, :k]             # p ≤ q up to fp32: nothing to draw instead
    coins = sampler.uniform(ctx, stream=1).view(-1, width)[:, :k]
    below = coins < pd if qd is None else coins * qd < pd
    n_accept = ((below | sure) & real).long().cumprod(-1).sum(-1)
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
