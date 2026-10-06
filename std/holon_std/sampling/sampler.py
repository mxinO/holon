"""Default sampler: greedy, or temperature with top-k / top-p / min-p filtering, plus requested logprobs."""
from __future__ import annotations

import torch
from holon.api import SampleOut, SamplingParams


def _filter(logits: torch.Tensor, p: SamplingParams) -> torch.Tensor:
    if p.top_k > 0:
        kth = logits.topk(min(p.top_k, logits.shape[-1])).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = logits.softmax(-1)
    if p.min_p > 0:
        logits = logits.masked_fill(probs < p.min_p * probs.max(), float("-inf"))
    if p.top_p < 1:
        sorted_probs, order = probs.sort(descending=True)
        drop = sorted_probs.cumsum(-1) - sorted_probs > p.top_p
        logits = logits.masked_fill(drop.scatter(-1, order, drop), float("-inf"))
    return logits


class Sampler:
    """Per-slot parameters, set on admission. A seeded request draws its token at position `p` from a generator
    seeded by (seed, p), so its samples depend neither on batching nor on preemption."""

    def __init__(self, max_seqs: int, device: torch.device) -> None:
        self.params: list[SamplingParams | None] = [None] * max_seqs

    def admit(self, slot: int, params: SamplingParams) -> None:
        self.params[slot] = params

    def sample(self, logits: torch.Tensor, ctx) -> SampleOut:
        plan = ctx.batch.plan
        sampled = plan.n_logits > 0
        params = [self.params[s] for s in plan.slots[sampled]]
        positions = (plan.start_upper + plan.n_query)[sampled].tolist()
        tokens = torch.empty(logits.shape[0], dtype=torch.long, device=logits.device)
        for i, (p, pos) in enumerate(zip(params, positions, strict=True)):
            if p.temperature == 0:
                tokens[i] = logits[i].argmax()
                continue
            gen = None
            if p.seed is not None:
                gen = torch.Generator(logits.device).manual_seed(hash((p.seed, pos)) & (2**63 - 1))
            probs = _filter(logits[i] / p.temperature, p).softmax(-1)
            tokens[i] = torch.multinomial(probs, 1, generator=gen)[0]
        if not any(p.logprobs is not None for p in params):
            return SampleOut(tokens)
        logprobs = logits.log_softmax(-1)
        top = logprobs.topk(max(max(p.logprobs or 0 for p in params), 1))
        return SampleOut(tokens, logprobs.gather(-1, tokens[:, None])[:, 0], top.indices, top.values)


class Plugin:
    """The `sampler` slot's plugin: creates one sampler per engine."""

    def register(self, regs) -> None:
        pass

    def sampler(self, max_seqs: int, device: torch.device) -> Sampler:
        return Sampler(max_seqs, device)
