"""A drafter that taps the target and samples its drafts — the Gate B seams (docs/dev/m04-spec.md §3.4, §3.5) with no
model of its own, everything on the device so it runs inside captured graphs:

- taps: the residual stream after the target's layers `tap.layers` (plugin option; default the first and the last)
  kept per (slot, position): a copy of the first (`copy`) and an fp32 accumulation of all (`accum`);
- drafts: each sampled from a law of Kq = 4 ids — the oracle's token at that position (ORACLE.sequences, written to
  the device at admission) and the three ids after it, at probabilities 0.7, 0.1, 0.1, 0.1 — published in the SeqTable
  with the drafts (`q_support`), drawn with the request's seed on a uniform stream of the drafter's own.
"""
import torch
from holon_std.sampling.sampler import uniform

from ..oracle import ORACLE

LAW = (0.7, 0.1, 0.1, 0.1)
STREAM = 7                       # the drafter's own coins (the verifier uses 1 and 2, the sampler 0)


class Plugin:
    def register(self, regs) -> None:
        pass

    def build(self, ctx, target):
        return Drafter(ctx, target)


class Drafter:
    q_support = len(LAW)

    def __init__(self, ctx, target) -> None:
        slots, length, hidden = ctx.options["max_seqs"] + 1, ctx.options["max_model_len"], target.spec.hidden
        option = ctx.options.get("plugins", {}).get("tap.layers")
        layers = [int(i) for i in option.split(",")] if option else [0, len(target.spec.layers) - 1]
        self.vocab, k = target.spec.vocab, ctx.options["max_draft_tokens"]
        self.copy = ctx.static("tap.copy", (slots, length, hidden), ctx.dtype)
        self.accum = ctx.static("tap.accum", (slots, length, hidden), torch.float32)
        self.oracle = ctx.static("tap.oracle", (slots, length + k + 1), torch.int64)
        self.law = torch.tensor(LAW, device=ctx.device)
        for n, layer in enumerate(layers):
            ctx.tap(layer, lambda h, step, first=n == 0: self._keep(h, step, first))

    def _keep(self, h: torch.Tensor, ctx, first: bool) -> None:
        dev = ctx.batch.dev
        row = torch.searchsorted(dev.query_start[1:], torch.arange(len(h), device=h.device), right=True)
        at = (dev.slots[row], dev.positions)
        if first:
            self.copy[at] = h
            self.accum[at] = h.float()
        else:
            self.accum.index_put_(at, h.float(), accumulate=True)

    def admit(self, slot: int, tokens: list[int]) -> None:
        full = next((s for s in ORACLE.sequences.values() if s[: len(tokens)] == tokens), [])[: self.oracle.shape[1]]
        self.oracle[slot].zero_()
        self.oracle[slot, : len(full)] = torch.tensor(full, dtype=torch.int64)

    def run(self, ctx, h: torch.Tensor) -> None:
        dev, table, plan = ctx.batch.dev, ctx.seq_table, ctx.batch.plan
        verify = bool((plan.n_logits > 1).any())
        slots = dev.slots if verify else dev.slots[dev.logits_rows]          # the rows that sampled
        k = plan.draft_k
        if not k or not len(dev.logits_rows):
            table.n_draft.index_fill_(0, slots, 0)
            return
        at = table.num_computed[slots][:, None] + 1 + torch.arange(k, device=h.device)    # the drafts' positions
        known = self.oracle[slots[:, None], at.clamp(max=self.oracle.shape[1] - 1)]     # the oracle's tokens there
        ids = (known[..., None] + torch.arange(len(LAW), device=h.device)) % self.vocab
        u = uniform(ctx.sampler.seed[slots][:, None].expand(-1, k), at, STREAM)
        pick = torch.searchsorted(self.law.cumsum(0), u[..., None], right=True).clamp_(max=len(LAW) - 1)
        table.draft_tokens[slots, :k] = ids.gather(-1, pick)[..., 0]
        table.draft_q_ids[slots, :k] = ids
        table.draft_q[slots, :k] = self.law
        table.n_draft.index_fill_(0, slots, k)
