"""Prompt lookup on the device (docs/dev/m04-spec.md §1, D7): drafts the tokens that followed the most recent earlier
occurrence of a request's last n tokens in its own history (n from `ngram.max` down to `ngram.min`, plugin options;
defaults 3 and 2). Agent turns resend their history, so tool output and code often repeat.

Each slot's history is a device row: the whole request at admission (prefix hits included), then every step's input
tokens and the token it sampled. Rejected drafts written past a row's accepted end are overwritten before they are
read (lookups read below the current length only). The lookup is fixed-shape over the whole row, so it runs inside
the step's graph; it writes no model state, so it drafts at the engine's maximum k on every step (`n_draft` 0 on a
miss), which lets the policy fall back to plain decoding and back.
"""
from __future__ import annotations

import torch


class Plugin:
    def register(self, regs) -> None:
        pass

    def build(self, ctx, target):
        opts = ctx.options.get("plugins", {})
        return NGramDrafter(ctx, int(opts.get("ngram.max", 3)), int(opts.get("ngram.min", 2)))


class NGramDrafter:
    def __init__(self, ctx, n_max: int, n_min: int) -> None:
        if not 1 <= n_min <= n_max:
            raise ValueError(f"ngram sizes {n_min}..{n_max}: need 1 <= min <= max")
        self.n = list(range(n_max, n_min - 1, -1))
        self.k = ctx.options["max_draft_tokens"]
        self.length = ctx.options["max_model_len"]
        self.history = ctx.static("drafter.ngram.history", (ctx.options["max_seqs"] + 1, self.length), torch.int64)
        self.history.fill_(-1)

    def admit(self, slot: int, tokens: list[int]) -> None:
        row = torch.tensor(tokens[: self.length], dtype=torch.int64, pin_memory=self.history.is_cuda)
        self.history[slot].fill_(-1)
        self.history[slot, : len(row)].copy_(row, non_blocking=True)

    def run(self, ctx, h: torch.Tensor) -> None:
        dev, table = ctx.batch.dev, ctx.seq_table
        rows = torch.repeat_interleave(torch.arange(len(dev.slots), device=dev.slots.device), dev.query_start.diff(),
                                       output_size=dev.input_ids.shape[0])        # a known size: no host sync
        self.history[dev.slots[rows], dev.positions] = dev.input_ids          # the step's inputs at their positions
        sampled = dev.slots[dev.logits_rows]
        if not len(sampled):
            return
        if bool((ctx.batch.plan.n_logits > 1).any()):                         # verify rows: one entry each
            sampled = dev.slots
        end = table.num_computed[sampled].clamp(max=self.length - 1)          # the new last token's position
        self.history[sampled, end] = table.last_token[sampled]
        drafts, n_draft = self._lookup(self.history[sampled], end + 1)
        table.draft_tokens[sampled, : self.k] = drafts
        table.n_draft[sampled] = n_draft

    def _lookup(self, hist: torch.Tensor, length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per row of `hist` [R, L] with `length` [R] tokens: the k tokens after the latest earlier match of its last n
        tokens (the largest n that matches), and how many of them exist."""
        r, span = hist.shape
        pos = torch.arange(span, device=hist.device)
        drafts = torch.zeros(r, self.k, dtype=torch.int64, device=hist.device)
        n_draft = torch.zeros(r, dtype=torch.int64, device=hist.device)
        for n in self.n:                                                       # longest n first; keep the first hit
            tail = hist.gather(1, (length[:, None] - n + torch.arange(n, device=hist.device)).clamp(min=0))
            windows = hist.unfold(1, n, 1)                                     # [R, L - n + 1, n]
            match = (windows == tail[:, None, :]).all(-1)
            starts = pos[: span - n + 1]
            match &= (starts[None, :] + n < length[:, None]) & (length[:, None] > n)   # earlier, a token after it
            last = torch.where(match, starts[None, :], -1).amax(1)            # the latest occurrence
            found = (last >= 0) & (n_draft == 0)
            begin = last + n
            take = (begin[:, None] + torch.arange(self.k, device=hist.device)).clamp(max=span - 1)
            count = (length - begin).clamp(min=0, max=self.k)
            drafts = torch.where(found[:, None], hist.gather(1, take), drafts)
            n_draft = torch.where(found, count, n_draft)
        return drafts.clamp(min=0), n_draft
