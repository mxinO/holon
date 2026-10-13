"""Drafts with the target checkpoint's multi-token-prediction head (docs/m04-spec.md D2, D3): whatever head the target's
model plugin declares as `draft_head("mtp")`, built as the auxiliary model "mtp" (its KV under `mtp.kv`).

Pairing (llama.cpp's): the head's row at position p takes the token at p and the target's final-norm hidden of p − 1,
so every head KV entry depends only on tokens ≤ p and prefix-cached pages share safely. After each step `run`:

1. runs the head over every token of the step (its KV), each token paired with its predecessor's hidden — inside the
   step, the previous row; for a row's first token, the hidden carried from the slot's last step if it was the
   position before, else zeros (a prefix hit's first token: costs acceptance, not correctness);
2. carries, per slot, the hidden of the row's last committed token (a verify row: the last accepted one);
3. for the rows that sampled, drafts k tokens greedily by k one-row sub-steps: the first pairs the new last token with
   that hidden, each next one its own draft with the head's previous output.
"""
from __future__ import annotations

import torch

NAMESPACE = "mtp"


class Plugin:
    def register(self, regs) -> None:
        pass

    def build(self, ctx, target):
        spec = target.draft_head(NAMESPACE)
        if spec is None:
            raise ValueError("drafters.mtp: the target's checkpoint declares no MTP draft head")
        return MTPDrafter(ctx, target, ctx.aux_model(NAMESPACE, spec, target.tensors()))


class MTPDrafter:
    def __init__(self, ctx, target, head) -> None:
        self.head = head
        self.embed, self.norm, self.lm_head = (target.shared(n) for n in ("embed", "norm", "lm_head"))
        slots, hidden = ctx.options["max_seqs"] + 1, target.spec.hidden
        self.carry = ctx.static("drafter.mtp.carry", (slots, hidden), ctx.dtype)      # per slot: a hidden ...
        self.carry_at = ctx.static("drafter.mtp.carry_at", (slots,), torch.int64)    # ... and its position
        self.carry_at.fill_(-1)

    def substeps(self, k: int) -> list[int]:
        return [1] * k

    def lookahead(self, k: int) -> int:
        return k

    def admit(self, slot: int, tokens: list[int]) -> None:
        self.carry_at[slot] = -1                    # nothing carried for the new owner

    def run(self, ctx, h: torch.Tensor) -> None:
        dev, table, plan = ctx.batch.dev, ctx.seq_table, ctx.batch.plan
        view = ctx.view(NAMESPACE)
        hn = self.norm(h)                                                # the lm_head input of every step token
        first = dev.query_start[:-1]
        slots = dev.slots
        prev = hn.roll(1, 0)                                             # each token's predecessor in its row …
        start = dev.positions[first]                                     # a request's first token carries nothing
        carried = self.carry[slots] * ((self.carry_at[slots] == start - 1) & (start > 0))[:, None]
        prev[first] = carried.to(prev.dtype)                             # … or the slot's carried hidden
        self.head(self.embed(dev.input_ids), prev, view)                 # the head's KV for the step's tokens
        verify = bool((plan.n_logits > 1).any())
        last = first + table.n_accept[slots] if verify else dev.query_start[1:] - 1
        self.carry[slots] = hn[last]
        self.carry_at[slots] = dev.positions[last]
        sampled = dev.slots[dev.logits_rows] if not verify else slots
        k = plan.draft_k
        if not k or not len(dev.logits_rows):
            table.n_draft.index_fill_(0, sampled, 0)
            return
        x, token = self.carry[sampled], table.last_token[sampled]
        for i in range(k):
            sub = view.substep(1, i)
            x = self.head(self.embed(token), x, sub)
            token = self.lm_head(x).argmax(-1)
            table.draft_tokens[sampled, i] = token
        table.n_draft.index_fill_(0, sampled, k)
