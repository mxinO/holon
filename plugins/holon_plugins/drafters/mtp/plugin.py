"""Drafts with the target checkpoint's multi-token-prediction head (docs/m04-spec.md D2, D3): whatever head the target's
model plugin declares as `draft_head("mtp")`, built as the auxiliary model "mtp" (its KV under `mtp.kv`).

Pairing (llama.cpp's): the head's row at position p takes the token at p and the target's final-norm hidden of p − 1,
so every head KV entry depends only on tokens ≤ p and prefix-cached pages share safely. A head whose `carries` is
"residual" (e.g. `std.spec.mtp_hc_head@1`, FN D9) takes the trunk's residual itself (`ModelSpec.residual`'s width and
dtype) instead of its final norm, returns its own, and maps it for the `lm_head` by its `out`; its token embedding is
the target embed's `inner` when it wraps one (`std.hyper.expand@1`'s, before the repeat into streams). The head's
weights are the target checkpoint's, or with the plugin option `mtp.path` a file of their own (e.g. Strata's MTP
pack). After each step `run`:

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
        path = ctx.options.get("plugins", {}).get("mtp.path")
        tensors = ctx.weight_source(path).tensors() if path else target.tensors()
        return MTPDrafter(ctx, target, ctx.aux_model(NAMESPACE, spec, tensors))


class MTPDrafter:
    def __init__(self, ctx, target, head) -> None:
        self.head = head
        self.embed, self.norm, self.lm_head = (target.shared(n) for n in ("embed", "norm", "lm_head"))
        self.residual = getattr(head, "carries", "norm") == "residual"
        self.out = head.out if self.residual else (lambda x: x)
        if self.residual:
            self.embed = getattr(self.embed, "inner", self.embed)
        r = target.spec.residual
        width, dtype = ((r.width, getattr(torch, r.dtype)) if self.residual and r is not None
                        else (target.spec.hidden, ctx.dtype))
        slots = ctx.options["max_seqs"] + 1
        self.dtype = ctx.dtype                      # what the head's predecessor hiddens are cast to
        self.carry = ctx.static("drafter.mtp.carry", (slots, width), dtype)          # per slot: a hidden ...
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
        hn = h if self.residual else self.norm(h)                        # what the head takes, of every step token
        first = dev.query_start[:-1]
        slots = dev.slots
        # each token's predecessor in its row, in the model's dtype (the head casts it so first: half a float32
        # residual's bytes in a prompt step's peak) …
        prev = torch.empty(hn.shape, dtype=self.dtype, device=hn.device)
        prev[1:] = hn[:-1]
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
            token = self.lm_head(self.out(x)).argmax(-1)
            table.draft_tokens[sampled, i] = token
        table.n_draft.index_fill_(0, sampled, k)
