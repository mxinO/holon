import torch

from ..oracle import ORACLE


class Plugin:
    """Drafts the oracle's tokens for the positions after each sampled row's new length — right, wrong (shifted),
    or mixed (wrong where (slot + position) % 3 == 0) — read on the device, so it works at overlap depth 1. Slots
    holding a sequence the oracle does not know get no real drafts (`n_draft` 0)."""

    def register(self, regs) -> None:
        pass

    def propose(self, ctx) -> None:
        dev, table = ctx.batch.dev, ctx.seq_table
        slots = dev.slots[dev.logits_rows]
        k = table.draft_tokens.shape[1]
        length = max(len(s) for s in ORACLE.sequences.values()) + k + 1
        oracle = torch.zeros(table.draft_tokens.shape[0], length, dtype=torch.int64)
        known = torch.zeros(table.draft_tokens.shape[0], dtype=torch.int64)
        for slot, full in ORACLE.by_slot.items():
            oracle[slot, : len(full)] = torch.tensor(full)
            known[slot] = 0 if slot in ORACLE.hidden_slots else k
        oracle = oracle.to(slots.device)
        pos = (table.num_computed[slots][:, None] + 1 + torch.arange(k, device=slots.device)).clamp(max=length - 1)
        drafts = oracle[slots[:, None], pos]
        wrong = (drafts + 1) % ORACLE.vocab
        if ORACLE.mode == "wrong":
            drafts = wrong
        elif ORACLE.mode == "mixed":
            drafts = torch.where((slots[:, None] + pos) % 3 == 0, wrong, drafts)
        table.draft_tokens[slots] = drafts
        table.n_draft[slots] = known.to(slots.device)[slots]
