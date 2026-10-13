import torch

from ..oracle import ORACLE


class Plugin:
    def register(self, regs) -> None:
        pass

    def build(self, ctx, target):
        return Drafter()


class Drafter:
    """Drafts the oracle's tokens for the positions after each sampled row's new length — right, wrong (shifted),
    or mixed (wrong where (slot + position) % 3 == 0) — read on the device, so it works at overlap depth 1. Slots
    holding a sequence the oracle does not know get no real drafts (`n_draft` 0).

    It also keeps what a real drafter's own state would see, for tests to check against the committed tokens: each
    slot's history from `admit` (ORACLE.history) and every step's input tokens at their positions (ORACLE.absorbed,
    and history too)."""

    def __init__(self) -> None:
        self.history = ORACLE.history = torch.full((64, 256), -1, dtype=torch.int64)
        self.absorbed = ORACLE.absorbed = torch.full((64, 256), -1, dtype=torch.int64)

    def admit(self, slot, tokens) -> None:
        self.history[slot] = self.absorbed[slot] = -1
        self.history[slot, : len(tokens)] = torch.tensor(tokens)

    def run(self, ctx, h) -> None:
        dev, table = ctx.batch.dev, ctx.seq_table
        assert h.shape[0] == dev.input_ids.shape[0]                    # the trunk's output for every step token
        rows = torch.repeat_interleave(torch.arange(len(dev.slots)), dev.query_start.diff().cpu())
        slots, positions = dev.slots.cpu()[rows], dev.positions.cpu()
        self.history[slots, positions] = self.absorbed[slots, positions] = dev.input_ids.cpu()
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
