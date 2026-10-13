import torch

from ..oracle import ORACLE


class Plugin:
    def register(self, regs) -> None:
        pass

    def build(self, ctx, target):
        return Drafter(ctx.options["max_seqs"] + 1, ctx.options["max_model_len"])


class Drafter:
    """Drafts the oracle's tokens for the positions after each sampled row's new length — right, wrong (shifted),
    or mixed (wrong where (slot + position) % 3 == 0) — read on the device, so it works at overlap depth 1. Slots
    holding a sequence the oracle does not know get no real drafts (`n_draft` 0).

    It also keeps what a real drafter's own state would see, for tests to check against the committed tokens: each
    slot's history from `admit` (ORACLE.history) and every step's input tokens at their positions (ORACLE.absorbed,
    and history too)."""

    def __init__(self, slots: int, length: int) -> None:
        self.history = ORACLE.history = torch.full((slots, length), -1, dtype=torch.int64)
        self.absorbed = ORACLE.absorbed = torch.full((slots, length), -1, dtype=torch.int64)

        self.rewound = None

    def admit(self, slot, tokens) -> None:
        self.history[slot] = self.absorbed[slot] = -1
        self.history[slot, : len(tokens)] = torch.tensor(tokens)

    def rewind(self, ctx) -> None:
        """Called after the verifier wrote n_accept, never past n_draft, and before `run`."""
        table, slots = ctx.seq_table, ctx.batch.dev.slots
        assert (ctx.batch.plan.n_logits > 1).all() and (table.n_accept[slots] <= table.n_draft[slots]).all()
        self.rewound = ctx.batch
        ORACLE.rewinds += 1

    def run(self, ctx, h) -> None:
        dev, table = ctx.batch.dev, ctx.seq_table
        assert h.shape[0] == dev.input_ids.shape[0]                    # the trunk's output for every step token
        verify = bool((ctx.batch.plan.n_logits > 1).any())
        assert (self.rewound is ctx.batch) == verify                   # rewound first, exactly on verify steps
        self.rewound = None
        if not verify:                                                 # sampled rows of a plain step accepted nothing
            assert (table.n_accept[dev.slots[dev.logits_rows]] == 0).all()
        ORACLE.max_position = max(ORACLE.max_position, int(dev.positions.max()))
        ORACLE.max_computed = max(ORACLE.max_computed, int(table.num_computed[dev.slots].max()))
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
        n_draft = known.to(slots.device)[slots]
        if ORACLE.real is not None:                                    # past the real drafts: ids no model accepts
            n_draft = n_draft.clamp(max=ORACLE.real)
            drafts = torch.where(torch.arange(k, device=slots.device) < n_draft[:, None], drafts, ORACLE.vocab + 7)
        table.draft_tokens[slots] = drafts
        table.n_draft[slots] = n_draft
        for slot, row, n in zip(slots.tolist(), drafts.tolist(), n_draft.tolist(), strict=True):
            ORACLE.written[slot] = row[:n]
