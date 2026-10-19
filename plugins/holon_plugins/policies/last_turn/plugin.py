"""`policies.last_turn` (docs/v1-kv-tiers.md §1, held out after `v1-kv-tiers-api`): Maya KG2's checkpoint at the last
turn's start, a `std.state.recurrent` policy by `Ref` (`recurrent.policy = "prompt_end,policies.last_turn@1"`).

An agent's next turn resends its history and appends a turn. Its admission matches the history's cached blocks, and
a hybrid's hit is trimmed back to the last checkpoint on them (the previous prompt's end); the turn's own start, where
the matched chain ends, gets no checkpoint, so a request sharing the history but not this turn (a retry, a sibling)
recomputes the previous answer. This policy checkpoints there: at the request's untrimmed match length, which is a
whole number of index blocks, so aligned for every kind.

**The proof's finding:** `SeqView` does not carry that length. `num_computed` is the trimmed hit; `branch_at` is the
matched chain's end only when its last block has cached continuations, which a turn that extends its history leaves
none of (its previous request was the chain's leaf). The minor it needs is one field, the untrimmed match length set
at admission (`SeqView.matched`); until the core has it, this unit refuses to build."""
from __future__ import annotations

import holon.api as api


def core_has_matched() -> bool:
    return "matched" in getattr(api.SeqView, "__annotations__", {})


class LastTurn:
    def __init__(self, ctx, spec: str | None = None) -> None:
        if not core_has_matched():
            raise RuntimeError("policies.last_turn needs SeqView.matched, the untrimmed match length at admission "
                               "(K's held-out proof finding, docs/v1-kv-tiers.md §13), which this core lacks")

    def next(self, view):
        return (view.matched, False) if view.matched > view.num_computed else None


class Plugin:
    def register(self, regs) -> None:
        regs.units.add("policies.last_turn", LastTurn)
