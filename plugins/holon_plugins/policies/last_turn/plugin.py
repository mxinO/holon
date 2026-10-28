"""`policies.last_turn` (docs/dev/v1-kv-tiers.md §1, held out after `v1-kv-tiers-api`): Maya KG2's checkpoint at the
last
turn's start, a `std.state.recurrent` policy by `Ref` (`recurrent.policy = "prompt_end,policies.last_turn@1"`).

An agent's next turn resends its history and appends a turn. Its admission matches the history's cached blocks, and
a hybrid's hit is trimmed back to the last checkpoint on them (the previous prompt's end); the turn's own start, where
the matched chain ends, gets no checkpoint, so a request sharing the history but not this turn (a retry, a sibling)
recomputes the previous answer. This policy checkpoints there: at `SeqView.matched`, the first admission's untrimmed
match length, a whole number of index blocks, so aligned for every kind.

The proof's finding: `SeqView` first lacked that length (`num_computed` is the trimmed hit, `branch_at` the match's end
only where cached continuations branch off); K added `matched` within ABI 1.4 (docs/dev/v1-kv-tiers.md §13)."""
from __future__ import annotations


class LastTurn:
    def __init__(self, ctx, spec: str | None = None) -> None:
        pass

    def next(self, view):
        return (view.matched, False) if view.matched > view.num_computed else None


class Plugin:
    def register(self, regs) -> None:
        regs.units.add("policies.last_turn", LastTurn)
