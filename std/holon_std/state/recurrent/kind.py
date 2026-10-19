"""`recurrent` state kind: fixed-size state per request (e.g. a linear-attention or SSM state) that a step overwrites
in place, so it cannot be shared by reference. Prefix reuse copies it: a checkpoint is a payload holding a request's
whole state at a position (a multiple of the spec's alignment), published into this kind's own pool and adopted by a
later request with the same prefix (docs/m03-qwen38.md D4).

Where checkpoints are taken is a list of policies, from the deployment's plugin options (docs/v1-kv-tiers.md D8):
- `recurrent.policy`: a comma list; the kind checkpoints at the earliest position any of them names.
  - `prompt_end` (the default): once per prompt, at its last aligned position — what an agent's next turn resends;
  - `every:<n>`: every n tokens (n a multiple of the alignment); a prefill step takes one of them in-step and ends one
    grid before the next, so a small n costs steps;
  - `branch`: where the request leaves a cached chain that continues (`SeqView.branch_at`: a shared system prompt's
    end, found rather than configured);
  - `answer:<k>`: a tentative checkpoint at each aligned position a decode row reaches; the request keeps its last k
    (`keep_tentative`; the core drops older ones). It needs a state tier (the engine refuses it without one);
  - a `Ref` (`name@major`) to a unit with `next(view) -> (at, tentative) | None`: the next position after
    `view.num_computed` it checkpoints at, and whether that payload is tentative; and `keep` (optional, default 0),
    the tentative payloads a request keeps for it. Positions a unit without `keep` calls tentative are not (nothing
    would keep them). Plugins implement it structurally.
  The kind keeps a request's last `keep_tentative` = the largest `keep` of its policies (the engine refuses one above
  0 without a state tier).
- `recurrent.checkpoints`: units of the checkpoint pool. Set, they are required: the ledger refuses a start that cannot
  give them (0: no checkpoints, no cuts). Unset, the default is one per request slot, best effort: a budget that does
  not fit them gives the pool 0 units, so the kind takes no checkpoints and cuts nothing, and says so once (stderr).
  With a state tier the pool only stages payloads between a step and its commit (D7).

Positions are multiples of the spec's alignment and of the page size: the prefix index files payloads per block of
`lcm(page, alignment)` tokens, and a payload off a block boundary could never be found.
"""
from __future__ import annotations

import math
import re
import sys

import numpy as np
from holon.api import HitLen, MemOp, Payload, Ref


class PromptEnd:
    def __init__(self, align: int) -> None:
        self.align = align

    def next(self, view):
        at = view.prompt_len // self.align * self.align
        return (at, False) if at > view.num_computed else None


class Every:
    def __init__(self, n: int) -> None:
        self.n = n

    def next(self, view):
        return (view.num_computed // self.n + 1) * self.n, False


class Branch:
    def next(self, view):
        return (view.branch_at, False) if view.branch_at > view.num_computed else None


class Answer:
    def __init__(self, align: int, keep: int) -> None:
        self.align, self.keep = align, keep

    def next(self, view):
        at = (view.num_computed // self.align + 1) * self.align
        return (at, True) if at > view.prompt_len else None


class _At:
    """A view as it will be once `at` tokens are computed (where the position after `at` lies)."""

    def __init__(self, view, at: int) -> None:
        self.view, self.num_computed = view, at

    def __getattr__(self, name: str):
        return getattr(self.view, name)


class Recurrent:
    def __init__(self, spec, pool, ctx) -> None:
        self.spec, self.pool = spec, pool
        self.align = math.lcm(spec.alignment, ctx.options.get("page_size", 16))
        opts = ctx.options.get("plugins", {})
        self.policies = [self._policy(name.strip(), ctx) for name in
                         str(opts.get("recurrent.policy", "prompt_end")).split(",") if name.strip()]
        self.keep_tentative = max((getattr(p, "keep", 0) for p in self.policies), default=0)
        asked = str(opts.get("recurrent.checkpoints", "")).strip()
        units = int(asked) if asked else ctx.options["max_seqs"]
        self.payloads = ctx.pool(f"{spec.name}.checkpoints", unit_bytes=spec.shape.unit_bytes, segments=spec.segments,
                                 min_units=units if asked else 0, pref_units=units)    # set: required
        self.warn = not asked and units > 0          # the default, best effort: say so if it got no units
        self.payload_pool = self.payloads.name

    def _policy(self, name: str, ctx):
        if name == "prompt_end":
            return PromptEnd(self.align)
        if name == "branch":
            return Branch()
        if (m := re.fullmatch(r"every:(\d+)", name)) and int(m[1]) > 0 and int(m[1]) % self.align == 0:
            return Every(int(m[1]))
        if (m := re.fullmatch(r"answer:(\d+)", name)) and int(m[1]) > 0:
            return Answer(self.align, int(m[1]))
        if "@" in name:
            return ctx.unit(Ref(name), spec=self.spec.name)
        raise ValueError(f"recurrent.policy {name!r}: prompt_end, branch, every:<n> (n a multiple of {self.align}), "
                         "answer:<k> or a policy unit's Ref")

    def _next(self, seq, start: int | None = None) -> tuple[int, bool] | None:
        """The earliest position after the computed tokens (or `start`) a policy names, and whether its payload is
        tentative (only when every policy naming it says so)."""
        view = seq if start is None else _At(seq, start)
        found = [(r[0], r[1] and getattr(p, "keep", 0) > 0) for p in self.policies
                 if (r := p.next(view)) is not None and r[0] > view.num_computed]
        if not found:
            return None
        at = min(a for a, _ in found)
        return at, all(t for a, t in found if a == at)

    def units_needed(self, seq, new_tokens: int) -> int:
        return 0                                  # the request slot owns its unit

    def capacity(self, seq) -> int:
        return 1 << 62

    def next_publish(self, seq) -> int | None:
        """The next position after the request's computed tokens where a policy checkpoints (none without
        checkpoint units)."""
        if not self.payloads.capacity:
            if self.warn:                            # once, at the first request
                self.warn = False
                print(f"holon: {self.payloads.name} got 0 units (the state budget did not fit the default "
                      f"recurrent.checkpoints): no prefix reuse for {self.spec.name}; set recurrent.checkpoints to "
                      "require them", file=sys.stderr, flush=True)
            return None
        found = self._next(seq)
        return found[0] if found is not None else None

    def cut(self, seq, start: int, n: int) -> int:
        """Where a prefill step must end so every checkpoint is taken: at the position, unless the spec's forward
        fills it in-step (`spec.in_step`: a grid the position meets from `start`; one per step, so a second position
        ends the step one grid before it, to be taken in-step by the next). A budget-truncated step is rounded down
        to the grid, so the next one starts on it."""
        at, grid, end = self.next_publish(seq), self.spec.in_step, start + n
        if at is not None and start < at < end:
            if not grid or (at - start) % grid:
                return at - start                                    # taken where the step ends (a MemOp copy)
            if (second := self._next(seq, at)) is not None and second[0] <= end:    # the next step's
                return max(second[0] - grid, at) - start
        if at is not None and grid and end < len(seq.tokens) and end % grid and end // grid * grid > start:
            return end // grid * grid - start                         # truncated by the step budget
        return n

    def publish(self, seq, at: int, in_step: bool = False):
        if self.payloads.num_free == 0:           # best effort: the index already evicted what it could
            return None
        unit = self.payloads.alloc(1)
        found = self._next(seq)
        payload = Payload((int(unit[0]),), self.payload_pool, found is not None and found[0] == at and found[1])
        if in_step:                               # the mixer's forward writes it (`BatchPlan.publishes`)
            return payload, []
        return payload, [MemOp(self.pool.name, np.array(seq.units(self.spec.name)), self.payload_pool, unit,
                               "post_sample")]

    def hit_length(self, idx, upto: int) -> HitLen:
        for i in reversed(range(idx.blocks(upto))):
            if idx.payload(i) is not None:
                return HitLen((i + 1) * idx.block)
        return HitLen(0)

    def adopt(self, seq, payload: Payload) -> list[MemOp]:
        """From `payload.pool`: this kind's pool, or a state tier's host copy (docs/v1-kv-tiers.md D7)."""
        return [MemOp(payload.pool, np.array(payload.units), self.pool.name, np.array(seq.units(self.spec.name)),
                      "pre_forward")]
