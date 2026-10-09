"""`recurrent` state kind: fixed-size state per request (e.g. a linear-attention or SSM state) that a step overwrites
in place, so it cannot be shared by reference. Prefix reuse copies it: a checkpoint is a payload holding a request's
whole state at a position (a multiple of the spec's alignment), published into this kind's own pool and adopted by a
later request with the same prefix (docs/m03-qwen38.md D4).

Where checkpoints are taken is a policy, from the deployment's plugin options:
- `recurrent.policy`: `"prompt_end"` (default: once per prompt, at its last aligned position — what an agent's next
  turn resends) or `"every:<n>"` (every n tokens, n a multiple of the alignment);
- `recurrent.checkpoints`: units of the checkpoint pool (default: one per request slot).
"""
from __future__ import annotations

import numpy as np
from holon.api import HitLen, MemOp, Payload


class Recurrent:
    verify_mode = "trivial"  # speculative rollback (replay or checkpoint slots) arrives with M0.4

    def __init__(self, spec, pool, ctx) -> None:
        self.spec, self.pool = spec, pool
        self.align = spec.alignment
        opts = ctx.options.get("plugins", {})
        policy = opts.get("recurrent.policy", "prompt_end")
        if policy == "prompt_end":
            self.every = None
        elif policy.startswith("every:") and int(policy[6:]) % self.align == 0 and int(policy[6:]) > 0:
            self.every = int(policy[6:])
        else:
            raise ValueError(f"recurrent.policy {policy!r}: 'prompt_end' or 'every:<n>', n a multiple of {self.align}")
        units = int(opts.get("recurrent.checkpoints", ctx.options["max_seqs"]))
        self.payloads = ctx.pool(f"{spec.name}.checkpoints", unit_bytes=spec.shape.unit_bytes, segments=spec.segments,
                                 pref_units=units)
        self.payload_pool = self.payloads.name

    def units_needed(self, seq, new_tokens: int) -> int:
        return 0                                  # the request slot owns its unit

    def capacity(self, seq) -> int:
        return 1 << 62

    def reclaimable(self, seq) -> list[int]:
        return []

    def next_publish(self, seq) -> int | None:
        """The next position after the request's computed tokens where the policy checkpoints."""
        start = seq.num_computed
        if self.every is not None:
            return (start // self.every + 1) * self.every
        at = seq.prompt_len // self.align * self.align
        return at if at > start else None

    def cut(self, seq, start: int, n: int) -> int:
        at = self.next_publish(seq)
        return at - start if at is not None and start < at < start + n else n

    def publish(self, seq, at: int):
        if self.payloads.num_free == 0:           # best effort: the index already evicted what it could
            return None
        unit = self.payloads.alloc(1)
        payload = Payload((int(unit[0]),), self.payload_pool)
        return payload, [MemOp(self.pool.name, np.array(seq.units(self.spec.name)), self.payload_pool, unit,
                               "post_sample")]

    def hit_length(self, idx, upto: int) -> HitLen:
        for i in reversed(range(idx.blocks(upto))):
            if idx.payload(i) is not None:
                return HitLen((i + 1) * idx.block)
        return HitLen(0)

    def adopt(self, seq, payload: Payload) -> list[MemOp]:
        return [MemOp(self.payload_pool, np.array(payload.units), self.pool.name, np.array(seq.units(self.spec.name)),
                      "pre_forward")]

    def fork(self, src, dsts) -> list[MemOp]:
        return []                                  # not built in M0.3: n > 1 siblings adopt the first's checkpoints

    def commit(self, slots, n_accept, path=None) -> None:
        pass

    def host_commit(self, view) -> None:
        pass
