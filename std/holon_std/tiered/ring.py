"""A prefill step's lent slots (docs/dev/v1-offload.md D4, D5, D8), all on the device — tables written by the step's
own kernels, no host sync: at the step's first resolve its layers' empty slots, then the policy's `lend(need)` —
whole experts, the coldest — serve the step; at its end they return to their layers as empty slots, each its
layer's first victim (the policy's next admissions fill them), so the cache never shrinks.

Two uses (`mode`, the host's choice at launch):
- `ring` (a step of at least `ring_tokens` prompt tokens): every piece a layer misses streams into its half of a double
  buffer of lent slots, layer l + 1's while layer l computes (`expert_prefetch@1`, after the work enqueued so far:
  the store's `ring_copy` "sm" copies from the device's tables on a few SMs, no host sync; "dma" reads the copy list
  back once — the step's one host sync — and copies on copy engines); a layer's resolve reads them in place (its
  `ring` table), so its expert passes, up to 4,096 tokens, fetch nothing but what a pass's own admissions unmap
  (into the graphed steps' staging slots). Lending makes misses, so the halves are sized after it: `ROUNDS` rounds of
  `lend(twice the most misses of a layer, per class, less what is held)`, the last asking for the worst case (twice
  the layer's experts). Should a class's slots still fall short — a cache that can lend no more of it —, that class's
  layers stage instead (their ring empty, their stage list its lent slots: the device's choice, reported in
  `ring_ok`), the other classes streaming.
- `stage`: the layer's routed misses are staged into lent slots (the resolve's `stage` list), where the store's own
  staging slots, sized for graphed steps, would not hold a pass's: a pass's worth (`prefill`) per class — or, where
  the step's passes fetch their misses in `parts` (`moe.pipeline`, every pass of the step pipelined), a ring of two
  parts: the rounds' `lend(twice a part of a layer's most absent pieces)` (a pass misses no more of a layer than it
  lacks; FN §21.16), the stage list those slots over and over, each part's fetch after the part two before it ran
  (`DeviceResolver._pipelined`).

A ring step under `ring_lend = "tail"` (docs/dev/v1-moe-prefill.md D11) first lends the tail — call 0, the policy's
`lend_tail`: the experts holding the last n slots of the tail's class, whole —, then makes the rounds (calls
1…ROUNDS, unchanged): the range is the step's expert passes' (D10's buffers), so it is no pool, ring or stage slot
and is not counted as held; the experts' other slots and the empty slots outside it serve as X's do. After call 0 no
resident holds a slot of the range, so the rounds' picks lie outside it and no admission of the step maps one (its
victims are residents; the empty slots are the step's).

Each class's lent slots are pooled (`pool`), the empty slots first; a layer's role takes its ring slots from the pool
of its class at its half's offset, its pieces in expert order; its stage list is the pool's first `prefill` slots (a
staging ring of parts: its first `want`, over and over)."""
from __future__ import annotations

import torch

from .residency import RETURN

BUFFERS = 2                                   # the ring's halves: layer l streams while layer l − 1 computes
ROUNDS = 4                                    # `lend` calls of a ring step, 1…ROUNDS (one suffices for staging)
SHORT = 4                                     # the store's error bit: a step's lent slots fell short of its use


class Lent:
    def __init__(self, store, mode: str, tail: tuple[int, int] | None = None, parts: int = 1) -> None:
        # `tail`: (the class, its last n slots) a tail step lends first; each class's slots from `end` on are not lent
        # to the ring and staging (its arena's end, the tail's start); `parts`: a staging step's ring of parts
        self.store, self.mode, self.tail, self.parts = store, mode, tail, parts
        self.width: torch.Tensor | None = None      # a staging ring's slots per class (its `want`), once lent
        self.end = store.lend_ends
        if tail is not None:                # this step's tail (its own passes' size): the rest of the class lends
            self.end = self.end.clone()
            self.end[tail[0]] -= tail[1]

    # -- at the step's first resolve -----------------------------------------------------------------------------------
    def begin(self) -> None:
        s = self.store
        res, classes = s.residency.device, s._class_index                    # [G, E, R], [G, R]
        groups, experts, roles = res.shape
        s.n_before.copy_(s.n_empty)                                          # the empty slots serve the step first
        s.n_empty.zero_()
        s.lent_call.fill_(-1)
        if self.tail is not None:                                            # call 0: the tail's holders, whole
            cls, n = self.tail
            s.lend_needs[0, cls].fill_(n)                                     # (a fill: no host sync)
            s.lend_tail(cls, n)
        # Held slots per class: the empty slots found, outside the tail (its recount once, after call 0), then each
        # round's picks, which lie outside it (no resident holds a range slot after call 0). Counted as they come:
        # this runs host-bound at the step's first resolve, so a recount per round would cost the step its launches.
        if self.tail is None:
            held = torch.zeros(len(s.classes), dtype=torch.int64, device=res.device)
            held.index_add_(0, classes.reshape(-1), s.n_before.long().repeat_interleave(roles))
        else:
            held = self._held(classes)
        # an auxiliary model's classes are not lent to (their layers never borrow: `done`)
        # a ring's halves, or a staging ring's two parts, follow the misses lending makes: `ROUNDS` rounds
        sized = self.mode == "ring" or self.parts > 1
        stage = torch.where(s.lending_classes, s.prefill, 0)
        for k in range(1, 1 + (ROUNDS if sized else 1)):
            want = self._want(res, classes) if sized else stage
            if k == ROUNDS:                                                  # the last: the worst case
                want = torch.where(held < want, BUFFERS * -(-experts // self.parts), want)
            if self.parts > 1:                                               # (a pass's worth: no ring needed)
                want = want.clamp(max=s.prefill)
            s.lend_needs[k] = (want - held).clamp_min(0)
            picked = s.lend(s.lend_needs[k], k).long()
            for r in range(roles):
                held.index_add_(0, classes[:, r], picked.sum(1))
        want = self._want(res, classes) if sized else stage
        if self.parts > 1:
            want = want.clamp(max=s.prefill)
        # Per class: the ring where its lent slots hold both halves, else its layers stage into them (a small class —
        # a type few layers share, FN's IQ1_M gate/up of 3 layers — cannot lend twice a layer's misses: it no longer
        # stages every class). `ring_ok`: 1 every class streamed, 2 some, 0 none.
        ok = held >= want if self.mode == "ring" else torch.zeros_like(held, dtype=torch.bool)
        s.ring_ok.copy_((ok.all().int() + (ok.any() & ~ok.all()).int() * 2).view(1))
        s.errors.bitwise_or_(((~ok) & (held < (want if self.parts > 1 else stage))).any().int() * SHORT)
        self._pool(res, classes)
        # a pass's misses past the ring (an earlier pass's admissions unmapped them) go to the graphed steps' staging
        # slots; a staged class's to its pool's first ones (a ring of parts: its first `want`, over and over)
        graphed = torch.cat([s.stage, s.stage[:, -1:].expand(roles, s.prefill - s.stage.shape[1])], 1)
        at = torch.arange(s.prefill, device=res.device).expand(len(s.classes), -1)
        if self.parts > 1:
            self.width = torch.where(s.lending_classes, want, s.prefill)
            at = torch.where(s.lending_classes[:, None], at % want.clamp_min(1)[:, None], at)
        lent = s.pool.gather(1, at)[classes]
        lent = torch.where(lent >= 0, lent, graphed[None, :, -1:])         # short (SHORT): never past the arena
        ringed = ok[classes]                                                # [G, R]
        ringed[s.borrowing_groups:] = False                                 # (an auxiliary layer: none)
        s.stage_tab.copy_(torch.where(ringed[..., None], graphed[None], lent))
        if self.mode == "ring":
            self._ring(res, classes, want // BUFFERS, ringed)
            if s.ring_copy == "dma":            # the step's one host sync: the second copy waits, after the first
                fetch, n = s.ring_host
                fetch.copy_(s.ring_fetch, non_blocking=True)
                n.copy_(s.ring_n)
            for g in range(min(BUFFERS, groups)):
                self._start(g)
        s.stage_width.copy_(torch.where(ok.all(), s.staging, s.prefill).view(1))

    def _want(self, res: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
        """Per class: twice the most pieces a layer of it misses (the target's layers) — a part of them, in a staging
        ring of parts."""
        missing = (res < 0).sum(1)                                           # [G, R]
        missing[self.store.borrowing_groups:] = 0
        want = torch.zeros(len(self.store.classes), dtype=torch.int64, device=res.device)
        most = want.scatter_reduce_(0, classes.reshape(-1), missing.reshape(-1), "amax")
        return -(-most // self.parts) * BUFFERS

    def _lent(self, classes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per layer and role, its empty stack's slots and its experts' lent slots [G, R, 2E], and which of them serve
        the step's ring and staging: the empty slots it found and the lent experts', outside the tail."""
        s = self.store
        groups, roles, experts = s.empty.shape
        empty = torch.arange(experts, device=classes.device) < s.n_before[:, None, None]     # [G, 1, E]
        lent = (s.lent_call >= 0)[:, None, :]                                                # [G, 1, E]
        slots = torch.cat([s.empty, s.lent_slots.permute(0, 2, 1)], 2)                     # [G, R, 2E]
        ok = torch.cat([empty.expand(groups, roles, experts), lent.expand(groups, roles, experts)], 2)
        return slots, ok & (slots < self.end[classes][:, :, None])

    def _held(self, classes: torch.Tensor) -> torch.Tensor:
        """Per class, the lent slots serving the step's ring and staging."""
        _, ok = self._lent(classes)
        held = torch.zeros(len(self.store.classes), dtype=torch.int64, device=classes.device)
        return held.index_add_(0, classes[:, :, None].expand_as(ok).reshape(-1), ok.reshape(-1).long())

    def _pool(self, res: torch.Tensor, classes: torch.Tensor) -> None:
        """Each class's lent slots in `pool` (past them −1): the layers' empty slots, then the lent experts', outside
        the tail."""
        s = self.store
        groups, experts, roles = res.shape
        slots, ok = self._lent(classes)
        slots, ok = slots.reshape(-1), ok.reshape(-1)
        cls = classes[:, :, None].expand(groups, roles, 2 * experts).reshape(-1)
        mine = (cls[None] == torch.arange(len(s.classes), device=res.device)[:, None]) & ok[None]   # [C, N]
        sink = s.pool.shape[1] - 1
        s.pool.fill_(-1).scatter_(1, torch.where(mine, mine.cumsum(1) - 1, sink), slots.expand(len(mine), -1))

    def _ring(self, res: torch.Tensor, classes: torch.Tensor, half: torch.Tensor, ringed: torch.Tensor) -> None:
        """The ring slots of every layer's missing pieces (its half at `half` per class), and their copies' tables;
        none for a layer's role whose class's slots fell short (`ringed` [G, R])."""
        s = self.store
        groups, experts, roles = res.shape
        missing = (res < 0) & ringed[:, None, :]                             # [G, E, R]
        rank = missing.cumsum(1) - 1
        base = (torch.arange(groups, device=res.device) % BUFFERS)[:, None] * half[classes]   # [G, R]
        at = (base[:, None, :] + rank).clamp(0, s.pool.shape[1] - 1)
        ring = torch.where(missing, s.pool[classes[:, None, :].expand_as(at), at], -1)
        s.ring.copy_(ring)
        index = torch.where(missing, rank, experts).permute(0, 2, 1)         # [G, R, E]: each piece's entry
        table = torch.full((groups, roles, 2, experts + 1), -1, dtype=torch.int32, device=res.device)
        table[:, :, 0].scatter_(2, index, s.host_units.permute(0, 2, 1).contiguous())
        table[:, :, 1].scatter_(2, index, ring.permute(0, 2, 1).contiguous())
        s.ring_fetch.copy_(table[..., :experts])
        s.ring_n.copy_(missing.sum(1))

    # -- per layer ----------------------------------------------------------------------------------------------------
    def _start(self, g: int) -> None:
        """Layer g's missing pieces into its half of the ring, after the work enqueued so far."""
        s = self.store
        for r, cls in enumerate(s.layer_classes[g]):
            if s.ring_copy == "dma":                     # the copy list on the host, from the pinned host pool
                fetch, n = s.ring_host
                host = cls.host.view(torch.uint8, (len(cls.pieces), cls.unit_bytes))
                s.prefetch[cls.name].start(g, fetch[g, r], n[g, r:r + 1], host, cls.slots())
            else:
                s.prefetch[cls.name].start(g, s.ring_fetch[g, r], s.ring_n[g, r:r + 1], cls.units(), cls.slots())

    def sources(self, g: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Before layer g's first pass: its ring (landed) and stage tables."""
        s = self.store
        if self.mode == "ring":
            for cls in s.layer_classes[g]:
                s.prefetch[cls.name].wait(g)
        return s.ring[g], s.stage_tab[g]

    def done(self, g: int) -> None:
        """After layer g's expert passes: its ring forgotten, layer g + 2's started; after the target's last, the lent
        slots returned to their layers, empty. An auxiliary model's layers (after the target's in the map) never
        borrow: their passes over a step's prompt rows may not run at all (the MTP head's front-only catch-up), and
        the loan must not outlive the step."""
        s = self.store
        last = s.borrowing_groups - 1
        if self.mode == "ring":
            s.ring[g].fill_(-1)
            if g + BUFFERS <= last:
                self._start(g + BUFFERS)
        if g == last:
            self._give_back()

    def _give_back(self) -> None:
        """Every layer's lent experts' slots on top of its empty stack, in the order they were lent (call — the tail's
        0, then the rounds' —, then expert id: the order of their delta records); the empty slots it found stay below
        them, as they were."""
        s = self.store
        groups, experts, roles = s.lent_slots.shape
        lent = s.lent_call >= 0
        ids = torch.arange(experts, device=lent.device)
        key = torch.where(lent, s.lent_call * experts + ids, (1 + ROUNDS) * experts)
        order = key.argsort(dim=1, stable=True)
        rank = torch.empty_like(order).scatter_(1, order, ids.expand(groups, experts))
        at = torch.where(lent, s.n_before[:, None] + rank, experts)[:, None, :].expand(groups, roles, experts)
        stacks = torch.cat([s.empty, s.empty[..., :1]], 2)
        s.empty.copy_(stacks.scatter_(2, at, s.lent_slots.permute(0, 2, 1).contiguous())[..., :experts])
        s.n_empty.copy_(s.n_before + lent.sum(1))
        # A marker record per group (g, RETURN, 0) where the slots came back: an auxiliary layer's sub-steps (FN D12a)
        # resolve after it, and their admissions pop the returned slots (the host mirror returns them there).
        g = torch.arange(groups, device=lent.device)
        s.delta[g, s.n_delta.long()] = torch.stack([g, torch.full_like(g, RETURN), torch.zeros_like(g)], 1).int()
        s.n_delta.add_(1)
