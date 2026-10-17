"""A prefill step's lent slots (docs/v1-offload.md D4, D5, D8), all on the device — tables written by the step's
own kernels, no host sync: at the step's first resolve its layers' empty slots, then the policy's `lend(need)` —
whole experts, the coldest — serve the step; at its end they return to their layers as empty slots, each its
layer's first victim (the policy's next admissions fill them), so the cache never shrinks.

Two uses (`mode`, the host's choice at launch):
- `ring` (a step of at least `ring_tokens` prompt tokens, on a layout whose every class can lend twice a layer's
  experts): every piece a layer misses streams into its half of a double buffer of lent slots, layer l + 1's while
  layer l computes (`expert_prefetch@1`, after the work enqueued so far); a layer's resolve reads them in place (its
  `ring` table), so its expert passes, up to 4,096 tokens, fetch nothing but what a pass's own admissions unmap
  (into the graphed steps' staging slots). Lending makes misses, so the halves are sized after it: `ROUNDS` rounds of
  `lend(twice the most misses of a layer, per class, less what is held)`, the last asking for the worst case (twice
  the layer's experts). Should the slots still fall short — a cache that can lend no more —, the step stages instead
  (its ring empty, its stage list the lent slots: the device's choice, reported in `ring_ok`).
- `stage`: the layer's routed misses are staged into lent slots (the resolve's `stage` list), where the store's own
  staging slots, sized for graphed steps, would not hold a pass's.

Each class's lent slots are pooled (`pool`), the empty slots first; a layer's role takes its ring slots from the pool
of its class at its half's offset, its pieces in expert order; its stage list is the pool's first `prefill` slots."""
from __future__ import annotations

import torch

BUFFERS = 2                                   # the ring's halves: layer l streams while layer l − 1 computes
ROUNDS = 4                                    # `lend` calls of a ring step (one suffices for staging)
SHORT = 4                                     # the store's error bit: a step's lent slots fell short of its use


class Lent:
    def __init__(self, store, mode: str) -> None:
        self.store, self.mode = store, mode

    # -- at the step's first resolve -----------------------------------------------------------------------------------
    def begin(self) -> None:
        s = self.store
        res, classes = s.residency.device, s._class_index                    # [G, E, R], [G, R]
        groups, experts, roles = res.shape
        s.n_before.copy_(s.n_empty)                                          # the empty slots serve the step first
        s.n_empty.zero_()
        s.lent_call.fill_(-1)
        held = torch.zeros(len(s.classes), dtype=torch.int64, device=res.device)
        held.index_add_(0, classes.reshape(-1), s.n_before.long().repeat_interleave(roles))
        stage = torch.full_like(held, s.prefill)
        for k in range(ROUNDS if self.mode == "ring" else 1):
            want = self._want(res, classes) if self.mode == "ring" else stage
            if k == ROUNDS - 1:                                              # the last: the worst case
                want = torch.where(held < want, 2 * experts, want)
            s.lend_needs[k] = (want - held).clamp_min(0)
            picked = s.lend(s.lend_needs[k], k).long()
            for r in range(roles):
                held.index_add_(0, classes[:, r], picked.sum(1))
        want = self._want(res, classes) if self.mode == "ring" else stage
        ring_ok = (held >= want).all() if self.mode == "ring" else torch.zeros((), dtype=torch.bool, device=res.device)
        s.ring_ok.copy_(ring_ok.view(1))
        s.errors.bitwise_or_((~ring_ok & (held < stage).any()).int() * SHORT)
        self._pool(res, classes)
        # a pass's misses past the ring (an earlier pass's admissions unmapped them) go to the graphed steps' staging
        # slots; a stage step's to the pool's first ones
        graphed = torch.cat([s.stage, s.stage[:, -1:].expand(roles, s.prefill - s.stage.shape[1])], 1)
        lent = s.pool[:, :s.prefill][classes]
        lent = torch.where(lent >= 0, lent, graphed[None, :, -1:])         # short (SHORT): never past the arena
        s.stage_tab.copy_(torch.where(ring_ok, graphed[None], lent))
        if self.mode == "ring":
            self._ring(res, classes, want // BUFFERS, ring_ok)
            for g in range(min(BUFFERS, groups)):
                self._start(g)
        s.stage_width.copy_(torch.where(ring_ok, s.staging, s.prefill).view(1))

    def _want(self, res: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
        """Per class: twice the most pieces a layer of it misses."""
        missing = (res < 0).sum(1)                                           # [G, R]
        want = torch.zeros(len(self.store.classes), dtype=torch.int64, device=res.device)
        return want.scatter_reduce_(0, classes.reshape(-1), missing.reshape(-1), "amax") * BUFFERS

    def _pool(self, res: torch.Tensor, classes: torch.Tensor) -> None:
        """Each class's lent slots in `pool` (past them −1): the layers' empty slots, then the lent experts'."""
        s = self.store
        groups, experts, roles = res.shape
        empty = torch.arange(experts, device=res.device) < s.n_before[:, None, None]         # [G, 1, E]
        lent = (s.lent_call >= 0)[:, None, :]                                                # [G, 1, E]
        slots = torch.cat([s.empty, s.lent_slots.permute(0, 2, 1)], 2).reshape(-1)         # [G, R, 2E]
        ok = torch.cat([empty.expand(groups, roles, experts), lent.expand(groups, roles, experts)], 2).reshape(-1)
        cls = classes[:, :, None].expand(groups, roles, 2 * experts).reshape(-1)
        mine = (cls[None] == torch.arange(len(s.classes), device=res.device)[:, None]) & ok[None]   # [C, N]
        sink = s.pool.shape[1] - 1
        s.pool.fill_(-1).scatter_(1, torch.where(mine, mine.cumsum(1) - 1, sink), slots.expand(len(mine), -1))

    def _ring(self, res: torch.Tensor, classes: torch.Tensor, half: torch.Tensor, ring_ok: torch.Tensor) -> None:
        """The ring slots of every layer's missing pieces (its half at `half` per class), and their copies' tables;
        none when the slots fell short."""
        s = self.store
        groups, experts, roles = res.shape
        missing = (res < 0) & ring_ok                                        # [G, E, R]
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
            s.prefetch[cls.name].start(g, s.ring_fetch[g, r], s.ring_n[g, r:r + 1], cls.units(), cls.slots())

    def sources(self, g: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Before layer g's first pass: its ring (landed) and stage tables."""
        s = self.store
        if self.mode == "ring":
            for cls in s.layer_classes[g]:
                s.prefetch[cls.name].wait(g)
        return s.ring[g], s.stage_tab[g]

    def done(self, g: int) -> None:
        """After layer g's expert passes: its ring forgotten, layer g + 2's started; after the last, the lent slots
        returned to their layers, empty."""
        s = self.store
        if self.mode == "ring":
            s.ring[g].fill_(-1)
            if g + BUFFERS < len(s.layers):
                self._start(g + BUFFERS)
        if g == len(s.layers) - 1:
            self._give_back()

    def _give_back(self) -> None:
        """Every layer's lent experts' slots on top of its empty stack, in the order they were lent (`lend` call,
        then expert id: the order of their delta records)."""
        s = self.store
        groups, experts, roles = s.lent_slots.shape
        lent = s.lent_call >= 0
        ids = torch.arange(experts, device=lent.device)
        key = torch.where(lent, s.lent_call * experts + ids, ROUNDS * experts)
        order = key.argsort(dim=1, stable=True)
        rank = torch.empty_like(order).scatter_(1, order, ids.expand(groups, experts))
        at = torch.where(lent, s.n_before[:, None] + rank, experts)[:, None, :].expand(groups, roles, experts)
        stacks = torch.cat([s.empty, s.empty[..., :1]], 2)
        s.empty.copy_(stacks.scatter_(2, at, s.lent_slots.permute(0, 2, 1).contiguous())[..., :experts])
        s.n_empty.copy_(s.n_before + lent.sum(1))
