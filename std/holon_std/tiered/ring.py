"""A prefill step's lent slots (docs/v1-offload.md D4, D5, D8): at the step's first resolve the policy's `lend(n)`
unmaps whole experts — the coldest — and their slots serve the step; at its end they return to their layers as empty
slots, each its layer's first victim (the policy's next admissions fill them), so the cache never shrinks.

Two uses (`mode`):
- `ring` (a step of at least `ring_tokens` prompt tokens): every piece a layer misses streams into a double buffer of
  lent slots, layer l + 1's while layer l computes, on a copy engine (`expert_prefetch@1`); a layer's resolve reads them
  in place (its `ring` table), so its expert passes, up to 4,096 tokens, fetch nothing.
- `stage`: the layer's routed misses are staged into lent slots (the resolve's `stage` list) where the store's own
  staging slots, sized for graphed steps, would not hold a pass's.

The step synchronizes once or twice at its start (the lending is the device's: the host reads the map back to know
the slots), never after."""
from __future__ import annotations

import numpy as np
import torch

BUFFERS = 2                                   # the ring's halves: layer l streams while layer l − 1 computes


class Lent:
    def __init__(self, store, mode: str) -> None:
        self.store, self.mode = store, mode
        self.lends: list[int] = []            # the `lend(n)` calls made, in order (the parity oracle's input)

    # -- at the step's first resolve -----------------------------------------------------------------------------------
    def begin(self) -> None:
        store = self.store
        groups, classes = len(store.layers), store.class_table
        before = store.residency.device.cpu().numpy().copy()        # a sync: the device's map, every step before done
        empty, n_empty = store.empty.cpu().numpy().copy(), store.n_empty.cpu().numpy().copy()
        store.n_empty.zero_()                                        # the empty slots first: they serve the step too
        self.slots = {c: [] for c in range(classes.max() + 1)}       # the lent slots per class
        self.owners = []                                             # (layer, its slots per role), as they return
        for g in range(groups):
            for j in range(n_empty[g]):
                self._own(g, empty[g, :, j], classes)
        lent = np.zeros(before.shape[:2], dtype=bool)
        n = self._want(before, classes)
        for _ in range(4):                                           # lending makes misses: lend until they fit
            short = n - self._held()
            if (short <= 0).all():
                break
            self.lends.append(int(short.max()))
            picked = store.lend(self.lends[-1]).cpu().numpy()
            for g, e in zip(*np.nonzero(picked & ~lent), strict=True):
                self._own(int(g), before[g, e], classes)
            lent |= picked
            n = self._want(np.where(lent[..., None], -1, before), classes)
        held, per = self._held(), n // BUFFERS if self.mode == "ring" else n
        if self.mode == "ring" and (held < per).any():               # not even one layer's worth: stage instead
            self.mode, per = "stage", self._need_stage(classes)
        if (held < per).any():
            raise RuntimeError(f"{store.name}: {held.tolist()} slots lent per class, {per.tolist()} needed (a cache "
                               "too small to stage a prefill pass)")
        self.after, self.half = np.where(lent[..., None], -1, before), per
        self.buffers = BUFFERS if (held >= BUFFERS * per).all() else 1
        if self.mode == "stage":                                     # a layer's stage list: its roles' classes' slots
            width = store._width[store._step] = int(per.min())
            self.stages = {key: _on([self.slots[c][:width] for c in key], store.ctx.device)
                           for key in {tuple(row) for row in classes.tolist()}}
            return
        self.pairs: dict[int, np.ndarray] = {}
        for g in range(min(self.buffers, groups)):
            self._start(g)

    def _own(self, g: int, slots: np.ndarray, classes: np.ndarray) -> None:
        """Slots (one per role) of layer g serve the step, and return to g at its end."""
        for r, slot in enumerate(slots.tolist()):
            self.slots[classes[g, r]].append(int(slot))
        self.owners.append((g, np.asarray(slots)))

    def _held(self) -> np.ndarray:
        return np.array([len(s) for s in self.slots.values()], dtype=np.int64)

    def _want(self, residency: np.ndarray, classes: np.ndarray) -> np.ndarray:
        """Slots per class the step needs: two layers' worth of misses (ring) or a pass's staging (stage)."""
        if self.mode == "stage":
            return self._need_stage(classes)
        want = np.zeros(classes.max() + 1, dtype=np.int64)
        for g in range(len(residency)):
            for r in range(classes.shape[1]):
                c = classes[g, r]
                want[c] = max(want[c], BUFFERS * int((residency[g, :, r] < 0).sum()))
        return want

    def _need_stage(self, classes: np.ndarray) -> np.ndarray:
        return np.full(classes.max() + 1, self.store.prefill, dtype=np.int64)

    # -- per layer ----------------------------------------------------------------------------------------------------
    def _start(self, g: int) -> None:
        """Layer g's missing pieces into its half of the ring, after the work enqueued so far."""
        store, half = self.store, g % self.buffers
        classes = store.class_table[g].tolist()
        ring = np.full(self.after.shape[1:], -1, dtype=np.int32)
        for r, c in enumerate(classes):
            cls = store.layer_classes[g][r]
            base = half * int(self.half[c])
            missing = np.flatnonzero(self.after[g, :, r] < 0)
            ring[missing, r] = np.asarray(self.slots[c][base:base + len(missing)])
            pairs = np.stack([store._host[g, missing, r], ring[missing, r]], 1)
            store.prefetch[cls.name].start(g, pairs, cls.host.view(torch.uint8, (len(cls.pieces), cls.unit_bytes)),
                                           cls.slots())
        self.pairs[g] = ring

    def sources(self, g: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Before layer g's first pass: its ring (landed) and stage tables."""
        store = self.store
        if self.mode == "stage":
            return store.ring[g], self.stages[tuple(store.class_table[g].tolist())]
        for r in range(store.roles):
            store.prefetch[store.layer_classes[g][r].name].wait(g)
        store.ring[g].copy_(_on(self.pairs.pop(g), store.ctx.device), non_blocking=True)
        return store.ring[g], store.stage

    def done(self, g: int) -> None:
        """After layer g's expert passes: its ring forgotten, layer g + 2's started; after the last, the lent slots
        returned to their layers, empty."""
        store = self.store
        if self.mode == "ring":
            store.ring[g].fill_(-1)
            if g + self.buffers < len(store.layers):
                self._start(g + self.buffers)
        if g == len(store.layers) - 1:
            store.give_back(self.owners)


def _on(values, device: torch.device) -> torch.Tensor:
    """Host ints on the device through pinned memory: an asynchronous copy, no host sync."""
    t = torch.as_tensor(np.asarray(values), dtype=torch.int32)
    return t.pin_memory().to(device, non_blocking=True) if device.type == "cuda" else t
