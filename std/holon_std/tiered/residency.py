"""The residency map: [groups, items, roles] -> the arena slot of each piece, or −1 − tier when it is not in VRAM
(the tier holding it). One device buffer the resolvers read, its host mirror, and host edits applied at a fixed
launch (docs/v1-offload.md D4: decided at commit c, applied at launch c + 2, unconditionally — the host owns the
regions it edits; the two-phase copy around an edit is its policy's).

An edit reaches the device as its own (index, value) pairs, written before the launch into an edit list the step's
forward scatters into the map (`apply`): entries it does not name keep what the device holds, so regions a device
resolver owns (its admissions) are never overwritten by the host's older view."""
from __future__ import annotations

import numpy as np
import torch

LAG = 2                       # launches between the commit deciding an edit and the launch applying it


class ResidencyMap:
    def __init__(self, ctx, name: str, shape: tuple[int, int, int]) -> None:
        size = int(np.prod(shape))
        self.host = np.full(shape, -1, dtype=np.int32)
        self._flat = ctx.static(f"{name}.residency", (size + 1,), torch.int32)   # the last entry takes no-ops
        self.device = self._flat[:size].view(shape)
        self._list = ctx.static(f"{name}.residency_edits", (2, size), torch.int64)  # (indices, values) of a launch
        self._edits: dict[int, dict[int, int]] = {}            # launch -> flat index -> value (the last one wins)

    def seed(self) -> None:
        """The host mirror to the device now (after a materialize, before any step), dropping pending edits."""
        self._edits.clear()
        self.device.copy_(torch.from_numpy(self.host))
        self._list[0].fill_(len(self._flat) - 1)

    def edit(self, commit: int, entries: list[tuple[int, int, int, int]]) -> None:
        """Entries (group, item, role, value) decided at the commit of step `commit`, applied at launch commit + LAG."""
        due = self._edits.setdefault(commit + LAG, {})
        for g, i, r, value in entries:
            due[int(np.ravel_multi_index((g, i, r), self.host.shape))] = value

    def writes(self, launch: int) -> list[tuple[torch.Tensor, np.ndarray]]:
        """The device write (`before_launch`) of the edit list due at `launch`, if any; the host mirror with them."""
        due = self._edits.pop(launch, None)
        if not due:
            return []
        pairs = np.array([list(due), list(due.values())], dtype=np.int64)
        self.host.reshape(-1)[pairs[0]] = pairs[1]
        return [(self._list[:, :len(due)], pairs)]

    def apply(self) -> None:
        """In the step's forward, before any resolver reads the map: scatter the launch's edit list, then empty it
        (graph-safe: fixed shapes, unused entries aim at the no-op entry)."""
        self._flat.scatter_(0, self._list[0], self._list[1].int())
        self._list[0].fill_(len(self._flat) - 1)
