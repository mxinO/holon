"""The residency map: [groups, items, roles] -> the arena slot of each piece, or −1 − tier when it is not in VRAM
(the tier holding it). One device buffer the resolvers read, its host mirror, and host edits applied at a fixed
launch (docs/v1-offload.md D4: decided at commit c, applied at launch c + 2, unconditionally — the host owns the
regions it edits; the two-phase copy around an edit is its policy's)."""
from __future__ import annotations

import numpy as np
import torch

LAG = 2                       # launches between the commit deciding an edit and the launch applying it


class ResidencyMap:
    def __init__(self, ctx, name: str, shape: tuple[int, int, int]) -> None:
        self.host = np.full(shape, -1, dtype=np.int32)
        self.device = ctx.static(f"{name}.residency", shape, torch.int32)
        self._edits: dict[int, list[tuple[int, int, int, int]]] = {}   # launch -> (group, item, role, value)

    def seed(self) -> None:
        """The host mirror to the device now (after a materialize, before any step), dropping pending edits."""
        self._edits.clear()
        self.device.copy_(torch.from_numpy(self.host))

    def edit(self, commit: int, entries: list[tuple[int, int, int, int]]) -> None:
        """Entries (group, item, role, value) decided at the commit of step `commit`, applied at launch commit + LAG."""
        self._edits.setdefault(commit + LAG, []).extend(entries)

    def writes(self, launch: int) -> list[tuple[torch.Tensor, np.ndarray]]:
        """The device write (`before_launch`) applying the edits due at `launch`, if any; the host mirror with them."""
        due = self._edits.pop(launch, None)
        if not due:
            return []
        for g, i, r, value in due:
            self.host[g, i, r] = value
        return [(self.device, self.host.copy())]
