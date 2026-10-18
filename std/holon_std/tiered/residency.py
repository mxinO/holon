"""The residency map: [groups, items, roles] -> the arena slot of each piece, or −1 − tier when it is not in VRAM
(the tier holding it). One device buffer the resolvers read, its host mirror, and host edits applied at a fixed
launch (docs/v1-offload.md D4: decided at commit c, applied at launch c + 2 — or later, a two-phase edit's second
half —, each with a precondition: the value the host expects the entry to hold; one that fails is not applied and
raises the device error word the commit checks, so a host edit never lands where a device resolver changed the map).

An edit reaches the device as its own (index, expected, value) triples, written before the launch into an edit list
the step's forward scatters into the map (`apply`): entries it does not name keep what the device holds, so regions a
device resolver owns are never overwritten by the host's older view. The host mirror is the device's map as of the
last commit: host edits launched at or before it, then the device's own changes that step logged (`commit`) — the
same on the overlapped and the synchronous paths."""
from __future__ import annotations

import numpy as np
import torch

LAG = 2                       # launches between the commit deciding an edit and the launch applying it
OWNER = 1                     # the edit error word's bit: a precondition failed


class ResidencyMap:
    def __init__(self, ctx, name: str, shape: tuple[int, int, int]) -> None:
        size = int(np.prod(shape))
        self.host = np.full(shape, -1, dtype=np.int32)
        self._flat = ctx.static(f"{name}.residency", (size + 1,), torch.int32)   # the last entry takes no-ops
        self.device = self._flat[:size].view(shape)
        self._list = ctx.static(f"{name}.residency_edits", (3, size), torch.int64)  # (index, expected, value)
        self._edits: dict[int, dict[int, tuple[int, int]]] = {}    # launch -> flat index -> (expected, value)
        self._launched: dict[int, np.ndarray] = {}                # launch -> its triples, until a commit mirrors them
        self.empty: list[list[np.ndarray]] = [[] for _ in range(shape[0])]   # per group: its empty slots (a stack)

    def seed(self) -> None:
        """The host mirror to the device now (after a materialize, before any step), dropping pending edits."""
        self._edits.clear()
        self._launched.clear()
        self.empty = [[] for _ in self.empty]
        self.device.copy_(torch.from_numpy(self.host))
        self._flat[-1] = 0
        self._clear()

    def edit(self, commit: int, entries: list[tuple[int, int, int, int, int]], later: int = 0) -> None:
        """Entries (group, item, role, expected, value) decided at the commit of step `commit`, applied at launch
        commit + LAG + `later`; an entry edited twice there keeps its first precondition and its last value."""
        due = self._edits.setdefault(commit + LAG + later, {})
        for g, i, r, expected, value in entries:
            index = int(np.ravel_multi_index((g, i, r), self.host.shape))
            due[index] = (due.get(index, (expected,))[0], value)

    def writes(self, launch: int) -> list[tuple[torch.Tensor, np.ndarray]]:
        """The device write (`before_launch`) of the edit list due at `launch`, if any."""
        due = self._edits.pop(launch, None)
        if not due:
            return []
        triples = np.array([list(due), *zip(*due.values(), strict=True)], dtype=np.int64)
        self._launched[launch] = triples
        return [(self._list[:, :len(due)], triples)]

    def apply(self, errors: torch.Tensor) -> None:
        """In the step's forward, before any resolver reads the map: scatter the launch's edits whose precondition
        holds, raise `errors` (OWNER) for the others, then empty the list (graph-safe: fixed shapes, unused entries
        aim at the no-op entry)."""
        index, expected, value = self._list
        now = self._flat[index]
        ok = now == expected.int()
        errors.bitwise_or_((~ok).any().int() * OWNER)
        self._flat.scatter_(0, index, torch.where(ok, value.int(), now))
        self._clear()

    def commit(self, step: int, records: np.ndarray, returned: bool = False) -> None:
        """The host mirror after step `step`: the host edits launched by it, then the step's delta log `records`
        ((group, a, b): a took b's slots, b unmapped; (group, a, −2 − j): a took the group's empty slots j, popped;
        (group, −1, b): b unmapped, lent), then — a step that `returned` its lent slots — every lent item's slots on
        top of its group's empty stack, in the order of their records."""
        flat = self.host.reshape(-1)
        for launch in sorted(x for x in self._launched if x <= step):
            index, _, value = self._launched.pop(launch)
            flat[index] = value
        back: list[list[np.ndarray]] = [[] for _ in self.empty]
        for g, a, b in records.tolist():
            if b <= -2:
                self.host[g, a] = self.empty[g][-2 - b]
                del self.empty[g][-2 - b:]
                continue
            if a >= 0:
                self.host[g, a] = self.host[g, b]
            elif returned:
                back[g].append(self.host[g, b].copy())
            self.host[g, b] = -1
        for g, slots in enumerate(back):
            self.empty[g] += slots

    def _clear(self) -> None:
        self._list[0].fill_(len(self._flat) - 1)
        self._list[1:].zero_()
