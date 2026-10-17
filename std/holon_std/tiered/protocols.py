"""What a placement policy and a host tier are to the tiered store (docs/v1-offload.md D3). Plugins implement them
structurally, named by a `Ref` in an open plugin option (`moe.placement`, `moe.tiers`), without importing std.

Items are (group, item) pairs — a MoE layer and one of its experts —, each held in one slot per role (gate/up,
down), every role's slot in its own slot class (role, ggml type, shape), global across groups."""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Budget(Protocol):
    def take(self, group: int, item: int) -> bool:
        """Reserve a free cache slot for each of the item's roles, if every role's class has one."""

    def warm_start(self, profile: np.ndarray) -> list[tuple[int, int]]:
        """F2's order (every policy's warm start): each group's 4 hottest items — the hottest of every group, then
        the second, … —, then the rest by count across groups; ties by item, then group."""


class PlacementPolicy(Protocol):
    """Built with the store's group shape (`items`, `roles`: keyword arguments after the Ref's own)."""

    resolve_variant: str | None   # the resolve contract its device state needs; None: `moe_resolve@1`
    device_owned: bool            # the device's resolver owns the whole cache: the host seeds it, never edits it

    def device_state(self) -> dict[str, tuple[tuple[int, ...], str]]:
        """Device buffers its resolve contract keeps across steps, per group: name -> (shape, dtype). The store holds
        each as [groups, *shape], zeroed at every seed, and passes a group's slice to its resolve by name."""

    def seed(self, profile: np.ndarray, budget: Budget) -> list[tuple[int, int]]:
        """After every materialize: the items to make resident, each first `budget.take`n; `profile` is routing
        counts [groups, items] (`moe.placement.profile`; zeros where it names no group, or without one)."""

    def after_commit(self, counts: np.ndarray, view: np.ndarray) -> list[tuple[int, int, bool]]:
        """At each commit: the step's routing counts [groups, items] of decode and verify rows, and the residency map
        as the host knows it ([groups, items, roles] -> slot | −1 − tier: the device's as of this step); returns
        swaps (group, item, resident) — per group as many items in as out, the i-th in taking the i-th out's slots,
        unmapped at launch c + 2, mapped at c + 2 + d (two-phase, D4); none from a device-owned policy."""


class Tier(Protocol):
    device_readable: bool         # kernels read its units in place (a mapped host pool)

    def capacity(self, cls: str, pieces: int) -> int:
        """How many of class `cls`'s `pieces` (role pieces of every item) this tier holds: its host pool's units."""

    def admit(self, cls: str, piece: int, unit: int) -> None:
        """Piece `piece` of class `cls` is now held in this tier's unit `unit` (the store fills it)."""

    def evict(self, cls: str, piece: int) -> None: ...

    def read(self, cls: str, piece: int) -> tuple[str, int]:
        """Where the piece's bytes are: ("host_slot", unit) for a device-readable tier."""
