"""What a placement policy and a host tier are to the tiered store (docs/v1-offload.md D3). Plugins implement them
structurally, named by a `Ref` in an open plugin option (`moe.placement`, `moe.tiers`), without importing std.

Items are (group, item) pairs — a MoE layer and one of its experts —, each held in one slot per role (gate/up,
down), every role's slot in its own slot class (role, ggml type, shape), global across groups."""
from __future__ import annotations

from typing import Protocol

import numpy as np
from holon.api import Extent


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
    heat: str | None              # (optional) the device state an exclusive RAM tier ranks experts by (D10: "hotter")
    frozen: bool                  # (optional) a host policy that never swaps after the seed

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
    """A host tier of the store's pieces (docs/v1-offload.md D3, D10), fastest first in `moe.tiers`; it owns its
    storage. The first tier is device-readable (pinned RAM); a later one holds what the first does not (the
    checkpoint's file, F9). With every piece in the first, the tiers are inclusive (each piece also in RAM); with fewer,
    exclusive: a piece lives in exactly one of VRAM, RAM and the slower tier, and the store moves it in the graph."""

    device_readable: bool         # kernels read its units in place (a mapped host pool)

    def holds(self, cls: str, pieces: int) -> int:
        """How many of a class's `pieces` it holds at once (all: an inclusive tier). Known while kernels are bound;
        the store forwards a tier its settings as keyword arguments at construction (the RAM tier's `ram`)."""

    def prepare(self, ctx, cls: str, unit_bytes: int, extents: list[list[Extent]], staging: int = 0,
                granule: int = 0) -> None:
        """Before the first materialize, once per class: its pieces, piece i the concatenation of `extents[i]`
        (`unit_bytes` in all). The tier takes its storage for them: a device-readable one a mapped host pool (units of
        `granule` bytes, a divisor of `unit_bytes`, so the store's MemOps copy whole units) holding `holds` pieces and
        `staging` more (where the slower tiers' reads land), filled when it holds every piece (piece i at i); a file
        tier its descriptors."""

    def units(self, cls: str):
        """A device-readable tier's pool of the class (`BuildCtx.pool`'s): its first `holds` pieces' units are RAM,
        the `staging` after them where slower reads land."""

    def read(self, cls: str, piece: int) -> tuple:
        """Where piece's bytes are: ("host_slot", unit) in a device-readable tier that holds every piece; in a file
        tier always ("pread", ((fd, direct, offset, nbytes), …)), its extents in order, each read with O_DIRECT or
        not (`direct`: through the page cache when 0) — the one kind `cpu_task@1`'s copy task reads (the store
        refuses any other at prepare)."""
