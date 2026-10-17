"""std.tiered.store@1: the engine's one tiered store of offloaded pieces (docs/v1-offload.md D5) — slot classes with
VRAM arenas and host tiers, the residency map, the placement policy, the per-step row kinds and counts.

A client (the MoE offload store, `std.moe.offload@1`) adds every piece while its model is finalized — (group, item,
role), its class and the checkpoint extents its bytes come from — and asks for each group's `DeviceResolver` when it
binds kernels. Then, through the engine's hooks:

- `before_materialize`: per class, an arena `[S, unit]` in VRAM — `top_k` null slots (zero weights, one per routing
  rank), `staging` slots, then cache slots: a fill pool growing to every piece, or the size `cache` pins — and a mapped
  host pool per tier, filled from the extents;
- `after_materialize` (after every materialize): null slots zeroed, the map reset to the tiers, the policy's seed
  admitted and copied into cache slots;
- `before_launch`: the step's row kinds per token (decode and verify rows: a mixed step's leading rows, or a row
  whose every query token has logits; prefill rows the others, so a one-token last prompt chunk counts as decode;
  graph padding past them), its counts zeroed, the map edits due;
- `after_commit`: the staging invariant (a layer's distinct misses fit its staging slots: F5a has no fallback), then
  the policy's edits from the step's counts.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np
import torch
from holon.api import Extent, Ref

from . import files
from .residency import ResidencyMap
from .resolver import DeviceResolver

PADDING, DECODE, PREFILL = 0, 1, 2           # row kinds (std.moe.kernels, `moe_resolve@1`)


@dataclasses.dataclass(eq=False)
class SlotClass:
    """Pieces of one role, ggml type and shape, from every group: one arena, one host pool per tier."""

    name: str
    unit_bytes: int
    null: int                                   # null slots [0, null), then staging slots
    staging: int
    pieces: list[tuple[int, int, int, list[Extent]]] = dataclasses.field(default_factory=list)  # group, item, role
    cache: int | None = None                    # pinned cache slots (None: fill)
    arena: object = None                        # its pools (`BuildCtx.pool`), from `before_materialize`
    host: object = None

    @property
    def max_slots(self) -> int:
        """Its arena's size when every piece is cached: what kernels over it are bound for."""
        return self.null + self.staging + len(self.pieces)

    def slots(self) -> torch.Tensor:
        """The arena as [S, unit] bytes (valid until the next materialize)."""
        return self.arena.view(torch.uint8, (self.arena.capacity, self.unit_bytes))

    def units(self) -> torch.Tensor:
        """The host pool as the device reads it, [pieces, unit] bytes."""
        return self.host.device_view(torch.uint8, (self.host.capacity, self.unit_bytes))


class TieredStore:
    def __init__(self, ctx, *, name: str, items: int, roles: int, top_k: int, staging: int, policy: Ref | str,
                 tiers: tuple[str, ...] = ("std.tiered.pinned@1",), cache: str = "", profile: str = "") -> None:
        self.ctx, self.name, self.items, self.roles, self.top_k, self.staging = ctx, name, items, roles, top_k, staging
        self.policy = ctx.unit(policy if isinstance(policy, Ref) else Ref(policy))
        if self.policy.resolve_variant is not None or self.policy.device_state():
            raise NotImplementedError(f"{name}: policies with resolve variants or device state land in step 3")
        self.tiers = [ctx.unit(Ref(t)) for t in tiers]
        if len(self.tiers) != 1 or not self.tiers[0].device_readable:
            raise NotImplementedError(f"{name}: one device-readable host tier (F5a), not {list(tiers)}")
        self._cache, self._profile = _parse_cache(cache), profile
        self.classes: dict[str, SlotClass] = {}
        self.groups: dict[int, dict[int, SlotClass]] = {}       # group -> role -> its class
        self.residency: ResidencyMap | None = None              # the device tables, once a resolver is asked for
        tokens = max(ctx.options.get("max_step_tokens", 1),
                     ctx.options.get("max_seqs", 1) * (1 + ctx.options.get("max_draft_tokens", 0)))
        self._kinds = np.zeros(tokens, dtype=np.uint8)
        self.kinds = ctx.static(f"{name}.kinds", (tokens,), torch.uint8)
        self.misses = ctx.static(f"{name}.misses", (1,), torch.int32)
        self.peak_misses = 0                    # the most distinct misses a layer had (`misses`, at commits)

    # -- while the model is finalized and bound ------------------------------------------------------------------------
    def add(self, group: int, item: int, role: int, cls: str, unit_bytes: int, extents: list[Extent]) -> SlotClass:
        """One piece: role `role` of item `item` of group `group`, in class `cls` (`unit_bytes` each), its bytes the
        concatenation of `extents`."""
        if self.residency is not None:
            raise RuntimeError(f"{self.name}: pieces added after kernels were bound (an auxiliary model's?)")
        c = self.classes.setdefault(cls, SlotClass(cls, unit_bytes, self.top_k, self.staging,
                                                   cache=self._cache.get(cls, self._cache.get("*"))))
        if c.unit_bytes != unit_bytes or sum(e.nbytes for e in extents) != unit_bytes:
            raise ValueError(f"{self.name}: class {cls} has {c.unit_bytes}-byte units, not {unit_bytes} "
                             f"({sum(e.nbytes for e in extents)} in the extents)")
        if self.groups.setdefault(group, {}).setdefault(role, c) is not c:
            raise ValueError(f"{self.name}: group {group}'s role {role} in two classes")
        c.pieces.append((group, item, role, extents))
        return c

    def resolver(self, group: int) -> DeviceResolver:
        if self.residency is None:
            self._order = {g: n for n, g in enumerate(sorted(self.groups))}
            shape = (len(self._order), self.items, self.roles)
            self.residency = ResidencyMap(self.ctx, self.name, shape)
            self.host_units = self.ctx.static(f"{self.name}.host", shape, torch.int32)
            self.counts = self.ctx.static(f"{self.name}.counts", shape[:2], torch.int32)
            self.telemetry = {"counts": self.counts, "misses": self.misses}
        g = self._order[group]
        return DeviceResolver(self.ctx, self, [self.groups[group][r] for r in range(self.roles)],
                              self.residency.device[g], self.host_units[g], self.counts[g])

    # -- engine hooks -------------------------------------------------------------------------------------------------
    def before_materialize(self, ctx) -> None:
        """Request every class's arena and host pool, and fill the host pools from the checkpoint."""
        jobs, host = [], np.full(self.residency.host.shape, -1, dtype=np.int32)
        for c in self.classes.values():
            base, n = c.null + c.staging, len(c.pieces)
            pref = n if c.cache is None else c.cache if isinstance(c.cache, int) else round(c.cache * n)
            if not 0 <= pref <= n:
                raise ValueError(f"{self.name}: class {c.name}: {pref} cache slots of {n} pieces")
            c.arena = ctx.pool(f"{self.name}.arena.{c.name}", unit_bytes=c.unit_bytes,
                               min_units=base + (0 if c.cache is None else pref), pref_units=base + pref,
                               fill=c.cache is None)
            units = self.tiers[0].capacity(c.name, n)
            c.host = ctx.pool(f"{self.name}.host.{c.name}", unit_bytes=c.unit_bytes, min_units=units,
                              pref_units=units, device="host", mapped=True)
            buffer = c.host.view(torch.uint8, (units, c.unit_bytes)).numpy()
            for piece, (g, i, r, extents) in enumerate(c.pieces):
                self.tiers[0].admit(c.name, piece, piece)
                _, host[self._order[g], i, r] = self.tiers[0].read(c.name, piece)
                at = 0
                for e in extents:
                    jobs.append((e, buffer[piece, at:at + e.nbytes]))
                    at += e.nbytes
        files.read(jobs)
        self._host = host
        self.host_units.copy_(torch.from_numpy(host))

    def after_materialize(self, ctx) -> None:
        """Reset the device state of the new pools: null slots, the map at the tiers, the policy's seed cached."""
        free, arenas, units = {}, {}, {}
        for c in self.classes.values():
            arenas[c.name], units[c.name] = c.slots(), c.host.view(torch.uint8, (c.host.capacity, c.unit_bytes))
            arenas[c.name].zero_()                          # the null slots; no slot left holding garbage
            free[c.name] = list(range(c.arena.capacity - 1, c.null + c.staging - 1, -1))   # popped lowest first
        self.residency.host[...] = -1                   # −1 − tier 0: every piece in the host tier
        groups = list(self._order)
        for g, i in self.policy.seed(self._profile_counts(), _Budget(self, free, groups)):
            for r, c in self.groups[groups[g]].items():
                slot = self.residency.host[g, i, r] = free[c.name].pop()
                arenas[c.name][slot].copy_(units[c.name][self._host[g, i, r]], non_blocking=True)
        self.residency.seed()
        for t in (self.counts, self.misses, self.kinds):
            t.zero_()

    def before_launch(self, step) -> tuple[list[tuple[torch.Tensor, np.ndarray]], list]:
        kinds = self._kinds
        kinds[:] = PADDING
        at = 0
        for row, (q, logits) in enumerate(zip(step.n_query.tolist(), step.n_logits.tolist(), strict=True)):
            decode = row < step.n_decode or (not step.n_decode and q == logits)   # decode and verify rows
            kinds[at:at + q] = DECODE if decode else PREFILL
            at += q
        return [(self.kinds, kinds), (self.counts, np.zeros(self.counts.shape, dtype=np.int32)),
                *self.residency.writes(step.step)], []

    def after_commit(self, step, telemetry) -> None:
        self.peak_misses = int(telemetry["misses"][0])
        if self.peak_misses > self.staging:
            raise RuntimeError(f"{self.name}: a layer had {self.peak_misses} distinct misses, {self.staging} staging "
                               "slots")
        if edits := self.policy.after_commit(telemetry["counts"].numpy(), self.residency.host):
            raise NotImplementedError(f"{self.name}: the policy edits the map ({len(edits)} edits): two-phase edits "
                                      "land in step 3")

    # -- helpers ------------------------------------------------------------------------------------------------------
    def _profile_counts(self) -> np.ndarray:
        shape = self.residency.host.shape[:2]
        if not self._profile:
            return np.zeros(shape)
        counts = np.asarray(json.loads(pathlib.Path(self._profile).read_text())["counts"], dtype=np.float64)
        if counts.shape != shape:
            raise ValueError(f"{self.name}: profile {self._profile} counts {counts.shape}, not [groups, items] {shape}")
        return counts


class _Budget:
    """Free cache slots per class while a policy seeds."""

    def __init__(self, store: TieredStore, free: dict[str, list[int]], groups: list[int]) -> None:
        self.store, self.free, self.groups = store, {name: len(slots) for name, slots in free.items()}, groups

    def take(self, group: int, item: int) -> bool:
        classes = [c.name for c in self.store.groups[self.groups[group]].values()]
        if any(self.free[c] < classes.count(c) for c in classes):
            return False
        for c in classes:
            self.free[c] -= 1
        return True


def _parse_cache(text: str) -> dict[str, int | float]:
    """`cache`: `<class>=<slots>` items, comma-separated, a slot count or (with a point) a fraction of the class's
    pieces; class `*` for the classes not named. Empty: every class fills free VRAM up to all its pieces."""
    out = {}
    for item in filter(None, (s.strip() for s in text.split(","))):
        name, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"cache item {item!r}: expected <class>=<slots or fraction>")
        out[name.strip()] = float(value) if "." in value else int(value)
    return out
