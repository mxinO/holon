"""std.tiered.store@1: the engine's one tiered store of offloaded pieces (docs/v1-offload.md D5) — slot classes with
VRAM arenas and host tiers, the residency map, the placement policy, the per-step row kinds and counts.

A client (the MoE offload store, `std.moe.offload@1`) adds every piece while its model is finalized — (group, item,
role), its class and the checkpoint extents its bytes come from — and asks for each group's `DeviceResolver` when it
binds kernels. Then, through the engine's hooks:

- `before_materialize`: one VRAM pool for every class's arena — a fill pool growing to every piece, or the size
  `cache` pins — and per class a mapped host pool of the tier, filled from the extents;
- `after_materialize` (after every materialize): the policy seeds whole items within the pool's bytes (and each
  class's pinned slots); then each class's arena `[S, unit]` is laid out in the pool — `top_k` null slots (zero
  weights, one per routing rank), `staging` slots, then the cache slots its seeded pieces fill —, the map reset to
  the tier and the seed copied in;
- `before_launch`: the step's row kinds per token (decode and verify rows: a mixed step's leading rows, or a row
  whose every query token has logits; prefill rows the others, so a one-token last prompt chunk counts as decode;
  graph padding past them), written as far as this step's or the last one's tokens reach; the map edits due, which
  the step's first resolve applies, zeroing the step's counts too (in the forward: in a graph, on the stream);
- `after_commit`: the staging invariant (a layer's distinct misses fit its staging slots: F5a has no fallback; `misses`
  is the running maximum since the last materialize), then the policy's edits from the step's counts.
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
ALIGN = 4096                                 # each class's arena starts at a multiple of it


@dataclasses.dataclass(eq=False)
class SlotClass:
    """Pieces of one role, ggml type and shape, from every group: an arena in the store's pool, a host pool."""

    name: str
    unit_bytes: int
    null: int                                   # null slots [0, null), then staging slots
    staging: int
    pieces: list[tuple[int, int, int, list[Extent]]] = dataclasses.field(default_factory=list)  # group, item, role
    cache: int | float | None = None            # pinned cache slots, or a fraction of its pieces (None: fill)
    host: object = None                         # its host pool (`BuildCtx.pool`), from `before_materialize`
    pool: object = None                         # the store's arena pool, and where in it its arena lies (bytes,
    at: int = 0                                 # slots), from `after_materialize`: views of a pool's buffer are
    size: int = 0                               # made per use, so none outlives a materialize

    @property
    def max_slots(self) -> int:
        """Its arena's size when every piece is cached: what kernels over it are bound for."""
        return self.null + self.staging + len(self.pieces)

    @property
    def cap(self) -> int:
        """The most cache slots it may have."""
        n = len(self.pieces)
        return n if self.cache is None else self.cache if isinstance(self.cache, int) else round(self.cache * n)

    def bytes(self, cached: int) -> int:
        return (self.null + self.staging + cached) * self.unit_bytes

    def slots(self) -> torch.Tensor:
        """The arena as [S, unit] bytes (valid until the next materialize)."""
        whole = self.pool.view(torch.uint8, (self.pool.capacity * self.pool.unit_bytes,))
        return whole[self.at:self.at + self.size * self.unit_bytes].view(self.size, self.unit_bytes)

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
        self._kinds, self._written = np.zeros(tokens, dtype=np.uint8), 0     # host side; the prefix not PADDING
        self.kinds = ctx.static(f"{name}.kinds", (tokens,), torch.uint8)
        self.misses = ctx.static(f"{name}.misses", (1,), torch.int32)   # a running maximum (`moe_resolve@1`)
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
                              self.residency.device[g], self.host_units[g], self.counts[g], first=g == 0)

    # -- engine hooks -------------------------------------------------------------------------------------------------
    def before_materialize(self, ctx) -> None:
        """Request the arena pool and every class's host pool, and fill the host pools from the checkpoint."""
        jobs, host = [], np.full(self.residency.host.shape, -1, dtype=np.int32)
        for c in self.classes.values():
            if not 0 <= c.cap <= len(c.pieces):
                raise ValueError(f"{self.name}: class {c.name}: {c.cap} cache slots of {len(c.pieces)} pieces")
            units = self.tiers[0].capacity(c.name, len(c.pieces))
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
        classes = list(self.classes.values())
        least = sum(c.bytes(0 if c.cache is None else c.cap) for c in classes) + ALIGN * len(classes)
        most = sum(c.bytes(c.cap) for c in classes) + ALIGN * len(classes)
        fill = any(c.cache is None for c in classes)
        self.granule = -(-min(c.unit_bytes for c in classes) // ALIGN) * ALIGN    # the pool's unit: about a piece
        self.arena = ctx.pool(f"{self.name}.arena", unit_bytes=self.granule, min_units=-(-least // self.granule),
                              pref_units=-(-most // self.granule), fill=fill)
        files.read(jobs)
        self._host = host
        self.host_units.copy_(torch.from_numpy(host))

    def after_materialize(self, ctx) -> None:
        """Seed the new pool: the policy's items within its bytes, each class's arena laid out for its share, the map
        at the tier but for the seed, copied in."""
        classes = list(self.classes.values())
        size = self.arena.capacity * self.granule
        budget = _Budget(self, size - sum(c.bytes(0) for c in classes) - ALIGN * len(classes))
        groups = list(self._order)
        seed = self.policy.seed(self._profile_counts(), budget)
        self.arena.view(torch.uint8, (size,)).zero_()         # the null slots; no slot left holding garbage
        free, arenas, at = {}, {}, 0
        for c in classes:
            c.pool, c.at, c.size = self.arena, at, c.null + c.staging + budget.taken[c.name]
            at = -(-(at + c.size * c.unit_bytes) // ALIGN) * ALIGN
            free[c.name] = list(range(c.size - 1, c.null + c.staging - 1, -1))      # popped lowest first
            arenas[c.name] = c.slots(), c.host.view(torch.uint8, (c.host.capacity, c.unit_bytes))
        self.residency.host[...] = -1                   # −1 − tier 0: every piece in the host tier
        if sorted(seed) != sorted(budget.granted):
            raise RuntimeError(f"{self.name}: the placement policy seeded other items than it took from the budget")
        for g, i in seed:
            for r, c in self.groups[groups[g]].items():
                slot = self.residency.host[g, i, r] = free[c.name].pop()
                arena, units = arenas[c.name]
                arena[slot].copy_(units[self._host[g, i, r]], non_blocking=True)
        self.residency.seed()
        for t in (self.counts, self.misses, self.kinds):
            t.zero_()
        self._written = 0

    def before_launch(self, step) -> tuple[list[tuple[torch.Tensor, np.ndarray]], list]:
        kinds, at = self._kinds, 0
        kinds[:self._written] = PADDING
        for row, (q, logits) in enumerate(zip(step.n_query.tolist(), step.n_logits.tolist(), strict=True)):
            decode = row < step.n_decode or (not step.n_decode and q == logits)   # decode and verify rows
            kinds[at:at + q] = DECODE if decode else PREFILL
            at += q
        reach, self._written = max(at, self._written), at          # the last step's rows become padding
        return [(self.kinds[:reach], kinds[:reach])] * bool(reach) + self.residency.writes(step.step), []

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
        """`profile`'s routing counts by group: a JSON file `{"layers": {"<group>": [count per item]}}` beside the
        checkpoint (its name relative to the checkpoint's directory, so plans record no host path); groups it does not
        name count zero."""
        counts = np.zeros(self.residency.host.shape[:2])
        if not self._profile:
            return counts
        name = pathlib.PurePath(self._profile)
        if name.is_absolute() or ".." in name.parts:
            raise ValueError(f"{self.name}: profile {self._profile!r}: a name relative to the checkpoint's directory")
        checkpoint = pathlib.Path(next(iter(self.classes.values())).pieces[0][3][0].path).parent
        for group, row in json.loads((checkpoint / name).read_text())["layers"].items():
            if int(group) not in self._order or len(row) != self.items:
                raise ValueError(f"{self.name}: profile {self._profile}: group {group} with {len(row)} counts; the "
                                 f"groups are {sorted(self._order)} of {self.items} items")
            counts[self._order[int(group)]] = row
        return counts


class _Budget:
    """What a policy seeds within: the arena pool's bytes past every class's null and staging slots, each class's
    pinned slots; an item takes a slot in each of its roles' classes."""

    def __init__(self, store: TieredStore, nbytes: int) -> None:
        self.left, self.taken, self.granted = nbytes, dict.fromkeys(store.classes, 0), []
        self.classes = [list(store.groups[g].values()) for g in store._order]      # per group (in map order)

    def take(self, group: int, item: int) -> bool:
        classes = self.classes[group]
        cost = sum(c.unit_bytes for c in classes)
        if cost > self.left or any(self.taken[c.name] + classes.count(c) > c.cap for c in classes):
            return False
        self.left -= cost
        self.granted.append((group, item))
        for c in classes:
            self.taken[c.name] += 1
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
