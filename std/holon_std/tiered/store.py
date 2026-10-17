"""std.tiered.store@1: the engine's one tiered store of offloaded pieces (docs/v1-offload.md D5) — slot classes with
VRAM arenas and host tiers, the residency map, the placement policy and its device state, the per-step row kinds,
counts and delta log.

A client (the MoE offload store, `std.moe.offload@1`) adds every piece while its model is finalized — (group, item,
role), its class and the checkpoint extents its bytes come from — and asks for each group's `DeviceResolver` when it
binds kernels. Then, through the engine's hooks:

- `before_materialize`: one VRAM pool for every class's arena — a fill pool growing to every piece, or the size
  `cache` pins — and per class a mapped host pool of the tier, filled from the extents; both pools' unit is the
  classes' common divisor, so a between-step copy of a piece is a MemOp of whole units;
- `after_materialize` (after every materialize): the policy seeds whole items within the pool's bytes (and each
  class's pinned slots); then each class's arena `[S, unit]` is laid out in the pool — `top_k` null slots (zero
  weights, one per routing rank), `staging` slots, then the cache slots its seeded pieces fill —, the map reset to
  the tier and the seed copied in, the policy's device state zeroed;
- `before_launch`: the step's row kinds per token (decode and verify rows: a mixed step's leading rows, or every row
  of a step whose rows all have logits for every query token; prefill rows the others — so a one-token last prompt
  chunk counts as decode only in a step of nothing else, and a step's decode rows lie in its first expert pass; graph
  padding past them), written as far as this step's or the last one's tokens reach; the map edits due, which
  the step's first resolve applies, zeroing the step's counts, error words and delta log too (in the forward: in a
  graph, on the stream); the copies of two-phase swaps due;
- `after_commit`: the error words, the staging invariant (a layer's distinct misses fit its staging slots: F5a has no
  fallback; `misses` is the running maximum since the last materialize), the host mirror brought to the step (host
  edits launched by it, then the device's delta log), the route taps, then the policy's swaps from the step's counts.

A host policy's swap is two-phase (D4): decided at commit c, the outgoing item is unmapped at launch c + 2 (its
routes become misses), its slots are overwritten with the incoming item's bytes by a MemOp on the core's copy stream
after step c + 2's device work, and launch c + 2 + `LAND` waits for that copy before the incoming item is mapped
there. Every host edit carries the value it expects (a failed precondition stops the engine at that step's commit).
"""
from __future__ import annotations

import dataclasses
import functools
import json
import math
import pathlib
from collections.abc import Callable

import numpy as np
import torch
from holon.api import Extent, MemOp, Ref

from . import files
from .residency import LAG, ResidencyMap
from .resolver import DeviceResolver

PADDING, DECODE, PREFILL = 0, 1, 2           # row kinds (std.moe.kernels, `moe_resolve@1`)
ALIGN = 4096                                 # each class's arena starts at a multiple of it (and of the pool's unit)
LAND = 2                                     # d: launches between a swap's unmap and its map (D4)
FLOOR = 4                                    # F2: each group's hottest items before the global ranking
POLICY = "placement.lfu@1"                   # F11, the default where the RAM tier holds every expert (D10, §13)


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
        """Its arena's size when every piece is cached (`linear_experts_arena@1`'s `slots`)."""
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
        return self.host.device_view(torch.uint8, (len(self.pieces), self.unit_bytes))

    def copy(self, unit: int, slot: int, after: int, lands_by: int) -> MemOp:
        """A MemOp copying host unit `unit` into arena slot `slot`, as whole units of the two pools."""
        per, base = self.unit_bytes // self.pool.unit_bytes, self.at // self.pool.unit_bytes
        run = np.arange(per)
        return MemOp(self.host.name, unit * per + run, self.pool.name, base + slot * per + run, after=after,
                     lands_by=lands_by)


class TieredStore:
    def __init__(self, ctx, *, name: str, items: int, roles: int, top_k: int, staging: int,
                 policy: Ref | str = "", tiers: tuple[str, ...] = ("std.tiered.pinned@1",), cache: str = "",
                 profile: str = "") -> None:
        self.ctx, self.name, self.items, self.roles, self.top_k, self.staging = ctx, name, items, roles, top_k, staging
        self.tiers = [ctx.unit(Ref(t)) for t in tiers]
        if len(self.tiers) != 1 or not self.tiers[0].device_readable:
            raise NotImplementedError(f"{name}: one device-readable host tier (F5a), not {list(tiers)}")
        ref = policy if isinstance(policy, Ref) else Ref(policy or POLICY)    # POLICY: the one tier holds every expert
        self.policy, self.policy_ref = ctx.unit(ref, items=items, roles=roles), ref.target       # (pinned, inclusive)
        self._cache, self._profile = _parse_cache(cache), profile
        self.classes: dict[str, SlotClass] = {}
        self.groups: dict[int, dict[int, SlotClass]] = {}       # group -> role -> its class
        self.residency: ResidencyMap | None = None              # the device tables, once a resolver is asked for
        self.resolvers: list[DeviceResolver] = []
        tokens = max(ctx.options.get("max_step_tokens", 1),
                     ctx.options.get("max_seqs", 1) * (1 + ctx.options.get("max_draft_tokens", 0)))
        self._kinds, self._written = np.zeros(tokens, dtype=np.uint8), 0     # host side; the prefix not PADDING
        self.kinds = ctx.static(f"{name}.kinds", (tokens,), torch.uint8)
        self.misses = ctx.static(f"{name}.misses", (1,), torch.int32)   # a running maximum (`moe_resolve@1`)
        self.errors = ctx.static(f"{name}.errors", (1,), torch.int32)   # `moe_resolve@1`'s, zeroed each step
        self.edit_errors = ctx.static(f"{name}.edit_errors", (1,), torch.int32)   # host edits' preconditions
        self.peak_misses = 0                    # the most distinct misses a layer had (`misses`, at commits)
        self.taps: list[Callable[[object, dict], None]] = []    # route taps: (step, telemetry) at each commit
        self.step = 0                                           # the step being launched (`before_launch`)
        self._copies: dict[int, list[MemOp]] = {}               # launch -> the swaps' copies handed in then

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
            self.limits = self.ctx.static(f"{self.name}.limits", (len(self._order), self.roles), torch.int32)
            self.state = {n: self.ctx.static(f"{self.name}.state.{n}", (len(self._order), *s), getattr(torch, dtype))
                          for n, (s, dtype) in self.policy.device_state().items()}
            # A step's map changes: at most every item of every group (an admission moves one in, one out).
            log = self.ctx.static(f"{self.name}.delta", (shape[0] * self.items + 1, 3), torch.int32)   # + a sink
            self.delta, self.n_delta = log[:-1], self.ctx.static(f"{self.name}.n_delta", (1,), torch.int32)
            self._log = log
            self.telemetry = {"counts": self.counts, "misses": self.misses, "errors": self.errors,
                              "edit_errors": self.edit_errors, "delta": self.delta, "n_delta": self.n_delta}
            self.zeroed = [self.counts, self.errors, self.edit_errors, self.n_delta]    # by the step's first resolve
        g = self._order[group]
        self.resolvers.append(DeviceResolver(self.ctx, self, g, [self.groups[group][r] for r in range(self.roles)],
                                             {n: t[g] for n, t in self.state.items()}))
        return self.resolvers[-1]

    def watch(self, errors: torch.Tensor) -> None:
        """A kernel's error word (`linear_experts_arena@1`'s `errors`): zeroed each step, a commit that finds it set
        stops the engine."""
        if not any(errors is t for t in self.zeroed):
            self.telemetry[f"kernel_errors.{len(self.zeroed) - 4}"] = errors
            self.zeroed.append(errors)

    # -- engine hooks -------------------------------------------------------------------------------------------------
    def before_materialize(self, ctx) -> None:
        """Request the arena pool and every class's host pool, and fill the host pools from the checkpoint."""
        jobs, host = [], np.full(self.residency.host.shape, -1, dtype=np.int32)
        classes = list(self.classes.values())
        self.granule = functools.reduce(math.gcd, (c.unit_bytes for c in classes))     # both pools' unit
        self._align = math.lcm(self.granule, ALIGN)
        for c in classes:
            if not 0 <= c.cap <= len(c.pieces):
                raise ValueError(f"{self.name}: class {c.name}: {c.cap} cache slots of {len(c.pieces)} pieces")
            units = self.tiers[0].capacity(c.name, len(c.pieces))
            per = c.unit_bytes // self.granule
            c.host = ctx.pool(f"{self.name}.host.{c.name}", unit_bytes=self.granule, min_units=units * per,
                              pref_units=units * per, device="host", mapped=True)
            buffer = c.host.view(torch.uint8, (units, c.unit_bytes)).numpy()
            for piece, (g, i, r, extents) in enumerate(c.pieces):
                self.tiers[0].admit(c.name, piece, piece)
                _, host[self._order[g], i, r] = self.tiers[0].read(c.name, piece)
                at = 0
                for e in extents:
                    jobs.append((e, buffer[piece, at:at + e.nbytes]))
                    at += e.nbytes
        least = sum(c.bytes(0 if c.cache is None else c.cap) for c in classes) + self._align * len(classes)
        most = sum(c.bytes(c.cap) for c in classes) + self._align * len(classes)
        fill = any(c.cache is None for c in classes)
        self.arena = ctx.pool(f"{self.name}.arena", unit_bytes=self.granule, min_units=-(-least // self.granule),
                              pref_units=-(-most // self.granule), fill=fill)
        files.read(jobs)
        self._host = host
        self.host_units.copy_(torch.from_numpy(host))
        self._class_index = torch.tensor([[classes.index(self.groups[g][r]) for r in range(self.roles)]
                                          for g in self._order], device=self.ctx.device)

    def after_materialize(self, ctx) -> None:
        """Seed the new pool: the policy's items within its bytes, each class's arena laid out for its share, the map
        at the tier but for the seed, copied in; the policy's state, counts and words zeroed."""
        classes = list(self.classes.values())
        size = self.arena.capacity * self.granule
        budget = _Budget(self, size - sum(c.bytes(0) for c in classes) - self._align * len(classes))
        groups = list(self._order)
        seed = self.policy.seed(self._profile_counts(), budget)
        self.arena.view(torch.uint8, (size,)).zero_()         # the null slots; no slot left holding garbage
        free, arenas, at = {}, {}, 0
        for c in classes:
            c.pool, c.at, c.size = self.arena, at, c.null + c.staging + budget.taken[c.name]
            at = -(-(at + c.size * c.unit_bytes) // self._align) * self._align
            free[c.name] = list(range(c.size - 1, c.null + c.staging - 1, -1))      # popped lowest first
            arenas[c.name] = c.slots(), c.host.view(torch.uint8, (len(c.pieces), c.unit_bytes))
        self.residency.host[...] = -1                   # −1 − tier 0: every piece in the host tier
        if sorted(seed) != sorted(budget.granted):
            raise RuntimeError(f"{self.name}: the placement policy seeded other items than it took from the budget")
        for g, i in seed:
            for r, c in self.groups[groups[g]].items():
                slot = self.residency.host[g, i, r] = free[c.name].pop()
                arena, units = arenas[c.name]
                arena[slot].copy_(units[self._host[g, i, r]], non_blocking=True)
        self.residency.seed()
        self._copies.clear()
        self.limits.copy_(torch.tensor([[c.size for _, c in sorted(self.groups[g].items())] for g in groups]))
        for t in (self.misses, self.kinds, *self.zeroed, *self.state.values()):
            t.zero_()
        self._written = 0

    def before_launch(self, step) -> tuple[list[tuple[torch.Tensor, np.ndarray]], list[MemOp]]:
        self.step, kinds, at = step.step, self._kinds, 0
        kinds[:self._written] = PADDING
        uniform = not step.n_decode and bool((step.n_query == step.n_logits).all())    # a decode or verify step
        for row, q in enumerate(step.n_query.tolist()):
            kinds[at:at + q] = DECODE if row < step.n_decode or uniform else PREFILL
            at += q
        reach, self._written = max(at, self._written), at          # the last step's rows become padding
        writes = [(self.kinds[:reach], kinds[:reach])] * bool(reach) + self.residency.writes(step.step)
        return writes, self._copies.pop(step.step, [])

    def after_commit(self, step, telemetry) -> None:
        if errors := {name: int(t[0]) for name, t in telemetry.items() if "errors" in name and int(t[0])}:
            raise RuntimeError(f"{self.name}: step {step.step}'s kernels flagged errors {errors} (errors: 1 = a "
                               "corrupt residency entry, 2 = the delta log overflowed; edit_errors: a host edit's "
                               "precondition failed; kernel_errors: the kernel's bits)")
        self.peak_misses = int(telemetry["misses"][0])
        if self.peak_misses > self.staging:
            raise RuntimeError(f"{self.name}: a layer had {self.peak_misses} distinct misses, {self.staging} staging "
                               "slots")
        self.residency.commit(step.step, telemetry["delta"][:int(telemetry["n_delta"][0])].numpy())
        for tap in self.taps:
            tap(step, telemetry)
        if swaps := self.policy.after_commit(telemetry["counts"].numpy(), self.residency.host):
            self._swap(step.step, swaps)

    # -- the policy's means -------------------------------------------------------------------------------------------
    def lend(self, n: int) -> torch.Tensor:
        """Device lending under a device-owned policy (D4; step 3b's ring calls it in a ring step's forward, eagerly,
        after the step's first resolve, which zeroes the log and the error words): the whole experts its resolve
        contract's `lend` picks, until every class has `n` slots (or all it has), are unmapped and logged `(group, −1,
        item)`; their scores stay. Returns the picks [groups, items]. Lent slots are not refilled yet: D4's return
        (a lent slot its layer's first victim) is step 3b's."""
        picked = self.resolvers[0].resolve.lend(n, self.residency.device, self._class_index, **self.state)
        self.residency.device.masked_fill_(picked[..., None], -1)
        flat = picked.reshape(-1)
        at = (self.n_delta + flat.cumsum(0) - 1).long()
        sink = len(self.delta)
        at = torch.where(flat & (at < sink), at, sink)
        groups, items = picked.shape
        g = torch.arange(groups, device=flat.device).repeat_interleave(items)
        record = torch.stack([g, torch.full_like(g, -1), torch.arange(items, device=flat.device).repeat(groups)], 1)
        self._log.index_copy_(0, at, record.int())
        self.errors.bitwise_or_(((self.n_delta + flat.sum()) > sink).int() * 2)          # OVERFLOW
        self.n_delta += flat.sum().int()
        return picked

    def _swap(self, commit: int, swaps: list[tuple[int, int, bool]]) -> None:
        """A host policy's swaps, two-phase: the outgoing items unmapped at launch commit + 2, the incoming copied into
        their slots after it and mapped at commit + 2 + LAND (every edit expecting what the host's view holds)."""
        if self.policy.device_owned:
            raise RuntimeError(f"{self.name}: the policy edits its device-owned cache")
        moves: dict[int, tuple[list[int], list[int]]] = {}
        for g, i, resident in swaps:
            moves.setdefault(g, ([], []))[not resident].append(i)
        view, groups, unmap, map_ = self.residency.host, list(self._order), [], []
        copies = self._copies.setdefault(commit + LAG, [])
        for g, (ins, outs) in moves.items():
            if len(ins) != len(outs):
                raise ValueError(f"{self.name}: group {g}: {len(ins)} items in, {len(outs)} out (a swap keeps slots)")
            for a, b in zip(ins, outs, strict=True):
                for r, c in self.groups[groups[g]].items():
                    slot = int(view[g, b, r])
                    unmap.append((g, b, r, slot, -1))
                    map_.append((g, a, r, int(view[g, a, r]), slot))
                    copies.append(c.copy(int(self._host[g, a, r]), slot, commit + LAG, commit + LAG + LAND))
        self.residency.edit(commit, unmap)
        self.residency.edit(commit, map_, later=LAND)

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

    @staticmethod
    def warm_start(profile: np.ndarray) -> list[tuple[int, int]]:
        groups, items = profile.shape
        ranked = np.lexsort((np.broadcast_to(np.arange(items), profile.shape), -profile), axis=-1)   # per group
        floor = [(g, int(ranked[g, n])) for n in range(min(FLOOR, items)) for g in range(groups)]
        group, item = np.divmod(np.arange(profile.size), items)
        rest = np.lexsort((group, item, -profile.ravel()))
        taken = set(floor)
        return floor + [(int(group[x]), int(item[x])) for x in rest if (int(group[x]), int(item[x])) not in taken]


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
