"""std.tiered.store@1: the engine's one tiered store of offloaded pieces (docs/dev/v1-offload.md D5) — slot classes with
VRAM arenas and host tiers, the residency map, the placement policy and its device state, the per-step row kinds,
counts and delta log.

A client (the MoE offload store, `std.moe.offload@1`) adds every piece while its model is finalized — (group, item,
role), its class and the checkpoint extents its bytes come from — and asks for each group's `DeviceResolver` when it
binds kernels. Then, through the engine's hooks:

- `before_materialize`: one VRAM pool for every class's arena — a fill pool growing to every piece, or the size
  `cache` pins — and per class a mapped host pool of the tier, filled from the extents; both pools' unit is the
  classes' common divisor, so a between-step copy of a piece is a MemOp of whole units; with `ring_lend = "tail"`
  the tail (docs/dev/v1-moe-prefill.md D11): its class and its slots, from the bytes the clients' passes take
  (`tail_bytes`);
- `after_materialize` (after every materialize): the policy seeds whole items within the pool's bytes (and each
  class's pinned slots: beside fill classes, a pinned class's hottest items first, so it gets them); then each
  class's arena `[S, unit]` is laid out in the pool — `top_k` null slots (zero weights, one per routing rank),
  `staging` slots, then the cache slots its seeded pieces fill —, the map reset to the tier and the seed copied in,
  the policy's device state zeroed;
- `before_launch`: the step's row kinds per token (decode and verify rows by the plan's `row_kind`, which the
  scheduler fills on every step it launches — a plan without it is refused; prefill rows the others, so a step's
  decode rows lie in its first expert pass; graph padding past them), written as far as this step's or the last
  one's tokens reach; the map edits due, which the step's first resolve applies, zeroing the step's counts, error
  words and delta log too (in the forward: in a graph, on the stream); the copies of two-phase swaps due;
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
import fnmatch
import functools
import json
import math
import os
import pathlib
from collections.abc import Callable

import numpy as np
import torch
from holon.api import Extent, MemOp, Ref, RowKind

from . import files
from .residency import LAG, ResidencyMap
from .resolver import ACTIVATION_LOAN, CPU_TASK, EXPERT_PREFETCH, TIERS, DeviceResolver
from .ring import ROUNDS, Lent

PADDING, DECODE, PREFILL = 0, 1, 2           # row kinds (std.moe.kernels, `moe_resolve@1`)
ALIGN = 4096                                 # each class's arena starts at a multiple of it (and of the pool's unit)
LAND = 2                                     # d: launches between a swap's unmap and its map (D4)
LOG = 2                                      # the delta log's records per group, per item
FLOOR = 4                                    # F2: each group's hottest items before the global ranking
POLICY = "placement.lfu@1"                   # F11, the default where the RAM tier holds every expert (D10, §13)
LOAN_MARGIN, LOAN_SLACK = 2.0, 20 << 20      # a lent activation peak's room for the caching allocator's segments:
# a MemPool in use keeps its freed segments (torch releases a private pool's cache only when no one uses it), so a
# step's segments are 1.6-1.75x its allocated peak (FN §23.4; 2.35-2.74x before 168562e, §21.12); lent in proportion
# to the step's prompt tokens. A step past its loan takes the rest from the device (the loan's overflow)


@dataclasses.dataclass(eq=False)
class SlotClass:
    """Pieces of one role, ggml type and shape, from every group: an arena in the store's pool, a host pool."""

    name: str
    unit_bytes: int
    null: int                                   # null slots [0, null), then staging slots
    staging: int
    pieces: list[tuple[int, int, int, list[Extent]]] = dataclasses.field(default_factory=list)  # group, item, role
    cache: int | float | None = None            # pinned cache slots, or a fraction of its pieces (None: fill)
    host: object = None                         # its host tier's pool (`Tier.units`), from `before_materialize`
    held: int = 0                               # its RAM units (every piece but under an exclusive tier)
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
        """The host tier's pool as the device reads it, [units, unit] bytes (RAM units, then staging units)."""
        return self.host.device_view(torch.uint8, (self.host.capacity * self.host.unit_bytes // self.unit_bytes,
                                                   self.unit_bytes))

    def host_view(self) -> torch.Tensor:
        """The same pool as the host sees it."""
        return self.host.view(torch.uint8, (self.host.capacity * self.host.unit_bytes // self.unit_bytes,
                                            self.unit_bytes))

    def copy(self, unit: int, slot: int, after: int, lands_by: int) -> MemOp:
        """A MemOp copying host unit `unit` into arena slot `slot`, as whole units of the two pools."""
        per, base = self.unit_bytes // self.pool.unit_bytes, self.at // self.pool.unit_bytes
        run = np.arange(per)
        return MemOp(self.host.name, unit * per + run, self.pool.name, base + slot * per + run, after=after,
                     lands_by=lands_by)


class TieredStore:
    def __init__(self, ctx, *, name: str, items: int, roles: int, top_k: int, staging: int, prefill: int = 0,
                 ring_tokens: int = 0, ring_copy: str = "sm", ring_lend: str = "coldest", stream_tokens: int = 0,
                 lend_peak: bool = False, pipeline: int = 1, policy: Ref | str = "",
                 tiers: tuple[str, ...] = ("std.tiered.pinned@1",), cache: str = "", profile: str = "",
                 ram: str = "") -> None:
        self.ctx, self.name, self.items, self.roles, self.top_k = ctx, name, items, roles, top_k
        # `staging`: slots for the graphed steps' misses; `prefill`: a prefill pass's (lent in its step when more, so
        # a pinned cache must be able to lend that many: else the staging slots hold them, as before D8);
        # `ring_tokens`: the prompt tokens of a step from which it streams through the ring (D8; 0: never)
        # `ring_copy`: the ring's copies on the device's SMs from its own tables ("sm"), or by copy engines from the
        # tables read back once per streaming step ("dma")
        self.staging, self.prefill, self.ring_tokens = staging, max(prefill, staging), ring_tokens
        # `stream_tokens`: a step of large passes (at least `ring_tokens` prompt tokens) streams every missing piece
        # from this many prompt tokens; below, it stages only its passes' routed misses into its lent slots (its tail
        # lent too)
        self.stream_tokens = stream_tokens
        # `pipeline` (FN step 7): a staging step's large passes fetch their misses in this many parts on a side stream,
        # each layer's resident entries computed first, then each part's as it lands (1: one fetch, then one pass).
        # Where every client pipelines its passes (`pipelining`), such a step lends its staging as a ring of parts
        # (`Lent`: twice a part of a layer's most absent pieces, each part's fetch after the part two before it ran)
        # and every pass of it pipelines (`pipelines`).
        self.pipeline, self._parts = pipeline, 1
        self.pipelining: list[Callable[[], bool]] = []       # per client: whether its passes always pipeline
        self._pipelining = False
        if ring_copy not in ("sm", "dma"):
            raise ValueError(f"{name}: ring_copy {ring_copy!r}: sm or dma")
        # `ring_lend`: what a streaming step lends — the policy's coldest whole experts ("coldest", D4), or first the
        # experts holding the last slots of the largest class arena, a contiguous range its clients' passes take their
        # buffers from (`tail`), then the coldest ("tail", docs/dev/v1-moe-prefill.md D11)
        if ring_lend not in ("coldest", "tail"):
            raise ValueError(f"{name}: ring_lend {ring_lend!r}: coldest or tail")
        self.ring_copy, self.ring_lend = ring_copy, ring_lend
        self.tail_bytes: list[Callable[..., int]] = []       # the bytes the clients' passes take from a tail
        #                                                      (their largest pass's; given a step's tokens, its)
        self._step_tail: tuple[int, int] | None = None       # a ring step's tail: the tail's last n it lends
        # `lend_peak` (FN step 7): a tail step's activations from the tail's last bytes (`lend_activations`, the core's
        # ActivationLender), so the engine reserves only the worst step that streams nothing
        self.lend_peak, self._loan, self._loan_bytes, self._loaned = lend_peak, None, 0, False
        self._loan_per_token, self._step_loan = 0.0, 0       # the loan's bytes per prompt token; this step's
        self.tail: tuple[int, int] | None = None             # (class index, its last n slots): set at materialize
        self._own_tail: tuple[int, int] | None = None        # a staging layout's own tail: (byte offset, bytes)
        self._fetch_streams: dict[bool, torch.Stream] = {}
        self._auxiliary: set[int] = set()                   # an auxiliary model's groups (`mark_auxiliary`)
        self.tiers = [ctx.unit(Ref(t), ram=ram) for t in tiers]    # `ram`: the RAM tier's units per class
        if not 1 <= len(self.tiers) <= 2 or not self.tiers[0].device_readable or \
                any(t.device_readable for t in self.tiers[1:]):
            raise NotImplementedError(f"{name}: a device-readable host tier, then at most one file tier, not "
                                      f"{list(tiers)}")
        self.exclusive = False                  # RAM holds fewer pieces than the store (D10): decided at the bind
        ref = policy if isinstance(policy, Ref) else Ref(policy or POLICY)    # POLICY: the one tier holds every expert
        self.policy, self.policy_ref = ctx.unit(ref, items=items, roles=roles), ref.target       # (pinned, inclusive)
        self._cache, self._profile = _parse_cache(cache), profile
        self.classes: dict[str, SlotClass] = {}
        self.groups: dict[int, dict[int, SlotClass]] = {}       # group -> role -> its class
        self.residency: ResidencyMap | None = None              # the device tables, made at `before_materialize`
        self.resolvers: list[DeviceResolver] = []
        self._watched: list[torch.Tensor] = []                  # kernels' error words watched before the tables
        tokens = max(ctx.options.get("max_step_tokens", 1),
                     ctx.options.get("max_seqs", 1) * (1 + ctx.options.get("max_draft_tokens", 0)))
        self._kinds, self._written = np.zeros(tokens, dtype=np.uint8), 0     # host side; the prefix not PADDING
        self.kinds = ctx.static(f"{name}.kinds", (tokens,), torch.uint8)
        self._decode = ctx.static(f"{name}.decode_kinds", (tokens,), torch.uint8)    # DECODE throughout (sub-steps)
        self._decode_set = False
        self.step, self.step_tokens = -1, 0     # the step being launched and its tokens (`before_launch`)
        self.misses = ctx.static(f"{name}.misses", (1,), torch.int32)   # the step's most (`moe_resolve@1`)
        self.errors = ctx.static(f"{name}.errors", (1,), torch.int32)   # `moe_resolve@1`'s, zeroed each step
        self.edit_errors = ctx.static(f"{name}.edit_errors", (1,), torch.int32)   # host edits' preconditions
        self.peak_misses = 0                    # the most distinct misses a layer had (`misses`, at commits)
        # Route taps: (step, record) at each commit; the record holds the step's routing `counts` [G, E] of decode
        # and verify rows, the experts any of its rows `routed` [G, E] (bool) and its victims' `scope` (what the call
        # holding its decode rows routed, under a variant), the delta log's `records` [n, 3] and,
        # when it borrowed slots (its empty ones, then lent), its `lends` (per call, its need per class: row 0 the
        # tail's, n at its class (D11; zero without one), then each `lend(need)` round's; else None).
        self.taps: list[Callable[[object, dict], None]] = []
        self._copies: dict[int, list[MemOp]] = {}               # launch -> the swaps' copies handed in then
        self._mode: str | None = None                           # the step being launched borrows: ring or stage
        self._borrowing: dict[int, str] = {}                    # step -> how it borrowed (its forward), until commit
        self.lending = False                                    # the layout lends (`after_materialize`)
        self.lent: Lent | None = None                           # its lent slots, from its first resolve to its end
        self.borrowed = {"ring": 0, "stage": 0}                 # steps that borrowed slots, by use (since a seed)

    # -- while the model is finalized and bound ------------------------------------------------------------------------
    def add(self, group: int, item: int, role: int, cls: str, unit_bytes: int, extents: list[Extent]) -> SlotClass:
        """One piece: role `role` of item `item` of group `group`, in class `cls` (`unit_bytes` each), its bytes the
        concatenation of `extents`."""
        if self.residency is not None:
            raise RuntimeError(f"{self.name}: pieces added after the device tables were made (at materialize)")
        c = self.classes.setdefault(cls, SlotClass(cls, unit_bytes, self.top_k, self.staging,
                                                   cache=_cache_of(self._cache, cls)))
        if c.unit_bytes != unit_bytes or sum(e.nbytes for e in extents) != unit_bytes:
            raise ValueError(f"{self.name}: class {cls} has {c.unit_bytes}-byte units, not {unit_bytes} "
                             f"({sum(e.nbytes for e in extents)} in the extents)")
        if self.groups.setdefault(group, {}).setdefault(role, c) is not c:
            raise ValueError(f"{self.name}: group {group}'s role {role} in two classes")
        c.pieces.append((group, item, role, extents))
        return c

    def resolver(self, group: int) -> DeviceResolver:
        """Group `group`'s resolver. Its device tables come at `before_materialize` (`_tables`), when every group has
        added its pieces: an auxiliary model's (an MTP layer's experts, FN D12a) are added after the target's kernels
        are bound."""
        self.resolvers.append(DeviceResolver(self.ctx, self, group, [self.groups[group][r] for r in range(self.roles)]))
        if self.ring_lend == "tail" and not hasattr(self.resolvers[0].resolve, "lend_tail"):
            raise ValueError(f"{self.name}: ring_lend tail (moe.ring_lend): {self.policy_ref}'s resolve "
                             f"({self.policy.resolve_variant or 'moe_resolve@1'}) does not lend a tail (`lend_tail`)")
        if self.residency is not None or any(self.tiers[0].holds(c.name, len(c.pieces)) < len(c.pieces)
                                             for c in self.classes.values()):
            self._tables()                       # exclusive tiers bind their I/O worker in the build: no later group
        return self.resolvers[-1]

    def _tables(self) -> None:
        """The device tables of every group added, then each resolver attached to its group's."""
        if self.residency is None:
            aux = {c.name for g in self._auxiliary for c in self.groups[g].values()} - \
                {c.name for g in self.groups if g not in self._auxiliary for c in self.groups[g].values()}
            pinned = [c.cap for c in self.classes.values() if c.cache is not None and c.name not in aux]
            for c in self.classes.values():
                c.held = self.tiers[0].holds(c.name, len(c.pieces))
            self.exclusive = any(c.held < len(c.pieces) for c in self.classes.values())
            ranked = self.policy.device_owned and getattr(self.policy, "heat", None) is not None
            if self.exclusive and (len(self.tiers) < 2 or not (ranked or getattr(self.policy, "frozen", False))):
                raise NotImplementedError(f"{self.name}: RAM holds fewer pieces than the store: a slower tier must "
                                          f"hold the rest, and the policy ({self.policy_ref}) own the cache on the "
                                          "device naming the heat it ranks by, or never edit it (host edits would "
                                          "copy from RAM units the graph moves)")
            if not self.policy.device_owned or any(cap < self.prefill for cap in pinned) or self.exclusive:
                # A host policy lends two-phase (not built), a small pinned cache cannot lend a pass's staging, an
                # exclusive RAM tier would demote what it lends (not built): the staging slots hold a prefill pass's
                # misses, and no step streams.
                self.ring_tokens = 0
                self.staging = self.prefill
                for c in self.classes.values():
                    c.staging = self.prefill
            self._order = {g: n for n, g in enumerate(sorted(self.groups))}
            shape = (len(self._order), self.items, self.roles)
            self.residency = ResidencyMap(self.ctx, self.name, shape)
            self.host_units = self.ctx.static(f"{self.name}.host", shape, torch.int32)
            self.counts = self.ctx.static(f"{self.name}.counts", shape[:2], torch.int32)
            self.limits = self.ctx.static(f"{self.name}.limits", (len(self._order), self.roles), torch.int32)
            self.state = {n: self.ctx.static(f"{self.name}.state.{n}", (len(self._order), *s), getattr(torch, dtype))
                          for n, (s, dtype) in self.policy.device_state().items()}
            self.routed = self.ctx.static(f"{self.name}.routed", shape[:2], torch.int32)
            # a call without a CPU share (`moe_resolve@1`'s `plan`, `jobs`): it takes nothing, its table unread
            self.no_plan = self.ctx.static(f"{self.name}.no_plan", (self.items + 1,), torch.int32)
            self.no_jobs = self.ctx.static(f"{self.name}.no_jobs", (1, 4), torch.int32)
            # The delta log, a region per group: a step's records of a group are its admissions (at most its misses,
            # E − R for R resident), its lent items (at most R) and the lent slots' returns (at most as many): 2E bound
            # them, and one more for the returned slots' marker (`Lent`; + a sink row for `lend`'s scatter).
            log = self.ctx.static(f"{self.name}.delta", (shape[0], LOG * self.items + 2, 3), torch.int32)
            self.delta, self.n_delta = log[:, :-1], self.ctx.static(f"{self.name}.n_delta", shape[:1], torch.int32)
            self._log = log
            self.telemetry = {"counts": self.counts, "routed": self.routed, "misses": self.misses,
                              "errors": self.errors, "edit_errors": self.edit_errors, "delta": self.delta,
                              "n_delta": self.n_delta}
            self.zeroed = [self.counts, self.routed, self.misses, self.errors, self.edit_errors, self.n_delta]
            self.ring = self.ctx.static(f"{self.name}.ring", shape, torch.int32)      # this step's ring slots
            self.empty = self.ctx.static(f"{self.name}.empty", (shape[0], self.roles, self.items), torch.int32)
            self.n_empty = self.ctx.static(f"{self.name}.n_empty", shape[:1], torch.int32)
            # A borrowing step's tables (`Lent`, on the device): the empty slots it found, the lent experts' slots and
            # `lend` call, each call's need per class, each class's pool of lent slots, the ring's copies, the stage
            # lists, and whether the ring held (else the step staged) with the stage width its misses had.
            n_classes = len(self.classes)
            self.n_before = self.ctx.static(f"{self.name}.n_before", shape[:1], torch.int32)
            self.lent_slots = self.ctx.static(f"{self.name}.lent_slots", shape, torch.int32)
            self.lent_call = self.ctx.static(f"{self.name}.lent_call", shape[:2], torch.int32)
            self.lend_needs = self.ctx.static(f"{self.name}.lend_needs", (1 + ROUNDS, n_classes), torch.int64)
            self.pool = self.ctx.static(f"{self.name}.pool", (n_classes, 2 * self.roles * math.prod(shape[:2]) + 1),
                                        torch.int32)
            self.ring_fetch = self.ctx.static(f"{self.name}.ring_fetch", (shape[0], self.roles, 2, self.items),
                                              torch.int32)
            self.ring_n = self.ctx.static(f"{self.name}.ring_n", (shape[0], self.roles), torch.int32)
            self.stage_tab = self.ctx.static(f"{self.name}.stage_tab", (shape[0], self.roles, self.prefill),
                                             torch.int32)
            self.ring_ok = self.ctx.static(f"{self.name}.ring_ok", (1,), torch.int32)
            pinned = self.ctx.device.type == "cuda"         # "dma": the copy list read back into pinned memory
            self.ring_host = (torch.empty(self.ring_fetch.shape, dtype=torch.int32, pin_memory=pinned),
                              torch.empty(self.ring_n.shape, dtype=torch.int32, pin_memory=pinned))
            self.stage_width = self.ctx.static(f"{self.name}.stage_width", (1,), torch.int32)
            self.telemetry |= {"lends": self.lend_needs, "ring_ok": self.ring_ok, "stage_width": self.stage_width}
            # The staging slots' list: a graphed step's, or (a layout too small to lend) a prefill pass's.
            self.stages = {n: self.ctx.static(f"{self.name}.stage.{n}", (self.roles, n), torch.int32)
                           for n in {self.staging, self.prefill}}
            self.layer_classes = [[self.groups[g][r] for r in range(self.roles)] for g in self.layers]
            if self.exclusive:
                self._exclusive_tables(shape)
            self.prefetch = {}
            for c in self.classes.values():
                facts = {"unit_bytes": c.unit_bytes}
                self.prefetch[c.name] = self.ctx.select_kernel(
                    EXPERT_PREFETCH, site=f"std.tiered[{self.name}].{c.name}", **(facts | {"copy": self.ring_copy})
                ).impl(self.ctx, **facts, copy=self.ring_copy)
            for errors in self._watched:
                self.watch(errors)
        for r in self.resolvers:
            if r.group is not None:
                continue
            g = self._order[r.group_id]
            r.attach(g, {n: t[g] for n, t in self.state.items()})
            if self.exclusive:                   # its SSD reads: a copy task on the I/O worker, bound after materialize
                self.reads.append(_Reads(self, g))
                r.io_slot = self.io.bind(self.reads[-1])

    def _exclusive_tables(self, shape: tuple[int, int, int]) -> None:
        """The exclusive tiers' device tables (`moe_tiers@1`'s), its kernel and the I/O worker (cpu_task@1, `io`)."""
        classes = list(self.classes.values())
        static = functools.partial(self.ctx.static, dtype=torch.int32)
        self.piece = static(f"{self.name}.piece", shape)
        self.classes32 = static(f"{self.name}.classes32", (shape[0], self.roles))
        self.owner = static(f"{self.name}.owner", (len(classes), max(c.held for c in classes)))
        self.capacity = static(f"{self.name}.capacity", (len(classes),))
        self.call_start = static(f"{self.name}.call_start", (1,))
        self.moves = static(f"{self.name}.moves", (self.roles, 4, self.prefill))
        self.n_moves = static(f"{self.name}.n_moves", (self.roles, 2))
        facts = {"experts": self.items, "top_k": self.top_k, "roles": self.roles}
        self.tier_op = self.ctx.select_kernel(TIERS, site=f"std.tiered[{self.name}]", **facts).impl(self.ctx, **facts)
        self.io = self.ctx.select_kernel(CPU_TASK, site=f"std.tiered[{self.name}].io", pool="io").impl(self.ctx,
                                                                                                       pool="io")
        self.watch(self.io.errors)               # (it fails open: a read it gave up on stops the engine at the commit)
        self.reads: list[_Reads] = []
        self.heat = self.state.get(getattr(self.policy, "heat", None) or "")

    @property
    def layers(self) -> list[int]:
        """The group (layer) ids, in map order: row g of the map, the counts and the records is layer `layers[g]`."""
        return sorted(self.groups)

    def watch(self, errors: torch.Tensor) -> None:
        """A kernel's error word (`linear_experts_arena@1`'s `errors`): zeroed each step, a commit that finds it set
        stops the engine."""
        if self.residency is None:              # before the tables: kept until `_tables`
            self._watched.append(errors)
        elif not any(errors is t for t in self.zeroed):
            self.telemetry[f"kernel_errors.{len(self.zeroed) - 6}"] = errors
            self.zeroed.append(errors)

    # -- engine hooks -------------------------------------------------------------------------------------------------
    def before_materialize(self, ctx) -> None:
        """The tiers take their storage (the RAM tier its pool, filled when it holds every piece), the arena pool is
        requested, and under exclusive tiers the I/O tables."""
        self._tables()
        classes = list(self.classes.values())
        self.granule = functools.reduce(math.gcd, (c.unit_bytes for c in classes))     # both pools' unit
        self._align = math.lcm(self.granule, ALIGN)
        for c in classes:
            if not 0 <= c.cap <= len(c.pieces):
                raise ValueError(f"{self.name}: class {c.name}: {c.cap} cache slots of {len(c.pieces)} pieces")
            extents = [x for *_, x in c.pieces]
            self.tiers[0].prepare(ctx, c.name, c.unit_bytes, extents, staging=c.staging if self.exclusive else 0,
                                  granule=self.granule)
            for t in self.tiers[1:]:
                t.prepare(ctx, c.name, c.unit_bytes, extents)
            c.host = self.tiers[0].units(c.name)
        host = np.full(self.residency.host.shape, -1, dtype=np.int32)
        if not self.exclusive:                   # inclusive: piece i in unit i, for good
            for c in classes:
                for piece, (g, i, r, _) in enumerate(c.pieces):
                    _, host[self._order[g], i, r] = self.tiers[0].read(c.name, piece)
        # One fill pool, pinned or not, so both profile the same minimum and grow above the paged floor alike (a fill's
        # counts then pin exactly): its minimum caches nothing and cannot lend, so that layout stages a prefill pass
        # (after_materialize); a pinned arena short of its pinned size is refused at the first launch (before_launch).
        # A layout that stages also keeps the tail's bytes for the large passes' buffers (`own_tail`), so the startup
        # profile's worst step — on this minimum — takes them from the pool, not from the activations it measures.
        self.tail = self._tail(lambda c: c.cap)
        least = sum((c.null + self.prefill) * c.unit_bytes for c in classes) + self._align * len(classes) + \
            self._own_bytes()
        most = sum(c.bytes(c.cap) for c in classes) + self._align * len(classes)
        self.arena = ctx.pool(f"{self.name}.arena", unit_bytes=self.granule, min_units=-(-least // self.granule),
                              pref_units=-(-most // self.granule), fill=True)
        self._pinned = -(-most // self.granule) if all(c.cache is not None for c in classes) else None
        self._host = host
        self.host_units.copy_(torch.from_numpy(host))
        self.class_table = np.array([[classes.index(self.groups[g][r]) for r in range(self.roles)]
                                     for g in self._order])
        self._class_index = torch.from_numpy(self.class_table).to(self.ctx.device)
        if self.exclusive:
            piece = np.zeros(self.residency.host.shape, dtype=np.int32)
            for c in classes:
                for n, (g, i, r, _) in enumerate(c.pieces):
                    piece[self._order[g], i, r] = n
            self._pieces = dict.fromkeys(self.classes, piece)
            self.piece.copy_(torch.from_numpy(piece))
            self.classes32.copy_(torch.from_numpy(self.class_table.astype(np.int32)))
            self.capacity.copy_(torch.tensor([c.held for c in classes], dtype=torch.int32))
            rows = 2 * (1 + self.prefill)
            self.io_pool = ctx.pool(f"{self.name}.io", unit_bytes=4 * rows, min_units=self.roles,
                                    pref_units=self.roles, device="host", mapped=True)
            self.descriptors = {}
            for c in classes:
                rows = []
                for n in range(len(c.pieces)):
                    kind, extents = self.tiers[1].read(c.name, n)
                    if kind != "pread":                  # the copy task's one kind (a page-cache tier: O_DIRECT off)
                        raise ValueError(f"{self.name}: {type(self.tiers[1]).__name__} read "
                                         f"{c.name}'s piece {n} as {kind!r}: a file tier's descriptors are 'pread' "
                                         "extents (fd, O_DIRECT, offset, nbytes)")
                    rows.append([tuple(x) for x in extents])
                width = max(map(len, rows))
                self.descriptors[c.name] = torch.tensor([x + [(0, 0, 0, 0)] * (width - len(x)) for x in rows],
                                                        dtype=torch.int64)
            self._ram = {c.name: np.full(c.held, -1) for c in classes}      # what each RAM unit holds (host's view)

    def decode_kinds(self, tokens: int) -> torch.Tensor:
        """Row kinds of `tokens` decode rows (an auxiliary model's sub-step, FN D12a)."""
        return self._decode[:tokens]

    def after_materialize(self, ctx) -> None:
        """Seed the new pool: the policy's items within its bytes, each class's arena laid out for its share, the map
        at the tier but for the seed, copied in; the policy's state, counts and words zeroed."""
        self._decode_set = False                # the decode kinds are written again at the next launch
        classes = list(self.classes.values())
        size = self.arena.capacity * self.granule
        groups = list(self._order)
        # A layout lends a prefill pass's staging (D5) when every class can; else its staging slots hold that much.
        # A class the seed leaves short of a pass's staging (one few groups share, or an auxiliary model's) is seeded
        # that many of its hottest items first (`floors`), if its cap allows: a few cache slots more for it, where
        # staging slots of a pass's size in every class would hold more and cache nothing.
        # A staging layout keeps its own tail at the pool's end (`own_tail`): the large passes' buffers.
        can_lend = self.policy.device_owned and not self.exclusive
        # Beside fill classes, a pinned class (`cache`: an MTP head's experts, kept whole) is floored at its pins: a
        # profile that never counted its group would otherwise seed it last, F2's few of it only.
        pinned = {c.name: c.cap for c in classes if c.cache is not None} if self._pinned is None else {}
        floors = dict(pinned)
        # the classes a borrowing step lends from: the target's (an auxiliary model's layers never borrow)
        lenders = {c.name for g in list(self._order)[:self.borrowing_groups] for c in self.groups[g].values()}
        for staging in sorted({self.staging, self.prefill}):
            for c in classes:
                c.staging = staging
            keep = self._own_bytes(loan=False) if staging == self.prefill else 0
            while True:
                budget = _Budget(self, size - keep - sum(c.bytes(0) for c in classes) - self._align * len(classes),
                                 floors)
                seed = self.policy.seed(self._profile_counts(), budget)
                short = {c.name: self.prefill for c in classes if budget.taken[c.name] < self.prefill
                         and c.name not in floors and c.cap >= self.prefill and c.name in lenders}
                if not (can_lend and staging < self.prefill and short):
                    break
                floors |= short
            self.lending = can_lend and all(n >= self.prefill for c, n in budget.taken.items() if c in lenders)
            if self.lending and self.tail:          # the tail's class must still stage a pass outside the tail: a
                cls, n = self.tail                  # small pool (the startup profile's minimum) stages instead
                c = classes[cls]
                self.lending = budget.taken[c.name] - n >= self.prefill
            if self.lending or staging == self.prefill:
                break
            floors = dict(pinned)
        if keep and not self.lending and self._loan is not None:    # a staging layout: the step-peak loan reserved in
            keep = self._own_bytes(loan=True)                       # its own tail, the cache that much smaller
            budget = _Budget(self, size - keep - sum(c.bytes(0) for c in classes) - self._align * len(classes), floors)
            seed = self.policy.seed(self._profile_counts(), budget)
        self._own_tail = (size - keep, keep) if keep and not self.lending else None
        self.stage = self.stages[classes[0].staging]
        self.arena.view(torch.uint8, (size,)).zero_()         # the null slots; no slot left holding garbage
        free, arenas, at = {}, {}, 0
        for c in classes:
            c.pool, c.at, c.size = self.arena, at, c.null + c.staging + budget.taken[c.name]
            at = -(-(at + c.size * c.unit_bytes) // self._align) * self._align
            free[c.name] = list(range(c.size - 1, c.null + c.staging - 1, -1))      # popped lowest first
            arenas[c.name] = c.slots(), c.host_view()
        # each class's arena end: the first slot a streaming step does not lend to its ring and staging (a tail step
        # takes its own tail off it, `Lent`)
        self.lend_ends = torch.tensor([c.size for c in classes], device=self.ctx.device)
        self.lending_classes = torch.tensor([c.name in lenders for c in classes], device=self.ctx.device)
        # −1 − tier: every piece in the RAM tier (inclusive), or on the SSD until placed (exclusive)
        self.residency.host[...] = -2 if self.exclusive else -1
        if sorted(seed) != sorted(budget.granted):
            raise RuntimeError(f"{self.name}: the placement policy seeded other items than it took from the budget")
        pending: dict[str, list[tuple[int, int]]] = {c.name: [] for c in classes}    # exclusive: (piece, slot)
        for g, i in seed:
            for r, c in self.groups[groups[g]].items():
                slot = self.residency.host[g, i, r] = free[c.name].pop()
                arena, units = arenas[c.name]
                if self.exclusive:
                    pending[c.name].append((self._piece_index(c, g, i, r), slot))
                else:
                    arena[slot].copy_(units[self._host[g, i, r]], non_blocking=True)
        if self.exclusive:
            self._seed_exclusive(seed, pending, arenas)
        self.residency.seed()
        self._copies.clear()
        self.ring.fill_(-1)
        self.stage.copy_(self.top_k + torch.arange(self.stage.shape[1]).expand(self.roles, -1))
        self._mode, self.lent, self._parts = None, None, 1
        self.limits.copy_(torch.tensor([[c.size for _, c in sorted(self.groups[g].items())] for g in groups]))
        for t in (self.kinds, self.n_empty, *self.zeroed, *self.state.values()):
            t.zero_()
        self._written, self.peak_misses = 0, 0
        self._sized = False                     # the arena's size is checked at the first launch after a materialize
        self.borrowed = dict.fromkeys(self.borrowed, 0)

    def io_device(self) -> torch.Tensor:
        """The SSD reads' request rows, as `moe_tiers@1` writes them (mapped)."""
        return self.io_pool.device_view(torch.int32, (self.roles, 1 + self.prefill, 2))

    def _piece_index(self, c: SlotClass, g: int, i: int, r: int) -> int:
        """The class's index of group g's (in map order) item i's role r."""
        return int(self._pieces[c.name][g, i, r])

    def _seed_exclusive(self, seed: list[tuple[int, int]], pending: dict[str, list[tuple[int, int]]],
                        arenas: dict) -> None:
        """Exclusive tiers' seed (D10): the VRAM items' pieces read from the checkpoint into their slots (through the
        staging units), then RAM: F2's order past the VRAM items, as far as every role's class has units (the RAM
        units reloaded where they hold something else); every other piece on the SSD."""
        groups, classes = list(self._order), list(self.classes.values())
        for c in classes:                                   # VRAM, a staging batch at a time
            arena, units = arenas[c.name]
            batch, host = pending[c.name], units.numpy()
            for at in range(0, len(batch), c.staging):
                part = batch[at:at + c.staging]
                files.read([(e, host[c.held + j, o:o + e.nbytes]) for j, (n, _) in enumerate(part)
                            for e, o in _offsets(c.pieces[n][3])])
                for j, (_, slot) in enumerate(part):
                    arena[slot].copy_(units[c.held + j])
        vram, used = set(seed), dict.fromkeys(self.classes, 0)
        host, owner = np.full(self.residency.host.shape, -1, dtype=np.int32), np.full(tuple(self.owner.shape), -1,
                                                                                        dtype=np.int32)
        for g, i in _Budget.f2(self._profile_counts()):
            role = self.groups[groups[g]]
            if (g, i) in vram or any(used[c.name] >= c.held for c in role.values()):
                continue
            for r, c in role.items():
                host[g, i, r] = used[c.name]
                owner[classes.index(c), used[c.name]] = g * self.items + i
                self.residency.host[g, i, r] = -1
                used[c.name] += 1
        held = self.owner.cpu().numpy()                      # what the device's RAM tier holds now
        jobs = []
        for k, c in enumerate(classes):
            units = arenas[c.name][1].numpy()
            for u in range(c.held):
                want, now = owner[k, u], held[k, u] if self._ram[c.name][u] >= 0 else -1
                if want >= 0 and want != now:
                    g, i = divmod(int(want), self.items)
                    r = next(r for r, x in self.groups[groups[g]].items() if x is c)
                    extents = c.pieces[self._piece_index(c, g, i, r)][3]
                    jobs += [(e, units[u, o:o + e.nbytes]) for e, o in _offsets(extents)]
                self._ram[c.name][u] = want
        files.read(jobs)
        self._host = host
        self.host_units.copy_(torch.from_numpy(host))
        self.owner.copy_(torch.from_numpy(owner))
        for reads in self.reads:
            reads.bind()

    def before_launch(self, step) -> tuple[list[tuple[torch.Tensor, np.ndarray]], list[MemOp]]:
        if not self._sized:                     # launches follow the final materialize (the startup profile's minimum
            self._check_pinned()                # pools never launch): the arena's size is final here
            self._sized = True
            self._pipelining = bool(self.pipelining) and all(f() for f in self.pipelining) and \
                all(r.fetches_parts for r in self.resolvers)
            if self._loan is not None and not (self.lending and self.tail and self.prefill >= self.items):
                if self._own_tail is None:
                    raise RuntimeError(f"{self.name}: the engine reserved activations for a lent tail "
                                       "(moe.lend_peak), but this layout neither lends nor keeps a tail")
                print(f"holon: {self.name}: this layout stages (its cache cannot lend a prompt step's staging): the "
                      "step-peak loan (moe.lend_peak) stays reserved, in the pool's own tail")
        self._step, kinds, at = step.step, self._kinds, 0
        kinds[:self._written] = PADDING
        if step.row_kind is None:                                    # launched plans are the scheduler's
            raise ValueError(f"{self.name}: step {step.step} launched without row kinds (`BatchPlan.row_kind`)")
        decode = np.isin(step.row_kind, (RowKind.DECODE, RowKind.VERIFY)).tolist()
        for q, d in zip(step.n_query.tolist(), decode, strict=True):
            kinds[at:at + q] = DECODE if d else PREFILL
            at += q
        prompt = int(sum(q for q, d in zip(step.n_query.tolist(), decode, strict=True) if not d))
        if not (self.lending and self.policy.device_owned):
            self._mode = None
        elif prompt and self.ring_tokens and prompt >= self.ring_tokens:
            self._mode = "ring" if prompt >= self.stream_tokens else "stage"
        elif prompt and min(self.top_k * at, self.prefill) > self.stage.shape[1]:
            self._mode = "stage"                                 # a pass's misses may pass the staging slots
        else:
            self._mode = None
        # a staging step of large passes whose clients all pipeline: its staging a ring of parts (`Lent`)
        self._parts = self.pipeline if self._mode == "stage" and self.pipeline > 1 and self._pipelining and \
            self.large_passes(at) and not self.exclusive else 1
        self._step_tail = None
        if self._loaned:
            self._loan.release()
            if held := self._loan.held():
                raise RuntimeError(f"{self.name}: {held} bytes the last tail step allocated in its activation loan "
                                   "outlive it (a buffer kept across steps, made in a lending step: the tail's experts "
                                   "overwrite it)")
        self._loaned = False
        self._step_loan = 0
        if self._mode is not None and self.large_passes(at) and self.tail:   # what its own passes take, not the most
            cls, n = self.tail
            if self._loan is not None:          # the activations of a step of `at` prompt tokens, at most the worst's
                self._step_loan = min(self._loan_bytes, self._step_loan_bytes(at))
            need = max((f(at) for f in self.tail_bytes), default=0) + self._step_loan
            self._step_tail = (cls, min(n, -(-need // list(self.classes.values())[cls].unit_bytes)))
        elif self._loan is not None and self._own_tail is not None and self.large_passes(at):   # a staging layout's
            self._step_loan = min(self._loan_bytes, self._step_loan_bytes(at))           # loan: its own tail's end
        reach, self._written = max(at, self._written), at          # the last step's rows become padding
        self.step, self.step_tokens = step.step, at
        writes = [(self.kinds[:reach], kinds[:reach])] * bool(reach) + self.residency.writes(step.step)
        if not self._decode_set:                # once after each materialize (statics come back zeroed)
            writes.append((self._decode, np.full(len(self._decode), DECODE, dtype=np.uint8)))
            self._decode_set = True
        return writes, self._copies.pop(step.step, [])

    def after_commit(self, step, telemetry) -> None:
        if errors := {name: int(t[0]) for name, t in telemetry.items() if "errors" in name and int(t[0])}:
            raise RuntimeError(f"{self.name}: step {step.step}'s kernels flagged errors {errors} (errors: 1 = a "
                               "corrupt residency entry, 2 = the delta log overflowed, 4 = a step's lent slots fell "
                               "short of a pass's staging, 8 = the CPU's share past its job table; edit_errors: a "
                               "host edit's precondition failed; kernel_errors: the kernel's bits, the CPU worker's "
                               "1 = a task failed, 2 = a wait timed out)")
        mode = self._borrowing.pop(step.step, None)
        misses = int(telemetry["misses"][0])
        width = int(telemetry["stage_width"][0]) if mode else self.stage.shape[1]
        self.peak_misses = max(self.peak_misses, misses)
        if misses > width:
            raise RuntimeError(f"{self.name}: a layer had {misses} distinct misses, {width} slots to stage them")
        delta, n = telemetry["delta"].numpy(), telemetry["n_delta"].numpy()
        routed = telemetry["routed"].numpy()
        record = {"counts": telemetry["counts"].numpy(), "routed": (routed & 1).astype(bool),
                  "scope": (routed & 2).astype(bool),           # what the call with decode rows routed (variants)
                  "cpu": (routed & 4).astype(bool),             # the experts the CPU took (F5b: never admitted)
                  "records": np.concatenate([delta[g, :n[g]] for g in range(len(n))]),
                  "lends": telemetry["lends"].numpy().copy() if mode else None}   # None: borrowed nothing
        if mode:
            self.borrowed["ring" if mode == "ring" and int(telemetry["ring_ok"][0]) else "stage"] += 1
            self.borrowed["partial"] = self.borrowed.get("partial", 0) + (int(telemetry["ring_ok"][0]) == 2)
        self.residency.commit(step.step, record["records"], returned=mode is not None)
        for tap in self.taps:
            tap(step, record)
        if swaps := self.policy.after_commit(telemetry["counts"].numpy(), self.residency.host):
            self._swap(step.step, swaps)

    def shutdown(self) -> None:
        """The I/O worker first (its reads use the file tier's descriptors), then the tiers — unless the worker left
        its coordinator behind (still in a read past its time bound): then the descriptors stay open."""
        if self._loan is not None:
            self._loan.release()
        service = getattr(self.io, "service", None) if self.exclusive else None    # (a native worker's)
        if service is not None:
            service.shutdown()
            if service.left:
                return
        for t in self.tiers:
            getattr(t, "shutdown", lambda: None)()

    # -- a step's forward ---------------------------------------------------------------------------------------------
    def large_passes(self, tokens: int) -> bool:
        """A step of `tokens` takes large expert passes (D8; its stacks' kernels' `pass_rows`,
        docs/dev/v1-moe-prefill.md
        D1): when its prompt could stream through the ring — by its size and the store's terms, never the layout's, so
        that the startup profile's worst step (on minimum pools, where a fill cache holds nothing to lend) runs such
        passes too. A pass's misses fit a prefill pass's staging (`prefill` = every item), lent or not."""
        return (self.policy.device_owned and bool(self.ring_tokens) and tokens >= self.ring_tokens
                and self.prefill >= self.items)

    def begin(self) -> None:
        """At the step's first resolve (after its words are zeroed and its edits applied): its lent slots, if any."""
        if self._mode is not None:                 # a graphed step never gets here: it never borrows
            self._borrowing[self._step] = self._mode
            self.lend_needs.zero_()
            self.lent = Lent(self, self._mode, self._step_tail, self._parts)
            self.lent.begin()

    def lend_activations(self, peak: int) -> int:
        """The core's ActivationLender, at the startup profile: a tail store (`lend_peak`) lends its tail steps — from
        `ring_tokens` prompt tokens — their activations, `peak` bytes and the allocator's segments (`LOAN_MARGIN`,
        `LOAN_SLACK`) past their passes' buffers at the tail's end, in proportion to the step's prompt tokens (`peak`
        is the worst step's, of `max_step_tokens`), where its device has `activation_loan@1`; 0 where it lends none."""
        if not (self.lend_peak and self.ring_lend == "tail" and self.ring_tokens and self.policy.device_owned
                and self.ctx.candidates(ACTIVATION_LOAN, {})):
            return 0
        loan = -(-(int(peak * LOAN_MARGIN) + LOAN_SLACK) // self._align) * self._align
        self._loan_bytes = loan
        try:
            tail = self._tail(lambda c: c.cap)         # the tail grows by the loan (for the final materialize)
        except ValueError:                             # its class cannot hold it beside a pass's staging
            self._loan_bytes = 0
            print(f"holon: {self.name}: the tail's class cannot hold the step-peak loan (moe.lend_peak) beside a "
                  "prefill pass's staging: the activations stay reserved")
            return 0
        self._loan = self.ctx.select_kernel(ACTIVATION_LOAN, site=f"std.tiered[{self.name}]").impl(self.ctx)
        self._own_tail = None           # the smaller profile next: its passes' buffers are its own, as they will be
        self._loan_per_token = peak * LOAN_MARGIN / max(self.ctx.options.get("max_step_tokens", 1), 1)
        self.tail = tail
        return self.ring_tokens

    def activations(self, step):
        """The core's ActivationLender, at an eager step's launch: a tail step's forward allocates from the tail's
        last bytes, its loan's (lent with the tail at its first resolve: nothing reads the tail's experts before, they
        are never read after); None for any other step."""
        if self._loan is None or not self._step_loan:
            return None
        if self._step_tail is not None:
            cls, n = self._step_tail
            c = list(self.classes.values())[cls]
            loan = c.slots()[c.size - n:].view(-1)[-self._step_loan:]
        else:                                   # a staging layout's own tail, its last bytes
            at, n = self._own_tail
            loan = self.arena.view(torch.uint8, (self.arena.capacity * self.granule,))[at:at + n][-self._step_loan:]
        self._loaned = True
        return self._loan.use(loan[-loan.data_ptr() % 512:])             # (torch's blocks are 512-byte aligned)

    @property
    def parts(self) -> int:
        """This step's staging ring of parts (`Lent`): its parts, or 1."""
        return self._parts

    def pipelines(self, tokens: int) -> int:
        """The fetch parts of a pass of `tokens` in this step: `pipeline` in a staging step's large passes — in every
        pass of a step whose staging is a ring of parts (its slots hold two parts of a layer's misses at a time) —,
        else 1."""
        if self._parts > 1:
            return self._parts
        return self.pipeline if self._mode == "stage" and self.large_passes(tokens) and not self.exclusive else 1

    def fetch_stream(self, split: bool = False) -> torch.Stream:
        """The stream a call's fetch runs on beside the main stream (`DeviceResolver`), one per store and kind: a
        split call's (`split`) at high priority — the hits' pass beside it has its blocks on every SM (launched early,
        PDL), so a fetch of the default priority found no SM until they drained, ≈ 40 µs of each layer of a 5080's
        verify pass (FN §23.8) —; a pipelined pass's parts at the default priority."""
        if split not in self._fetch_streams:
            self._fetch_streams[split] = torch.Stream(device=self.ctx.device, priority=-1 if split else 0)
        return self._fetch_streams[split]

    def tail_view(self) -> torch.Tensor | None:
        """The step's lent tail (D11) as bytes — its class's last n slots, contiguous —, from the step's first resolve
        to its end; a staging layout's own tail (the pool's last bytes) in any step; else None."""
        if self.lent is None or self.lent.tail is None:
            if self._own_tail is None:
                return None
            at, n = self._own_tail              # fixed bytes of the pool, used within a step: graphs may hold them
            return self.arena.view(torch.uint8, (self.arena.capacity * self.granule,))[at:at + n - self._step_loan]
        cls, n = self.lent.tail
        c = list(self.classes.values())[cls]
        view = c.slots()[c.size - n:].view(-1)
        return view[:len(view) - self._step_loan]           # (the activation loan's bytes past the passes')

    def sources(self, g: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Layer g's ring and stage tables for the step (at its first pass)."""
        return self.lent.sources(g) if self.lent is not None else (self.ring[g], self.stage)

    def layer_done(self, g: int) -> None:
        if self.lent is not None:
            self.lent.done(g)
            if g == self.borrowing_groups - 1:
                self.lent = None

    def mark_auxiliary(self, group: int) -> None:
        """`group` is an auxiliary model's (FN D12a): after the target's in the map, never borrowing (`Lent.done`)."""
        self._auxiliary.add(group)

    @property
    def borrowing_groups(self) -> int:
        """The groups a borrowing step lends to, first in the map: the target's."""
        return len(self.groups) - len(self._auxiliary)

    # -- the policy's means -------------------------------------------------------------------------------------------
    def lend(self, need: torch.Tensor, call: int = 1) -> torch.Tensor:
        """Device lending under a device-owned policy (D4; a prefill step's `Lent` calls it at the step's first
        resolve, which zeroes the log and the error words, eagerly, no host sync): the whole experts its resolve
        contract's `lend` picks for `need` (slots per class) are unmapped and logged `(group, −1, item)`, their slots
        kept with the call's number (1…ROUNDS; 0 is the tail's) for their return to their layers, empty, at the step's
        end; their scores stay. Returns the picks [groups, items]."""
        picked = self.resolvers[0].resolve.lend(need, self.residency.device, self._class_index, **self.state)
        return self._unmap(picked, call)

    def lend_tail(self, cls: int, n: int) -> torch.Tensor:
        """A tail step's call 0 (docs/dev/v1-moe-prefill.md D11): the whole experts holding any of class `cls`'s last
        `n`
        slots, as the resolve contract's `lend_tail` picks them (the policy's spares there released), unmapped and
        logged as `lend`'s. Returns the picks [groups, items]."""
        picked = self.resolvers[0].resolve.lend_tail(cls, n, self.residency.device, self._class_index, self.limits,
                                                     **self.state)
        return self._unmap(picked, 0)

    def _unmap(self, picked: torch.Tensor, call: int) -> torch.Tensor:
        """`picked` [groups, items] unmapped in every role and logged `(group, −1, item)`, their slots kept with the
        call's number."""
        self.lent_slots.copy_(torch.where(picked[..., None], self.residency.device, self.lent_slots))
        self.lent_call.masked_fill_(picked, call)
        self.residency.device.masked_fill_(picked[..., None], -1)
        groups, items = picked.shape
        sink = self.delta.shape[1]
        at = (self.n_delta[:, None] + picked.cumsum(1) - 1).long()           # each group's records after its own
        at = torch.where(picked & (at < sink), at, sink)
        g = torch.arange(groups, device=picked.device)[:, None].expand(groups, items)
        e = torch.arange(items, device=picked.device)[None].expand(groups, items)
        self._log.view(-1, 3).index_copy_(0, (g * (sink + 1) + at).reshape(-1),
                                          torch.stack([g, -torch.ones_like(g), e], -1).reshape(-1, 3).int())
        lent = picked.sum(1).int()
        self.errors.bitwise_or_(((self.n_delta + lent) > sink).any().int() * 2)          # OVERFLOW
        self.n_delta += lent
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
    def _own_bytes(self, loan: bool = True) -> int:
        """The bytes a staging layout keeps for the large passes' buffers — and, `loan`, the step-peak loan's
        (`moe.lend_peak`: reserved there where the cache cannot lend it, its activations at the end) —, aligned."""
        if not self.tail:
            return 0
        cls, n = self.tail
        nbytes = n * list(self.classes.values())[cls].unit_bytes - (0 if loan else self._loan_bytes)
        return -(-nbytes // self._align) * self._align

    def _step_loan_bytes(self, at: int) -> int:
        """The loan of a step of `at` prompt tokens: its share of the worst step's, aligned (at most the worst's)."""
        return -(-(int(self._loan_per_token * at) + LOAN_SLACK) // self._align) * self._align

    def _tail(self, slots: Callable[[SlotClass], int]) -> tuple[int, int] | None:
        """The tail under `ring_lend = "tail"` (D11), where steps stream: the largest class arena — the first role's
        class most groups have, ties to the lower class index — and its last n slots, enough for the clients'
        `tail_bytes`; refused where that class's other cache slots (`slots(c)` cache slots, less the tail's) cannot
        lend a prefill pass's staging (D5): a tail step then always stages, never short."""
        if self.ring_lend != "tail" or not self.ring_tokens:        # (`resolver` stops streaming where none lends)
            return None
        classes = list(self.classes.values())
        first = [classes.index(self.groups[g][0]) for g in self._order]
        cls = max(sorted(set(first)), key=first.count)
        c = classes[cls]
        n = -(-(max((f() for f in self.tail_bytes), default=0) + self._loan_bytes) // c.unit_bytes)
        if slots(c) - n < self.prefill:
            raise ValueError(f"{self.name}: ring_lend tail (moe.ring_lend): class {c.name} lends {slots(c)} cache "
                             f"slots, {n} of them its tail ({n * c.unit_bytes} bytes): the rest cannot stage a "
                             f"prefill pass's {self.prefill} misses")
        return cls, n

    def _check_pinned(self) -> None:
        """D4: placement never follows free memory. A pinned arena that did not get its pinned size (it grows above
        the paged floor like a fill) refuses to serve."""
        if self._pinned is not None and self.arena.capacity < self._pinned:
            raise RuntimeError(
                f"{self.name}: the pinned cache ({', '.join(f'{c.name}={c.cap}' for c in self.classes.values())}) "
                f"does not fit above the paged floor: the arena got {self.arena.capacity * self.granule / 2**30:.3f} "
                f"of {self._pinned * self.granule / 2**30:.3f} GiB")

    @property
    def profile_dir(self) -> pathlib.Path:
        """The directory a profile's name is relative to: the experts' file's, as the weight source reads it (a hub
        cache's snapshot links resolve into its blob directory), so `holon_tools profile` writes where this reads."""
        return pathlib.Path(next(iter(self.classes.values())).pieces[0][3][0].path).parent

    @property
    def checkpoint(self) -> list[list]:
        """The files the pieces are read from, `[name, bytes]` each (as resolved: a hub cache's blob is named by its
        content hash), which a profile records: profiles of different files may share a directory (a hub cache's
        blobs)."""
        paths = {e.path for c in self.classes.values() for *_, extents in c.pieces for e in extents}
        return sorted([os.path.basename(p), os.path.getsize(p)] for p in paths)

    def _profile_counts(self) -> np.ndarray:
        """`profile`'s routing counts by group: a JSON file `{"checkpoint": …, "layers": {"<group>": [count per
        item]}}` beside the checkpoint (its name relative to the checkpoint's directory, so plans record no host path),
        refused unless it was counted on this checkpoint's files (or some of them: a profile of the target serves a
        deployment whose auxiliary model, an MTP head, adds its own file and group); groups it does not name count
        zero."""
        counts = np.zeros(self.residency.host.shape[:2])
        if not self._profile:
            return counts
        name = pathlib.PurePath(self._profile)
        if name.is_absolute() or ".." in name.parts:
            raise ValueError(f"{self.name}: profile {self._profile!r}: a name relative to the checkpoint's directory")
        data = json.loads((self.profile_dir / name).read_text())
        files = data.get("checkpoint") or []
        if not files or any(f not in self.checkpoint for f in files):
            raise ValueError(f"{self.name}: profile {self._profile} was counted on {data.get('checkpoint')}, not on "
                             f"this checkpoint {self.checkpoint} (`holon_tools profile` writes one per file)")
        for group, row in data["layers"].items():
            if int(group) not in self._order or len(row) != self.items:
                raise ValueError(f"{self.name}: profile {self._profile}: group {group} with {len(row)} counts; the "
                                 f"groups are {sorted(self._order)} of {self.items} items")
            counts[self._order[int(group)]] = row
        return counts


class _Budget:
    """What a policy seeds within: the arena pool's bytes past every class's null and staging slots, each class's
    pinned slots; an item takes a slot in each of its roles' classes. `floors` (class -> items): the store's layout
    wants that many of the class's items first (`warm_start`)."""

    def __init__(self, store: TieredStore, nbytes: int, floors: dict[str, int] | None = None) -> None:
        self.left, self.taken, self.granted = nbytes, dict.fromkeys(store.classes, 0), []
        self.classes = [list(store.groups[g].values()) for g in store._order]      # per group (in map order)
        self.floors = floors or {}

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

    def warm_start(self, profile: np.ndarray) -> list[tuple[int, int]]:
        """F2's order (`f2`), each floored class's first `floors[c]` items of it moved to the front."""
        order = self.f2(profile)
        if not self.floors:
            return order
        front = {}
        for name, n in self.floors.items():
            mine = [(g, i) for g, i in order if any(c.name == name for c in self.classes[g])]
            front |= dict.fromkeys(mine[:n])
        return list(front) + [x for x in order if x not in front]

    @staticmethod
    def f2(profile: np.ndarray) -> list[tuple[int, int]]:
        groups, items = profile.shape
        ranked = np.lexsort((np.broadcast_to(np.arange(items), profile.shape), -profile), axis=-1)   # per group
        floor = [(g, int(ranked[g, n])) for n in range(min(FLOOR, items)) for g in range(groups)]
        group, item = np.divmod(np.arange(profile.size), items)
        rest = np.lexsort((group, item, -profile.ravel()))
        taken = set(floor)
        return floor + [(int(group[x]), int(item[x])) for x in rest if (int(group[x]), int(item[x])) not in taken]


def _offsets(extents: list[Extent]) -> list[tuple[Extent, int]]:
    """Each extent with its offset in the piece (the pieces are their extents, concatenated)."""
    out, at = [], 0
    for e in extents:
        out.append((e, at))
        at += e.nbytes
    return out


class _Reads:
    """A layer's SSD reads (exclusive tiers): `cpu_task@1`'s copy task over its roles' descriptors, RAM tier pools and
    request rows, bound once the pools exist (the I/O worker reads `native` when it starts, after this)."""

    def __init__(self, store: TieredStore, g: int) -> None:
        self.store, self.g, self.task = store, g, None

    def bind(self) -> None:
        if self.task is None:
            s = self.store
            io = s.io_pool.view(torch.int32, (s.roles, 1 + s.prefill, 2))
            self.task = s.io.copy([(s.descriptors[c.name], c.host_view(), io[r])
                                   for r, c in enumerate(s.layer_classes[self.g])])

    @property
    def native(self):
        return self.task.native

    def __call__(self) -> None:
        self.task()


def _parse_cache(text: str) -> dict[str, int | float]:
    """`cache`: `<class>=<slots>` items, comma-separated, a slot count or (with a point) a fraction of the class's
    pieces; a class a name or a pattern (`aux.*`: an MTP head's classes; `*` the classes nothing else names, see
    `_cache_of`). Empty: every class fills free VRAM up to all its pieces."""
    out = {}
    for item in filter(None, (s.strip() for s in text.split(","))):
        name, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"cache item {item!r}: expected <class>=<slots or fraction>")
        out[name.strip()] = float(value) if "." in value else int(value)
    return out


def _cache_of(cache: dict[str, int | float], cls: str) -> int | float | None:
    """Class `cls`'s pinned cache: its own item, else the first pattern (fnmatch) naming it, `*` last; None: fill."""
    if cls in cache:
        return cache[cls]
    return next((v for k, v in sorted(cache.items(), key=lambda kv: kv[0] == "*") if fnmatch.fnmatchcase(cls, k)),
                None)
