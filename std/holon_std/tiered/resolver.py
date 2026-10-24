"""A group's in-graph resolve and fetch (docs/v1-offload.md D5): its routed ids to arena slots per role through the
residency map (`moe_resolve@1`, or the policy's variant contract with the policy's device state), every miss gathered
from the host tier into its slot — a staging slot, or the slot a device-owned policy admitted it into
(`expert_fetch@1`, F5a). Under exclusive tiers (D10) `moe_tiers@1` then places the call's victims and SSD misses, the
I/O worker reads its SSD pieces (`cpu_task@1`'s copy task), and after the gather the victims are demoted into their RAM
units and the admitted moved from staging into their slots."""
from __future__ import annotations

import torch

from .ring import SHORT

RESOLVE, FETCH, EXPERT_PREFETCH = "moe_resolve@1", "expert_fetch@1", "expert_prefetch@1"
ACTIVATION_LOAN = "activation_loan@1"
TIERS, CPU_TASK = "moe_tiers@1", "cpu_task@1"


class DeviceResolver:
    def __init__(self, ctx, store, group_id: int, classes: list) -> None:
        self.store, self.group_id, self.classes = store, group_id, classes
        self.group = None                     # its row in the store's tables, once attached (`attach`)
        self._prepared = False                # the pass is prepared, not yet resolved
        self.io_slot = None                   # exclusive tiers: its SSD reads' slot on the I/O worker
        facts = {"experts": store.items, "top_k": store.top_k, "roles": store.roles, "staging": store.staging}
        site = f"std.tiered[{store.name}]"
        op = store.policy.resolve_variant or RESOLVE
        self.resolve = ctx.select_kernel(op, site=site, **facts).impl(ctx, **facts)
        self._fetch = []
        for c in classes:
            facts = {"unit_bytes": c.unit_bytes, "staging": c.staging}
            self._fetch.append(ctx.select_kernel(FETCH, site=f"{site}.{c.name}", **facts).impl(ctx, **facts))

    @property
    def fetches_parts(self) -> bool:
        """Its fetch fetches a list part by part (`expert_fetch@1`'s `parts`, every role's in one launch)."""
        fetch = self._fetch[0]
        return bool(getattr(fetch, "parts", False)) and getattr(fetch, "gather_roles", None) is not None

    def attach(self, group: int, state: dict[str, torch.Tensor]) -> None:
        """Its row `group` of the store's device tables (made when every group has its pieces)."""
        store = self.store
        self.group, self.state = group, state
        self.residency, self.host = store.residency.device[group], store.host_units[group]
        self.limits, self.counts = store.limits[group], store.counts[group]

    def prepare(self, first: int) -> None:
        """Before the first use of the pass starting at the step's token `first`: on the step's first, its counts and
        error words from zero, its edits applied and (a prefill step) its slots borrowed; on the layer's first, its ring
        and stage tables. Once per pass (a route transform's map and the resolve both need it)."""
        store = self.store
        if self._prepared:
            return
        self._prepared = True
        if not self.group and not first:
            for t in store.zeroed:
                t.zero_()
            store.residency.apply(store.edit_errors)
            store.begin()
        if not first:
            self.ring, self.stage = store.sources(self.group)

    def present(self, first: int) -> torch.Tensor:
        """The layer's map with this step's ring pieces in their ring slots: what a route transform reads."""
        self.prepare(first)
        return torch.where(self.ring >= 0, self.ring, self.residency)

    def __call__(self, ids: torch.Tensor, first: int, plan: torch.Tensor | None = None,
                 jobs: torch.Tensor | None = None, before_fetch=None,
                 kinds: torch.Tensor | None = None, split: bool = False, parts: int = 1):
        """Slots [roles, T, k] for `ids` [T, k], the step's tokens `first`… (their row kinds, or `kinds`), the misses
        fetched; and
        `cpu` [T, k], the entries the CPU computes (F5b: its share by `plan`, its jobs in `jobs`; none without), handed
        to `before_fetch(cpu)` before the fetch (the CPU's jobs start, then the GPU's misses come over PCIe: D5)."""
        store = self.store
        kinds = store.kinds[first:first + len(ids)] if kinds is None else kinds     # given: a drafter sub-step's
        if len(kinds) != len(ids):
            raise ValueError(f"tokens {first}…{first + len(ids)} past the step's {len(store.kinds)} row kinds")
        self.prepare(first)
        if store.exclusive:
            store.call_start.copy_(store.n_delta[self.group:self.group + 1])
        split = split and getattr(self.resolve, "splits", False) and not store.exclusive and before_fetch is None
        parts = parts if self.fetches_parts and not store.exclusive and before_fetch is None else 1
        if store.parts > 1 and parts <= 1 and self.group < store.borrowing_groups:
            raise RuntimeError(f"{store.name}: a pass of layer {self.group} did not pipeline in a step whose staging "
                               "is a ring of parts (its slots hold two parts of a layer's misses)")
        defer = (store.cpu_admit or getattr(store.policy, "defer_cpu", False)) and before_fetch is not None and \
            getattr(self.resolve, "defers", False)
        # the policy's fill of empty slots (FN §21.15): the target's layers, their pieces listed apart (`later`)
        fill = getattr(self.resolve, "fills", False) and self.group < store.borrowing_groups and not store.exclusive
        fills = {"fill": fill} if getattr(self.resolve, "fills", False) else {}
        out = self.resolve(
            ids, kinds, self.residency, self.host, self.limits, self.counts, store.routed[self.group], self.ring,
            self.stage, store.empty[self.group], store.n_empty[self.group:self.group + 1], store.misses, store.errors,
            store.delta[self.group], store.n_delta[self.group:self.group + 1], self.group,
            store.no_plan if plan is None else plan, store.no_jobs if jobs is None else jobs,
            **({"split": True} if split else {}), **({"defer": True} if defer or fill else {}), **fills,
            **self.state)
        slots, fetch, n, cpu = out[:4]
        if fill and not defer:                   # the fill's pieces: copied after the layer (`fetch_later`)
            store.fetch_later(self._fetch, self.classes, out[-2], out[-1])
        self._prepared = False
        if split:                                # the fetch on a side stream; the hits' passes run meanwhile
            main = torch.accelerator.current_stream(ids.device)
            side = store.fetch_stream(split=True)
            side.wait_stream(main)
            with side:
                self._gather(fetch, n)
            fetch.record_stream(side)            # the main stream's allocations must not reuse them meanwhile
            n.record_stream(side)
            return slots, cpu, out[4], lambda: main.wait_stream(side)
        if parts > 1:                            # the misses part by part on a side stream, each part's entries known
            return self._pipelined(ids, slots, fetch, n, parts)
        if before_fetch is not None:
            before_fetch(cpu)
        if store.exclusive:
            g = self.group
            store.tier_op(ids, kinds, g, fetch, n, self.stage, store.delta[g], store.call_start,
                          store.n_delta[g:g + 1], store.residency.device, store.host_units, store.piece,
                          store.classes32, store.owner, store.capacity, store.heat, store.io_device(), store.moves,
                          store.n_moves)
            store.io.signal(self.io_slot)
            store.io.wait(self.io_slot)
        if store.cpu_admit or getattr(store.policy, "defer_cpu", False):
            store.wait_later()                   # an earlier call's admitted CPU experts off the link first (a fill's
                                                 # share it: joined at the target's last layer only)
        self._gather(fetch, n)
        if defer:                                # the CPU's admitted experts: their pieces after the call
            store.fetch_later(self._fetch, self.classes, out[4], out[5])
        if store.exclusive:
            for r, (c, gather) in enumerate(zip(self.classes, self._fetch, strict=True)):
                gather(store.moves[r, 0:2], store.n_moves[r, 0:1], c.slots(), c.units())      # demotions
                gather(store.moves[r, 2:4], store.n_moves[r, 1:2], c.slots(), c.slots())      # admitted, staged
        return slots, cpu

    def _pipelined(self, ids: torch.Tensor, slots: torch.Tensor, fetch: torch.Tensor, n: torch.Tensor, parts: int):
        """(slots, part [T, k], landed): the misses' fetch in `parts` parts on the store's fetch stream — part i the
        entries [i·q, (i + 1)·q) of each role's list, q = ⌈n / parts⌉ —, an entry's part the latest of its pieces'
        (−1: resident or a null slot; a fetched piece's rank: its host unit's place in the role's list, staged or
        admitted). Parts 0 and 1 start now; `landed[i]()` — called in order, once the work of the parts before i is
        enqueued — starts part i + 1's fetch after that work (a staging ring of parts reuses part i − 1's slots:
        `Lent`), then makes the current stream wait for part i — in such a ring; else every part starts now."""
        store = self.store
        main = torch.accelerator.current_stream(slots.device)
        side = store.fetch_stream()
        side.wait_stream(main)
        events: list[torch.Event] = []
        ring = store.lent.width if store.lent is not None and store.lent.width is not None and \
            self.group < store.borrowing_groups else None          # a staging ring's slots per class

        def start(i: int) -> None:
            if i >= 2 and ring is not None:      # after the part two before it ran: its slots may be these
                side.wait_stream(main)
            with side:
                self._fetch[0].gather_roles(fetch, n, [c.units() for c in self.classes],
                                            [c.slots() for c in self.classes], part=i, parts=parts)
                events.append(torch.Event(device=slots.device))
                events[i].record(side)

        def landed(i: int) -> None:
            if i + 1 < parts and i >= 1 and ring is not None:
                start(i + 1)
            main.wait_event(events[i])
        for i in range(min(2, parts) if ring is not None else parts):
            start(i)
        fetch.record_stream(side)
        n.record_stream(side)
        width = self.stage.shape[1]
        part = torch.full(slots.shape[1:], -1, dtype=torch.int64, device=slots.device)
        live = ids >= 0
        for r, c in enumerate(self.classes):
            units = c.units().shape[0]
            listed = torch.arange(width, device=slots.device) < n[r]
            rank = torch.full((units + 1,), -1, dtype=torch.int64, device=slots.device)
            rank.scatter_(0, torch.where(listed, fetch[r, 0].long(), units), torch.arange(width, device=slots.device))
            rank[units] = -1
            at = torch.where(live, rank[self.host[ids.long().clamp(min=0), r].long()], -1)
            q = (n[r:r + 1].long().clamp(max=width) + parts - 1) // parts
            # an entry's part, the latest of its roles' (each role's list the same experts: residency is whole)
            part = torch.maximum(part, torch.where(at >= 0, at // q.clamp(min=1), -1))
            if ring is not None:                 # two parts past a ring the list wraps: never (a pass misses no more
                w = ring[int(store.class_table[self.group, r])]                # than it lacks)
                store.errors.bitwise_or_(((2 * q > w) & (n[r:r + 1].clamp(max=width) > w)).int() * SHORT)
        return slots, part, [lambda i=i: landed(i) for i in range(parts)]

    def _gather(self, fetch: torch.Tensor, n: torch.Tensor) -> None:
        roles = getattr(self._fetch[0], "gather_roles", None)
        if roles is not None:                    # every role's misses in one launch
            roles(fetch, n, [c.units() for c in self.classes], [c.slots() for c in self.classes])
        else:
            for r, (c, gather) in enumerate(zip(self.classes, self._fetch, strict=True)):
                gather(fetch[r], n[r:r + 1], c.units(), c.slots())

    def done(self) -> None:
        """After the layer's expert passes (the step's ring moves on)."""
        self.store.layer_done(self.group)
