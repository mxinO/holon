"""A group's in-graph resolve and fetch (docs/v1-offload.md D5): its routed ids to arena slots per role through the
residency map (`moe_resolve@1`, or the policy's variant contract with the policy's device state), every miss gathered
from the host tier into its slot — a staging slot, or the slot a device-owned policy admitted it into
(`expert_fetch@1`, F5a). Under exclusive tiers (D10) `moe_tiers@1` then places the call's victims and SSD misses, the
I/O worker reads its SSD pieces (`cpu_task@1`'s copy task), and after the gather the victims are demoted into their RAM
units and the admitted moved from staging into their slots."""
from __future__ import annotations

import torch

RESOLVE, FETCH, EXPERT_PREFETCH = "moe_resolve@1", "expert_fetch@1", "expert_prefetch@1"
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
                 kinds: torch.Tensor | None = None, split: bool = False):
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
        out = self.resolve(
            ids, kinds, self.residency, self.host, self.limits, self.counts, store.routed[self.group], self.ring,
            self.stage, store.empty[self.group], store.n_empty[self.group:self.group + 1], store.misses, store.errors,
            store.delta[self.group], store.n_delta[self.group:self.group + 1], self.group,
            store.no_plan if plan is None else plan, store.no_jobs if jobs is None else jobs,
            **({"split": True} if split else {}), **self.state)
        slots, fetch, n, cpu = out[:4]
        self._prepared = False
        if split:                                # the fetch on a side stream; the hits' passes run meanwhile
            main = torch.cuda.current_stream(ids.device)
            side = store.fetch_stream()
            side.wait_stream(main)
            with torch.cuda.stream(side):
                self._gather(fetch, n)
            fetch.record_stream(side)            # the main stream's allocations must not reuse them meanwhile
            n.record_stream(side)
            return slots, cpu, out[4], lambda: main.wait_stream(side)
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
        self._gather(fetch, n)
        if store.exclusive:
            for r, (c, gather) in enumerate(zip(self.classes, self._fetch, strict=True)):
                gather(store.moves[r, 0:2], store.n_moves[r, 0:1], c.slots(), c.units())      # demotions
                gather(store.moves[r, 2:4], store.n_moves[r, 1:2], c.slots(), c.slots())      # admitted, staged
        return slots, cpu

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
