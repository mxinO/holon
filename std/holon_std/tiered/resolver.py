"""A group's in-graph resolve and fetch (docs/v1-offload.md D5): its routed ids to arena slots per role through the
residency map (`moe_resolve@1`, or the policy's variant contract with the policy's device state), every miss gathered
from the host tier into its slot — a staging slot, or the slot a device-owned policy admitted it into
(`expert_fetch@1`, F5a)."""
from __future__ import annotations

import torch

RESOLVE, FETCH, EXPERT_PREFETCH = "moe_resolve@1", "expert_fetch@1", "expert_prefetch@1"


class DeviceResolver:
    def __init__(self, ctx, store, group: int, classes: list, state: dict[str, torch.Tensor]) -> None:
        self.store, self.group, self.classes, self.state = store, group, classes, state
        self._prepared = False                # the pass is prepared, not yet resolved
        self.residency, self.host = store.residency.device[group], store.host_units[group]
        self.limits, self.counts = store.limits[group], store.counts[group]
        facts = {"experts": store.items, "top_k": store.top_k, "roles": store.roles, "staging": store.staging}
        site = f"std.tiered[{store.name}]"
        op = store.policy.resolve_variant or RESOLVE
        self.resolve = ctx.select_kernel(op, site=site, **facts).impl(ctx, **facts)
        self._fetch = []
        for c in classes:
            facts = {"unit_bytes": c.unit_bytes, "staging": c.staging}
            self._fetch.append(ctx.select_kernel(FETCH, site=f"{site}.{c.name}", **facts).impl(ctx, **facts))

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

    def __call__(self, ids: torch.Tensor, first: int) -> torch.Tensor:
        """Slots [roles, T, k] for `ids` [T, k], the step's tokens `first`… (their row kinds); the misses fetched."""
        store = self.store
        kinds = store.kinds[first:first + len(ids)]
        if len(kinds) != len(ids):
            raise ValueError(f"tokens {first}…{first + len(ids)} past the step's {len(store.kinds)} row kinds")
        self.prepare(first)
        slots, fetch, n = self.resolve(ids, kinds, self.residency, self.host, self.limits, self.counts,
                                       store.routed[self.group], self.ring, self.stage, store.empty[self.group],
                                       store.n_empty[self.group:self.group + 1], store.misses, store.errors,
                                       store.delta[self.group], store.n_delta[self.group:self.group + 1], self.group,
                                       **self.state)
        self._prepared = False
        for r, (c, gather) in enumerate(zip(self.classes, self._fetch, strict=True)):
            gather(fetch[r], n[r:r + 1], c.units(), c.slots())
        return slots

    def done(self) -> None:
        """After the layer's expert passes (the step's ring moves on)."""
        self.store.layer_done(self.group)
