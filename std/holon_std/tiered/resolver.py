"""A group's in-graph resolve and fetch (docs/v1-offload.md D5): its routed ids to arena slots per role through the
residency map (`moe_resolve@1`, or the policy's variant), every miss gathered from the host tier into staging slots
(`expert_fetch@1`, F5a)."""
from __future__ import annotations

import torch

RESOLVE, FETCH = "moe_resolve@1", "expert_fetch@1"


class DeviceResolver:
    def __init__(self, ctx, store, classes: list, residency: torch.Tensor, host: torch.Tensor,
                 counts: torch.Tensor) -> None:
        self.store, self.classes = store, classes
        self.residency, self.host, self.counts = residency, host, counts
        facts = {"experts": store.items, "top_k": store.top_k, "roles": store.roles, "staging": store.staging}
        site = f"std.tiered[{store.name}]"
        self._resolve = ctx.select_kernel(RESOLVE, site=site, **facts).impl(ctx, **facts)
        self._fetch = []
        for c in classes:
            facts = {"unit_bytes": c.unit_bytes, "staging": c.staging}
            self._fetch.append(ctx.select_kernel(FETCH, site=f"{site}.{c.name}", **facts).impl(ctx, **facts))

    def __call__(self, ids: torch.Tensor, first: int) -> torch.Tensor:
        """Slots [roles, T, k] for `ids` [T, k], the step's tokens `first`… (their row kinds); the misses fetched."""
        kinds = self.store.kinds[first:first + len(ids)]
        if len(kinds) != len(ids):
            raise ValueError(f"tokens {first}…{first + len(ids)} past the step's {len(self.store.kinds)} row kinds")
        slots, fetch, n = self._resolve(ids, kinds, self.residency, self.host, self.counts, self.store.misses)
        for r, (c, gather) in enumerate(zip(self.classes, self._fetch, strict=True)):
            gather(fetch[r], n[r:r + 1], c.units(), c.slots(), c.null)
        return slots
