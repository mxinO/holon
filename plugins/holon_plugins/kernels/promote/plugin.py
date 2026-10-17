"""placement.promote's resolve contract `moe_resolve.promote@1` (F10) on CUDA: the policy's own kernel plugin, as
docs/v1-offload.md D3 has a resolve variant after the std offload API's tag ship its CUDA provider (kernels.expert_io
takes no new policy codes). A call is three launches on the current stream: the promotion CTA (stamp, promotions
into empty slots and spares: the map, the spares and the delta log), the store's own `moe_resolve@1` — whichever
provider serves it here — over the map as promoted, and the settling CTA (the promoted pieces' fetches after the staged
ones, the absent spares refilled from the least recently routed). Contract and oracle: placement.promote
(reference.py). `lend` runs eagerly (ring steps are never graphed) in torch ops.

The native module is built ahead of time (`python -m holon_plugins.kernels.promote.prebuild`); without it an instance
cannot be created and the error names that command."""
from __future__ import annotations

import torch
from holon.api import Constraint, KernelSpec

from . import extension

OP, RESOLVE = "moe_resolve.promote", "moe_resolve@1"
SHARED_BYTES = 48 << 10                     # the promotion CTA's four [E] tables, beside its static words


class Resolve:
    def __init__(self, ctx, *, experts: int, top_k: int, roles: int, staging: int) -> None:
        self.lib, self.experts, self.top_k, self.staging = extension.load(ctx.fingerprint.sm), experts, top_k, staging
        facts = {"experts": experts, "top_k": top_k, "roles": roles, "staging": staging}
        self.resolve = ctx.select_kernel(RESOLVE, site="kernels.promote", **facts).impl(ctx, **facts)

    def __call__(self, ids, kinds, residency, host, limits, counts, routed, ring, stage, empty, n_empty, misses,
                 errors, delta, n_delta, group, plan, jobs, last, clock, spare):
        out = torch.empty(1 + self.experts, dtype=torch.int32, device=ids.device)   # the promoted, in order
        self.lib.promote(ids, kinds, residency, limits, empty, n_empty, routed, errors, delta, n_delta, group, last,
                         clock, spare, out, self.staging)
        slots, fetch, n_fetch, cpu = self.resolve(ids, kinds, residency, host, limits, counts, routed, ring, stage,
                                                  empty, n_empty, misses, errors, delta, n_delta, group, plan, jobs)
        self.lib.settle(out, residency, host, limits, empty, n_empty, fetch, n_fetch, misses, last, clock, spare,
                        self.top_k, self.staging)
        return slots, fetch, n_fetch, cpu

    def lend(self, need: torch.Tensor, residency: torch.Tensor, classes: torch.Tensor, last: torch.Tensor,
             **state) -> torch.Tensor:
        """Whole resident experts least recently routed first, ties by expert id then layer; for each class c the
        first need[c] of them holding a slot of c. Device ops, no host sync."""
        groups, experts, _ = residency.shape
        whole = (residency >= 0).all(-1).t().reshape(-1)                      # [E·G], in (expert, layer) order
        key = torch.where(whole, last.t().reshape(-1), torch.iinfo(torch.int32).max)
        order = torch.sort(key, stable=True).indices                        # stable: ties keep (expert, layer) order
        holds = torch.zeros(groups, len(need), dtype=torch.bool, device=key.device).scatter_(1, classes, True)
        held = holds.repeat(experts, 1)[order] & whole[order, None]          # [E·G, C] in the order
        take = (held & (held.cumsum(0) <= need.to(key.device)[None])).any(1)
        picked = torch.zeros_like(whole).scatter_(0, order, take)
        return picked.view(experts, groups).t()


class Plugin:
    def register(self, regs) -> None:
        def supports(f) -> bool:
            return 4 * f["experts"] * 4 + 64 <= SHARED_BYTES
        regs.kernels.add(OP, KernelSpec(op=OP, provider="kernels.promote", impl=Resolve,
                                        requires=Constraint(sms=frozenset(extension.ARCHS)), supports=supports,
                                        props=frozenset({"graph_safe", "deterministic"}), priority=10))
