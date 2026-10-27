"""placement.lru@1 (F4, docs/dev/v1-offload.md D1, D4): LRU fetch-on-miss over logical-step timestamps, the comparison
arm of F11. The cache is the device's: the host seeds it (F2's warm start) and never edits it; in the graph, the
policy's own resolve contract `moe_resolve.lru@1` (reference.py) stamps each layer's routes and gathers every miss
it can straight into the slot of the least recently routed resident the step does not need. CUDA provider:
kernels.expert_io; the CPU provider here is the oracle."""
from __future__ import annotations

import numpy as np
import torch
from holon.api import Constraint, KernelSpec

from . import reference

OP = "moe_resolve.lru"


class Lru:
    resolve_variant = OP + "@1"
    device_owned = True                       # the whole cache: no host edits

    def __init__(self, ctx, *, items: int, roles: int, **site) -> None:
        self.items = items

    def device_state(self) -> dict[str, tuple[tuple[int, ...], str]]:
        return {"last": ((self.items,), "int32"), "clock": ((1,), "int32")}

    def seed(self, profile: np.ndarray, budget) -> list[tuple[int, int]]:
        return [(g, i) for g, i in budget.warm_start(profile) if budget.take(g, i)]

    def after_commit(self, counts: np.ndarray, view: np.ndarray) -> list:
        return []


class Resolve:
    """The CPU provider: the oracle, writing what the contract writes in place."""

    def __init__(self, ctx, *, experts: int, top_k: int, roles: int, staging: int) -> None:
        self.staging = staging

    def __call__(self, ids, kinds, residency, host, limits, counts, routed, ring, stage, empty, n_empty, misses,
                 errors, delta, n_delta, group, plan, jobs, last, clock):
        out = reference.moe_resolve_lru(ids, kinds, residency, host, limits, self.staging, last, clock, group, ring,
                                        stage, empty, int(n_empty[0]), plan)
        counts += out["counts"]
        routed |= out["routed"]
        misses.clamp_(min=int(out["n_fetch"].max()))
        residency.copy_(out["residency"])
        last.copy_(out["last"])
        clock.copy_(out["clock"])
        at = int(n_delta[0])
        for i, record in enumerate(out["records"]):
            if at + i < len(delta):
                delta[at + i] = torch.tensor(record)
            else:
                out["errors"] |= reference.OVERFLOW
        n_delta += len(out["records"])
        n_empty.fill_(out["n_empty"])
        errors |= out["errors"]
        jobs[0, 0] = len(out["jobs"])
        for i, row in enumerate(out["jobs"]):
            jobs[1 + i, :len(row)] = torch.tensor(row)
        return out["slots"], out["fetch"], out["n_fetch"], out["cpu"]

    def lend(self, need, residency, classes, last, clock) -> torch.Tensor:
        return reference.lend(need.cpu().numpy(), residency, classes, last, clock)


class Plugin:
    def register(self, regs) -> None:
        regs.units.add("placement.lru", Lru)
        regs.ops.add(OP, reference.moe_resolve_lru)
        regs.kernels.add(OP, KernelSpec(op=OP, provider="placement.lru", impl=Resolve,
                                        requires=Constraint(devices=frozenset({"cpu"})),
                                        props=frozenset({"deterministic", "graph_safe"})))
