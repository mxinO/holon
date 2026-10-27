"""placement.promote@1 (F10, docs/dev/v1-offload.md D1, §13): Project Maya's promote-on-fetch — each layer keeps 3 spare
slots the device owns; in the graph a decode miss is promoted into a spare (in route order) and becomes resident, and
at the step boundary each spare taken is refilled by evicting the layer's least recently routed expert. The host seeds
the cache (F2's warm start) and never edits it. The policy's own resolve contract `moe_resolve.promote@1`
(reference.py) owns the spares, its device state; the CPU provider here is the oracle, the CUDA provider is the
policy's own kernel plugin, kernels.promote."""
from __future__ import annotations

import numpy as np
import torch
from holon.api import Constraint, KernelSpec

from . import reference

OP = "moe_resolve.promote"


class Promote:
    resolve_variant = OP + "@1"
    device_owned = True                       # the whole cache, spares included: no host edits

    def __init__(self, ctx, *, items: int, roles: int, **site) -> None:
        self.items, self.roles = items, roles

    def device_state(self) -> dict[str, tuple[tuple[int, ...], str]]:
        return {"last": ((self.items,), "int32"), "clock": ((1,), "int32"),
                "spare": ((reference.SPARES, 1 + self.roles), "int32")}

    def seed(self, profile: np.ndarray, budget) -> list[tuple[int, int]]:
        return [(g, i) for g, i in budget.warm_start(profile) if budget.take(g, i)]

    def after_commit(self, counts: np.ndarray, view: np.ndarray) -> list:
        return []


class Resolve:
    """The CPU provider: the oracle, writing what the contract writes in place."""

    def __init__(self, ctx, *, experts: int, top_k: int, roles: int, staging: int) -> None:
        self.staging = staging

    def __call__(self, ids, kinds, residency, host, limits, counts, routed, ring, stage, empty, n_empty, misses,
                 errors, delta, n_delta, group, plan, jobs, last, clock, spare):
        out = reference.moe_resolve_promote(ids, kinds, residency, host, limits, self.staging, last, clock, spare,
                                            group, ring, stage, empty, int(n_empty[0]), plan)
        counts += out["counts"]
        routed |= out["routed"]
        misses.clamp_(min=int(out["n_fetch"].max()))
        for t, name in ((residency, "residency"), (last, "last"), (clock, "clock"), (spare, "spare")):
            t.copy_(out[name])
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

    def lend(self, need, residency, classes, last, clock, spare) -> torch.Tensor:
        return reference.lend(need.cpu().numpy(), residency, classes, last, clock, spare)


class Plugin:
    def register(self, regs) -> None:
        regs.units.add("placement.promote", Promote)
        regs.ops.add(OP, reference.moe_resolve_promote)
        regs.kernels.add(OP, KernelSpec(op=OP, provider="placement.promote", impl=Resolve,
                                        requires=Constraint(devices=frozenset({"cpu"})),
                                        props=frozenset({"deterministic", "graph_safe"})))
