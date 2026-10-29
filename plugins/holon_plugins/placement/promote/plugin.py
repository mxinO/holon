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


BUDGET = 1        # RAM -> VRAM promotions a layer a decode call (docs/dev/v1-glm.md D8t: Maya admits every miss; one
#                   keeps the hit rate and halves the promoted bytes, replayed on GLM's decode routes)


class Promote:
    resolve_variant = OP + "@1"
    device_owned = True                       # the whole cache, spares included: no host edits
    exclusive_flag = "exclusive"              # the device state the store sets under exclusive tiers
    heat = "last"                             # what RAM's coldest is by under exclusive tiers: the least recent
    reserved = reference.SPARES               # each layer's spares: cache slots a step's lending never takes
    prompt_admissions = False                 # decode misses only (a prompt step's staged pieces never move)

    def __init__(self, ctx, *, items: int, roles: int, **site) -> None:
        self.items, self.roles = items, roles
        # D8t: at most BUDGET RAM misses a layer promotes a decode call (SSD misses and spare owners besides), the
        # resolve contract's `budget` fact; a test's admit-all arm sets BUDGET None: Maya's promote-on-fetch
        self.resolve_facts = {} if BUDGET is None else {"budget": BUDGET}

    def device_state(self) -> dict[str, tuple[tuple[int, ...], str]]:
        return {"last": ((self.items,), "int32"), "clock": ((1,), "int32"),
                "spare": ((reference.SPARES, 1 + self.roles), "int32"),
                "exclusive": ((1,), "int32")}         # 1 under exclusive tiers (the store's): refills, returns logged

    def seed(self, profile: np.ndarray, budget) -> list[tuple[int, int]]:
        return [(g, i) for g, i in budget.warm_start(profile) if budget.take(g, i)]

    def after_commit(self, counts: np.ndarray, view: np.ndarray) -> list:
        return []


class Resolve:
    """The CPU provider: the oracle, writing what the contract writes in place."""

    def __init__(self, ctx, *, experts: int, top_k: int, roles: int, staging: int, budget: int | None = None) -> None:
        self.staging, self.budget = staging, budget

    def __call__(self, ids, kinds, residency, host, limits, counts, routed, ring, stage, empty, n_empty, misses,
                 errors, delta, n_delta, group, plan, jobs, last, clock, spare, exclusive):
        out = reference.moe_resolve_promote(ids, kinds, residency, host, limits, self.staging, last, clock, spare,
                                            group, ring, stage, empty, int(n_empty[0]), plan, bool(exclusive[0]),
                                            budget=self.budget)
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

    def lend(self, need, residency, classes, last, clock, spare, **state) -> torch.Tensor:
        return reference.lend(need.cpu().numpy(), residency, classes, last, clock, spare)

    def lend_tail(self, cls, n, residency, classes, limits, spare, **state) -> torch.Tensor:
        return reference.release_tail(cls, n, residency, classes, limits, spare)


class Plugin:
    def register(self, regs) -> None:
        regs.units.add("placement.promote", Promote)
        regs.ops.add(OP, reference.moe_resolve_promote)
        regs.kernels.add(OP, KernelSpec(op=OP, provider="placement.promote", impl=Resolve,
                                        requires=Constraint(devices=frozenset({"cpu"})),
                                        props=frozenset({"deterministic", "graph_safe"})))
