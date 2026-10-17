"""placement.lfu@1 (F11, docs/v1-offload.md D1, D4): decayed-count admission on fetch, the offload store's default
placement policy. The cache is the device's: the host seeds it (F2's warm start) and never edits it; in the graph,
the policy's own resolve contract `moe_resolve.lfu@1` (reference.py) decays and counts each layer's routes and
gathers a miss straight into the slot of the coldest resident the step does not need, when the miss is hotter.
CUDA provider: kernels.expert_io; the CPU provider here is the oracle."""
from __future__ import annotations

import numpy as np
import torch
from holon.api import Constraint, KernelSpec

from . import reference

OP = "moe_resolve.lfu"


class Lfu:
    resolve_variant = OP + "@1"
    device_owned = True                       # the whole cache: no host edits

    def __init__(self, ctx, *, items: int, roles: int, **site) -> None:
        self.items = items

    def device_state(self) -> dict[str, tuple[tuple[int, ...], str]]:
        return {"score": ((self.items,), "float32")}

    def seed(self, profile: np.ndarray, budget) -> list[tuple[int, int]]:
        return [(g, i) for g, i in budget.warm_start(profile) if budget.take(g, i)]

    def after_commit(self, counts: np.ndarray, view: np.ndarray) -> list:
        return []


class Resolve:
    """The CPU provider: the oracle, writing what the contract writes in place."""

    def __init__(self, ctx, *, experts: int, top_k: int, roles: int, staging: int) -> None:
        self.staging = staging

    def __call__(self, ids, kinds, residency, host, limits, counts, routed, ring, stage, empty, n_empty, misses,
                 errors, delta, n_delta, group, score):
        out = reference.moe_resolve_lfu(ids, kinds, residency, host, limits, self.staging, score, group, ring, stage,
                                        empty, int(n_empty[0]))
        counts += out["counts"]
        routed |= out["routed"]
        misses.clamp_(min=int(out["n_fetch"].max()))
        residency.copy_(out["residency"])
        score.copy_(out["score"])
        at = int(n_delta[0])
        for i, record in enumerate(out["records"]):
            if at + i < len(delta):
                delta[at + i] = torch.tensor(record)
            else:
                out["errors"] |= reference.OVERFLOW
        n_delta += len(out["records"])
        n_empty.fill_(out["n_empty"])
        errors |= out["errors"]
        return out["slots"], out["fetch"], out["n_fetch"]

    def lend(self, need, residency, classes, score) -> torch.Tensor:
        return reference.lend(need.cpu().numpy(), residency, classes, score)


class Plugin:
    def register(self, regs) -> None:
        regs.units.add("placement.lfu", Lfu)
        regs.ops.add(OP, reference.moe_resolve_lfu)
        regs.kernels.add(OP, KernelSpec(op=OP, provider="placement.lfu", impl=Resolve,
                                        requires=Constraint(devices=frozenset({"cpu"})),
                                        props=frozenset({"deterministic", "graph_safe"})))
