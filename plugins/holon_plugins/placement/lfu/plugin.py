"""placement.lfu@1 (F11, docs/v1-offload.md D1, D4): decayed-count admission on fetch, the offload store's default
placement policy. The cache is the device's: the host seeds it (F2's warm start) and never edits it; in the graph,
the policy's own resolve contract `moe_resolve.lfu@1` (reference.py) decays and counts each layer's routes and
gathers a miss straight into the slot of the coldest resident the step does not need, when the miss is hotter. It
lends a streaming step its coldest experts, or first a tail (`lend_tail`, docs/v1-moe-prefill.md D11).
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
    heat = "score"                            # an exclusive RAM tier ranks by it (D10)

    def __init__(self, ctx, *, items: int, roles: int, **site) -> None:
        self.items = items
        fill = fill_terms(ctx)
        if fill["fill_every"] > 0:
            self.replay_terms = fill          # what the L1 replay needs (judge.placement)

    def device_state(self) -> dict[str, tuple[tuple[int, ...], str]]:
        return {"score": ((self.items,), "float32"), "tick": ((1,), "int32")}

    def seed(self, profile: np.ndarray, budget) -> list[tuple[int, int]]:
        return [(g, i) for g, i in budget.warm_start(profile) if budget.take(g, i)]

    def after_commit(self, counts: np.ndarray, view: np.ndarray) -> list:
        return []


def fill_terms(ctx) -> dict:
    """The fill's options (FN §21.15): `lfu.fill_every` (0: off), `lfu.fill` (experts a due call fills)."""
    opts = ctx.options.get("plugins", {})
    return {"fill_every": int(opts.get("lfu.fill_every", 0)), "fill": int(opts.get("lfu.fill", 1))}


class Resolve:
    """The CPU provider: the oracle, writing what the contract writes in place."""

    defers = True                          # the CPU's experts admitted, their fetch apart (`defer`)
    budget, gain = None, 0.0               # F11's terms (placement.budget's provider sets its own)

    def __init__(self, ctx, *, experts: int, top_k: int, roles: int, staging: int) -> None:
        self.staging = staging
        terms = fill_terms(ctx)
        self.fill, self.fill_every = terms["fill"], terms["fill_every"]
        self.fills = self.fill_every > 0 and self.fill > 0

    def __call__(self, ids, kinds, residency, host, limits, counts, routed, ring, stage, empty, n_empty, misses,
                 errors, delta, n_delta, group, plan, jobs, score, tick=None, defer: bool = False,
                 fill: bool = True):
        out = reference.moe_resolve_lfu(ids, kinds, residency, host, limits, self.staging, score, group, ring, stage,
                                        empty, int(n_empty[0]), plan, defer, self.budget, self.gain,
                                        None if tick is None else int(tick[0]), self.fill,
                                        self.fill_every if fill else 0)
        if tick is not None:
            tick.fill_(out["tick"])
        counts += out["counts"]
        routed |= out["routed"]
        misses.clamp_(min=int(max(out["n_fetch"].max(), out["n_later"].max() if defer else 0)))
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
        jobs[0, 0] = len(out["jobs"])
        for i, row in enumerate(out["jobs"]):
            jobs[1 + i, :len(row)] = torch.tensor(row)
        return (out["slots"], out["fetch"], out["n_fetch"], out["cpu"]) + (
            (out["later"], out["n_later"]) if defer else ())

    def lend(self, need, residency, classes, score, tick=None) -> torch.Tensor:
        return reference.lend(need.cpu().numpy(), residency, classes, score)

    def lend_tail(self, cls, n, residency, classes, limits, score, tick=None) -> torch.Tensor:
        return reference.lend_tail(cls, n, residency, classes, limits)


class Plugin:
    def register(self, regs) -> None:
        regs.units.add("placement.lfu", Lfu)
        regs.ops.add(OP, reference.moe_resolve_lfu)
        regs.kernels.add(OP, KernelSpec(op=OP, provider="placement.lfu", impl=Resolve,
                                        requires=Constraint(devices=frozenset({"cpu"})),
                                        props=frozenset({"deterministic", "graph_safe"})))
