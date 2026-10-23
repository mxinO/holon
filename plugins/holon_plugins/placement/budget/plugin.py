"""placement.budget@1 (FN step 7): F11's decayed-count admission held back, as Strata's adaptive tier is, so a CPU
share pays — the CPU's experts admitted as any miss, their copies after the layer while the CPU serves the call, and
a candidate admitted only past its victim by `budget.min_gain` (at most `budget.per_layer` a call). The policy's
resolve contract `moe_resolve.budget@1` (reference.py); its CUDA provider is kernels.expert_io's (F11's kernel with
the cap and the margin), the CPU provider here is the oracle."""
from __future__ import annotations

from holon.api import Constraint, KernelSpec

from ..lfu.plugin import Lfu
from ..lfu.plugin import Resolve as LfuResolve
from . import reference

OP = "moe_resolve.budget"


def _terms(ctx) -> dict:
    opts = ctx.options.get("plugins", {})
    return {"budget": int(opts.get("budget.per_layer", -1)), "gain": float(opts.get("budget.min_gain", 0.5))}


class Budget(Lfu):
    resolve_variant = OP + "@1"
    defer_cpu = True                          # the CPU's experts admitted (the call's `defer`)

    def __init__(self, ctx, *, items: int, roles: int, **site) -> None:
        super().__init__(ctx, items=items, roles=roles, **site)
        self.replay_terms = {**getattr(self, "replay_terms", {}), **_terms(ctx)}   # the L1 replay's (judge.placement)


class Resolve(LfuResolve):
    """The CPU provider: the oracle with the policy's terms."""

    def __init__(self, ctx, **facts) -> None:
        super().__init__(ctx, **facts)
        terms = _terms(ctx)
        self.budget, self.gain = None if terms["budget"] < 0 else terms["budget"], terms["gain"]


class Plugin:
    def register(self, regs) -> None:
        regs.units.add("placement.budget", Budget)
        regs.ops.add(OP, reference.moe_resolve_budget)
        regs.kernels.add(OP, KernelSpec(op=OP, provider="placement.budget", impl=Resolve,
                                        requires=Constraint(devices=frozenset({"cpu"})),
                                        props=frozenset({"deterministic", "graph_safe"})))

