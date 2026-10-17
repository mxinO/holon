"""transforms.tail_skip (A1, docs/v1-offload.md D1, D3, §6): Strata's route tail skip as a route transform, the
contract `moe_route_transform.tail_skip@1` this plugin owns (reference.py). A step's missed expert whose best routing
rank is at least `tail_skip.rank` (a plugin option; default 7) is dropped, its weight not renormalized. `approx`:
bound only by the explicit option `moe.route_transform = tail_skip`, human-gated (L2 KL, top-1, a task check), never in
a default lock. Providers: the oracle on the CPU, one Triton program on CUDA (kernel.py)."""
from __future__ import annotations

from holon.api import Constraint, KernelSpec

from . import reference

OP = "moe_route_transform.tail_skip"


def _rank(ctx) -> int:
    rank = int(ctx.options.get("plugins", {}).get("tail_skip.rank", reference.RANK))
    if rank < 1:
        raise ValueError(f"transforms.tail_skip: tail_skip.rank {rank}: at least 1 (0 would drop every miss)")
    return rank


class Reference:
    """The CPU provider: the oracle."""

    def __init__(self, ctx, *, experts: int, top_k: int) -> None:
        self.rank = _rank(ctx)

    def __call__(self, ids, weights, residency):
        return reference.tail_skip(ids, weights, residency, self.rank)


def _cuda(ctx, *, experts: int, top_k: int):
    from .kernel import TailSkip  # Triton only where CUDA runs it

    return TailSkip(ctx, experts=experts, top_k=top_k, rank=_rank(ctx))


class Plugin:
    def register(self, regs) -> None:
        regs.ops.add(OP, reference.tail_skip)
        regs.kernels.add(OP, KernelSpec(op=OP, provider="transforms.tail_skip", impl=Reference,
                                        requires=Constraint(devices=frozenset({"cpu"})),
                                        props=frozenset({"deterministic"})))
        regs.kernels.add(OP, KernelSpec(op=OP, provider="transforms.tail_skip", impl=_cuda,
                                        requires=Constraint(devices=frozenset({"cuda"})),
                                        props=frozenset({"deterministic", "graph_safe"})))
