"""A toy route transform (`moe_route_transform.drop_last@1`, docs/dev/v1-offload.md D3): every token's last routing rank
is dropped (id −1); weights unchanged; the layer's map is checked for its shape only. Its own contract, oracle and
provider (torch ops, any device), as a held-out transform plugin's would be."""
import torch
from holon.api import Constraint, KernelSpec

OP = "moe_route_transform.drop_last"


def drop_last(ids: torch.Tensor, weights: torch.Tensor, residency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if residency.dim() != 2:
        raise ValueError(f"drop_last: a layer's map [E, R], not {tuple(residency.shape)}")
    out = ids.clone()
    out[:, -1] = -1
    return out, weights


class DropLast:
    calls = 0

    def __init__(self, ctx, *, experts: int, top_k: int) -> None:
        pass

    def __call__(self, ids, weights, residency):
        DropLast.calls += 1
        return drop_last(ids, weights, residency)


class Plugin:
    def register(self, regs) -> None:
        regs.ops.add(OP, drop_last)
        regs.kernels.add(OP, KernelSpec(op=OP, provider="tests.drop_last", impl=DropLast,
                                        requires=Constraint(devices=frozenset({"cpu", "cuda"})),
                                        props=frozenset({"deterministic", "graph_safe"})))
