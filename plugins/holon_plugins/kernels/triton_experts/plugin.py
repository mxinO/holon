"""Our Triton grouped GEMM as the `linear_experts@2` kernel for dense expert stacks (`dense/<dtype>`; op contract:
std.moe). Correct and graph-safe; not tuned (vLLM / SGLang's fused-MoE kernels are the performance reference)."""
from __future__ import annotations

import torch
from holon.api import Constraint, KernelSpec

from .kernel import grouped

OP = "linear_experts"


class LinearExperts:
    def __init__(self, ctx, *, glu: str, **facts) -> None:
        self.glu = glu

    def __call__(self, x: torch.Tensor, parts: dict, ids: torch.Tensor) -> torch.Tensor:
        return grouped(x, parts["weight"], ids, self.glu)


def _supports(f: dict) -> bool:
    return (f["layout"] in ("dense/bfloat16", "dense/float16") and f["out_dtype"] == "float32"
            and f["glu"] in ("none", "swiglu") and f["rows_per_token"] in (1, f["top_k"]))


class Plugin:
    def register(self, regs) -> None:
        regs.kernels.add(OP, KernelSpec(op=OP, provider="kernels.triton_experts", impl=LinearExperts,
                                        requires=Constraint(sm_min=80), supports=_supports,
                                        props=frozenset({"graph_safe", "deterministic"})))
