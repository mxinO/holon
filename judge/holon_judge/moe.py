"""The floors' MoE block (docs/m02-moe.md §4.6): HF's sparse-MoE block replaced by one built from std.moe's reference
functions, so a floor computes routing, the expert path and the combine with the engine's precision (design D3: an
F32 router, f32 expert outputs and SwiGLU, an f32 slot-ordered combine) — never looser than the engine — while the
rest of the model is HF's. D3 is defined once, in `reference.py`; its hash goes into the floor's meta."""
from __future__ import annotations

import functools
import hashlib
from collections.abc import Callable

import torch
from holon_std.moe import reference


def reference_hash() -> str:
    with open(reference.__file__, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def mirror(model: torch.nn.Module, operand_dtype: torch.dtype | None = None,
           experts: Callable | None = None, prefix: str = "") -> int:
    """Replace every sparse-MoE block's forward in `model` (its routers kept F32, as `golden.record` loads a floor's).
    `operand_dtype`: a dense layout's operand rounding (bf16 HF checkpoints); `experts(name, x, weight, ids, glu)`
    stands in for `reference.linear_experts` (Tier B's emulation), `name` the block's (`prefix` + its name in
    `model`, e.g. one streamed layer's) + `.gate_up` or `.down`. Returns the number of blocks replaced."""
    n = 0
    for name, module in model.named_modules():
        if type(module).__name__.endswith("SparseMoeBlock"):
            module.forward = functools.partial(_forward, module, prefix + name, operand_dtype, experts)
            n += 1
    return n


def _forward(block: torch.nn.Module, name: str, operand_dtype: torch.dtype | None, experts: Callable | None,
             hidden: torch.Tensor) -> torch.Tensor:
    router = block.gate
    x = hidden.reshape(-1, hidden.shape[-1])
    weights, ids = reference.moe_topk(torch.nn.functional.linear(x.float(), router.weight.float()), router.top_k,
                                      router.norm_topk_prob)
    apply = experts or (lambda _, *a: reference.linear_experts(*a, operand_dtype=operand_dtype))
    h = apply(f"{name}.gate_up", x[:, None], block.experts.gate_up_proj, ids, "swiglu")
    y = apply(f"{name}.down", h, block.experts.down_proj, ids, "none")
    return reference.moe_combine(y, weights, hidden.dtype).reshape(hidden.shape)
