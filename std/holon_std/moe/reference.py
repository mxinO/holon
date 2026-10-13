"""Reference for std.moe (oracles; also the floors' MoE block in the judge): softmax top-k routing, expert stacks
applied per (token, slot), the slot-ordered combine. fp32 throughout, as the op contracts specify."""
from __future__ import annotations

import torch

CLAMP = 6.1e-5            # llama.cpp's guard on the top-k sum (never reached by softmax: the sum is ≥ k/E)


def moe_topk(logits: torch.Tensor, top_k: int, renormalize: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """The `moe_topk@1` oracle: `(weights [T, k] f32, ids [T, k] int32)`."""
    x = logits.float()
    x = torch.where(x.isnan(), torch.finfo(torch.float32).min, x)
    probs = x.softmax(-1)
    order = probs.sort(dim=-1, descending=True, stable=True)            # ties to the lower index
    weights, ids = order.values[:, :top_k], order.indices[:, :top_k]
    if renormalize:
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(CLAMP)
    return weights.contiguous(), ids.int().contiguous()


def linear_experts(x: torch.Tensor, weight: torch.Tensor, ids: torch.Tensor, glu: str = "none",
                   operand_dtype: torch.dtype | None = None) -> torch.Tensor:
    """The `linear_experts@2` oracle over the stack's weights `weight` [E, out, in] (dequantized, or dense):
    `y [T, k, out_y]` f32. `operand_dtype`: a dense layout rounds the activation to the weight dtype first."""
    t, k = ids.shape
    xs = x.expand(t, k, x.shape[-1]) if x.shape[1] == 1 else x
    xs = xs.to(operand_dtype).float() if operand_dtype is not None else xs.float()
    y = xs.new_zeros(t, k, weight.shape[1])
    flat = ids.long()
    for e in flat.unique().tolist():                                    # one product per routed expert
        rows = flat == e
        y[rows] = xs[rows] @ weight[e].float().T
    if glu == "swiglu":
        gate, up = y.chunk(2, dim=-1)
        y = torch.nn.functional.silu(gate) * up
    elif glu != "none":
        raise ValueError(f"unknown glu {glu!r}")
    return y


def moe_combine(y: torch.Tensor, weights: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """The `moe_combine@1` oracle: `Σ_j w[t, j] · y[t, j]` in f32, slot by slot, cast to `dtype`."""
    acc = torch.zeros(y.shape[0], y.shape[2], dtype=torch.float32, device=y.device)
    for j in range(y.shape[1]):
        acc += weights[:, j, None].float() * y[:, j].float()
    return acc.to(dtype)
