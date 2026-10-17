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


def expert_parts(parts: dict[str, torch.Tensor], experts: int, merged: bool) -> list[list[dict[str, torch.Tensor]]]:
    """A `linear_experts@2` stack's parts per expert, then per unit it was merged from (gate's, then up's when
    `merged`; else one): a part of rank ≥ 3 is per row, [E, rows, …], each unit taking its share of the expert's rows;
    one of rank ≤ 2 is per tensor, [E] or [E, 1], or merged [2, E] (gate's row, then up's). For a dequantizer that
    reads one unit's parts at a time."""
    units = 2 if merged else 1

    def part(p: torch.Tensor, e: int, u: int) -> torch.Tensor:
        if p.dim() >= 3:
            return p[e].chunk(units)[u]
        return p[u, e] if merged else p.reshape(experts)[e]
    return [[{r: part(p, e, u) for r, p in parts.items()} for u in range(units)] for e in range(experts)]


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


def moe_resolve(ids: torch.Tensor, kinds: torch.Tensor, residency: torch.Tensor, host: torch.Tensor,
                limits: torch.Tensor, staging: int, ring: torch.Tensor | None = None,
                stage: torch.Tensor | None = None) -> tuple[torch.Tensor, ...]:
    """The `moe_resolve@1` oracle: `(slots [R, T, k], fetch [R, 2, F] (−1 past n_fetch), n_fetch [R], counts [E],
    routed [E], errors)`, `counts` the increments of this call, `routed` the experts it saw routed, `errors` the bits it
    sets. `ring` defaults to none, `stage` to the staging slots k + i."""
    from .kernels import DECODE, PADDING, RANGE

    t, k = ids.shape
    experts, roles = residency.shape
    ring = torch.full((experts, roles), -1) if ring is None else ring.long().cpu()
    stage = (k + torch.arange(staging)).expand(roles, staging) if stage is None else stage.long().cpu()
    ids, kinds = ids.long().cpu(), kinds.long().cpu()
    residency, host = residency.long().cpu(), host.long().cpu()
    live = (kinds[:, None] != PADDING) & (ids >= 0)                     # [T, k]: routed entries
    routed = ids.clamp_min(0)
    slots = torch.arange(k).expand(roles, t, k).clone()                 # null slot j by default
    width = stage.shape[1]
    fetch = torch.full((roles, 2, width), -1, dtype=torch.int64)
    n_fetch = torch.zeros(roles, dtype=torch.int64)
    errors = 0
    for r in range(roles):
        missing = sorted({e for e in routed[live].tolist() if residency[e, r] < 0 and ring[e, r] < 0})
        rank = {e: i for i, e in enumerate(missing)}
        n_fetch[r] = len(missing)
        for i, e in enumerate(missing[:width]):
            fetch[r, :, i] = torch.tensor([host[e, r], stage[r, i]])
        for a, j in live.nonzero().tolist():
            e = int(routed[a, j])
            at = int(residency[e, r]) if residency[e, r] >= 0 else int(ring[e, r])
            if e in rank:
                slots[r, a, j] = stage[r, min(rank[e], width - 1)]
            elif k + staging <= at < int(limits[r]):
                slots[r, a, j] = at
            else:                                                       # corrupt: null slot j, flagged
                errors |= RANGE
    counted = routed[live & (kinds[:, None] == DECODE)]
    counts = torch.bincount(counted, minlength=experts)
    routed_ = torch.zeros(experts, dtype=torch.int32)
    routed_[routed[live]] = 1
    return slots.int(), fetch.int(), n_fetch.int(), counts.int(), routed_, errors


def expert_fetch(fetch: torch.Tensor, n: int, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """The `expert_fetch@1` oracle: `dst` with units `src[fetch[0, i]]` in slots `fetch[1, i]` for i < min(n,
    staging)."""
    out = dst.clone()
    for unit, slot in fetch[:, :n].T.tolist():
        out[slot] = src[unit].to(out.device)
    return out


def expert_prefetch(pairs, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """The `expert_prefetch@1` oracle (a `start` and its `wait`): `dst` with unit `src[u]` in slot s for each (u, s)."""
    out = dst.clone()
    for unit, slot in torch.as_tensor(pairs).tolist():
        out[slot] = src[unit].to(out.device)
    return out
