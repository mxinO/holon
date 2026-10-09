"""Oracles of the Gated DeltaNet ops (contracts in `kernels.py`): pure torch, f32 math, token by token — the exact
recurrence HF's `torch_recurrent_gated_delta_rule` runs, in the kernels' state layout ([Hv, Dv, Dk] per unit; HF keeps
[Hk·r, Dk, Dv]). States are updated in place, as the kernels do."""
from __future__ import annotations

import torch
import torch.nn.functional as F

L2_EPS = 1e-6


def _rows(query_start: torch.Tensor) -> list[tuple[int, int]]:
    qs = query_start.tolist()
    return list(zip(qs[:-1], qs[1:], strict=True))


def causal_conv1d(x: torch.Tensor, weight: torch.Tensor, state: torch.Tensor, slots: torch.Tensor,
                  query_start: torch.Tensor, has_init: torch.Tensor) -> torch.Tensor:
    """Per row: silu(depthwise causal conv over (state ‖ row)) → y; state ← the last K-1 inputs."""
    kernel = weight.shape[1]
    y = torch.empty_like(x)
    w = weight.float()
    for r, (lo, hi) in enumerate(_rows(query_start)):
        if hi == lo:
            continue
        unit = int(slots[r])
        prev = state[unit].float() if bool(has_init[r]) else x.new_zeros(x.shape[1], kernel - 1, dtype=torch.float32)
        seq = torch.cat([prev, x[lo:hi].float().T], dim=1)                     # [C, K-1 + n]
        windows = seq.unfold(1, kernel, 1)                                     # [C, n, K]
        y[lo:hi] = F.silu((windows * w[:, None, :]).sum(-1)).T.to(x.dtype)
        state[unit] = seq[:, -(kernel - 1):].to(state.dtype)
    return y


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + L2_EPS)


def head_map(heads_k: int, heads_v: int, v_head_order: str) -> torch.Tensor:
    """The q/k head each v-head reads: `j // (Hv/Hk)` (grouped, HF) or `j % Hk` (tiled, GGUF)."""
    j = torch.arange(heads_v)
    if v_head_order == "grouped":
        return j // (heads_v // heads_k)
    if v_head_order == "tiled":
        return j % heads_k
    raise ValueError(f"unknown v_head_order {v_head_order!r}")


def gated_delta_rule(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                     a_log: torch.Tensor, dt_bias: torch.Tensor, state: torch.Tensor, slots: torch.Tensor,
                     query_start: torch.Tensor, has_init: torch.Tensor, v_head_order: str) -> torch.Tensor:
    """The gated delta rule per row, token by token in f32; state [units, Hv, Dv, Dk] f32 updated in place."""
    heads_v, dv, dk = state.shape[1:]
    idx = head_map(q.shape[1], heads_v, v_head_order).to(q.device)
    qn = l2norm(q.float())[:, idx] * dk ** -0.5                                 # [T, Hv, Dk]
    kn = l2norm(k.float())[:, idx]
    g = -a_log.float().exp() * F.softplus(a.float() + dt_bias.float())          # [T, Hv]
    beta = torch.sigmoid(b.float())
    o = torch.empty(v.shape, dtype=torch.float32, device=v.device)
    for r, (lo, hi) in enumerate(_rows(query_start)):
        if hi == lo:
            continue
        unit = int(slots[r])
        s = state[unit].clone() if bool(has_init[r]) else state.new_zeros(heads_v, dv, dk)
        for t in range(lo, hi):
            s = s * g[t].exp()[:, None, None]
            delta = beta[t][:, None] * (v[t].float() - (s * kn[t][:, None, :]).sum(-1))     # [Hv, Dv]
            s = s + delta[:, :, None] * kn[t][:, None, :]
            o[t] = (s * qn[t][:, None, :]).sum(-1)
        state[unit] = s
    return o.to(v.dtype)


def gated_rmsnorm(o: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """HF's `RMSNormGated`: the norm in f32 cast to o's dtype, times the weight, times silu(z) in f32."""
    x = o.float()
    normed = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)).to(o.dtype)
    return ((weight * normed).float() * F.silu(z.float())).to(o.dtype)
