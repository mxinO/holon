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
    """Per row: silu(depthwise causal conv over (state ‖ row)) → y; state ← the last K-1 inputs. Each output is computed
    for its token alone, so a row of n tokens is bitwise n one-token calls (CPU math varies with tensor length)."""
    kernel = weight.shape[1]
    y = torch.empty_like(x)
    w = weight.float()
    for r, (lo, hi) in enumerate(_rows(query_start)):
        if hi == lo:
            continue
        unit = int(slots[r])
        prev = state[unit].float() if bool(has_init[r]) else x.new_zeros(x.shape[1], kernel - 1, dtype=torch.float32)
        seq = torch.cat([prev, x[lo:hi].float().T], dim=1)                     # [C, K-1 + n]
        for t in range(hi - lo):
            y[lo + t] = F.silu((seq[:, t: t + kernel] * w).sum(-1)).to(x.dtype)
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
                     query_start: torch.Tensor, has_init: torch.Tensor, v_head_order: str,
                     snapshot: torch.Tensor | None = None, snaps: dict[int, tuple[int, int]] | None = None,
                     out_slots: torch.Tensor | None = None) -> torch.Tensor:
    """The gated delta rule per row, token by token in f32; state [units, Hv, Dv, Dk] f32 updated in place (into
    `out_slots[r]` instead of `slots[r]` when given); `snaps` {row: (offset, unit)}: the row's state after its first
    `offset` tokens into `snapshot[unit]`. Every per-token quantity is computed for its token alone, so a row of n
    tokens is bitwise n one-token calls (CPU math varies with tensor length)."""
    heads_v, dv, dk = state.shape[1:]
    idx = head_map(q.shape[1], heads_v, v_head_order).to(q.device)
    o = torch.empty(v.shape, dtype=torch.float32, device=v.device)
    for r, (lo, hi) in enumerate(_rows(query_start)):
        if hi == lo:
            continue
        unit = int(slots[r])
        s = state[unit].clone() if bool(has_init[r]) else state.new_zeros(heads_v, dv, dk)
        for t in range(lo, hi):
            qn = l2norm(q[t: t + 1].float())[0, idx] * dk ** -0.5                  # [Hv, Dk]
            kn = l2norm(k[t: t + 1].float())[0, idx]
            g = -a_log.float().exp() * F.softplus(a[t].float() + dt_bias.float())  # [Hv]
            beta = torch.sigmoid(b[t].float())
            s = s * g.exp()[:, None, None]
            delta = beta[:, None] * (v[t].float() - (s * kn[:, None, :]).sum(-1))      # [Hv, Dv]
            s = s + delta[:, :, None] * kn[:, None, :]
            o[t] = (s * qn[:, None, :]).sum(-1)
            if snaps and r in snaps and t + 1 - lo == snaps[r][0]:
                snapshot[snaps[r][1]] = s
        state[unit if out_slots is None else int(out_slots[r])] = s
    return o.to(v.dtype)


def gated_rmsnorm(o: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """HF's `RMSNormGated`: the norm in f32 cast to o's dtype, times the weight, times silu(z) in f32."""
    x = o.float()
    normed = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)).to(o.dtype)
    return ((weight * normed).float() * F.silu(z.float())).to(o.dtype)
