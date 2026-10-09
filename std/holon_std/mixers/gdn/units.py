"""The Gated DeltaNet mixer (docs/m03-qwen38.md §4.2): `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a` →
`causal_conv1d` → `gated_delta_rule` → `gated_rmsnorm` → `out_proj`.

Per-request state is two `Slot` specs of one planner group (`recurrent`): the conv state ([C, K-1] per layer, in the
model dtype — its inputs are model-dtype projections, so storing them is exact) and the delta-rule state ([Hv, Dv, Dk]
per layer, f32). A request's state is the unit at its request slot; checkpoints are the `recurrent` kind's business.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import torch
from holon.api import Ref, Slot, StateSpec
from torch import nn

from . import kernels

LINEAR = "std.ops.linear@1"
GROUP = "recurrent"
CHUNK = 64                         # the delta rule's chunk: resuming at its multiples is exact (S3)


@dataclasses.dataclass(eq=False)
class GDNMeta:
    """One step's metadata for every GDN layer: the delta-rule instance and its plan, rows and their slots."""

    rule: Any
    handle: Any
    slots: torch.Tensor
    query_start: torch.Tensor
    graph_key: Any
    has_init: torch.Tensor | None = None


class Planner:
    def __init__(self, rule: Any) -> None:
        self.rule = rule

    def plan_bounds(self, batch, pages) -> GDNMeta:
        lens = batch.plan.n_query.tolist()
        decode = all(n == 1 for n in lens)
        return GDNMeta(self.rule, self.rule.plan(lens), batch.dev.slots, batch.dev.query_start,
                       ("gdn", len(lens)) if decode else None)

    def prepare_device(self, meta: GDNMeta, ctx) -> None:
        dev = ctx.batch.dev
        meta.has_init = dev.positions[dev.query_start[:-1]] > 0            # a row that starts a request: no state


def _facts(params: dict, dtype: str) -> dict:
    return {"heads_k": params["heads_k"], "heads_v": params["heads_v"], "head_k_dim": params["head_k_dim"],
            "head_v_dim": params["head_v_dim"], "v_head_order": params.get("v_head_order", "grouped"),
            "dtype": dtype}


class GDN(nn.Module):
    @classmethod
    def state_specs(cls, params: dict, layers: tuple[int, ...], ctx) -> list[StateSpec]:
        channels = 2 * params["heads_k"] * params["head_k_dim"] + params["heads_v"] * params["head_v_dim"]
        conv = channels * (params.get("conv_kernel", 4) - 1) * ctx.dtype.itemsize
        ssm = params["heads_v"] * params["head_v_dim"] * params["head_k_dim"] * 4
        n = len(layers)
        return [StateSpec("conv_state", "recurrent@1", Slot(conv * n), layers, "gdn.conv", segments=n,
                          alignment=CHUNK, page_group=GROUP),
                StateSpec("ssm_state", "recurrent@1", Slot(ssm * n), layers, "gdn.ssm", segments=n,
                          alignment=CHUNK, page_group=GROUP)]

    @classmethod
    def planner(cls, params: dict, specs: list[StateSpec], ctx) -> Planner:
        facts = _facts(params, str(ctx.dtype).removeprefix("torch."))
        return Planner(ctx.select_kernel(kernels.GATED_DELTA_RULE, site=GROUP, **facts).impl(ctx, **facts))

    def __init__(self, ctx, *, layer: int, hidden: int, heads_k: int, heads_v: int, head_k_dim: int,
                 head_v_dim: int, conv_kernel: int = 4, v_head_order: str = "grouped", norm_eps: float = 1e-6) -> None:
        super().__init__()
        if v_head_order not in kernels.ORDERS or heads_v % heads_k:
            raise ValueError(f"v_head_order {v_head_order!r} / heads {heads_k}, {heads_v}: unsupported")
        self.hk, self.hv, self.dk, self.dv, self.kernel = heads_k, heads_v, head_k_dim, head_v_dim, conv_kernel
        self.channels = 2 * heads_k * head_k_dim + heads_v * head_v_dim
        self.in_proj_qkv = ctx.unit(Ref(LINEAR, in_features=hidden, out_features=self.channels))
        self.in_proj_z = ctx.unit(Ref(LINEAR, in_features=hidden, out_features=heads_v * head_v_dim))
        self.in_proj_b = ctx.unit(Ref(LINEAR, in_features=hidden, out_features=heads_v))
        self.in_proj_a = ctx.unit(Ref(LINEAR, in_features=hidden, out_features=heads_v))
        self.out_proj = ctx.unit(Ref(LINEAR, in_features=heads_v * head_v_dim, out_features=hidden))
        self.conv_weight = nn.Parameter(torch.empty(self.channels, conv_kernel, dtype=ctx.dtype), requires_grad=False)
        self.A_log = nn.Parameter(torch.empty(heads_v, dtype=torch.float32), requires_grad=False)
        self.dt_bias = nn.Parameter(torch.empty(heads_v, dtype=torch.float32), requires_grad=False)
        self.norm_weight = nn.Parameter(torch.empty(head_v_dim, dtype=ctx.dtype), requires_grad=False)
        dtype = str(ctx.dtype).removeprefix("torch.")
        site = f"std.mixers.gdn[layer={layer}]"
        conv = {"channels": self.channels, "kernel_size": conv_kernel, "dtype": dtype}
        norm = {"dim": head_v_dim, "eps": norm_eps, "dtype": dtype}
        self._conv = ctx.select_kernel(kernels.CAUSAL_CONV1D, site=site, **conv).impl(ctx, **conv)
        self._norm = ctx.select_kernel(kernels.GATED_RMSNORM, site=site, **norm).impl(ctx, **norm)
        conv_bound, ssm_bound = ctx.state("conv_state", layer), ctx.state("ssm_state", layer)
        self._state = None if conv_bound is None else (conv_bound, ssm_bound, conv_bound[0].layers.index(layer))
        self._rule = None
        if conv_bound is None:                     # a model-only build: whole sequences from zero state
            facts = {"heads_k": heads_k, "heads_v": heads_v, "head_k_dim": head_k_dim, "head_v_dim": head_v_dim,
                     "v_head_order": v_head_order, "dtype": dtype}
            self._rule = ctx.select_kernel(kernels.GATED_DELTA_RULE, site=site, **facts).impl(ctx, **facts)

    def _states(self) -> tuple[torch.Tensor, torch.Tensor]:
        (conv_spec, conv_pool), (_, ssm_pool), index = self._state
        n = len(conv_spec.layers)
        conv = conv_pool.view(self.conv_weight.dtype, (n, conv_pool.capacity, self.channels, self.kernel - 1))
        ssm = ssm_pool.view(torch.float32, (n, ssm_pool.capacity, self.hv, self.dv, self.dk))
        return conv[index], ssm[index]

    def forward(self, x: torch.Tensor, ctx) -> torch.Tensor:
        t = x.shape[0]
        if self._state is None:
            qs = ctx.batch.dev.query_start
            lens = qs.diff().tolist()
            rows = len(lens)
            conv_state = x.new_zeros(rows, self.channels, self.kernel - 1)
            ssm_state = torch.zeros(rows, self.hv, self.dv, self.dk, dtype=torch.float32, device=x.device)
            slots = torch.arange(rows, device=x.device)
            has_init = torch.zeros(rows, dtype=torch.bool, device=x.device)
            rule, handle = self._rule, self._rule.plan(lens)
        else:
            meta = ctx.meta(GROUP)
            conv_state, ssm_state = self._states()
            qs, slots, has_init, rule, handle = meta.query_start, meta.slots, meta.has_init, meta.rule, meta.handle
        mixed = self._conv(self.in_proj_qkv(x), self.conv_weight, conv_state, slots, qs, has_init)
        q, k, v = mixed.split([self.hk * self.dk, self.hk * self.dk, self.hv * self.dv], dim=-1)
        o = rule(handle, q.reshape(t, self.hk, self.dk), k.reshape(t, self.hk, self.dk),
                 v.reshape(t, self.hv, self.dv), self.in_proj_a(x), self.in_proj_b(x), self.A_log, self.dt_bias,
                 ssm_state, slots, has_init)
        o = self._norm(o, self.in_proj_z(x).reshape(t, self.hv, self.dv), self.norm_weight)
        return self.out_proj(o.reshape(t, self.hv * self.dv))
