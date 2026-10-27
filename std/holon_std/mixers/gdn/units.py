"""The Gated DeltaNet mixer (docs/dev/m03-qwen38.md §4.2): `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a` →
`causal_conv1d` → `gated_delta_rule` → `gated_rmsnorm` → `out_proj`.

Per-request state is two `Slot` specs of one planner group (`recurrent`): the conv state ([C, K-1] per layer, in the
model dtype — its inputs are model-dtype projections, so storing them is exact) and the delta-rule state ([Hv, Dv, Dk]
per layer, f32). A request's state is the unit at its request slot; checkpoints are the `recurrent` kind's business,
except that this forward fills those the core lists inside a prefill step (`BatchPlan.publishes`, on the 64-token
chunk grid both specs declare as `in_step`): the delta-rule kernel stores its f32 state at that chunk, and the conv
payload gets the K − 1 conv inputs before the position, all from the step's own rows.

Speculative verify rows (M0.4 D6) leave both states at their pre-verify values S0: the conv window and the row's conv
inputs and gates are saved, and the delta rule writes its final state to the padding slot's unit (a sink nothing
reads). Once the verifier has accepted n drafts, `rewind` reruns the delta rule from S0 over the row with a = b = −1e4
past its first 1 + n tokens — g = −exp(A)·softplus(−1e4) = 0 and β = sigmoid(−1e4) = 0 leave S unchanged there —
writing the state after the accepted tokens; its convolution is recomputed from the saved window. Both delta-rule
calls are planned `per_token`, so the result is bitwise what 1 + n one-token decode steps would leave.

The glue runs once per step for every layer (docs/dev/m07-perf.md D5): the replay buffers have a layer axis (one buffer
per name, each layer's slice at its index in the group), and the group's first layer — `rewind_modules` visits a
model's layers in order; no counter, so graph warm-up and capture behave alike — calls `gdn_rewind`, one launch that
masks every layer's saved gates and writes every layer's conv window at the accepted end into the pool, from window ‖
saved x. Each layer's convolution then takes the saved window buffer as its state (rows `arange(R)`), so it never
touches the pool.
"""
from __future__ import annotations

import dataclasses
import itertools
from typing import Any

import torch
from holon.api import Ref, Slot, StateSpec
from torch import nn

from . import kernels

LINEAR = "std.ops.linear@1"
GROUP = "recurrent"
CONV, SSM = "conv_state", "ssm_state"   # the two Slot specs (their names in BatchPlan.publishes)
CHUNK = 64                         # the delta rule's chunk: resuming at its multiples is exact (S3)


@dataclasses.dataclass(eq=False)
class GDNMeta:
    """One step's metadata for every GDN layer: the delta-rule instance and its plan, rows and their slots."""

    rule: Any
    handle: Any
    slots: torch.Tensor
    query_start: torch.Tensor
    graph_key: Any
    verify: bool = False                             # every row a verify row (planned per token; `rewind` follows)
    rewind: Any = None                               # the group's `gdn_rewind` instance (speculative deployments)
    n_decode: int = 0                                # a mixed step: its leading decode rows, planned as such …
    chunks: Any = None                               # … and the delta rule's plan of the prefill rows after them
    has_init: torch.Tensor | None = None
    ssm_pool: str | None = None                      # in-step checkpoints: the delta-rule payloads' pool (the
    conv: tuple[str, torch.Tensor, torch.Tensor] | None = None   # kernel has their rows); the conv payloads' pool,
                                                     # token indices [P·(K−1)] and units [P]


class Planner:
    """The group's delta-rule plans per step. A mixed step (`BatchPlan.n_decode` n > 0, docs/dev/m07-perf.md §3.1) is
    planned per row kind: its n decode rows as a decode step's (the recurrent kernel), the prefill chunks after them
    as a prefill step's (chunked, their in-step checkpoints' rows re-based), so each row computes as in its own
    unmixed step."""

    mixes_row_kinds = True

    def __init__(self, rule: Any, conv_kernel: int, rewind: Any = None) -> None:
        self.rule, self.window, self.rewind = rule, conv_kernel - 1, rewind

    def plan_bounds(self, batch, pages) -> GDNMeta:
        plan, n = batch.plan, batch.plan.n_decode
        lens = plan.n_query.tolist()
        decode = all(t == 1 for t in lens) and not n
        verify = bool(len(lens)) and not decode and not n and bool((plan.n_logits == plan.n_query).all())
        if not verify and bool((plan.n_logits > 1).any()):      # rewind needs every row of a step to be a verify row
            raise RuntimeError("gdn: a step mixing verify rows with other rows")
        pubs = {spec: [p for p in plan.publishes if p.spec == spec] for spec in (CONV, SSM)}
        ssm = [(p.row - n, p.offset, p.payload.units[0]) for p in pubs[SSM]]
        meta = GDNMeta(self.rule, self.rule.plan(lens[:n] if n else lens, () if n else ssm, per_token=verify),
                       batch.dev.slots, batch.dev.query_start,
                       ("gdn", len(lens)) if decode else ("gdn.verify", len(lens), lens[0]) if verify else None,
                       verify, self.rewind, n, self.rule.plan(lens[n:], ssm) if n else None)
        if ssm:
            meta.ssm_pool = pubs[SSM][0].payload.pool
        if conv := pubs[CONV]:                   # the window before each position, from the step's rows
            starts, device = [0, *itertools.accumulate(lens)], batch.dev.slots.device
            rows = [starts[p.row] + p.offset - self.window + j for p in conv for j in range(self.window)]
            meta.conv = (conv[0].payload.pool, _on(rows, device), _on([p.payload.units[0] for p in conv], device))
        return meta

    def prepare_device(self, meta: GDNMeta, ctx) -> None:
        dev = ctx.batch.dev
        meta.has_init = dev.positions[dev.query_start[:-1]] > 0            # a row that starts a request: no state


def _on(values: list[int], device: torch.device) -> torch.Tensor:
    """Host ints on the device through pinned memory: an asynchronous copy, no host sync."""
    return torch.tensor(values, dtype=torch.long, pin_memory=device.type == "cuda").to(device, non_blocking=True)


def _channels(params: dict) -> int:
    """The conv's channels: q, k and v."""
    return 2 * params["heads_k"] * params["head_k_dim"] + params["heads_v"] * params["head_v_dim"]


def _facts(params: dict, dtype: str) -> dict:
    return {"heads_k": params["heads_k"], "heads_v": params["heads_v"], "head_k_dim": params["head_k_dim"],
            "head_v_dim": params["head_v_dim"], "v_head_order": params.get("v_head_order", "grouped"),
            "dtype": dtype}


class GDN(nn.Module):
    @classmethod
    def state_specs(cls, params: dict, layers: tuple[int, ...], ctx) -> list[StateSpec]:
        conv = _channels(params) * (params.get("conv_kernel", 4) - 1) * ctx.dtype.itemsize
        ssm = params["heads_v"] * params["head_v_dim"] * params["head_k_dim"] * 4
        n = len(layers)
        return [StateSpec(CONV, "recurrent@1", Slot(conv * n), layers, "gdn.conv", segments=n,
                          alignment=CHUNK, page_group=GROUP, in_step=CHUNK),
                StateSpec(SSM, "recurrent@1", Slot(ssm * n), layers, "gdn.ssm", segments=n,
                          alignment=CHUNK, page_group=GROUP, in_step=CHUNK)]

    @classmethod
    def planner(cls, params: dict, specs: list[StateSpec], ctx) -> Planner:
        dtype = str(ctx.dtype).removeprefix("torch.")
        facts, rewind = _facts(params, dtype), None
        if ctx.options.get("max_draft_tokens"):     # verify rows write their final state elsewhere (D6)
            facts["out_slots"] = True
            glue = {"channels": _channels(params), "kernel_size": params.get("conv_kernel", 4),
                    "heads_v": params["heads_v"], "dtype": dtype}
            rewind = ctx.select_kernel(kernels.GDN_REWIND, site=GROUP, **glue).impl(ctx, **glue)
        rule = ctx.select_kernel(kernels.GATED_DELTA_RULE, site=GROUP, **facts).impl(ctx, **facts)
        return Planner(rule, params.get("conv_kernel", 4), rewind)

    def __init__(self, ctx, *, layer: int, hidden: int, heads_k: int, heads_v: int, head_k_dim: int,
                 head_v_dim: int, conv_kernel: int = 4, v_head_order: str = "grouped", norm_eps: float = 1e-6,
                 out_gate: str = "silu", merge_ab: bool = False) -> None:
        super().__init__()
        # a and b as one projection (finalize_weights): one GEMM of 2·HV rows, on CUDA (a CPU BLAS's bits for the
        # merged shape depend on the call's rows, which a prefix hit's must not)
        self._merge_ab = merge_ab and torch.device(ctx.device).type == "cuda"
        if v_head_order not in kernels.ORDERS or heads_v % heads_k or out_gate not in kernels.OUT_GATES:
            raise ValueError(f"v_head_order {v_head_order!r} / heads {heads_k}, {heads_v} / out_gate {out_gate!r}: "
                             "unsupported")
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
        norm = {"dim": head_v_dim, "eps": norm_eps, "dtype": dtype} | (
            {"out_gate": out_gate} if out_gate != "silu" else {})        # D8: other sites' facts stay as they were
        self._conv = ctx.select_kernel(kernels.CAUSAL_CONV1D, site=site, **conv).impl(ctx, **conv)
        self._norm = ctx.select_kernel(kernels.GATED_RMSNORM, site=site, **norm).impl(ctx, **norm)
        conv_bound, ssm_bound = ctx.state(CONV, layer), ctx.state(SSM, layer)
        self._state = None if conv_bound is None else (conv_bound, ssm_bound, conv_bound[0].layers.index(layer))
        self._rule = self._replay = None
        if conv_bound is not None and ctx.options.get("max_draft_tokens"):     # verify rows: what `rewind` reruns
            rows, n = ctx.options["max_seqs"], len(conv_bound[0].layers)
            tokens = rows * (1 + ctx.options["max_draft_tokens"])
            self._shared = (ctx.static("gdn.replay.x", (n, tokens, self.channels), ctx.dtype),     # every layer's
                            ctx.static("gdn.replay.a", (n, tokens, heads_v), ctx.dtype),
                            ctx.static("gdn.replay.b", (n, tokens, heads_v), ctx.dtype),
                            ctx.static("gdn.replay.window", (n, rows, self.channels, conv_kernel - 1), ctx.dtype))
            self._replay = tuple(buf[self._state[2]] for buf in self._shared)                  # this layer's
            self._rows = ctx.static("gdn.replay.rows", (rows,), torch.long)  # the replay conv's state: the window's
            if self._state[2] == 0:
                self._rows.copy_(torch.arange(rows, device=self._rows.device))
            self._sink = rows                       # the padding slot's unit: never read (graph padding rows only)
        if conv_bound is None:                     # a model-only build: whole sequences from zero state
            facts = {"heads_k": heads_k, "heads_v": heads_v, "head_k_dim": head_k_dim, "head_v_dim": head_v_dim,
                     "v_head_order": v_head_order, "dtype": dtype}
            self._rule = ctx.select_kernel(kernels.GATED_DELTA_RULE, site=site, **facts).impl(ctx, **facts)

    def _states(self, conv_pool=None, ssm_pool=None) -> tuple[torch.Tensor, torch.Tensor]:
        """This layer's view of the conv and delta-rule pools (the specs' own, or payload pools of their layout)."""
        (conv_spec, own_conv), (_, own_ssm), index = self._state
        ssm_pool, n = ssm_pool or own_ssm, len(conv_spec.layers)
        ssm = ssm_pool.view(torch.float32, (n, ssm_pool.capacity, self.hv, self.dv, self.dk))
        return self._conv_states(conv_pool or own_conv)[index], ssm[index]

    def _conv_states(self, pool) -> torch.Tensor:
        """Every layer's conv states in `pool`: [layers, units, C, K-1]."""
        n = len(self._state[0][0].layers)
        return pool.view(self.conv_weight.dtype, (n, pool.capacity, self.channels, self.kernel - 1))

    def finalize_weights(self, ctx) -> None:
        """`merge_ab`: a's and b's projections as one unit (a's rows then b's) where the linears allow — at a verify
        pass's few rows cuBLAS runs each tiny GEMM in a kernel of its own of ≈ 15 µs (FN step 7 item 8: 1.1 ms of a
        5080 verify pass), the merged one split-K in ≈ 5."""
        merge = getattr(type(self.in_proj_a), "merge", None)
        if self._merge_ab and not hasattr(self, "in_proj_ab") and merge is not None:
            merged = merge(self.in_proj_a, self.in_proj_b)
            if merged is not None:
                self.in_proj_ab = merged

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
        h = getattr(self.in_proj_qkv, "prepare_input", lambda x: x)(x)    # one input, four projections (m07 D6)
        if hasattr(self, "in_proj_ab"):                 # (column views: a rule that takes strided rows reads them)
            a, b = (y if getattr(rule, "strided", False) else y.contiguous()
                    for y in self.in_proj_ab(h).split(self.hv, dim=-1))
        else:
            a, b = self.in_proj_a(h), self.in_proj_b(h)
        qkv = self.in_proj_qkv(h)
        out_slots = None
        if self._state is not None and meta.verify:      # keep S0: save what `rewind` reruns, write the state aside
            saved_x, saved_a, saved_b, window = self._replay
            saved_x[:t].copy_(qkv)
            saved_a[:t].copy_(a)
            saved_b[:t].copy_(b)
            window[: len(slots)].copy_(conv_state.index_select(0, slots.long()))
            out_slots = torch.full_like(slots, self._sink)
        mixed = self._conv(qkv, self.conv_weight, conv_state, slots, qs, has_init)
        snapshot = None
        if self._state is not None:
            if meta.conv is not None:                       # in-step conv checkpoints: the inputs before each position
                pool, rows, units = meta.conv
                window = qkv.index_select(0, rows).view(len(units), self.kernel - 1, self.channels).transpose(1, 2)
                self._states(conv_pool=ctx.pool(pool))[0].index_copy_(0, units, window.to(conv_state.dtype))
            if meta.ssm_pool is not None:
                snapshot = self._states(ssm_pool=ctx.pool(meta.ssm_pool))[1]
        if self._state is not None and meta.chunks is not None:     # a mixed step: decode rows, then the chunks
            n = meta.n_decode
            o = torch.cat([self._rule_call(rule, handle, mixed[:n], a[:n], b[:n], ssm_state, slots[:n], has_init[:n]),
                           self._rule_call(rule, meta.chunks, mixed[n:], a[n:], b[n:], ssm_state, slots[n:],
                                           has_init[n:], snapshot)])
        else:
            o = self._rule_call(rule, handle, mixed, a, b, ssm_state, slots, has_init, snapshot, out_slots)
        o = self._norm(o, self.in_proj_z(h).reshape(t, self.hv, self.dv), self.norm_weight)
        return self.out_proj(o.reshape(t, self.hv * self.dv))

    def _rule_call(self, rule, handle, mixed, a, b, ssm_state, slots, has_init, snapshot=None, out_slots=None):
        t = mixed.shape[0]
        q, k, v = mixed.split([self.hk * self.dk, self.hk * self.dk, self.hv * self.dv], dim=-1)
        return rule(handle, q.reshape(t, self.hk, self.dk), k.reshape(t, self.hk, self.dk),
                    v.reshape(t, self.hv, self.dv), a, b, self.A_log, self.dt_bias, ssm_state, slots, has_init,
                    snapshot=snapshot, out_slots=out_slots)

    def rewind(self, ctx) -> None:
        """After a verify step: the conv window and the delta-rule state after each row's accepted tokens (D6). The
        group's first layer runs every layer's glue (D5), so a model's GDN layers rewind in order (`rewind_modules`)."""
        if self._state is None or not (meta := ctx.meta(GROUP)).verify:
            return
        saved_x, saved_a, saved_b, window = self._replay
        rows, t = len(meta.slots), int(ctx.batch.plan.n_query.sum())
        if self._state[2] == 0:
            x, a, b, windows = self._shared
            meta.rewind(self._conv_states(self._state[0][1]), windows, x, a, b, meta.slots, ctx.seq_table.n_accept,
                        t // rows)
        mixed = self._conv(saved_x[:t], self.conv_weight, window, self._rows[:rows], meta.query_start, meta.has_init)
        self._rule_call(meta.rule, meta.handle, mixed, saved_a[:t], saved_b[:t], self._states()[1], meta.slots,
                        meta.has_init, out_slots=meta.slots)
