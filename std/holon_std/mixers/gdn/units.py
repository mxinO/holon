"""The Gated DeltaNet mixer (docs/m03-qwen38.md §4.2): `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a` →
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

The glue runs once per step for every layer (docs/m07-perf.md D5): the replay buffers have a layer axis (one buffer
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
    subs: list | None = None                         # KDA's prompt-step sub-batches (`KDAPlanner`), and its conv
    conv_pool: str | None = None                     # payloads' pool


class Planner:
    """The group's delta-rule plans per step. A mixed step (`BatchPlan.n_decode` n > 0, docs/m07-perf.md §3.1) is
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


def _state_views(state: tuple, conv_dtype: torch.dtype, conv_shape: tuple, ssm_shape: tuple, conv_pool=None,
                 ssm_pool=None) -> tuple[torch.Tensor, torch.Tensor]:
    """A layer's view of the conv and delta-rule pools (the specs' own, or payload pools of their layout): conv states
    [units, *conv_shape] of `conv_dtype`, delta-rule states [units, *ssm_shape] f32. `state`: (conv bound, ssm bound,
    the layer's index among the specs' layers), as GDN and KDA hold it."""
    (conv_spec, own_conv), (_, own_ssm), index = state
    conv_pool, ssm_pool, n = conv_pool or own_conv, ssm_pool or own_ssm, len(conv_spec.layers)
    conv = conv_pool.view(conv_dtype, (n, conv_pool.capacity, *conv_shape))
    return conv[index], ssm_pool.view(torch.float32, (n, ssm_pool.capacity, *ssm_shape))[index]


def _publish_conv(target: torch.Tensor, x: torch.Tensor, rows: torch.Tensor, units: torch.Tensor, kernel: int) -> None:
    """In-step conv checkpoints: the K−1 inputs before each checkpointed position (`rows`, K−1 per unit), into
    `target`'s `units`."""
    window = x.index_select(0, rows).view(len(units), kernel - 1, x.shape[1]).transpose(1, 2)
    target.index_copy_(0, units, window.to(target.dtype))


def _fresh_rows(x: torch.Tensor, ctx, conv_shape: tuple, ssm_shape: tuple):
    """A model-only build's rows, each a whole sequence from zero state: (lens, query_start, conv states, delta-rule
    states, slots, has_init)."""
    qs = ctx.batch.dev.query_start
    lens = qs.diff().tolist()
    rows = len(lens)
    conv = x.new_zeros(rows, *conv_shape)
    ssm = torch.zeros(rows, *ssm_shape, dtype=torch.float32, device=x.device)
    return (lens, qs, conv, ssm, torch.arange(rows, device=x.device),
            torch.zeros(rows, dtype=torch.bool, device=x.device))


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
        return _state_views(self._state, self.conv_weight.dtype, (self.channels, self.kernel - 1),
                            (self.hv, self.dv, self.dk), conv_pool, ssm_pool)

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
            lens, qs, conv_state, ssm_state, slots, has_init = _fresh_rows(
                x, ctx, (self.channels, self.kernel - 1), (self.hv, self.dv, self.dk))
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
                _publish_conv(self._states(conv_pool=ctx.pool(pool))[0], qkv, rows, units, self.kernel)
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


@dataclasses.dataclass(eq=False)
class SubBatch:
    """A KDA step's sub-batch (docs/v1-glm.md D4, D9): tokens [lo, hi) of the step, its row pieces (`rows`: the step
    rows they belong to; `cont`: pieces after a row's first, which continue its state), their plan, the snapshots
    falling at a piece's end (the piece's final state, copied) and the conv payloads inside it."""

    lo: int
    hi: int
    handle: Any
    rows: torch.Tensor
    cont: torch.Tensor
    query_start: torch.Tensor
    at_end: list[tuple[int, int]]
    conv: tuple[torch.Tensor, torch.Tensor] | None
    slots: torch.Tensor | None = None
    has_init: torch.Tensor | None = None


class KDAPlanner(Planner):
    """GDN's planner (mixed steps, in-step checkpoints) for the KDA rule, and with `rows` > 0 a prompt step longer than
    `rows` tokens planned as sub-batches (D9): its rows packed in order into sub-batches of at most `rows` tokens, a
    longer row split at multiples of `rows` from its start (each piece continuing the previous one's state), so the
    whole mixer runs a sub-batch at a time. Its in-step checkpoints are re-based per piece (a snapshot at a piece's end
    is that piece's final state; a conv payload belongs to the piece its position ends); only behind `rows > 0`, so
    GDN's plans are untouched (B3-m2)."""

    def __init__(self, rule: Any, conv_kernel: int, rows: int) -> None:
        super().__init__(rule, conv_kernel)
        self.rows = rows

    def plan_bounds(self, batch, pages) -> GDNMeta:
        plan = batch.plan
        lens = plan.n_query.tolist()
        if not self.rows or plan.n_decode or sum(lens) <= self.rows or bool((plan.n_logits > 1).any()):
            return super().plan_bounds(batch, pages)
        pieces = []                                             # (row, offset in the row, length)
        for r, n in enumerate(lens):
            pieces += [(r, a, min(self.rows, n - a)) for a in range(0, n, self.rows)]
        subs, cur, size = [], [], 0
        for p in pieces:
            if cur and size + p[2] > self.rows:
                subs.append(cur)
                cur, size = [], 0
            cur.append(p)
            size += p[2]
        subs.append(cur)
        pubs = {spec: [p for p in plan.publishes if p.spec == spec] for spec in (CONV, SSM)}
        device, starts, out = batch.dev.slots.device, [0, *itertools.accumulate(lens)], []
        for sub in subs:
            lo = starts[sub[0][0]] + sub[0][1]
            snaps, at_end, conv_rows, conv_units = [], [], [], []
            for i, (r, a, n) in enumerate(sub):
                for p in pubs[SSM]:
                    if p.row == r and a < p.offset < a + n:
                        snaps.append((i, p.offset - a, p.payload.units[0]))
                    elif p.row == r and p.offset == a + n:
                        at_end.append((i, p.payload.units[0]))
                for p in pubs[CONV]:
                    if p.row == r and a < p.offset <= a + n:
                        first = starts[r] + p.offset - lo - self.window
                        conv_rows += [first + j for j in range(self.window)]
                        conv_units.append(p.payload.units[0])
            n_lens = [n for _, _, n in sub]
            out.append(SubBatch(lo, lo + sum(n_lens), self.rule.plan(n_lens, snaps),
                                _on([r for r, _, _ in sub], device),
                                _on([int(a > 0) for _, a, _ in sub], device).bool(),
                                _on([0, *itertools.accumulate(n_lens)], device).int(), at_end,
                                (_on(conv_rows, device), _on(conv_units, device)) if conv_units else None))
        return GDNMeta(self.rule, None, batch.dev.slots, batch.dev.query_start, None, subs=out,
                       ssm_pool=pubs[SSM][0].payload.pool if pubs[SSM] else None,
                       conv_pool=pubs[CONV][0].payload.pool if pubs[CONV] else None)

    def prepare_device(self, meta: GDNMeta, ctx) -> None:
        super().prepare_device(meta, ctx)
        for sub in meta.subs or ():
            sub.slots = meta.slots.index_select(0, sub.rows)
            sub.has_init = meta.has_init.index_select(0, sub.rows) | sub.cont


class KDA(nn.Module):
    """GLM-5.3-Flash's Kimi delta attention mixer (docs/v1-glm.md D4; HF's `Glm5NextTextLinearAttention`): q, k and v
    projections → their causal convs (f32 weights, one conv over q ‖ k ‖ v) → `kda@1` with the per-channel forget gate
    `f_b(f_a(x))` and β = σ(b(x)) → `gated_rmsnorm@1` (sigmoid of `g_b(g_a(x))`, f32 through the weight: `round =
    "none"`) → `o_proj`. Its state is GDN's two `Slot` specs (the conv windows of q, k, v in the model dtype; the
    recurrent [H, Dv, Dk] f32), GDN's planner and in-step checkpoints. With `rows` > 0 a prompt step runs as sub-batches
    of at most `rows` tokens (D9, `KDAPlanner`): bitwise the unsplit recurrence (the rule's contract), the projections'
    bits per sub-batch. Speculative verify rows (MTP, deferred by D7) are not served. Its state group and specs are
    GDN's names (`recurrent`, `CONV` / `SSM`): one model cannot mix GDN and KDA layers (the group would be declared
    twice); none does."""

    @classmethod
    def state_specs(cls, params: dict, layers: tuple[int, ...], ctx) -> list[StateSpec]:
        width = params["heads"] * params["head_dim"]
        conv = 3 * width * (params.get("conv_kernel", 4) - 1) * ctx.dtype.itemsize
        ssm = params["heads"] * params["head_dim"] ** 2 * 4
        n = len(layers)
        return [StateSpec(CONV, "recurrent@1", Slot(conv * n), layers, "kda.conv", segments=n,
                          alignment=CHUNK, page_group=GROUP, in_step=CHUNK),
                StateSpec(SSM, "recurrent@1", Slot(ssm * n), layers, "kda.ssm", segments=n,
                          alignment=CHUNK, page_group=GROUP, in_step=CHUNK)]

    @classmethod
    def planner(cls, params: dict, specs: list[StateSpec], ctx) -> Planner:
        if ctx.options.get("max_draft_tokens"):
            raise ValueError("std.mixers.kda@1: speculative verify rows are not served (MTP is deferred, "
                             "docs/v1-glm.md D7)")
        facts = cls._facts(params, str(ctx.dtype).removeprefix("torch."))
        rule = ctx.select_kernel(kernels.KDA, site=GROUP, **facts).impl(ctx, **facts)
        return KDAPlanner(rule, params.get("conv_kernel", 4), params.get("rows", 0))

    @staticmethod
    def _facts(params: dict, dtype: str) -> dict:
        return {"heads": params["heads"], "head_k_dim": params["head_dim"], "head_v_dim": params["head_dim"],
                "lower_bound": float(params.get("lower_bound", -5.0)), "dtype": dtype}

    def __init__(self, ctx, *, layer: int, hidden: int, heads: int, head_dim: int, conv_kernel: int = 4,
                 lower_bound: float = -5.0, norm_eps: float = 1e-5, rows: int = 0) -> None:
        super().__init__()
        if rows % CHUNK:
            raise ValueError(f"std.mixers.kda@1: rows {rows} must be a multiple of {CHUNK} (the rule's chunk)")
        self.h, self.d, self.kernel, self.rows = heads, head_dim, conv_kernel, rows
        self._dtype = ctx.dtype                    # the conv state's (its inputs are the projections' outputs)
        width = heads * head_dim
        self.channels = 3 * width
        lin = lambda i, o: ctx.unit(Ref(LINEAR, in_features=i, out_features=o))       # noqa: E731
        self.q_proj, self.k_proj, self.v_proj = lin(hidden, width), lin(hidden, width), lin(hidden, width)
        self.f_a_proj, self.f_b_proj = lin(hidden, head_dim), lin(head_dim, width)
        self.b_proj = lin(hidden, heads)
        self.g_a_proj, self.g_b_proj = lin(hidden, head_dim), lin(head_dim, width)
        self.o_proj = lin(width, hidden)
        f32 = lambda *s: nn.Parameter(torch.empty(*s, dtype=torch.float32), requires_grad=False)   # noqa: E731
        self.conv_q, self.conv_k, self.conv_v = (f32(width, conv_kernel) for _ in range(3))   # as stored (F32)
        self.conv_weight: torch.Tensor | None = None                 # q ‖ k ‖ v (finalize_weights)
        self.A_log, self.dt_bias, self.norm_weight = f32(heads), f32(width), f32(head_dim)
        dtype = str(ctx.dtype).removeprefix("torch.")
        site = f"std.mixers.kda[layer={layer}]"
        conv = {"channels": self.channels, "kernel_size": conv_kernel, "dtype": dtype} | (
            {"weight_dtype": "float32"} if dtype != "float32" else {})        # (the convs stay f32; not dtype)
        norm = {"dim": head_dim, "eps": norm_eps, "dtype": dtype, "out_gate": "sigmoid", "round": "none"}
        self._conv = ctx.select_kernel(kernels.CAUSAL_CONV1D, site=site, **conv).impl(ctx, **conv)
        self._norm = ctx.select_kernel(kernels.GATED_RMSNORM, site=site, **norm).impl(ctx, **norm)
        conv_bound, ssm_bound = ctx.state(CONV, layer), ctx.state(SSM, layer)
        self._state = None if conv_bound is None else (conv_bound, ssm_bound, conv_bound[0].layers.index(layer))
        self._rule = None
        if conv_bound is None:                     # a model-only build: whole sequences from zero state
            facts = self._facts({"heads": heads, "head_dim": head_dim, "lower_bound": lower_bound}, dtype)
            self._rule = ctx.select_kernel(kernels.KDA, site=site, **facts).impl(ctx, **facts)

    def finalize_weights(self, ctx) -> None:
        """The three convs as one (q ‖ k ‖ v), as the mixed projection's channels lie."""
        if self.conv_weight is None:
            self.conv_weight = torch.cat([self.conv_q, self.conv_k, self.conv_v]).contiguous()
            self.conv_q = self.conv_k = self.conv_v = None

    def _states(self, conv_pool=None, ssm_pool=None) -> tuple[torch.Tensor, torch.Tensor]:
        return _state_views(self._state, self._dtype, (self.channels, self.kernel - 1), (self.h, self.d, self.d),
                            conv_pool, ssm_pool)

    def forward(self, x: torch.Tensor, ctx) -> torch.Tensor:
        if self._state is None:
            return self._whole_sequences(x, ctx)
        meta = ctx.meta(GROUP)
        if meta.subs:                                       # into one output: a piece's at a time (C-m5)
            out = x.new_empty(x.shape)
            for s in meta.subs:
                out[s.lo:s.hi] = self._sub_batch(x[s.lo:s.hi], s, meta, ctx)
            return out
        conv_state, ssm_state = self._states()
        qkv, f, b, gate_in = self._project(x)
        mixed = self._conv(qkv, self.conv_weight, conv_state, meta.slots, meta.query_start, meta.has_init)
        snapshot = None
        if meta.conv is not None:                           # in-step conv checkpoints: the inputs before each position
            pool, rows, units = meta.conv
            _publish_conv(self._states(conv_pool=ctx.pool(pool))[0], qkv, rows, units, self.kernel)
        if meta.ssm_pool is not None:
            snapshot = self._states(ssm_pool=ctx.pool(meta.ssm_pool))[1]
        if meta.chunks is not None:                         # a mixed step: decode rows, then the chunks
            n = meta.n_decode
            o = torch.cat([self._rule_call(meta.rule, meta.handle, mixed[:n], f[:n], b[:n], ssm_state,
                                           meta.slots[:n], meta.has_init[:n]),
                           self._rule_call(meta.rule, meta.chunks, mixed[n:], f[n:], b[n:], ssm_state,
                                           meta.slots[n:], meta.has_init[n:], snapshot)])
        else:
            o = self._rule_call(meta.rule, meta.handle, mixed, f, b, ssm_state, meta.slots, meta.has_init, snapshot)
        return self._out(o, gate_in)

    def _sub_batch(self, x: torch.Tensor, sub: SubBatch, meta: GDNMeta, ctx) -> torch.Tensor:
        """One sub-batch of a prompt step (`KDAPlanner`): its pieces continue their rows' states in the pool."""
        conv_state, ssm_state = self._states()
        qkv, f, b, gate_in = self._project(x)
        mixed = self._conv(qkv, self.conv_weight, conv_state, sub.slots, sub.query_start, sub.has_init)
        if sub.conv is not None:
            _publish_conv(self._states(conv_pool=ctx.pool(meta.conv_pool))[0], qkv, *sub.conv, self.kernel)
        snapshot = self._states(ssm_pool=ctx.pool(meta.ssm_pool))[1] if meta.ssm_pool is not None else None
        o = self._rule_call(meta.rule, sub.handle, mixed, f, b, ssm_state, sub.slots, sub.has_init, snapshot)
        for i, unit in sub.at_end:                          # a checkpoint at a piece's end: the state it left
            snapshot[unit].copy_(ssm_state[sub.slots[i]])
        return self._out(o, gate_in)

    def _project(self, x: torch.Tensor):
        qkv = torch.cat([self.q_proj(x), self.k_proj(x), self.v_proj(x)], dim=-1)
        return qkv, self.f_b_proj(self.f_a_proj(x)), self.b_proj(x), x

    def _rule_call(self, rule, handle, mixed, f, b, ssm_state, slots, has_init, snapshot=None):
        t, w = mixed.shape[0], self.h * self.d
        q, k, v = mixed.split([w, w, w], dim=-1)
        shape = (t, self.h, self.d)
        return rule(handle, q.reshape(shape), k.reshape(shape), v.reshape(shape), f, b, self.A_log, self.dt_bias,
                    ssm_state, slots, has_init, snapshot=snapshot)

    def _out(self, o: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t = o.shape[0]
        gate = self.g_b_proj(self.g_a_proj(x)).reshape(t, self.h, self.d)
        return self.o_proj(self._norm(o, gate, self.norm_weight).reshape(t, self.h * self.d))

    def _whole_sequences(self, x: torch.Tensor, ctx) -> torch.Tensor:
        """A model-only build: each sequence from zero state."""
        lens, qs, conv_state, ssm_state, slots, has_init = _fresh_rows(
            x, ctx, (self.channels, self.kernel - 1), (self.h, self.d, self.d))
        qkv, f, b, gate_in = self._project(x)
        mixed = self._conv(qkv, self.conv_weight, conv_state, slots, qs, has_init)
        o = self._rule_call(self._rule, self._rule.plan(lens), mixed, f, b, ssm_state, slots, has_init)
        return self._out(o, gate_in)
