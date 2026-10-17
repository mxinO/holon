"""std.moe units: the MoE ffn (router, top-k, expert stacks, combine) and its expert stacks (docs/m02-moe.md §4.2)."""
from __future__ import annotations

import torch
from holon.api import QTensor, Ref
from torch import nn

from . import kernels

CHUNK = 512                        # tokens per pass through the expert path: bounds its f32 activations


def _dtype(t: torch.dtype) -> str:
    return str(t).removeprefix("torch.")


def _hold(param: nn.Parameter, parts: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """`parts` on the device by role, the first (the primary part) held by `param`: what the parameter held (meta,
    or a part `prepare` replaced) is swapped out, the Parameter object kept."""
    (role, primary), *rest = parts.items()
    torch.utils.swap_tensors(param, nn.Parameter(primary.to(device), requires_grad=False))
    return {role: param} | {r: t.to(device) for r, t in rest}


def _act_quant(ctx) -> str:
    """The deployment's `quant.act` plugin option for quantized weights (default `a8`, docs/m06-lowp.md D3)."""
    act = ctx.options.get("plugins", {}).get("quant.act", "a8")
    if act not in kernels.ACT_QUANT:
        raise ValueError(f"quant.act {act!r}: one of {list(kernels.ACT_QUANT)} for quantized weights")
    return act


def _one_act_mode(ctx, spec, site: str) -> None:
    """Mixed steps (`chunk_with_decode` > 0, docs/m07-perf.md §3.1) run decode rows and a prefill chunk in one call, on
    the path the call's row count picks: a kernel whose paths compute in different activation modes (its spec's
    `config["act"]`, path -> mode) would then run a decode row at a precision its unmixed step does not use. The
    option is the engine's effective value: 0 with a drafter, whose steps alternate and never mix."""
    modes = spec.config.get("act", {})
    if ctx.options.get("chunk_with_decode", 0) > 0 and len(set(modes.values())) > 1:
        raise ValueError(f"{site}: chunk_with_decode > 0 needs one activation mode on every path; {spec.provider}'s "
                         f"kernel computes {modes}")


class ExpertLinear(nn.Module):
    """A stack of E experts' [out, in] weights, applied per (token, slot) by `linear_experts@2`.

    Dense (an HF checkpoint) it is the parameter itself, layout `dense/<dtype>`. Quantized, `adopt_weights` takes the
    whole stack (a GGUF file's, whose parts hold the experts' rows in order) or one expert's slice at a time (the first
    materializes each part as [E, …] on the device, docs/m06-lowp.md D6); either way every part is held per expert,
    [E, …], the first (the primary part) by `weight`. `merge(gate, up)` builds one gate/up stack per expert that
    applies SwiGLU (`glu="swiglu"`); the kernel is bound after merges, and its `prepare`, if any, replaces the parts
    and their layout. `act_dtype`: its input's dtype (the model dtype for gate/up, f32 for down, which takes the
    SwiGLU output)."""

    def __init__(self, ctx, *, experts: int, in_features: int, out_features: int, top_k: int,
                 rows_per_token: int, act_dtype: torch.dtype, device: torch.device | None = None) -> None:
        super().__init__()
        self.experts, self.in_features, self.out_features = experts, in_features, out_features
        self.top_k, self.rows_per_token, self.glu, self._ctx = top_k, rows_per_token, "none", ctx
        self.act_dtype, self.device = act_dtype, device or ctx.device     # where its parts are held
        self.extents = {}                          # a whole adopted stack's, per part (QTensor.extents)
        self.weight = nn.Parameter(torch.empty(experts, out_features, in_features, dtype=ctx.dtype),
                                   requires_grad=False)
        self.parts: dict[str, torch.Tensor] | None = {"weight": self.weight}     # by role; dense: the parameter
        self.layout = None                         # the parts' Layout (adopted, then prepared); None: dense
        self._kernel = None

    def layout_fact(self) -> str:
        if self.layout is None:
            return f"dense/{_dtype(self.weight.dtype)}"
        return f"{self.layout.scheme.elem}/{self.layout.name}"

    def adopt_weights(self, ctx, name: str, q, index: int | None = None) -> bool:
        stack = (self.experts, self.out_features, self.in_features)
        if name != "weight" or tuple(q.logical_shape) != (stack if index is None else stack[1:]):
            return False
        if index is None:
            self.layout, self.extents = q.layout, dict(q.extents)
            self.parts = _hold(self.weight, {r: t.unflatten(0, (self.experts, -1)) for r, t in q.parts.items()},
                               self.device)
            return True
        if self.weight.is_meta:                              # the first slice: each part as [E, *its shape]
            self.layout = q.layout
            self.parts = _hold(self.weight, {r: torch.empty(self.experts, *t.shape, dtype=t.dtype, device=self.device)
                                             for r, t in q.parts.items()}, self.device)
        elif q.layout != self.layout or {r: (t.dtype, t.shape) for r, t in q.parts.items()} != \
                {r: (t.dtype, t.shape[1:]) for r, t in self.parts.items()}:
            return False
        for role, t in q.parts.items():
            self.parts[role][index].copy_(t)
        return True

    @staticmethod
    def merge(gate: ExpertLinear, up: ExpertLinear) -> ExpertLinear | None:
        """One stack applying SwiGLU: per expert gate's rows, then up's. Parts merge by shape (docs/m06-lowp.md D5):
        one with a value per expert ([E] or [E, 1]) is stacked [2, E], gate's row then up's; one with `out / g` rows
        per expert ([E, out / g, …]) is concatenated per expert (a stored row is a whole block run, or its rows'
        scales). None if the layouts, shapes or parts differ; gate and up give up their weights."""
        if (gate.layout, gate.in_features, gate.out_features, gate.experts, gate.act_dtype) != \
                (up.layout, up.in_features, up.out_features, up.experts, up.act_dtype) \
                or gate.weight.dtype != up.weight.dtype or gate.parts.keys() != up.parts.keys():
            return None
        parts = {}
        for role, g in gate.parts.items():
            if g.dim() <= 2 and g.numel() == gate.experts:      # one value per expert, [E] or [E, 1]
                parts[role] = torch.stack([g.reshape(-1), up.parts[role].reshape(-1)])
            elif g.dim() > 1 and g.shape[1] and gate.out_features % g.shape[1] == 0:
                parts[role] = torch.cat([g, up.parts[role]], dim=1)
            else:
                return None
        with torch.device("meta"):
            merged = ExpertLinear(gate._ctx, experts=gate.experts, in_features=gate.in_features,
                                  out_features=2 * gate.out_features, top_k=gate.top_k,
                                  rows_per_token=gate.rows_per_token, act_dtype=gate.act_dtype,
                                  device=gate.device)
        merged.layout, merged.glu = gate.layout, "swiglu"
        merged.parts = _hold(merged.weight, parts, gate.device)
        gate.weight = up.weight = gate.parts = up.parts = None
        return merged

    def bind_kernels(self, ctx) -> None:
        if self.weight is None:                                   # merged away
            return
        # once: `prepare` replaces the layout the facts are read from (a second bind would select for its output)
        assert self._kernel is None, "ExpertLinear.bind_kernels runs once"
        self._kernel = bind_experts(ctx, self.facts(ctx))
        if self.layout is not None and hasattr(self._kernel, "prepare"):          # docs/m06-lowp.md D4
            q = QTensor(self.layout, (self.experts, self.out_features, self.in_features), self.parts)
            if (prepared := self._kernel.prepare(q)) is not q:
                self.layout, self.parts = prepared.layout, _hold(self.weight, prepared.parts, ctx.device)

    def facts(self, ctx) -> dict:
        """Its `linear_experts@2` facts."""
        return {"layout": self.layout_fact(), "experts": self.experts, "in": self.in_features,
                "out": self.out_features, "top_k": self.top_k, "rows_per_token": self.rows_per_token,
                "glu": self.glu, "act_dtype": _dtype(self.act_dtype), "out_dtype": "float32",
                "act_quant": "a16" if self.layout is None else _act_quant(ctx)}

    @property
    def kernel(self):
        """Its bound `linear_experts@2` instance."""
        return self._kernel

    def forward(self, x, ids: torch.Tensor) -> torch.Tensor:
        """`x` [T, R, in], or a handle a `moe_router@1` instance prepared for this kernel (std.moe.kernels)."""
        return self._kernel(x, self.parts, ids)


def bind_experts(ctx, facts: dict, op: str = kernels.LINEAR_EXPERTS, site: str = "std.moe.experts"):
    """The `op` kernel instance (`linear_experts@2`, or `linear_experts_arena@1`) for `facts`, selected at its site."""
    site = site + "[" + ",".join(f"{k}={v}" for k, v in facts.items()) + "]"
    spec = ctx.select_kernel(op, site=site, **facts)
    _one_act_mode(ctx, spec, site)
    return spec.impl(ctx, **facts)


class Resident:
    """std.moe.resident@1, the default expert store: every expert in VRAM, as the weight source stacked it (docs/
    v1-offload.md D5: the merge is the store's)."""

    def __init__(self, ctx, **site) -> None:
        self.device = ctx.device                       # where expert stacks are held

    @staticmethod
    def merge(gate: ExpertLinear, up: ExpertLinear) -> ExpertLinear | None:
        return ExpertLinear.merge(gate, up)

    @staticmethod
    def stack(down: ExpertLinear) -> ExpertLinear:
        return down

    @staticmethod
    def route(ids: torch.Tensor, first: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The ids each stack reads for tokens `first`…: the experts' own."""
        return ids, ids

    @staticmethod
    def transform(ids: torch.Tensor, weights: torch.Tensor, first: int) -> tuple[torch.Tensor, torch.Tensor]:
        return ids, weights

    @staticmethod
    def pass_tokens(tokens: int) -> int:
        """Tokens per expert pass: one chunk."""
        return CHUNK

    @staticmethod
    def finish() -> None:
        pass


def _store(ctx, **site):
    """The expert store the `moe.store` plugin option names (a unit Ref, e.g. `std.moe.offload@1`); resident without
    it."""
    target = ctx.options.get("plugins", {}).get("moe.store")
    return Resident(ctx) if target is None else ctx.unit(target if isinstance(target, Ref) else Ref(target), **site)


class MoE(nn.Module):
    """Mixture of experts: softmax top-k routing over an f32 router (`moe_router@1`, or the f32 linear and
    `moe_topk@1`), gate/up/down expert stacks with SwiGLU, the weighted combine. The expert path runs in f32 (design
    D3), CHUNK tokens at a time."""

    def __init__(self, ctx, *, hidden: int, experts: int, top_k: int, inter: int, renormalize: bool,
                 layer: int = 0) -> None:
        super().__init__()
        self.layer = layer
        self._store = _store(ctx, layer=layer, experts=experts, top_k=top_k)
        if ctx.options.get("plugins", {}).get("moe.route_transform") and isinstance(self._store, Resident):
            raise ValueError("std.moe@1: moe.route_transform drops routes, which only an offload store serves "
                             "(moe.store = std.moe.offload@1)")
        self.router = nn.Parameter(torch.empty(experts, hidden, dtype=torch.float32), requires_grad=False)
        stack = {"experts": experts, "top_k": top_k, "device": self._store.device}
        self.gate = ExpertLinear(ctx, in_features=hidden, out_features=inter, rows_per_token=1, act_dtype=ctx.dtype,
                                 **stack)
        self.up = ExpertLinear(ctx, in_features=hidden, out_features=inter, rows_per_token=1, act_dtype=ctx.dtype,
                               **stack)
        self.down = ExpertLinear(ctx, in_features=inter, out_features=hidden, rows_per_token=top_k,
                                 act_dtype=torch.float32, **stack)          # the f32 SwiGLU output
        self._gate_up: ExpertLinear | None = None      # the expert store's stacks (`finalize_weights`)
        self._down: ExpertLinear | None = None
        dtype = _dtype(ctx.dtype)
        topk = {"experts": experts, "top_k": top_k, "renormalize": renormalize}
        site = f"std.moe[experts={experts},top_k={top_k}]"
        # the router and top-k in one kernel where a provider serves them (bounded: its own f32 order), else the f32
        # linear and `moe_topk@1`
        router = topk | {"hidden": hidden, "dtype": dtype}
        self._router = (ctx.select_kernel(kernels.MOE_ROUTER, site=site, **router).impl(ctx, **router)
                        if ctx.candidates(kernels.MOE_ROUTER, router) else None)
        self._topk = None if self._router else ctx.select_kernel(kernels.MOE_TOPK, site=site, **topk).impl(ctx, **topk)
        self._prepare = getattr(self._router, "route_prepare", None)
        combine = {"hidden": hidden, "top_k": top_k, "dtype": dtype, "y_dtype": "float32"}
        self._combine = ctx.select_kernel(kernels.MOE_COMBINE, site=f"std.moe[hidden={hidden},top_k={top_k}]",
                                          **combine).impl(ctx, **combine)

    def finalize_weights(self, ctx) -> None:
        self._gate_up = self._store.merge(self.gate, self.up)
        if self._gate_up is None:
            raise ValueError(f"layer {self.layer}: gate and up experts must share a layout to merge "
                             f"(gate {self.gate.layout_fact()}, up {self.up.layout_fact()})")
        self._down = self._store.stack(self.down)

    def _route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(weights, ids, the gate/up input) for up to CHUNK tokens: the router's route_prepare where it has one, else
        its routing (or the f32 linear and top-k) and `x[:, None]`."""
        if self._prepare is not None:
            return self._prepare(x, self.router, self._gate_up.kernel)
        weights, ids = (self._router(x, self.router) if self._router is not None
                        else self._topk(nn.functional.linear(x.float(), self.router)))
        return weights, ids, x[:, None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(x, None)

    def add_residual(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """`residual + self(x)`, the add in the combine's kernel where it has one (`moe_combine@1`'s `add`): bitwise
        the composition (the ffn unit interface's optional member, std.blocks.prenorm)."""
        if not hasattr(self._combine, "add"):
            return residual + self._forward(x, None)
        return self._forward(x, residual)

    def _forward(self, x: torch.Tensor, residual: torch.Tensor | None) -> torch.Tensor:
        """The router and top-k per CHUNK tokens; the expert path per pass of the store's size (a CHUNK, or whole
        CHUNKs in a streaming offload step: the step's last partial chunk is a pass of its own, so every row's chunk
        and kernel path are a chunk's); with `residual`, its add in the combine."""
        out, tokens = torch.empty_like(x), x.shape[0]
        size, full = self._store.pass_tokens(tokens), tokens // CHUNK * CHUNK
        passes = [(a, min(a + size, full)) for a in range(0, full, size)] + [(full, tokens)] * (full < tokens)
        for a, b in passes:
            xc = x[a:b]
            routes = [self._route(xc[c:c + CHUNK]) for c in range(0, b - a, CHUNK)]
            if len(routes) == 1:                       # the gate/up input: maybe prepared by the router's launch
                weights, ids, xin = routes[0]
            else:
                weights, ids = (torch.cat(t) for t in zip(*(r[:2] for r in routes), strict=True))
                xin = xc[:, None]
            ids, weights = self._store.transform(ids, weights, a)
            gate_up, down = self._store.route(ids, a)
            h = self._gate_up(xin, gate_up)                           # [t, k, inter] f32, SwiGLU applied
            if residual is None:
                self._combine(self._down(h, down), weights, out[a:b])
            else:
                self._combine.add(self._down(h, down), weights, residual[a:b], out[a:b])
        self._store.finish()
        return out
