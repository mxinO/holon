"""std.moe units: the MoE ffn (router, top-k, expert stacks, combine) and its expert stacks (docs/dev/m02-moe.md
§4.2)."""
from __future__ import annotations

import torch
from holon.api import QTensor, Ref
from torch import nn

from . import kernels

CHUNK = kernels.CHUNK              # tokens per router call; an expert pass is whole CHUNKs (`pass_tokens`)
SPLIT_ROWS = 16                    # passes of at most this many tokens (decode, verify) split an offload store's call
ALIGN = 256                        # a passed `scratch`'s alignment (`linear_experts@2`, docs/dev/v1-moe-prefill.md D10)


def _dtype(t: torch.dtype) -> str:
    return str(t).removeprefix("torch.")


def _hold(param: nn.Parameter, parts: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """`parts` on the device by role, the first (the primary part) held by `param`: what the parameter held (meta,
    or a part `prepare` replaced) is swapped out, the Parameter object kept."""
    (role, primary), *rest = parts.items()
    torch.utils.swap_tensors(param, nn.Parameter(primary.to(device), requires_grad=False))
    return {role: param} | {r: t.to(device) for r, t in rest}


def _act_quant(ctx) -> str:
    """The deployment's `quant.act` plugin option for quantized weights (default `a8`, docs/dev/m06-lowp.md D3)."""
    act = ctx.options.get("plugins", {}).get("quant.act", "a8")
    if act not in kernels.ACT_QUANT:
        raise ValueError(f"quant.act {act!r}: one of {list(kernels.ACT_QUANT)} for quantized weights")
    return act


def _one_act_mode(ctx, spec, site: str) -> None:
    """Mixed steps (`chunk_with_decode` > 0, docs/dev/m07-perf.md §3.1) run decode rows and a prefill chunk in one call,
    on
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
    materializes each part as [E, …] on the device, docs/dev/m06-lowp.md D6); either way every part is held per expert,
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
        """One stack applying SwiGLU: per expert gate's rows, then up's. Parts merge by shape (docs/dev/m06-lowp.md D5):
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
        if self.layout is not None and hasattr(self._kernel, "prepare"):          # docs/dev/m06-lowp.md D4
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

    def forward(self, x, ids: torch.Tensor, out: torch.Tensor | None = None,
                scratch: torch.Tensor | None = None) -> torch.Tensor:
        """`x` [T, R, in], or a handle a `moe_router@1` instance prepared for this kernel (std.moe.kernels); D10's
        buffers where the kernel takes them."""
        return self._kernel(x, self.parts, ids, **passed(self._kernel, out=out, scratch=scratch))


def takes_buffers(kernel) -> bool:
    """Whether a `linear_experts@2` instance takes passed buffers (it defines `scratch_bytes`,
    docs/dev/v1-moe-prefill.md
    D10)."""
    return hasattr(kernel, "scratch_bytes")


def passed(kernel, **given) -> dict:
    """The keywords of D10's buffers `given` for a call of `kernel`: those passed, if it takes them; else none."""
    return {k: v for k, v in given.items() if v is not None} if takes_buffers(kernel) else {}


def pass_scratch(stacks, tokens: int, device: torch.device) -> torch.Tensor | None:
    """D10's allocating fallback, where no tail is lent: one buffer for a pass of `tokens`, of the larger of its
    stacks' kernels' `scratch_bytes`, for both calls (gate/up, then down); None where none needs any."""
    nbytes = max((s.kernel.scratch_bytes(tokens) for s in stacks if takes_buffers(s.kernel)), default=0)
    if not nbytes:
        return None
    buf = torch.empty(nbytes + ALIGN - 1, dtype=torch.uint8, device=device)
    at = -buf.data_ptr() % ALIGN
    return buf[at:at + nbytes]


def pass_rows(stacks) -> int:
    """The rows per call every one of `stacks`' bound kernels takes at its chunks' bits (`pass_rows`, a multiple of
    CHUNK; CHUNK where a kernel does not say)."""
    rows = min(getattr(s.kernel, "pass_rows", CHUNK) for s in stacks)
    assert rows >= CHUNK and rows % CHUNK == 0, f"std.moe: pass_rows {rows} is not a whole number of {CHUNK}-row chunks"
    return rows


def bind_experts(ctx, facts: dict, op: str = kernels.LINEAR_EXPERTS, site: str = "std.moe.experts"):
    """The `op` kernel instance (`linear_experts@2`, or `linear_experts_arena@1`) for `facts`, selected at its site."""
    site = site + "[" + ",".join(f"{k}={v}" for k, v in facts.items()) + "]"
    spec = ctx.select_kernel(op, site=site, **facts)
    _one_act_mode(ctx, spec, site)
    return spec.impl(ctx, **facts)


class Resident:
    """std.moe.resident@1, the default expert store: every expert in VRAM, as the weight source stacked it (docs/
    v1-offload.md D5: the merge is the store's).

    An expert store's members, as `MoE` calls them (std.moe.offload@1 is the other store):
    - `merge(gate, up)` / `stack(down)`: the layer's gate/up and down stacks, at `finalize_weights`;
    - `transform(ids, weights, first)`: the route transform's ids and weights for tokens `first`…, per pass;
    - `route(ids, first, x)`: the ids each stack reads (an arena's slots), per pass;
    - `patch(y)`: down's outputs with any rows computed elsewhere written in (F5b's CPU share);
    - `pass_tokens(tokens)`: the tokens per expert pass of a step of `tokens` (whole CHUNKs);
    - `pass_buffers(tokens)`: D10's lent `(gate/up's out, down's out, scratch)` for a pass, or None: the pass
      allocates its own (`pass_scratch`);
    - `finish()`: after the layer's last pass of the step."""

    def __init__(self, ctx, **site) -> None:
        self.device = ctx.device                       # where expert stacks are held
        self._stacks: list[ExpertLinear] = []          # the layer's gate/up and down stacks (`pass_tokens`)

    def merge(self, gate: ExpertLinear, up: ExpertLinear) -> ExpertLinear | None:
        merged = ExpertLinear.merge(gate, up)
        self._stacks += [merged] if merged is not None else []
        return merged

    def fused(self, gate_up: ExpertLinear) -> ExpertLinear:
        gate_up.glu = "swiglu"
        self._stacks.append(gate_up)
        return gate_up

    def stack(self, down: ExpertLinear) -> ExpertLinear:
        self._stacks.append(down)
        return down

    @staticmethod
    def route(ids: torch.Tensor, first: int, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The ids each stack reads for tokens `first`… (inputs `x`): the experts' own."""
        return ids, ids

    @staticmethod
    def patch(y: torch.Tensor) -> torch.Tensor:
        """The experts' outputs, as computed."""
        return y

    @staticmethod
    def transform(ids: torch.Tensor, weights: torch.Tensor, first: int) -> tuple[torch.Tensor, torch.Tensor]:
        return ids, weights

    def pass_tokens(self, tokens: int) -> int:
        """Tokens per expert pass: as many as both stacks' kernels take at their chunks' bits
        (docs/dev/v1-moe-prefill.md
        D1), on a step of at least a chunk."""
        return pass_rows(self._stacks) if tokens >= CHUNK else CHUNK

    @staticmethod
    def pass_buffers(tokens: int) -> None:
        """No lent memory: the pass allocates its buffers (D10's fallback)."""
        return None

    @staticmethod
    def finish() -> None:
        pass


def _store(ctx, **site):
    """The expert store the `moe.store` plugin option names (a unit Ref, e.g. `std.moe.offload@1`); resident without
    it."""
    target = ctx.options.get("plugins", {}).get("moe.store")
    return Resident(ctx) if target is None else ctx.unit(target if isinstance(target, Ref) else Ref(target), **site)


class MoE(nn.Module):
    """Mixture of experts: softmax top-k routing over an f32 router (held bf16 where `moe.router_dtype` asks and its
    values allow; `moe_router@1`, or the f32 linear and
    `moe_topk@1`), gate/up/down expert stacks with SwiGLU, the weighted combine. The router runs CHUNK tokens at a
    time, or a whole pass where it declares `pass_rows` (D3′); the expert path, in f32 (design D3), a pass of the
    store's size (whole CHUNKs). `fused_gate_up`: the checkpoint stores one gate/up stack, per expert gate's rows then
    up's (transformers' `gate_up_proj`; e.g. Qwen3.8-Flash-Next's MTP pack, FN D9), loaded as `gate_up`."""

    def __init__(self, ctx, *, hidden: int, experts: int, top_k: int, inter: int, renormalize: bool,
                 layer: int = 0, fused_gate_up: bool = False) -> None:
        super().__init__()
        self.layer = layer
        self._store = _store(ctx, layer=layer, experts=experts, top_k=top_k)
        if ctx.options.get("plugins", {}).get("moe.route_transform") and isinstance(self._store, Resident):
            raise ValueError("std.moe@1: moe.route_transform drops routes, which only an offload store serves "
                             "(moe.store = std.moe.offload@1)")
        self.router = nn.Parameter(torch.empty(experts, hidden, dtype=torch.float32), requires_grad=False)
        stack = {"experts": experts, "top_k": top_k, "device": self._store.device}
        if fused_gate_up:
            self.gate_up = ExpertLinear(ctx, in_features=hidden, out_features=2 * inter, rows_per_token=1,
                                        act_dtype=ctx.dtype, **stack)
        else:
            self.gate = ExpertLinear(ctx, in_features=hidden, out_features=inter, rows_per_token=1,
                                     act_dtype=ctx.dtype, **stack)
            self.up = ExpertLinear(ctx, in_features=hidden, out_features=inter, rows_per_token=1,
                                   act_dtype=ctx.dtype, **stack)
        self.down = ExpertLinear(ctx, in_features=inter, out_features=hidden, rows_per_token=top_k,
                                 act_dtype=torch.float32, **stack)          # the f32 SwiGLU output
        # an offload store's decode and verify passes run their resident entries while the misses land (`moe.split=on`;
        # off by default: mmvq computes a null slot's entries in full, so the two passes cost twice the expert work —
        # an RTX A5000's IQ2_XS MTP decode 68 against 73 tok/s — until it skips them)
        self._split = hasattr(self._store, "route_split") and \
            str(ctx.options.get("plugins", {}).get("moe.split", "off")) == "on"
        self._pipelined = hasattr(self._store, "route_pipelined")       # (a staging step's large passes: the store's)
        self._gate_up: ExpertLinear | None = None      # the expert store's stacks (`finalize_weights`)
        self._down: ExpertLinear | None = None
        dtype = _dtype(ctx.dtype)
        topk = {"experts": experts, "top_k": top_k, "renormalize": renormalize}
        site = f"std.moe[experts={experts},top_k={top_k}]"
        # the router and top-k in one kernel where a provider serves them (bounded: its own f32 order), else the f32
        # linear and `moe_topk@1`
        router = topk | {"hidden": hidden, "dtype": dtype}
        # `moe.router_dtype=bfloat16`: the router held in bf16 (half its bytes), which the checkpoint's values must be
        # exactly (finalize_weights checks); its kernels take them as f32, the same logits
        self._router_bf16 = str(ctx.options.get("plugins", {}).get("moe.router_dtype", "float32")) == "bfloat16"
        if self._router_bf16:
            router["router_dtype"] = "bfloat16"
        self._router = (ctx.select_kernel(kernels.MOE_ROUTER, site=site, **router).impl(ctx, **router)
                        if ctx.candidates(kernels.MOE_ROUTER, router) else None)
        self._topk = None if self._router else ctx.select_kernel(kernels.MOE_TOPK, site=site, **topk).impl(ctx, **topk)
        self._prepare = getattr(self._router, "route_prepare", None)
        self._route_rows = getattr(self._router, "pass_rows", CHUNK)  # rows one router call takes at its chunks' bits
        combine = {"hidden": hidden, "top_k": top_k, "dtype": dtype, "y_dtype": "float32"}
        self._combine = ctx.select_kernel(kernels.MOE_COMBINE, site=f"std.moe[hidden={hidden},top_k={top_k}]",
                                          **combine).impl(ctx, **combine)

    def finalize_weights(self, ctx) -> None:
        if self._router_bf16:
            w = self.router.detach()
            if not torch.equal(w.bfloat16().float(), w):
                raise ValueError(f"layer {self.layer}: moe.router_dtype bfloat16, but the router's values are not "
                                 "exactly bf16 (keep float32)")
            torch.utils.swap_tensors(self.router, nn.Parameter(w.bfloat16(), requires_grad=False))
        if hasattr(self, "gate_up"):
            if not hasattr(self._store, "fused"):
                raise ValueError(f"layer {self.layer}: {type(self._store).__name__} holds no fused gate/up stacks")
            self._gate_up = self._store.fused(self.gate_up)
        else:
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
                        else self._topk(nn.functional.linear(x.float(), self.router.float())))
        return weights, ids, x[:, None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(x, None)

    joins = True                             # add_residual takes a `join` (std.ffn.sum@1's side stream)

    def add_residual(self, x: torch.Tensor, residual: torch.Tensor, join=None) -> torch.Tensor:
        """`residual + self(x)`, the add in the combine's kernel where it has one (`moe_combine@1`'s `add`): bitwise
        the composition (the ffn unit interface's optional member, std.blocks.prenorm). `join()`, given, runs just
        before `residual` is first read (a residual computed on another stream joins there, std.ffn.sum@1)."""
        if not hasattr(self._combine, "add"):
            y = self._forward(x, None)
            if join is not None:
                join()
            return residual + y
        return self._forward(x, residual, join)

    def _parts(self, xin, gate_up, down, part, landed, out_h, out_y, scratch) -> torch.Tensor:
        """A pipelined pass: the entries of part −1 (resident pieces, null slots) first, then each part's as its
        fetch lands, every call computing its own entries (the others' ids −1, their rows untouched) into one `out`
        per stack — each row as the one call over every entry gives it (a call's rows do not depend on its others)."""
        tokens, k = part.shape
        if out_h is None:
            out_h = torch.empty(tokens, k, self._gate_up.out_y, device=part.device)
        if out_y is None:
            out_y = torch.empty(tokens, k, self._down.out_y, device=part.device)
        for i in range(-1, len(landed)):
            if i >= 0:
                landed[i]()
            mine = part == i
            h = self._gate_up(xin, torch.where(mine, gate_up, -1), out_h, scratch)
            self._down(h, torch.where(mine, down, -1), out_y, scratch)
        return out_y

    def _forward(self, x: torch.Tensor, residual: torch.Tensor | None, join=None) -> torch.Tensor:
        """The router and top-k per CHUNK tokens; the expert path per pass of the store's size (whole CHUNKs, up to
        what the stacks' kernels take at their chunks' bits: the step's last partial chunk is a pass of its own, so
        every row's chunk and kernel path are a chunk's); with `residual`, its add in the combine."""
        out, tokens = torch.empty_like(x), x.shape[0]
        size, full = self._store.pass_tokens(tokens), tokens // CHUNK * CHUNK
        passes = [(a, min(a + size, full)) for a in range(0, full, size)] + [(full, tokens)] * (full < tokens)
        for a, b in passes:
            xc = x[a:b]
            # the router per CHUNK, or per pass where its rows' bits do not depend on the call (its `pass_rows`)
            step = b - a if (b - a) % CHUNK == 0 and b - a <= self._route_rows else CHUNK
            routes = [self._route(xc[c:c + step]) for c in range(0, b - a, step)]
            if len(routes) == 1:                       # the gate/up input: maybe prepared by the router's launch
                weights, ids, xin = routes[0]
            else:
                weights, ids = (torch.cat(t) for t in zip(*(r[:2] for r in routes), strict=True))
                xin = xc[:, None]
            ids, weights = self._store.transform(ids, weights, a)
            misses = landed = part = None
            if self._split and b - a <= SPLIT_ROWS and self._store.pipelines(b - a) <= 1:   # the resident entries
                # first, while the misses' fetch lands (not in a step whose staging is a ring of parts: it pipelines)
                (gate_up, down), misses, landed = self._store.route_split(ids, a, xc)
            elif self._pipelined:                     # a staging step's large pass: its misses part by part
                (gate_up, down), part, landed = self._store.route_pipelined(ids, a, xc)
            else:
                gate_up, down = self._store.route(ids, a, xc)
            # D10's buffers: the step's lent tail (D11), else one scratch for both calls, allocated per pass
            out_h, out_y, scratch = self._store.pass_buffers(b - a) or (
                None, None, pass_scratch((self._gate_up, self._down), b - a, x.device))
            if part is not None:                      # each part's entries (−1 ids: skipped) into the same buffers
                y = self._parts(xin, gate_up, down, part, landed, out_h, out_y, scratch)
            else:
                h = self._gate_up(xin, gate_up, out_h, scratch)           # [t, k, inter] f32, SwiGLU applied
                y = self._store.patch(self._down(h, down, out_y, scratch))    # (F5b: the CPU's rows in place)
            if misses is not None:                    # then the fetched ones: each entry ran in one of the two,
                landed()                              # its null slot's zero in the other
                y.add_(self._down(self._gate_up(xin, misses[0], None, scratch), misses[1], None, scratch))
            if residual is None:
                self._combine(y, weights, out[a:b])
            else:
                if join is not None:
                    join()
                    join = None
                self._combine.add(y, weights, residual[a:b], out[a:b])
        if join is not None:                          # no pass read the residual (no tokens): joined all the same
            join()
        self._store.finish()
        return out
