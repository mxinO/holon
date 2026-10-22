"""std.moe.offload@1: a MoE layer's experts through the engine's one tiered store (`std.tiered.store@1`, shared by
every MoE layer: docs/v1-offload.md D5). Expert stacks stay where the weight source maps them; the merge records each
expert's gate and up extents (gate's rows, then up's: the resident merge's layout) and down's, one piece per role in
the class of its (role, ggml type, shape). Per chunk of tokens the layer's ids resolve to arena slots — misses
gathered into staging slots — and `linear_experts_arena@1` runs over each class's arena.

Options (EngineConfig.plugin_options): `moe.placement` (a policy's unit Ref; default `placement.lfu@1`, F11, while
the RAM tier holds every expert, also the RAM-short default (F11 with deferred admission); `std.tiered.static@1` is F1),
`moe.placement.profile` (routing counts written by `holon_tools profile`, JSON `{"checkpoint": [[file, bytes], …],
"layers": {"<layer>": [count per expert]}}`, refused on another checkpoint's files, named relative to the
checkpoint's directory), `moe.tiers` (host tiers' Refs, comma-separated; default `std.tiered.pinned@1`), `moe.ram`
(the RAM tier's units per class, forwarded to it: fewer than every expert makes RAM exclusive, a slower tier holding
the rest), `moe.cache` (cache slots per class, pinned: `<class>=<n>` or `<class>=<fraction>`, `*` for every class not
named; default: fill free VRAM), `moe.ring_tokens` (D8: a step of at least this many prompt tokens streams every
layer's missing experts through lent ring slots in passes of up to the stacks' kernels' `pass_rows` tokens; 0: never;
default 512), `moe.ring_copy` (`sm`: the ring's copies a small kernel beside the pass, no host sync; `dma`: copy
engines, one host sync per streaming step; default `sm`), `moe.ring_lend` (what a streaming step lends: `coldest`,
the policy's coldest whole experts, X's; `tail`, first the experts holding the last slots of the largest class arena,
a contiguous range its expert passes take their buffers from, then the coldest: docs/v1-moe-prefill.md D10, D11;
default `coldest`), `moe.stream_tokens` (a step of at least `ring_tokens` prompt tokens streams only from this many:
below, its large passes stage their routed misses into its lent slots, as Strata's smaller prompts do; default 0),
`moe.lend_peak` (`on`: under `ring_lend = tail`, a streaming step's activations come from its lent tail, the engine
reserving only the worst smaller step's peak; default `off`), `moe.route_transform`
(`approx`: the name of a route transform, whose contract `moe_route_transform.<name>@1` its plugin provides; its
dropped routes, negative ids, are dropped end to end), `moe.cpu_plan`, `moe.cpu_isa` and `moe.cpu_min_tokens` (F5b,
the CPU's share of the misses: `cpu.py`)."""
from __future__ import annotations

import torch
from holon.api import Extent, QTensor, Ref
from torch import nn

from . import kernels
from .cpu import CpuShare, plan_table
from .units import ALIGN, CHUNK, ExpertLinear, bind_experts, pass_rows, passed, takes_buffers

GATE_UP, DOWN = 0, 1                           # roles
KEY = "std.moe.offload"                        # the store's shared-object key (`ctx.shared`, the engine's lookup)
CPU = "std.moe.cpu"                            # F5b's (`CpuShare`)
RING_TOKENS = 512                              # D8: a step of this many prompt tokens streams (§13: the 5080's)


def _store(ctx, experts: int, top_k: int):
    """The engine's tiered store for MoE experts, built once by the first layer with the engine's context."""
    opts = ctx.options.get("plugins", {})
    profile = str(opts.get("moe.placement.profile", ""))
    return ctx.unit(Ref("std.tiered.store@1", name="moe", items=experts, roles=2, top_k=top_k,
                        staging=min(experts, top_k * _rows(ctx)), prefill=min(experts, top_k * CHUNK),
                        ring_tokens=int(opts.get("moe.ring_tokens", RING_TOKENS)),
                        ring_copy=str(opts.get("moe.ring_copy", "sm")),
                        ring_lend=str(opts.get("moe.ring_lend", "coldest")),
                        stream_tokens=int(opts.get("moe.stream_tokens", 0)),
                        lend_peak=str(opts.get("moe.lend_peak", "off")) == "on",
                        policy=str(opts.get("moe.placement", "")),
                        tiers=tuple(t.strip() for t in str(opts.get("moe.tiers", "std.tiered.pinned@1")).split(",")),
                        cache=str(opts.get("moe.cache", "")), profile=profile,
                        ram=str(opts.get("moe.ram", ""))))


def _rows(ctx) -> int:
    """The most decode and verify tokens of a step."""
    return ctx.options.get("max_seqs", 1) * (1 + ctx.options.get("max_draft_tokens", 0))


class Offload(nn.Module):
    device = torch.device("cpu")                # expert stacks are held where the weight source maps them

    def __init__(self, ctx, *, layer: int, experts: int, top_k: int) -> None:
        super().__init__()
        self.store = ctx.shared(KEY, lambda root: _store(root, experts, top_k))
        # An auxiliary model's layer (an MTP head's, FN D12a) is the store's next group, numbered when its stacks are
        # added (after the target's); its passes read the step's row kinds over the step's tokens, then, in the
        # drafter's sub-steps (one decode row per drafting sequence), decode kinds.
        self._aux = ctx is not self.store.ctx
        self._seen, self._main = None, 0
        policy = self.store.policy
        if (policy.resolve_variant or policy.device_state()) and _rows(ctx) > CHUNK:
            raise NotImplementedError(f"std.moe.offload@1: {self.store.policy_ref}'s state changes once per call "
                                      f"with decode rows (D4), and {_rows(ctx)} decode and verify tokens of a step "
                                      f"would pass one expert pass ({CHUNK} tokens)")
        if (self.store.items, self.store.top_k) != (experts, top_k):
            raise ValueError(f"std.moe.offload@1: layer {layer} has {experts} experts top-{top_k}, the store "
                             f"{self.store.items} top-{self.store.top_k}")
        self.layer, self._resolver, self._transform = (None if self._aux else layer), None, None
        self._max_pass = ctx.options.get("max_step_tokens")       # a pass's most tokens (none: the kernels' pass_rows)
        opts = ctx.options.get("plugins", {})
        self._transform_name = opts.get("moe.route_transform")
        self._facts = {"experts": experts, "top_k": top_k}
        # F5b: the CPU's share, unless the split was measured on another CPU tier (S5: F5a there)
        plan, isa = plan_table(str(opts.get("moe.cpu_plan", "")), experts), str(opts.get("moe.cpu_isa", ""))
        self._cpu = ctx.shared(CPU, lambda root: CpuShare(root, self.store, plan=plan, tokens=_rows(root),
                                                          min_tokens=int(opts.get("moe.cpu_min_tokens", 1)))) \
            if plan and isa in ("", ctx.fingerprint.cpu_isa) else None
        self._stacks, self._slot, self._pending = [], None, None

    def _group(self) -> None:
        if self.layer is None:                  # an auxiliary model's layer: after every group added so far
            self.layer = max(self.store.groups, default=-1) + 1
            self.store.mark_auxiliary(self.layer)

    def fused(self, gate_up: ExpertLinear) -> ArenaExperts:
        """A fused gate/up stack (per expert gate's rows then up's: the arena's layout) as pieces of the store."""
        self._group()
        stack = ArenaExperts(self, gate_up, "swiglu", gate_up.out_features, GATE_UP, [_extents(gate_up)])
        gate_up.weight = gate_up.parts = None
        self._stacks.append(stack)
        return stack

    def merge(self, gate: ExpertLinear, up: ExpertLinear) -> ArenaExperts | None:
        """The gate/up stack as pieces of the store: per expert, gate's extent then up's (nothing is read or moved)."""
        if (gate.layout, gate.in_features, gate.out_features, gate.experts) != \
                (up.layout, up.in_features, up.out_features, up.experts):
            return None
        self._group()
        stack = ArenaExperts(self, gate, "swiglu", 2 * gate.out_features, GATE_UP,
                             [_extents(gate), _extents(up)])
        gate.weight = up.weight = gate.parts = up.parts = None
        self._stacks.append(stack)
        return stack

    def stack(self, down: ExpertLinear) -> ArenaExperts:
        self._group()
        stack = ArenaExperts(self, down, "none", down.out_features, DOWN, [_extents(down)])
        down.weight = down.parts = None
        self._stacks.append(stack)
        return stack

    def bind_kernels(self, ctx) -> None:
        self._resolver = self.store.resolver(self.layer)
        self.store.tail_bytes.append(self._tail_bytes)           # read at materialize, its stacks' kernels bound
        if self._cpu is not None:
            self._slot = self._cpu.bind(ctx, *self._stacks)
        if self._transform_name:                     # docs/v1-offload.md D3: an explicit option, never by priority
            op = f"moe_route_transform.{self._transform_name}@1"
            self._transform = ctx.select_kernel(op, site=f"std.moe.offload[{self.layer}]",
                                                **self._facts).impl(ctx, **self._facts)

    def transform(self, ids: torch.Tensor, weights: torch.Tensor, first: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The route transform's (`moe.route_transform`), with the layer's map as the resolve will read it (after the
        step's lending, the ring's pieces in their slots) and graph padding's routes dropped (−1: the transform sees
        the step's rows only); else the routes as they are."""
        if self._transform is None:
            return ids, weights
        padding = self.store.kinds[first:first + len(ids), None] == kernels.PADDING
        return self._transform(torch.where(padding, -1, ids), weights, self._resolver.present(first))

    def pass_tokens(self, tokens: int) -> int:
        """The expert pass of the step: in a streaming step (D8) as many tokens as both stacks' kernels take at
        their chunks' bits (docs/v1-moe-prefill.md D1), else std.moe@1's chunk."""
        return pass_rows(self._stacks) if self.store.large_passes(tokens) else CHUNK

    def _layout(self, tokens: int) -> list[int]:
        """D10's buffers of a pass of `tokens`, in bytes: one `scratch` for both stacks' kernels (the larger
        `scratch_bytes`), then gate/up's `out` and down's — 0 for a kernel that does not take them."""
        k = self.store.top_k
        return [max((s.kernel.scratch_bytes(tokens) for s in self._stacks if takes_buffers(s.kernel)), default=0),
                *(tokens * k * s.out_y * 4 if takes_buffers(s.kernel) else 0 for s in self._stacks)]

    def _tail_bytes(self, tokens: int | None = None) -> int:
        """The bytes a lent tail must hold for this layer's largest pass (its stacks' `pass_rows` within the step
        budget; within a step of `tokens`, given), each buffer at an ALIGN boundary of a range starting anywhere."""
        tokens = min(pass_rows(self._stacks), self._max_pass or pass_rows(self._stacks), tokens or 1 << 62)
        return sum(-(-n // ALIGN) * ALIGN for n in self._layout(tokens)) + ALIGN - 1

    def pass_buffers(self, tokens: int) -> tuple | None:
        """D10's buffers of a pass of `tokens` in the step's lent tail (D11: (gate/up's out, down's out, scratch),
        None for a kernel that takes none); None where the step lends no tail (the pass allocates them)."""
        tail = self.store.tail_view()
        if tail is None:
            return None
        at, views = -tail.data_ptr() % ALIGN, []
        for nbytes in self._layout(tokens):
            views.append(tail[at:at + nbytes] if nbytes else None)
            at += -(-nbytes // ALIGN) * ALIGN
        if at > len(tail):                       # past the bound configuration's pass (never): allocated instead
            return None
        scratch, *outs = views
        k = self.store.top_k
        h, y = (None if v is None else v.view(torch.float32).view(tokens, k, s.out_y)
                for v, s in zip(outs, self._stacks, strict=True))
        return h, y, scratch

    def route(self, ids: torch.Tensor, first: int, x: torch.Tensor) -> torch.Tensor:
        """Each role's arena slots for `ids` (the step's tokens `first`…, inputs `x`), misses fetched; with F5b, a
        small pass's CPU share is published to the CPU (its entries read null slots: zeros until `patch`)."""
        kinds = self._kinds(len(ids))
        if self._cpu is None or not self._cpu.takes(len(ids)):
            self._pending = None
            return self._resolver(ids, first, kinds=kinds)[0]
        slots, self._pending = self._resolver(ids, first, *self._cpu.args(),
                                              before_fetch=lambda cpu: self._cpu.publish(self._slot, x, cpu),
                                              kinds=kinds)
        return slots

    def route_split(self, ids: torch.Tensor, first: int, x: torch.Tensor):
        """`route` with the fetched entries apart (`moe_resolve@1`'s `split`): (hits [R, T, k], misses [R, T, k],
        join) — the fetch on a side stream that `join()` makes the current stream wait for —, or (slots, None, None)
        where the pass does not split (F5b's CPU share: `route`'s; exclusive tiers, a resolve without it)."""
        if self._cpu is not None and self._cpu.takes(len(ids)):
            return self.route(ids, first, x), None, None
        self._pending = None
        out = self._resolver(ids, first, kinds=self._kinds(len(ids)), split=True)
        return (out[0], out[2], out[3]) if len(out) == 4 else (out[0], None, None)

    def _kinds(self, tokens: int) -> torch.Tensor | None:
        """The row kinds of an auxiliary layer's pass: the step's (None) while its passes cover the step's tokens, then
        decode kinds (the drafter's sub-steps). A target layer's are the step's."""
        if not self._aux:
            return None
        if self._seen != self.store.step:
            self._seen, self._main = self.store.step, 0
        if self._main < self.store.step_tokens:
            self._main += tokens
            return None
        return self.store.decode_kinds(tokens)

    def patch(self, y: torch.Tensor) -> torch.Tensor:
        """The experts' outputs `y` [T, k, H] of the pass `route` resolved, its CPU share's rows in place."""
        if self._pending is not None:
            self._cpu.collect(self._slot, y, self._pending)
        return y

    def finish(self) -> None:
        self._resolver.done()


class ArenaExperts(nn.Module):
    """One role's stack as its slot class's arena: `linear_experts@2` over [S, out, row_bytes] with slot ids."""

    def __init__(self, owner: Offload, like: ExpertLinear, glu: str, out: int, role: int,
                 extents: list[list[Extent]]) -> None:
        super().__init__()
        self._facts, self.layout, self.in_features = like.facts, like.layout, like.in_features   # (not a child)
        self._store = owner.store
        self.glu, self.out, self.out_y = glu, out, out // 2 if glu == "swiglu" else out
        name = f"{('gate_up', 'down')[role]}.{like.layout.scheme.elem.removeprefix('ggml.')}.{out}x{like.in_features}"
        if owner._aux:                  # an auxiliary model's pieces in classes of their own (an MTP head's: its draft
            name = f"aux.{name}"        # sub-steps never compete with the target's layers for slots; floored whole)
        unit = sum(x[0].nbytes for x in extents)
        for e in range(like.experts):
            self.cls = owner.store.add(owner.layer, e, role, name, unit, [x[e] for x in extents])
        self._kernel = None

    def layout_fact(self, ctx) -> str:
        return self._facts(ctx)["layout"]

    def bind_kernels(self, ctx) -> None:
        facts = self._facts(ctx) | {"out": self.out, "glu": self.glu, "slots": self.cls.max_slots}
        self._kernel = bind_experts(ctx, facts, kernels.LINEAR_EXPERTS_ARENA, "std.moe.arena")
        self._store.watch(self._kernel.errors)
        if hasattr(self._kernel, "prepare"):
            q = QTensor(self.layout, (facts["experts"], self.out, self.in_features), {"blocks": torch.empty(0)})
            if self._kernel.prepare(q) is not q:
                raise ValueError(f"std.moe.offload@1: {facts['layout']}'s kernel repacks its weights; the offload "
                                 "store serves stacks as the checkpoint stores them")

    @property
    def kernel(self):
        """Its bound `linear_experts_arena@1` instance."""
        return self._kernel

    def forward(self, x, slots: torch.Tensor, out: torch.Tensor | None = None,
                scratch: torch.Tensor | None = None) -> torch.Tensor:
        """`x` [T, R, in], or a handle a `moe_router@1` instance prepared for this kernel (std.moe.kernels); D10's
        buffers where the kernel takes them."""
        arena = self.cls.slots()
        return self._kernel(x, {"blocks": arena.view(len(arena), self.out, -1)}, slots,
                            **passed(self._kernel, out=out, scratch=scratch))


def _extents(stack: ExpertLinear) -> list[Extent]:
    """Each expert's extent in a whole adopted stack (its rows lie in expert order in the checkpoint)."""
    if stack.layout is None or stack.parts.keys() != {"blocks"} or "blocks" not in stack.extents:
        raise ValueError(f"std.moe.offload@1: layer stacks of {stack.layout_fact()} without one file extent; the "
                         "offload store serves GGUF expert stacks (ggml block layouts)")
    whole, n = stack.extents["blocks"], stack.experts
    per = whole.nbytes // n
    return [Extent(whole.path, whole.offset + e * per, per) for e in range(n)]
