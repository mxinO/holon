"""std.moe.offload@1: a MoE layer's experts through the engine's one tiered store (`std.tiered.store@1`, shared by
every MoE layer: docs/v1-offload.md D5). Expert stacks stay where the weight source maps them; the merge records each
expert's gate and up extents (gate's rows, then up's: the resident merge's layout) and down's, one piece per role in
the class of its (role, ggml type, shape). Per chunk of tokens the layer's ids resolve to arena slots — misses
gathered into staging slots — and `linear_experts_arena@1` runs over each class's arena.

Options (EngineConfig.plugin_options): `moe.placement` (a policy's unit Ref; default `std.tiered.static@1`, F1),
`moe.placement.profile` (routing counts, JSON `{"layers": {"<layer>": [count per expert]}}`, named relative to the
checkpoint's directory), `moe.tiers` (host tiers' Refs,
comma-separated; default `std.tiered.pinned@1`), `moe.cache` (cache slots per class, pinned: `<class>=<n>` or
`<class>=<fraction>`, `*` for every class not named; default: fill free VRAM)."""
from __future__ import annotations

import torch
from holon.api import Extent, QTensor, Ref
from torch import nn

from . import kernels
from .units import CHUNK, ExpertLinear, bind_experts

GATE_UP, DOWN = 0, 1                           # roles


def _store(ctx, experts: int, top_k: int):
    """The engine's tiered store for MoE experts, built once by the first layer with the engine's context."""
    opts = ctx.options.get("plugins", {})
    tokens = max(ctx.options.get("max_step_tokens", 1),
                 ctx.options.get("max_seqs", 1) * (1 + ctx.options.get("max_draft_tokens", 0)))
    profile = str(opts.get("moe.placement.profile", ""))
    return ctx.unit(Ref("std.tiered.store@1", name="moe", items=experts, roles=2, top_k=top_k,
                        staging=min(experts, top_k * min(CHUNK, tokens)),
                        policy=str(opts.get("moe.placement", "std.tiered.static@1")),
                        tiers=tuple(t.strip() for t in str(opts.get("moe.tiers", "std.tiered.pinned@1")).split(",")),
                        cache=str(opts.get("moe.cache", "")), profile=profile))


class Offload(nn.Module):
    device = torch.device("cpu")                # expert stacks are held where the weight source maps them

    def __init__(self, ctx, *, layer: int, experts: int, top_k: int) -> None:
        super().__init__()
        self.store = ctx.shared("std.moe.offload", lambda root: _store(root, experts, top_k))
        if ctx is not self.store.ctx:
            raise NotImplementedError("std.moe.offload@1: an auxiliary model's experts (an MTP layer's) need its "
                                      "sub-steps' row kinds; not yet")
        if (self.store.items, self.store.top_k) != (experts, top_k):
            raise ValueError(f"std.moe.offload@1: layer {layer} has {experts} experts top-{top_k}, the store "
                             f"{self.store.items} top-{self.store.top_k}")
        self.layer, self._resolver = layer, None

    def merge(self, gate: ExpertLinear, up: ExpertLinear) -> ArenaExperts | None:
        """The gate/up stack as pieces of the store: per expert, gate's extent then up's (nothing is read or moved)."""
        if (gate.layout, gate.in_features, gate.out_features, gate.experts) != \
                (up.layout, up.in_features, up.out_features, up.experts):
            return None
        stack = ArenaExperts(self, gate, "swiglu", 2 * gate.out_features, GATE_UP,
                             [_extents(gate), _extents(up)])
        gate.weight = up.weight = gate.parts = up.parts = None
        return stack

    def stack(self, down: ExpertLinear) -> ArenaExperts:
        stack = ArenaExperts(self, down, "none", down.out_features, DOWN, [_extents(down)])
        down.weight = down.parts = None
        return stack

    def bind_kernels(self, ctx) -> None:
        self._resolver = self.store.resolver(self.layer)

    def route(self, ids: torch.Tensor, first: int) -> torch.Tensor:
        """Each role's arena slots for `ids` (the step's tokens `first`…), misses fetched."""
        return self._resolver(ids, first)


class ArenaExperts(nn.Module):
    """One role's stack as its slot class's arena: `linear_experts@2` over [S, out, row_bytes] with slot ids."""

    def __init__(self, owner: Offload, like: ExpertLinear, glu: str, out: int, role: int,
                 extents: list[list[Extent]]) -> None:
        super().__init__()
        self._facts, self.layout, self.in_features = like.facts, like.layout, like.in_features   # (not a child)
        self._store = owner.store
        self.glu, self.out = glu, out
        name = f"{('gate_up', 'down')[role]}.{like.layout.scheme.elem.removeprefix('ggml.')}.{out}x{like.in_features}"
        unit = sum(x[0].nbytes for x in extents)
        for e in range(like.experts):
            self.cls = owner.store.add(owner.layer, e, role, name, unit, [x[e] for x in extents])
        self._kernel = None

    def bind_kernels(self, ctx) -> None:
        facts = self._facts(ctx) | {"out": self.out, "glu": self.glu, "slots": self.cls.max_slots}
        self._kernel = bind_experts(ctx, facts, kernels.LINEAR_EXPERTS_ARENA, "std.moe.arena")
        self._store.watch(self._kernel.errors)
        if hasattr(self._kernel, "prepare"):
            q = QTensor(self.layout, (facts["experts"], self.out, self.in_features), {"blocks": torch.empty(0)})
            if self._kernel.prepare(q) is not q:
                raise ValueError(f"std.moe.offload@1: {facts['layout']}'s kernel repacks its weights; the offload "
                                 "store serves stacks as the checkpoint stores them")

    def forward(self, x: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        arena = self.cls.slots()
        return self._kernel(x, {"blocks": arena.view(len(arena), self.out, -1)}, slots)


def _extents(stack: ExpertLinear) -> list[Extent]:
    """Each expert's extent in a whole adopted stack (its rows lie in expert order in the checkpoint)."""
    if stack.layout is None or stack.parts.keys() != {"blocks"} or "blocks" not in stack.extents:
        raise ValueError(f"std.moe.offload@1: layer stacks of {stack.layout_fact()} without one file extent; the "
                         "offload store serves GGUF expert stacks (ggml block layouts)")
    whole, n = stack.extents["blocks"], stack.experts
    per = whole.nbytes // n
    return [Extent(whole.path, whole.offset + e * per, per) for e in range(n)]
