"""Toys proving FN's core seams (docs/dev/v1-flash-next.md §4) before a model uses them.

- C1, a residual wider than `hidden`: `toy.wide_embed@1` repeats its inner embedding into two float32 streams
  ([T, 2H]); `toy.wide_block@1` runs its inner block on the first stream and adds the result to the second; and
  `toy.wide_norm@1` sums the streams back to [T, H] in the model's dtype for its inner norm (the final norm). The
  drafter (`drafter.toy_streams@1`: prompt lookup, the n-gram drafter) records the trunk output it is handed and taps
  layer 0, keeping a copy in its own static buffer (graph-safe).
- C2 and C3, state declared by the block: `toy.planned_block@1` is the wide block with a `pre` unit at the layers
  `pre_at`, whose state it declares and plans by forwarding the `pre` Ref's class hooks (`BuildCtx.unit_class`); the
  plain `toy.wide_block@1` forwards only its specs (no planner: the core must refuse its Slot state).
- `toy.running_sum@1`: Slot state, a request's running sum of its token ids (int64, exact) under the `recurrent`
  kind, filled in-step on its `in_step` grid; every token's residual gets (sum mod 97) / 97, so a wrong state changes
  the logits. Verify rows keep the state before the step and `rewind` sets the sum after each row's accepted tokens.
  As a block's `pre` it is spec `toy.sum`; as a layer's mixer (`name`), spec `toy.sum.<name>` in its own group.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import torch
from holon.api import Ref, Slot, StateSpec
from holon_plugins.drafters.ngram.plugin import NGramDrafter
from torch import nn

GROUP = "toy.sum"                      # the spec (and group) of a running sum; a second one adds its own `name`


class Plugin:
    def register(self, regs: Any) -> None:
        for name, unit in (("toy.wide_embed", WideEmbed), ("toy.wide_norm", WideNorm), ("toy.wide_block", WideBlock),
                           ("toy.planned_block", PlannedBlock), ("toy.running_sum", RunningSum)):
            regs.units.add(name, unit)

    def build(self, ctx: Any, target: Any) -> StreamsDrafter:
        return StreamsDrafter(ctx, target.spec.residual)


class StreamsDrafter(NGramDrafter):
    def __init__(self, ctx: Any, residual: Any) -> None:
        super().__init__(ctx, 3, 2)
        self.seen: set[tuple[str, tuple[int, ...], torch.dtype]] = set()        # (who, shape after T, dtype)
        tokens = max(ctx.options["max_step_tokens"], ctx.options["max_seqs"] * (1 + ctx.options["max_draft_tokens"]))
        self.copy = ctx.static("toy.tap", (tokens, residual.width), getattr(torch, residual.dtype))
        ctx.tap(0, self._tap)

    def _tap(self, h: torch.Tensor, ctx: Any) -> None:
        self.seen.add(("tap", tuple(h.shape[1:]), h.dtype))
        self.copy[: len(h)].copy_(h)

    def run(self, ctx: Any, h: torch.Tensor) -> None:
        self.seen.add(("run", tuple(h.shape[1:]), h.dtype))
        super().run(ctx, h)


class WideEmbed(nn.Module):
    def __init__(self, ctx: Any, *, vocab: int, hidden: int, inner: Ref) -> None:
        super().__init__()
        self.inner = ctx.unit(inner, vocab=vocab, hidden=hidden)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        e = self.inner(ids).float()
        return torch.cat([e, e], dim=-1)


class WideNorm(nn.Module):
    def __init__(self, ctx: Any, *, dim: int, inner: Ref) -> None:
        super().__init__()
        self.inner, self.dim, self.dtype = ctx.unit(inner, dim=dim), dim, ctx.dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner((x[:, : self.dim] + x[:, self.dim:]).to(self.dtype))


class WideBlock(nn.Module):
    """[T, 2H] f32: the inner block on the first stream; its update also added to the second. With `pre` at this
    layer, the `pre` unit runs first, on both streams."""

    @classmethod
    def state_specs(cls, params: dict, layers: tuple[int, ...], ctx: Any) -> list[StateSpec]:
        pre, at = params.get("pre"), tuple(i for i in layers if i in params.get("pre_at", ()))
        hooks = ctx.unit_class(pre) if pre is not None and at else None
        return hooks.state_specs(dict(pre.params), at, ctx) if hasattr(hooks, "state_specs") else []

    def __init__(self, ctx: Any, *, layer: int, hidden: int, mixer: Ref, ffn: Ref, norm: Ref, inner: Ref,
                 inner_norm: Ref, pre: Ref | None = None, pre_at: tuple[int, ...] = ()) -> None:
        super().__init__()
        self.inner = ctx.unit(inner, layer=layer, hidden=hidden, mixer=mixer, ffn=ffn, norm=inner_norm)
        self.pre = ctx.unit(pre, layer=layer, hidden=2 * hidden) if pre is not None and layer in pre_at else None
        self.hidden, self.dtype = hidden, ctx.dtype

    def forward(self, x: torch.Tensor, ctx: Any) -> torch.Tensor:
        if self.pre is not None:
            x = self.pre(x, ctx)
        a, b = x[:, : self.hidden], x[:, self.hidden:]
        new = self.inner(a.to(self.dtype), ctx).float()
        return torch.cat([new, b + (new - a)], dim=-1)


class PlannedBlock(WideBlock):
    @classmethod
    def planner(cls, params: dict, specs: list[StateSpec], ctx: Any) -> Any:
        pre = params["pre"]
        return ctx.unit_class(pre).planner(dict(pre.params), specs, ctx)


@dataclasses.dataclass(eq=False)
class SumMeta:
    graph_key: Any
    slots: torch.Tensor
    query_start: torch.Tensor
    verify: bool
    publish: tuple[str, torch.Tensor, torch.Tensor] | None = None    # in-step: payload pool, token indices, units
    has_init: torch.Tensor | None = None


class SumPlanner:
    mixes_row_kinds = True

    def __init__(self, spec: str) -> None:
        self.spec = spec

    def plan_bounds(self, batch: Any, pages: Any) -> SumMeta:
        plan = batch.plan
        lens = plan.n_query.tolist()
        decode = bool(lens) and all(t == 1 for t in lens) and not plan.n_decode
        verify = bool(lens) and not decode and not plan.n_decode and bool((plan.n_logits == plan.n_query).all())
        key = ("toy.sum", len(lens)) if decode else ("toy.sum.verify", len(lens), lens[0]) if verify else None
        meta = SumMeta(key, batch.dev.slots, batch.dev.query_start, verify)
        if pubs := [p for p in plan.publishes if p.spec == self.spec]:     # the sum after the first `offset` tokens
            starts, device = plan.n_query.cumsum() - plan.n_query, batch.dev.slots.device
            meta.publish = (pubs[0].payload.pool,
                            torch.tensor([int(starts[p.row]) + p.offset - 1 for p in pubs], device=device),
                            torch.tensor([p.payload.units[0] for p in pubs], device=device))
        return meta

    def prepare_device(self, meta: SumMeta, ctx: Any) -> None:
        dev = ctx.batch.dev
        meta.has_init = dev.positions[dev.query_start[:-1]] > 0              # a row that starts its request: 0


class RunningSum(nn.Module):
    @classmethod
    def state_specs(cls, params: dict, layers: tuple[int, ...], ctx: Any) -> list[StateSpec]:
        if len(layers) != 1:
            raise ValueError("toy.running_sum: one layer")
        return [StateSpec(_spec(params), "recurrent@1", Slot(8), layers, "toy.sum.int64",
                          alignment=params.get("alignment", 64), in_step=params.get("in_step", 64))]

    @classmethod
    def planner(cls, params: dict, specs: list[StateSpec], ctx: Any) -> SumPlanner:
        return SumPlanner(_spec(params))

    def __init__(self, ctx: Any, *, layer: int, hidden: int, name: str = "", alignment: int = 64,
                 in_step: int = 64) -> None:
        super().__init__()
        self.spec = _spec({"name": name})
        self.state = ctx.state(self.spec, layer)               # None: a model-only build (sums from zero)
        if self.state is not None:
            tokens = ctx.options["max_seqs"] * (1 + ctx.options["max_draft_tokens"])
            tokens = max(tokens, ctx.options["max_step_tokens"])
            self.saved = ctx.static(f"{self.spec}.saved", (tokens,), torch.int64)

    def _sums(self, pool: Any) -> torch.Tensor:
        return pool.view(torch.int64, (pool.capacity,))

    def forward(self, x: torch.Tensor, ctx: Any) -> torch.Tensor:
        ids, t = ctx.batch.dev.input_ids, x.shape[0]
        if self.state is None:
            qs = ctx.batch.dev.query_start
            s0 = torch.zeros(len(qs) - 1, dtype=torch.int64, device=x.device)
            meta = None
        else:
            meta = ctx.meta(self.spec)
            qs, sums = meta.query_start, self._sums(self.state[1])
            s0 = torch.where(meta.has_init, sums[meta.slots], 0)
        rows = torch.repeat_interleave(torch.arange(len(qs) - 1, device=x.device), qs.diff(), output_size=t)
        c = torch.cumsum(ids, 0)
        running = c - (c - ids)[qs[:-1]][rows] + s0[rows]          # each token's sum, itself included
        if meta is not None:
            if meta.verify:                                       # keep the sum before the step for `rewind`
                self.saved[:t].copy_(running)
            else:
                sums.index_copy_(0, meta.slots.long(), running[qs[1:] - 1])
            if meta.publish is not None:
                pool, at, units = meta.publish
                self._sums(ctx.pool(pool)).index_copy_(0, units, running[at])
        return x + (running % 97).to(x.dtype)[:, None] / 97

    def rewind(self, ctx: Any) -> None:
        if self.state is None or not (meta := ctx.meta(self.spec)).verify:
            return
        accepted = ctx.seq_table.n_accept[meta.slots]          # the row's first 1 + n tokens stand
        self._sums(self.state[1]).index_copy_(0, meta.slots.long(), self.saved[meta.query_start[:-1] + accepted])


def _spec(params: dict) -> str:
    return f"{GROUP}.{params['name']}" if params.get("name") else GROUP
