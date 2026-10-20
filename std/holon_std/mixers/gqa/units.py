from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch
from holon.api import Paged, Ref, StateSpec
from torch import nn

from . import kernels

LINEAR = "std.ops.linear@1"


@dataclasses.dataclass(eq=False)
class AttnMeta:
    """One step's attention metadata for every layer of the group: the kernel this step uses and its plan, and
    (from `prepare_device` on) where this step's keys and values go: page unit and offset per token."""

    kernel: Any
    handle: Any
    pages: Any
    write_unit: torch.Tensor | None = None
    write_off: torch.Tensor | None = None

    @property
    def graph_key(self) -> Any:
        """Hashable if this step's attention may be replayed from a captured graph, else None."""
        key = getattr(self.kernel, "graph_key", None)
        return None if key is None else key(self.handle)


class Planner:
    """Owns this group's kernel instances; a step is all prefill chunks, all decode rows, all verify rows of one
    width, or (`BatchPlan.n_decode` > 0) decode rows then prefill chunks: a mixed step (docs/m07-perf.md §3.1). Its
    decode rows go through the prefill kernel when that kernel's spec has the `serves_decode_rows` prop (it serves
    them as well as the decode kernel, e.g. by splitting their KV), else through the decode kernel and the chunks
    through the prefill kernel, as two calls."""

    mixes_row_kinds = True

    def __init__(self, prefill: Any, decode: Any, verify: Any, needs_host_lens: bool, prefill_mixes: bool) -> None:
        self.prefill, self.decode, self.verify = prefill, decode, verify
        # Decode steps plan from host lengths (e.g. FlashInfer), so a request whose length is only known on the
        # device (an uncommitted verify row) must sit them out; the core reads this (design §7.3).
        self.needs_host_lens = needs_host_lens
        self.prefill_mixes = prefill_mixes

    def plan_bounds(self, batch, pages) -> AttnMeta:
        n_query, n_logits, n = batch.plan.n_query, batch.plan.n_logits, batch.plan.n_decode
        if n and not self.prefill_mixes:
            decode, chunks = (_rows(batch, pages, a, b) for a, b in ((0, n), (n, len(n_query))))
            kernel = _Split(n, self.decode, self.prefill)
            return AttnMeta(kernel, (self.decode.plan(*decode), self.prefill.plan(*chunks)), pages)
        if n:
            kernel = self.prefill
        elif (n_query == 1).all():
            kernel = self.decode
        elif (n_logits == n_query).all() and (n_query == n_query[0]).all():
            kernel = self.verify
        else:
            kernel = self.prefill
        return AttnMeta(kernel, kernel.plan(batch, pages), pages)

    def prepare_device(self, meta: AttnMeta, ctx) -> None:
        pages = meta.pages                                       # once per step, not per layer
        meta.write_unit, meta.write_off = pages.write_index // pages.page, pages.write_index % pages.page
        prepare = getattr(meta.kernel, "prepare", None)          # in-graph part of a kernel's plan (optional)
        if prepare is not None:
            prepare(meta.handle)


@dataclasses.dataclass(frozen=True)
class _Split:
    """A mixed step's attention as two calls: its `n` decode rows (one token each, first) by the decode kernel, the
    prefill chunks after them by the prefill kernel. Eager only (no `graph_key`)."""

    n: int
    decode: Any
    prefill: Any

    def prepare(self, handle) -> None:
        for kernel, h in zip((self.decode, self.prefill), handle, strict=True):
            if (prepare := getattr(kernel, "prepare", None)) is not None:
                prepare(h)

    def run(self, q: torch.Tensor, kv, handle) -> torch.Tensor:
        return torch.cat([self.decode.run(q[:self.n], kv, handle[0]), self.prefill.run(q[self.n:], kv, handle[1])])


def _rows(batch, pages, lo: int, hi: int) -> tuple[Any, Any]:
    """Rows lo … hi-1 of a step as a batch and pages of their own (a mixed step's slices; host plan, device views):
    what attention kernels plan and run from. Host tokens, the shape key, publishes and the logits indices stay the
    step's — no attention kernel reads them."""
    plan, dev = batch.plan, batch.dev
    qs = np.concatenate([[0], np.cumsum(plan.n_query)])
    t0, t1 = int(qs[lo]), int(qs[hi])
    sub = dataclasses.replace(plan, slots=plan.slots[lo:hi], n_query=plan.n_query[lo:hi],
                              n_logits=plan.n_logits[lo:hi], start_upper=plan.start_upper[lo:hi],
                              token_src=plan.token_src[lo:hi], n_decode=0)
    sub_dev = dataclasses.replace(dev, input_ids=dev.input_ids[t0:t1], positions=dev.positions[t0:t1],
                                  query_start=dev.query_start[lo:hi + 1] - t0, slots=dev.slots[lo:hi])
    return (type(batch)(sub, sub_dev),
            dataclasses.replace(pages, lens=pages.lens[lo:hi], write_index=pages.write_index[t0:t1]))


class GQA(nn.Module):
    """Grouped-query attention with optional per-head q/k norm. Rope and norms are injected units.

    `output_gate="sigmoid"`: `q_proj` also emits a gate, interleaved per head (each head's `head_dim` queries, then its
    `head_dim` gate values), and the attention output is multiplied by `sigmoid(gate)` before `o_proj`.

    With engine state bound, keys and values live in a paged pool shared by all layers of this mixer, one pool
    segment per layer — [layers, units, 2 (k/v), kv_heads, page, head_dim] (HND, the layout every selected
    attention kernel must read) — so each layer's slice is contiguous for kernels. In a model-only build (no
    state) it attends over each whole sequence with the `attention.causal@1` kernel.

    `causal=False` (a row's new tokens see each other both ways) and `window` W > 0 (sliding-window attention) reach
    every attention kernel it selects as facts (kernels.py). `write_kv` writes a step's keys and values without
    attending (a drafter's context).

    With engine state, the q/k norms, the rotation and the KV write run as one `attention.qk_rope_kv@1` kernel on every
    step (docs/m07-perf.md D3) when the norm and rope units answer `fusion_facts()` and the KV writer's spec offers it
    (`config["fused"]`); otherwise, and in a model-only build, as the units' composition.

    **The surface a composing unit may use** (e.g. `std.mixers.qsa@1`, which holds a GQA built by Ref and attends with
    kernels of its own; stable within a major): `heads`, `kv_heads`, `head_dim`; `o_proj` (the output projection
    unit); `kv_layer()` (this layer's pages, engine builds); `project(x, ctx)` → (q [T, heads, D] normed and rotated,
    the output gate [T, heads, D] or None, k, v [T, kv_heads, D]); `project_and_write(x, ctx, kv)` → (q, gate), the
    keys and values written at the step's positions; `write_kv(x, ctx)`. Everything else is private.
    """

    @staticmethod
    def kv_layout(ctx) -> str:
        """The KV layout of the deployment's `kv.format` plugin option (default: the model dtype)."""
        fmt = ctx.options.get("plugins", {}).get("kv.format")
        if fmt not in kernels.FORMATS:
            raise ValueError(f"kv.format {fmt!r}: one of {sorted(f for f in kernels.FORMATS if f)}")
        return kernels.FORMATS[fmt]

    @classmethod
    def state_specs(cls, params: dict, layers: tuple[int, ...], ctx) -> list[StateSpec]:
        page, layout = ctx.options.get("page_size", 16), cls.kv_layout(ctx)
        per_layer = kernels.page_bytes(layout, page, params["kv_heads"], params["head_dim"], ctx.dtype.itemsize)
        return [StateSpec("kv", "paged@1", Paged(per_layer * len(layers), page), layers, layout,
                          segments=len(layers))]

    @staticmethod
    def masks(params: dict) -> dict:
        """The `causal` / `window` knobs as attention facts, given only when not default (kernels.py): every attention
        op this mixer selects gets them, so a provider that cannot mask so is never chosen — the build fails instead of
        attending unwindowed."""
        return {k: params[k] for k, default in (("causal", True), ("window", 0)) if params.get(k, default) != default}

    @classmethod
    def planner(cls, params: dict, specs: list[StateSpec], ctx) -> Planner:
        dtype, layout = str(ctx.dtype).removeprefix("torch."), specs[0].layout
        facts = {"heads": params["heads"], "kv_heads": params["kv_heads"], "head_dim": params["head_dim"],
                 "page": specs[0].shape.tokens_per_unit, "q_dtype": dtype, "kv_dtype": kernels.kv_dtype(layout, dtype),
                 "layout": layout, "max_seqs": ctx.options["max_seqs"]} | cls.masks(params)
        if ctx.options.get("planner.device_lens"):   # lengths only on the device (drafter sub-steps): verify kernel
            verify = ctx.select_kernel(kernels.VERIFY, site=specs[0].group, **facts).impl(ctx, **facts)
            return Planner(verify, verify, verify, needs_host_lens=False, prefill_mixes=False)
        ops = (kernels.PREFILL, kernels.DECODE, kernels.VERIFY)
        chosen = [ctx.select_kernel(op, site=specs[0].group, **facts) for op in ops]
        return Planner(*(spec.impl(ctx, **facts) for spec in chosen),
                       needs_host_lens="needs_host_lens" in chosen[1].props,
                       prefill_mixes="serves_decode_rows" in chosen[0].props)

    def __init__(self, ctx, *, layer: int, hidden: int, heads: int, kv_heads: int, head_dim: int, rope: Ref,
                 qk_norm: Ref | None = None, output_gate: str = "none", causal: bool = True, window: int = 0) -> None:
        super().__init__()
        if output_gate not in ("none", "sigmoid"):
            raise ValueError(f"output_gate {output_gate!r}: expected 'none' or 'sigmoid'")
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.gated = output_gate == "sigmoid"
        self.q_width = heads * head_dim * (2 if self.gated else 1)
        self.q_proj = ctx.unit(Ref(LINEAR, in_features=hidden, out_features=self.q_width))
        self.k_proj = ctx.unit(Ref(LINEAR, in_features=hidden, out_features=kv_heads * head_dim))
        self.v_proj = ctx.unit(Ref(LINEAR, in_features=hidden, out_features=kv_heads * head_dim))
        self.o_proj = ctx.unit(Ref(LINEAR, in_features=heads * head_dim, out_features=hidden))
        self.q_norm = ctx.unit(qk_norm, dim=head_dim) if qk_norm else None
        self.k_norm = ctx.unit(qk_norm, dim=head_dim) if qk_norm else None
        self.rope = ctx.unit(rope, head_dim=head_dim)
        self._qkv: nn.Module | None = None          # q, k and v in one matmul …
        self._qk: nn.Module | None = None           # … or q and k (a quantized v of another type stays apart)
        bound = ctx.state("kv", layer)
        self._kv = None if bound is None else (bound[1], bound[0].layers.index(layer), len(bound[0].layers),
                                                bound[0].shape.tokens_per_unit, ctx.dtype, bound[0].layout)
        self._causal = self._write = self._fused = None
        if bound is not None:
            facts = {"layout": bound[0].layout, "kv_heads": kv_heads, "head_dim": head_dim,
                     "page": bound[0].shape.tokens_per_unit, "dtype": str(ctx.dtype).removeprefix("torch.")}
            spec = ctx.select_kernel(kernels.KV_WRITE, site=f"std.mixers.gqa[layer={layer}]", **facts)
            self._write = spec.impl(ctx, **facts)
            self._fused = self._fusion(ctx, spec, facts, layer)
        if bound is None:
            facts = {"heads": heads, "kv_heads": kv_heads, "head_dim": head_dim,
                     "dtype": str(ctx.dtype).removeprefix("torch.")} | self.masks({"causal": causal, "window": window})
            self._causal = ctx.select_kernel(kernels.CAUSAL, site=f"std.mixers.gqa[layer={layer}]", **facts).impl(
                ctx, **facts)

    def _fusion(self, ctx, write, facts: dict, layer: int):
        """The `attention.qk_rope_kv@1` instance (D3), or None: the norm units must answer `fusion_facts()` alike, the
        rope unit too, and the KV writer's spec `write` must offer the fused kernel."""
        if self.q_norm is None or kernels.QK_ROPE_KV not in write.config.get("fused", ()):
            return None
        norm, k_norm, rope = (getattr(u, "fusion_facts", lambda: None)() for u in (self.q_norm, self.k_norm, self.rope))
        if norm is None or norm != k_norm or rope is None:
            return None
        facts = facts | {"heads": self.heads, "gated": self.gated} | norm | rope
        return ctx.select_kernel(kernels.QK_ROPE_KV, site=f"std.mixers.gqa[layer={layer}]", **facts).impl(ctx, **facts)

    def kv_layer(self):
        """This layer's pages as its layout's kernels read them (kernels.py): a tensor, or q8_0's (values, scales)."""
        pool, index, n_layers, page, dtype, layout = self._kv
        shape = (pool.capacity, 2, self.kv_heads, page, self.head_dim)
        if layout == "kv.hnd.q8_0":
            n = 2 * self.kv_heads * page * self.head_dim
            units = pool.view(torch.uint8, (n_layers, pool.capacity, pool.unit_bytes // n_layers))[index]
            return (units[:, :n].view(torch.int8).view(shape),
                    units[:, n:].view(torch.float16).view(*shape[:-1], self.head_dim // kernels.Q8_BLOCK))
        dtype = torch.float8_e4m3fn if layout == "kv.hnd.fp8" else dtype
        return pool.view(dtype, (n_layers, *shape))[index]

    def finalize_weights(self, ctx) -> None:
        merge = getattr(type(self.q_proj), "merge", None)          # one matmul for q, k, v where the linears allow
        if merge is not None:
            self._qkv = merge(self.q_proj, self.k_proj, self.v_proj)
            self._qk = None if self._qkv is not None else merge(self.q_proj, self.k_proj)

    def _projections(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """q [T, heads, D] (gated: [T, heads, 2D], each head's queries, then its gate values), k and v [T, kv_heads, D]
        of `x`, before the norms and the rotation: views of the projections' output."""
        t = x.shape[0]
        if self._qkv is not None:
            q, k, v = self._qkv(x).split(
                [self.q_width, self.kv_heads * self.head_dim, self.kv_heads * self.head_dim], dim=-1)
        elif self._qk is not None:
            h = getattr(self._qk, "prepare_input", lambda x: x)(x)      # one input, two projections (D6)
            q, k = self._qk(h).split([self.q_width, self.kv_heads * self.head_dim], dim=-1)
            v = self.v_proj(h)
        else:
            h = getattr(self.q_proj, "prepare_input", lambda x: x)(x)
            q, k, v = self.q_proj(h), self.k_proj(h), self.v_proj(h)
        return (q.reshape(t, self.heads, self.q_width // self.heads), k.reshape(t, self.kv_heads, self.head_dim),
                v.reshape(t, self.kv_heads, self.head_dim))

    def _gate(self, q: torch.Tensor) -> torch.Tensor | None:
        return q[..., self.head_dim:] if self.gated else None

    def project(self, x: torch.Tensor, ctx) -> tuple[torch.Tensor, ...]:
        """q, the output gate (or None), k and v of `x`, normed and rotated at the step's positions."""
        q, k, v = self._projections(x)
        q, gate = q[..., :self.head_dim], self._gate(q)
        if self.gated:
            q = q.contiguous()                                   # the norm and rope kernels take dense rows
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = self.rope(q, k, ctx.batch.dev.positions)
        return q, gate, k, v

    def project_and_write(self, x: torch.Tensor, ctx, kv) -> tuple[torch.Tensor, torch.Tensor | None]:
        """q normed and rotated and the output gate (or None) of `x`; its keys and values written at the step's
        positions — by the fused kernel (D3), else by the composition and `kv_write@1`."""
        meta = ctx.meta("kv")
        if self._fused is None:
            q, gate, k, v = self.project(x, ctx)
            self._write(k, v, kv, meta.write_unit, meta.write_off)
            return q, gate
        q, k, v = self._projections(x)
        return self._fused(q, k, v, ctx.batch.dev.positions, self.q_norm.weight, self.k_norm.weight, kv,
                           meta.write_unit, meta.write_off), self._gate(q)

    def write_kv(self, x: torch.Tensor, ctx) -> None:
        """Write the keys and values of `x` at the step's positions without attending: a drafter's context from
        another model's features (engine builds)."""
        self.project_and_write(x, ctx, self.kv_layer())

    def forward(self, x: torch.Tensor, ctx) -> torch.Tensor:
        t = x.shape[0]
        if self._kv is None:
            q, gate, k, v = self.project(x, ctx)
            o = self._causal(q, k, v, ctx.batch.dev.query_start)
        else:
            meta, kv = ctx.meta("kv"), self.kv_layer()
            q, gate = self.project_and_write(x, ctx, kv)
            o = meta.kernel.run(q, kv, meta.handle)
        o = o.reshape(t, self.heads, self.head_dim)
        if gate is not None:
            o = o * torch.sigmoid(gate)
        return self.o_proj(o.reshape(t, self.heads * self.head_dim))
