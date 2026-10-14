from __future__ import annotations

import dataclasses
from typing import Any

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
    """Owns this group's kernel instances; each step is all prefill chunks, all decode rows or all verify rows of one
    width (v0 scheduler)."""

    def __init__(self, prefill: Any, decode: Any, verify: Any, needs_host_lens: bool) -> None:
        self.prefill, self.decode, self.verify = prefill, decode, verify
        # Decode steps plan from host lengths (e.g. FlashInfer), so a request whose length is only known on the
        # device (an uncommitted verify row) must sit them out; the core reads this (design §7.3).
        self.needs_host_lens = needs_host_lens

    def plan_bounds(self, batch, pages) -> AttnMeta:
        n_query, n_logits = batch.plan.n_query, batch.plan.n_logits
        if (n_query == 1).all():
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
            return Planner(verify, verify, verify, needs_host_lens=False)
        ops = (kernels.PREFILL, kernels.DECODE, kernels.VERIFY)
        chosen = [ctx.select_kernel(op, site=specs[0].group, **facts) for op in ops]
        decode_props = chosen[1].props
        return Planner(*(spec.impl(ctx, **facts) for spec in chosen), needs_host_lens="needs_host_lens" in decode_props)

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

    def _kv_layer(self):
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
        return (q.reshape(t, self.heads, -1), k.reshape(t, self.kv_heads, self.head_dim),
                v.reshape(t, self.kv_heads, self.head_dim))

    def _gate(self, q: torch.Tensor) -> torch.Tensor | None:
        return q[..., self.head_dim:] if self.gated else None

    def _project(self, x: torch.Tensor, ctx) -> tuple[torch.Tensor, ...]:
        """q, the output gate (or None), k and v of `x`, normed and rotated at the step's positions."""
        q, k, v = self._projections(x)
        q, gate = q[..., :self.head_dim], self._gate(q)
        if self.gated:
            q = q.contiguous()                                   # the norm and rope kernels take dense rows
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = self.rope(q, k, ctx.batch.dev.positions)
        return q, gate, k, v

    def _project_and_write(self, x: torch.Tensor, ctx, kv) -> tuple[torch.Tensor, torch.Tensor | None]:
        """q normed and rotated and the output gate (or None) of `x`; its keys and values written at the step's
        positions — by the fused kernel (D3), else by the composition and `kv_write@1`."""
        meta = ctx.meta("kv")
        if self._fused is None:
            q, gate, k, v = self._project(x, ctx)
            self._write(k, v, kv, meta.write_unit, meta.write_off)
            return q, gate
        q, k, v = self._projections(x)
        return self._fused(q, k, v, ctx.batch.dev.positions, self.q_norm.weight, self.k_norm.weight, kv,
                           meta.write_unit, meta.write_off), self._gate(q)

    def write_kv(self, x: torch.Tensor, ctx) -> None:
        """Write the keys and values of `x` at the step's positions without attending: a drafter's context from
        another model's features (engine builds)."""
        self._project_and_write(x, ctx, self._kv_layer())

    def forward(self, x: torch.Tensor, ctx) -> torch.Tensor:
        t = x.shape[0]
        if self._kv is None:
            q, gate, k, v = self._project(x, ctx)
            o = self._causal(q, k, v, ctx.batch.dev.query_start)
        else:
            meta, kv = ctx.meta("kv"), self._kv_layer()
            q, gate = self._project_and_write(x, ctx, kv)
            o = meta.kernel.run(q, kv, meta.handle)
        o = o.reshape(t, self.heads, self.head_dim)
        if gate is not None:
            o = o * torch.sigmoid(gate)
        return self.o_proj(o.reshape(t, self.heads * self.head_dim))
