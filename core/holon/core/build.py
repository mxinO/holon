"""Build a model from its ModelSpec and load weights through its WeightMap (design §5.1).

A `Ref("std.ops.rmsnorm@1")` is served by the plugin whose manifest provides `unit.std.ops.rmsnorm@1`.
"""
from __future__ import annotations

import dataclasses
import functools
import re
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from torch import nn

from holon.api import (
    Batch,
    BatchPlan,
    DeviceBatch,
    ModelSpec,
    Ref,
    SeqTable,
    ShapeKey,
    StepCtx,
    TokenSrc,
    WeightMap,
)

from .plugins import PluginError, PluginSet


class BuildContext:
    """Instantiates units by Ref; the only way models and units obtain other units (invariant 2)."""

    def __init__(self, plugins: PluginSet, device: torch.device, dtype: torch.dtype = torch.bfloat16) -> None:
        self.plugins = plugins
        self.device = device
        self.dtype = dtype                 # parameter dtype for units that do not override it
        self.used: dict[str, str] = {}     # unit name -> providing plugin (for the lock)
        self._buffers: dict[str, torch.Tensor] = {}

    def unit(self, ref: Ref, **site: Any) -> nn.Module:
        plugin = self.plugins.provider("unit", ref)
        self.plugins.load(plugin)
        self.used[ref.name] = plugin
        return self.plugins.regs.units.get(ref.name)(self, **dict(ref.params), **site)

    def static(self, name: str, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """A named buffer allocated once and reused (a stable address for graph capture)."""
        buf = self._buffers.get(name)
        if buf is None:
            buf = self._buffers[name] = torch.zeros(shape, dtype=dtype, device=self.device)
        elif buf.shape != shape or buf.dtype != dtype:
            raise PluginError(f"static buffer {name!r} requested with two different shapes/dtypes")
        return buf

    def workspace(self, name: str, nbytes: int) -> torch.Tensor:
        return self.static(f"workspace.{name}", (nbytes,), torch.uint8)

    def state(self, spec_name: str) -> Any:
        raise PluginError("state pools are bound by the engine; a model-only build has none")


@dataclasses.dataclass
class EagerStepCtx:
    """A minimal StepCtx for stateless reference runs: whole sequences from position 0, no KV cache."""

    batch: Batch
    seq_table: SeqTable | None = None
    metas: dict[str, Any] = dataclasses.field(default_factory=dict)

    def meta(self, group: str) -> Any:
        return self.metas[group]

    @classmethod
    def whole_sequences(cls, ids: list[list[int]], device: torch.device | str) -> EagerStepCtx:
        """One step that prefills every sequence in `ids` and returns logits for every token."""
        lens = np.array([len(s) for s in ids], dtype=np.int32)
        flat = np.concatenate([np.asarray(s, dtype=np.int64) for s in ids])
        plan = BatchPlan(step=0, slots=np.arange(len(ids), dtype=np.int32), n_query=lens, n_logits=lens,
                         start_upper=np.zeros_like(lens), token_src=np.full_like(lens, TokenSrc.HOST),
                         host_tokens=flat, shape=ShapeKey(int(lens.sum()), len(ids)))
        starts = np.concatenate([[0], lens.cumsum()])
        dev = DeviceBatch(input_ids=torch.as_tensor(flat, device=device),
                          positions=torch.as_tensor(np.concatenate([np.arange(n) for n in lens]), device=device),
                          query_start=torch.as_tensor(starts, device=device),
                          slots=torch.as_tensor(plan.slots, device=device),
                          logits_index=torch.arange(int(lens.sum()), device=device))
        return cls(Batch(plan, dev))


class Model(nn.Module):
    """embed → blocks → norm → lm_head, all built from the spec's Refs."""

    def __init__(self, spec: ModelSpec, ctx: BuildContext) -> None:
        super().__init__()
        self.spec = spec
        self.embed = ctx.unit(spec.embed, vocab=spec.vocab, hidden=spec.hidden)
        self.layers = nn.ModuleList(
            ctx.unit(spec.block, layer=i, hidden=spec.hidden, mixer=d.mixer, ffn=d.ffn, norm=spec.norm)
            for i, d in enumerate(spec.layers)
        )
        self.norm = ctx.unit(spec.norm, dim=spec.hidden)
        self.lm_head = ctx.unit(spec.lm_head, vocab=spec.vocab, hidden=spec.hidden, embed=self.embed)

    def forward(self, ctx: StepCtx) -> torch.Tensor:
        dev = ctx.batch.dev
        h = self.embed(dev.input_ids)
        for layer in self.layers:
            h = layer(h, ctx)
        return self.lm_head(self.norm(h[dev.logits_index]))


def build_model(spec: ModelSpec, plugins: PluginSet, device: torch.device,
                dtype: torch.dtype = torch.bfloat16) -> tuple[Model, BuildContext]:
    """Build on the meta device (no memory), then materialize parameters directly on `device`."""
    ctx = BuildContext(plugins, device, dtype)
    with torch.device("meta"):
        model = Model(spec, ctx)
    return model.to_empty(device=device).eval(), ctx


@functools.lru_cache(maxsize=256)
def _compile(pattern: str) -> re.Pattern[str]:
    parts = re.split(r"\{(\w+)\}", pattern)
    regex = "".join(re.escape(p) if i % 2 == 0 else (r"(?P<i>\d+)" if p == "i" else f"(?P<{p}>.+)")
                    for i, p in enumerate(parts))
    return re.compile(regex + r"\Z")


def map_name(name: str, weights: WeightMap) -> tuple[str, str | None] | None:
    """Checkpoint tensor name → (parameter path, transform), or None if the tensor is skipped."""
    if weights.root and name.startswith(weights.root):
        name = name[len(weights.root):]
    if any(name.startswith(s) for s in weights.skip):
        return None
    for rule in weights.rules:
        m = _compile(rule.pattern).match(name)
        if m:
            return rule.target.format(**m.groupdict()), rule.transform
    raise PluginError(f"no weight rule maps checkpoint tensor {name!r}")


@torch.no_grad()
def load_weights(model: Model, ctx: BuildContext, tensors: Iterable[tuple[str, torch.Tensor]]) -> None:
    """Fill every parameter exactly once from the checkpoint, then run each unit's `finalize_weights`."""
    params = dict(model.named_parameters())
    filled: set[str] = set()
    for name, tensor in tensors:
        mapped = map_name(name, model.spec.weights)
        if mapped is None:
            continue
        target, transform = mapped
        if target not in params:
            raise PluginError(f"{name!r} maps to {target!r}, which the model does not have")
        if target in filled:
            raise PluginError(f"{target!r} filled twice (last from {name!r})")
        if transform:
            tensor = ctx.plugins.regs.transforms.get(transform)(tensor)
        if params[target].shape != tensor.shape:
            raise PluginError(f"{target}: checkpoint shape {tuple(tensor.shape)} != {tuple(params[target].shape)}")
        params[target].copy_(tensor)
        filled.add(target)
    if missing := sorted(set(params) - filled):
        raise PluginError(f"parameters not loaded: {missing[:8]}{' …' if len(missing) > 8 else ''}")
    for module in reversed(list(model.modules())):  # children first: composites may read finalized children
        if module is not model and hasattr(module, "finalize_weights"):
            module.finalize_weights(ctx)
