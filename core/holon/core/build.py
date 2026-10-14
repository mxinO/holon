"""Build a model from its ModelSpec and load weights through its WeightMap (design §5.1).

A `Ref("std.ops.rmsnorm@1")` is served by the plugin whose manifest provides `unit.std.ops.rmsnorm@1`.
"""
from __future__ import annotations

import dataclasses
import functools
import re
import sys
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from torch import nn

from holon.api import (
    Batch,
    DeviceBatch,
    KernelSpec,
    ModelSpec,
    QTensor,
    Ref,
    SeqTable,
    StepCtx,
    WeightMap,
    parse_versioned,
)

from . import platform
from .batching import host_plan
from .plugins import PluginError, PluginSet


class BuildContext:
    """Instantiates units by Ref; the only way models and units obtain other units (invariant 2)."""

    def __init__(self, plugins: PluginSet, device: torch.device, dtype: torch.dtype = torch.bfloat16,
                 options: dict[str, Any] | None = None, ledger: Any = None) -> None:
        self.plugins = plugins
        self.ledger = ledger               # the engine's memory ledger (None: a model-only build)
        self.device = device
        self.dtype = dtype                 # parameter dtype for units that do not override it
        self.fingerprint = platform.fingerprint(device)
        self.options = options or {}
        self.used: dict[str, str] = {}     # unit name -> providing plugin (for the lock)
        self.kernels: dict[str, dict[str, Any]] = {}   # "<site>/<op>" -> provider, facts, props, config (lock)
        self._selected: dict[tuple, KernelSpec] = {}
        self.states: dict[tuple[str, int], tuple[Any, Any]] = {}  # (local spec name, layer) -> (spec, pool)
        self._buffers: dict[str, torch.Tensor] = {}
        self.aux: Any = None               # the engine's AuxModels, which `aux_model` builds into (engine builds)
        self.taps: dict[int, list[Any]] = {}   # layer -> consumers of the residual stream after it (`Model.trunk`)
        self.sources: dict[str, str] = {}      # path -> identity of every checkpoint `weight_source` opened (plans)

    def unit_factory(self, ref: Ref) -> Any:
        """The registered factory (usually the unit class) serving `ref`, loading its plugin if needed."""
        plugin = self.plugins.provider("unit", ref)
        self.plugins.load(plugin)
        self.used[ref.name] = plugin
        return self.plugins.regs.units.get(ref.name)

    def unit(self, ref: Ref, **site: Any) -> nn.Module:
        return self.unit_factory(ref)(self, **dict(ref.params), **site)

    def select_kernel(self, op: str, *, site: str, **facts: Any) -> KernelSpec:
        key = (op, tuple(sorted(facts.items())))
        spec = self._selected.get(key)
        if spec is None:
            spec = self._selected[key] = self._choose(op, facts)
        self.kernels[f"{site}/{op}"] = {"provider": spec.provider, "facts": facts, "props": sorted(spec.props),
                                        "config": spec.config}
        return spec

    def candidates(self, op: str, facts: dict[str, Any]) -> list[KernelSpec]:
        """Every kernel for versioned `op` that runs here and supports `facts`, loading its available providers."""
        name, major = parse_versioned(op)
        for plugin in self.plugins.providers(f"kernel.{name}", major):
            if self.plugins.available(plugin, self.device.type):
                self.plugins.load(plugin)
        sm = self.fingerprint.sm or None
        return [s for s in self.plugins.regs.kernels.get(name)
                if s.requires.allows(self.device.type, sm) and (s.supports is None or s.supports(facts))]

    def _choose(self, op: str, facts: dict[str, Any]) -> KernelSpec:
        name = parse_versioned(op)[0]
        pinned = self.options.get("kernels", {}).get(name)
        found = [s for s in self.candidates(op, facts) if pinned in (None, s.provider)]
        if pinned is None and (preferred := self.options.get("kernel_preferences", {}).get(name)) is not None:
            if chosen := [s for s in found if s.provider == preferred]:
                found = chosen
            else:                                   # soft: a lock's pin from another class or another kernel set
                print(f"holon: {op}: preferred {preferred!r} does not serve {facts} on {self.device.type} sm "
                      f"{self.fingerprint.sm}; default priority", file=sys.stderr, flush=True)
        if not found:
            raise PluginError(f"no kernel for {op} on {self.device.type} sm {self.fingerprint.sm} with {facts}"
                              + (f" from pinned provider {pinned!r}" if pinned else ""))
        best = max(s.priority for s in found)
        top = [s for s in found if s.priority == best]
        if len(top) > 1:
            raise PluginError(f"{op}: providers {[s.provider for s in top]} tie at priority {best}")
        return top[0]

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

    def pool(self, name: str, *, unit_bytes: int, segments: int = 1, min_units: int = 0, pref_units: int = 0) -> Any:
        if self.ledger is None:
            raise PluginError(f"pool {name!r}: a model-only build has no memory ledger")
        return self.ledger.plugin_pool(name, unit_bytes=unit_bytes, segments=segments, min_units=min_units,
                                       pref_units=pref_units)

    def bind_state(self, spec: Any, pool: Any) -> None:
        """Make `pool` the state of `spec` for its layers (engine builds; model-only builds bind nothing)."""
        for layer in spec.layers:
            if (spec.name, layer) in self.states:
                raise PluginError(f"state {spec.name!r} bound twice for layer {layer}")
            self.states[(spec.name, layer)] = (spec, pool)

    def state(self, spec_name: str, layer: int) -> tuple[Any, Any] | None:
        return self.states.get((spec_name, layer))

    def aux_model(self, namespace: str, spec: Any, tensors: Iterable[tuple[str, torch.Tensor | QTensor]],
                  options: dict[str, Any] | None = None) -> nn.Module:
        if self.aux is None:
            raise PluginError(f"aux model {namespace!r}: a model-only build has no engine state")
        return self.aux.build(namespace, spec, tensors, options)

    def weight_source(self, path: str) -> Any:
        from .assembly import open_source

        source = open_source(self.plugins, path)
        self.sources[str(path)] = source.identity()
        return source

    def tap(self, layer: int, consume: Any) -> None:
        self.taps.setdefault(layer, []).append(consume)

    def child(self, namespace: str, options: dict[str, Any] | None = None) -> ChildContext:
        return ChildContext(self, namespace, options)


class ChildContext:
    """The BuildCtx an auxiliary model's units see: the parent's, with state names, static buffers, plugin pools and
    lock sites under `<namespace>.`, so the same units build beside the target without colliding with it; `options`
    adds deployment options (e.g. `planner.device_lens` for drafter sub-steps). Kernel workspaces are shared: target
    and aux kernels run in stream order."""

    def __init__(self, parent: BuildContext, namespace: str, options: dict[str, Any] | None = None) -> None:
        self.parent, self.namespace = parent, namespace
        self.plugins, self.device, self.dtype = parent.plugins, parent.device, parent.dtype
        self.fingerprint = parent.fingerprint
        self.options = parent.options | (options or {})

    def unit_factory(self, ref: Ref) -> Any:
        return self.parent.unit_factory(ref)

    def unit(self, ref: Ref, **site: Any) -> nn.Module:
        return self.parent.unit_factory(ref)(self, **dict(ref.params), **site)       # nested units see the child too

    def select_kernel(self, op: str, *, site: str, **facts: Any) -> KernelSpec:
        return self.parent.select_kernel(op, site=f"{self.namespace}.{site}", **facts)

    def state(self, spec_name: str, layer: int) -> tuple[Any, Any] | None:
        return self.parent.state(f"{self.namespace}.{spec_name}", layer)

    def bind_state(self, spec: Any, pool: Any) -> None:
        self.parent.bind_state(spec, pool)            # specs arrive already under the namespace

    def static(self, name: str, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        return self.parent.static(f"{self.namespace}.{name}", shape, dtype)

    def workspace(self, name: str, nbytes: int) -> torch.Tensor:
        return self.parent.workspace(name, nbytes)

    def pool(self, name: str, **sizes: Any) -> Any:
        return self.parent.pool(f"{self.namespace}.{name}", **sizes)


@dataclasses.dataclass
class StepContext:
    """The core's StepCtx: the step's batch plus each state group's planner metadata."""

    batch: Batch
    metas: dict[str, Any] = dataclasses.field(default_factory=dict)
    seq_table: SeqTable | None = None
    pages: dict[str, Any] = dataclasses.field(default_factory=dict)    # the core's page views (engine steps)
    prep: Any = None                                                    # the engine's device-prep inputs
    pools: dict[str, Any] = dataclasses.field(default_factory=dict)     # the ledger's pools, by name
    substeps: Any = None                       # the drafter sub-steps planned with this step (Executor)
    sampler: Any = None                        # the engine's sampler (verifiers draw through it)
    namespace: str = ""                        # set by `view`: whose names `meta` and `substep` use

    def meta(self, group: str) -> Any:
        return self.metas[group]

    def view(self, namespace: str) -> StepContext:
        """This step as an auxiliary model's units see it: its groups' metadata, pages and in-step publishes under
        their local names (`<namespace>.kv` → `kv`)."""
        pre = namespace + "."
        local = lambda d: {k[len(pre):]: v for k, v in d.items() if k.startswith(pre)}    # noqa: E731
        return dataclasses.replace(self, batch=local_batch(self.batch, namespace), metas=local(self.metas),
                                   pages=local(self.pages), namespace=namespace)

    def substep(self, q: int, offset: int) -> StepContext:
        """The next of the drafter's sub-steps (`Drafter.substeps`): q tokens per sampled row at positions
        `num_computed + offset + j`, with its own buffers, ready to run (in this context's namespace)."""
        if self.substeps is None:
            raise RuntimeError("this step planned no sub-steps (Drafter.substeps)")
        sub = self.substeps.next(q, offset)
        return sub.view(self.namespace) if self.namespace else sub

    def pool(self, name: str) -> Any:
        return self.pools[name]

    @classmethod
    def whole_sequences(cls, ids: list[list[int]], device: torch.device | str) -> StepContext:
        """A stateless step that prefills every sequence in `ids` and returns logits for every token (reference
        runs of a model built without state)."""
        lens = np.array([len(s) for s in ids], dtype=np.int32)
        flat = np.concatenate([np.asarray(s, dtype=np.int64) for s in ids])
        plan = host_plan(0, np.arange(len(ids), dtype=np.int32), np.zeros_like(lens), lens, lens, flat)
        starts = np.concatenate([[0], lens.cumsum()])
        dev = DeviceBatch(input_ids=torch.as_tensor(flat, device=device),
                          positions=torch.as_tensor(np.concatenate([np.arange(n) for n in lens]), device=device),
                          query_start=torch.as_tensor(starts, device=device),
                          slots=torch.as_tensor(plan.slots, device=device),
                          logits_index=torch.arange(int(lens.sum()), device=device),
                          logits_rows=torch.repeat_interleave(torch.arange(len(ids)), torch.as_tensor(lens)).to(device))
        return cls(Batch(plan, dev))


def local_batch(batch: Batch, namespace: str) -> Batch:
    """`batch` with only the in-step publishes of `namespace`'s specs, under their local names."""
    pre = namespace + "."
    publishes = [dataclasses.replace(p, spec=p.spec[len(pre):]) for p in batch.plan.publishes if p.spec.startswith(pre)]
    return Batch(dataclasses.replace(batch.plan, publishes=publishes), batch.dev)


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
        self.taps = ctx.taps               # filled after the build, by a drafter (`BuildCtx.tap`)

    def forward(self, ctx: StepCtx) -> torch.Tensor:
        """Logits of the step's `logits_index` rows."""
        return self.head(self.trunk(ctx)[ctx.batch.dev.logits_index])

    def trunk(self, ctx: StepCtx) -> torch.Tensor:
        """Final hidden states of every token of the step, before the norm; the residual stream after each tapped
        layer goes to its consumers on the way."""
        h = self.embed(ctx.batch.dev.input_ids)
        for i, layer in enumerate(self.layers):
            h = layer(h, ctx)
            for consume in self.taps.get(i, ()):
                consume(h, ctx)
        return h

    def head(self, h: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(h))


def build_model(spec: ModelSpec, ctx: BuildContext) -> Model:
    """Build on the meta device: no parameter takes memory until `load_weights` fills it, in its stored form."""
    with torch.device("meta"):
        model = Model(spec, ctx)
    return model.eval()


@functools.lru_cache(maxsize=256)
def _compile(pattern: str, stack: str | None = None) -> re.Pattern[str]:
    parts = re.split(r"\{(\w+)\}", pattern)
    if stack is not None and stack not in parts[1::2]:
        raise PluginError(f"weight rule {pattern!r}: stack {stack!r} is not one of its captures")
    regex = "".join(re.escape(p) if i % 2 == 0 else (f"(?P<{p}>\\d+)" if p in ("i", stack) else f"(?P<{p}>.+)")
                    for i, p in enumerate(parts))
    return re.compile(regex + r"\Z")


def map_name(name: str, weights: WeightMap) -> tuple[str, str | None, int | None] | None:
    """Checkpoint tensor name → (parameter path, transform, stack index), or None if the tensor is skipped."""
    if weights.root and name.startswith(weights.root):
        name = name[len(weights.root):]
    if any(name.startswith(s) for s in weights.skip) or (weights.only and not name.startswith(weights.only)):
        return None
    for rule in weights.rules:
        m = _compile(rule.pattern, rule.stack).match(name)
        if m:
            return rule.target.format(**m.groupdict()), rule.transform, int(m[rule.stack]) if rule.stack else None
    raise PluginError(f"no weight rule maps checkpoint tensor {name!r}")


def _adopt(model: nn.Module, target: str, q: QTensor, transform: str | None, ctx: BuildContext, **index: int) -> None:
    """Hand a quantized tensor to the unit owning `target`: `adopt_weights(ctx, leaf, q)`, or
    `adopt_weights(…, index=i)` for slice `i` of a stack (the unit keeps its stack's first dimension while loading)."""
    if transform:
        raise PluginError(f"{target}: transform {transform!r} cannot apply to a quantized tensor")
    owner, _, leaf = target.rpartition(".")
    unit = model.get_submodule(owner)
    if not getattr(unit, "adopt_weights", None) or not unit.adopt_weights(ctx, leaf, q, **index):
        raise PluginError(f"{target}: {type(unit).__name__} cannot read layout {q.layout}")


def _fill_slice(model: nn.Module, param: nn.Parameter, target: str, index: int, name: str,
                value: torch.Tensor | QTensor, transform: str | None, ctx: BuildContext, got: set[int]) -> None:
    """Copy one checkpoint tensor into slice `index` of a stacked parameter, materializing the stack on its first
    slice (in place: no transient copy of the whole stack); a quantized one goes to its unit."""
    if not 0 <= index < param.shape[0]:
        raise PluginError(f"{name!r}: index {index} outside the {param.shape[0]} slices of {target!r}")
    if index in got:
        raise PluginError(f"{target}[{index}] filled twice (last from {name!r})")
    got.add(index)
    if isinstance(value, QTensor):
        return _adopt(model, target, value, transform, ctx, index=index)
    if transform:
        value = ctx.plugins.regs.transforms.get(transform)(value)
    if tuple(value.shape) != tuple(param.shape[1:]):
        raise PluginError(f"{name!r}: shape {tuple(value.shape)} != a slice of {target!r} {tuple(param.shape[1:])}")
    if param.is_meta:
        torch.utils.swap_tensors(param, nn.Parameter(torch.empty(param.shape, dtype=param.dtype, device=ctx.device),
                                                     requires_grad=False))
    param.data[index].copy_(value.to(dtype=param.dtype))


@torch.no_grad()
def load_weights(model: nn.Module, ctx: BuildContext, tensors: Iterable[tuple[str, torch.Tensor | QTensor]],
                 weights: WeightMap | None = None) -> None:
    """Fill every parameter exactly once from the checkpoint by `weights` (default: the model spec's), then run each
    unit's `finalize_weights`.

    A dense tensor is copied to the device in the parameter's dtype. A quantized one (`QTensor`) goes to the module
    owning the parameter — `adopt_weights(ctx, name, qtensor)`, with `index=i` for slice `i` of a stack, stores it in
    its own form and returns True, or False if it cannot read that layout (design §5.5; the core never looks inside a
    layout)."""
    weights = weights or model.spec.weights
    params = dict(model.named_parameters())
    filled: set[str] = set()
    slices: dict[str, set[int]] = {}                    # stacked parameters: the slices filled so far
    for name, value in tensors:
        mapped = map_name(name, weights)
        if mapped is None:
            continue
        target, transform, index = mapped
        if target not in params:
            raise PluginError(f"{name!r} maps to {target!r}, which the model does not have")
        if (index is None and target in slices) or (index is not None and target in filled):
            raise PluginError(f"{target!r} filled both whole and by stack slices (last from {name!r})")
        if index is not None:
            _fill_slice(model, params[target], target, index, name, value, transform, ctx,
                        slices.setdefault(target, set()))
            continue
        if target in filled:
            raise PluginError(f"{target!r} filled twice (last from {name!r})")
        if isinstance(value, QTensor):
            _adopt(model, target, value, transform, ctx)
        else:
            if transform:
                value = ctx.plugins.regs.transforms.get(transform)(value)
            param = params[target]
            if param.shape != value.shape:
                raise PluginError(f"{target}: checkpoint shape {tuple(value.shape)} != {tuple(param.shape)}")
            # The meta parameter becomes the loaded tensor in place (the Parameter object others hold is kept).
            torch.utils.swap_tensors(param, nn.Parameter(value.to(device=ctx.device, dtype=param.dtype),
                                                         requires_grad=False))
        filled.add(target)
    for target, got in slices.items():
        if missing := sorted(set(range(params[target].shape[0])) - got):
            raise PluginError(f"{target}: stack slices not loaded: {missing[:8]}{' …' if len(missing) > 8 else ''}")
    if missing := sorted(n for n, p in model.named_parameters() if p.is_meta):
        raise PluginError(f"parameters not loaded: {missing[:8]}{' …' if len(missing) > 8 else ''}")
    del params                                      # weights merges give up are freed as they go
    for module in reversed(list(model.modules())):  # children first: composites may read finalized children
        if module is not model and hasattr(module, "finalize_weights"):
            module.finalize_weights(ctx)
    for module in model.modules():                  # after merges: units bind the kernels of the weights they hold
        if hasattr(module, "bind_kernels"):
            module.bind_kernels(ctx)
