"""Engine assembly (design §7): the checkpoint's weight source and the model plugin for its architecture, and the
state the model's mixers declare — bound to ledger pools and state kinds, with one planner per state group and a
prefix index.
"""
from __future__ import annotations

import collections
import dataclasses
import fnmatch
import math
import pathlib
from typing import Any

import torch
from torch import nn

from holon.api import AuxSpec, Batch, ModelSpec, Paged, Ref, StateSpec, canonical_hash

from .build import BuildContext, load_weights
from .memory import Ledger
from .plugins import PluginError, PluginSet
from .prefix import PrefixIndex
from .state import BoundSpec


def resolve_model(plugins: PluginSet, cfg: Any) -> tuple[Any, Any]:
    """The checkpoint's weight source and the model plugin that provides its architecture (`cfg`: EngineConfig)."""
    kind = cfg.source or detect_source(plugins, cfg.model)
    plugins.load(plugins.owner(f"weight_source.{kind}")[0])
    source = plugins.regs.weight_sources.get(kind)(cfg.model)
    return source, plugins.load(plugins.owner(f"arch.{source.arch}")[0])


def detect_source(plugins: PluginSet, model: str) -> str:
    """The weight-source kind whose manifest `detect` patterns match `model` — its file name, or a file inside it —
    read from manifests alone (no plugin is imported to decide). Exactly one must match."""
    path = pathlib.Path(model)
    names = [path.name, *(p.name for p in path.iterdir())] if path.is_dir() else [path.name]
    kinds = {r.name.removeprefix("weight_source.") for inst in plugins.installed.values()
             for r in inst.manifest.provides if r.name.startswith("weight_source.")
             if any(fnmatch.fnmatch(n, pattern) for pattern in inst.manifest.detect for n in names)}
    if len(kinds) != 1:
        raise PluginError(f"{model}: {'no' if not kinds else 'several'} weight sources recognise it "
                          f"({sorted(kinds)}); name one with EngineConfig.source")
    return kinds.pop()


class _PagesPlanner:
    """Default planner: kernels read the core's `Pages` directly."""

    def plan_bounds(self, batch: Batch, pages: Any) -> Any:
        return pages

    def prepare_device(self, meta: Any, ctx: Any) -> None:
        pass


def bind_state(spec: ModelSpec, ctx: BuildContext, plugins: PluginSet, ledger: Ledger, cfg: Any
               ) -> tuple[list[BoundSpec], dict[str, Any]]:
    """Each mixer group's state specs → pools and bound state kinds; one planner per state group."""
    return _bind_layers(spec.layers, ctx, plugins, ledger, cfg)


class AuxModels:
    """The auxiliary models drafters build (`BuildCtx.aux_model`): each one's layers' state bound under its namespace,
    its units built in a child context that maps their state names there, its weights loaded by its own map. Their
    bound specs and planners join the engine's; drafter sub-steps get planners of their own (`sub_planners`)."""

    def __init__(self, ctx: BuildContext, plugins: PluginSet, ledger: Ledger, cfg: Any) -> None:
        self.ctx, self.plugins, self.ledger, self.cfg = ctx, plugins, ledger, cfg
        self.bound: list[BoundSpec] = []
        self.planners: dict[str, Any] = {}
        self.modules: list[tuple[str, nn.Module]] = []      # (namespace, module)
        self._layers: list[tuple[str, tuple]] = []

    def build(self, namespace: str, spec: AuxSpec, tensors: Any) -> nn.Module:
        if any(namespace == ns for ns, _ in self.modules):
            raise PluginError(f"aux model {namespace!r} built twice")
        child = self.ctx.child(namespace)
        bound, planners = _bind_layers(spec.layers, child, self.plugins, self.ledger, self.cfg, namespace)
        self.bound += bound
        self.planners |= planners
        with torch.device("meta"):
            module = child.unit(spec.unit, layers=spec.layers)
        load_weights(module.eval(), child, tensors, spec.weights)
        self.modules.append((namespace, module))
        self._layers.append((namespace, spec.layers))
        return module

    def sub_planners(self, n: int) -> dict[str, list[Any]]:
        """`n` planners per auxiliary state group, one per drafter sub-step: kernel instances hold per-step buffers,
        and a sub-step's lengths exist only on the device (`planner.device_lens`)."""
        out: dict[str, list[Any]] = {}
        for namespace, layers in self._layers:
            child = self.ctx.child(namespace, {"planner.device_lens": True})
            for _ in range(n):
                for group, planner in _bind_layers(layers, child, self.plugins, self.ledger, self.cfg, namespace,
                                                   plan_only=True)[1].items():
                    if getattr(planner, "needs_host_lens", False):
                        raise PluginError(f"{group}: drafter sub-steps need planners that read lengths on the device")
                    out.setdefault(group, []).append(planner)
        return out


def _bind_layers(layers: tuple, ctx: Any, plugins: PluginSet, ledger: Ledger, cfg: Any, namespace: str = "",
                 plan_only: bool = False) -> tuple[list[BoundSpec], dict[str, Any]]:
    """Bind the state the layers' mixers declare (`namespace`: under `<namespace>.`); `plan_only`: no binding, new
    planners only."""
    by_mixer: dict[Ref, list[int]] = collections.defaultdict(list)
    for i, layer in enumerate(layers):
        by_mixer[layer.mixer].append(i)
    bound, planners = [], {}
    for ref, at in by_mixer.items():
        cls = ctx.unit_factory(ref)
        specs = cls.state_specs(dict(ref.params), tuple(at), ctx) if hasattr(cls, "state_specs") else []
        if namespace:                      # e.g. "kv" → "mtp.kv": its own pools, tables and planners
            specs = [dataclasses.replace(s, name=f"{namespace}.{s.name}",
                                         page_group=s.page_group and f"{namespace}.{s.page_group}") for s in specs]
        if not plan_only:             # kinds see the engine's context: the specs carry their full names
            bound += [_bind(s, getattr(ctx, "parent", ctx), plugins, ledger, cfg) for s in specs]
        for group in dict.fromkeys(s.group for s in specs):
            if group in planners:
                raise PluginError(f"state group {group!r} declared by two mixers")
            members = [s for s in specs if s.group == group]
            planners[group] = (cls.planner(dict(ref.params), members, ctx) if hasattr(cls, "planner")
                               else _PagesPlanner())
    return bound, planners


def _bind(spec: StateSpec, ctx: BuildContext, plugins: PluginSet, ledger: Ledger, cfg: Any) -> BoundSpec:
    shape = spec.shape
    if isinstance(shape, Paged):
        sizing = {"min_units": -(-cfg.max_model_len // shape.tokens_per_unit), "tokens_per_unit": shape.tokens_per_unit}
    else:                                  # every request slot, the padding slot too, owns its units (api.Slot)
        n = (cfg.max_seqs + 1) * (1 + shape.draft_slots)
        sizing = {"min_units": n, "pref_units": n}
    pool = ledger.request(f"state.{spec.name}", unit_bytes=shape.unit_bytes, segments=spec.segments,
                          null_unit=isinstance(shape, Paged), **sizing)
    kind = Ref(spec.kind)
    plugins.load(plugins.provider("state_kind", kind))
    ctx.bind_state(spec, pool)
    return BoundSpec(spec, pool, plugins.regs.state_kinds.get(kind.name)(spec, pool, ctx))


def prefix_index(spec: ModelSpec, bound: list[BoundSpec], enabled: bool, pools: dict[str, Any]) -> PrefixIndex | None:
    """A prefix index over blocks every paged spec can share whole pages of and every spec's alignment divides;
    other state joins through its kind's payloads (`pools`: every pool, by name). None when disabled."""
    if not enabled or not bound:
        return None
    paged = [b for b in bound if isinstance(b.spec.shape, Paged)]
    block = math.lcm(*(b.spec.shape.tokens_per_unit for b in paged), *(b.spec.alignment for b in bound))
    # Model + recipe identity. Enough within one process; weight updates (v1) and persistent tiers must add the
    # checkpoint identity and weight version (design §6.3).
    namespace = int(canonical_hash(spec).removeprefix("sha256:")[:16], 16)
    return PrefixIndex(namespace, block, {b.spec.name: b.pool for b in paged}, pools)
