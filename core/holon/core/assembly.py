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
import sys
from typing import Any

import torch
from torch import nn

from holon.api import AuxSpec, Batch, ModelSpec, Paged, Ref, Resolved, StateSpec, canonical_hash

from .build import BuildContext, load_weights
from .executor import profile_rows
from .memory import Ledger
from .plugins import PluginError, PluginSet
from .prefix import PrefixIndex
from .state import BoundSpec


def resolve_model(plugins: PluginSet, cfg: Any) -> tuple[Any, Any]:
    """The checkpoint's weight source and the model plugin that provides its architecture (`cfg`: EngineConfig)."""
    source = open_source(plugins, cfg.model, cfg.source)
    return source, plugins.load(plugins.owner(f"arch.{source.arch}")[0])


def open_source(plugins: PluginSet, path: str, kind: str | None = None) -> Any:
    """The weight source of `kind` (default: the one whose manifest recognises `path`) reading `path`."""
    kind = kind or detect_source(plugins, path)
    plugins.load(plugins.owner(f"weight_source.{kind}")[0])
    return plugins.regs.weight_sources.get(kind)(path)


def claim_ref(plugins: PluginSet, ref: str, kind: str | None = None) -> Resolved:
    """The one weight-source plugin's claim of `ref`, a model reference that is not a local path (e.g. a hub id),
    through the optional `resolve` hook; with `kind`, only the plugin providing that weight source is asked. A
    ValueError from a hook is a hint, shown when no plugin claims the reference."""
    claims, hints = {}, []
    for name, inst in sorted(plugins.installed.items()):
        provides = {r.name for r in inst.manifest.provides}
        if inst.manifest.slot != "weight_source" or (kind is not None and f"weight_source.{kind}" not in provides):
            continue
        resolve = getattr(plugins.load(name), "resolve", None)
        try:
            if resolve is not None and (claim := resolve(ref)) is not None:
                claims[name] = claim
        except ValueError as e:
            hints.append(f"{name}: {e}")
    if len(claims) > 1:
        raise PluginError(f"{ref}: several weight sources resolve it ({sorted(claims)}); name one with --source")
    if not claims:
        raise PluginError(f"{ref}: no such file or directory, and no weight source resolves it"
                          + "".join(f"\n  {h}" for h in hints))
    return claims.popitem()[1]


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
    """Default planner: kernels read the core's `Pages` directly, the same for every step (mixed ones included)."""

    mixes_row_kinds = True

    def plan_bounds(self, batch: Batch, pages: Any) -> Any:
        return pages

    def prepare_device(self, meta: Any, ctx: Any) -> None:
        pass


def bind_state(spec: ModelSpec, ctx: BuildContext, plugins: PluginSet, ledger: Ledger, cfg: Any
               ) -> tuple[list[BoundSpec], dict[str, Any]]:
    """Each mixer group's and the block's state specs → pools and bound state kinds; one planner per state group."""
    return _bind_layers(spec.layers, ctx, plugins, ledger, cfg, block=spec.block)


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

    def build(self, namespace: str, spec: AuxSpec, tensors: Any, options: dict[str, Any] | None = None) -> nn.Module:
        if any(namespace == ns for ns, _ in self.modules):
            raise PluginError(f"aux model {namespace!r} built twice")
        child = self.ctx.child(namespace, options)
        bound, planners = _bind_layers(spec.layers, child, self.plugins, self.ledger, self.cfg, namespace)
        self.bound += bound
        self.planners |= planners
        with torch.device("meta"):
            module = child.unit(spec.unit, layers=spec.layers)
        load_weights(module.eval(), child, tensors, spec.weights)
        self.modules.append((namespace, module))
        self._layers.append((namespace, spec.layers, options or {}))
        return module

    def sub_planners(self, n: int) -> dict[str, list[Any]]:
        """`n` planners per auxiliary state group, one per drafter sub-step: kernel instances hold per-step buffers,
        and a sub-step's lengths exist only on the device (`planner.device_lens`)."""
        out: dict[str, list[Any]] = {}
        if n and (recurrent := [b.spec.name for b in self.bound if not isinstance(b.spec.shape, Paged)]):
            raise PluginError(f"{recurrent}: drafter sub-steps over per-request (recurrent) aux state are not "
                              "supported — a sub-step would advance it past the committed tokens")
        for namespace, layers, options in self._layers:
            child = self.ctx.child(namespace, options | {"planner.device_lens": True})
            for _ in range(n):
                for group, planner in _bind_layers(layers, child, self.plugins, self.ledger, self.cfg, namespace,
                                                   plan_only=True)[1].items():
                    if getattr(planner, "needs_host_lens", False):
                        raise PluginError(f"{group}: drafter sub-steps need planners that read lengths on the device")
                    out.setdefault(group, []).append(planner)
        return out


def _bind_layers(layers: tuple, ctx: Any, plugins: PluginSet, ledger: Ledger, cfg: Any, namespace: str = "",
                 plan_only: bool = False, block: Ref | None = None) -> tuple[list[BoundSpec], dict[str, Any]]:
    """Bind the state the layers' mixers declare, and the `block`'s over all layers (`namespace`: under
    `<namespace>.`); `plan_only`: no binding, new planners only. A group without a planner gets the default one,
    which no step's graph can capture: the block's Slot state is refused for it, a mixer's says so."""
    by_mixer: dict[Ref, list[int]] = collections.defaultdict(list)
    for i, layer in enumerate(layers):
        by_mixer[layer.mixer].append(i)
    bound, planners = [], {}
    owners = [(ref, tuple(at), False) for ref, at in by_mixer.items()] + ([(block, tuple(range(len(layers))), True)]
                                                                         if block is not None else [])
    for ref, at, is_block in owners:
        cls = ctx.unit_class(ref)
        specs = cls.state_specs(dict(ref.params), at, ctx) if hasattr(cls, "state_specs") else []
        if namespace:                      # e.g. "kv" → "draft.kv": its own pools, tables and planners
            specs = [dataclasses.replace(s, name=f"{namespace}.{s.name}",
                                         page_group=s.page_group and f"{namespace}.{s.page_group}") for s in specs]
        if not plan_only:             # kinds see the engine's context: the specs carry their full names
            bound += [_bind(s, getattr(ctx, "parent", ctx), plugins, ledger, cfg) for s in specs]
        for group in dict.fromkeys(s.group for s in specs):
            if group in planners:
                raise PluginError(f"state group {group!r} declared twice (by two mixers, or a mixer and the block)")
            members = [s for s in specs if s.group == group]
            if not hasattr(cls, "planner"):
                if is_block and not all(isinstance(s.shape, Paged) for s in members):
                    raise PluginError(f"state group {group!r}: the block {ref.target} declares Slot state without "
                                      "a planner")
                if cfg.cuda_graphs and ctx.device.type == "cuda" and not plan_only:     # once: not per sub-step
                    print(f"holon: state group {group!r} has no planner: steps run without CUDA graphs",
                          file=sys.stderr, flush=True)
            planners[group] = (cls.planner(dict(ref.params), members, ctx) if hasattr(cls, "planner")
                               else _PagesPlanner())
    return bound, planners


def _bind(spec: StateSpec, ctx: BuildContext, plugins: PluginSet, ledger: Ledger, cfg: Any) -> BoundSpec:
    shape = spec.shape
    if isinstance(shape, Paged):         # one full-length request, or (clamp_context) the profiled step's rows
        rows = profile_rows(cfg) if cfg.clamp_context else [cfg.max_model_len]
        page = shape.tokens_per_unit
        sizing = {"min_units": sum(-(-r // page) for r in rows), "tokens_per_unit": page}
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
    if len(aligns := {b.spec.alignment for b in bound if not isinstance(b.spec.shape, Paged)}) > 1:   # one lcm grid
        print(f"holon: non-paged state of alignments {sorted(aligns)}: prefix hits need one", file=sys.stderr)
    block = math.lcm(*(b.spec.shape.tokens_per_unit for b in paged), *(b.spec.alignment for b in bound))
    # Model + recipe identity. Enough within one process; weight updates (v1) and persistent tiers must add the
    # checkpoint identity and weight version (design §6.3).
    namespace = int(canonical_hash(spec).removeprefix("sha256:")[:16], 16)
    return PrefixIndex(namespace, block, {b.spec.name: b.pool for b in paged}, pools)
