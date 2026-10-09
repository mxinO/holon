"""Engine assembly (design §7): the checkpoint's weight source and the model plugin for its architecture, and the
state the model's mixers declare — bound to ledger pools and state kinds, with one planner per state group and a
prefix index.
"""
from __future__ import annotations

import collections
import fnmatch
import math
import pathlib
from typing import Any

from holon.api import Batch, ModelSpec, Paged, Ref, StateSpec, canonical_hash

from .build import BuildContext
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
    by_mixer: dict[Ref, list[int]] = collections.defaultdict(list)
    for i, layer in enumerate(spec.layers):
        by_mixer[layer.mixer].append(i)
    bound, planners = [], {}
    for ref, layers in by_mixer.items():
        cls = ctx.unit_factory(ref)
        specs = cls.state_specs(dict(ref.params), tuple(layers), ctx) if hasattr(cls, "state_specs") else []
        bound += [_bind(s, ctx, plugins, ledger, cfg) for s in specs]
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
