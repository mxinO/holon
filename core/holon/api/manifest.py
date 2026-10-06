"""Plugin manifests, slots and resources (design §4.1, §4.2).

A manifest is read without importing the plugin, so everything the resolver needs — resources, numerics class,
graph policy, overlap depth, ordering, shape axes — lives here and nowhere else.
"""
from __future__ import annotations

import dataclasses
import enum
import pathlib
import tomllib
import types
from typing import Any, Literal, get_args

from .terms import parse_versioned


class Cardinality(enum.Enum):
    SINGLE = "single"                # exactly one per Plan
    PER_SITE = "per_site"            # bound per site (layer, op instance, state spec, store)
    ORDERED_MULTI = "ordered_multi"  # ordered list, `after`/`before` constraints
    MULTI = "multi"                  # any number, keyed by name


SLOTS: types.MappingProxyType[str, Cardinality] = types.MappingProxyType({
    **dict.fromkeys(["model", "platform", "launcher", "schedule_policy", "sampler", "verifier", "draft_policy",
                     "eviction_policy"], Cardinality.SINGLE),
    **dict.fromkeys(["unit", "kernel", "state_kind", "placement_policy", "codec", "weight_source"],
                    Cardinality.PER_SITE),
    **dict.fromkeys(["plan_transform", "logits_processor", "checkpoint_policy"], Cardinality.ORDERED_MULTI),
    **dict.fromkeys(["frontend", "dialect", "converter", "tier", "mover", "drafter", "encoder", "connector",
                     "exporter"], Cardinality.MULTI),
})

Numerics = Literal["policy", "exact", "bounded", "approx"]
GraphPolicy = Literal["full", "piecewise", "eager", "host"]
_NUMERICS = set(get_args(Numerics))
_GRAPH = set(get_args(GraphPolicy))


@dataclasses.dataclass(frozen=True, slots=True)
class Resource:
    """A typed resource `ns.name@major`, e.g. `store.tiered@1`. Matching on major means "compatible"."""

    name: str
    major: int

    @classmethod
    def parse(cls, text: str) -> Resource:
        return cls(*parse_versioned(text))

    def __str__(self) -> str:
        return f"{self.name}@{self.major}"


TunableType = Literal["int", "float", "bool", "choice"]
TunableApply = Literal["live", "restart"]
_TUNABLE_PY = {"int": int, "float": (int, float), "bool": bool}


@dataclasses.dataclass(frozen=True, slots=True)
class Tunable:
    type: TunableType
    default: Any
    range: tuple[Any, Any] | None = None
    choices: tuple[Any, ...] | None = None
    log: bool = False
    apply: TunableApply = "restart"

    def __post_init__(self) -> None:
        if self.type not in get_args(TunableType) or self.apply not in get_args(TunableApply):
            raise ValueError(f"invalid tunable type/apply: {self.type!r}/{self.apply!r}")
        py = _TUNABLE_PY.get(self.type)
        if py and (not isinstance(self.default, py) or (self.type != "bool" and isinstance(self.default, bool))):
            raise ValueError(f"{self.type} tunable has default {self.default!r}")
        if self.type == "choice" and (not self.choices or self.default not in self.choices):
            raise ValueError(f"choice tunable: default {self.default!r} not in {self.choices!r}")
        if self.range is not None and not self.range[0] <= self.default <= self.range[1]:
            raise ValueError(f"tunable default {self.default!r} outside range {self.range!r}")


@dataclasses.dataclass(frozen=True, slots=True)
class Provenance:
    """Where borrowed code comes from (design §5.6 rule 5)."""

    source: str              # repository URL
    commit: str
    license: str             # SPDX
    files: tuple[str, ...] = ()


_SECTIONS: dict[str, set[str]] = {
    "plugin": {"name", "version", "entry", "slot", "license", "core_abi", "slot_abi", "variant_of", "implements",
               "derived_from"},
    "resources": {"provides", "requires"},
    "properties": {"numerics", "graph_policy", "max_depth", "after", "before", "shape_axes"},
    "dependencies": {"python"},
    "reference": {"module"},
    "tunables": None,  # any names; each value is validated as a Tunable
    "support": {"platforms"},
    "provenance": {"source", "commit", "license", "files"},
}


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class Manifest:
    name: str
    version: str
    entry: str                         # "module:attr", relative to the plugin package; imported only if selected
    slot: str
    license: str
    core_abi: str = ">=0.0"
    slot_abi: str = "0"
    variant_of: str | None = None      # lineage of forks
    implements: str | None = None      # "<unit>@sem:<hash>" for numeric units, or "<slot>@<major>"
    derived_from: str | None = None    # copied recipes / explicit old-version copies
    provides: tuple[Resource, ...] = ()
    requires: tuple[Resource, ...] = ()
    numerics: Numerics = "exact"
    graph_policy: GraphPolicy = "full"
    max_depth: int = 1                 # largest overlap depth tolerated (design §7.3)
    after: tuple[str, ...] = ()        # ordering for ordered_multi slots
    before: tuple[str, ...] = ()
    shape_axes: dict[str, tuple[Any, ...]] = dataclasses.field(default_factory=dict)  # finite graph-capture axes
    dependencies: tuple[str, ...] = ()  # third-party imports allowed (invariant 2)
    reference: str | None = None       # oracle module of a numeric unit
    tunables: dict[str, Tunable] = dataclasses.field(default_factory=dict)
    platforms: tuple[str, ...] = ()
    provenance: Provenance | None = None

    @property
    def cardinality(self) -> Cardinality:
        return SLOTS[self.slot]

    @classmethod
    def load(cls, path: pathlib.Path) -> Manifest:
        """Read `plugin.toml`. Every unknown section or key and every invalid value is an error naming the file."""
        try:
            with path.open("rb") as f:
                return cls._from_toml(tomllib.load(f))
        except (ValueError, TypeError, KeyError, tomllib.TOMLDecodeError) as e:
            raise ValueError(f"{path}: {e}") from None

    @classmethod
    def _from_toml(cls, data: dict[str, Any]) -> Manifest:
        for section, body in data.items():
            if section not in _SECTIONS:
                raise ValueError(f"unknown section [{section}]")
            allowed = _SECTIONS[section]
            if allowed is not None and set(body) - allowed:
                raise ValueError(f"unknown keys in [{section}]: {sorted(set(body) - allowed)}")
        if "plugin" not in data:
            raise ValueError("missing [plugin] section")
        props, res = data.get("properties", {}), data.get("resources", {})
        m = cls(
            **data["plugin"],
            provides=tuple(map(Resource.parse, res.get("provides", ()))),
            requires=tuple(map(Resource.parse, res.get("requires", ()))),
            numerics=props.get("numerics", "exact"),
            graph_policy=props.get("graph_policy", "full"),
            max_depth=props.get("max_depth", 1),
            after=tuple(props.get("after", ())),
            before=tuple(props.get("before", ())),
            shape_axes={k: tuple(v) for k, v in props.get("shape_axes", {}).items()},
            dependencies=tuple(data.get("dependencies", {}).get("python", ())),
            reference=data.get("reference", {}).get("module"),
            tunables={k: _tunable(v) for k, v in data.get("tunables", {}).items()},
            platforms=tuple(data.get("support", {}).get("platforms", ())),
            provenance=_provenance(data["provenance"]) if "provenance" in data else None,
        )
        if m.slot not in SLOTS:
            raise ValueError(f"unknown slot {m.slot!r}")
        if m.numerics not in _NUMERICS or m.graph_policy not in _GRAPH:
            raise ValueError(f"invalid numerics/graph_policy: {m.numerics!r}/{m.graph_policy!r}")
        if type(m.max_depth) is not int or m.max_depth < 0:
            raise ValueError(f"max_depth must be a non-negative int, got {m.max_depth!r}")
        return m


def _provenance(v: dict[str, Any]) -> Provenance:
    return Provenance(**{**v, "files": tuple(v.get("files", ()))})


def _tunable(v: dict[str, Any]) -> Tunable:
    v = dict(v)
    for key in ("range", "choices"):
        if key in v:
            v[key] = tuple(v[key])
    return Tunable(**v)
