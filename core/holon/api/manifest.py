"""Plugin manifests, slots and resources (design §4.1, §4.2).

A manifest is read without importing the plugin, so everything the resolver needs — resources, numerics class,
graph policy, ordering, shape axes — lives here and nowhere else.
"""
from __future__ import annotations

import dataclasses
import enum
import operator
import pathlib
import re
import tomllib
import types
from typing import Any, Literal, get_args

from .terms import parse_versioned

ABI = "1.1"    # the plugin ABI this core implements (design §4.5); manifests declare the range they need (`core_abi`)
_COMPARE = {">=": operator.ge, ">": operator.gt, "<=": operator.le, "<": operator.lt, "==": operator.eq}


def abi_allows(spec: str, version: str = ABI) -> bool:
    """Whether `version` is in a manifest's `core_abi` range, comma-separated clauses such as ">=1.0,<2.0"."""
    def key(v: str) -> tuple[int, ...]:
        return (*map(int, v.split(".")), 0, 0)[:3]

    clauses = [re.fullmatch(r"\s*(>=|<=|==|>|<)\s*(\d+(?:\.\d+){0,2})\s*", c) for c in spec.split(",")]
    if not all(clauses):
        raise ValueError(f"core_abi {spec!r}: expected clauses such as '>=1.0,<2.0'")
    return all(_COMPARE[m[1]](key(version), key(m[2])) for m in clauses)


class Cardinality(enum.Enum):
    SINGLE = "single"                # exactly one per Plan
    PER_SITE = "per_site"            # bound per site (layer, op instance, state spec, store)
    ORDERED_MULTI = "ordered_multi"  # ordered list, `after`/`before` constraints
    MULTI = "multi"                  # any number, keyed by name


# The slots a plugin may fill: each has plugins and the core loads them. A new slot is a minor ABI addition
# (design §4.2).
SLOTS: types.MappingProxyType[str, Cardinality] = types.MappingProxyType({
    **dict.fromkeys(["model", "sampler", "verifier", "draft_policy"], Cardinality.SINGLE),
    **dict.fromkeys(["unit", "kernel", "state_kind", "weight_source"], Cardinality.PER_SITE),
    "logits_processor": Cardinality.ORDERED_MULTI,
    **dict.fromkeys(["frontend", "drafter"], Cardinality.MULTI),
})

Numerics = Literal["policy", "exact", "bounded", "approx"]
_NUMERICS = set(get_args(Numerics))


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


@dataclasses.dataclass(frozen=True, slots=True)
class Option:
    """A plugin option the plugin reads (`EngineConfig.plugin_options`, manifest `[options]`): its help line, its
    default as display text, and, for a closed set, the values it accepts (`holon serve` checks them before a start)."""

    help: str
    default: str | None = None
    choices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.help, str) or not self.help:
            raise ValueError("an option needs a help string")
        strings = (*self.choices, self.default or "") if isinstance(self.choices, tuple) else (None,)
        if not all(isinstance(x, str) for x in strings):
            raise ValueError(f"an option's default and choices are strings: {self.default!r}, {self.choices!r}")


@dataclasses.dataclass(frozen=True, slots=True)
class Provenance:
    """Where borrowed code comes from (design §5.6 rule 5)."""

    source: str              # repository URL
    commit: str
    license: str             # SPDX
    files: tuple[str, ...] = ()


_SECTIONS: dict[str, set[str]] = {
    "plugin": {"name", "version", "entry", "slot", "license", "core_abi", "derived_from"},
    "resources": {"provides", "requires"},
    "properties": {"numerics", "after", "before"},
    "dependencies": {"python"},
    "reference": {"module"},
    "support": {"platforms", "detect"},
    "provenance": {"source", "commit", "license", "files"},
    "build": {"module"},
    "options": None,   # any keys (the option names); each value is validated as an Option
}


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class Manifest:
    name: str
    version: str
    entry: str                         # "module:attr", relative to the plugin package; imported only if selected
    slot: str
    license: str
    core_abi: str                      # the plugin ABI versions it works with, e.g. ">=1.0,<2.0" (required)
    derived_from: str | None = None    # copied recipes / explicit old-version copies
    provides: tuple[Resource, ...] = ()
    requires: tuple[Resource, ...] = ()  # kernels: the op contracts they implement
    numerics: Numerics = "exact"
    after: tuple[str, ...] = ()        # ordering for ordered_multi slots
    before: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()  # third-party top-level modules it imports (invariant 2)
    reference: str | None = None       # oracle module of a numeric unit
    platforms: tuple[str, ...] = ()
    detect: tuple[str, ...] = ()       # weight sources: file-name patterns of their checkpoints (e.g. "config.json")
    provenance: Provenance | None = None
    build: str | None = None           # a module of the plugin's package: `python -m <it> --sm N` builds its native
    #                                    code for SM N ahead of time (`holon prebuild`; the CLI is part of the ABI)
    options: dict[str, Option] = dataclasses.field(default_factory=dict)   # the plugin options it reads

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
        abi = data.get("plugin", {}).get("core_abi")    # first: a manifest of another ABI gets this message
        if not isinstance(abi, str) or not abi_allows(abi):
            raise ValueError(f"[plugin] core_abi {abi!r} does not admit this core's plugin ABI {ABI}"
                             if abi else f"[plugin] core_abi missing (this core implements plugin ABI {ABI})")
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
            after=tuple(props.get("after", ())),
            before=tuple(props.get("before", ())),
            dependencies=tuple(data.get("dependencies", {}).get("python", ())),
            reference=data.get("reference", {}).get("module"),
            platforms=tuple(data.get("support", {}).get("platforms", ())),
            detect=tuple(data.get("support", {}).get("detect", ())),
            provenance=_provenance(data["provenance"]) if "provenance" in data else None,
            build=data.get("build", {}).get("module"),
            options={k: Option(**{**v, "choices": tuple(v["choices"]) if isinstance(v.get("choices"), list)
                                  else v.get("choices", ())})
                     for k, v in data.get("options", {}).items()},
        )
        if m.slot not in SLOTS:
            raise ValueError(f"unknown slot {m.slot!r}")
        if m.numerics not in _NUMERICS:
            raise ValueError(f"invalid numerics: {m.numerics!r}")
        return m


def _provenance(v: dict[str, Any]) -> Provenance:
    return Provenance(**{**v, "files": tuple(v.get("files", ()))})
