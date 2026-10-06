"""Canonical terms and hashing (design §4.6, §5.1).

Recipes emit plain data — `Ref`s, dataclasses, tuples, dicts, scalars. Identity is the hash of the *emitted term*:
dict keys and sets are ordered canonically, and dataclass fields left at their default are omitted, so refactoring a
recipe — or adding a defaulted field to a type (an additive minor) — never changes an existing identity.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any


def parse_versioned(text: str) -> tuple[str, int]:
    """Parse `name@major` (e.g. `std.mixers.gqa@1`)."""
    name, sep, major = text.partition("@")
    if not name or not sep or not major.isdigit():
        raise ValueError(f"expected 'name@major', got {text!r}")
    return name, int(major)


@dataclasses.dataclass(frozen=True, slots=True, init=False, eq=False)
class Ref:
    """A reference to a unit by name and interface major, plus its constructor parameters.

    `Ref("std.mixers.gqa@1", heads=32, kv_heads=8)`. Parameters are stored sorted, so keyword order never affects
    identity, and must be canonical terms. Use `with_` to derive a modified copy (`dataclasses.replace` is not
    supported).
    """

    name: str
    major: int
    params: tuple[tuple[str, Any], ...]
    _key: str = dataclasses.field(repr=False, compare=False)

    def __init__(self, target: str, /, **params: Any) -> None:
        name, major = parse_versioned(target)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "major", major)
        object.__setattr__(self, "params", tuple(sorted(params.items())))
        try:  # the canonical key validates the parameters once and serves equality and hashing
            object.__setattr__(self, "_key", _dumps(to_term(self)))
        except TypeError as e:
            raise ValueError(f"Ref {target!r}: parameters must be canonical terms ({e})") from None

    @property
    def target(self) -> str:
        return f"{self.name}@{self.major}"

    def param(self, key: str, default: Any = None) -> Any:
        return dict(self.params).get(key, default)

    def with_(self, **params: Any) -> Ref:
        return Ref(self.target, **{**dict(self.params), **params})

    # Equality follows identity: Ref(x=1) and Ref(x=True) are different terms.
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Ref) and self._key == other._key

    def __hash__(self) -> int:
        return hash(self._key)


def _is_default(f: dataclasses.Field, value: Any) -> bool:
    if f.default is not dataclasses.MISSING:
        return value == f.default
    if f.default_factory is not dataclasses.MISSING:
        return value == f.default_factory()
    return False


def to_term(obj: Any) -> Any:
    """Convert a value into a JSON-compatible canonical term."""
    if isinstance(obj, Ref):
        return {"ref": obj.target, "params": {k: to_term(v) for k, v in obj.params}}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        fields = {f.name: to_term(getattr(obj, f.name)) for f in dataclasses.fields(obj)
                  if not _is_default(f, getattr(obj, f.name))}
        return {"type": type(obj).__name__, **fields}
    if isinstance(obj, dict):
        if not all(isinstance(k, str) for k in obj):
            raise TypeError("canonical dict keys must be str")
        return {k: to_term(obj[k]) for k in sorted(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_term(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted((to_term(v) for v in obj), key=_dumps)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    raise TypeError(f"not a canonical term: {type(obj).__name__}")


def _dumps(term: Any) -> str:
    return json.dumps(term, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(obj: Any) -> str:
    """`sha256:<hex>` of the canonical JSON encoding of `obj`."""
    return "sha256:" + hashlib.sha256(_dumps(to_term(obj)).encode()).hexdigest()
