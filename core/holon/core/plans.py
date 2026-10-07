"""Reading the lock history (design §4.3, §4.4): what an engine needs to start from a lock. Locks are written only by
the tools (`holon_tools`), which also compute plans.

A lock lives at `<plans>/<model_id>/<class>/<n>.lock`; the newest `n` is what runs, rollback is the one before.
Model ids and classes may contain "/" (a hub id; `<gpu>/<toolchain>`), which become directories. A lock file is
TOML whose `digest` is the canonical hash of everything else in it and whose `plan_id` is the canonical hash of its
plan; both are checked on read, so a hand edit fails to load.
"""
from __future__ import annotations

import hashlib
import pathlib
import tomllib
import typing
from collections.abc import Mapping
from typing import Any

import torch

from holon.api import Binding, Fingerprint, Lock, Plan, UnitPin, canonical_hash, parse_versioned

from .engine import EngineConfig
from .plugins import Installed

CORE_UNIT = "holon"          # the core pins its own source like a unit: a core change reaches every plan
CORE_DIR = pathlib.Path(__file__).resolve().parents[1]
_NOT_SOURCE_DIRS = {"__pycache__", "build", "dist"}
_NOT_SOURCE_SUFFIXES = {".pyc", ".pyo", ".so"}


def src_hash(path: pathlib.Path) -> str:
    """A unit's `src`: sha256 over the sorted relative paths and contents of its source files — everything under
    `path` but hidden files, caches and build products. Content only, so a checkout and an installed copy agree."""
    h = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file() and _is_source(p.relative_to(path))):
        h.update(f.relative_to(path).as_posix().encode() + b"\0" + f.read_bytes() + b"\0")
    return "sha256:" + h.hexdigest()


def _is_source(rel: pathlib.PurePath) -> bool:
    return rel.suffix not in _NOT_SOURCE_SUFFIXES and not any(
        part.startswith(".") or part in _NOT_SOURCE_DIRS or part.endswith(".egg-info") for part in rel.parts)


def unit_dirs(installed: Mapping[str, Installed]) -> dict[str, pathlib.Path]:
    """Unit name -> source directory, the core included."""
    return {CORE_UNIT: CORE_DIR} | {name: i.path for name, i in installed.items()}


def current_srcs(installed: Mapping[str, Installed]) -> dict[str, str]:
    """Unit name -> src of the code installed now (for `nearest_class`: a lock pinning other code is outdated)."""
    return {name: src_hash(path) for name, path in unit_dirs(installed).items()}


def lock_overrides(lock: Lock) -> dict[str, Any]:
    """EngineConfig overrides that rebuild the lock's deployment on its checkpoint: the engine settings and a kernel
    pin per op (site keys are `<site>/<op>`)."""
    pins: dict[str, str] = {}
    for key, binding in lock.plan.sites.items():
        op = parse_versioned(key.rpartition("/")[2])[0]
        if pins.setdefault(op, binding.impl) != binding.impl:
            raise ValueError(f"lock binds {op} to both {pins[op]} and {binding.impl}; engine pins are per op")
    hints = typing.get_type_hints(EngineConfig)
    settings = {k: getattr(torch, v) if hints.get(k) is torch.dtype else v for k, v in lock.plan.engine.items()}
    return settings | {"kernels": pins}


def loads(text: str) -> Lock:
    d = tomllib.loads(text)
    digest = d.pop("digest")
    if canonical_hash(d) != digest:
        raise ValueError(f"digest {digest} does not match the content ({canonical_hash(d)}): edited by hand?")
    plan = Plan(model_id=d["model_id"], fingerprint_class=d["fingerprint_class"], recipe=d["recipe"],
                commit=d["deploy"]["commit"], uv_lock=d["deploy"]["uv_lock"],
                units={k: UnitPin(**v) for k, v in d["units"].items()},
                sites={k: Binding(**v) for k, v in d["sites"].items()}, engine=d["engine"])
    if plan.plan_id != d["plan_id"]:
        raise ValueError(f"plan_id {d['plan_id']} does not match the plan ({plan.plan_id})")
    return Lock(plan, d["snapshot"].get("trace"), tuple(d["snapshot"]["uncovered"]), d["sm"])


def read(path: str | pathlib.Path) -> Lock:
    try:
        return loads(pathlib.Path(path).read_text())
    except (KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as e:
        raise ValueError(f"{path}: {e}") from None


def history(plans: str | pathlib.Path, model_id: str, cls: str) -> list[pathlib.Path]:
    """Lock files of one (model, class), oldest first."""
    if any(part in ("", ".", "..") for part in f"{model_id}/{cls}".split("/")):
        raise ValueError(f"not a model id and class: {model_id!r}, {cls!r}")
    return numbered(pathlib.Path(plans, model_id, cls))


def numbered(d: pathlib.Path) -> list[pathlib.Path]:
    """The `<n>.lock` files of one history directory, by `n`."""
    return sorted((p for p in d.glob("*.lock") if p.stem.isdigit()), key=lambda p: int(p.stem))


def newest(plans: str | pathlib.Path, model_id: str, cls: str) -> Lock | None:
    found = history(plans, model_id, cls)
    return read(found[-1]) if found else None


def nearest_class(plans: str | pathlib.Path, model_id: str, fp: Fingerprint,
                  current: Mapping[str, str] | None = None) -> tuple[Lock, bool] | None:
    """The lock to start from, and whether it is `unverified` (borrowed from another class): this class first, then
    the same GPU on another toolchain, the same SM, the same toolchain. With `current` (unit -> src installed now) a
    lock pinning other code is outdated and treated as missing (design §4.4). None: no usable lock."""
    def usable(lock: Lock) -> bool:
        return current is None or all(current.get(name) == pin.src for name, pin in lock.plan.units.items())

    best = None
    root = pathlib.Path(plans, model_id)
    for cls in sorted({p.parent.relative_to(root).as_posix() for p in root.glob("*/*/*.lock")}):
        lock = newest(plans, model_id, cls)
        if lock is None or not usable(lock):
            continue
        gpu, _, toolchain = cls.partition("/")
        rank = (cls == fp.cls, gpu == fp.gpu, lock.sm == fp.sm, toolchain == fp.toolchain)
        if best is None or rank > best[0]:
            best = (rank, lock)
    return None if best is None else (best[1], not best[0][0])
