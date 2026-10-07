"""Plans of built engines and the lock history (design §4.3, §4.4, §4.6).

A lock lives at `<plans>/<model_id>/<class>/<n>.lock`; the newest `n` is what runs, rollback is the one before.
Model ids and classes may contain "/" (a hub id; `<gpu>/<toolchain>`), which become directories.

Lock files are TOML (read with stdlib `tomllib`), written canonically by `dumps`: top-level keys, then `[deploy]`,
`[engine]`, `[snapshot]`, and `[units]` and `[sites]` with one sorted line per entry, so a changed unit or binding is
one changed line in review. `plan_id` is stored and re-checked on read, which rejects hand edits.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import re
import subprocess
import tomllib
import typing
from collections.abc import Mapping
from typing import Any

import torch

from holon import __version__
from holon.api import Binding, Fingerprint, Lock, Plan, UnitPin, canonical_hash, parse_versioned

from . import platform
from .engine import Engine, EngineConfig, resolve_model
from .plugins import Installed

CORE_UNIT = "holon"          # the core pins its own source like a unit: a core change reaches every plan
_CORE_DIR = pathlib.Path(__file__).resolve().parents[1]
# EngineConfig fields a plan does not record: identified elsewhere (model_id, class, sites) or per-host budgets.
_NOT_PLANNED = {"model", "device", "kernels", "memory_fraction", "state_memory"}


# -- the plan of a built engine ---------------------------------------------------------------------------------------

def plan_of(engine: Engine) -> Plan:
    """The deployment `engine` runs: checkpoint, class, recipe, code (the core and every loaded plugin), the kernel
    bound at every site and the engine settings that shape outputs."""
    source, _ = resolve_model(engine.plugins, engine.cfg)
    root = _git(_CORE_DIR, "rev-parse", "--show-toplevel")
    uv_lock = pathlib.Path(root, "uv.lock") if root else None
    units = {CORE_UNIT: UnitPin(__version__, src_hash(_CORE_DIR))}
    units |= {name: unit_pin(engine.plugins.installed[name]) for name in engine.plugins.loaded}
    sites = {key: Binding(k["provider"], {"facts": k["facts"], "props": k["props"]}
                          | ({"launch": k["config"]} if k["config"] else {}))
             for key, k in engine.ctx.kernels.items()}
    settings = {f.name: getattr(engine.cfg, f.name) for f in dataclasses.fields(engine.cfg)
                if f.name not in _NOT_PLANNED}
    return Plan(model_id=source.identity(), fingerprint_class=platform.fingerprint(engine.device).cls,
                recipe=canonical_hash(engine.spec), commit=_git(_CORE_DIR, "rev-parse", "HEAD") or "unknown",
                uv_lock=_sha256(uv_lock.read_bytes()) if uv_lock and uv_lock.is_file() else "none",
                units=units, sites=sites, engine={k: str(v).removeprefix("torch.") if isinstance(v, torch.dtype)
                                                  else v for k, v in settings.items()})


def unit_pin(inst: Installed) -> UnitPin:
    ref = inst.manifest.reference
    sem = _sha256((inst.path / f"{ref.replace('.', '/')}.py").read_bytes()) if ref else None
    return UnitPin(inst.manifest.version, src_hash(inst.path), sem)


def src_hash(path: pathlib.Path) -> str:
    """The git tree hash of directory `path` when it matches HEAD there; else (uncommitted edits, or code installed
    outside git) a content hash — a pin never names code other than what runs."""
    if _git(path, "status", "--porcelain", "--", ".") == "" and (tree := _git(path, "rev-parse", "HEAD:./")):
        return tree
    h = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file() and not _hidden(p.relative_to(path))):
        h.update(f.relative_to(path).as_posix().encode() + b"\0" + f.read_bytes() + b"\0")
    return "sha256:" + h.hexdigest()


def unit_dirs(installed: Mapping[str, Installed]) -> dict[str, pathlib.Path]:
    """Unit name -> source directory, the core included."""
    return {CORE_UNIT: _CORE_DIR} | {name: i.path for name, i in installed.items()}


def current_srcs(installed: Mapping[str, Installed]) -> dict[str, str]:
    """Unit name -> src of the code installed now (for `nearest_class`: a lock pinning other code is outdated)."""
    return {name: src_hash(path) for name, path in unit_dirs(installed).items()}


def uncovered(plan: Plan) -> tuple[str, ...]:
    """Sites whose kernel is not run-to-run deterministic: a change touching them never auto-advances (§7.10)."""
    return tuple(sorted(s for s, b in plan.sites.items() if "deterministic" not in b.config.get("props", ())))


def diff(a: Plan, b: Plan) -> dict[str, tuple[Any, Any]]:
    """What changed from `a` to `b`: "<field>", "unit:<name>", "site:<key>" or "engine:<setting>" -> (old, new),
    None where absent."""
    out: dict[str, tuple[Any, Any]] = {f: (getattr(a, f), getattr(b, f))
                                       for f in ("model_id", "fingerprint_class", "recipe", "commit", "uv_lock")
                                       if getattr(a, f) != getattr(b, f)}
    for kind, x, y in (("unit", a.units, b.units), ("site", a.sites, b.sites), ("engine", a.engine, b.engine)):
        for key in sorted(x.keys() | y.keys()):
            if key not in x or key not in y or canonical_hash(x[key]) != canonical_hash(y[key]):
                out[f"{kind}:{key}"] = (x.get(key), y.get(key))
    return out


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


# -- lock files -------------------------------------------------------------------------------------------------------

def dumps(lock: Lock) -> str:
    p = lock.plan
    doc: dict[str, Any] = {"plan_id": p.plan_id, "model_id": p.model_id, "fingerprint_class": p.fingerprint_class,
                           "sm": lock.sm, "recipe": p.recipe, "deploy": {"commit": p.commit, "uv_lock": p.uv_lock},
                           "engine": dict(sorted(p.engine.items())),
                           "snapshot": {"trace": lock.snapshot, "uncovered": list(lock.uncovered)}}
    if lock.snapshot is None:
        del doc["snapshot"]["trace"]
    doc["units"] = {k: {f: v for f, v in dataclasses.asdict(u).items() if v is not None}
                    for k, u in sorted(p.units.items())}
    doc["sites"] = {k: {f: v for f, v in dataclasses.asdict(b).items() if v} | {"impl": b.impl}
                    for k, b in sorted(p.sites.items())}
    lines = ["# Written by the holon tools; never edit by hand (AGENTS.md)."]
    lines += [f"{_key(k)} = {_inline(v)}" for k, v in doc.items() if not isinstance(v, dict)]
    for table, body in ((k, v) for k, v in doc.items() if isinstance(v, dict)):
        lines += ["", f"[{table}]", *(f"{_key(k)} = {_inline(v)}" for k, v in body.items())]
    text = "\n".join(lines) + "\n"
    # The writer covers the values locks hold (JSON-like terms); anything it would mangle must fail here.
    if tomllib.loads(text) != json.loads(json.dumps(doc)):
        raise ValueError("lock does not round-trip through TOML")
    return text


def loads(text: str) -> Lock:
    d = tomllib.loads(text)
    plan = Plan(model_id=d["model_id"], fingerprint_class=d["fingerprint_class"], recipe=d["recipe"],
                commit=d["deploy"]["commit"], uv_lock=d["deploy"]["uv_lock"],
                units={k: UnitPin(**v) for k, v in d["units"].items()},
                sites={k: Binding(**v) for k, v in d["sites"].items()}, engine=d["engine"])
    if plan.plan_id != d["plan_id"]:
        raise ValueError(f"plan_id {d['plan_id']} does not match the content ({plan.plan_id}): edited by hand?")
    return Lock(plan, d["snapshot"].get("trace"), tuple(d["snapshot"]["uncovered"]), d["sm"])


def read(path: str | pathlib.Path) -> Lock:
    try:
        return loads(pathlib.Path(path).read_text())
    except (KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as e:
        raise ValueError(f"{path}: {e}") from None


def _key(k: str) -> str:
    return k if re.fullmatch(r"[A-Za-z0-9_-]+", k) else json.dumps(k, ensure_ascii=False)


def _inline(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(map(_inline, v)) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{_key(k)} = {_inline(x)}" for k, x in sorted(v.items())) + " }" if v else "{}"
    raise TypeError(f"cannot write {type(v).__name__} into a lock")


# -- history ----------------------------------------------------------------------------------------------------------

def history(plans: str | pathlib.Path, model_id: str, cls: str) -> list[pathlib.Path]:
    """Lock files of one (model, class), oldest first."""
    if any(part in ("", ".", "..") for part in f"{model_id}/{cls}".split("/")):
        raise ValueError(f"not a model id and class: {model_id!r}, {cls!r}")
    return _numbered(pathlib.Path(plans, model_id, cls))


def newest(plans: str | pathlib.Path, model_id: str, cls: str) -> Lock | None:
    found = history(plans, model_id, cls)
    return read(found[-1]) if found else None


def append(plans: str | pathlib.Path, lock: Lock) -> pathlib.Path:
    """Write `lock` as the next entry of its history."""
    found = history(plans, lock.plan.model_id, lock.plan.fingerprint_class)
    path = pathlib.Path(plans, lock.plan.model_id, lock.plan.fingerprint_class,
                        f"{int(found[-1].stem) + 1 if found else 1}.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:                 # exclusive: a concurrent writer fails instead of overwriting
        f.write(dumps(lock))
    return path


def heads(plans: str | pathlib.Path) -> list[Lock]:
    """The newest lock of every history under `plans`."""
    dirs = sorted({p.parent for p in pathlib.Path(plans).rglob("*.lock")})
    return [read(found[-1]) for d in dirs if (found := _numbered(d))]


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


# -- helpers ----------------------------------------------------------------------------------------------------------

def _numbered(d: pathlib.Path) -> list[pathlib.Path]:
    return sorted((p for p in d.glob("*.lock") if p.stem.isdigit()), key=lambda p: int(p.stem))


def _git(cwd: pathlib.Path, *args: str) -> str | None:
    """Output of a git command in `cwd`; None outside a repository or without git."""
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _hidden(rel: pathlib.PurePath) -> bool:
    return any(part.startswith(".") or part == "__pycache__" for part in rel.parts)


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()
