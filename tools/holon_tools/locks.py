"""Plans of built engines and writing the lock history (design §4.3, §4.4, §4.6); reading is `holon.core.plans`.

Lock files are TOML written canonically by `dumps`: a `digest` of the rest, top-level keys, then `[deploy]`,
`[engine]`, `[snapshot]`, and `[units]` and `[sites]` with one sorted line per entry, so a changed unit or binding is
one changed line in review.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import re
import subprocess
from typing import Any

import torch
from holon import __version__
from holon.api import Binding, Lock, Plan, UnitPin, canonical_hash
from holon.core import platform
from holon.core.engine import Engine
from holon.core.plans import CORE_DIR, CORE_UNIT, history, loads, numbered, read, src_hash
from holon.core.plugins import Installed

# EngineConfig fields a plan does not record: identified elsewhere (model_id, class, sites) or per-host budgets.
_NOT_PLANNED = {"model", "device", "kernels", "memory_fraction", "state_memory"}


def plan_of(engine: Engine) -> Plan:
    """The deployment `engine` runs: checkpoint, class, recipe, the code that runs (the core, the plugins of the
    units it built, the kernels bound at its sites, and its single plugins — model, weight source, sampler, state
    kinds …; kernel providers loaded only as candidates are left out), every site's kernel binding, and the engine
    settings that shape outputs."""
    source = engine.source
    root = _git(CORE_DIR, "rev-parse", "--show-toplevel")
    uv_lock = pathlib.Path(root, "uv.lock") if root else None
    bound = {k["provider"] for k in engine.ctx.kernels.values()}
    runs = {name for name in engine.plugins.loaded if name in bound or name in engine.ctx.used.values()
            or engine.plugins.installed[name].manifest.slot not in ("unit", "kernel")}
    if hidden := sorted(n for n in runs if engine.plugins.installed[n].manifest.slot == "weight_source"
                        and not engine.plugins.installed[n].manifest.detect):
        # A lock serves the checkpoints the engine detects; a source never detected (a judge fixture) never serves.
        raise ValueError(f"{hidden[0]}: a weight source without detect patterns cannot be planned")
    units = {CORE_UNIT: UnitPin(__version__, src_hash(CORE_DIR))}
    units |= {name: unit_pin(engine.plugins.installed[name]) for name in sorted(runs)}
    sites = {key: Binding(k["provider"], {"facts": k["facts"], "props": k["props"]}
                          | ({"launch": k["config"]} if k["config"] else {}))
             for key, k in engine.ctx.kernels.items()}
    settings = {f.name: getattr(engine.cfg, f.name) for f in dataclasses.fields(engine.cfg)
                if f.name not in _NOT_PLANNED} | {"source": source.kind}         # resolved, not the setting
    return Plan(model_id=source.identity(), fingerprint_class=platform.fingerprint(engine.device).cls,
                recipe=canonical_hash(engine.spec), commit=_git(CORE_DIR, "rev-parse", "HEAD") or "unknown",
                uv_lock=_sha256(uv_lock.read_bytes()) if uv_lock and uv_lock.is_file() else "none",
                units=units, sites=sites, engine={k: str(v).removeprefix("torch.") if isinstance(v, torch.dtype)
                                                  else v for k, v in settings.items()})


def unit_pin(inst: Installed) -> UnitPin:
    ref = inst.manifest.reference
    sem = _sha256((inst.path / f"{ref.replace('.', '/')}.py").read_bytes()) if ref else None
    return UnitPin(inst.manifest.version, src_hash(inst.path), sem)


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
    doc = {"digest": canonical_hash(doc)} | doc
    lines = ["# Written by the holon tools; never edit by hand (AGENTS.md)."]
    lines += [f"{_key(k)} = {_inline(v)}" for k, v in doc.items() if not isinstance(v, dict)]
    for table, body in ((k, v) for k, v in doc.items() if isinstance(v, dict)):
        lines += ["", f"[{table}]", *(f"{_key(k)} = {_inline(v)}" for k, v in body.items())]
    text = "\n".join(lines) + "\n"
    loads(text)      # the digest check: anything this writer would mangle fails here, not when the lock is read
    return text


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
    return [read(found[-1]) for d in dirs if (found := numbered(d))]


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


def _git(cwd: pathlib.Path, *args: str) -> str | None:
    """Output of a git command in `cwd`; None outside a repository or without git."""
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()
