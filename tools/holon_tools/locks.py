"""Writing the lock history (design §4.3, §4.4, §4.6); plans, reading and diffs are `holon.core.plans`.

Lock files are TOML written canonically by `dumps`: a `digest` of the rest, top-level keys, then `[deploy]`,
`[engine]`, `[snapshot]`, and `[units]` and `[sites]` with one sorted line per entry, so a changed unit or binding is
one changed line in review.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import re
import subprocess
from typing import Any

from holon.api import Lock, Plan, canonical_hash
from holon.core import plans
from holon.core.engine import Engine
from holon.core.plans import CORE_DIR, history, loads, numbered, read


def plan_of(engine: Engine, model_id: str | None = None) -> Plan:
    """`plans.plan_of` at the checked-out commit, under `model_id` (an existing history's id for the build)."""
    return plans.plan_of(engine, model_id, _git(CORE_DIR, "rev-parse", "HEAD") or "unknown")


def uncovered(plan: Plan) -> tuple[str, ...]:
    """Sites whose kernel is not run-to-run deterministic: a change touching them never auto-advances (§7.10)."""
    return tuple(sorted(s for s, b in plan.sites.items() if "deterministic" not in b.config.get("props", ())))


def dumps(lock: Lock) -> str:
    p = lock.plan
    doc: dict[str, Any] = {"plan_id": p.plan_id, "model_id": p.model_id, "fingerprint_class": p.fingerprint_class,
                           "sm": lock.sm, "recipe": p.recipe, "deploy": {"commit": p.commit, "uv_lock": p.uv_lock},
                           "engine": dict(sorted(p.engine.items())),
                           "snapshot": {"trace": lock.snapshot, "uncovered": list(lock.uncovered)}}
    if lock.snapshot is None:
        del doc["snapshot"]["trace"]
    if p.sources:
        doc["sources"] = dict(sorted(p.sources.items()))
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

