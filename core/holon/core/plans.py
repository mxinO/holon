"""Plans and the lock history (design §4.3, §4.4; release design D2): the plan a built engine runs, the lock a
checkpoint starts from, and how they differ. Locks are written only by the tools (`holon_tools`).

A lock lives at `<plans>/<model_id>/<class>/<n>.lock`; the newest `n` is what runs, rollback is the one before.
Model ids and classes may contain "/" (a hub id; `<gpu>/<toolchain>`), which become directories. A lock file is
TOML whose `digest` is the canonical hash of everything else in it and whose `plan_id` is the canonical hash of its
plan; both are checked on read, so a hand edit fails to load.
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import pathlib
import tomllib
import typing
from collections.abc import Iterable, Mapping
from typing import Any

import torch

from holon import __version__
from holon.api import Binding, Fingerprint, Lock, Plan, UnitPin, canonical_hash, parse_versioned

from . import platform
from .engine import EngineConfig
from .plugins import Installed

CORE_UNIT = "holon"          # the core pins its own source like a unit: a core change reaches every plan
CORE_DIR = pathlib.Path(__file__).resolve().parents[1]
TREE = CORE_DIR.parents[1]   # the source tree holding the core: its `plans/` and `uv.lock`
RUNTIME = ("holonlm", "holonlm-std", "holonlm-plugins")     # what a user installs; `uv_lock` hashes their closure
# EngineConfig fields a plan does not record: identified elsewhere (model_id, class, sites), per-host budgets and
# cores (a plugin pins the pool sizes that matter in its own options), or how the deployment was arrived at (pins and
# preferences show as sites; the clamp as `max_model_len`).
NOT_PLANNED = {"model", "device", "kernels", "kernel_preferences", "memory_fraction", "state_memory", "clamp_context",
               "host_memory", "paged_min_tokens", "loop_core"}
# The `src` scheme, written into every pin. Scheme 1 ("sha256:…", locks before it) hashed every file's bytes; a lock
# pinning another scheme is outdated like one pinning other code, and the next propagation re-pins it.
SRC_SCHEME = "src2"
_NOT_SOURCE_DIRS = {"__pycache__", "build", "dist"}              # anywhere, as in scheme 1
_NOT_SOURCE_SUFFIXES = {".pyc", ".pyo", ".so", ".md"}
_NOT_RUNTIME = {"tests", "docs", "examples", "conftest.py"}        # at the unit's top level only, with test_*
_DOCUMENTED = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def src_hash(path: pathlib.Path) -> str:
    """A unit's `src`: `src_digest` of the files under `path`. Content only, so a checkout and an installed copy
    agree."""
    return src_digest((f.relative_to(path).as_posix(), f.read_bytes()) for f in path.rglob("*") if f.is_file())


def src_digest(files: Iterable[tuple[str, bytes]]) -> str:
    """sha256 over the sorted relative paths and contents of a unit's runtime files: everything but hidden files,
    caches, build products, `*.md`, and the unit's own tests, docs and examples (top-level `tests/`, `docs/`,
    `examples/`, `test_*`, `conftest.py`; a vendored tree's are code). Python files count by their syntax tree without
    docstrings, so comments, docstrings and formatting never outdate a lock."""
    h = hashlib.sha256()
    for rel, data in sorted(files):
        if _is_source(pathlib.PurePosixPath(rel)):
            h.update(rel.encode() + b"\0" + (_python(data) if rel.endswith(".py") else data) + b"\0")
    return f"{SRC_SCHEME}:sha256:{h.hexdigest()}"


def _is_source(rel: pathlib.PurePath) -> bool:
    top = rel.parts[0]
    return not (rel.suffix in _NOT_SOURCE_SUFFIXES or top in _NOT_RUNTIME or top.startswith("test_") or any(
        part.startswith(".") or part in _NOT_SOURCE_DIRS or part.endswith(".egg-info") for part in rel.parts))


def _python(data: bytes) -> bytes:
    """A Python file's syntax tree without docstrings, rendered by `_node`; its bytes if it does not parse."""
    try:
        return repr(_node(ast.parse(data))).encode()
    except (SyntaxError, ValueError, RecursionError):
        return data


def _node(node: Any) -> Any:
    """Node type and fields, leaving out locations and empty fields, so the rendering does not depend on the Python
    version adding optional fields (`type_params`, …)."""
    if isinstance(node, list):
        return [_node(x) for x in node]
    if not isinstance(node, ast.AST):
        return repr(node)
    out = [type(node).__name__]
    for key in node._fields:
        if (value := getattr(node, key, None)) is None or value == []:
            continue
        if key == "body" and isinstance(node, _DOCUMENTED) and ast.get_docstring(node, clean=False) is not None:
            value = value[1:]
        out += [key, _node(value)]
    return out


def unit_dirs(installed: Mapping[str, Installed]) -> dict[str, pathlib.Path]:
    """Unit name -> source directory, the core included."""
    return {CORE_UNIT: CORE_DIR} | {name: i.path for name, i in installed.items()}


def current_srcs(installed: Mapping[str, Installed]) -> dict[str, str]:
    """Unit name -> src of the code installed now (for `nearest_class`: a lock pinning other code is outdated)."""
    return {name: src_hash(path) for name, path in unit_dirs(installed).items()}


def plan_of(engine: Any, model_id: str | None = None, commit: str = "unknown") -> Plan:
    """The deployment `engine` runs: checkpoint (`model_id`, default its identity), class, recipe, the code that runs
    (the core, the plugins of the units it built, the kernels bound at its sites, and its single plugins — model,
    weight source, sampler, state kinds …; kernel providers loaded only as candidates are left out), every site's
    kernel binding, and the engine settings that shape outputs."""
    bound = {k["provider"] for k in engine.ctx.kernels.values()}
    runs = {name for name in engine.plugins.loaded if name in bound or name in engine.ctx.used.values()
            or engine.plugins.installed[name].manifest.slot not in ("unit", "kernel")}
    units = {CORE_UNIT: UnitPin(__version__, src_hash(CORE_DIR))}
    units |= {name: unit_pin(engine.plugins.installed[name]) for name in sorted(runs)}
    sites = {key: Binding(k["provider"], {"facts": k["facts"], "props": k["props"]}
                          | ({"launch": k["config"]} if k["config"] else {}))
             for key, k in engine.ctx.kernels.items()}
    settings = {f.name: getattr(engine.cfg, f.name) for f in dataclasses.fields(engine.cfg)
                if f.name not in NOT_PLANNED} | {"source": engine.source.kind}         # resolved, not the setting
    # An option naming another checkpoint (a drafter's) is planned as that checkpoint's identity, not its host path.
    opened = engine.ctx.sources
    sources = {k: opened[str(v)] for k, v in settings["plugin_options"].items() if str(v) in opened}
    unplanned = sources.keys() | unplanned_options(engine.plugins.installed)
    settings["plugin_options"] = {k: v for k, v in settings["plugin_options"].items() if k not in unplanned}
    cls = lock_class(platform.fingerprint(engine.device), engine.ctx.cpu_kernels)
    return Plan(model_id=model_id or engine.source.identity(), fingerprint_class=cls,
                recipe=canonical_hash(engine.spec), commit=commit, uv_lock=uv_lock(TREE / "uv.lock"), units=units,
                sites=sites, engine={k: str(v).removeprefix("torch.") if isinstance(v, torch.dtype) else v
                                     for k, v in settings.items()}, sources=sources)


def lock_class(fp: Fingerprint, cpu_kernels: bool) -> str:
    """The class of a plan's locks: the fingerprint's, and with kernels that run on the CPU (whose results differ by
    instruction set) the CPU's tier inside its first segment, e.g. "rtx-5080+x86-avx2/cu130-torch2.11" (lock discovery
    globs two segments)."""
    return f"{fp.gpu}+{fp.cpu_isa}/{fp.toolchain}" if cpu_kernels and fp.cpu_isa else fp.cls


def unplanned_options(installed: Mapping[str, Installed]) -> set[str]:
    """The plugin options declared only by frontend plugins, which the engine never loads, or `planned = false`: they
    cannot shape outputs, so plans leave them out. An undeclared option stays planned (release design §14.2)."""
    slots: dict[str, set[str | None]] = {}
    for inst in installed.values():
        for key, option in inst.manifest.options.items():
            slots.setdefault(key, set()).add(inst.manifest.slot if option.planned else None)
    return {key for key, found in slots.items() if found <= {"frontend", None}}


def state_identity(engine: Any) -> str | None:
    """What state a tier persisted must have been made under to be reused (docs/v1-kv-tiers.md D11): the plan's id
    without the tier's own options (byte-neutral by tier invariance) and the clamp-dependent `max_model_len`; None if
    the engine's plan cannot be computed."""
    try:
        plan = plan_of(engine)
    except Exception:  # noqa: BLE001 — then nothing persisted is trusted
        return None
    own = engine.plugins.installed[engine.cfg.slots["tier"]].manifest.options
    options = {k: v for k, v in plan.engine["plugin_options"].items() if k not in own}
    return dataclasses.replace(plan, engine=plan.engine | {"plugin_options": options, "max_model_len": None}).plan_id


def unit_pin(inst: Installed) -> UnitPin:
    ref = inst.manifest.reference
    sem = _sha256(_python((inst.path / f"{ref.replace('.', '/')}.py").read_bytes())) if ref else None
    return UnitPin(inst.manifest.version, src_hash(inst.path), sem)


def uv_lock(path: pathlib.Path) -> str:
    """The hash of the runtime closure in `uv.lock`: the sorted (name, version, source) of every package the runtime
    distributions reach, so dev-only pins never move a plan; "none" without a lock file."""
    if not path.is_file():
        return "none"
    packages = {p["name"]: p for p in tomllib.loads(path.read_text())["package"]}
    seen, todo = set(), [{"name": name} for name in RUNTIME]
    while todo:                      # (package, extras) pairs: torch reaches the CUDA libraries through extras
        dep = todo.pop()
        if (key := (dep["name"], tuple(dep.get("extra", ())))) not in seen and dep["name"] in packages:
            seen.add(key)
            package = packages[dep["name"]]
            todo += package.get("dependencies", [])
            todo += [d for e in key[1] for d in package.get("optional-dependencies", {}).get(e, [])]
    return canonical_hash(sorted({(n, packages[n].get("version", ""), str(packages[n]["source"])) for n, _ in seen}))


def diff(a: Plan, b: Plan) -> dict[str, tuple[Any, Any]]:
    """What changed from `a` to `b`: "<field>", "unit:<name>", "site:<key>", "engine:<setting>" or "source:<option>" ->
    (old, new), None where absent."""
    out: dict[str, tuple[Any, Any]] = {f: (getattr(a, f), getattr(b, f))
                                       for f in ("model_id", "fingerprint_class", "recipe", "commit", "uv_lock")
                                       if getattr(a, f) != getattr(b, f)}
    for kind, x, y in (("unit", a.units, b.units), ("site", a.sites, b.sites), ("engine", a.engine, b.engine),
                       ("source", a.sources, b.sources)):
        for key in sorted(x.keys() | y.keys()):
            if key not in x or key not in y or canonical_hash(x[key]) != canonical_hash(y[key]):
                out[f"{kind}:{key}"] = (x.get(key), y.get(key))
    return out


def served(engine: Any, lock: str | None) -> dict[str, Any]:
    """What an engine process reports once ready: its plan id and, started from `lock`, that lock and how the plan
    differs from it (planned under the lock's model id and commit, so only real differences show)."""
    try:
        if lock is None:
            return {"plan_id": plan_of(engine).plan_id, "lock": None, "diff": []}
        want = read(lock).plan
        plan = plan_of(engine, want.model_id, want.commit)
    except Exception as e:  # noqa: BLE001 — a report never stops serving
        return {"plan_id": None, "lock": lock, "diff": [f"plan not computed: {type(e).__name__}: {e}"]}
    return {"plan_id": plan.plan_id, "lock": str(lock), "diff": [
        f"{k}: {old} -> {new}" if k.startswith("engine:") else k for k, (old, new) in diff(want, plan).items()]}


def lock_overrides(lock: Lock) -> dict[str, Any]:
    """EngineConfig overrides that rebuild the lock's deployment on its checkpoint: the engine settings and a kernel
    preference per op (site keys are `<site>/<op>`; an op bound to two providers is left to default priority)."""
    pins: dict[str, str | None] = {}
    for key, binding in lock.plan.sites.items():
        op = parse_versioned(key.rpartition("/")[2])[0]
        if pins.setdefault(op, binding.impl) != binding.impl:
            pins[op] = None
    hints = typing.get_type_hints(EngineConfig)
    settings = {k: getattr(torch, v) if hints.get(k) is torch.dtype else v for k, v in lock.plan.engine.items()}
    return settings | {"kernel_preferences": {op: p for op, p in pins.items() if p is not None}}


def loads(text: str) -> Lock:
    d = tomllib.loads(text)
    digest = d.pop("digest")
    if canonical_hash(d) != digest:
        raise ValueError(f"digest {digest} does not match the content ({canonical_hash(d)}): edited by hand?")
    plan = Plan(model_id=d["model_id"], fingerprint_class=d["fingerprint_class"], recipe=d["recipe"],
                commit=d["deploy"]["commit"], uv_lock=d["deploy"]["uv_lock"],
                units={k: UnitPin(**v) for k, v in d["units"].items()},
                sites={k: Binding(**v) for k, v in d["sites"].items()}, engine=d["engine"],
                sources=d.get("sources", {}))
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


def model_ids(plans: str | pathlib.Path, build: str | None = None) -> list[str]:
    """The model ids with a lock history under `plans` (`<model_id>/<gpu>/<toolchain>/<n>.lock`); with `build`, those
    of that checkpoint build (`<name>@<build>`, whatever name its file was found under)."""
    root = pathlib.Path(plans)
    locks = (p.relative_to(root) for p in root.rglob("*.lock"))
    ids = {p.parents[2].as_posix() for p in locks if len(p.parts) > 3}
    return sorted(i for i in ids if build is None or i.endswith(f"@{build}"))


def history_id(plans: str | pathlib.Path, identity: str) -> str:
    """The model id the checkpoint `identity` (`<name>@<build>`) has lock histories under, found by its build alone —
    the same file downloaded elsewhere or renamed finds them —, else `identity` (a model without locks yet)."""
    found = model_ids(plans, identity.rpartition("@")[2])
    if len(found) > 1:
        raise ValueError(f"{plans}: one checkpoint build has lock histories under several model ids: {found}")
    return found[0] if found else identity


def find(plans: str | pathlib.Path, identity: str, fp: Fingerprint, current: Mapping[str, str] | None = None
         ) -> tuple[Lock, bool, pathlib.Path] | None:
    """The lock the checkpoint `identity` starts from: `nearest_class` of its history (`history_id`)."""
    return nearest_class(plans, history_id(plans, identity), fp, current)


def nearest_class(plans: str | pathlib.Path, model_id: str, fp: Fingerprint,
                  current: Mapping[str, str] | None = None) -> tuple[Lock, bool, pathlib.Path] | None:
    """The lock to start from, whether it is `unverified` (borrowed from another class), and its file: this class
    first — the CPU's tier's (`lock_class`) before the one without a tier, both verified —, then the same GPU and
    toolchain on another tier, the same GPU on another toolchain, the same SM, the same toolchain. With `current` (unit
    -> src installed now) a lock pinning other code is outdated and treated as missing (design §4.4). None: no usable
    lock."""
    def usable(lock: Lock) -> bool:
        return current is None or all(current.get(name) == pin.src for name, pin in lock.plan.units.items())

    best = None
    root = pathlib.Path(plans, model_id)
    for cls in sorted({p.parent.relative_to(root).as_posix() for p in root.glob("*/*/*.lock")}):
        found = history(plans, model_id, cls)
        if not found or not usable(lock := read(path := found[-1])):
            continue
        gpu, _, toolchain = cls.partition("/")
        gpu, _, tier = gpu.partition("+")
        own = gpu == fp.gpu and toolchain == fp.toolchain
        rank = (own and tier in ("", fp.cpu_isa), gpu == fp.gpu, lock.sm == fp.sm, toolchain == fp.toolchain,
                tier == fp.cpu_isa)
        if best is None or rank > best[0]:
            best = (rank, lock, path)
    return None if best is None else (best[1], not best[0][0], best[2])


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()
