"""The plugin ABI's listing (release design §7): what plugins may rely on, rendered from the code and compared with
`core/abi.txt`. A difference fails unless the listing is regenerated (`--write`) with `holon.api.ABI` bumped: an
addition needs a minor bump, a removal or change a major one (below 1.0: the minor number, as manifests' ranges read
it). Also fails when the core looks up a hook by name (`hasattr` / `getattr` with a literal) that no Protocol of
`holon.api` declares.

Listed: every public name of `holon.api` (dataclass fields in order with types and defaults; Protocol members with
their signatures, required or optional, and who implements the Protocol), the optional hooks (`holon.api.hooks`),
`EngineConfig`, the `BuildCtx.options` keys the core provides, the manifest schema, the string vocabularies (the
API's and the IPC frame and failure kinds), and the `op.*` / `unit.*` contracts first-party plugins provide.
Run in the project environment."""
from __future__ import annotations

import ast
import dataclasses
import enum
import inspect
import sys
import tomllib
import typing

import holon.api as api
from _common import CORE, ROOT, fail
from holon.api import hooks, manifest
from holon.core import engine, ipc

LISTING = ROOT / "core" / "abi.txt"
# Protocols the core implements and plugins call (the rest are implemented by plugins).
CORE_IMPLEMENTS = {"BuildCtx", "EngineClient", "KernelRegistry", "Ledger", "Pool", "PrefixView",
                   "Registries", "Registry", "SeqView", "StepCtx", "TargetView"}
OWN_ATTRIBUTES = {"parent"}   # looked up on the core's own objects (a child build context's parent), not on plugins


def _members(cls: type) -> list[str]:
    out = []
    for name, value in cls.__dict__.items():
        if name.startswith("_") and name != "__call__":
            continue
        func = value.__func__ if isinstance(value, (classmethod, staticmethod)) else value
        if callable(func):
            prefix = "classmethod " if isinstance(value, classmethod) else ""
            out.append(f"{prefix}{name}{inspect.signature(func)}")
        elif isinstance(value, property):
            out.append(f"property {name}")
    out += [f"{name}: {kind}" for name, kind in cls.__dict__.get("__annotations__", {}).items()
            if not name.startswith("_")]
    return out


def render(name: str, obj: object, optional: bool = False) -> list[str]:
    if dataclasses.is_dataclass(obj):
        params = obj.__dataclass_params__
        head = f"dataclass {name}" + (" frozen" if params.frozen else "")
        fields = [f"  {f.name}: {f.type}" + ("" if f.default is dataclasses.MISSING and f.default_factory is
                                             dataclasses.MISSING else f" = {_default(f)}")
                  for f in dataclasses.fields(obj)]
        return [head, *fields, *(f"  {m}" for m in _members(obj) if "(" in m)]
    if isinstance(obj, type) and issubclass(obj, enum.Enum):
        return [f"enum {name}", *(f"  {m.name} = {m.value!r}" for m in obj)]
    if isinstance(obj, type) and getattr(obj, "_is_protocol", False):
        bases = [b.__name__ for b in obj.__bases__ if b is not typing.Protocol and b.__name__ != "Generic"]
        mark = "optional" if optional else "required"
        who = "core" if name in CORE_IMPLEMENTS else "plugins"
        return [f"protocol {name}({', '.join(bases)}) {mark}, implemented by {who}", *(f"  {m}" for m in _members(obj))]
    if isinstance(obj, type):
        return [f"class {name}({', '.join(b.__name__ for b in obj.__bases__)})", *(f"  {m}" for m in _members(obj))]
    if callable(obj):
        return [f"def {name}{inspect.signature(obj)}"]
    value = dict(obj) if isinstance(obj, typing.Mapping) else sorted(obj) if isinstance(obj, frozenset) else obj
    return [f"const {name} = {value!r}"]


def _default(f: dataclasses.Field) -> str:
    return f"{f.default_factory.__name__}()" if f.default is dataclasses.MISSING else repr(f.default)


def options_keys() -> list[str]:
    """String keys of the dict literals the core builds `BuildCtx.options` from (`options = {…}`, `… | {…}`)."""
    keys = set()
    for path in (CORE / "core").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            literal = None
            if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "options" for t in node.targets):
                literal = node.value
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr) and \
                    getattr(node.left, "id", "") == "options":
                literal = node.right
            if isinstance(literal, ast.Dict):
                keys |= {k.value for k in literal.keys if isinstance(k, ast.Constant)}
    return sorted(keys)


def contracts() -> list[str]:
    out = set()
    for path in (*ROOT.glob("std/holon_std/**/plugin.toml"), *ROOT.glob("plugins/holon_plugins/**/plugin.toml")):
        m = tomllib.loads(path.read_text())
        out |= {f"{r}  ({m['plugin']['name']})" for r in m.get("resources", {}).get("provides", ())
                if r.startswith(("op.", "unit."))}
    return sorted(out)


def listing() -> str:
    lines = [f"abi {api.ABI}", "", "# holon.api"]
    for name in sorted(api.__all__):
        if name != "ABI" and getattr(obj := getattr(api, name), "__module__", "") != hooks.__name__:  # hooks: below
            lines += render(name, obj)
    lines += ["", "# holon.api.hooks (optional)"]
    for name, obj in sorted(vars(hooks).items()):
        if isinstance(obj, type) and obj.__module__ == hooks.__name__:
            lines += render(name, obj, optional=True)
    lines += ["", "# EngineConfig", *render("EngineConfig", engine.EngineConfig)]
    lines += ["", "# BuildCtx.options keys the core provides", *options_keys()]
    lines += ["", "# manifest sections and keys",
              *(f"[{s}] {sorted(keys) if keys else '<any>'}" for s, keys in manifest._SECTIONS.items())]
    lines += ["", "# string vocabularies",
              *(f"{name} = {sorted(getattr(manifest, name).__args__)}" for name in ("Numerics", "GraphPolicy")),
              f"ipc frames = {sorted(k.decode() for k in ipc._DECODE)}",
              f"ipc failed kinds = {sorted({*ipc.FAILED_KINDS.values(), 'error'})}"]
    lines += ["", "# op and unit contracts provided by first-party plugins", *contracts()]
    return "\n".join(lines) + "\n"


def hook_names() -> list[str]:
    """`hasattr` / `getattr` literals in the core that no holon.api Protocol declares."""
    declared = set(OWN_ATTRIBUTES)
    for module in (api, hooks):
        for obj in vars(module).values():
            if isinstance(obj, type) and getattr(obj, "_is_protocol", False):
                declared |= set(obj.__dict__) | set(obj.__dict__.get("__annotations__", {}))
    out = []
    for path in sorted(CORE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") in ("hasattr", "getattr") and \
                    len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                if (name := node.args[1].value) not in declared:
                    out.append(f"{path.relative_to(ROOT)}:{node.lineno}: {name!r} is not declared in holon.api")
    return out


def _version(listing: str) -> tuple[int, ...]:
    return (*map(int, listing.split("\n", 1)[0].split()[1].split(".")), 0, 0)[:3]


def _entries(listing: str) -> set[tuple[str, str]]:
    """Each line of a listing with the item it belongs to (its last unindented line)."""
    out, head = set(), ""
    for line in listing.splitlines()[1:]:
        head = line if not line.startswith("  ") else head
        out.add((head, line))
    return out


def bump_error(old: str, new: str) -> str | None:
    """Why the listing `new` may not replace the committed `old` under its version, or None. A removal or change, or a
    member added to a required Protocol plugins implement (plugins of the old ABI lack it), needs a major bump (below
    1.0: the minor number, as manifests' ranges read it); any other addition a minor one."""
    was, now, before, after = _version(old), _version(new), _entries(old), _entries(new)
    breaking = sorted(line for _, line in before - after) + sorted(
        line for head, line in after - before if line != head and "required, implemented by plugins" in head
        and (head, head) in before)
    major = now[0] > was[0] or (was[0] == 0 and now[:2] > was[:2])
    if breaking and not major:
        return f"a removal, change or required plugin member needs a major ABI bump (from {old.split()[1]}): " \
               f"{breaking[:5]}"
    if not breaking and now <= was:
        return f"an addition needs a minor ABI bump (from {old.split()[1]})"
    return None


def main() -> None:
    new = listing()
    msgs = hook_names()
    if "--write" in sys.argv:
        if LISTING.exists() and LISTING.read_text() != new and (why := bump_error(LISTING.read_text(), new)):
            fail([why], "ABI listing written")
        LISTING.write_text(new)
    elif not LISTING.exists() or LISTING.read_text() != new:
        old = LISTING.read_text().splitlines() if LISTING.exists() else []
        msgs += [f"- {line}" for line in old if line not in new.splitlines()][:20]
        msgs += [f"+ {line}" for line in new.splitlines() if line not in old][:20]
        msgs.append("the ABI changed: bump holon.api.ABI and regenerate with `python scripts/ci/check_abi.py --write`")
    fail(msgs, f"ABI {api.ABI} listing (core/abi.txt) and hook names")


if __name__ == "__main__":
    main()
