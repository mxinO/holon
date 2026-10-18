"""The plugin ABI's listing (release design §7): what plugins may rely on, rendered from the code and compared with
`core/abi.txt`. A difference fails unless the listing is regenerated (`--write`) with `holon.api.ABI` bumped: an
addition needs a minor bump — so do keyword-only parameters with defaults added to a member plugins only call (of a
Protocol the core implements, or of a dataclass), and a field's default listing the fields its class gained —; a
removal or change, a member added to a required Protocol plugins implement, or a dataclass field added without a
default needs a major one (below 1.0: the minor number, as manifests' ranges read it). Also fails when the core looks
up a hook by name (`hasattr` / `getattr` with a literal) that no Protocol of `holon.api` declares.

Listed: every public name of `holon.api` (dataclass fields by index with types and defaults; Protocol members with
their signatures, required or optional, and who implements the Protocol), the optional hooks (`holon.api.hooks`),
`EngineConfig`, the `BuildCtx.options` keys the core provides, the manifest schema and the string vocabularies (the
API's and the IPC frame and failure kinds).

The contracts the first-party plugins provide (every resource but `kernel.*`, `arch.*`, `model.*`) are the plugins'
ABI, not the core's: each
distribution lists its own (`std/contracts.txt`, `plugins/contracts.txt`). A new contract is free; one removed (or
provided by another plugin) needs the same contract at a higher major.

`--base REF` (run_all.sh passes its base ref) also holds every listing to these rules against REF's, so a change
cannot skip its bump by regenerating the committed listing. Run in the project environment."""
from __future__ import annotations

import ast
import dataclasses
import enum
import inspect
import subprocess
import sys
import tomllib
import typing

import holon.api as api
from _common import CORE, ROOT, fail
from holon.api import hooks, manifest
from holon.core import engine, ipc

LISTING = ROOT / "core" / "abi.txt"
CONTRACTS = {"std": ROOT / "std" / "contracts.txt", "plugins": ROOT / "plugins" / "contracts.txt"}
# Protocols the core implements and plugins call (the rest are implemented by plugins).
CORE_IMPLEMENTS = {"BuildCtx", "EngineClient", "EntryView", "KernelRegistry", "Pool", "PrefixView",
                   "Registries", "Registry", "SeqView", "StepCtx", "Stream", "TargetView", "TierCtx"}
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
        fields = [f"  #{i} {f.name}: {f.type}" + ("" if f.default is dataclasses.MISSING and f.default_factory is
                                                  dataclasses.MISSING else f" = {_default(f)}")
                  for i, f in enumerate(dataclasses.fields(obj))]
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
    if isinstance(obj, typing.Mapping):                 # e.g. SLOTS: one entry per line, so a new one is an addition
        return elements(f"const {name}", (f"{k}: {getattr(v, 'value', v)}" for k, v in sorted(obj.items())))
    if isinstance(obj, frozenset):                      # e.g. KERNEL_PROPS
        return elements(f"const {name}", sorted(obj))
    # Ordered vocabularies a consumer switches over (FINISH_REASONS, ERROR_KINDS) and unions (ParseEvent) stay one
    # line: a new value breaks a consumer that handles each one, so any change is major.
    return [f"const {name} = {obj!r}"]


def elements(head: str, values: typing.Iterable[str]) -> list[str]:
    """A set-like item, one element per line under its head: an element added is an addition (minor), one removed or
    changed a removal (major)."""
    return [head, *(f"  {v}" for v in values)]


def sections(schema: dict[str, set[str] | None]) -> list[str]:
    """The manifest schema: each section's keys one per line; a section taking any keys on one line."""
    out = []
    for section, keys in schema.items():
        out += elements(f"[{section}]", sorted(keys)) if keys is not None else [f"[{section}] <any>"]
    return out


def _default(f: dataclasses.Field) -> str:
    if f.default is dataclasses.MISSING:
        return f"{f.default_factory.__name__}()"
    return f"frozenset({sorted(f.default)!r})" if isinstance(f.default, frozenset) else repr(f.default)


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


NOT_CONTRACTS = ("kernel.", "arch.", "model.")     # implementations of contracts, and what a model plugin builds


def contracts(dist: str) -> str:
    """The contracts the plugins of distribution `dist` provide — every resource but kernels, architectures and models
    (ops, units, state kinds, weight transforms, the grammar constraint, slot interfaces) —, one per line with its
    provider."""
    out = set()
    for path in ROOT.glob(f"{dist}/holon_{dist}/**/plugin.toml"):
        m = tomllib.loads(path.read_text())
        out |= {f"{r}  ({m['plugin']['name']})" for r in m.get("resources", {}).get("provides", ())
                if not r.startswith(NOT_CONTRACTS)}
    return "".join(f"{line}\n" for line in sorted(out))


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
              *sections(manifest._SECTIONS)]
    lines += ["", "# string vocabularies",
              *elements("Numerics", sorted(manifest.Numerics.__args__)),
              *elements("ipc frames", sorted(k.decode() for k in ipc._DECODE)),
              *elements("ipc failed kinds", sorted({*ipc.FAILED_KINDS.values(), "error"}))]
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


class _Sets(ast.NodeTransformer):
    """`frozenset(…)` and `set(…)` of literals as one constant per value: `frozenset([])` is `frozenset()`."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in ("frozenset", "set") and not node.keywords \
                and len(node.args) <= 1:
            try:
                items = ast.literal_eval(node.args[0]) if node.args else ()
            except ValueError:
                return node
            return ast.Constant((node.func.id, tuple(sorted(items, key=repr))))
        return node


def _value(text: str) -> str:
    """A listed default, compared by its syntax tree with sets of literals normalised (never evaluated)."""
    return ast.dump(_Sets().visit(ast.parse(text, mode="eval")))


def extends(old: str, new: str, defaults: dict[tuple[str, str], str]) -> bool:
    """Whether the line `new` lists what `old` does with additions only: a member with keyword-only parameters added,
    each with a default; a field whose default, a dataclass instance, shows the fields added to its class at their
    defaults (`defaults`: (class, field) -> its listed default), so the instance is the same."""
    if old.strip().startswith("#"):
        (field, _, x), (same, _, y) = old.partition(" = "), new.partition(" = ")
        try:
            x, y = ast.parse(x, mode="eval").body, ast.parse(y, mode="eval").body
        except SyntaxError:
            return False
        if not (field == same and isinstance(x, ast.Call) and isinstance(y, ast.Call) and isinstance(y.func, ast.Name)
                and ast.dump(x) == ast.dump(ast.Call(y.func, y.args, y.keywords[:len(x.keywords)]))):
            return False
        try:
            return all(_value(ast.unparse(k.value)) == _value(defaults[y.func.id, k.arg])
                       for k in y.keywords[len(x.keywords):])
        except (KeyError, SyntaxError):
            return False
    try:
        a, b = (ast.parse(f"def {line.strip()}: ...").body[0] for line in (old, new))
    except SyntaxError:
        return False
    n, y = len(a.args.kwonlyargs), b.args
    kept = ast.arguments(posonlyargs=y.posonlyargs, args=y.args, vararg=y.vararg, kwonlyargs=y.kwonlyargs[:n],
                         kw_defaults=y.kw_defaults[:n], kwarg=y.kwarg, defaults=y.defaults)
    return (ast.dump(ast.FunctionDef(a.name, a.args, [], [], a.returns))
            == ast.dump(ast.FunctionDef(b.name, kept, [], [], b.returns)) and None not in y.kw_defaults[n:])


def bump_error(old: str, new: str) -> str | None:
    """Why the listing `new` may not replace `old` under its version, or None. A removal or change, a member added to a
    required Protocol plugins implement (plugins of the old ABI lack it), or a dataclass field added without a default
    (callers of the old ABI do not pass it) needs a major bump (below 1.0: the minor number, as manifests' ranges read
    it); any other addition a minor one, keyword-only parameters with defaults added to a member of a Protocol the core
    implements or of a dataclass included (`extends`)."""
    was, now, before, after = _version(old), _version(new), _entries(old), _entries(new)
    def breaks(head: str, line: str) -> bool:
        if line == head or (head, head) not in before:            # a new item
            return False
        return "required, implemented by plugins" in head or (head.startswith("dataclass ") and " = " not in line)

    defaults = {(head.split()[1], line.split()[1].rstrip(":")): line.partition(" = ")[2] for head, line in after
                if head.startswith("dataclass ") and line.strip().startswith("#") and " = " in line}

    def extended(head: str, line: str) -> bool:      # by a line of the same item, which plugins only call or read
        return ("implemented by core" in head or head.startswith("dataclass ")) and any(
            h == head and extends(line, new, defaults) for h, new in after - before)

    breaking = sorted(line for head, line in before - after if not extended(head, line)) + sorted(
        line for head, line in after - before if breaks(head, line))
    major = now[0] > was[0] or (was[0] == 0 and now[:2] > was[:2])
    if breaking and not major:
        return f"a removal, change, required plugin member or field without a default needs a major ABI bump " \
               f"(from {old.split()[1]}): {breaking[:5]}"
    if not breaking and now <= was:
        return f"an addition needs a minor ABI bump (from {old.split()[1]})"
    return None


def contract_error(old: str, new: str) -> str | None:
    """Why the contracts listing `new` may not replace `old`, or None: a contract removed, or provided by another
    plugin, needs the same contract at a higher major (its provider's breaking change); new contracts are free."""
    def majors(listing: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for line in listing.splitlines():
            name, major = api.parse_versioned(line.split()[0])
            out[name] = max(out.get(name, 0), major)
        return out

    gone = sorted(set(old.splitlines()) - set(new.splitlines()))
    newest = majors(new)
    bad = [line for line in gone if newest.get(api.parse_versioned(line.split()[0])[0], 0)
           <= api.parse_versioned(line.split()[0])[1]]
    return f"contracts removed or changed without a new major: {bad[:5]}" if bad else None


def _at(base: str, path) -> str | None:
    """`path` as committed at `base` (None if absent there)."""
    out = subprocess.run(["git", "show", f"{base}:{path.relative_to(ROOT)}"], cwd=ROOT, capture_output=True, text=True)
    return out.stdout if out.returncode == 0 else None


def main() -> None:
    rendered = {LISTING: (listing(), bump_error)} | {path: (contracts(dist), contract_error)
                                                      for dist, path in CONTRACTS.items()}
    msgs = hook_names()
    base = sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else None
    if base and subprocess.run(["git", "rev-parse", "--verify", "-q", base], cwd=ROOT, capture_output=True).returncode:
        print(f"skip ABI bump check: base {base!r} not found")
        base = None
    for path, (new, rule) in rendered.items():
        committed = path.read_text() if path.exists() else None
        if "--write" in sys.argv:
            if committed is not None and committed != new and (why := rule(committed, new)):
                fail([f"{path.relative_to(ROOT)}: {why}"], "ABI listings written")
            path.write_text(new)
        elif committed != new:
            old = (committed or "").splitlines()
            msgs += [f"{path.relative_to(ROOT)}: - {line}" for line in old if line not in new.splitlines()][:20]
            msgs += [f"{path.relative_to(ROOT)}: + {line}" for line in new.splitlines() if line not in old][:20]
            msgs.append(f"{path.relative_to(ROOT)} changed: regenerate it with `python scripts/ci/check_abi.py "
                        "--write` (core/abi.txt: after bumping holon.api.ABI)")
        if base and (was := _at(base, path)) is not None and was != new and (why := rule(was, new)):
            msgs.append(f"{path.relative_to(ROOT)} against {base}: {why}")
    fail(msgs, f"ABI {api.ABI} listing (core/abi.txt), contracts and hook names")


if __name__ == "__main__":
    main()
