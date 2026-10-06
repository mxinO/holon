"""Plugin discovery, loading and registries (design §4.1–§4.3).

Plugins are found through the `holon.plugins` entry-point group: each entry names a package whose directory holds
`plugin.toml`. Manifests are read without executing plugin code. Named objects a plugin registers (units, state
kinds, schemes, …) must be declared in its manifest as resources — `provides = ["unit.std.ops.rmsnorm@1"]` — and
`Ref`s resolve through those declarations. Every failure is an error, never a warning (invariant 7).
"""
from __future__ import annotations

import dataclasses
import importlib
import importlib.metadata
import importlib.util
import pathlib
from collections.abc import Iterable, Iterator
from typing import Any

from holon.api import KernelSpec, Manifest, Ref

ENTRY_GROUP = "holon.plugins"


class PluginError(RuntimeError):
    pass


class Registry[T]:
    """A name → object table that refuses silent overwrites."""

    def __init__(self, what: str) -> None:
        self.what = what
        self._items: dict[str, T] = {}

    def add(self, key: str, value: T) -> None:
        if key in self._items:
            raise PluginError(f"duplicate {self.what} {key!r}")
        self._items[key] = value

    def get(self, key: str) -> T:
        try:
            return self._items[key]
        except KeyError:
            raise PluginError(f"unknown {self.what} {key!r}") from None

    def items(self) -> Iterable[tuple[str, T]]:
        return self._items.items()

    def __contains__(self, key: str) -> bool:
        return key in self._items


class KernelRegistry:
    """Many kernels per op: `add(op, spec)` appends; `get(op)` returns all candidates."""

    def __init__(self) -> None:
        self._items: dict[str, list[KernelSpec]] = {}

    def add(self, op: str, spec: KernelSpec) -> None:
        self._items.setdefault(op, []).append(spec)

    def get(self, op: str) -> list[KernelSpec]:
        return list(self._items.get(op, ()))


# Registry field -> resource namespace a plugin must declare to add to it.
_DECLARED = {"ops": "op", "units": "unit", "state_kinds": "state_kind", "schemes": "scheme",
             "converters": "converter", "dialects": "dialect", "transforms": "transform"}


@dataclasses.dataclass
class Registries:
    ops: Registry = dataclasses.field(default_factory=lambda: Registry("op"))
    kernels: KernelRegistry = dataclasses.field(default_factory=KernelRegistry)
    units: Registry = dataclasses.field(default_factory=lambda: Registry("unit"))
    state_kinds: Registry = dataclasses.field(default_factory=lambda: Registry("state kind"))
    schemes: Registry = dataclasses.field(default_factory=lambda: Registry("scheme"))
    converters: Registry = dataclasses.field(default_factory=lambda: Registry("converter"))
    dialects: Registry = dataclasses.field(default_factory=lambda: Registry("dialect"))
    transforms: Registry = dataclasses.field(default_factory=lambda: Registry("weight transform"))


class _Declared:
    """A registry view that only accepts keys the plugin declared in `provides`."""

    def __init__(self, inner: Registry, namespace: str, manifest: Manifest) -> None:
        self._inner, self._ns, self._manifest = inner, namespace, manifest
        self._allowed = {r.name for r in manifest.provides}

    def add(self, key: str, value: Any) -> None:
        if f"{self._ns}.{key}" not in self._allowed:
            raise PluginError(f"plugin {self._manifest.name!r} registers {self._ns} {key!r} "
                              f"without declaring '{self._ns}.{key}@<major>' in provides")
        self._inner.add(key, value)

    def get(self, key: str) -> Any:
        return self._inner.get(key)

    def items(self) -> Iterable[tuple[str, Any]]:
        return self._inner.items()


class _OwnKernels:
    """A kernel-registry view that only accepts specs naming the registering plugin as provider."""

    def __init__(self, inner: KernelRegistry, manifest: Manifest) -> None:
        self._inner, self._manifest = inner, manifest

    def add(self, op: str, spec: KernelSpec) -> None:
        if spec.provider != self._manifest.name:
            raise PluginError(f"plugin {self._manifest.name!r} registers a kernel for {op!r} "
                              f"with provider {spec.provider!r}")
        self._inner.add(op, spec)

    def get(self, op: str) -> list[KernelSpec]:
        return self._inner.get(op)


def _scoped(regs: Registries, manifest: Manifest) -> Registries:
    views = {f: _Declared(getattr(regs, f), ns, manifest) for f, ns in _DECLARED.items()}
    return Registries(kernels=_OwnKernels(regs.kernels, manifest), **views)  # type: ignore[arg-type]


@dataclasses.dataclass(frozen=True)
class Installed:
    manifest: Manifest
    package: str             # the plugin's package; `manifest.entry` is relative to it


def discover(group: str = ENTRY_GROUP) -> dict[str, Installed]:
    """Find all installed plugins and read their manifests (no plugin code is executed)."""
    found: dict[str, Installed] = {}
    for ep in importlib.metadata.entry_points(group=group):
        manifest = Manifest.load(_package_dir(ep.value) / "plugin.toml")
        if manifest.name != ep.name:
            raise PluginError(f"entry point {ep.name!r} points at manifest named {manifest.name!r}")
        if manifest.name in found:
            raise PluginError(f"plugin {manifest.name!r} installed twice")
        found[manifest.name] = Installed(manifest, ep.value)
    return found


def _package_dir(module: str) -> pathlib.Path:
    """Locate a package directory without importing it or its parents (`find_spec` on a top-level name does not
    execute the package). A namespace package spread over several directories uses its first location."""
    top, *rest = module.split(".")
    spec = importlib.util.find_spec(top)
    if spec is None or not spec.submodule_search_locations:
        raise PluginError(f"plugin package {module!r} not found")
    path = pathlib.Path(next(iter(spec.submodule_search_locations))).joinpath(*rest)
    if not (path / "plugin.toml").is_file():
        raise PluginError(f"{module!r}: no plugin.toml in {path}")
    return path


class PluginSet:
    """The installed plugins, an index of the resources they provide, and the registries of those loaded so far."""

    def __init__(self, installed: dict[str, Installed]) -> None:
        self.installed = installed
        self.regs = Registries()
        self._loaded: dict[str, Any] = {}
        # Interface resources (e.g. "kernel.attention") may have several providers for the resolver to choose
        # from; registered names (units, ops, state kinds, …) must have exactly one.
        self._providers: dict[str, list[tuple[str, int]]] = {}
        unique = set(_DECLARED.values())
        for name, inst in installed.items():
            for r in inst.manifest.provides:
                others = self._providers.setdefault(r.name, [])
                if others and r.name.split(".", 1)[0] in unique:
                    raise PluginError(f"{r.name!r} provided by both {others[0][0]!r} and {name!r}")
                others.append((name, r.major))

    def by_slot(self, slot: str) -> Iterator[Manifest]:
        return (i.manifest for i in self.installed.values() if i.manifest.slot == slot)

    def provider(self, namespace: str, ref: Ref) -> str:
        """The plugin declaring `<namespace>.<ref.name>` at `ref.major`."""
        resource = f"{namespace}.{ref.name}"
        found = self._providers.get(resource, [])
        if len(found) != 1:
            raise PluginError(f"{resource!r} needs exactly one provider, found {[p for p, _ in found]}")
        plugin, major = found[0]
        if major != ref.major:
            raise PluginError(f"{ref.target}: {plugin!r} provides {resource}@{major}")
        return plugin

    def load(self, name: str) -> Any:
        """Import a plugin's entry class, instantiate it and let it register; idempotent."""
        if name not in self._loaded:
            inst = self.installed.get(name)
            if inst is None:
                raise PluginError(f"plugin {name!r} is not installed")
            module_name, _, attr = inst.manifest.entry.partition(":")
            try:
                plugin = getattr(importlib.import_module(f"{inst.package}.{module_name}"), attr)()
                plugin.register(_scoped(self.regs, inst.manifest))
            except PluginError:
                raise
            except Exception as e:  # noqa: BLE001 — any import/registration failure becomes a named plugin error
                raise PluginError(f"plugin {name!r} failed to load: {type(e).__name__}: {e}") from e
            self._loaded[name] = plugin
        return self._loaded[name]
