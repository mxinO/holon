"""Ops, kernels and plugin registration (design §5.3, §5.4, §4.3).

An Op is a reference implementation registered under a name; a Kernel is one implementation of an Op with
constraints and properties. A plugin's `register(regs)` adds its objects explicitly; nothing registers at import.
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

    from .state import StateKindFactory


@dataclasses.dataclass(frozen=True, slots=True)
class Constraint:
    """Where a kernel may run. Empty fields mean "no constraint"."""

    sm_min: int | None = None              # compute capability as 10*major+minor, e.g. 86, 89, 120
    sm_max: int | None = None
    sms: frozenset[int] = frozenset()      # exactly these (e.g. the SMs a prebuilt binary carries)
    devices: frozenset[str] = frozenset({"cuda"})
    cpu_isa: frozenset[str] = frozenset()  # exactly these CPU tiers (`Fingerprint.cpu_isa`; a binary per flag set)

    def allows(self, device: str, sm: int | None) -> bool:
        return device in self.devices and (sm is None or not ((self.sm_min and sm < self.sm_min) or (
            self.sm_max and sm > self.sm_max) or (self.sms and sm not in self.sms)))


KERNEL_PROPS = frozenset({"graph_safe", "deterministic", "needs_host_lens", "serves_decode_rows"})  # KernelSpec.props


@dataclasses.dataclass(frozen=True, eq=False)
class KernelSpec:
    op: str
    provider: str
    impl: Callable[..., Any]
    requires: Constraint = Constraint()
    supports: Callable[[Any], bool] | None = None  # shape predicate, evaluated at plan time
    props: frozenset[str] = frozenset()            # of KERNEL_PROPS
    config: dict[str, Any] = dataclasses.field(default_factory=dict)       # pinned launch config (no autotune)
    priority: int = 0                               # agent variants register < 0 until promoted

    def __post_init__(self) -> None:
        if not self.props <= KERNEL_PROPS:
            raise ValueError(f"{self.provider} {self.op}: unknown kernel props {sorted(self.props - KERNEL_PROPS)}")


class Registry[T](Protocol):
    def add(self, key: str, value: T) -> None: ...

    def get(self, key: str) -> T: ...


class KernelRegistry(Protocol):
    """Many kernels per op: `add(op, spec)` appends; `get(op)` returns all candidates."""

    def add(self, op: str, spec: KernelSpec) -> None: ...

    def get(self, op: str) -> list[KernelSpec]: ...


class Registries(Protocol):
    """Engine-owned registries a plugin adds to. Named entries must be declared in the plugin's manifest
    (`provides = ["unit.<name>@<major>", …]`); duplicates are errors."""

    ops: Registry[Callable[..., Any]]           # op name -> reference implementation
    kernels: KernelRegistry
    units: Registry[Callable[..., Any]]         # unit name -> factory(ctx, **params)
    state_kinds: Registry[StateKindFactory]
    transforms: Registry[Callable[[Any], Any]]  # weight value transforms, e.g. "neg_exp"
    weight_sources: Registry[Callable[..., Any]]  # source kind -> factory(path) -> WeightSource


class Plugin(Protocol):
    def register(self, regs: Registries) -> None: ...
