"""Plans and lockfiles (design §4.3, §4.4, §4.6).

A `Plan` is the resolved deployment of one model on one fingerprint class; `plan_id` hashes exactly that. A `Lock`
adds the verification results recorded for the plan, which may be re-recorded without changing the plan's identity.
`(commit, lock)` fully defines a deployment.
"""
from __future__ import annotations

import dataclasses
from typing import Any

from .terms import canonical_hash


@dataclasses.dataclass(frozen=True, slots=True)
class UnitPin:
    version: str
    src: str                 # git tree hash of the unit directory
    sem: str | None = None   # hash of reference.py + parameter schema (numeric units)


@dataclasses.dataclass(frozen=True, eq=False)
class Binding:
    """What runs at one site: a unit or kernel, with its pinned config and tunables."""

    impl: str
    config: dict[str, Any] = dataclasses.field(default_factory=dict)
    tunables: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True, eq=False)
class Plan:
    model_id: str            # includes the checkpoint build (repo + file hash)
    fingerprint_class: str
    recipe: str              # canonical hash of the emitted ModelSpec
    commit: str
    uv_lock: str
    units: dict[str, UnitPin]
    sites: dict[str, Binding]
    # Core settings that shape outputs (step budget, page size, graphs, overlap, dtype …); budgets are per host.
    engine: dict[str, Any] = dataclasses.field(default_factory=dict)
    # Other checkpoints the engine opened (a drafter's), by the deployment option naming them: their identity, never
    # the host path (a deployment input).
    sources: dict[str, str] = dataclasses.field(default_factory=dict)

    @property
    def plan_id(self) -> str:
        return canonical_hash(self)


@dataclasses.dataclass(frozen=True, eq=False)
class Lock:
    plan: Plan
    snapshot: str | None = None        # bit hash of the snapshot trace (design §7.10); None: not run-identical
    uncovered: tuple[str, ...] = ()    # sites whose production kernel is not run-to-run deterministic
    sm: int = 0                        # SM of the class (implied by its SKU), for the nearest-class fallback
