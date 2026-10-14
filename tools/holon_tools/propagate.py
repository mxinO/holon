"""`propagate`: carry a model's lock to the code checked out now, by one of the outcomes of design §4.6.

    unchanged  the closure is what the lock pins (only the commit differs): nothing to write
    approve    judge L2 vs the reference passes: the new lock (and its snapshot) needs one approval
    fail       L2 fails, or the new snapshot trace is not run-identical although every kernel claims determinism
    advance    only with `allow_advance`: the snapshot trace is byte-identical to the lock's and no uncovered site
               changed — appended without a human

Auto-advance also needs changed-line coverage (every changed line in the model's closure ran during the trace,
design §4.6), which is not recorded yet; until it is, a byte-identical trace is evidence for the approval, and
advancing without it is an explicit opt-in.

A plan with uncovered sites (kernels that are not run-to-run deterministic) may have no run-identical snapshot; its
lock then records none, and every change to it goes through L2 and approval (design §7.10).

The commit is recorded in the lock but does not make a plan "changed": a lock committed in commit C can never name
C, so counting it would make every re-run (CI's isolation-check) write another lock. The core is pinned as a unit
instead, so core changes are seen. Not yet checked for approval: "no L2 metric regresses beyond δ vs the current
lock", which needs the lock's L2 metrics recorded.
"""
from __future__ import annotations

import dataclasses
import gc
import pathlib
from typing import Any, Protocol

import torch
from holon.api import Binding, Lock, Plan, UnitPin, canonical_hash
from holon.core import plans
from holon.core.engine import Engine, EngineConfig
from holon_judge import snapshot
from holon_judge.compare import compare
from holon_judge.engine import score
from holon_judge.golden import model_meta
from holon_judge.recording import PromptSet, Recording
from holon_judge.snapshot import Request, Snapshot

from . import locks


@dataclasses.dataclass(frozen=True)
class Outcome:
    verdict: str                       # unchanged | advance | approve | fail
    changes: dict[str, tuple[Any, Any]]  # plans.diff(current lock's plan, new plan)
    evidence: list[str]
    lock: Lock | None = None           # what to append (advance; approve once approved)
    snapshot: Snapshot | None = None   # the new lock's snapshot trace (None: not run-identical)


class Checks(Protocol):
    def plan(self) -> Plan:
        """The plan the code checked out now builds under the lock's pins."""

    def snapshot(self) -> Snapshot:
        """One run of the snapshot trace on a fresh engine."""

    def l2(self) -> tuple[bool, str]:
        """The judge's bounded check against the reference: (passed, report)."""


def trace_hash(snap: Snapshot) -> str:
    return canonical_hash({"streams": snap.streams, "steps": snap.steps})


def decide(current: Lock | None, checks: Checks, sm: int, previous: Snapshot | None = None,
           allow_advance: bool = False) -> Outcome:
    """The outcome for a model whose newest lock is `current` (None: a new model; its first lock always needs an
    approval). `previous` is `current`'s stored snapshot, used only to locate a divergence."""
    plan = checks.plan()
    changes = plans.diff(current.plan, plan) if current else {}
    if current is not None and changes.keys() <= {"commit"}:
        return Outcome("unchanged", changes, [f"closure unchanged since commit {current.plan.commit}"])
    first, again = checks.snapshot(), checks.snapshot()
    racy = locks.uncovered(plan)
    if (unstable := snapshot.diff(first, again)) and not racy:
        return Outcome("fail", changes, [f"snapshot trace is not run-identical: {unstable}"])
    lock = Lock(plan, None if unstable else trace_hash(first), racy, sm)
    if unstable:
        evidence = [f"snapshot trace is not run-identical ({unstable}), with uncovered sites {list(racy)}"]
    elif current is None:
        evidence = ["new model: first lock"]
    elif lock.snapshot != current.snapshot:
        evidence = [f"snapshot trace differs from the lock's: {_divergence(previous, first)}"]
    elif touched := _touched(current, plan, changes):
        evidence = [f"snapshot trace identical, but uncovered sites changed: {touched}"]
    elif allow_advance:
        return Outcome("advance", changes, ["snapshot trace byte-identical"], lock, first)
    else:
        evidence = ["snapshot trace byte-identical (auto-advance awaits changed-line coverage)"]
    evidence.append(f"trace covered step widths {_widths(first.meta)}, prefix tokens saved "
                    f"{first.meta.get('tokens_saved', '-')}")
    passed, report = checks.l2()
    return Outcome("approve" if passed else "fail", changes, [*evidence, report], lock if passed else None,
                   None if unstable else first)


def _divergence(previous: Snapshot | None, new: Snapshot) -> str:
    if previous is None:
        return "trace hash differs"
    if previous.meta.get("trace") != new.meta.get("trace"):
        return "the trace's requests changed (prompt set or trace definition)"
    return snapshot.diff(previous, new) or "trace hash differs"


def _touched(current: Lock, plan: Plan, changes: dict[str, tuple[Any, Any]]) -> list[str]:
    """Uncovered sites, of the old plan or the new, whose binding or kernel code changed: identical outputs prove
    nothing about a kernel that is not run-to-run deterministic."""
    def changed(site: str) -> bool:
        return f"site:{site}" in changes or any(f"unit:{p.sites[site].impl}" in changes
                                                for p in (current.plan, plan) if site in p.sites)
    return sorted(filter(changed, set(current.uncovered) | set(locks.uncovered(plan))))


def _widths(meta: dict) -> str:
    """Step widths as ranges, e.g. "1-49, 51-57"."""
    runs: list[list[int]] = []
    for w in meta.get("step_widths", []):
        if runs and w == runs[-1][-1] + 1:
            runs[-1].append(w)
        else:
            runs.append([w])
    return ", ".join(f"{r[0]}-{r[-1]}" if len(r) > 1 else str(r[0]) for r in runs) or "-"


@dataclasses.dataclass
class L2Inputs:
    prompts: PromptSet
    golden: pathlib.Path
    floor: pathlib.Path


class EngineChecks:
    """The real checks, on engines built in this process from `config` (model, device, dtype, kernel pins).

    Every snapshot engine gets the same state pools — they set admission and preemption, so the schedule the trace
    records: `pool_bytes` (the lock's stored snapshot records the bytes its pools took) or else the bytes the first
    engine's measured pools took. Engines built one after another measure different free memory (a predecessor's
    cached blocks, the process's own growth, another session's load); an explicit `config.state_memory` wins."""

    def __init__(self, config: EngineConfig, trace: list[Request], l2: L2Inputs | None,
                 pool_bytes: int | None = None, model_id: str | None = None) -> None:
        self.config, self.trace, self.inputs, self.pool_bytes = config, trace, l2, pool_bytes
        self.model_id = model_id           # the build's existing history, if it has one (else the checkpoint identity)
        self._built: Engine | None = None

    def plan(self) -> Plan:
        self._built = self._pinned()
        return locks.plan_of(self._built, self.model_id)

    def snapshot(self) -> Snapshot:
        engine, self._built = self._built, None       # the plan's engine runs the first trace: one build fewer
        return snapshot.record(engine or self._pinned(), self.trace,
                               {"trace": canonical_hash(self.trace), "pool_bytes": self.pool_bytes})

    def _pinned(self) -> Engine:
        engine = self._fresh(dataclasses.replace(self.config, state_memory=self.config.state_memory or
                                                 self.pool_bytes))
        if self.pool_bytes is None:
            self.pool_bytes = sum(p.capacity * p.unit_bytes for p in engine.ledger.pools.values())
        return engine

    def l2(self) -> tuple[bool, str]:
        if self.inputs is None:
            return False, "L2: no golden and floor given"
        golden, prompts = Recording.load(self.inputs.golden), self.inputs.prompts
        need = max(len(p.ids) for p in prompts.prompts) + 1
        engine = self._fresh(dataclasses.replace(self.config, max_model_len=max(self.config.max_model_len, need)))
        meta = model_meta(pathlib.Path(self.config.model)) | {"dtype": str(self.config.dtype).removeprefix("torch.")}
        result = compare(golden, score(engine, prompts, golden.k, meta), Recording.load(self.inputs.floor))
        return bool(result.passed), result.table()

    def _fresh(self, config: EngineConfig) -> Engine:
        self._built = None
        gc.collect()         # free the previous engine (cached blocks too) before the next one takes memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return Engine(config)


def snapshot_path(plans_dir: str | pathlib.Path, plan: Plan, digest: str) -> pathlib.Path:
    """Where a lock's snapshot is stored: next to its history, by content."""
    return pathlib.Path(plans_dir, plan.model_id, plan.fingerprint_class, "snapshots",
                        digest.removeprefix("sha256:") + ".json")


def write(plans_dir: str | pathlib.Path, outcome: Outcome) -> pathlib.Path:
    """Append the outcome's lock and store its snapshot, if it has one."""
    lock, snap = outcome.lock, outcome.snapshot
    if lock is None:
        raise ValueError(f"nothing to write for outcome {outcome.verdict!r}")
    if snap is not None and lock.snapshot is not None:
        path = snapshot_path(plans_dir, lock.plan, lock.snapshot)
        if not path.exists():
            dataclasses.replace(snap, meta=snap.meta | {"plan_id": lock.plan.plan_id}).save(path)
    return locks.append(plans_dir, lock)


def report(outcome: Outcome) -> str:
    lines = [f"verdict: {outcome.verdict}"]
    lines += [f"  {key}: {_short(old)} -> {_short(new)}" for key, (old, new) in outcome.changes.items()]
    return "\n".join(lines + [f"  {e}" for e in outcome.evidence])


def _short(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, UnitPin):
        return f"{v.version} {v.src[:16]}"
    if isinstance(v, Binding):
        return f"{v.impl} {canonical_hash(v)[7:19]}"       # the hash tells a config change from no change
    return str(v)
