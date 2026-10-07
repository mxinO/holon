"""`propagate`: carry a model's lock to the code checked out now, by one of the outcomes of design §4.6.

    unchanged  the closure is what the lock pins (only the commit differs): nothing to write
    advance    the snapshot trace is byte-identical to the lock's and no uncovered site changed: appended, no human
    approve    otherwise, judge L2 vs the reference passes: the new lock (and its snapshot) needs one approval
    fail       L2 fails, or the new snapshot trace is not run-identical although every kernel claims determinism

A plan with uncovered sites (kernels that are not run-to-run deterministic) may have no run-identical snapshot; its
lock then records none, and every change to it goes through L2 and approval (design §7.10).

The commit is recorded in the lock but does not make a plan "changed": a lock committed in commit C can never name
C, so counting it would make every re-run (CI's isolation-check) write another lock. The core is pinned as a unit
instead, so core changes are seen. Not yet checked: changed-line coverage of the snapshot trace (auto-advance should
also require every changed line in the closure to have run) and "no L2 metric regresses beyond δ vs the current
lock" for approval, which needs the lock's L2 metrics recorded.
"""
from __future__ import annotations

import dataclasses
import gc
import pathlib
from typing import Any, Protocol

from holon.api import Binding, Lock, Plan, SamplingParams, UnitPin, canonical_hash
from holon.core import plans
from holon.core.engine import Engine, EngineConfig
from holon_judge import snapshot
from holon_judge.compare import compare
from holon_judge.engine import score
from holon_judge.golden import model_meta
from holon_judge.recording import PromptSet, Recording
from holon_judge.snapshot import Snapshot

Trace = list[tuple[list[int], SamplingParams]]


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


def decide(current: Lock | None, checks: Checks, sm: int, previous: Snapshot | None = None) -> Outcome:
    """The outcome for a model whose newest lock is `current` (None: a new model; its first lock always needs an
    approval). `previous` is `current`'s stored snapshot, used only to locate a divergence."""
    plan = checks.plan()
    changes = plans.diff(current.plan, plan) if current else {}
    if current is not None and changes.keys() <= {"commit"}:
        return Outcome("unchanged", changes, [f"closure unchanged since commit {current.plan.commit}"])
    first, again = checks.snapshot(), checks.snapshot()
    racy = plans.uncovered(plan)
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
    else:
        return Outcome("advance", changes, ["snapshot trace byte-identical"], lock, first)
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
    return sorted(filter(changed, set(current.uncovered) | set(plans.uncovered(plan))))


def snapshot_trace(prompts: PromptSet, n: int = 8, prompt_len: int = 128) -> Trace:
    """The snapshot trace (design §7.10) as far as M0 serves it: greedy requests, seeded sampling with n = 2 and
    logprobs, prompt logprobs, and staggered lengths so the decode batch shrinks through several graph buckets;
    the engine's step budget chunks the prefill. Changing it re-records every model's snapshot (one approval)."""
    trace = []
    for i, p in enumerate(prompts.prompts[:n]):
        params = SamplingParams(temperature=0.0, max_tokens=8 + 2 * i, ignore_eos=True)
        if i % 3 == 1:
            params = dataclasses.replace(params, temperature=0.8, top_p=0.9, seed=i, n=2, logprobs=5)
        elif i % 3 == 2:
            params = dataclasses.replace(params, prompt_logprobs=5)
        trace.append((p.ids[:prompt_len], params))
    return trace


@dataclasses.dataclass
class L2Inputs:
    prompts: PromptSet
    golden: pathlib.Path
    floor: pathlib.Path


class EngineChecks:
    """The real checks, on engines built in this process from `config` (model, device, dtype, kernel pins)."""

    def __init__(self, config: EngineConfig, trace: Trace, l2: L2Inputs | None) -> None:
        self.config, self.trace, self.inputs = config, trace, l2
        self._built: Engine | None = None

    def plan(self) -> Plan:
        self._built = self._fresh(self.config)
        return plans.plan_of(self._built)

    def snapshot(self) -> Snapshot:
        engine, self._built = self._built, None       # the plan's engine runs the first trace: one build fewer
        return snapshot.record(engine or self._fresh(self.config), self.trace, {"trace": canonical_hash(self.trace)})

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
        gc.collect()         # free the previous engine before the next one measures the memory left for it
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
    return plans.append(plans_dir, lock)


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
