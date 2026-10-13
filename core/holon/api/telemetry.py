"""Telemetry events (design §7.9). Phase timings are diagnostic; verdicts are measured by the judge as a client."""
from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(slots=True)
class Event:
    plan_id: str
    step: int
    ts_ns: int
    phase_us: dict[str, float]            # schedule, transform, forward, fill (masks), sample, commit
    n_seqs: int
    n_tokens: int
    shape: str
    pools: dict[str, tuple[int, int]] = dataclasses.field(default_factory=dict)   # used, capacity
    prefix: dict[str, int] = dataclasses.field(default_factory=dict)             # hits, misses, tokens_saved
    spec: dict[str, int] = dataclasses.field(default_factory=dict)               # proposed, accepted
    counters: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)  # per plugin namespace
