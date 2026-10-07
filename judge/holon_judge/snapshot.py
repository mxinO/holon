"""Snapshots: a fixed trace's outputs, recorded so two runs can be compared for byte identity (design §7.10,
"run-identical").

The driver submits every request of the trace in order before the first step, so arrival order and batch
composition are fixed by the trace and the engine's scheduler. Per engine step it records, for each delta, the
request's trace index, the sequence index within the request (`n > 1`) and a sha256 over everything the delta carries
except the engine-assigned id: sequence index, tokens, logprobs, prompt logprobs and finish reason. Floats are hashed
through their shortest round-trip repr, so equal digests mean bitwise-equal values. Steps that emit nothing (prefill
chunks) are kept, which also pins the schedule.
"""
from __future__ import annotations

import collections
import dataclasses
import hashlib
import json
import pathlib

from holon.api import Logprobs, SamplingParams, TokenDelta

from .engine import EngineLike


@dataclasses.dataclass(frozen=True)
class Snapshot:
    streams: list[list[list[int]]]     # generated tokens per trace request, per sequence of that request
    steps: list[list[list]]            # per engine step: [trace index, sequence index, delta sha256] in delta order
    meta: dict = dataclasses.field(default_factory=dict)

    def save(self, path: str | pathlib.Path) -> None:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(path).write_text(json.dumps(dataclasses.asdict(self), sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | pathlib.Path) -> Snapshot:
        return cls(**json.loads(pathlib.Path(path).read_text()))


def _lp(x: Logprobs | None) -> list | None:
    return None if x is None else [x.token, x.top_ids, x.top]


def delta_digest(delta: TokenDelta) -> str:
    body = [delta.index, delta.token_ids, _lp(delta.logprobs), _lp(delta.prompt_logprobs), delta.finish_reason]
    return hashlib.sha256(json.dumps(body).encode()).hexdigest()


def record(engine: EngineLike, trace: list[tuple[list[int], SamplingParams]], meta: dict | None = None) -> Snapshot:
    """Run `trace` (prompt ids and sampling params per request) to completion and snapshot its outputs."""
    for i, (_, params) in enumerate(trace):
        # The engine draws a fresh seed for unseeded sampling, which would make every run differ.
        if params.temperature > 0 and params.seed is None:
            raise ValueError(f"trace request {i} samples without a seed")
    index = {engine.add(ids, params): i for i, (ids, params) in enumerate(trace)}
    streams: list[list[list[int]]] = [[[] for _ in range(params.n)] for _, params in trace]
    steps = []
    while engine.has_work():
        step = []
        for delta in engine.step():
            i = index[delta.rid]
            streams[i][delta.index].extend(delta.token_ids)
            step.append([i, delta.index, delta_digest(delta)])
        steps.append(step)
    return Snapshot(streams, steps, meta or {})


def diff(a: Snapshot, b: Snapshot) -> str | None:
    """None when the runs are identical, else where they first diverge."""
    for s, (x, y) in enumerate(zip(a.steps, b.steps, strict=False)):  # lengths compared below
        if x != y:
            cx, cy = collections.Counter(map(tuple, x)), collections.Counter(map(tuple, y))
            seqs = sorted({(i, j) for i, j, _ in (cx - cy) + (cy - cx)})
            return f"step {s}: " + (f"deltas differ for (request, sequence) {seqs}" if seqs else "delta order differs")
    if len(a.steps) != len(b.steps):
        return f"step count {len(a.steps)} != {len(b.steps)}"
    return None
