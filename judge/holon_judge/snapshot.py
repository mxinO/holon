"""Snapshots: a fixed trace's outputs, recorded so two runs can be compared for byte identity (design §7.10,
"run-identical").

The driver submits the trace's requests in order before the first step, except those that wait for an earlier
request to finish (`Request.after`), which it submits in the step that request finishes. Arrival order and batch
composition are thus fixed by the trace and the engine's scheduler. Per engine step it records, for each delta, the
request's trace index, the sequence index within the request (`n > 1`) and a sha256 over everything the delta carries
except the engine-assigned id: sequence index, tokens, logprobs, prompt logprobs and finish reason. Floats are hashed
through their shortest round-trip repr, so equal digests mean bitwise-equal values. Steps that emit nothing (prefill
chunks) are kept, which also pins the schedule.
"""
from __future__ import annotations

import collections
import dataclasses
import gc
import hashlib
import json
import pathlib
from collections.abc import Callable

from holon.api import Logprobs, SamplingParams, TokenDelta

from .engine import EngineLike
from .recording import PromptSet


@dataclasses.dataclass(frozen=True)
class Request:
    ids: list[int]
    params: SamplingParams
    after: int | None = None           # submitted when this earlier request (trace index) has finished


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


def record(engine: EngineLike, trace: list[Request], meta: dict | None = None) -> Snapshot:
    """Run `trace` to completion and snapshot its outputs. The meta records the distinct numbers of deltas per step
    (for decode steps, the batch sizes the trace exercised) and, if the engine reports it, the prefix-cache tokens
    saved."""
    for i, r in enumerate(trace):
        # The engine draws a fresh seed for unseeded sampling, which would make every run differ.
        if r.params.temperature > 0 and r.params.seed is None:
            raise ValueError(f"trace request {i} samples without a seed")
        if r.after is not None and not 0 <= r.after < i:
            raise ValueError(f"trace request {i} waits for request {r.after}, which does not come before it")
    index: dict[str, int] = {}
    waiting: dict[int, list[int]] = collections.defaultdict(list)
    left = [r.params.n for r in trace]                 # unfinished sequences per request

    def submit(i: int) -> None:
        index[engine.add(trace[i].ids, trace[i].params)] = i

    for i, r in enumerate(trace):
        if r.after is None:
            submit(i)
        else:
            waiting[r.after].append(i)
    streams: list[list[list[int]]] = [[[] for _ in range(r.params.n)] for r in trace]
    steps = []
    while engine.has_work():
        step = []
        for delta in engine.step():
            i = index[delta.rid]
            streams[i][delta.index].extend(delta.token_ids)
            step.append([i, delta.index, delta_digest(delta)])
            if delta.finish_reason is not None:
                left[i] -= 1
                if left[i] == 0:
                    for j in waiting.pop(i, []):
                        submit(j)
        steps.append(step)
    stats = engine.stats() if hasattr(engine, "stats") else {}
    measured = {"step_widths": sorted({len(s) for s in steps if s})}
    if "prefix_tokens_saved" in stats:
        measured["tokens_saved"] = stats["prefix_tokens_saved"]
    return Snapshot(streams, steps, measured | (meta or {}))


def repeat(build: Callable[[int | None], EngineLike], requests: list[Request], runs: int
           ) -> tuple[list[Snapshot], list[dict[str, int]]]:
    """`runs` snapshots of the trace, each on a fresh engine `build(state_memory)`: the first sizes its pools from the
    memory it measures (None), the others get the bytes those pools took, so every run has the same pools — they set
    admission and preemption, so the schedule, and engines built one after another in a process measure different
    free memory. Each engine is freed before the next; returns the snapshots and each run's pool sizes."""
    import torch

    snaps, pools, budget = [], [], None
    for _ in range(runs):
        engine = build(budget)
        ledger = getattr(engine, "ledger", None)
        if ledger is not None:
            pools.append({name: p.capacity for name, p in ledger.pools.items()})
            budget = ledger.device_bytes()
        else:
            pools.append({})
        snaps.append(record(engine, requests))
        if hasattr(engine, "close"):
            engine.close()
        del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return snaps, pools


def trace(prompts: PromptSet, width: int = 64, short_len: int = 32, long_len: int = 2048) -> list[Request]:
    """The snapshot trace (design §7.10), as far as M0 serves it:

    - request 0: the longest prompt (at most `long_len` tokens) — long context;
    - request 1: that prompt extended, submitted once request 0 has finished — a prefix-cache hit;
    - `width - 2` short requests (windows of `short_len` tokens over the prompt set) whose `max_tokens` grow by one,
      so once they decode together the batch shrinks about one sequence per step through every graph bucket up to
      `width`. The early finishers carry seeded sampling (one with n = 2), generated-token logprobs and prompt
      logprobs: logprob steps run eagerly, so they must not overlap the large buckets for long.

    Prefill is chunked by the engine's step budget. Changing the trace re-records every model's snapshot."""
    stream = [t for p in prompts.prompts for t in p.ids]
    longest = max(prompts.prompts, key=lambda p: len(p.ids)).ids[:long_len]
    greedy = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)
    requests = [Request(longest, greedy), Request(longest + stream[:16], greedy, after=0)]
    special = {1: {"logprobs": 5}, 2: {"prompt_logprobs": 5}, 3: {"temperature": 0.8, "top_p": 0.9, "seed": 3, "n": 2}}
    for i in range(width - 2):
        start = (i * short_len) % max(1, len(stream) - short_len)
        params = dataclasses.replace(greedy, max_tokens=4 + i, **special.get(i, {}))
        if i % 8 == 5:
            params = dataclasses.replace(params, temperature=0.8, top_k=50, seed=i)
        requests.append(Request(stream[start:start + short_len], params))
    return requests


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
