"""Backend-agnostic trace replay with client-side timing (stdlib-only: runs inside every backend's environment).

The driver owns the clock: it releases requests at their arrival times (or `gap` after the turn they continue),
calls `Backend.step()` while work is in flight, and timestamps every token when `step()` hands it back — so an
engine that reports a token one step late (overlap scheduling) is charged for it, as a client would be.
"""
from __future__ import annotations

import dataclasses
import heapq
import itertools
import time
from typing import Protocol

from workloads import tokens

Delta = tuple[int, list[int], bool]     # (request id, new token ids, finished)


class Backend(Protocol):
    """An engine driven one scheduling step at a time. In-process engines implement it directly; an HTTP backend
    would implement `step()` as "collect the stream chunks that arrived"."""

    def add(self, rid: int, prompt: list[int], max_tokens: int) -> None: ...
    def step(self) -> list[Delta]: ...
    def has_work(self) -> bool: ...
    def info(self) -> dict: ...


@dataclasses.dataclass
class Request:
    sid: int
    turn: int
    prompt: list[int]
    output: int
    release: float                       # seconds since replay start
    first: float | None = None
    last: float | None = None
    generated: list[int] = dataclasses.field(default_factory=list)

    def row(self) -> list:
        return [self.sid, self.turn, len(self.prompt), self.output, len(self.generated),
                round(self.release, 6), round(self.first, 6), round(self.last, 6)]


ROW = ["sid", "turn", "prompt", "output", "generated", "release", "first", "last"]
_rids = itertools.count()               # unique across the replays of one process (warmup included)


def replay(backend: Backend, sessions: list[dict], salt: str) -> tuple[list[Request], list[tuple]]:
    """Run every session to completion. Returns the finished requests and one (wall s, CPU s, tokens, first
    tokens) record per step; CPU time is the whole process's, measured around `step()` only."""
    system = tokens(salt, "system", max(s["system"] for s in sessions))
    by_id = {s["id"]: s for s in sessions}
    ready = [(s["arrival"], s["id"], 0, []) for s in sessions]   # (release, sid, turn, history)
    heapq.heapify(ready)
    live: dict[int, Request] = {}
    done: list[Request] = []
    steps: list[tuple] = []
    t0 = time.perf_counter()
    while ready or live:
        now = time.perf_counter() - t0
        while ready and ready[0][0] <= now:
            release, sid, i, history = heapq.heappop(ready)
            s, t = by_id[sid], by_id[sid]["turns"][i]
            prompt = (history or system[:s["system"]]) + tokens(salt, f"{sid}.{i}", t["append"])
            rid = next(_rids)
            live[rid] = Request(sid, i, prompt, t["output"], release)
            backend.add(rid, prompt, t["output"])
        if not backend.has_work():
            if not ready:
                raise RuntimeError(f"backend is idle with {len(live)} requests unfinished")
            time.sleep(max(0.0, ready[0][0] - now))
            continue
        w0, c0 = time.perf_counter(), time.process_time()
        deltas = backend.step()
        c1, w1 = time.process_time(), time.perf_counter()
        firsts = 0
        for rid, toks, finished in deltas:
            r = live.get(rid)
            if r is None:                # a token for a request already finished (overlap tail)
                continue
            if r.first is None:
                r.first, firsts = w1 - t0, firsts + 1
            r.last = w1 - t0
            r.generated += toks
            if finished:
                done.append(live.pop(rid))
                s = by_id[r.sid]
                if r.turn + 1 < len(s["turns"]):
                    nxt = s["turns"][r.turn + 1]
                    heapq.heappush(ready, (r.last + nxt["gap"], r.sid, r.turn + 1, r.prompt + r.generated))
        steps.append((w1 - w0, c1 - c0, sum(len(d[1]) for d in deltas), firsts))
    while backend.has_work():            # drain in-flight work so the next replay starts clean
        backend.step()
    return done, steps


def warmup_sessions(sessions: list[dict], max_output: int = 16) -> list[dict]:
    """First turns of every session, all at t=0 with short outputs: compiles/captures the prefill and decode shapes
    the measured replay will hit (replayed under another salt, so no prompt is cached for the measured run)."""
    return [{**s, "arrival": 0.0, "turns": [{**s["turns"][0], "output": min(s["turns"][0]["output"], max_output)}]}
            for s in sessions]
