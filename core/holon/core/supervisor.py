"""The v0 supervisor: engine lifecycle, crash recovery, crash attribution and rollback (design §7.7, §4.4).

Frontends talk to the `Supervisor` as their `EngineClient`. It owns the engine process and watches it: the client
notices a dead process, and a periodic `stats` ping with a timeout catches a hung one. After a crash it restarts the
engine and resubmits every in-flight request from its committed tokens — each unfinished sequence continues as its
prompt plus the tokens already streamed, with its own seed and the remaining token budget. Sampling is a pure
function of (seed, position), so the stream goes on exactly where it stopped: the frontend sees no duplicate or
missing tokens, only a pause.

Attribution. Each crash strikes the requests in flight. Most crashes are transient, so requests are resubmitted
together. A request with `poison_crashes` strikes is a repeat suspect. If it is the only one in a crash and other
requests have completed since its first strike (others survive), it is poison: it fails with an error, nothing rolls
back, and the crashes it was in stop counting against the configuration. Repeat suspects that cannot be told apart go
on probation — they rerun one at a time beside other requests, so a later crash isolates the culprit. Whatever the
attribution, a request that has lived through `max_crashes` crashes (in flight or waiting) fails rather than wait for
another restart: that bounds how long any request can stall.

Rollback. Every crash no quarantine explains is charged to the configuration. `rollback_crashes` charges, or as many
failed starts in a row, restart the engine on the previous entry of its config history (rollback = restart on the
previous lock, §4.4). With no previous entry the engine keeps restarting on the oldest one — requests stay bounded by
`max_crashes` — unless it cannot even start: then the supervisor gives up, every request fails and `failed()`
returns, so the server exits and its process manager decides what next.

Strikes and charges are forgotten once an engine has served `stable_s` without crashing — measured in engine uptime,
not wall time, so slow restarts and hang timeouts cannot make repeated crashes look unrelated.

Seam: the history is a plain list of engine configs, oldest first. The lock registry (`plans/<model>/<class>/*.lock`)
will supply it — the newest lock last — once lock tooling lands; nothing else here depends on where it comes from.
"""
from __future__ import annotations

import asyncio
import collections
import contextlib
import dataclasses
import secrets
import time
import uuid
from collections.abc import AsyncIterator, Callable, Coroutine, Sequence
from typing import Any

from holon.api import SamplingParams, TokenDelta

from .engine import sequence_seed
from .ipc import EngineProcess, engine_main


@dataclasses.dataclass(frozen=True)
class Policy:
    ping_every: float = 5.0          # seconds between health pings
    ping_timeout: float = 60.0       # the engine answers between steps: no answer this long means it hangs
    start_timeout: float = 600.0     # a start (weights, pool sizing, graph capture) taking longer has failed
    stable_s: float = 60.0           # uptime without a crash after which strikes and charges are forgotten
    poison_crashes: int = 2          # strikes that make a request a quarantine suspect
    max_crashes: int = 5             # crashes a request lives through before it fails instead of waiting again
    rollback_crashes: int = 4        # unexplained crashes (or failed starts in a row) that roll back


@dataclasses.dataclass(eq=False)
class _Request:
    rid: str
    prompt: list[int]
    params: SamplingParams                   # with a concrete seed: resubmissions must draw the same tokens
    queue: asyncio.Queue[TokenDelta] = dataclasses.field(default_factory=asyncio.Queue)
    out: dict[int, list[int]] = dataclasses.field(default_factory=dict)    # tokens streamed, per sequence
    done: set[int] = dataclasses.field(default_factory=set)               # finished sequences
    engine: EngineProcess | None = None      # the engine it runs on; None while it waits or once it is done
    pumps: dict[str, asyncio.Task[None]] = dataclasses.field(default_factory=dict)   # engine rid -> task
    strikes: int = 0                         # crashes it was in flight for
    crashes: int = 0                         # crashes it lived through, in flight or waiting
    survivors_before: int = 0                # requests completed before its first strike
    attempt: int = 0

    @property
    def finished(self) -> bool:
        return len(self.done) == self.params.n

    def submissions(self) -> list[tuple[str, list[int], SamplingParams, int | None]]:
        """Engine requests (rid, tokens, params, sequence index) that continue it: the whole request while nothing
        has streamed, else one per unfinished sequence — its prompt plus the tokens streamed, its own seed, the
        remaining budget, and prompt logprobs only if they were not delivered yet."""
        self.attempt += 1
        if not self.out and not self.done:
            return [(f"{self.rid}#{self.attempt}", self.prompt, self.params, None)]
        p, subs = self.params, []
        for i in sorted(set(range(p.n)) - self.done):
            out = self.out.get(i, [])
            params = dataclasses.replace(p, n=1, seed=sequence_seed(p.seed, i),
                                         max_tokens=None if p.max_tokens is None else p.max_tokens - len(out),
                                         prompt_logprobs=p.prompt_logprobs if i == 0 and not out else None)
            subs.append((f"{self.rid}#{self.attempt}.{i}", self.prompt + out, params, i))
        return subs


class Supervisor:
    """An `EngineClient` that keeps an engine process serving (see the module docstring)."""

    def __init__(self, history: Sequence[Any], policy: Policy | None = None,
                 target: Callable[..., None] = engine_main) -> None:
        if not history:
            raise ValueError("the config history is empty")
        self.history, self.policy, self._target = list(history), policy or Policy(), target
        self.version = len(self.history) - 1            # the history entry the engine runs
        self._engine: EngineProcess | None = None       # None while (re)starting
        self._ready = asyncio.Event()                   # set while an engine serves, or once given up
        self._requests: dict[str, _Request] = {}
        self._waiting: list[_Request] = []              # to submit once an engine is ready
        self._probation: collections.deque[_Request] = collections.deque()
        self._probe: _Request | None = None             # the one repeat suspect allowed on the engine
        self._survivors = 0                             # requests completed so far
        self._charged: list[frozenset[str]] = []        # unexplained crashes: the requests in flight at each
        self._up_since = 0.0                            # when the current engine became ready
        self._failure: str | None = None
        self._gave_up = asyncio.Event()
        self._tasks: set[asyncio.Task[None]] = set()
        self._start_s: list[float] = []                 # each engine start, process launch -> ready
        self.counters = dict.fromkeys(("crashes", "resubmitted", "quarantined", "rollbacks"), 0)

    # -- lifecycle --------------------------------------------------------------------------------------------------
    async def start(self) -> None:
        await self._launch()
        if self._failure is not None:
            raise RuntimeError(self._failure)
        self._spawn(self._watch())

    async def close(self) -> None:
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)      # a restart in progress stops its engine
        self._give_up("the supervisor shut down")
        if self._engine is not None:
            await self._engine.stop()
            self._engine = None

    async def failed(self) -> str:
        """Wait until the supervisor gives up; why."""
        await self._gave_up.wait()
        assert self._failure is not None
        return self._failure

    # -- EngineClient ---------------------------------------------------------------------------------------------
    async def generate(self, token_ids: list[int], params: SamplingParams, rid: str | None = None
                       ) -> AsyncIterator[TokenDelta]:
        if self._failure is not None:
            raise RuntimeError(self._failure)
        rid = rid or uuid.uuid4().hex
        if rid in self._requests:
            raise ValueError(f"request id {rid!r} is already in flight")
        seed = params.seed if params.seed is not None else secrets.randbits(63)
        req = _Request(rid, list(token_ids), dataclasses.replace(params, seed=seed))
        self._requests[rid] = req
        self._waiting.append(req)
        self._dispatch()
        finished = 0
        try:
            while finished < params.n:
                delta = await req.queue.get()
                if delta.finish_reason is not None:
                    finished += params.n if delta.finish_reason == "error" else 1
                yield delta
        finally:
            del self._requests[rid]
            for task in req.pumps.values():        # cancelling a pump aborts its engine request
                task.cancel()
            req.done.update(range(params.n))       # nobody reads it any more: never resubmit it
            self._leave(req)

    async def abort(self, rid: str) -> None:
        if (req := self._requests.get(rid)) is not None:
            self._end(req, "abort")

    async def stats(self) -> dict[str, Any]:
        while True:
            engine = await self._serving()
            try:
                stats = await engine.client.stats()
                break
            except RuntimeError:
                if engine.client.failure is None:
                    raise
                self._crashed(engine, engine.client.failure)
        restarts = self._start_s[1:]
        return stats | self.counters | {"config": self.version, "restarts": len(restarts),
                                        "start_s": self._start_s[0], "restart_s": restarts[-1] if restarts else None,
                                        "restart_max_s": max(restarts, default=None)}

    # -- requests on the engine -----------------------------------------------------------------------------------
    def _dispatch(self) -> None:
        """Submit what waits, and the next suspect on probation if none is on the engine."""
        if self._engine is None:
            return
        if self._probe is None and self._probation:
            self._probe = self._probation.popleft()
            self._waiting.append(self._probe)
        for req in self._waiting:
            req.engine = self._engine
            if req.attempt:
                self.counters["resubmitted"] += 1
            for rid, tokens, params, index in req.submissions():
                req.pumps[rid] = asyncio.create_task(self._pump(req, self._engine, rid, tokens, params, index))
        self._waiting.clear()

    async def _pump(self, req: _Request, engine: EngineProcess, rid: str, tokens: list[int], params: SamplingParams,
                    index: int | None) -> None:
        """Forward one engine request's deltas to `req` until it finishes or the engine fails."""
        try:
            async for d in engine.client.generate(tokens, params, rid):
                if d.finish_reason == "error" and d.error == engine.client.failure:
                    break                                         # the engine failed, not this request
                d.rid, d.index = req.rid, d.index if index is None else index
                req.out.setdefault(d.index, []).extend(d.token_ids)
                req.queue.put_nowait(d)
                if d.finish_reason is not None:
                    req.done.update(range(req.params.n) if d.finish_reason == "error" else (d.index,))
                    if req.finished:
                        if d.finish_reason != "error":
                            self._survivors += 1
                        self._leave(req)
        except Exception as e:  # noqa: BLE001 — a dead engine refuses new requests; anything else ends this one
            if engine.client.failure is None:
                self._end(req, "error", f"supervisor failed: {type(e).__name__}: {e}")
        if engine.client.failure is not None:
            self._crashed(engine, engine.client.failure)

    def _end(self, req: _Request, reason: str, error: str | None = None) -> None:
        """Finish the request's unfinished sequences here (abort, quarantine, giving up)."""
        for task in req.pumps.values():
            task.cancel()
        for i in sorted(set(range(req.params.n)) - req.done):
            req.queue.put_nowait(TokenDelta(req.rid, [], index=i, finish_reason=reason, error=error))
        req.done.update(range(req.params.n))
        self._leave(req)

    def _leave(self, req: _Request) -> None:
        """`req` no longer needs the engine."""
        req.engine = None
        if req in self._waiting:
            self._waiting.remove(req)
        if req in self._probation:
            self._probation.remove(req)
        if self._probe is req:
            self._probe = None
            self._dispatch()

    # -- crashes ----------------------------------------------------------------------------------------------------
    def _crashed(self, engine: EngineProcess, reason: str) -> None:
        if engine is self._engine:                                # once per engine, however it is noticed
            self._engine = None
            self._ready.clear()
            self._spawn(self._recover(engine, reason))

    async def _recover(self, engine: EngineProcess, reason: str) -> None:
        inflight = [r for r in self._requests.values() if r.engine is engine and not r.finished]
        for req in inflight:                      # streamed tokens stand; whatever was still buffered is redrawn
            for task in req.pumps.values():
                task.cancel()
            req.pumps.clear()
            req.engine = None
        if self._probe in inflight:
            self._probe = None
        for req in self._requests.values():
            req.crashes += 1
        await engine.stop(graceful=False)
        self.counters["crashes"] += 1
        self._attribute([r for r in inflight if not r.finished], reason)        # some may have left meanwhile
        await self._launch()

    def _attribute(self, inflight: list[_Request], reason: str) -> None:
        for req in inflight:
            if not req.strikes:
                req.survivors_before = self._survivors
            req.strikes += 1
        repeat = [r for r in inflight if r.strikes >= self.policy.poison_crashes]
        if len(repeat) == 1 and self._survivors > repeat[0].survivors_before:
            poison = repeat[0]
            self.counters["quarantined"] += 1
            self._charged = [rids for rids in self._charged if poison.rid not in rids]    # it explains them
            self._end(poison, "error", f"request quarantined: in flight at {poison.strikes} engine crashes while "
                                       f"other requests completed (last: {reason})")
        else:
            self._charge(inflight)
        for req in list(self._requests.values()):       # waiting ones too: probation must not stall anyone forever
            if not req.finished and req.crashes >= self.policy.max_crashes:
                self._end(req, "error", f"not resubmitted: {req.crashes} engine crashes since it arrived "
                                        f"(last: {reason})")
        resubmit = [r for r in inflight if not r.finished and r not in repeat]
        self._probation.extend(r for r in repeat if not r.finished)
        self._waiting[:0] = resubmit                    # back before requests that arrived during the restart

    def _charge(self, inflight: list[_Request]) -> None:
        """Charge an unexplained crash to the configuration; too many roll it back."""
        self._charged.append(frozenset(r.rid for r in inflight))
        if len(self._charged) >= self.policy.rollback_crashes:
            self._roll_back()                    # with nothing older, keep restarting: max_crashes bounds requests

    def _forgive(self) -> None:
        """The engine has served `stable_s` without crashing: earlier crashes count against nobody."""
        self._charged.clear()
        for req in self._requests.values():
            req.strikes = req.crashes = 0

    def _roll_back(self) -> bool:
        """Restart on the previous config in the history from now on; False if there is none."""
        self._charged.clear()
        if self.version == 0:
            return False
        self.version -= 1
        self.counters["rollbacks"] += 1
        return True

    async def _launch(self) -> None:
        failed = 0                                      # consecutive failed starts
        while self._failure is None:
            begin = time.monotonic()
            try:
                engine = await asyncio.wait_for(EngineProcess.start(self.history[self.version], self._target),
                                                self.policy.start_timeout)
            except (RuntimeError, TimeoutError) as e:
                failed += 1
                if failed == self.policy.rollback_crashes:
                    failed = 0
                    if not self._roll_back():
                        self._give_up(f"the engine fails to start on every configuration in its history "
                                      f"(last: {str(e) or 'start timed out'})")
                continue
            self._start_s.append(time.monotonic() - begin)
            self._up_since = time.monotonic()
            self._engine = engine
            self._ready.set()
            self._dispatch()
            return

    async def _watch(self) -> None:
        while True:
            await asyncio.sleep(self.policy.ping_every)
            if (engine := self._engine) is None:
                continue
            try:
                await asyncio.wait_for(engine.client.stats(), self.policy.ping_timeout)
            except (RuntimeError, TimeoutError):          # dead, or hung (recovery kills it)
                self._crashed(engine, engine.client.failure or
                              f"engine did not answer a health ping within {self.policy.ping_timeout:g} s")
                continue
            if engine is self._engine and time.monotonic() - self._up_since >= self.policy.stable_s:
                self._forgive()

    async def _serving(self) -> EngineProcess:
        while True:
            await self._ready.wait()
            if self._failure is not None:
                raise RuntimeError(self._failure)
            if self._engine is not None:
                return self._engine

    def _give_up(self, reason: str) -> None:
        if self._failure is None:
            self._failure = reason
        for req in list(self._requests.values()):
            self._end(req, "error", self._failure)
        self._ready.set()
        self._gave_up.set()

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._done)

    def _done(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and (e := task.exception()) is not None:    # never leave requests waiting
            self._give_up(f"supervisor failed: {type(e).__name__}: {e}")


@contextlib.asynccontextmanager
async def supervised(history: Sequence[Any], policy: Policy | None = None,
                     target: Callable[..., None] = engine_main) -> AsyncIterator[Supervisor]:
    """A started supervisor over the newest config in `history`; shuts the engine down on exit."""
    supervisor = Supervisor(history, policy, target)
    await supervisor.start()
    try:
        yield supervisor
    finally:
        await supervisor.close()
