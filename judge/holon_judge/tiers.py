"""Tier checks (docs/dev/v1-kv-tiers.md acceptance 2, §6): `holon_judge tiers`.

- **Invariance** (C2's reference). A serial trace (`bounded.Trace`: sessions sharing a system prompt, each turn
  resending its history plus a user chunk; two warm-up requests first make the system prompt's end a branch point);
  before every turn after the first, a regenerate of the previous prompt cut to whole blocks, whose last block is
  recomputed while only the tier holds it (promote-into-place). The reference
  runs each session's turns back to back with no filler, so nothing it hits has left the device; the tested run, on
  the same configuration, interleaves the sessions with a filler before each return that takes every device unit, so
  every return comes through the tier. The output streams and each request's `cached_tokens` must be identical,
  with at least `MIN_LOADS` loads and a block promoted into place. Runs whose `cached_tokens` differ are not compared
  (the hit sets differ: a capacity question, judged by `bounded`).
- **Null test** (hybrids). Tiers off, `prompt_end` checkpoints only, a checkpoint pool of 4 vs 16 units on the
  back-to-back trace: identical outputs (the pool's size alone moves no byte); refused, as invariance, when
  `cached_tokens` differ (a state memory too small for 16 units also shrinks the paged pools).
- **Run-identity.** A concurrent closed-loop trace (every session sends its next turn, and a regenerate of its last
  prompt, once its requests are done; the first requests found waiting for their loads are aborted and sent again a
  few steps later) on fresh engines, each with the tier's copies
  delayed: on the GPU an unseeded sleep before each copy on the tier's streams, on the CPU `holon_judge.deferred`
  with a seed per run. Step lists (slots and query tokens per step) and outputs must be identical, with at least
  `MIN_CONCURRENT_LOADS` loads. Two variants open a plant's window: `reuse` aborts the gated request with the most
  loading blocks and sends a prefill-heavy request in the same round instead of resending it (the late copy may meet
  units the newcomer takes), and `fork` sends a fork (the finished prompt plus 128 tokens) in place of each
  regenerate (its adopt of the prompt's checkpoint runs in a long prefill step).
- **With a level below host** (`state_tier.levels`): the level's reads count too (at least `MIN_LOADS` per
  invariance run, `MIN_CONCURRENT_LOADS` per run-identity run), every level I/O job gets an unseeded sleep before it
  (`Delayed`'s bound on the GPU, `Deferred.io_delay` on the CPU), and the **restart check** (`restart`, 3f's two
  engines) runs: the level restored after a clean close gives the outputs and `cached_tokens` of an engine that never
  restarted, and after a crash nothing is restored.
- **Audit**, after every step of every run: resident blocks prefix-closed, each block's resident-children count
  (`kids`), every paged unit's references equal to its holders (requests and the index), admissions refused only with
  no slot free, the tier's own page drops leaf first (no child keeps a page copy below a dropped one), and every
  payload filed at its position (the block it ends: (depth + 1) × block, the position the kind published it at).

Planted bugs (`planted.TIER_PLANTS`) run the same checks; each counts how often it fired and the checks count how
many outcomes it changed against the clean runs (requests, steps, audit findings the clean runs do not have), each of
which fails its check. A plant is caught when it fired and changed an outcome; one that fired and changed nothing
fails its own check.
"""
from __future__ import annotations

import collections
import contextlib
import gc
import pathlib
import random
import shutil
import threading
import time
from collections.abc import Callable, Iterator
from unittest import mock

import numpy as np
import torch
from holon.api import Paged, SamplingParams

from . import deferred
from .bounded import Trace

MIN_LOADS = 8                    # tier invariance: loads of the tested run (acceptance 2)
LOOPS = {"identity": {}, "reuse": {"heavy": True}, "fork": {"fork": 128}}   # run-identity's closed-loop variants
MIN_CONCURRENT_LOADS = 32        # run-identity: loads per run
STALL_STEPS = 2000               # steps in a row without a token: the engine makes no progress


# -- the tier's copy streams under test ---------------------------------------------------------------------------
class Delayed:
    """The GPU: an unseeded sleep of up to `most` seconds on the tier's stream before each of its copies, and before
    each level I/O job (`io_sleep`: instead, 0 or `io_sleep` seconds, even odds: a job lands at once or steps late)."""

    def __init__(self, most: float, io_sleep: float | None = None) -> None:
        self.cycles = int(most * torch.cuda.get_device_properties(torch.cuda.current_device()).clock_rate * 1e3)
        self.rng, self.io_delay, self.io_sleep = random.SystemRandom(), most, io_sleep

    def wrap(self, make: Callable[[], object]) -> object:
        return _Sleepy(make(), self)

    def pump(self) -> None:
        pass


class _Sleepy:
    def __init__(self, inner, owner: Delayed) -> None:
        self.inner, self.owner = inner, owner

    def run(self, fn, after=()):
        n = self.owner.rng.randrange(self.owner.cycles + 1)
        return self.inner.run(lambda: (torch.cuda._sleep(n), fn())[1], after)


class Deferred:
    """The CPU: `holon_judge.deferred`'s streams (a copy lands 0 … `most` launches late unless waited for). With
    `io_lag`, a level I/O job's completion is also deferred, 0 … `io_lag` launches (seeded), unless something waits
    for its ticket, and at each commit's start and each `Engine.step` a seeded coin per deferred job completes it
    early: what reads completion without waiting (the free-on-ticket plant) sees it at varying points."""

    io_delay = 0.005                       # seconds: the level's I/O jobs, unseeded (up to)

    def __init__(self, seed: int, most: int = 2, io_lag: int = 0) -> None:
        self.streams = deferred.Deferred(seed, most=most)
        self.io_lag, self.rng, self.lagged, self.lock = io_lag, random.Random(seed + 1000), [], threading.Lock()

    def wrap(self, make: Callable[[], object]) -> object:
        return self.streams.stream()

    def pump(self) -> None:
        self.streams.pump()
        with self.lock:
            due = [job for at, job in self.lagged if at <= self.streams.launches]
            self.lagged = [(at, job) for at, job in self.lagged if at > self.streams.launches]
        for job in due:
            job.ticket._event.set()

    def complete(self, job) -> None:
        with self.lock:
            at = self.streams.launches + self.rng.randint(0, self.io_lag)
            if at > self.streams.launches:
                self.lagged.append((at, job))
                return
        job.ticket._event.set()

    def coin(self) -> None:
        """At a commit's start or a step's entry: each deferred completion lands now on a seeded coin."""
        with self.lock:
            due = [(at, job) for at, job in self.lagged if self.rng.random() < 0.5]
            self.lagged = [x for x in self.lagged if x not in due]
        for _, job in due:
            job.ticket._event.set()

    def release(self, ticket) -> None:
        """Someone waits for `ticket`: its completion lands now."""
        with self.lock:
            mine = [(at, job) for at, job in self.lagged if job.ticket is ticket]
            self.lagged = [x for x in self.lagged if x not in mine]
        for _, job in mine:
            job.ticket._event.set()


@contextlib.contextmanager
def copy_streams(streams: Delayed | Deferred | None) -> Iterator[None]:
    """Engines built inside give their tier `streams`' copy streams (only the tier's: the core's other copies keep
    theirs)."""
    if streams is None:
        yield
        return
    from holon.core.engine import Engine

    tier_ctx = Engine._tier_ctx

    def planted(self):
        ctx = tier_ctx(self)
        make = ctx.stream
        ctx.stream = lambda: streams.wrap(make)
        return ctx
    from holon_std.state_tier.iothreads import IoThreads

    submit, rng = IoThreads.submit, random.SystemRandom()

    def pause() -> None:                                     # an unseeded sleep before each level I/O job
        bimodal = getattr(streams, "io_sleep", None)
        time.sleep(rng.choice((0.0, bimodal)) if bimodal else rng.uniform(0, streams.io_delay))

    def delayed(self, fn, deps=(), read=False):
        return submit(self, fn, [pause, *deps], read)
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(Engine, "_tier_ctx", planted))
        stack.enter_context(mock.patch.object(IoThreads, "submit", delayed))
        if getattr(streams, "io_lag", 0):                     # deferred I/O completions
            from holon_std.state_tier.iothreads import Ticket

            wait = Ticket.wait

            def waited(ticket):
                streams.release(ticket)
                wait(ticket)
            stack.enter_context(mock.patch.object(IoThreads, "_complete", lambda io, job: streams.complete(job)))
            stack.enter_context(mock.patch.object(Ticket, "wait", waited))
            stack.enter_context(mock.patch.object(Ticket, "synchronize", waited))
            step, commit = Engine.step, Engine._commit

            def stepped(engine, *a, **k):
                streams.coin()
                return step(engine, *a, **k)

            def committed(engine, *a, **k):
                streams.coin()
                return commit(engine, *a, **k)
            stack.enter_context(mock.patch.object(Engine, "step", stepped))
            stack.enter_context(mock.patch.object(Engine, "_commit", committed))
        yield


# -- one engine, instrumented -------------------------------------------------------------------------------------
class Probe:
    """An engine with its steps recorded (slots, starts and query tokens), its tier's loads counted (page loads, payload
    loads, blocks promoted into place) and the audit run after every step."""

    def __init__(self, engine, streams: Delayed | Deferred | None = None) -> None:
        self.engine, self.streams = engine, streams
        self.steps: list[tuple[list[int], list[int], list[int]]] = []
        self.counts: collections.Counter = collections.Counter()
        self.findings: collections.Counter = collections.Counter()
        self.kernels: list[str] = []
        state, index, schedule = engine.state, engine.state.index, engine.sched.schedule

        def recorded():
            out = schedule()
            if out is not None:
                self.steps.append((out[0].slots.tolist(), out[0].start_upper.tolist(), out[0].n_query.tolist()))
            return out
        engine.sched.schedule = recorded
        admit = state.admit

        def admitting(seq):
            free = bool(state._free_slots)
            ok = admit(seq)
            if not ok and free:
                self.findings["admission refused with a slot free"] += 1
            return ok
        state.admit = admitting
        register = state.register

        def registering(seq, spec, at, payload, done=None):
            bare = [e for key in state.chain.get(seq.slot, ()) if (e := index.get(key)) is not None and
                    spec not in e.payloads]
            register(seq, spec, at, payload, done)
            if any(spec in e.payloads and (e.depth + 1) * index.block != at for e in bare):   # where it was filed
                self.findings["a payload filed away from its position"] += 1
        state.register = registering
        if streams is not None:                      # a launch went by: copies due by now land
            launching = state.launching

            def pumped(seqs):
                streams.pump()
                return launching(seqs)
            state.launching = pumped
        self.leveled = False
        if index is None or index.tier is None:
            return
        self.leveled = getattr(index.tier, "level", None) is not None
        tier, publish, promote, publishing = index.tier, state.publish, index.promote, [False]
        self.kernels = sorted({f"{getattr(op, '__module__', '')}.{getattr(op, '__qualname__', type(op).__qualname__)}"
                               for op in getattr(tier, "op", {}).values()})      # std.state_tier's copy ops
        load, dropped = tier.load, tier.dropped

        def loading(keys, units, payloads, after):
            if keys or payloads:
                self.counts["page_loads" if keys else "payload_loads"] += 1
            self.counts["loaded_blocks"] += len(keys)
            return load(keys, units, payloads, after)

        def promoting(entries, units, after):
            self.counts["promoted_into_place"] += len(entries) if publishing[0] else 0
            return promote(entries, units, after)

        def published(seq):
            publishing[0] = True
            try:
                return publish(seq)
            finally:
                publishing[0] = False

        def leaf_first():
            out, gone = dropped(), set()
            for key, spec in out:
                if spec is not None or (e := index.get(key)) is None:
                    continue
                if any(c not in gone and (x := index.get(c)) is not None and (c in index.lower or x.lower)
                       for c in e.children):
                    self.findings["the tier dropped a page copy with a child copy below it"] += 1
                gone.add(key)
                if key in index.lower:                  # the core drops it with its subtree
                    stack = [key]
                    while stack:
                        gone.update(children := index.get(stack.pop()).children)
                        stack += children
            return out
        tier.load, index.promote, state.publish, tier.dropped = loading, promoting, published, leaf_first

    def step(self) -> list:
        deltas = self.engine.step()
        for finding in audit(self.engine):
            self.findings[finding] += 1
        return deltas

    def drive(self, on_delta: Callable[[object], None], before: Callable[[], None] = lambda: None,
              pending: Callable[[], bool] = lambda: False) -> None:
        """Step until no work is left (and nothing is `pending`), handing every delta to `on_delta`; `before` runs
        before each step."""
        idle = 0
        while self.engine.has_work() or pending():
            before()
            deltas = self.step()
            idle = 0 if any(d.token_ids for d in deltas) else idle + 1
            if idle > STALL_STEPS:
                raise RuntimeError(f"no token in {STALL_STEPS} steps: the engine makes no progress")
            for d in deltas:
                on_delta(d)

    def close(self) -> None:
        tier = self.engine.tier
        if tier is not None and callable(getattr(tier, "stats", None)):
            self.counts.update({f"tier.{k}": v for k, v in tier.stats().items()})
        self.engine.close()
        del self.engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def audit(engine) -> list[str]:
    """The index's invariants (docs/dev/v1-kv-tiers.md D6) between steps: what is wrong, by kind."""
    state, index, out = engine.state, engine.state.index, []
    if index is None:
        return out
    for e in index.entries.values():
        if e.parent is not None and e.parent not in index.entries:
            out.append("a resident block under a non-resident parent")
        if e.kids != sum(c in index.entries for c in e.children):
            out.append("a block's kids differ from its resident children")
    for b in state.bound:
        if not isinstance(b.spec.shape, Paged):
            continue
        want = np.zeros(b.pool.capacity, dtype=np.int64)
        for held in state.held.values():
            np.add.at(want, np.asarray(held.get(b.spec.name, ()), dtype=np.int64), 1)
        for e in index.entries.values():
            np.add.at(want, np.asarray(e.units.get(b.spec.name, ()), dtype=np.int64), 1)
        have = b.pool.refcount(np.arange(b.pool.capacity)).astype(np.int64)
        if b.pool.null is not None:
            want[b.pool.null] = have[b.pool.null]
        if (want != have).any():
            out.append("paged units whose references differ from their holders (requests and the index)")
    return out


def tiers_off(options: dict) -> dict:
    """Plugin options for the same configuration without a tier: no `answer:<k>` checkpoint policy (tentative
    checkpoints need a tier, docs/dev/v1-kv-tiers.md D8)."""
    if "recurrent.policy" not in options:
        return options
    kept = [p for p in str(options["recurrent.policy"]).split(",") if not p.strip().startswith("answer:")]
    return options | {"recurrent.policy": ",".join(kept) or "prompt_end"}


def hybrid(engine) -> bool:
    """Whether a state kind publishes payloads (checkpoints)."""
    return any(hasattr(b.kind, "payload_pool") for b in engine.state.bound)


# -- the traces -----------------------------------------------------------------------------------------------------
def _params(n: int) -> SamplingParams:
    return SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)


def ask(probe: Probe, prompt: list[int], max_tokens: int) -> tuple[list[int], int | None]:
    """One request answered greedily on its own: (output, cached_tokens)."""
    rid, out, cached = probe.engine.add(prompt, _params(max_tokens)), [], [None]

    def take(d):
        if d.rid == rid:
            out.extend(d.token_ids)
            cached[0] = d.cached_tokens if d.cached_tokens is not None else cached[0]
    probe.drive(take)
    return out, cached[0]


def _random(trace: Trace, n: int, length: int, seed: int) -> list[list[int]]:
    """`n` prompts of `length` random tokens of the trace's vocabulary."""
    vocab, rng = sorted({t for c in trace.chunks for x in c for t in x} | set(trace.system)), random.Random(seed)
    return [[rng.choice(vocab) for _ in range(length)] for _ in range(n)]


def fillers(probe: Probe, trace: Trace) -> list[list[list[int]]]:
    """Two sets of filler prompts (used alternately), each taking every unit of the smallest paged pool: random
    tokens of the trace's vocabulary, one block short of `max_model_len` at most. Each set comes back through the tier
    itself (a load), so the tier holds two sets, not one per return."""
    engine = probe.engine
    tokens = min((p.capacity - p.has_null) * p.tokens_per_unit for p in engine.ledger.pools.values()
                 if p.tokens_per_unit)
    length = min(tokens, engine.cfg.max_model_len - engine.state.index.block - 1)
    return [_random(trace, -(-tokens // length), length, seed) for seed in (0, 1)]


def serial(probe: Probe, trace: Trace, interleaved: bool, filler: list[list[list[int]]] | None = None) -> dict:
    """The serial trace: (kind, session, turn) -> (output, cached_tokens), kind "turn" or "regenerate" ("warm": two
    requests first make the system prompt's end a branch point, so where a hybrid's `branch` checkpoints fall does not
    depend on the order of the sessions). Back to back (each session's turns in a row) or interleaved (turn by turn),
    with the `filler` sets (alternately) before every turn of the interleaved run."""
    block, sessions, turns = probe.engine.state.index.block, len(trace.chunks), len(trace.chunks[0])
    order = [(s, t) for t in range(turns) for s in range(sessions)] if interleaved else \
        [(s, t) for s in range(sessions) for t in range(turns)]
    history, prompts, out = {s: list(trace.system) for s in range(sessions)}, {}, {}
    for i, extra in enumerate(_random(trace, 2, block, seed=2)):    # the system prompt's end is a branch point
        out["warm", i, 0] = ask(probe, trace.system + extra, trace.max_tokens)
    for i, (s, t) in enumerate(order):
        for prompt in filler[i % 2] if filler else ():
            ask(probe, prompt, 1)
        if t:
            last = prompts[s, t - 1]
            out["regenerate", s, t] = ask(probe, last[:len(last) // block * block], trace.max_tokens)
        prompts[s, t] = history[s] + trace.chunks[s][t]
        out["turn", s, t] = ask(probe, prompts[s, t], trace.max_tokens)
        history[s] = prompts[s, t] + out["turn", s, t][0]
    return out


def closed_loop(probe: Probe, trace: Trace, aborts: int = 8, resend: int = 3, heavy: bool = False,
                fork: int = 0) -> list[tuple]:
    """Every session at once; once a session's requests are done it sends its next turn (history resent plus its
    chunk) together with a regenerate of its previous prompt cut to whole blocks (the two share blocks the tier may
    hold: one promotes what the other recomputes), or with `fork`, a fork: that prompt plus `fork` random tokens. The
    first `aborts` requests found waiting for their loads are aborted and sent again `resend` steps later (their
    loading blocks meanwhile held by the index alone, so evictable); with `heavy`, the one with the most loading blocks
    is aborted and not sent again, and a prefill-heavy request (session -1: random tokens, as long as a filler) is sent
    in the same round. Every output delta in order, as (steps scheduled by then, session, kind, turn, tokens)."""
    engine, block, sessions = probe.engine, probe.engine.state.index.block, range(len(trace.chunks))
    answers: dict[int, dict] = {s: {} for s in [*sessions, -1]}
    live, left, later, tick, trail = {}, [aborts], [], [0], []
    heavies = _random(trace, aborts, len(fillers(probe, trace)[0][0]), seed=3) if heavy else []

    def loading(rid) -> int:
        return sum(len(engine.state.index.loads.get(id(load), (None, ()))[1]) for q in engine._live[rid]
                   for load in engine.state.waits.get(q.slot, ()))

    def send(s: int, kind: str, t: int, prompt: list[int]) -> None:
        answers[s][kind, t] = []
        live[engine.add(prompt, _params(trace.max_tokens))] = (s, kind, t, prompt)

    def before():
        tick[0] += 1
        for item in [x for x in later if x[0] == tick[0]]:
            later.remove(item)
            send(*item[1])
        gated = sorted(rid for rid, seqs in engine._live.items() if any(q.slot in engine.state.due for q in seqs))
        if left[0] and gated:
            left[0] -= 1
            probe.counts["aborted_gated"] += 1
            rid = max(gated, key=loading) if heavy else gated[0]
            engine.abort(rid)
            if heavy:
                live.pop(rid)
                send(-1, "heavy", left[0], heavies[left[0]])
            else:
                later.append((tick[0] + resend, live.pop(rid)))

    def take(d):
        if d.rid not in live:
            return
        s, kind, t, prompt = live[d.rid]
        answers[s][kind, t].extend(d.token_ids)
        trail.append((len(probe.steps), s, kind, t, tuple(d.token_ids)))
        if not d.finish_reason:
            return
        del live[d.rid]
        if kind == "turn" and t + 1 < len(trace.chunks[s]):
            send(s, *(("fork", t + 1, prompt + _random(trace, 1, fork, seed=100 * s + t)[0]) if fork else
                      ("regenerate", t + 1, prompt[:len(prompt) // block * block])))
            send(s, "turn", t + 1, prompt + answers[s][kind, t] + trace.chunks[s][t + 1])
    for s in sessions:
        send(s, "turn", 0, list(trace.system) + trace.chunks[s][0])
    probe.drive(take, before, lambda: bool(later))
    return trail


# -- the checks -----------------------------------------------------------------------------------------------------
Build = Callable[..., object]   # build(tier=True, options=None, **EngineConfig overrides) -> Engine: `tier` False drops
                                # the slot `tier`, `options` go over the plugin options


class RunFailed(RuntimeError):
    """A run raised: what its audit had found by then."""

    def __init__(self, error: BaseException, findings: collections.Counter) -> None:
        super().__init__(f"{type(error).__name__}: {error}")
        self.findings = findings


def _run(build: Build, streams, body: Callable[[Probe], object], **overrides) -> tuple[object, Probe]:
    """`body(probe)` on a fresh engine with the tier's copies on `streams`; the probe is closed, its counts kept."""
    try:
        with copy_streams(streams):
            probe = Probe(build(**overrides), streams)
    except Exception as e:
        raise RunFailed(e, collections.Counter()) from e
    probe.hybrid = hybrid(probe.engine)
    try:
        out = body(probe)
    except Exception as e:
        with contextlib.suppress(Exception):         # a failed engine may fail to close too: the first error counts
            probe.close()
        raise RunFailed(e, probe.findings) from e
    try:
        probe.close()
    except Exception as e:
        raise RunFailed(e, probe.findings) from e
    return out, probe


def _key(k: tuple) -> str:
    return " ".join(map(str, k))


def invariance(build: Build, trace: Trace, streams: Callable[[], object]) -> dict:
    """Tier invariance against C2's reference (module docstring)."""
    ref, rp = _run(build, streams(), lambda p: serial(p, trace, interleaved=False))
    got, tp = _run(build, streams(), lambda p: serial(p, trace, interleaved=True, filler=fillers(p, trace)))
    cached = sorted(_key(k) for k in ref if ref[k][1] != got[k][1])
    outputs = sorted(_key(k) for k in ref if ref[k][0] != got[k][0])
    failures = [f"cached_tokens differ for {len(cached)} requests (first {cached[:3]}): not compared"] if cached else [
        f"outputs differ for {len(outputs)} requests (first {outputs[:3]})"] if outputs else []
    loads = tp.counts["page_loads"] + tp.counts["payload_loads"]
    if loads < MIN_LOADS or not tp.counts["promoted_into_place"]:
        failures.append(f"the tested run made {loads} loads (at least {MIN_LOADS}) and promoted "
                        f"{tp.counts['promoted_into_place']} blocks into place (at least 1)")
    if rp.counts["page_loads"]:
        failures.append(f"the reference loaded pages {rp.counts['page_loads']} times: the device pool cannot hold a "
                        "session")
    if tp.leveled and tp.counts["tier.level_reads"] < MIN_LOADS:
        failures.append(f"the tested run read {tp.counts['tier.level_reads']} copies from the level below host (at "
                        f"least {MIN_LOADS}): the host level holds every return")
    findings = rp.findings + tp.findings
    failures += [f"audit: {k} ({n} steps)" for k, n in findings.items()]
    return {"hybrid": tp.hybrid, "tier_kernels": tp.kernels, "requests": len(ref),
            "tokens_saved": sum(c or 0 for _, c in got.values()),
            "reference": dict(rp.counts), "tested": dict(tp.counts), "cached_differ": cached,
            "outputs_differ": outputs, "changed": len(set(cached) | set(outputs)), "findings": dict(findings),
            "verdict": "refused" if cached else "pass" if not failures else "fail", "failures": failures,
            "_outcomes": got}


def null_test(build: Build, trace: Trace) -> dict:
    """Tiers off, `prompt_end` only, a checkpoint pool of 4 vs 16 units, back to back (hybrids only): outputs identical
    where the hits are. Refused, as invariance, when `cached_tokens` differ: the ledger gives paged state what the
    checkpoint pool does not take, so a state memory too small for 16 units moves page evictions too (`paged_units`)."""
    runs, paged, findings = {}, {}, collections.Counter()

    def body(p):
        return serial(p, trace, interleaved=False), {n: q.capacity for n, q in p.engine.ledger.pools.items()
                                                     if q.tokens_per_unit}
    for units in (4, 16):
        (runs[units], paged[units]), probe = _run(build, None, body, tier=False, options={
            "recurrent.policy": "prompt_end", "recurrent.checkpoints": str(units)})
        findings += probe.findings
    cached = sorted(_key(k) for k in runs[4] if runs[4][k][1] != runs[16][k][1])
    outputs = sorted(_key(k) for k in runs[4] if runs[4][k][0] != runs[16][k][0])
    failures = ([f"cached_tokens differ for {len(cached)} requests (first {cached[:3]}): not compared"] if cached else
                [f"outputs differ for {len(outputs)} requests (first {outputs[:3]})"] if outputs else [])
    failures += [f"audit: {k} ({n} steps)" for k, n in findings.items()]
    return {"requests": len(runs[4]), "tokens_saved": sum(c or 0 for _, c in runs[4].values()), "paged_units": paged,
            "cached_differ": cached, "outputs_differ": outputs, "failures": failures,
            "verdict": "refused" if cached and not findings else "fail" if failures else "pass"}


def restart(build: Build, trace: Trace, streams: Callable[[int], object], root: pathlib.Path) -> dict:
    """3f's two-engine restart (acceptance 2: the level restored after a restart; `ssd.persist = close`). Each
    session's first turn is parked, the fillers after it, so its blocks go down to the level. Engine A then resumes
    every session's second turn with no restart; engine B closes instead, and B2, started on B's directory, restores
    and resumes them. B2's outputs and `cached_tokens` must be A's (a record restore discards shows as fewer cached
    tokens), with blocks restored. Then B2 stops without its snapshot, as a crash would: B3, started on that directory,
    must restore nothing."""
    shutil.rmtree(root, ignore_errors=True)
    opts = {k: {"state_tier.ssd.dir": str(root / k), "state_tier.ssd.persist": "close"} for k in ("a", "b")}
    sessions, findings = range(len(trace.chunks)), collections.Counter()

    def park(p):
        out = {s: ask(p, list(trace.system) + trace.chunks[s][0], trace.max_tokens) for s in sessions}
        for prompt in fillers(p, trace)[0]:
            ask(p, prompt, 1)
        return {s: list(trace.system) + trace.chunks[s][0] + out[s][0] for s in sessions}, out

    def resume(p, history):
        return {s: ask(p, history[s] + trace.chunks[s][1], trace.max_tokens) for s in sessions}

    def crash(p, history):
        out = resume(p, history)
        p.engine.tier.level.persist = "off"                     # no snapshot at its close: a crash leaves none
        return out

    def run(body, key, i):
        out, probe = _run(build, streams(i), body, options=opts[key])
        findings.update(probe.findings)
        return out, probe.counts
    (parked_a, resumed_a), _ = run(lambda p: (lambda h: (h[1], resume(p, h[0])))(park(p)), "a", 0)
    (history, parked_b), _ = run(park, "b", 1)
    resumed_b, counts_b2 = run(lambda p: crash(p, history), "b", 2)
    _, counts_b3 = run(lambda p: None, "b", 3)
    keys = sorted(resumed_a)
    cached = [s for s in keys if resumed_a[s][1] != resumed_b[s][1]]
    outputs = [s for s in keys if resumed_a[s][0] != resumed_b[s][0]] + [s for s in keys if parked_a[s] != parked_b[s]]
    restored, after_crash = counts_b2.get("tier.restored_blocks", 0), counts_b3.get("tier.restored_blocks", 0)
    failures = ([f"cached_tokens differ for sessions {cached} after the restart"] if cached else []) + (
        [f"outputs differ for sessions {outputs}"] if outputs else []) + (
        ["nothing was restored"] if not restored else []) + (
        [f"{after_crash} blocks restored after a crash"] if after_crash else [])
    failures += [f"audit: {k} ({n} steps)" for k, n in findings.items()]
    return {"sessions": len(keys), "restored_blocks": restored, "restored_after_crash": after_crash,
            "level_reads_after_restart": counts_b2.get("tier.level_reads", 0),
            "cached_tokens": sum(c or 0 for _, c in resumed_b.values()), "cached_differ": cached,
            "outputs_differ": outputs, "changed": len(set(cached) | set(outputs)) + bool(after_crash),
            "findings": dict(findings), "verdict": "fail" if failures else "pass", "failures": failures}


def identity(build: Build, trace: Trace, streams: Callable[[int], object], runs: int = 3, clean: dict | None = None,
             loop: dict | None = None, **overrides) -> dict:
    """Run-identity over `runs` fresh engines, `streams(i)` the i-th run's copy streams: the same steps, outputs and
    counts (all but wall times: the tier's decisions, where copies sit, are I/O-timing free, D4). With `clean` (a clean
    report: a plant's runs), each run is also held to the clean outputs on the clean schedule: up to the first step
    that differs, a run's outputs must be the clean run's (the same steps compute the same bytes); a schedule that
    moves (`moved`) is reported, and judged only by the runs' identity with each other. `loop`: closed_loop's
    variant (`LOOPS`)."""
    got = []
    for i in range(runs):
        trail, probe = _run(build, streams(i), lambda p: closed_loop(p, trace, **(loop or {})), **overrides)
        got.append((probe.steps, trail, dict(probe.counts), probe.findings))
        leveled = probe.leveled
    diffs, changed, moved = [], 0, []
    decided = [{k: v for k, v in c.items() if not k.endswith("_s")} for _, _, c, _ in got]   # all but wall times
    for i, (steps, trail, _, _) in enumerate(got[1:], start=1):
        if steps != got[0][0] or trail != got[0][1]:
            changed += 1
            diffs.append(f"run {i} differs from run 0: {_differs(got[0], (steps, trail))}")
        elif moved_counts := sorted(k for k in decided[0].keys() | decided[i].keys()
                                    if decided[0].get(k) != decided[i].get(k)):
            changed += 1                     # the tier decided differently (where copies sit) under other I/O timing
            diffs.append(f"run {i} differs from run 0: the tier's counts {moved_counts}")
    for i, (steps, trail, _, _) in enumerate(got if clean else []):
        base_steps, base_trail = clean["_runs"][0][:2]
        same = next((k for k, (a, b) in enumerate(zip(steps, base_steps, strict=False)) if a != b),
                    min(len(steps), len(base_steps)))
        moved.append(None if steps == base_steps else same)
        wrong = [x for x, y in zip(_upto(trail, same), _upto(base_trail, same), strict=False) if x != y]
        if wrong:
            changed += len(wrong)
            diffs.append(f"run {i}: {len(wrong)} outputs differ from the clean run's on its schedule (first "
                         f"{wrong[0][1:4]})")
    loads = [c.get("page_loads", 0) + c.get("payload_loads", 0) for _, _, c, _ in got]
    findings = sum((f for *_, f in got), collections.Counter())
    failures = diffs + [f"audit: {k} ({n} steps)" for k, n in findings.items()]
    if overrides.get("tier", True) and min(loads) < MIN_CONCURRENT_LOADS:
        failures.append(f"loads per run {loads}: at least {MIN_CONCURRENT_LOADS}")
    level_reads = [c.get("tier.level_reads", 0) for _, _, c, _ in got]
    if leveled and min(level_reads) < MIN_CONCURRENT_LOADS:
        failures.append(f"level reads per run {level_reads}: at least {MIN_CONCURRENT_LOADS}")
    return {"runs": runs, "steps": len(got[0][0]), "loads": loads, "counts": [c for _, _, c, _ in got],
            "diffs": diffs, "changed": changed, "findings": dict(findings),
            **({"moved_at_step": moved} if clean else {}),
            "verdict": "fail" if failures else "pass", "failures": failures, "_runs": got}


def _upto(trail: list, step: int) -> list:
    return [x for x in trail if x[0] <= step]


def _differs(a: tuple, b: tuple) -> str:
    if a[0] == b[0]:
        return "outputs"
    return f"steps from step {next((k for k, (x, y) in enumerate(zip(a[0], b[0], strict=False)) if x != y),
                                    min(len(a[0]), len(b[0])))}"


# -- plants -----------------------------------------------------------------------------------------------------------
def judge_plant(name: str, plant: Callable[[], contextlib.AbstractContextManager], checks: tuple[str, ...],
                run: dict[str, Callable[[], dict]], clean: dict) -> dict:
    """Run plant `name`'s `checks` (by name in `run`) with the plant in place: how often it fired, what each check
    found, and how many outcomes it changed against the clean runs (`clean`: the clean reports by check)."""
    out, fired = {}, collections.Counter()
    for check in checks:
        with plant() as counter:
            try:
                report = run[check]()
            except RunFailed as e:          # a crash: what the audit found before it counts, the crash does not
                report = {"verdict": "fail" if e.findings else "crash", "findings": dict(e.findings),
                          "failures": [f"audit: {k} ({n} steps)" for k, n in e.findings.items()] + [f"then {e}"]}
            fired += counter
        known = clean.get(check, {}).get("findings", {})          # audit findings of the clean runs too: not its
        report["changed"] = report.get("changed", 0) + sum(n for k, n in report.get("findings", {}).items()
                                                           if k not in known)
        out[check] = {k: v for k, v in report.items() if not k.startswith("_")}
    changed = sum(r["changed"] for r in out.values() if r["verdict"] == "fail")      # what failed a check (not
                                                                # a refused comparison: the hit sets differ)
    return {"fired": dict(fired), "changed": changed, "checks": out, "verdict": "caught" if fired["fired"] and changed
            else "fired, changed nothing" if fired["fired"] else "never fired"}


def strip(report: dict) -> dict:
    """A report without its raw runs (for JSON)."""
    return {k: strip(v) if isinstance(v, dict) else v for k, v in report.items()
            if not (isinstance(k, str) and k.startswith("_"))}            # (the null test's pools are keyed by units)

