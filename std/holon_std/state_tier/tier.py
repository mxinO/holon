"""std.state_tier (docs/v1-kv-tiers.md D3–D11): copies of prefix-index blocks in pinned host memory below the device,
and optionally on one `Level` below that (`state_tier.levels`: the built-in `ssd`, or a `Ref` to a level unit), the
slot `tier`'s plugin.

Storage. One mapped host pool per device pool it copies: a paged spec's pages, a kind's payloads. `host.bytes` is
split between pages (`host.pages`, a share) and payloads, each part over its pools by their unit bytes, so every paged
spec, and every payload spec, gets as many units. Page host pools mirror the device's layer-major shape, or are
unit-major (one segment per unit, one iovec) when a level is configured (D10); payload host pools keep the kind's own
shape, so its adopt MemOps stay S2-legal. Copies are whole units, row by row (`state_copy@1`, kernels.py): loads on one
copy stream, stores on another, so a store queued ahead never delays a load. The level gets the same split of its own
bytes.

Decisions (the determinism contract, holon.api.tier): every placement, refusal and eviction is a function of the
calls received and the step hooks, never of when a copy or an I/O job lands. Pages are evicted least recently used
and leaf first (a block with a child copy stays), payloads least recently stored or adopted; nothing being loaded is
evicted before `loaded`. Page stores follow `host.admit` (`publish`: every block; `hit:<n>`: blocks matched n times;
`evict`: only at eviction); payload stores always succeed, evicting other payloads (the attach-time floor makes that
possible).

The level (D11). A copy the host level evicts is written back to it (it is on the level from that call on; the host
units are free at once, and their next writer waits for the write), except a tentative payload and a superseded one
(walking down single-child links from its block reaches a block with an ordinary payload of the same spec): those are
dropped. A copy on both levels leaves the host level without a write, and the level's own copies go redundant ones
first (those the host level also holds), then least recently used leaves (a child copy on either level keeps a page
copy). Below `host.reserve` free units (default: one `max_model_len` load's), `before_launch` writes host leaves back.
A load of a copy only the level holds reads it into host units, which it holds from then on; one read per copy, so a
later load of it waits for that read.

I/O (`iothreads.py`). Jobs run on `ssd.io_threads` threads in the order decided, but while a read is queued no write
starts except one that frees units a queued read needs (D11's read priority, rev 8.3); a job first waits, on the host,
for what it depends on (the fences of the host units it fills or reads, the copy's previous job). A load that reads
from the level has its H2D enqueued by an I/O thread once its reads are done, in the order loads were decided, unless
it was abandoned first (`loaded(load, None)` completes the wait for that enqueue as a no-op). The engine's host wait
for an enqueue is `stats()["load_enqueue_wait_s"]`.

Fences. A host unit keeps what must pass before it is reused, one fence per ordered source (a later one implies the
earlier): the latest writer of its bytes (a store's D2H, or a level read), the latest step that read it (`loaded`'s
`done`), the latest load copy that read it no step waited for, and the latest write-back from it. A reuse waits for
them (on the copy stream, or on the host before an I/O job). A device read of a host unit follows its latest writer.

The clock. Virtual seconds advanced at each launch by `step_cost` = a + b·tokens + c·rows. Page loads queue on the
H2D lane at `h2d_rate`; a load is due once the lane has carried its pages (one of at most `stall_bytes` is due at once,
its step waits, but it still occupies the lane), and `loaded` brings the clock up to the due time it waited for.
Payload loads from the host level take no lane (their adopt reads the host copy in-step): due at once. A level read
first passes the level's lane at its read rate, behind the writes it depends on (charged at the write rate divided
by the level's mixed factor); other write-backs go on a lane of their own at the write rate (they wait for reads).
`stats` counts, besides stores, loads, write-backs and evictions, the gated loads (due after the clock when issued)
and the steps launched while one is not due."""
from __future__ import annotations

import collections
import dataclasses
import re
import threading
import time
from typing import Any

import numpy as np
import torch
from holon.api import BlockRecord, Payload, Ref

from .iothreads import IoThreads, Ticket
from .kernels import STATE_COPY
from .reference import state_copy
from .ssd import SsdLevel

UNITS = {"": 1, "k": 1 << 10, "m": 1 << 20, "g": 1 << 30, "kib": 1 << 10, "mib": 1 << 20, "gib": 1 << 30}


def nbytes(text: object) -> int:
    """`16GiB`, `512MiB`, `1048576` -> bytes."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]*)\s*", str(text))
    if m is None or m[2].lower() not in UNITS:
        raise ValueError(f"state_tier: {text!r} is not a byte size (e.g. 16GiB)")
    return int(float(m[1]) * UNITS[m[2].lower()])


@dataclasses.dataclass(eq=False)
class HostLoad:
    """A load (holon.api.Load). Its fences: the copy into device units, or the stores (or level reads) that wrote its
    payloads' host units; a load reading the level first waits, on the host, for its copy's enqueue (or abandonment)."""

    fences: list
    due: float
    payloads: dict
    keys: list                  # the copies it reads: (key, None) pages, (key, spec) payloads
    reads: list                 # the host units it reads: (pool, unit)
    copy: Any                   # its copy into device units (page loads), else None; set by an I/O thread if `level`
    level: bool = False         # it reads the level: its copy is enqueued once those reads are done
    told: set = dataclasses.field(default_factory=set)
    tier: Any = None
    enqueued: bool = False
    abandoned: bool = False

    def _enqueue_wait(self) -> None:
        if self.level and not self.enqueued:
            start, cv = time.perf_counter(), self.tier.cv
            with cv:
                cv.wait_for(lambda: self.enqueued or self.abandoned)
            self.tier.counts["load_enqueue_wait_s"] += time.perf_counter() - start

    def wait(self) -> None:
        self._enqueue_wait()
        for f in self.fences if not self.level else [self.copy] if self.copy is not None else []:
            f.wait()

    def synchronize(self) -> None:
        self._enqueue_wait()
        for f in self.fences if not self.level else [self.copy] if self.copy is not None else []:
            f.synchronize()


class HostTier:
    def __init__(self, ctx: Any) -> None:
        opts = ctx.options.get("plugins", {})
        get = lambda key, default: opts.get(f"state_tier.{key}", default)      # noqa: E731
        self.ctx, self.page_size = ctx, ctx.options.get("page_size", 16)
        self.host_bytes, self.page_share = nbytes(get("host.bytes", "8GiB")), float(get("host.pages", "0.25"))
        admit = str(get("host.admit", "publish"))
        if not (admit in ("publish", "evict") or re.fullmatch(r"hit:[1-9]\d*", admit)):
            raise ValueError(f"state_tier.host.admit {admit!r}: publish, evict or hit:<n>")
        self.admit, self.min_hits = admit.split(":")[0], int(admit.split(":")[1]) if ":" in admit else 0
        self.a, self.b, self.c = (float(x) for x in str(get("step_cost", "0.015,0.0004,0.0005")).split(","))
        self.h2d_rate, self.stall = float(get("h2d_rate", "43e9")), nbytes(get("stall_bytes", "32MiB"))
        self.reserve_opt = get("host.reserve", None)
        levels = [x.strip() for x in str(get("levels", "")).split(",") if x.strip()]
        if len(levels) > 1:
            raise ValueError(f"state_tier.levels {levels}: K binds one level below host")
        self.threads_n = int(get("ssd.io_threads", "2"))
        self.level = None
        if levels == ["ssd"]:
            self.level = SsdLevel({k: get(f"ssd.{k}", None) for k in ("dir", "direct", "persist", "read", "write",
                                                                       "mixed")}
                                  | {"bytes": nbytes(get("ssd.bytes", "64GiB"))}, self.threads_n)
        elif levels:
            self.level = ctx.unit(Ref(levels[0]), options=dict(opts))
        self.t = self.lane = self.gated = 0.0       # the clock, the H2D lane's end, the latest gated load's due
        self.nvme = self.wlane = 0.0                # the level's lane: reads and the writes they wait on; the others
        self.copies: dict[tuple[int, str | None], dict[str, list[int]]] = {}   # (key, None | spec) -> pool -> units
        self.low: dict[tuple[int, str | None], dict[str, int]] = {}           # on the level: copy -> pool -> units
        self.specs: dict[int, set[str | None]] = {}                  # key -> its copies' (None: pages, spec) anywhere
        self.lru: dict[bool, collections.OrderedDict] = {True: collections.OrderedDict(),   # pages, payloads:
                                                         False: collections.OrderedDict()}  # least recent first
        self.low_lru: collections.OrderedDict = collections.OrderedDict()
        self.meta: dict[int, tuple[int | None, tuple[int, ...]]] = {}    # key -> (parent, tokens), copies anywhere
        self.tree: dict[int | None, set[int]] = {}                   # parent -> its children among those keys
        self.parent: dict[int, int | None] = {}                      # keys with a page copy anywhere
        self.kids: dict[int | None, set[int]] = {}                   # key -> its children with a page copy anywhere
        self.host_kids: dict[int | None, set[int]] = {}              # ... with a host page copy
        self.loading: collections.Counter = collections.Counter()    # copy -> loads not yet `loaded`
        self.reading: collections.Counter = collections.Counter()    # host unit -> loads not yet `loaded`
        self.limbo: set[tuple[str, int]] = set()                     # released while still read
        self.fences: dict[tuple[str, int], dict[str, Any]] = collections.defaultdict(dict)   # -> source -> fence
        self.written: dict[tuple[str, int], Any] = {}
        self.jobs: dict[tuple[int, str | None], Any] = {}            # copy -> its latest level job
        self.enqueues: Any = None                                    # the latest load enqueue job
        self.hits: collections.Counter = collections.Counter()
        self.tentative: set[tuple[int, str]] = set()
        self.pending: list[tuple[int, str | None]] = []
        self.counts: collections.Counter = collections.Counter()
        self.cv = threading.Condition()

    # -- attach --------------------------------------------------------------------------------------------------
    def attach(self, ctx: Any) -> list:
        self.dev, self.block = dict(ctx.pools), ctx.block
        pages = {k: p for k, p in self.dev.items() if p.name != k}     # a paged spec's pool, by its spec's name
        payloads = {k: p for k, p in self.dev.items() if k not in pages}
        self.pages = set(pages)
        share = {k: self.page_share * p.unit_bytes / sum(q.unit_bytes for q in pages.values())
                 for k, p in pages.items()}
        share |= {k: (1 - self.page_share if pages else 1) * p.unit_bytes / sum(
            q.unit_bytes for q in payloads.values()) for k, p in payloads.items()}     # as many units per spec
        self.per_block = {k: max(1, self.block // self.page_size) for k in pages}
        load = {k: ctx.max_model_len // self.block * self.per_block[k] for k in pages} | {k: 1 for k in payloads}
        floor = {k: ctx.max_seqs * n for k, n in load.items()} | {k: ctx.max_seqs + 1 for k in payloads}
        short = {k: (int(self.host_bytes * share[k] // p.unit_bytes), floor[k]) for k, p in self.dev.items()
                 if self.host_bytes * share[k] // p.unit_bytes < floor[k]}
        if short:
            raise RuntimeError("state_tier.host.bytes too small for the loads in flight: " + ", ".join(
                f"{k} gets {n} units, needs {f}" for k, (n, f) in short.items()))
        self.unit_major = set(pages) if self.level is not None else set()
        self.host = {k: ctx.host_pool(f"state_tier.{k}", unit_bytes=p.unit_bytes,
                                      segments=1 if k in self.unit_major else p.segments,
                                      units=int(self.host_bytes * share[k] // p.unit_bytes), mapped=True)
                     for k, p in self.dev.items()}
        self.free = {k: collections.deque(range(p.capacity)) for k, p in self.host.items()}
        self.reserve = {k: int(self.reserve_opt) if self.reserve_opt is not None else load[k] for k in self.host}
        self.d2h, self.h2d = ctx.stream(), ctx.stream()
        self.op = {(k, d): self._bind(k, p, d) for k, p in self.dev.items() for d in ("h2d", "d2h")}
        if self.level is None:
            return []
        self.io = IoThreads(self.threads_n)
        self.low_cap = dict(self.level.bind(ctx, {k: (p.unit_bytes, p.segments, share[k]) for k, p in self.dev.items()},
                                            ctx.identity))
        self.low_free = dict(self.low_cap)
        rates = self.level.rates()
        self.read_rate, self.write_rate, self.mixed = (float(rates[k]) for k in ("read", "write", "mixed"))
        records = self.level.restore()
        for r in records:                                              # parents first
            self._meta(r.key, r.parent, r.tokens)
            if r.units:
                self._low_add((r.key, None), dict(self.per_block), r.parent)
            for spec, p in r.payloads.items():
                self._low_add((r.key, spec), {p.pool: len(p.units)}, None)
        self.counts["restored_blocks"] += len(records)
        return [BlockRecord(r.key, r.parent, r.tokens, r.units, {s: Payload((), p.pool) for s, p in r.payloads.items()})
                for r in records]

    def _bind(self, name: str, pool: Any, direction: str):
        facts = {"direction": direction, "chunk": pool.unit_bytes // pool.segments}
        if not self.ctx.candidates(STATE_COPY, facts):
            return state_copy                                   # the oracle: correct everywhere, only slower
        return self.ctx.select_kernel(STATE_COPY, site=f"std.state_tier.{name}.{direction}", **facts).impl(
            self.ctx, **facts)

    # -- the protocol --------------------------------------------------------------------------------------------
    def clock(self) -> float:
        return self.t

    def store(self, blocks, after, evicting: bool):
        accepted, jobs = set(), []
        for b in blocks:
            self._meta(b.key, b.parent, b.tokens)
            if b.units:
                k = (b.key, None)
                if k not in self.copies and self._admits(b.key, evicting):
                    host = self._room({spec: len(us) for spec, us in b.units.items()}, pages=True)
                    if host is not None:
                        self._add(k, host, b.parent)
                        jobs += [(spec, b.units[spec], host[spec]) for spec in host]
                        self.counts["page_stores" if not evicting else "evict_stores"] += 1
                    else:
                        self.counts["refused_pages"] += 1
                accepted |= {k} & (self.copies.keys() | self.low.keys())
            for spec, p in b.payloads.items():
                k = (b.key, spec)
                if k not in self.copies and k not in self.low:
                    host = self._room({p.pool: len(p.units)}, pages=False)
                    if host is None:
                        raise RuntimeError(f"state_tier: no host unit for a payload of {p.pool} (every copy loading)")
                    self._add(k, host, None)
                    jobs.append((p.pool, p.units, host[p.pool]))
                    self.counts["payload_stores"] += 1
                (self.tentative.add if p.tentative else self.tentative.discard)(k)
                accepted.add(k)
        self._forget([b.key for b in blocks])
        reuse = {id(f): f for name, _, hu in jobs for u in hu for f in self.fences.pop((name, u), {}).values()}
        fence = self._run(self.d2h, lambda: self._copy(jobs, "d2h"), [*after, *reuse.values()])
        for name, _, hu in jobs:
            for u in hu:
                self.written[name, u] = self.fences[name, u]["store"] = fence
        return accepted, fence

    def load(self, keys, units, payloads, after):
        jobs, reads, out, pages, read_bytes, dependent = [], [], {}, 0, 0, []
        copies = [(key, None) for key in keys] + list(payloads)
        self.loading.update(copies)                 # from now on: no eviction of its copies, read-backs' included
        for k in copies:                            # the level's copies come to host units first (one read each)
            if k not in self.copies:
                read_bytes += self._read_back(k, dependent)
        level = bool(read_bytes)
        for i, key in enumerate(keys):
            host = self.copies[key, None]                 # a copy the core loads is one the tier still holds
            for spec, us in units.items():
                per = len(us) // len(keys)
                jobs.append((spec, us[i * per:(i + 1) * per], host[spec]))
                pages += per * self.dev[spec].unit_bytes
        for key, spec in payloads:
            (pool, hu), = self.copies[key, spec].items()
            out[key, spec] = Payload(tuple(hu), self.host[pool].name)
            reads += [(pool, u) for u in hu]
            self.counts["payload_load_bytes"] += len(hu) * self.dev[pool].unit_bytes
        reads += [(name, u) for name, _, hu in jobs for u in hu]
        written = list({id(f): f for r in reads if (f := self.written.get(r)) is not None}.values())
        load = HostLoad([], self.t, out, copies, reads, None, level and bool(jobs), tier=self)
        if not jobs:
            load.fences = written
        elif load.level:                              # enqueued by an I/O thread, in the order loads are decided
            self._after_reads(load, jobs, after, written)
        else:
            load.copy = self._run(self.h2d, lambda: self._copy(jobs, "h2d"), [*after, *written])
            load.fences = [load.copy]
        self.reading.update(reads)
        for k in copies:
            self.lru[k[1] is None].move_to_end(k)
        if level:                                     # D4: the level's lane, then the H2D lane
            self.nvme = max(self.nvme, self.t) + read_bytes / self.read_rate
            load.due = self.nvme
        if pages:
            self.lane = max(self.lane, load.due if level else self.t) + pages / self.h2d_rate
            if level or pages > self.stall:
                load.due = self.lane
        if load.due > self.t:                                 # its readers wait for the clock: gated
            self.counts["gated_loads"] += 1
            self.gated = max(self.gated, load.due)
        self.counts["loads"] += 1
        self.counts["loaded_blocks"] += len(keys)
        self.counts["load_bytes"] += pages
        return load

    def loaded(self, load: HostLoad, done) -> None:
        kind = "abandoned" if done is None else "fence"
        if kind in load.told:
            raise RuntimeError(f"state_tier: loaded(…, {kind}) twice for one load")
        if done is None:
            with self.cv:                            # its copy is enqueued already, or never
                load.abandoned = True
                self.cv.notify_all()
        if not load.told:
            for counter, items in ((self.loading, load.keys), (self.reading, load.reads)):
                for x in items:
                    counter[x] -= 1
                    if not counter[x]:
                        del counter[x]
            for r in load.reads:
                if r in self.limbo and not self.reading[r]:
                    self.limbo.discard(r)
                    self.free[r[0]].append(r[1])
            # Its readers' fence, from the first call only: a later abandonment adds nothing (the first `done`, or
            # its copy, already covers it) and must not replace a newer load's fence on a unit reused since. A load
            # reading the level is its own fence (its copy may not be enqueued yet; enqueues keep the load order).
            step = done is not None and done is not load
            fence = done if step else load if load.level else load.copy
            if fence is not None:                    # a payload load's fences: the stores, held
                for r in load.reads:
                    self.fences[r]["step" if step else "level" if load.level else "load"] = fence
        load.told.add(kind)
        self.t = max(self.t, load.due)
        self.counts[kind] += 1

    def hit(self, keys, adopted) -> None:
        for key in keys:
            self.hits[key] += 1
            for lru, k in ((self.lru[True], (key, None)), (self.low_lru, (key, None))):
                if k in lru:
                    lru.move_to_end(k)
        for k in adopted:
            for lru in (self.lru[False], self.low_lru):
                if k in lru:
                    lru.move_to_end(k)

    def settled(self, payloads) -> None:
        self.tentative.difference_update(payloads)

    def drop(self, keys, spec: str | None = None) -> None:
        for key in keys:
            for s in [s for s in self.specs.get(key, ()) if spec is None or s == spec]:
                self._release((key, s))
                self._low_release((key, s))
            if spec is None:
                self.hits.pop(key, None)
        self._forget(keys)

    def dropped(self):
        out, self.pending = self.pending, []
        return out

    def clear(self) -> None:
        for k in list(self.copies):
            self._release(k)
        for k in list(self.low):
            self._low_release(k)
        self.hits.clear()
        self._forget(list(self.meta))

    def close(self) -> None:
        if self.level is None:
            return
        persist = getattr(self.level, "persist", "off") == "close"
        if persist:                                  # every host copy not on the level goes there (tentatives too)
            self.tentative.clear()
            for k in [k for k in self.copies if k not in self.low]:
                if self._low_room(self._need(k), k):
                    self._write_back(k, urgent=False)
        self.io.close()
        records = [BlockRecord(key, *self.meta[key], {s: () for s in self.pages} if (key, None) in self.low else {},
                               {s: Payload((0,) * sum(self.low[key, s].values()), next(iter(self.low[key, s])))
                                for s in specs if s is not None and (key, s) in self.low})
                   for key, specs in self.specs.items() if any((key, s) in self.low for s in specs)]
        self.level.close(records if persist else [])

    # -- step hooks, stats ---------------------------------------------------------------------------------------
    def before_launch(self, plan):
        self.counts["gated_steps"] += self.t < self.gated                 # launched while a gated load waits
        self.t += self.a + self.b * int(plan.n_query.sum()) + self.c * len(plan.slots)
        if self.level is not None:                   # D11's watermark: host leaves go to the level
            self.jobs = {k: j for k, j in self.jobs.items() if k in self.low or not j.ticket.done()}
            for name, want in self.reserve.items():
                if len(self.free[name]) < want:
                    self._room({name: want}, pages=name in self.pages, take=False)
        return [], []

    def stats(self) -> dict[str, int]:
        held = {k: p.capacity - len(self.free[k]) for k, p in self.host.items()}
        out = dict(self.counts) | {f"host_units.{k}": n for k, n in held.items()}
        if self.level is not None:
            out |= {f"level_units.{k}": self.low_cap[k] - n for k, n in self.low_free.items()}
        return out

    # -- the host level ------------------------------------------------------------------------------------------
    def _admits(self, key: int, evicting: bool) -> bool:
        if self.admit == "hit":
            return self.hits[key] >= self.min_hits
        return evicting or self.admit == "publish"

    def _add(self, k, host: dict[str, list[int]], parent: int | None) -> None:
        self.copies[k] = host
        self.specs.setdefault(k[0], set()).add(k[1])
        self.lru[k[1] is None][k] = None
        if k[1] is None:
            self._link(k[0], parent, self.host_kids)

    def _link(self, key: int, parent: int | None, kids: dict) -> None:
        self.parent[key] = parent
        kids.setdefault(parent, set()).add(key)      # also before its parent's (write-back stores leaves first)
        if kids is self.host_kids:
            self.kids.setdefault(parent, set()).add(key)

    def _room(self, need: dict[str, int], pages: bool, take: bool = True,
              dependent: list | None = None) -> dict[str, list[int]] | None:
        """Host units for one copy (`take`; else only room), evicting its kind's least recently used copies (pages:
        leaves of the host level) that no load reads: to the level when it takes them (their writes are
        `dependent` jobs of the caller's read, if any), else dropped; None if that cannot make room."""
        kids = self.host_kids if self.level is not None else self.kids
        tried: set = set()
        while any(len(self.free[k]) < n for k, n in need.items()):
            victim = next((k for k in self.lru[pages] if k not in tried and not self.loading[k]
                           and not (pages and kids.get(k[0])) and any(name in need for name in self.copies[k])), None)
            if victim is None:
                return None
            tried.add(victim)
            if victim in self.low:                   # on the level already: the host copy just goes
                self._release(victim)
                self.counts["host_evicted_redundant"] += 1
            elif self.level is not None and self._keeps(victim) and self._low_room(self._need(victim), victim):
                self._write_back(victim, urgent=dependent is not None, dependent=dependent)
            elif not (pages and self.kids.get(victim[0])):           # dropped (a block with a child copy stays)
                self._release(victim)
                self.pending.append(victim)
                self.counts["evicted_pages" if pages else "evicted_payloads"] += 1
        return {k: [self.free[k].popleft() for _ in range(n)] for k, n in need.items()} if take else {}

    def _release(self, k) -> None:
        """The host copy of `k` goes (its units free at once; their reuse waits for their fences)."""
        if k not in self.copies:
            return
        for name, us in self.copies.pop(k).items():
            for u in us:
                if self.reading[name, u]:                 # a load not yet `loaded` reads it: kept till then
                    self.limbo.add((name, u))
                else:
                    self.free[name].append(u)
        del self.lru[k[1] is None][k]
        if k[1] is None:
            self._unlink(k[0], self.host_kids)
        if k not in self.low:
            self._gone(k)

    def _gone(self, k) -> None:
        """`k` has no copy left anywhere."""
        self.tentative.discard(k)
        self.specs[k[0]].discard(k[1])
        if k[1] is None:
            self._unlink(k[0], self.kids)
            del self.parent[k[0]]
            self.hits.pop(k[0], None)
        if not self.specs[k[0]]:
            del self.specs[k[0]]
            self._forget([k[0]])

    def _unlink(self, key: int, kids: dict) -> None:
        parent = self.parent[key]
        kids[parent].discard(key)
        if not kids[parent]:
            del kids[parent]

    def _meta(self, key: int, parent: int | None, tokens) -> None:
        if key not in self.meta:
            self.meta[key] = (parent, tuple(tokens))
            self.tree.setdefault(parent, set()).add(key)

    def _forget(self, keys) -> None:
        """Blocks with no copy left: their records go."""
        for key in keys:
            if key not in self.specs and key in self.meta:
                parent = self.meta.pop(key)[0]
                self.tree[parent].discard(key)
                if not self.tree[parent]:
                    del self.tree[parent]

    def _need(self, k) -> dict[str, int]:
        return {name: len(us) for name, us in self.copies[k].items()} if k in self.copies else dict(self.low[k])

    # -- the level -----------------------------------------------------------------------------------------------
    def _keeps(self, k) -> bool:
        """Whether a host eviction of `k` writes it back: not a tentative payload, nor a superseded one (D11)."""
        if k[1] is None:
            return True
        if k in self.tentative:
            return False
        key = k[0]
        while len(children := self.tree.get(key, ())) == 1:          # the tier's view: blocks it holds a copy of
            (key,) = children
            if ((key, k[1]) in self.copies or (key, k[1]) in self.low) and (key, k[1]) not in self.tentative:
                return False
        return True

    def _low_room(self, need: dict[str, int], keep) -> bool:
        """Logical room on the level for `need` units, evicting its redundant copies first (also on host), then its
        least recently used leaves no load reads; False if that cannot make it."""
        while any(self.low_free[k] < n for k, n in need.items()):
            fits = lambda k: k != keep and not self.loading[k] and any(name in need for name in self.low[k])  # noqa
            victim = next((k for k in self.low_lru if fits(k) and k in self.copies), None) or next(
                (k for k in self.low_lru if fits(k) and not (k[1] is None and self.kids.get(k[0]))), None)
            if victim is None:
                return False
            redundant = victim in self.copies
            self._low_release(victim)
            if not redundant:
                self.pending.append(victim)
                self.counts["level_evicted"] += 1
        return True

    def _low_add(self, k, units: dict[str, int], parent: int | None) -> None:
        self.low[k] = units
        self.low_lru[k] = None
        for name, n in units.items():
            self.low_free[name] -= n
        if k not in self.copies:
            self.specs.setdefault(k[0], set()).add(k[1])
            if k[1] is None:
                self._link(k[0], parent, self.kids)

    def _low_release(self, k) -> None:
        if k not in self.low:
            return
        for name, n in self.low.pop(k).items():
            self.low_free[name] += n
        del self.low_lru[k]
        self.io.then(self.jobs.get(k), lambda: self.level.drop(*k))   # once its last job is done
        if k not in self.copies:
            self._gone(k)

    def _submit(self, k, fn, deps=(), read: bool = False):
        """A level job on copy `k`, after the copy's previous one (moved ahead of reads if it is a queued write)."""
        prev = self.jobs.get(k)
        if prev is not None:
            self.io.urgent(prev)
        job = self.io.submit(fn, [*([prev.ticket] if prev is not None else []), *deps], read)
        self.jobs[k] = job
        return job

    def _write_back(self, k, urgent: bool, dependent: list | None = None) -> None:
        """Write `k`'s host copy to the level (decided now: it is on the level, its host units free; their reuse waits
        for the write). `urgent`: a read waits for it (the read queue, charged at the mixed rate)."""
        host, parent = self.copies[k], self.parent.get(k[0])
        units = {name: len(us) for name, us in host.items()}
        src = {name: [self._iovec(name, u) for u in us] for name, us in host.items()}
        deps = [self.written[name, u] for name, us in host.items() for u in us
                if (name, u) in self.written]
        nbytes_ = sum(len(us) * self.host[name].unit_bytes for name, us in host.items())
        self._low_add(k, units, parent)
        job = self._submit(k, lambda: self.level.write(k[0], k[1], src), deps, read=urgent)
        if dependent is not None:
            dependent.append(job)
        for name, us in host.items():
            for u in us:
                self.fences[name, u]["io"] = job.ticket
        self._release(k)
        if urgent:
            self.nvme = max(self.nvme, self.t) + nbytes_ / (self.write_rate * self.mixed)
        else:
            self.wlane = max(self.wlane, self.t) + nbytes_ / self.write_rate
        self.counts["write_backs"] += 1
        self.counts["write_back_bytes"] += nbytes_

    def _read_back(self, k, dependent: list) -> int:
        """Bring the level's copy of `k` to host units (decided now: the host holds it, written by the read)."""
        need = dict(self.low[k])
        host = self._room(need, pages=k[1] is None, dependent=dependent)
        if host is None:
            raise RuntimeError(f"state_tier: no host units to read {k} back (every copy loading)")
        dst = {name: [self._iovec(name, u) for u in us] for name, us in host.items()}
        deps = [f for name, us in host.items() for u in us for f in self.fences.pop((name, u), {}).values()]
        for job in dependent:                        # the writes freeing these units go first (D11)
            self.io.urgent(job)
        job = self._submit(k, lambda: self.level.read(k[0], k[1], dst), deps, read=True)
        self._add(k, host, self.parent.get(k[0]) if k[1] is None else None)
        self.low_lru.move_to_end(k)
        for name, us in host.items():
            for u in us:
                self.written[name, u] = self.fences[name, u]["store"] = job.ticket
        self.counts["level_reads"] += 1
        self.counts["level_payload_reads"] += k[1] is not None
        nbytes_ = sum(len(us) * self.host[name].unit_bytes for name, us in host.items())
        self.counts["level_read_bytes"] += nbytes_
        return nbytes_

    def _after_reads(self, load: HostLoad, jobs: list, after: list, written: list) -> None:
        """Queue the load's H2D enqueue after its reads (they wrote `written`'s host units) and earlier enqueues."""
        prev = self.enqueues
        deps = ([prev.ticket] if prev is not None else []) + written
        self.enqueues = self.io.submit(lambda: self._enqueue(load, jobs, after), deps, read=True)

    def _enqueue(self, load: HostLoad, jobs: list, after: list) -> None:
        """On an I/O thread, once the load's reads are done: its H2D, unless it was abandoned first."""
        with self.cv:
            if not load.abandoned:
                load.copy = self.h2d.run(lambda: self._copy(jobs, "h2d"), list(after))
            load.enqueued = True
            self.cv.notify_all()

    @staticmethod
    def _run(stream, fn, after: list):
        """`stream.run(fn)` after `after`; I/O tickets among them are waited for on the host first (a stream cannot
        wait for them, and off CUDA it waits for nothing)."""
        for f in after:
            if isinstance(f, Ticket):
                f.wait()
        return stream.run(fn, [f for f in after if not isinstance(f, Ticket)])

    def _iovec(self, name: str, u: int) -> list[memoryview]:
        """Host unit `u` of pool `name` as segment buffers: one for a unit-major pool, else one per segment."""
        pool = self.host[name]
        seg = pool.unit_bytes // pool.segments
        view = pool.view(torch.uint8, (pool.segments, pool.capacity, seg)).numpy()
        return [memoryview(view[s, u]) for s in range(pool.segments)]

    # -- copies --------------------------------------------------------------------------------------------------
    def _copy(self, jobs: list, direction: str) -> None:
        """Every job's units, row by row: one op call per device pool (D10)."""
        by_pool: dict[str, list[tuple[Any, Any]]] = collections.defaultdict(list)
        for name, dev_units, host_units in jobs:
            by_pool[name].append((dev_units, host_units))
        for name, pairs in by_pool.items():
            dev, host = self.dev[name], self.host[name]
            dev_rows = self._rows(dev, [u for pair in pairs for u in pair[0]], False)
            host_rows = self._rows(dev, [u for pair in pairs for u in pair[1]], name in self.unit_major, host.capacity)
            d, h = self._view(dev, dev.segments, False), self._view(host, dev.segments, True)
            if direction == "h2d":
                self.op[name, direction](h, host_rows, d, dev_rows)
            else:
                self.op[name, direction](d, dev_rows, h, host_rows)

    @staticmethod
    def _rows(dev: Any, units: list[int], unit_major: bool, capacity: int | None = None) -> torch.Tensor:
        """Unit u of segment s is row s · capacity + u (row u · segments + s in a unit-major pool)."""
        s, u = np.arange(dev.segments, dtype=np.int64)[:, None], np.asarray(units, np.int64)[None]
        rows = u * dev.segments + s if unit_major else s * (capacity or dev.capacity) + u
        return torch.from_numpy(rows.ravel())

    @staticmethod
    def _view(pool: Any, segments: int, host: bool) -> torch.Tensor:
        shape = (segments * pool.capacity, pool.unit_bytes // segments)
        return pool.device_view(torch.uint8, shape) if host else pool.view(torch.uint8, shape)
