"""std.state_tier's host level (docs/v1-kv-tiers.md D3–D8, D10): copies of prefix-index blocks in pinned host memory
below the device, the slot `tier`'s plugin.

Storage. One mapped host pool per device pool it copies, of the same unit shape (`[segments, capacity, chunk]`): a
paged spec's pages, a kind's payloads. `host.bytes` is split between pages (`host.pages`, a share) and payloads, each
part over its pools by their unit bytes. Copies are whole units, row by row (`state_copy@1`, kernels.py): loads on one
copy stream, stores on another, so a store queued ahead never delays a load.

Decisions (the determinism contract, holon.api.tier): every placement, refusal and eviction is a function of the
calls received and the step hooks, never of when a copy lands. Pages are evicted least recently used and leaf first
(a block with a child copy stays), payloads least recently stored or adopted; nothing being loaded is evicted before
`loaded`. Page stores follow `host.admit` (`publish`: every block; `hit:<n>`: blocks matched n times; `evict`: only at
eviction); payload stores always succeed, evicting other payloads (the attach-time floor makes that possible).

Fences. A host unit keeps what must pass before it is reused, one fence per ordered source (a later one implies the
earlier): the latest store that wrote it, the latest step that read it (`loaded`'s `done`) and the latest load copy
that read it no step waited for; a reuse waits for them on the copy stream. A device read of a host unit follows the
store that wrote it: a load's copy waits for it, and a payload load's fence is it.

The clock. Virtual seconds advanced at each launch by `step_cost` = a + b·tokens + c·rows. Page loads queue on the
H2D lane at `h2d_rate`; a load is due once the lane has carried its pages (one of at most `stall_bytes` is due at once,
its step waits, but it still occupies the lane), and `loaded` brings the clock up to the due time it waited for.
Payload loads take no lane (their adopt reads the host copy in-step): due at once."""
from __future__ import annotations

import collections
import dataclasses
import re
from typing import Any

import numpy as np
import torch
from holon.api import Payload

from .kernels import STATE_COPY
from .reference import state_copy

UNITS = {"": 1, "k": 1 << 10, "m": 1 << 20, "g": 1 << 30, "kib": 1 << 10, "mib": 1 << 20, "gib": 1 << 30}


def nbytes(text: object) -> int:
    """`16GiB`, `512MiB`, `1048576` -> bytes."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]*)\s*", str(text))
    if m is None or m[2].lower() not in UNITS:
        raise ValueError(f"state_tier: {text!r} is not a byte size (e.g. 16GiB)")
    return int(float(m[1]) * UNITS[m[2].lower()])


@dataclasses.dataclass(eq=False)
class HostLoad:
    """A load (holon.api.Load): its fences are the copy into device units, or the stores that wrote the payloads."""

    fences: list
    due: float
    payloads: dict
    keys: list                  # the copies it reads: (key, None) pages, (key, spec) payloads
    reads: list                 # the host units it reads: (pool, unit)
    copy: Any                   # its copy into device units (page loads), else None
    told: set = dataclasses.field(default_factory=set)

    def wait(self) -> None:
        for f in self.fences:
            f.wait()

    def synchronize(self) -> None:
        for f in self.fences:
            f.synchronize()

    def done(self) -> bool:
        return all(f.done() for f in self.fences)


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
        self.t = self.lane = 0.0
        self.copies: dict[tuple[int, str | None], dict[str, list[int]]] = {}   # (key, None | spec) -> pool -> units
        self.specs: dict[int, set[str | None]] = {}                  # key -> its copies' (None: pages, spec)
        self.lru: dict[bool, collections.OrderedDict] = {True: collections.OrderedDict(),   # pages, payloads:
                                                         False: collections.OrderedDict()}  # least recent first
        self.parent: dict[int, int | None] = {}
        self.kids: dict[int | None, set[int]] = {}                   # key -> its children with a page copy
        self.loading: collections.Counter = collections.Counter()    # copy -> loads not yet `loaded`
        self.reading: collections.Counter = collections.Counter()    # host unit -> loads not yet `loaded`
        self.limbo: set[tuple[str, int]] = set()                     # released while still read
        self.fences: dict[tuple[str, int], dict[str, Any]] = collections.defaultdict(dict)   # -> source -> fence
        self.written: dict[tuple[str, int], Any] = {}
        self.hits: collections.Counter = collections.Counter()
        self.tentative: set[tuple[int, str]] = set()
        self.pending: list[tuple[int, str | None]] = []
        self.counts: collections.Counter = collections.Counter()

    # -- attach --------------------------------------------------------------------------------------------------
    def attach(self, ctx: Any) -> list:
        self.dev, self.block = dict(ctx.pools), ctx.block
        pages = {k: p for k, p in self.dev.items() if p.name != k}     # a paged spec's pool, by its spec's name
        payloads = {k: p for k, p in self.dev.items() if k not in pages}
        self.pages = set(pages)
        share = {k: self.host_bytes * self.page_share * p.unit_bytes / sum(q.unit_bytes for q in pages.values())
                 for k, p in pages.items()}
        share |= {k: self.host_bytes * (1 - self.page_share if pages else 1) * p.unit_bytes / sum(
            q.unit_bytes for q in payloads.values()) for k, p in payloads.items()}     # as many units per spec
        floor = {k: ctx.max_seqs * (ctx.max_model_len // self.block) * max(1, self.block // self.page_size)
                 for k in pages} | {k: ctx.max_seqs + 1 for k in payloads}
        short = {k: (int(share[k] // p.unit_bytes), floor[k]) for k, p in self.dev.items()
                 if share[k] // p.unit_bytes < floor[k]}
        if short:
            raise RuntimeError("state_tier.host.bytes too small for the loads in flight: " + ", ".join(
                f"{k} gets {n} units, needs {f}" for k, (n, f) in short.items()))
        self.host = {k: ctx.host_pool(f"state_tier.{k}", unit_bytes=p.unit_bytes, segments=p.segments,
                                      units=int(share[k] // p.unit_bytes), mapped=True) for k, p in self.dev.items()}
        self.free = {k: collections.deque(range(p.capacity)) for k, p in self.host.items()}
        self.d2h, self.h2d = ctx.stream(), ctx.stream()
        self.op = {(k, d): self._bind(k, p, d) for k, p in self.dev.items() for d in ("h2d", "d2h")}
        return []

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
                accepted |= {k} & self.copies.keys()
            for spec, p in b.payloads.items():
                k = (b.key, spec)
                if k not in self.copies:
                    host = self._room({p.pool: len(p.units)}, pages=False)
                    if host is None:
                        raise RuntimeError(f"state_tier: no host unit for a payload of {p.pool} (every copy loading)")
                    self._add(k, host, None)
                    jobs.append((p.pool, p.units, host[p.pool]))
                    self.counts["payload_stores"] += 1
                (self.tentative.add if p.tentative else self.tentative.discard)(k)
                accepted.add(k)
        reuse = {id(f): f for name, _, hu in jobs for u in hu for f in self.fences.pop((name, u), {}).values()}
        fence = self.d2h.run(lambda: self._copy(jobs, "d2h"), [*after, *reuse.values()])
        for name, _, hu in jobs:
            for u in hu:
                self.written[name, u] = self.fences[name, u]["store"] = fence
        return accepted, fence

    def load(self, keys, units, payloads, after):
        jobs, reads, out, copies, pages = [], [], {}, [], 0
        for i, key in enumerate(keys):
            host = self.copies[key, None]                 # a copy the core loads is one the tier still holds
            for spec, us in units.items():
                per = len(us) // len(keys)
                jobs.append((spec, us[i * per:(i + 1) * per], host[spec]))
                pages += per * self.dev[spec].unit_bytes
            copies.append((key, None))
        for key, spec in payloads:
            (pool, hu), = self.copies[key, spec].items()
            out[key, spec] = Payload(tuple(hu), self.host[pool].name)
            reads += [(pool, u) for u in hu]
            self.counts["payload_load_bytes"] += len(hu) * self.dev[pool].unit_bytes
            copies.append((key, spec))
        reads += [(name, u) for name, _, hu in jobs for u in hu]
        written = list({id(f): f for r in reads if (f := self.written.get(r)) is not None}.values())
        copy = self.h2d.run(lambda: self._copy(jobs, "h2d"), [*after, *written]) if jobs else None
        self.loading.update(copies)
        self.reading.update(reads)
        for k in copies:
            self.lru[k[1] is None].move_to_end(k)
        if pages:                                             # D4: queued on the H2D lane behind earlier loads
            self.lane = max(self.lane, self.t) + pages / self.h2d_rate
        self.counts["loads"] += 1
        self.counts["loaded_blocks"] += len(keys)
        self.counts["load_bytes"] += pages
        return HostLoad([copy] if copy is not None else written, self.lane if pages > self.stall else self.t, out,
                        copies, reads, copy)

    def loaded(self, load: HostLoad, done) -> None:
        kind = "abandoned" if done is None else "fence"
        if kind in load.told:
            raise RuntimeError(f"state_tier: loaded(…, {kind}) twice for one load")
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
            if done is not None and done is not load and load.copy is not None:   # a step read it: was it in?
                self.counts["late_loads"] += not load.copy.done()
        load.told.add(kind)
        step = done is not None and done is not load
        if (fence := done if step else load.copy) is not None:       # a payload load's fences: the stores, held
            for r in load.reads:
                self.fences[r]["step" if step else "load"] = fence
        self.t = max(self.t, load.due)
        self.counts[kind] += 1

    def hit(self, keys, adopted) -> None:
        for key in keys:
            self.hits[key] += 1
            if (key, None) in self.copies:
                self.lru[True].move_to_end((key, None))
        for k in adopted:
            if k in self.copies:
                self.lru[False].move_to_end(k)

    def settled(self, payloads) -> None:
        self.tentative.difference_update(payloads)

    def drop(self, keys, spec: str | None = None) -> None:
        for key in keys:
            for s in [s for s in self.specs.get(key, ()) if spec is None or s == spec]:
                self._release((key, s))
            if spec is None:
                self.hits.pop(key, None)

    def dropped(self):
        out, self.pending = self.pending, []
        return out

    def clear(self) -> None:
        for k in list(self.copies):
            self._release(k)
        self.hits.clear()

    def close(self) -> None:
        pass

    # -- step hooks, stats ---------------------------------------------------------------------------------------
    def before_launch(self, plan):
        self.t += self.a + self.b * int(plan.n_query.sum()) + self.c * len(plan.slots)
        return [], []

    def stats(self) -> dict[str, int]:
        held = {k: p.capacity - len(self.free[k]) for k, p in self.host.items()}
        return dict(self.counts) | {f"host_units.{k}": n for k, n in held.items()}

    # -- storage -------------------------------------------------------------------------------------------------
    def _admits(self, key: int, evicting: bool) -> bool:
        if self.admit == "hit":
            return self.hits[key] >= self.min_hits
        return evicting or self.admit == "publish"

    def _add(self, k, host: dict[str, list[int]], parent: int | None) -> None:
        self.copies[k] = host
        self.specs.setdefault(k[0], set()).add(k[1])
        self.lru[k[1] is None][k] = None
        if k[1] is None:
            self.parent[k[0]] = parent
            self.kids.setdefault(parent, set()).add(k[0])   # also before its parent's (write-back stores leaves first)

    def _room(self, need: dict[str, int], pages: bool) -> dict[str, list[int]] | None:
        """Host units for one copy, evicting its kind's least recently used copies (pages: leaves only) that no load
        reads; None if that cannot make room."""
        while any(len(self.free[k]) < n for k, n in need.items()):
            victim = next((k for k in self.lru[pages] if not self.loading[k] and not (pages and self.kids.get(k[0]))
                           and any(name in need for name in self.copies[k])), None)
            if victim is None:
                return None
            self._release(victim)
            self.pending.append(victim)
            self.counts["evicted_pages" if pages else "evicted_payloads"] += 1
        return {k: [self.free[k].popleft() for _ in range(n)] for k, n in need.items()}

    def _release(self, k) -> None:
        for name, us in self.copies.pop(k).items():
            for u in us:
                if self.reading[name, u]:                 # a load not yet `loaded` reads it: kept till then
                    self.limbo.add((name, u))
                else:
                    self.free[name].append(u)
        del self.lru[k[1] is None][k]
        self.tentative.discard(k)
        self.specs[k[0]].discard(k[1])
        if not self.specs[k[0]]:
            del self.specs[k[0]]
        if k[1] is None:
            parent = self.parent.pop(k[0])
            self.kids[parent].discard(k[0])
            if not self.kids[parent]:
                del self.kids[parent]
            self.hits.pop(k[0], None)

    def _copy(self, jobs: list, direction: str) -> None:
        """Every job's units, row by row: one op call per device pool (D10)."""
        by_pool: dict[str, list[tuple[Any, Any]]] = collections.defaultdict(list)
        for name, dev_units, host_units in jobs:
            by_pool[name].append((dev_units, host_units))
        for name, pairs in by_pool.items():
            dev, host = self.dev[name], self.host[name]
            dev_rows, host_rows = (self._rows(p, [u for pair in pairs for u in pair[i]]) for i, p in ((0, dev),
                                                                                                    (1, host)))
            d, h = self._view(dev, False), self._view(host, True)
            if direction == "h2d":
                self.op[name, direction](h, host_rows, d, dev_rows)
            else:
                self.op[name, direction](d, dev_rows, h, host_rows)

    @staticmethod
    def _rows(pool: Any, units: list[int]) -> torch.Tensor:
        """Unit u of segment s is row s · capacity + u."""
        rows = np.arange(pool.segments, dtype=np.int64)[:, None] * pool.capacity + np.asarray(units, np.int64)[None]
        return torch.from_numpy(rows.ravel())

    @staticmethod
    def _view(pool: Any, host: bool) -> torch.Tensor:
        shape = (pool.segments * pool.capacity, pool.unit_bytes // pool.segments)
        return pool.device_view(torch.uint8, shape) if host else pool.view(torch.uint8, shape)
