"""K-M5 (milestone K step 0, docs/v1-kv-tiers.md §8): a CPU replay of the real core `PrefixIndex`, `StateManager` and
`Scheduler` on the gate traces, with a simulated tier and a step cost model in place of the device.

The replay drives the core the way `Engine.step` does at overlap depth 0 (schedule → `step_ops` → `launching` → the
tier's `before_launch` → `launched` → advance → commit: tokens, `publish`, `register`), on the RTX 5080's 27B lock:
paged q8_0 KV (16-token pages, 64-token blocks) and two recurrent specs shaped like the GDN's conv and ssm state
(in-step publishes on the 64-token grid, a checkpoint pool per spec). The pools hold no real bytes: units are what
count, and every byte figure comes from the unit sizes below.

The simulated tier implements `holon.api.StateTier` without bytes: a host level of `host.bytes` and an optional SSD
level below it, write-through page stores at insert (admission `publish`), mandatory payload stores at registration,
leaf-first LRU per level (pages and payloads in one recency order; payloads by store or adopt), copies redundant with
the host level dropped first on SSD, SSD write-back below the `host.reserve` watermark, its own drops through
`dropped()`, and loads due over virtual lanes (H2D, D2H, one NVMe lane) on a clock the step cost model advances.

Usage: python spikes/k0_tiers/m5/replay.py [--only agent_park] [--jobs 4] [--out evidence/k0_m5_replay_cpu_<date>.json]
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime
import functools
import heapq
import itertools
import json
import multiprocessing
import pathlib
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "bench"))
import workloads  # noqa: E402
from holon.api import HitLen, MemOp, Paged, Payload, SamplingParams, Seq, Slot, StateSpec, Status  # noqa: E402
from holon.core.memory import Ledger  # noqa: E402
from holon.core.prefix import PrefixIndex  # noqa: E402
from holon.core.sched import Scheduler  # noqa: E402
from holon.core.state import BoundSpec, StateManager  # noqa: E402

# -- the deployment: the RTX 5080's 27B lock (max_seqs 4, chunk_with_decode 512, page 16, q8_0 KV) -------------------
PAGE, BLOCK, MAX_SEQS, CHUNK, MAX_STEP, MAX_LEN = 16, 64, 4, 512, 2048, 49152
KV_TOKEN = 34_816                                    # q8_0 KV of the 16 attention layers, bytes per token
GDN = {"conv": 61_440 * 48, "ssm": 3 * 2**20 * 48}  # one checkpoint: 2.8 MiB + 144 MiB = 147 MiB (154 MB), 48 layers
CHECKPOINT = sum(GDN.values())
KV_TOKENS_AT_4 = 57_376          # the lock's KV tokens with 4 checkpoint units (bench evidence, `kv_tokens`)
PER_UNIT = CHECKPOINT // KV_TOKEN   # KV tokens one more checkpoint unit costs in VRAM (4,421)
GiB = 2**30
BARS = "evidence/bars_release-qwen3.8-27b-ud-iq3_s_rtx-5080_2026-10-16.json"
BENCH = "evidence/bench_release-qwen3.8-27b-ud-iq3_s_rtx-5080_2026-10-16.json"
RATES = {"h2d": 47.5e9, "d2h": 45.0e9, "read": 2.6e9, "write": 1.5e9}   # see `rates()`
SALT = "k-m5"
POLICIES = {"pe": "prompt_end", "pe+br": "prompt_end,branch", "pe+br+a2": "prompt_end,branch,answer:2"}
DONE = type("Done", (), {"wait": lambda self: None, "synchronize": lambda self: None})()   # a step's end


def cost_model() -> tuple[tuple[float, float, float], dict]:
    """a + b·tokens + c·rows (seconds) from the lock's bars and bench evidence: a batch-1 decode step (tg128), a
    batch-4 decode step (decode_b8 at batch 4, the bench's step wall p50) and a 512-token chunk at depth 8K
    (pp512 @ d8192: the agent traces resume at 4–19K)."""
    bars, bench = (json.loads((ROOT / f).read_text()) for f in (BARS, BENCH))
    one = 1 / bars["results"]["tg128"]["holon"]["statistic"]                    # a + b + c
    four = bench["summary"]["decode_b8"]["holon"]["decode"]["wall_ms_p50"] / 1e3  # a + 4b + 4c
    chunk = 512 / bars["results"]["pp512 @ d8192"]["holon"]["statistic"]         # a + 512b + c
    b = (chunk - one) / 511
    c = (four - one) / 3 - b
    a = one - b - c
    return (a, b, c), {"decode_b1_s": one, "decode_b4_s": four, "chunk512_d8192_s": chunk,
                       "a_s": a, "b_s_per_token": b, "c_s_per_row": c,
                       "measured_pp512_d0_s": 512 / bars["results"]["pp512"]["holon"]["statistic"]}


def rates() -> tuple[dict, str]:
    """H2D 47.5 GB/s (X's M1 blob rate; K-M2's best fragmented path is 0.99× of its blob), D2H 45 GB/s (K-M2's blob
    rate); NVMe from K-M4 when its evidence exists (the 2-thread O_DIRECT read of 557,056 B units, the sustained
    write past the SLC cache), else 2.6 / 1.5 GB/s placeholders."""
    out, source = dict(RATES), "placeholders (H2D X's M1 47.5 GB/s; D2H K-M2's 45 GB/s blob; NVMe 2.6 / 1.5 GB/s)"
    if m4 := sorted(ROOT.glob("evidence/k0_m4_nvme_rtx-5080_*.json")):
        m = json.loads(m4[-1].read_text())["m4"]
        out["read"] = 1e9 * next(r["gbps"] for r in m["rows"] if (r["kind"], r["mode"], r["shape"], r["threads"])
                                 == ("read", "direct", "unit_557056", 2))
        out["write"] = 1e9 * m["sustained"]["mean_gbps"]
        source = (f"H2D X's M1 47.5 GB/s; D2H K-M2's 45 GB/s blob; NVMe from {m4[-1].name}: 2-thread O_DIRECT read "
                  "of 557,056 B units, sustained write mean (its mixed-I/O rule is STOP: measured 0.76-0.84x of "
                  "the shared-lane model this replay uses)")
    return out, source


# -- state kinds -------------------------------------------------------------------------------------------------
class PagedKV:
    """std's paged kind."""

    def __init__(self, spec: StateSpec) -> None:
        self.spec = spec

    def units_needed(self, seq, new_tokens: int) -> int:
        return max(0, -(-(seq.num_computed + new_tokens) // PAGE) - len(seq.units(self.spec.name)))

    def capacity(self, seq) -> int:
        return len(seq.units(self.spec.name)) * PAGE


class Recurrent:
    """std's recurrent kind (`std/holon_std/state/recurrent/kind.py`) with the policies K adds (D8): `prompt_end`,
    `branch` (at `SeqView.branch_at`, in-step during prefill) and `answer:<k>` (a tentative payload at each aligned
    position a decode row reaches; the request keeps its last k). The kind takes the earliest position over its
    policies; a prefill step meeting two positions ends one grid before the second, as std's `every` does."""

    def __init__(self, spec: StateSpec, pool, payloads, policy: str) -> None:
        self.spec, self.pool, self.payloads, self.payload_pool = spec, pool, payloads, payloads.name
        names = policy.split(",")
        self.prompt_end, self.branch = "prompt_end" in names, "branch" in names
        self.keep_tentative = next((int(n[7:]) for n in names if n.startswith("answer:")), 0)

    def units_needed(self, seq, new_tokens: int) -> int:
        return 0

    def capacity(self, seq) -> int:
        return 1 << 62

    def _next(self, seq, start: int) -> int | None:
        at = []
        if self.prompt_end and (p := seq.prompt_len // BLOCK * BLOCK) > start:
            at.append(p)
        if self.branch and seq.branch_at > start:
            at.append(seq.branch_at)
        if self.keep_tentative and start >= seq.prompt_len:            # a decode row
            at.append((start // BLOCK + 1) * BLOCK)
        return min(at, default=None)

    def next_publish(self, seq) -> int | None:
        return None if self.payloads.capacity == 0 else self._next(seq, seq.num_computed)

    def cut(self, seq, start: int, n: int) -> int:
        at, grid, end = self.next_publish(seq), self.spec.in_step, start + n
        if at is not None and start < at < end:
            if not grid or (at - start) % grid:
                return at - start
            if (second := self._next(seq, at)) is not None and second <= end:
                return second - grid - start
        if at is not None and grid and end < len(seq.tokens) and end % grid and end // grid * grid > start:
            return end // grid * grid - start
        return n

    def publish(self, seq, at: int, in_step: bool = False):
        if self.payloads.num_free == 0:
            return None
        unit = self.payloads.alloc(1)
        payload = Payload((int(unit[0]),), self.payload_pool, at > seq.prompt_len)    # decode-produced: tentative
        if in_step:
            return payload, []
        return payload, [MemOp(self.pool.name, np.array(seq.units(self.spec.name)), self.payload_pool, unit,
                               "post_sample")]

    def hit_length(self, idx, upto: int) -> HitLen:
        for i in reversed(range(idx.blocks(upto))):
            if idx.payload(i) is not None:
                return HitLen((i + 1) * idx.block)
        return HitLen(0)

    def adopt(self, seq, payload: Payload) -> list[MemOp]:
        return [MemOp(self.payload_pool, np.array(payload.units), self.pool.name,
                      np.array(seq.units(self.spec.name)), "pre_forward")]


class Index(PrefixIndex):
    """The core's PrefixIndex unchanged, with counters around two of its methods."""

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.n: collections.Counter = collections.Counter()
        self.scans: list[tuple[float, int]] = []          # evict_payloads calls that scanned: (seconds, entries)

    def _evict(self, e) -> None:
        super()._evict(e)
        self.n["demoted" if e.key in self.lower else "dropped"] += 1

    def evict_payloads(self, pool: str, need: int) -> None:
        if self.payload_pools[pool].num_free >= need:
            return super().evict_payloads(pool, need)
        t0 = time.perf_counter()
        super().evict_payloads(pool, need)
        self.scans.append((time.perf_counter() - t0, len(self.entries)))


# -- time: the step cost model, and the simulated tier ------------------------------------------------------------
class Clock:
    """Simulated seconds: each launch costs a + b·tokens + c·rows; idle time jumps to the next release."""

    def __init__(self, cost: tuple[float, float, float]) -> None:
        self.t, self.cost = 0.0, cost

    def before_launch(self, plan):
        a, b, c = self.cost
        self.t += a + b * int(plan.n_query.sum()) + c * len(plan.slots)
        return [], []

    def idle(self, until: float) -> None:
        self.t = max(self.t, until)


class StoreFence:
    """A store's end on the D2H lane. The free rule waits for it at the next launch; at depth 1 (the locks) that is
    the launch after next, by when it has landed, so it never stalls the replay (depth 0); lateness is recorded."""

    __slots__ = ("tier", "due")

    def __init__(self, tier: SimTier, due: float) -> None:
        self.tier, self.due = tier, due

    def wait(self) -> None:
        self.tier.store_late = max(self.tier.store_late, self.due - self.tier.t)

    synchronize = wait


class SimLoad:
    """A load: gated until `due` on the tier's clock; a launch that waits for it starts once it lands."""

    def __init__(self, tier: SimTier, due: float, lands: float, payloads: dict, items: list) -> None:
        self.tier, self.due, self.lands, self.payloads, self.items = tier, due, lands, payloads, items
        self.told: set[str] = set()
        self.abandoned = False

    def wait(self) -> None:
        if not self.abandoned and self.lands > self.tier.t:
            self.tier.stalls.append(self.lands - self.tier.t)
            self.tier.t = self.lands

    synchronize = wait


@dataclasses.dataclass(eq=False)
class Item:
    nbytes: int
    parent: int | None            # pages: the parent block's key
    host: bool = False
    ssd: bool = False
    loading: int = 0              # loads not yet `loaded`
    tentative: bool = False       # a tentative payload: never written back to SSD (rev 8.1)


def timed(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kw):
        t0 = time.perf_counter()
        try:
            return fn(self, *args, **kw)
        finally:
            self.spent += time.perf_counter() - t0
    return wrapper


class SimTier(Clock):
    """`holon.api.StateTier` without bytes (module docstring). Decisions read logical byte counts only."""

    def __init__(self, cost, host_bytes: int, ssd_bytes: int, reserve: int, rates: dict, unit: dict[str, int],
                 stall_bytes: int = 0) -> None:
        super().__init__(cost)
        self.cap, self.reserve, self.rates, self.unit = {"host": host_bytes, "ssd": ssd_bytes}, reserve, rates, unit
        self.stall_bytes = stall_bytes
        self.items: dict[tuple[int, str | None], Item] = {}
        self.specs: dict[int, set] = collections.defaultdict(set)          # key -> its items' specs (None: pages)
        self.lru = {"host": collections.OrderedDict(), "ssd": collections.OrderedDict()}
        self.used = {"host": 0, "ssd": 0}
        self.used_payloads = {"host": 0, "ssd": 0}
        self.kids: collections.Counter = collections.Counter()       # page key -> child pages held (any level)
        self.host_kids: collections.Counter = collections.Counter()  # page key -> child pages on host
        self.lane = {"h2d": 0.0, "d2h": 0.0, "nvme": 0.0}
        self.pending: list = []
        self.n: collections.Counter = collections.Counter()
        self.peak: collections.Counter = collections.Counter()
        self.log: list = []           # loads since the step's start or the last admission: (tokens, bytes) by level
        self.spent, self.store_late = 0.0, 0.0
        self.stalls: list[float] = []
        self.phantom: list[float] = []   # how far a design-literal `loaded` would advance the clock for unread loads

    def attach(self, ctx) -> list:
        return []

    @timed
    def clock(self) -> float:
        return self.t

    # -- the protocol ---------------------------------------------------------------------------------------------
    @timed
    def store(self, blocks, after, evicting):
        accepted, moved = set(), 0
        for b in blocks:
            if b.units:
                k = (b.key, None)
                if k not in self.items:
                    nb = sum(len(us) * self.unit[spec] for spec, us in b.units.items())
                    if self._room("host", nb, keep=(b.parent, None)):
                        self._new(k, Item(nb, b.parent))
                        self._add(k, "host")
                        moved += nb
                        self.n["evict_stores" if evicting else "page_stores"] += 1
                    else:
                        self.n["refused_pages"] += 1
                if k in self.items:
                    accepted.add(k)
            for spec, p in b.payloads.items():
                k = (b.key, spec)
                if k not in self.items:
                    nb = self.unit[p.pool]
                    if not self._room("host", nb, keep=None, mandatory=True):
                        raise RuntimeError(f"payload store of {nb} B refused: the host level is too small")
                    self._new(k, Item(nb, None, tentative=p.tentative))
                    self._add(k, "host")
                    moved += nb
                    self.n["payload_stores"] += 1
                accepted.add(k)
        self.lane["d2h"] = max(self.lane["d2h"], self.t) + moved / self.rates["d2h"]
        self.n["store_bytes"] += moved
        return accepted, StoreFence(self, self.lane["d2h"])

    @timed
    def load(self, keys, units, payloads, after):
        ks = [(k, None) for k in keys] + list(payloads)
        if missing := [k for k in ks if k not in self.items]:     # the core loads only what the tier holds
            raise RuntimeError(f"the core loads {missing[0]}, a copy the tier dropped"
                               + (" (reported by dropped(), not yet polled)" if missing[0] in self.pending else ""))
        items = [self.items[k] for k in ks]
        for it in items:
            it.loading += 1
        tokens, nbytes, reads = collections.Counter(), collections.Counter(), 0
        for k, it in zip(ks, items, strict=True):
            level = "ram" if it.host else "ssd"
            nbytes[level] += it.nbytes
            tokens[level] += BLOCK if k[1] is None else 0
            if not it.host:                          # read into host units, then H2D
                reads += it.nbytes
                if not self._room("host", it.nbytes, keep=None, mandatory=True):
                    raise RuntimeError("no host units for an SSD read")
                self._add(k, "host")
            self._touch(k)
        ready = self.t
        if reads:
            ready = self.lane["nvme"] = max(self.lane["nvme"], self.t) + reads / self.rates["read"]
        total = sum(nbytes.values())
        lands = self.lane["h2d"] = max(self.lane["h2d"], ready) + total / self.rates["h2d"]
        self.log.append((tokens, nbytes))
        for level in nbytes:
            self.n[f"load_bytes_{level}"] += nbytes[level]
            self.n[f"load_tokens_{level}"] += tokens[level]
        self.n["loads"] += 1
        due = self.t if total <= self.stall_bytes else lands
        return SimLoad(self, due, lands, {k: Payload((0,), "plugin.simtier.host") for k in payloads}, items)

    @timed
    def loaded(self, load, done) -> None:
        kind = "none" if done is None else "fence"
        assert kind not in load.told, f"loaded(…, {kind}) twice"
        if not load.told:
            for it in load.items:
                it.loading -= 1
        load.told.add(kind)
        if done is None:
            load.abandoned = True
            self.n["abandoned"] += 1
        elif done is load:
            self.n["waiterless"] += 1
        else:
            self.n["loaded"] += 1
        if done is None or done is load:
            self.phantom.append(max(0.0, load.lands - self.t))

    @timed
    def hit(self, keys, adopted) -> None:
        for key in keys:
            if (key, None) in self.items:
                self._touch((key, None))
        for k in adopted:
            if k in self.items:
                self._touch(k)

    @timed
    def settled(self, payloads) -> None:
        for k in payloads:
            if k in self.items:
                self.items[k].tentative = False
                self.n["settled"] += 1

    @timed
    def drop(self, keys, spec=None) -> None:
        for key in keys:
            for s in [s for s in self.specs.get(key, ()) if spec is None or s == spec]:
                if s is None and self.items[key, s].loading:
                    self.n["contract_dropped_while_loading"] += 1
                self._forget((key, s), report=False)

    @timed
    def dropped(self) -> list:
        out, self.pending = self.pending, []
        return out

    def clear(self) -> None:
        for k in list(self.items):
            self._forget(k, report=False)

    def close(self) -> None:
        pass

    def before_launch(self, plan):
        super().before_launch(plan)
        if self.cap["ssd"]:                          # the write-back watermark (D11)
            while self.cap["host"] - self.used["host"] < self.reserve:
                k = self._victim("host", None)
                if k is None:
                    break
                self._evict("host", k)
                self.n["watermark_demotions"] += 1
        return [], []

    # -- storage --------------------------------------------------------------------------------------------------
    def _new(self, k, it: Item) -> None:
        self.items[k] = it
        self.specs[k[0]].add(k[1])
        if k[1] is None and it.parent is not None:
            self.kids[it.parent] += 1

    def _add(self, k, level: str) -> None:
        it = self.items[k]
        setattr(it, level, True)
        self.used[level] += it.nbytes
        self.used_payloads[level] += it.nbytes if k[1] is not None else 0
        self.lru[level][k] = None
        if level == "host" and k[1] is None and it.parent is not None:
            self.host_kids[it.parent] += 1
        self.peak[level] = max(self.peak[level], self.used[level])
        self.peak[f"{level}_payloads"] = max(self.peak[f"{level}_payloads"], self.used_payloads[level])
        self.peak[f"{level}_pages"] = max(self.peak[f"{level}_pages"], self.used[level] - self.used_payloads[level])

    def _remove(self, k, level: str) -> None:
        it = self.items[k]
        setattr(it, level, False)
        self.used[level] -= it.nbytes
        self.used_payloads[level] -= it.nbytes if k[1] is not None else 0
        del self.lru[level][k]
        if level == "host" and k[1] is None and it.parent is not None:
            self.host_kids[it.parent] -= 1

    def _forget(self, k, report: bool) -> None:
        it = self.items[k]
        for level in ("host", "ssd"):
            if getattr(it, level):
                self._remove(k, level)
        del self.items[k]
        self.specs[k[0]].discard(k[1])
        if not self.specs[k[0]]:
            del self.specs[k[0]]
        if k[1] is None and it.parent is not None:
            self.kids[it.parent] -= 1
        if report:
            self.pending.append(k)
            self.n["tier_dropped_pages" if k[1] is None else "tier_dropped_payloads"] += 1

    def _touch(self, k) -> None:
        for level in ("host", "ssd"):
            if k in self.lru[level]:
                self.lru[level].move_to_end(k)

    def _victim(self, level: str, keep, any_: bool = False):
        """The least recently used copy at `level` that may go: not being loaded, not `keep`; a leaf at that level
        (pages with no child page there; on SSD, none in the tier); on SSD, copies the host level also holds first."""
        lru = self.lru[level]
        if level == "ssd":
            for k in lru:
                if self.items[k].host and not self.items[k].loading:
                    return k
        kids = self.host_kids if level == "host" else self.kids
        for k in lru:
            if k != keep and not self.items[k].loading and (any_ or k[1] is not None or not kids[k[0]]):
                return k
        return None

    def _room(self, level: str, nb: int, keep, mandatory: bool = False) -> bool:
        while self.used[level] + nb > self.cap[level]:
            k = self._victim(level, keep) or (self._victim(level, keep, any_=True) if mandatory else None)
            if k is None:
                return False
            self._evict(level, k)
        return True

    def _evict(self, level: str, k) -> None:
        """Host: write back to SSD (a copy SSD holds already is just freed), else drop; SSD: drop unless on host."""
        it = self.items[k]
        self._remove(k, level)
        if (level == "host" and self.cap["ssd"] and not it.ssd and not it.tentative
                and self._room("ssd", it.nbytes, keep=None)):
            self.lane["nvme"] = max(self.lane["nvme"], self.t) + it.nbytes / self.rates["write"]
            self._add(k, "ssd")
            self.n["writebacks"] += 1
            self.n["writeback_bytes"] += it.nbytes
            self.n["writeback_payload_bytes" if k[1] is not None else "writeback_page_bytes"] += it.nbytes
        elif not it.host and not it.ssd:
            self._forget(k, report=True)
        elif level == "host":
            self.n["host_freed_redundant"] += 1


# -- the replay ---------------------------------------------------------------------------------------------------
@dataclasses.dataclass(eq=False)
class Req:
    sid: int
    turn: int
    prompt: int
    output: int
    release: float
    gen: list[int]
    history: list[int]                         # its prompt
    first: float | None = None
    last: float | None = None
    cached: int | None = None                  # its first admission's hit
    tier_tokens: collections.Counter = dataclasses.field(default_factory=collections.Counter)
    load_bytes: collections.Counter = dataclasses.field(default_factory=collections.Counter)


@dataclasses.dataclass(frozen=True)
class Config:
    trace: str
    name: str
    policy: str
    staging: int
    host_gib: float = 0.0                     # 0: no tier (VRAM-only, today)
    ssd_gib: float = 0.0


def check(sim: Sim) -> None:
    """D6's invariants (as core/tests/test_state_liveness.py checks them) plus the core's and the tier's agreement:
    every copy the index counts on is in the tier or among its drops not yet polled."""
    idx, every = sim.index, sim.index.entries | sim.index.lower
    assert not idx.entries.keys() & idx.lower.keys()
    children = collections.defaultdict(set)
    for e in every.values():
        assert e.parent is None or e.parent in every, "an entry's parent is gone"
        if e.parent is not None:
            children[e.parent].add(e.key)
    for e in every.values():
        assert e.children == children[e.key], "children differ from a recount"
        assert e.kids == sum(c in idx.entries for c in e.children), "kids differ from a recount"
    for e in idx.entries.values():
        assert e.parent is None or e.parent in idx.entries, "resident entries are not prefix-closed"
    held = [u for e in idx.entries.values() for u in e.units.get("kv", ())]
    assert sorted(np.flatnonzero(idx.cached("kv"))) == sorted(held), "the cached mask differs from the units"
    if (tier := sim.tier) is not None:
        gone = set(tier.pending)
        for e in every.values():
            if e.lower:
                assert (e.key, None) in tier.items or (e.key, None) in gone, "a lower entry the tier does not hold"
            for spec, p in e.payloads.items():
                assert p.units or (e.key, spec) in tier.items or (e.key, spec) in gone, "a payload the tier lost"


class Sim:
    def __init__(self, cfg: Config, cost, rates: dict, checks: bool = False) -> None:
        self.cfg, self.checks = cfg, checks
        kv_tokens = KV_TOKENS_AT_4 - (cfg.staging - 4) * PER_UNIT
        self.kv_tokens = kv_tokens // PAGE * PAGE
        ledger = Ledger(torch.device("cpu"))
        kv = ledger.request("state.kv", unit_bytes=8, min_units=kv_tokens // PAGE, tokens_per_unit=PAGE,
                            null_unit=True)
        kv_spec = StateSpec("kv", "paged@1", Paged(8, PAGE), (0,), "gqa")
        bound, unit = [BoundSpec(kv_spec, kv, PagedKV(kv_spec))], {"kv": PAGE * KV_TOKEN}
        for name, nbytes in GDN.items():
            spec = StateSpec(name, "recurrent@1", Slot(48), (0,), f"gdn.{name}", segments=48, alignment=BLOCK,
                             in_step=BLOCK)
            pool = ledger.request(f"state.{name}", unit_bytes=48, segments=48, min_units=MAX_SEQS + 1,
                                  pref_units=MAX_SEQS + 1)
            payloads = ledger.plugin_pool(f"{name}.checkpoints", unit_bytes=48, segments=48, min_units=cfg.staging,
                                          pref_units=cfg.staging)
            unit[payloads.name] = nbytes
            bound.append(BoundSpec(spec, pool, Recurrent(spec, pool, payloads, cfg.policy)))
        ledger.materialize(None)
        self.payload_pools = [b.kind.payloads for b in bound[1:]]
        self.index = Index(1, BLOCK, {"kv": kv}, ledger.pools)
        self.tier = None
        if cfg.host_gib:
            reserve = MAX_LEN * KV_TOKEN + CHECKPOINT if cfg.ssd_gib else 0   # one max_model_len load
            self.tier = SimTier(cost, int(cfg.host_gib * GiB), int(cfg.ssd_gib * GiB), reserve, rates, unit)
            self.index.restore(self.tier.attach(None))
            self.index.tier = self.tier
        self.clock = self.tier or Clock(cost)
        self.state = StateManager(bound, MAX_SEQS, MAX_LEN, torch.device("cpu"), on_admit=self._admitted,
                                  index=self.index)
        self.sched = Scheduler(self.state, MAX_STEP, MAX_LEN, chunk_with_decode=CHUNK,
                               clock=self.tier.clock if self.tier is not None else (lambda: 0.0))
        self.reqs: dict[Seq, Req] = {}
        self.readmissions = 0
        self.steps: list[tuple] = []          # (core s, tier s, rows, tokens)
        self.staged: list[int] = []           # payload units staged per step (the larger spec)
        self.pubs: list[int] = []             # publishes per step (the larger spec)

    def _admitted(self, seq: Seq) -> None:
        req = self.reqs[seq]
        log = self.tier.log if self.tier is not None else []
        if req.cached is None:
            req.cached = seq.num_computed
            for tokens, nbytes in log:
                req.tier_tokens.update(tokens)
                req.load_bytes.update(nbytes)
        else:
            self.readmissions += 1
        log.clear()

    def step(self) -> list[Req]:
        """One engine step at depth 0; the requests it finished."""
        tier, state, perf = self.tier, self.state, time.perf_counter
        spent0 = tier.spent if tier is not None else 0.0
        t0 = perf()
        if tier is not None:
            self.index.poll()
            tier.log.clear()
        scheduled = self.sched.schedule()
        if scheduled is None:
            raise RuntimeError("the scheduler made no progress with requests pending")
        plan, seqs = scheduled
        ops = state.step_ops(plan, seqs)
        loads = state.launching(seqs)
        t1 = perf()
        staged = max(p.capacity - p.num_free for p in self.payload_pools)
        self.clock.before_launch(plan)            # the executor's launch: the tier's step hook, the step's cost
        now = self.clock.t
        t2 = perf()
        state.launched(loads, DONE)
        state.unpin(ops.pins)
        rows = list(zip(seqs, plan.n_query.tolist(), plan.n_logits.tolist(), strict=True))
        for seq, n, m in rows:
            seq.num_computed += n
            if m:
                seq.pending += m
                if seq.status is not Status.RUNNING:
                    self.sched.finished_prefill(seq)
        emitted, finished, chains = [], [], state.chain
        if tier is not None:
            self.index.poll()                     # Engine._commit: what the tier dropped in its hooks
        for seq, _, m in rows:                    # Engine._commit
            if not m:
                if seq.status is not Status.FINISHED and self._completes(seq, chains):
                    state.publish(seq)
                continue
            seq.pending -= m
            if seq.status is Status.FINISHED:
                continue
            req = self.reqs[seq]
            seq.tokens.append(req.gen[len(seq.tokens) - seq.prompt_len])
            done = len(seq.tokens) - seq.prompt_len >= req.output or len(seq.tokens) >= MAX_LEN
            if self._completes(seq, chains):
                state.publish(seq)
            if done:
                self.sched.finish(seq, "length")
                finished.append(req)
            emitted.append(req)
        for seq, spec, at, payload in ops.publishes:
            state.register(seq, spec, at, payload, DONE)
        t3 = perf()
        tier_s = (tier.spent - spent0) if tier is not None else 0.0
        self.steps.append(((t1 - t0) + (t3 - t2) - tier_s, tier_s, len(rows), int(plan.n_query.sum())))
        self.staged.append(staged)
        self.pubs.append(max(sum(1 for _, s, _, _ in ops.publishes if s == name) for name in GDN))
        for req in emitted:
            req.first = now if req.first is None else req.first
            req.last = now
        if self.checks:
            check(self)
        return finished

    @staticmethod
    def _completes(seq: Seq, chains: dict) -> bool:
        chain = chains.get(seq.slot)
        return chain is not None and min(seq.num_computed, len(seq.tokens)) // BLOCK > len(chain)

    def run(self, sessions: list[dict]) -> list[Req]:
        """bench/driver.py's release rules in simulated time: arrival, `gap` after the turn it continues, `after`,
        and round-robin per `client`; generated tokens resent verbatim in the next prompt."""
        system = workloads.tokens(SALT, "system", max(s["system"] for s in sessions))
        by_id = {s["id"]: s for s in sessions}
        new = {s["id"]: workloads.appends(SALT, s) for s in sessions}
        clients: dict = {}
        for s in sessions:
            clients.setdefault(s.get("client"), []).append(s["id"])
        clients.pop(None, None)
        ready = [(s["arrival"], s["id"], 0, []) for s in sessions if "after" not in s
                 and ("client" not in s or clients[s["client"]][0] == s["id"])]
        heapq.heapify(ready)
        followers = {s["after"]: s for s in sessions if "after" in s}
        parked: dict[int, tuple[int, list[int]]] = {}
        done: list[Req] = []
        rid = 0
        while True:
            while ready and ready[0][0] <= self.clock.t:
                release, sid, i, history = heapq.heappop(ready)
                s, t = by_id[sid], by_id[sid]["turns"][i]
                prompt = (history or system[:s["system"]]) + new[sid][i]
                seq = Seq(rid=str(rid), tokens=list(prompt),
                          params=SamplingParams(max_tokens=t["output"], ignore_eos=True))
                rid += 1
                self.reqs[seq] = Req(sid, i, len(prompt), t["output"], release,
                                     workloads.tokens(SALT, f"gen.{sid}.{i}", t["output"]), prompt)
                self.sched.add(seq)
            if not (self.sched.waiting or self.sched.running):
                if not ready:
                    break
                self.clock.idle(ready[0][0])
                continue
            for r in self.step():
                done.append(r)
                s = by_id[r.sid]
                history = r.history + r.gen
                if "client" in s:
                    parked[r.sid] = (r.turn + 1, history)
                    order = clients[s["client"]]
                    at = order.index(r.sid)
                    for sid in order[at + 1:] + order[:at + 1]:
                        turn_, hist = parked.get(sid, (0, []))
                        if turn_ < len(by_id[sid]["turns"]):
                            heapq.heappush(ready, (r.last + by_id[sid]["turns"][turn_]["gap"], sid, turn_, hist))
                            break
                elif r.turn + 1 < len(s["turns"]):
                    heapq.heappush(ready, (r.last + s["turns"][r.turn + 1]["gap"], r.sid, r.turn + 1, history))
                elif (f := followers.get(r.sid)) is not None:
                    heapq.heappush(ready, (r.last + f["arrival"], f["id"], 0, []))
        return done


# -- metrics ------------------------------------------------------------------------------------------------------
def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))]


def summarize(sim: Sim, reqs: list[Req], wall: float) -> dict:
    tier, idx = sim.tier, sim.index
    ttft = lambda rs: [r.first - r.release for r in rs]                                     # noqa: E731
    group = lambda rs: {"n": len(rs), "ttft_p50_s": pct(ttft(rs), 50), "ttft_p90_s": pct(ttft(rs), 90)}  # noqa: E731
    resumes = [r for r in reqs if r.turn > 0]
    out = {
        "config": dataclasses.asdict(sim.cfg), "kv_tokens_vram": sim.kv_tokens, "requests": len(reqs),
        "ttft": group(reqs), "first_turns": group([r for r in reqs if r.turn == 0]),
        "turns_1_5": group([r for r in reqs if r.turn < 5]), "turns_6_plus": group([r for r in reqs if r.turn >= 5]),
        "resumes": {**group(resumes),
                    "recomputed_tokens_p50": pct([r.prompt - r.cached for r in resumes], 50),
                    "recomputed_tokens_p90": pct([r.prompt - r.cached for r in resumes], 90),
                    "load_bytes_p50": pct([sum(r.load_bytes.values()) for r in resumes], 50)},
        "throughput_tok_s": sum(r.output for r in reqs) / (max(r.last for r in reqs) - min(r.release for r in reqs)),
        "makespan_s": max(r.last for r in reqs),
        "prompt_tokens": sum(r.prompt for r in reqs), "hit_tokens": {
            "vram": sum(r.cached - sum(r.tier_tokens.values()) for r in reqs),
            "ram": sum(r.tier_tokens["ram"] for r in reqs), "ssd": sum(r.tier_tokens["ssd"] for r in reqs)},
        "resident_share": {
            name: pct([1 - sum(r.tier_tokens.values()) / r.cached for r in rs if r.turn > 0 and r.cached], 50)
            for name, rs in (("turns_1_5", [r for r in reqs if r.turn < 5]),
                             ("turns_6_plus", [r for r in reqs if r.turn >= 5]))},
        "index": dict(idx.n), "preemptions": sim.sched.preemptions, "readmissions": sim.readmissions,
        "dropped_publishes": sim.state.dropped_publishes,
        "staging_peak_depth0": max(sim.staged), "staging_demand_depth1_est": max(
            (a + b for a, b in itertools.pairwise(sim.pubs)), default=0),
        "parked_end": {"resident_entries": len(idx.entries), "lower_entries": len(idx.lower),
                       "tokens": (len(idx.entries) + len(idx.lower)) * BLOCK},
        "steps": len(sim.steps), "replay_wall_s": wall,
    }
    core = [s[0] * 1e3 for s in sim.steps]
    out["host_ms"] = {"core_p50": pct(core, 50), "core_p99": pct(core, 99), "core_mean": statistics.fmean(core),
                      "decode_steps_core_p50": pct([s[0] * 1e3 for s in sim.steps if s[3] == s[2]], 50),
                      "decode_steps_core_p99": pct([s[0] * 1e3 for s in sim.steps if s[3] == s[2]], 99)}
    if idx.scans:
        us = [s[0] * 1e6 for s in idx.scans]
        out["evict_payloads_scans"] = {"calls": len(us), "us_p50": pct(us, 50), "us_p99": pct(us, 99),
                                       "us_max": max(us), "entries_p50": pct([s[1] for s in idx.scans], 50),
                                       "share_of_core_time": sum(us) / 1e3 / sum(core)}
    if tier is not None:
        sim_ms = [s[1] * 1e3 for s in sim.steps]
        out["host_ms"] |= {"sim_tier_p50": pct(sim_ms, 50), "sim_tier_p99": pct(sim_ms, 99)}
        out["tier"] = dict(tier.n) | {
            "peak_host_bytes": tier.peak["host"], "peak_host_page_bytes": tier.peak["host_pages"],
            "peak_host_payload_bytes": tier.peak["host_payloads"], "peak_ssd_bytes": tier.peak["ssd"],
            "end_host_bytes": tier.used["host"], "end_ssd_bytes": tier.used["ssd"],
            "end_items": len(tier.items), "load_stalls": len(tier.stalls), "load_stall_s": sum(tier.stalls),
            "load_stall_p90_ms": pct([s * 1e3 for s in tier.stalls], 90),
            "store_fence_late_max_ms": tier.store_late * 1e3,
            "phantom_catchup_n": sum(1 for p in tier.phantom if p > 0),
            "phantom_catchup_s": sum(tier.phantom)}
    return out


def run_config(cfg: Config, cost, rates_, checks: bool = False) -> dict:
    sessions = workloads.load(ROOT / "bench/traces" / f"{cfg.trace.removesuffix('_2t')}.jsonl")
    if cfg.trace.endswith("_2t"):
        sessions = [{**s, "turns": s["turns"][:2]} for s in sessions]
    sim = Sim(cfg, cost, rates_, checks)
    t0 = time.perf_counter()
    reqs = sim.run(sessions)
    return summarize(sim, reqs, time.perf_counter() - t0)


def configs(traces: list[str]) -> list[Config]:
    out = []
    for trace in traces:
        for p in POLICIES:
            out.append(Config(trace, f"vram/{p}", POLICIES[p], 4))
            for gib in (8, 16, 32):
                out.append(Config(trace, f"ram{gib}/{p}", POLICIES[p], 5, gib))
        full = POLICIES["pe+br+a2"]
        out += [Config(trace, "ram4+ssd/pe+br+a2", full, 5, 4, 100), Config(trace, "ram8+ssd/pe+br+a2", full, 5, 8,
                                                                              100),
                Config(trace, "ram4+ssd8/pe+br+a2", full, 5, 4, 8)]        # a full SSD: the tier drops
        if trace == "agent_park":
            out += [Config(trace, f"ram16/pe+br+a2/s{s}", full, s, 16) for s in (4, 6, 8)]
    return out


def verdicts(results: list[dict]) -> dict:
    """The rows the predictions name: 3c (agent_park, tiered vs VRAM-only today), 3e (claude_code 2 turns, branch on
    vs off, tiers on in both), staging, host time."""
    by = {(r["config"]["trace"], r["config"]["name"]): r for r in results}
    ratio = lambda a, b: None if a is None or not b else round(a / b, 4)                       # noqa: E731
    out: dict = {}
    if (base := by.get(("agent_park", "vram/pe"))) is not None:
        out["3c"] = {name: {
            "ttft_p50": ratio(r["ttft"]["ttft_p50_s"], base["ttft"]["ttft_p50_s"]),
            "ttft_p90": ratio(r["ttft"]["ttft_p90_s"], base["ttft"]["ttft_p90_s"]),
            "turns_1_5_ttft_p50": ratio(r["turns_1_5"]["ttft_p50_s"], base["turns_1_5"]["ttft_p50_s"]),
            "turns_6_plus_ttft_p50": ratio(r["turns_6_plus"]["ttft_p50_s"], base["turns_6_plus"]["ttft_p50_s"]),
            "throughput": ratio(r["throughput_tok_s"], base["throughput_tok_s"])}
            for (trace, name), r in by.items() if trace == "agent_park" and name != "vram/pe"}
    out["3e"] = {}
    for (trace, name), r in by.items():
        off = by.get((trace, name.replace("/pe+br+a2", "/pe").replace("/pe+br", "/pe")))
        if trace == "claude_code_2t" and "+br" in name and off is not None:
            out["3e"][f"{name} vs {off['config']['name']}"] = ratio(r["first_turns"]["ttft_p50_s"],
                                                                    off["first_turns"]["ttft_p50_s"])
    tiered = [r for r in results if r["config"]["host_gib"]]
    out["staging"] = {"dropped_publishes_max": max(r["dropped_publishes"] for r in tiered),
                      "peak_depth0_max": max(r["staging_peak_depth0"] for r in tiered),
                      "demand_depth1_est_max": max(r["staging_demand_depth1_est"] for r in tiered)}
    out["host_ms"] = {f"{t} {n}": {k: round(v, 4) for k, v in by[t, n]["host_ms"].items()}
                      for t, n in by if n in ("vram/pe", "ram16/pe+br+a2")}
    return out


PREDICTIONS = json.loads((ROOT / "evidence/k0_m5_replay_cpu_2026-10-18.json").read_text())[
    "predictions_fixed_before_measuring"] if (ROOT / "evidence/k0_m5_replay_cpu_2026-10-18.json").exists() else {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=["agent_park", "claude_code", "claude_code_2t", "agent"])
    ap.add_argument("--configs", nargs="*", help="config names to run (default: all)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", action="store_true", help="check the index invariants after every step (slow)")
    ap.add_argument("--tier", nargs=2, type=float, metavar=("HOST_GIB", "SSD_GIB"),
                    help="instead of the matrix: one tiered config per trace (prompt_end,branch,answer:2)")
    args = ap.parse_args()
    cost, derivation = cost_model()
    rates_, source = rates()
    todo = [c for c in configs(args.only) if not args.configs or c.name in args.configs]
    if args.tier:
        host, ssd = args.tier
        todo = [Config(t, f"ram{host:g}+ssd{ssd:g}/pe+br+a2", POLICIES["pe+br+a2"], 5, host, ssd) for t in args.only]
    with multiprocessing.get_context("fork").Pool(args.jobs) as pool:
        results = pool.starmap(run_config, [(c, cost, rates_, args.check) for c in todo])
    for r in results:
        c = r["config"]
        print(f"{c['trace']:15s} {c['name']:22s} ttft p50 {r['ttft']['ttft_p50_s']:.3f}"
              f" p90 {r['ttft']['ttft_p90_s']:.3f} | thr {r['throughput_tok_s']:.1f} | hit {r['hit_tokens']}"
              f" | core p50 {r['host_ms']['core_p50']:.3f} p99 {r['host_ms']['core_p99']:.3f} ms"
              f" | drops {r['dropped_publishes']}")
    if args.out:
        base = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout
        record = {
            "measurement": "K-M5 CPU replay of the core prefix index, state manager and scheduler with a simulated "
                           "tier (docs/v1-kv-tiers.md §8)",
            "date": datetime.date.today().isoformat(),
            "command": "python spikes/k0_tiers/m5/replay.py " + " ".join(sys.argv[1:]),
            "core": {"base_commit": base.strip(), "state": "uncommitted step-1 seams (KS1-KS5) over the base"},
            "predictions_fixed_before_measuring": PREDICTIONS,
            "deployment": {"lock": "Qwen3.8-27B UD-IQ3_S rtx-5080 (10.lock): max_seqs 4, chunk_with_decode 512, "
                                   "max_step_tokens 2048, page 16, max_model_len 49152, q8_0 KV",
                           "kv_token_bytes": KV_TOKEN, "checkpoint_bytes": GDN, "kv_tokens_at_4_units": KV_TOKENS_AT_4,
                           "kv_tokens_per_staging_unit": PER_UNIT, "block_tokens": BLOCK, "overlap_depth": 0},
            "cost_model": {"sources": [BARS, BENCH], **derivation}, "rates": rates_, "rates_source": source,
            "verdicts": verdicts(results),
            "results": results,
        }
        pathlib.Path(args.out).write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
