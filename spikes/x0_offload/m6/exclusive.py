"""X step 5's input (design D10): M6's replay with an exclusive RAM tier (CPU only, numpy).

An expert lives in exactly one of VRAM, RAM or SSD. VRAM holds a fraction of all experts (the per-layer quotas and
warm start of `replay.quotas`), RAM a fraction of the rest (seeded by the profile's next ranks), SSD the remainder
(the file is always valid, so leaving RAM for SSD costs no I/O). A step's distinct decode (layer, expert) keys are
served from VRAM (hit), RAM (H2D gather) or SSD (read into pinned host memory, then H2D). Variants:

  f11_ssd       F11 (`lfu`, M6's rule) with SSD-only victims: the in-graph admission overwrites its victim, which
                becomes SSD-only; the admitted expert's RAM slot is released and reusable `free_lag` steps later; a
                host policy promotes hot SSD-only experts into RAM (free slots first, then over RAM's coldest when
                strictly hotter), deciding after commit c and landing at c + `promo_lag`. `promo` = background: the
                promotions' SSD reads share the 2-worker reader with demand reads (budget, stalls and waits below);
                off: no promotion; free: promotions cost nothing (a bound)
  f11_deferred  F11 with deferred admission: the miss is staged this step (no admission in place); the pairing's
                victim is unmapped and demoted D2H into RAM (a free slot, else over RAM's coldest when hotter, else
                SSD-only); the miss's copy goes D2D into the victim's slot and its map flips at c + `flip` (default
                2). In the window the miss is still read from its old tier (its RAM slot is kept until the flip, then
                released). Candidates exclude pending misses
  f10_spares    F10 promote-on-fetch (Maya, M6's `promote_on_fetch`, 3 spares per layer): misses in route order land
                in the spares; each step boundary refills them by evicting the layer's LRU residents, each demoted D2H
                into RAM if hotter than RAM's victim (a free slot ranks coldest), else SSD-only. A promoted miss's
                RAM slot is released as under f11_ssd
  lfu_allram    `lfu` with RAM holding every expert (the reference; RAM fraction ignored)

A demotion (D2H) lands for reads at the next step (D10: it precedes the next H2D on the same copy stream); until a
copy lands its expert's tier code says SSD. Sensitivities: `_nopromo` / `_freepromo` (f11_ssd's promotion off /
free), `_flip1` (f11_deferred's map flips at c+1: no window), `_dlag2` (demotions land at c+2, as if the host picked
their RAM slots), `_rlag2` (f10's refill is a host edit applied before launch c+2, D4), `_keep` (a demand SSD read
that stays out of VRAM is kept in RAM, in a free slot or over RAM's coldest when hotter: the read already landed in
host memory, so it costs no I/O; Maya's rule sends it back to disk).

RAM heat is the decayed count of F11 (x0.915 per step, +1 per routed decode entry) for every variant
(`--ram-decay` sets it apart).

Modelled time per token (serial; per step, summed over steps / decode tokens):
  GPU     (dense + every routed expert's bytes) / GPU_EFF, GPU_EFF = a batch-1 token's bytes / 4.5 ms (the RTX 5080's
          all-resident estimate, design §13), so a batch-1 step costs 4.5 ms whatever the placement
  H2D     (RAM + SSD misses' bytes) / 47.5 GB/s (M1's gather)
  D2H     demotion bytes / 47.5 GB/s
  SSD     demand bytes / 2.6 GB/s (M7: 2 workers saturate; an expert is 3 file extents, so both workers are fed);
          `ms_ssd_latency` instead floors each layer with an SSD read at one expert's depth-1 read time (M7's 0.77 ms
          per 1.46 MB and 1.30 ms per 2.8 MB, interpolated: 1.24 / 1.40 ms for 2.65 / 3.06 MB)
  promo   f11_ssd background: the reader drains promotions during the step's non-SSD time; the host issues one only
          while the backlog fits that idle time over the `promo_lag` - 1 steps before landing (this step's as the
          estimate); a demand read of a promoted expert completes its promotion; a promotion still unread at its
          landing step stalls that step for its rest; a step with backlog pays, per layer with demand SSD reads, the
          residual of an in-flight extent on the first of 2 busy workers (a third of an extent's depth-1 read, 0.19 ms)

    python spikes/x0_offload/m6/exclusive.py --traces local/tmp/x0-m6 --out <json> [--jobs 24] [--ram-decay 0.99]
    python spikes/x0_offload/m6/exclusive.py --report <json> [<ram-decay jsons>] [--evidence <record>]
    python spikes/x0_offload/m6/exclusive.py --selfcheck
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from replay import Decode, _evict_order, _first_n_per_layer, _pair_swap, decode_steps, load, quotas  # noqa: E402

# The vehicle (Qwen3-30B-A3B Q4_K_M), from the file's tensor table (replay.expert_bytes; M6's record): gate / up Q4_K,
# down Q6_K in layers 0-5, every third from 8 to 41 and 42-47, Q4_K elsewhere.
Q6_DOWN = set(range(6)) | set(range(8, 42, 3)) | set(range(42, 48))
EBYTES = np.array([3_059_712.0 if layer in Q6_DOWN else 2_654_208.0 for layer in range(48)])
DENSE = 822_523_904.0
T_B1 = 4.5e-3                          # all-resident GPU time per batch-1 token, RTX 5080 (design §13 roofline)
GPU_EFF = (DENSE + 8 * EBYTES.sum()) / T_B1
PCIE = 47.5e9                          # M1 gather, H2D; D2H demotion at the same rate
SSD = 2.6e9                            # M7, 2 workers
LAT = lambda b: 0.77e-3 + (b - 1.46e6) * (1.30e-3 - 0.77e-3) / (2.8e6 - 1.46e6)  # noqa: E731  depth-1 read time
EXTENT_WAIT = LAT(EBYTES.mean() / 3) / 3  # residual of an in-flight extent, first of 2 busy workers

VRAM = (0.10, 0.25, 0.47)
RAMF = (0.25, 0.50, 0.75)
VARIANTS = {"f11_ssd": {"promo": "background"}, "f11_ssd_nopromo": {"promo": "off"},
            "f11_ssd_freepromo": {"promo": "free"}, "f11_deferred": {"flip": 2}, "f11_deferred_flip1": {"flip": 1},
            "f10_spares": {"spares": 3}, "lfu_allram": {}, "f11_ssd_keep": {"promo": "background", "keep": True},
            "f11_deferred_keep": {"flip": 2, "keep": True}, "f11_deferred_flip1_keep": {"flip": 1, "keep": True},
            "f10_spares_keep": {"spares": 3, "keep": True}, "f11_deferred_dlag2": {"flip": 2, "demote_lag": 2},
            "f11_deferred_flip1_dlag2": {"flip": 1, "demote_lag": 2},
            "f10_spares_dlag2": {"spares": 3, "demote_lag": 2}, "f10_spares_rlag2": {"spares": 3, "refill_lag": 2}}
MAIN = ("f11_ssd", "f11_deferred", "f10_spares", "lfu_allram")   # (a), (b), (c), (d)
DECAY = 0.7 ** 0.25
FREE_LAG = 2                           # a released RAM slot is reusable after the host reads the delta log (D4: c+2)
PROMO_LAG = 4                          # host decision after commit c lands at c+2+d, d = 2 (D4)
SPLITS = ((("chat", "code"), ("agent", "mixed")), (("agent",), ("chat", "code", "mixed")))   # M6's held-out splits


class RamTier:
    """The exclusive RAM tier: residency, a fixed slot count, slots released with a lag, copies landing at a step."""

    def __init__(self, ram: np.ndarray, cap: int):
        self.ram, self.cap = ram, cap
        self.free = cap - int(ram.sum())       # ready free slots
        self.later: list[tuple[int, int]] = []  # released slots: (ready step, count)
        self.incoming: list[tuple[int, np.ndarray]] = []   # copies in flight: (landing step, keys); tier code SSD
        self.inflight = np.zeros(len(ram), bool)

    def release(self, keys: np.ndarray, s: int) -> None:
        if len(keys):
            self.ram[keys] = False
            self.later.append((s + FREE_LAG, len(keys)))

    def tick(self, s: int, res: np.ndarray) -> None:
        """Start of step s: released slots become free; copies due land (a key admitted to VRAM meanwhile releases
        its slot instead)."""
        self.free += sum(n for t, n in self.later if t <= s)
        self.later = [(t, n) for t, n in self.later if t > s]
        for t, keys in self.incoming:
            if t <= s:
                land = keys[~res[keys]]
                self.ram[land], self.inflight[keys] = True, False
                if len(keys) > len(land):
                    self.later.append((s + FREE_LAG, len(keys) - len(land)))
        self.incoming = [(t, k) for t, k in self.incoming if t > s]

    def place(self, keys: np.ndarray, u: np.ndarray, land: int, protect: np.ndarray | None = None,
              ebytes: np.ndarray | None = None, max_bytes: float = np.inf) -> np.ndarray:
        """Give RAM slots to `keys` (hottest first, at most `max_bytes` of them): ready free slots, then over RAM's
        coldest unprotected resident while strictly hotter (that one becomes SSD-only at once). The placed keys read
        as SSD until step `land`. Returns them; a key whose earlier copy is still in flight keeps that slot."""
        keys = keys[~self.inflight[keys]]
        keys = keys[np.argsort(-u[keys], kind="stable")]
        if ebytes is not None:
            keys = keys[ebytes[keys].cumsum() <= max_bytes]
        if not len(keys):
            return keys
        nf = min(self.free, len(keys))
        rest = keys[nf:]
        placed = [keys[:nf]]
        self.free -= nf
        if len(rest):
            ev = self.ram if protect is None else self.ram & ~protect
            cand = np.flatnonzero(ev)
            k = min(len(rest), len(cand))
            if k:
                cold = cand[np.argpartition(u[cand], k - 1)[:k]] if k < len(cand) else cand
                cold = cold[np.argsort(u[cold], kind="stable")]
                ok = np.cumprod(u[rest[:k]] > u[cold]).astype(bool)
                self.ram[cold[ok]] = False
                placed.append(rest[:k][ok])
        out = np.concatenate(placed)
        if len(out):
            self.inflight[out] = True
            self.incoming.append((land, out))
        return out

    def check(self, res: np.ndarray) -> None:
        assert not (self.ram & res).any() and not (self.ram & self.inflight).any()
        assert int(self.ram.sum() + self.inflight.sum()) + self.free + sum(n for _, n in self.later) == self.cap


def simulate(tr: Decode, variant: str, quota: np.ndarray, seed: np.ndarray, rank: np.ndarray, ram_seed: np.ndarray,
             ebytes: np.ndarray, knobs: dict | None = None, check: bool = False) -> dict:
    """One replay with an exclusive RAM tier (`ram_seed`: its warm start; its size is the tier's slot count)."""
    L, E, LE = tr.L, tr.E, tr.L * tr.E
    kind = "_".join(variant.split("_")[:2])
    kn = {"demote_lag": 1, "refill_lag": 1} | VARIANTS[variant] | (knobs or {})
    dlag = kn["demote_lag"]
    res = seed.copy()
    u = np.zeros(LE)
    h = u if kn.get("ram_decay", DECAY) == DECAY else np.zeros(LE)   # RAM heat (F11's counts unless set apart)
    last = np.full(LE, -1, np.int64)
    free_sp = np.zeros(L, np.int64)                    # f10_spares: empty spare slots per layer
    if kind == "f10_spares":
        free_sp = np.where(quota < E, np.minimum(kn["spares"], quota - 1), 0)
        res &= rank < np.repeat(quota - free_sp, E)
    allram = kind == "lfu_allram"
    tier = RamTier(np.ones(LE, bool) if allram else ram_seed & ~res, LE if allram else int(ram_seed.sum()))
    pending = np.zeros(LE, bool)                       # f11_deferred: staged, map flips later
    flips: dict[int, np.ndarray] = {}
    queued = np.zeros(LE, bool)                        # f10_spares: refill evictions decided, not yet applied
    evq: list[tuple[int, np.ndarray]] = []
    promos: list[list] = []                            # f11_ssd: [landing step, keys, unread bytes, read by demand]
    need = np.zeros(LE, bool)
    layer_eb = ebytes[::E]
    tot = dict.fromkeys(("hit", "ram_fetch", "ssd_read", "demotion", "promotion", "dense"), 0.0)
    tt = dict.fromkeys(("gpu", "h2d", "d2h", "ssd", "ssd_latency", "promo_stall", "promo_wait"), 0.0)
    hits = need_n = admits = 0
    for s in range(len(tr.tokens)):
        if not allram:
            tier.tick(s, res)
        if s in flips:                                 # f11_deferred: the staged misses' maps flip
            m = flips.pop(s)
            res[m], pending[m] = True, False
            tier.release(m[tier.ram[m]], s)
        stall = sum(p[2] for p in promos if p[0] <= s) / SSD    # f11_ssd: promotions still unread at landing
        promos = [p for p in promos if p[0] > s]
        backlog = sum(p[2] for p in promos)
        a, b = tr.ptr[s], tr.ptr[s + 1]
        keys = tr.key[a:b]
        hit = res[keys]
        miss = keys[~hit]
        inram = tier.ram[miss]
        hb, rb = ebytes[keys[hit]].sum(), ebytes[miss[inram]].sum()
        ssd_keys = miss[~inram]
        sb = ebytes[ssd_keys].sum()
        ssd_layers = np.unique(ssd_keys // E)
        for p in promos:                               # a demand read of a promoted expert completes its promotion
            got = np.isin(p[1], ssd_keys) & ~p[3]
            p[2], p[3] = max(0.0, p[2] - ebytes[p[1][got]].sum()), p[3] | got
        db = pb = 0.0
        hits += int(hit.sum())
        need_n += len(keys)
        u *= DECAY
        np.add.at(u, keys, tr.cnt[a:b])
        if h is not u:
            h *= kn["ram_decay"]
            np.add.at(h, keys, tr.cnt[a:b])
        if kind in ("lfu_allram", "f11_ssd") and len(miss):
            need[keys] = True
            cin, cout = _pair_swap(res, need, u, miss, L, E, gain=0.0, u_min=-1.0, limit=None, apply=True)
            need[keys] = False
            admits += len(cin)
            if not allram:                             # victims become SSD-only; admitted RAM slots are released
                tier.release(cin[tier.ram[cin]], s)
        elif kind == "f11_deferred" and len(miss):
            cand = miss[~pending[miss]]
            need[keys] = True
            cin, cout = _pair_swap(res, need, u, cand, L, E, gain=0.0, u_min=-1.0, limit=None, apply=False)
            need[keys] = False
            admits += len(cin)
            res[cout] = False                          # unmapped; demoted D2H before the staged copy lands
            if len(cin):
                pending[cin] = True
                flips[s + kn["flip"]] = cin
            db = ebytes[tier.place(cout, h, s + dlag, protect=pending)].sum()
        elif kind == "f10_spares":
            last[keys] = s
            prom = miss[_first_n_per_layer(miss, E, free_sp)]
            res[prom] = True
            admits += len(prom)
            tier.release(prom[tier.ram[prom]], s)
            n_prom = np.bincount(prom // E, minlength=L)
            free_sp -= n_prom
            if len(prom):                              # refill: evict the LRU residents, applied refill_lag later
                order = _evict_order(np.where(res & ~queued, last, np.int64(1 << 40)), L, E)
                ev = order[np.arange(E)[None, :] < n_prom[:, None]]
                queued[ev] = True
                evq.append((s + kn["refill_lag"] - 1, ev))
            for t, ev in evq:                          # at the boundary before launch t + 1: unmap and demote
                if t <= s:
                    res[ev], queued[ev] = False, False
                    free_sp += np.bincount(ev // E, minlength=L)
                    db += ebytes[tier.place(ev, h, s + dlag)].sum()
            evq = [(t, ev) for t, ev in evq if t > s]
        if kn.get("keep") and len(ssd_keys):         # demand SSD reads that stayed out of VRAM, kept in RAM
            kept = ssd_keys[~res[ssd_keys] & ~pending[ssd_keys] & ~tier.inflight[ssd_keys] & ~tier.ram[ssd_keys]]
            tier.place(kept, h, s + 1, protect=pending)
        # time
        gpu = (DENSE + hb + rb + sb) / GPU_EFF
        h2d, d2h = (rb + sb) / PCIE, db / PCIE
        t_ns = gpu + h2d + d2h
        wait = EXTENT_WAIT * len(ssd_layers) if backlog > 0 and kn.get("promo") == "background" else 0.0
        ssd_lat = sum(max(ebytes[ssd_keys[ssd_keys // E == la]].sum() / SSD, LAT(layer_eb[la])) for la in ssd_layers)
        if promos and kn.get("promo") == "background":  # the reader drains the backlog in the step's non-SSD time
            room = t_ns * SSD
            for p in promos:
                d = min(room, p[2])
                p[2] -= d
                room -= d
        if kind == "f11_ssd" and kn["promo"] != "off":   # the host's promotions, decided after this step's commit
            cand = np.flatnonzero(~res & ~tier.ram & ~tier.inflight & (h > 0))
            cap = np.inf                               # background: what the reader can drain before landing
            if kn["promo"] == "background":
                cap = (PROMO_LAG - 1) * t_ns * SSD - sum(p[2] for p in promos)
            got = tier.place(cand, h, s + PROMO_LAG, ebytes=ebytes, max_bytes=cap)
            if len(got):
                pb = ebytes[got].sum()
                promos.append([s + PROMO_LAG, got, pb if kn["promo"] == "background" else 0.0,
                               np.zeros(len(got), bool)])
        for k, v in (("hit", hb), ("ram_fetch", rb), ("ssd_read", sb), ("demotion", db), ("promotion", pb),
                     ("dense", DENSE)):
            tot[k] += v
        for k, v in (("gpu", gpu), ("h2d", h2d), ("d2h", d2h), ("ssd", sb / SSD), ("ssd_latency", ssd_lat),
                     ("promo_stall", stall), ("promo_wait", wait)):
            tt[k] += v
        if check and not allram:
            tier.check(res)
            mapped = (res | pending).reshape(L, E).sum(1) + free_sp
            assert (mapped == quota).all() and not (res & pending).any()
    if not allram:
        tier.check(res)
    ntok = int(tr.tokens.sum())
    base = tt["gpu"] + tt["h2d"] + tt["d2h"] + tt["promo_stall"] + tt["promo_wait"]
    return {"hit_rate": hits / need_n, "tokens": ntok, "steps": len(tr.tokens), "admissions_per_token": admits / ntok,
            "bytes_per_token": {k: v / ntok for k, v in tot.items()},
            "ms_parts_per_token": {k: 1e3 * v / ntok for k, v in tt.items()},
            "ms_per_token": 1e3 * (base + tt["ssd"]) / ntok,
            "ms_per_token_ssd_latency": 1e3 * (base + tt["ssd_latency"]) / ntok}


def setup(profile: np.ndarray, L: int, E: int, vf: float, rf: float) -> tuple:
    """VRAM quotas / seed / rank (M6's), and the RAM seed: the profile's top `rf` of the experts not in VRAM."""
    q, seed, rank = quotas(profile, L, E, round(vf * L * E))
    p = profile + 1e-9 * (1 - np.arange(L * E) % E / E)
    order = np.argsort(np.where(seed, np.inf, -p), kind="stable")
    ram = np.zeros(L * E, bool)
    ram[order[:round(rf * (L * E - int(seed.sum())))]] = True
    return q, seed, rank, ram


# ---------------------------------------------------------------- driver

_TR: dict[str, Decode] = {}


_KNOBS: dict = {}


def _job(job: tuple) -> tuple:
    test, train, vf, rf, variant = job
    tr = _TR[test]
    prof = sum(_TR[n].counts for n in train)
    q, seed, rank, ram = setup(prof, tr.L, tr.E, vf, rf)
    return job, simulate(tr, variant, q, seed, rank, ram, np.repeat(EBYTES, tr.E), _KNOBS)


def run(traces: str, jobs: int, variants: tuple[str, ...]) -> dict:
    t0 = time.time()
    for p in sorted(pathlib.Path(traces).glob("*.npz")):
        _TR[p.stem] = decode_steps(load(p), 128)
    todo = []
    for train, tests in SPLITS:
        for test, vf in itertools.product(tests, VRAM):
            todo.append((test, train, vf, 1.0, "lfu_allram"))
            todo += [(test, train, vf, rf, v) for rf in RAMF for v in variants if v != "lfu_allram"]
    with mp.get_context("fork").Pool(jobs) as pool:
        done = pool.map(_job, todo, chunksize=1)
    cells: dict = {}
    for (test, train, vf, rf, v), r in done:
        cell = cells.setdefault(f"{test}|{'+'.join(train)}", {}).setdefault(f"{round(100 * vf)}%", {})
        for rk in ([f"{round(100 * x)}%" for x in RAMF] if v == "lfu_allram" else [f"{round(100 * rf)}%"]):
            cell.setdefault(rk, {})[v] = r
    return {"cells": cells, "replay_s": round(time.time() - t0, 1)}


def _mean(cells: dict, vk: str, rk: str, v: str, key: str = "ms_per_token") -> float:
    return float(np.mean([c[vk][rk][v][key] for c in cells.values()]))


def wins(cells: dict, variants: tuple[str, ...], key: str = "ms_per_token") -> dict:
    """Per (VRAM, RAM) cell: how many of the held-out pairs each variant has the least time in, and the range of
    the others' time over the winner's."""
    out: dict = {}
    for by_v in cells.values():
        for vk, by_r in by_v.items():
            for rk, r in by_r.items():
                ms = {v: r[v][key] for v in variants}
                best = min(ms, key=ms.get)
                cell = out.setdefault(f"{vk} / {rk}", {"wins": {}, "ratio": []})
                cell["wins"][best] = cell["wins"].get(best, 0) + 1
                cell["ratio"].append(max(ms.values()) / ms[best])
    return out


SENS = {"(a)'s promotion": ("f11_ssd", "f11_ssd_nopromo", "f11_ssd_freepromo", "f11_ssd_keep"),
        "(b) and (c): flip, demotion and refill lags": ("f11_deferred", "f11_deferred_flip1", "f11_deferred_dlag2",
                                                        "f11_deferred_flip1_dlag2", "f10_spares", "f10_spares_dlag2",
                                                        "f10_spares_rlag2"),
        "(b) and (c) keeping demand SSD reads in RAM": ("f11_deferred_keep", "f11_deferred_flip1_keep",
                                                        "f10_spares_keep")}
DUELS = (MAIN[:3], ("f11_deferred_flip1", "f10_spares"), ("f11_deferred_dlag2", "f10_spares_dlag2"),
         ("f11_deferred", "f10_spares_rlag2"), ("f11_deferred_keep", "f10_spares_keep"),
         ("f11_deferred_flip1_keep", "f10_spares_keep"))
BYTES_CELLS = (("10%", "50%"), ("25%", "25%"), ("25%", "50%"), ("25%", "75%"), ("47%", "50%"))


def report(rec: dict, others: tuple[dict, ...] = ()) -> str:
    """The README's tables: `rec` at the defaults; `others`, the same grid at other RAM heat decays."""
    md = []

    def table(head: list[str], rows: list[list]) -> None:
        md.append("| " + " | ".join(head) + " |")
        md.append("|" + "---|" * len(head))
        md.extend("| " + " | ".join(str(x) for x in r) + " |" for r in rows)
        md.append("")

    cells = rec["cells"]
    have = next(iter(next(iter(next(iter(cells.values())).values())).values()))
    grid = [(f"{round(100 * x)}%", f"{round(100 * y)}%") for x in VRAM for y in RAMF]
    md.append("Mean over the 5 held-out pairs, ms per token: (a)-(d), SSD at 2.6 GB/s (bold: least of (a)-(c)); "
              "then (a) / (b) / (c) with the per-layer depth-1 floor\n")
    rows = []
    for vk, rk in grid:
        m = {v: _mean(cells, vk, rk, v) for v in MAIN}
        best = min(MAIN[:3], key=m.get)
        rows.append([vk, rk] + [f"**{m[v]:.2f}**" if v == best else f"{m[v]:.2f}" for v in MAIN]
                    + [" / ".join(f"{_mean(cells, vk, rk, v, 'ms_per_token_ssd_latency'):.2f}" for v in MAIN[:3])])
    table(["VRAM", "RAM of rest"] + [f"({c}) {v}" for c, v in zip("abcd", MAIN, strict=True)]
          + ["floor (a) / (b) / (c)"], rows)
    for title, group in SENS.items():
        sens = [v for v in group if v in have]
        md.append(f"Sensitivities, {title}: mean over the 5 pairs, ms per token\n")
        table(["VRAM", "RAM of rest"] + sens, [[vk, rk] + [f"{_mean(cells, vk, rk, v):.2f}" for v in sens]
                                              for vk, rk in grid])
    md.append("D2H demotion overlapped with H2D (PCIe is full duplex), mean over the 5 pairs, ms per token\n")
    hid = [v for v in ("f11_deferred", "f10_spares", "f11_deferred_flip1", "f11_deferred_flip1_keep",
                       "f10_spares_keep") if v in have]
    table(["VRAM", "RAM of rest"] + hid, [[vk, rk] + [
        f"{_mean(cells, vk, rk, v) - np.mean([c[vk][rk][v]['ms_parts_per_token']['d2h'] for c in cells.values()]):.2f}"
        for v in hid] for vk, rk in grid])
    for duel in DUELS:
        if all(v in have for v in duel):
            w = wins(cells, duel)
            md.append(f"Least time per cell over the 5 pairs, {' vs '.join(duel)}: wins; slowest / fastest\n")
            table(["VRAM / RAM"] + list(duel) + ["slowest / fastest (min-max)"],
                  [[k] + [c["wins"].get(v, 0) for v in duel] + [f"{min(c['ratio']):.2f}-{max(c['ratio']):.2f}"]
                   for k, c in w.items()])
    for o in others:
        rd = o["knobs"]["ram_decay"]
        md.append(f"RAM heat decay {rd} per step instead of F11's {DECAY:.3f}, mean over the 5 pairs, ms per token\n")
        sv = [v for v in MAIN[:3] + ("f11_deferred_flip1", "f10_spares_keep", "f11_deferred_flip1_keep")
              if v in have]
        table(["VRAM", "RAM of rest"] + sv, [[vk, rk] + [f"{_mean(o['cells'], vk, rk, v):.2f}" for v in sv]
                                             for vk, rk in grid])
    pv = [v for v in MAIN + ("f11_deferred_flip1",) if v in have]
    md.append("Every held-out pair, ms per token (bold: least of (a)-(c))\n")
    rows = []
    for pair, by_v in cells.items():
        for vk, rk in grid:
            r = by_v[vk][rk]
            best = min(MAIN[:3], key=lambda v: r[v]["ms_per_token"])
            rows.append([pair.replace("|", " / "), vk, rk] + [
                f"**{r[v]['ms_per_token']:.2f}**" if v == best else f"{r[v]['ms_per_token']:.2f}" for v in pv])
    table(["test / profile", "VRAM", "RAM of rest"] + pv, rows)
    md.append("Bytes per decode token by path (MB), admissions per token and time parts (ms), held-out agent "
              "(profile chat+code)\n")
    rows = []
    agent = cells["agent|chat+code"]
    for vk, rk in BYTES_CELLS:
        for v in pv + [x for x in ("f11_deferred_keep", "f10_spares_keep") if x in have]:
            r = agent[vk][rk][v]
            b, pt = r["bytes_per_token"], r["ms_parts_per_token"]
            rows.append([vk, rk, v, f"{100 * r['hit_rate']:.1f}"] + [
                f"{b[k] / 1e6:.1f}" for k in ("hit", "ram_fetch", "ssd_read", "demotion", "promotion")] + [
                f"{r['admissions_per_token']:.1f}", f"{pt['gpu']:.2f}", f"{pt['h2d']:.2f}", f"{pt['d2h']:.2f}",
                f"{pt['ssd']:.2f}", f"{pt['promo_stall'] + pt['promo_wait']:.2f}", f"{r['ms_per_token']:.2f}"])
    table(["VRAM", "RAM", "variant", "hit %", "VRAM hit", "RAM fetch", "SSD read", "demotion", "promotion",
           "admits / tok", "GPU", "H2D", "D2H", "SSD", "promo stall + wait", "ms / tok"], rows)
    return "\n".join(md)


CELL_FIELDS = ("hit_pct", "ms_per_token", "ms_per_token_ssd_floor", "admissions_per_token", "mb_vram_hit",
               "mb_ram_fetch", "mb_ssd_read", "mb_demotion_d2h", "mb_promotion_ssd", "ms_promo_stall_wait")


def evidence(recs: list[dict]) -> dict:
    """The committed record: settings, every pair's results at the defaults (rows of CELL_FIELDS), and the
    per-cell means and wins at every RAM heat decay run."""
    grid = [(f"{round(100 * x)}%", f"{round(100 * y)}%") for x in VRAM for y in RAMF]
    main = recs[0]
    out = {k: main[k] for k in ("model", "rates", "knobs", "replay_s")}
    out["mean_ms_per_token"] = {}
    for r in recs:
        have = next(iter(next(iter(next(iter(r["cells"].values())).values())).values()))
        out["mean_ms_per_token"][f"ram_decay_{r['knobs']['ram_decay']:.3f}"] = {
            f"{vk} / {rk}": {v: round(_mean(r["cells"], vk, rk, v), 3) for v in have} for vk, rk in grid}
    out["wins"] = {" vs ".join(d): wins(main["cells"], d) for d in DUELS}
    for w in out["wins"].values():
        for c in w.values():
            c["ratio"] = [round(min(c["ratio"]), 3), round(max(c["ratio"]), 3)]
    row = lambda r: [round(x, 3) for x in (  # noqa: E731
        100 * r["hit_rate"], r["ms_per_token"], r["ms_per_token_ssd_latency"], r["admissions_per_token"],
        *(r["bytes_per_token"][k] / 1e6 for k in ("hit", "ram_fetch", "ssd_read", "demotion", "promotion")),
        r["ms_parts_per_token"]["promo_stall"] + r["ms_parts_per_token"]["promo_wait"])]
    out["cell_fields"] = CELL_FIELDS
    out["cells"] = {pair: {f"{vk} / {rk}": {v: row(r) for v, r in by_v[vk][rk].items()} for vk, rk in grid}
                    for pair, by_v in main["cells"].items()}
    return out


def selfcheck() -> None:
    """Synthetic traces: lfu_allram reproduces M6's lfu hits; every variant hits 100% with VRAM holding everything;
    the tier stays exclusive and its slot count fixed (asserted inside simulate)."""
    from replay import simulate as m6
    rng = np.random.default_rng(1)
    L, E, k, R = 3, 16, 2, 600
    hot = rng.permutation(E)[:6]
    ids = np.where(rng.random((R, L, k)) < 0.8, rng.choice(hot, (R, L, k)), rng.integers(0, E, (R, L, k)))
    ids[..., 1] = (ids[..., 0] + 1 + rng.integers(0, E - 1, (R, L))) % E
    tr = decode_steps({"ids": ids.astype(np.uint8), "step": np.arange(R) // 2, "decode": np.ones(R, bool),
                       "name": "syn"}, E)
    eb = np.full(L * E, 2.8e6)
    for vf, rf in ((0.25, 0.25), (0.25, 0.75), (0.5, 0.5)):
        q, seed, rank, ram = setup(tr.counts, L, E, vf, rf)
        r = {v: simulate(tr, v, q, seed, rank, ram, eb, check=True) for v in VARIANTS}
        assert abs(r["lfu_allram"]["hit_rate"] - m6(tr, "lfu", q, seed, rank, eb, 0.0)["hit_rate"]) < 1e-12
        assert r["lfu_allram"]["bytes_per_token"]["ssd_read"] == 0
        print(vf, rf, {v: (round(x["hit_rate"], 3), round(x["ms_per_token"], 2)) for v, x in r.items()})
    q, seed, rank, ram = setup(tr.counts, L, E, 1.0, 0.5)
    assert all(simulate(tr, v, q, seed, rank, ram, eb)["hit_rate"] == 1.0 for v in VARIANTS)
    print("selfcheck ok")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", help="directory of capture.py's .npz files")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--out")
    ap.add_argument("--report", nargs="+", help="records written by --out (the default run first, then --ram-decay "
                    "runs): print the README's tables")
    ap.add_argument("--evidence", help="with --report: also write the condensed evidence record here")
    ap.add_argument("--ram-decay", type=float, help="RAM heat's decay per step (default: F11's counts, 0.915)")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        return selfcheck()
    if args.report:
        recs = [json.loads(pathlib.Path(f).read_text()) for f in args.report]
        if args.evidence:
            pathlib.Path(args.evidence).write_text(json.dumps(evidence(recs), indent=1) + "\n")
        return print(report(recs[0], tuple(recs[1:])))
    if args.ram_decay is not None:
        _KNOBS["ram_decay"] = args.ram_decay
    rec = run(args.traces, args.jobs, tuple(args.variants))
    rec["model"] = {"expert_bytes": sorted(set(EBYTES.tolist())), "dense_bytes_per_step": DENSE}
    rec["rates"] = {"gpu_all_resident_ms_batch1": T_B1 * 1e3, "gpu_eff_gbps": GPU_EFF / 1e9, "h2d_gbps": PCIE / 1e9,
                    "d2h_gbps": PCIE / 1e9, "ssd_gbps": SSD / 1e9,
                    "ssd_depth1_ms": {str(int(x)): LAT(x) * 1e3 for x in sorted(set(EBYTES.tolist()))},
                    "extent_wait_ms": EXTENT_WAIT * 1e3}
    rec["knobs"] = {"variants": VARIANTS, "decay": DECAY, "ram_decay": _KNOBS.get("ram_decay", DECAY),
                    "free_lag": FREE_LAG, "promo_lag": PROMO_LAG,
                    "splits": SPLITS, "vram": VRAM, "ram_of_rest": RAMF}
    text = json.dumps(rec, indent=1)
    if args.out:
        pathlib.Path(args.out).write_text(text)
    print(report(rec))


if __name__ == "__main__":
    main()
