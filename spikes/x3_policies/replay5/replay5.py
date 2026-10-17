"""X step 3b item 5: prompt-time admission and D8 ring return for F11 (`lfu`).  Whole-trace replay in step order.

Extends M6's replay (spikes/x0_offload/m6/replay.py, imported read-only) to every step of a capture trace, prefill
and mixed steps included. Scoring is decode only (prefill streams or stages its experts under D8 in every variant, so
its cost is taken as unchanged); prefill-time admissions are reported apart as bytes that are either free (the miss
is streamed / staged straight into its victim slot, as decode misses are) or paid (a separate H2D copy).

lfu variants (decay x0.915 only on steps with decode rows; pairing per layer per step: the step's non-resident
candidates by count desc against residents the step (decode + prefill rows) does not need by count asc, ties by
expert id, admit iff strictly greater, stop at the first failing pair):
  A    D4 as specified, no lending modelled: prefill rows neither count nor admit
  B    prefill rows count +1 per routed entry and their misses are candidates
  C    prefill misses are candidates but prefill rows do not count (only experts with decode history can enter)
  C2   prefill rows count +1.0 per (chunk, layer, expert) routed (presence, not entries) and their misses are
       candidates: a nominal count of 1 per prefill chunk that routed the expert
  BwW  prefill rows count W x (the expert's routed entries / the chunk's rows) and their misses are candidates: a
       chunk adds W decode tokens' worth at the prompt's routing frequency (W = 12 ~ a converged decode history,
       1 / (1 - 0.915) = 11.8)
  lru     M6's lru (F4): decode rows touch and admit; prefill ignored
  lru_pf  FreeToken-style: prefill rows touch (timestamps) and their misses admit too
`A_m6` is A with M6's victim mask (decode keys only); it and `lru_m6` must equal M6's `lfu` / `lru` exactly.

D8 ring variants (prefill never counts; A's rule otherwise). A step whose prefill rows number >= THR is a streaming
(ring) step: at its launch the device lends N whole experts, N = RING x max_l(E - quota_l) (RING = 2: a double buffer
of the largest layer's non-resident set; 1: single), the layer's empty slots first, then the globally coldest
residents by decayed count, ties by layer then expert id. Every expert non-resident at launch is streamed. At return:
  R0  lent slots come back empty; an empty slot is its layer's first victim, filled unconditionally by the next
      admissions (that layer's candidates by count desc, ties by id; the rest pair as A)
  R1  each layer's empty slots are refilled from that layer's streamed experts not resident after the step, by decayed
      decode count desc, ties by expert id (the lent experts are themselves streamed, so this restores or upgrades)
  R2  as R1 but ranked by the step's prefill routed entries desc, then decayed count desc, then expert id
`Bw12_Rx` = Bw12's counts and prefill admission with ring return Rx. Suffix `_s` = single-buffer ring (RING 1); `_all` = every prefill step lends (THR 1). Steps below THR stage and
never admit or lend.

    python replay5.py --traces <x0-m6 dir> --gguf <Q4_K_M.gguf> --out replay5.json [--check] [--jobs 12]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "x0_offload" / "m6"))
import replay as m6  # noqa: E402

RATES = (25e9, 47.5e9, 50e9)
GPU = m6.GPU
DECAY = m6.DEFAULTS["lfu"]["decay"]
FRACTIONS = (0.10, 0.25, 0.47)
BUCKETS = (0, 8, 32, 128)          # decode position of a row in its sequence: [0,8) [8,32) [32,128) [128,inf)
THR = 512                          # D8 crossover on the 5080: chunks from here stream through the ring


@dataclasses.dataclass(frozen=True)
class V:
    fam: str = "lfu"               # lfu | lru
    pweight: str | None = None     # prefill increments: None | entries | presence | freq (pw x entries / chunk rows)
    pw: float = 0.0
    padmit: bool = False           # prefill misses are candidates
    m6need: bool = False           # victim mask from decode keys only (M6)
    ret: str | None = None         # ring return: None (no lending) | R0 | R1 | R2
    ring: int = 2                  # ring size in largest-layer non-resident sets
    thr: int = THR


VARIANTS = {"A": V(), "A_m6": V(m6need=True), "B": V(pweight="entries", padmit=True), "C": V(padmit=True),
            "C2": V(pweight="presence", padmit=True),
            **{f"Bw{w}": V(pweight="freq", pw=float(w), padmit=True) for w in (1, 4, 12, 32)},
            **{f"Bw12_{r}": V(pweight="freq", pw=12.0, padmit=True, ret=r) for r in ("R0", "R1", "R2")}, "lru": V("lru"), "lru_pf": V("lru", padmit=True),
            **{r: V(ret=r) for r in ("R0", "R1", "R2")},
            **{f"{r}_s": V(ret=r, ring=1) for r in ("R0", "R1", "R2")},
            **{f"{r}_all": V(ret=r, thr=1) for r in ("R0", "R1", "R2")}}
CHECK_ONLY = {"lru_m6": V("lru", m6need=True)}


@dataclasses.dataclass
class Full:
    name: str
    L: int
    E: int
    dptr: np.ndarray       # [S+1] decode keys of each step (distinct, by layer then route order)
    dkey: np.ndarray
    dcnt: np.ndarray
    pptr: np.ndarray       # [S+1] prefill keys of each step
    pkey: np.ndarray
    pcnt: np.ndarray
    tokens: np.ndarray     # [S] decode rows per step
    prows: np.ndarray      # [S] prefill rows per step
    rptr: np.ndarray       # [S+1] decode rows of each step
    rkey: np.ndarray       # [Rd, L*k] keys per decode row
    rbucket: np.ndarray    # [Rd] warm-up bucket of the row
    seqs: int


def _group(ids: np.ndarray, sidx: np.ndarray, S: int, E: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """m6.decode_steps' grouping for arbitrary rows: distinct keys per step, by layer then route order."""
    R, L, k = ids.shape
    st = np.repeat(sidx, L * k)
    lay = np.tile(np.repeat(np.arange(L), k), R)
    key = lay * E + ids.reshape(-1).astype(np.int64)
    order = np.arange(R * L * k) % k + np.repeat(np.arange(R), L * k) * k
    srt = np.lexsort((order, lay, st))
    st, key = st[srt], key[srt]
    _, first, cnt = np.unique(st * (L * E) + key, return_index=True, return_counts=True)
    keep = np.argsort(first, kind="stable")
    st, key, cnt = st[first[keep]], key[first[keep]], cnt[keep]
    ptr = np.concatenate([[0], np.cumsum(np.bincount(st, minlength=S))])
    return ptr, key, cnt.astype(float)


def full_steps(raw: dict, E: int) -> Full:
    step, dec, seq, ids = raw["step"], raw["decode"], raw["seq"], raw["ids"]
    _, sidx = np.unique(step, return_inverse=True)
    S = int(sidx.max()) + 1
    L, k = ids.shape[1], ids.shape[2]
    dptr, dkey, dcnt = _group(ids[dec], sidx[dec], S, E)
    pptr, pkey, pcnt = _group(ids[~dec], sidx[~dec], S, E)
    tokens = np.bincount(sidx[dec], minlength=S)
    assert np.all(np.diff(sidx[dec]) >= 0)                       # decode rows in step order
    rptr = np.concatenate([[0], np.cumsum(tokens)])
    rkey = (np.arange(L)[None, :, None] * E + ids[dec].astype(np.int64)).reshape(-1, L * k)
    dseq = seq[dec]                                               # position of each decode row in its sequence
    pos = np.zeros(len(dseq), np.int64)
    o = np.argsort(dseq, kind="stable")
    s_sorted = dseq[o]
    pos[o] = np.arange(len(o)) - np.searchsorted(s_sorted, s_sorted, side="left")
    rbucket = np.searchsorted(np.array(BUCKETS), pos, side="right") - 1
    return Full(raw["name"], L, E, dptr, dkey, dcnt, pptr, pkey, pcnt, tokens, np.bincount(sidx[~dec], minlength=S),
                rptr, rkey, rbucket, int(len(np.unique(seq))))


def _rank_in_layer(keys: np.ndarray, *scores_desc: np.ndarray, E: int) -> tuple[np.ndarray, np.ndarray]:
    """`keys` sorted per layer by the scores (descending, first score most significant), ties by key; and each
    sorted key's rank within its layer."""
    lay = keys // E
    o = np.lexsort((keys,) + tuple(-s for s in reversed(scores_desc)) + (lay,))
    k, lay = keys[o], lay[o]
    return k, np.arange(len(k)) - np.searchsorted(lay, lay, side="left")


def _admit(res, need, u, cand, free, L, E) -> np.ndarray:
    """A's pairing, after the layer's empty slots (R0's returned ring) take its best candidates unconditionally."""
    filled = cand[:0]
    if len(cand) and free.any():
        c, r = _rank_in_layer(cand, u[cand], E=E)
        take = r < free[c // E]
        filled, cand = c[take], c[~take]
        res[filled] = True
    if not len(cand):
        return filled
    cin, _ = m6._pair_swap(res, need, u, cand, L, E, gain=0.0, u_min=-1.0, limit=None, apply=True)
    return np.concatenate([filled, cin])


def simulate(tr: Full, variant: str, quota, seed, rank, ebytes: np.ndarray, dense: float) -> dict:
    v = (VARIANTS | CHECK_ONLY)[variant]
    L, E, LE = tr.L, tr.E, tr.L * tr.E
    res = seed.copy()
    u = np.zeros(LE)
    last = np.full(LE, -1, np.int64)
    need = np.zeros(LE, bool)
    mark = np.zeros(LE, bool)
    cnt_arr = np.ones(LE)
    parr = np.zeros(LE)
    zero = np.zeros(L, np.int64)
    n_ring = v.ring * int((E - quota).max())
    tot = {"hit": 0.0, "fetch": 0.0, "dense": 0.0, "prefill_admission": 0.0}
    t_free = np.zeros(len(RATES))
    hits = need_n = n_padm = 0
    ring = {"steps": 0, "lent": 0, "lent_restored": 0, "kept": 0}
    nb = len(BUCKETS)
    b_ent, b_hitent, b_rows, b_fetch, b_hitb = (np.zeros(nb) for _ in range(5))
    for s in range(len(tr.tokens)):
        dk, dc = tr.dkey[tr.dptr[s]:tr.dptr[s + 1]], tr.dcnt[tr.dptr[s]:tr.dptr[s + 1]]
        pk, pc = tr.pkey[tr.pptr[s]:tr.pptr[s + 1]], tr.pcnt[tr.pptr[s]:tr.pptr[s + 1]]
        ntok = int(tr.tokens[s])
        is_ring = v.ret is not None and tr.prows[s] >= v.thr
        if is_ring:                                       # lend at launch: empty slots first, then the coldest
            ring["steps"] += 1
            empties = int((quota - res.reshape(L, E).sum(1)).sum())
            n_lend = max(0, n_ring - empties)
            rk_ = np.flatnonzero(res)
            lent = rk_[np.lexsort((rk_, u[rk_]))[:n_lend]]
            res[lent] = False
            stream = ~res                                 # every non-resident expert streams through the ring
            ring["lent"] += len(lent)
        dmiss = dk[~res[dk]]
        if ntok:                                          # score the decode rows against the residency at launch
            hit = res[dk]
            hb, fb = ebytes[dk[hit]].sum(), ebytes[dmiss].sum()
            hits += int(hit.sum())
            need_n += len(dk)
            tot["hit"] += hb
            tot["fetch"] += fb
            tot["dense"] += dense
            for i, p in enumerate(RATES):
                t_free[i] += (dense + hb + fb) / GPU + fb / p
            rk = tr.rkey[tr.rptr[s]:tr.rptr[s + 1]]       # per-row attribution (a shared miss split by its rows)
            cnt_arr[dk] = dc
            h = res[rk]
            share = ebytes[rk] / cnt_arr[rk]
            rb = tr.rbucket[tr.rptr[s]:tr.rptr[s + 1]]
            np.add.at(b_ent, rb, rk.shape[1])
            np.add.at(b_hitent, rb, h.sum(1))
            np.add.at(b_rows, rb, 1)
            np.add.at(b_fetch, rb, (share * ~h).sum(1))
            np.add.at(b_hitb, rb, (share * h).sum(1) + dense / ntok)
            cnt_arr[dk] = 1.0
        nd = dk if v.m6need else np.concatenate([dk, pk])  # the step's needs (victim mask)
        if v.fam == "lfu":
            if ntok:
                u *= DECAY
            u[dk] += dc                                   # distinct within a step: plain fancy add is exact
            if v.pweight == "entries":
                u[pk] += pc
            elif v.pweight == "presence":
                u[pk] += 1.0
            elif v.pweight == "freq" and len(pk):
                u[pk] += v.pw * pc / tr.prows[s]
            cand = dmiss
            if v.padmit and len(pk):
                mark[dmiss] = True
                cand = np.concatenate([dmiss, pk[~res[pk] & ~mark[pk]]])
                mark[dmiss] = False
            if len(cand):
                free = zero if (is_ring or v.ret is None) else quota - res.reshape(L, E).sum(1)
                need[nd] = True
                cin = _admit(res, need, u, cand, free, L, E)
                need[nd] = False
                if v.padmit and len(cin):
                    mark[dmiss] = True
                    pin = cin[~mark[cin]]
                    mark[dmiss] = False
                    n_padm += len(pin)
                    tot["prefill_admission"] += ebytes[pin].sum()
            if is_ring and v.ret in ("R1", "R2"):         # return: refill the empty slots from the ring
                free = quota - res.reshape(L, E).sum(1)
                c = np.flatnonzero(stream & ~res)
                if v.ret == "R1":
                    c, r = _rank_in_layer(c, u[c], E=E)
                else:
                    parr[pk] = pc
                    c, r = _rank_in_layer(c, parr[c], u[c], E=E)
                    parr[pk] = 0.0
                keep = c[r < free[c // E]]
                res[keep] = True
                ring["kept"] += len(keep)
                ring["lent_restored"] += int(res[lent].sum())
        else:                                             # lru family
            if v.padmit:
                mark[dk] = True
                keys = np.concatenate([dk, pk[~mark[pk]]])
                mark[dk] = False
                keys = keys[np.argsort(keys // E, kind="stable")]     # by layer; decode keys first in a layer
            else:
                keys = dk
            if not len(keys):
                continue
            last[keys] = s
            miss = keys[~res[keys]]
            if len(miss):
                need[nd] = True
                n_miss = np.bincount(miss // E, minlength=L)
                evictable = (res & ~need).reshape(L, E).sum(1)
                n_ev = np.minimum(n_miss, evictable)
                order = m6._evict_order(np.where(res & ~need, last, m6.BIG), L, E)
                res[order[np.arange(E)[None, :] < n_ev[:, None]]] = False
                adm = miss[m6._first_n_per_layer(miss, E, n_ev)]
                res[adm] = True
                need[nd] = False
                if v.padmit:
                    mark[dmiss] = True
                    pin = adm[~mark[adm]]
                    mark[dmiss] = False
                    n_padm += len(pin)
                    tot["prefill_admission"] += ebytes[pin].sum()
        cnt = res.reshape(L, E).sum(1)
        assert (cnt <= quota).all() and (v.ret is not None or (cnt == quota).all()), (variant, s)
    ntok = int(tr.tokens.sum())
    pa = tot["prefill_admission"]
    rate = lambda p: f"pcie{p / 1e9:g}"  # noqa: E731
    out = {"hit_rate": hits / need_n, "tokens": ntok,
           "bytes_per_token": {k: x / ntok for k, x in tot.items()},
           "prefill_admissions": n_padm, "prefill_admission_mb_per_prompt": pa / tr.seqs / 1e6,
           "ms_per_token": {rate(p): 1e3 * t_free[i] / ntok for i, p in enumerate(RATES)},
           "ms_per_token_paid": {rate(p): 1e3 * (t_free[i] + pa / p) / ntok for i, p in enumerate(RATES)},
           "warmup": {f"[{lo},{hi})": {"rows": int(b_rows[j]), "hit_rate": b_hitent[j] / max(1, b_ent[j]),
                                       "fetch_mb_per_token": b_fetch[j] / max(1, b_rows[j]) / 1e6,
                                       "ms_per_token_pcie47.5": 1e3 * (b_hitb[j] / GPU + b_fetch[j] / GPU
                                                                       + b_fetch[j] / 47.5e9) / max(1, b_rows[j])}
                      for j, (lo, hi) in enumerate(zip(BUCKETS, BUCKETS[1:] + ("inf",)))}}
    if v.ret is not None:
        out["ring"] = ring | {"ring_experts": n_ring}
    return out


G: dict = {}                       # shared with forked workers


def _cell(job: tuple[str, str, float]) -> tuple[str, str, dict]:
    n, pname, f = job
    tr, prof = G["fulls"][n], G["profs"][pname]
    q = m6.quotas(prof, tr.L, tr.E, round(f * tr.L * tr.E))
    return n, f"{round(f * 100)}%", {v: simulate(tr, v, *q, G["ekey"], G["dense"]) for v in VARIANTS}


ISO = ("A", "B", "C", "C2", "Bw1", "Bw4", "Bw12", "Bw32", "R0", "R1", "R2", "Bw12_R0", "Bw12_R1", "Bw12_R2", "lru",
       "lru_pf")


def _iso(job: tuple[str, str, float]) -> tuple[str, str, dict]:
    """Isolated requests (the x4_gap bench's shape): each sequence with >= THR prefill rows and >= 32 decode rows is
    replayed alone from the profile seed (its prefill steps, then its decode); tokens-weighted means."""
    n, pname, f = job
    raw, prof = G["raws"][n], G["profs"][pname]
    seq, dec = raw["seq"], raw["decode"]
    pre_n, dec_n = np.bincount(seq[~dec], minlength=seq.max() + 1), np.bincount(seq[dec], minlength=seq.max() + 1)
    pick = np.flatnonzero((pre_n >= THR) & (dec_n >= 32))
    acc: dict = {}
    for q_ in pick:
        m = seq == q_
        tr = full_steps({k: raw[k][m] for k in ("ids", "step", "decode", "seq")} | {"name": f"{n}/{q_}"}, G["E"])
        q = m6.quotas(prof, tr.L, tr.E, round(f * tr.L * tr.E))
        for v in ISO:
            r = simulate(tr, v, *q, G["ekey"], G["dense"])
            a = acc.setdefault(v, {"tokens": 0, "hit": 0.0, "fetch": 0.0, "padm": 0.0, "ms": np.zeros(len(RATES)),
                                   "wrows": np.zeros(len(BUCKETS)), "whit": np.zeros(len(BUCKETS)),
                                   "wfetch": np.zeros(len(BUCKETS))})
            t = r["tokens"]
            a["tokens"] += t
            a["hit"] += r["hit_rate"] * t
            a["fetch"] += r["bytes_per_token"]["fetch"] * t
            a["padm"] += r["bytes_per_token"]["prefill_admission"] * t
            a["ms"] += np.array(list(r["ms_per_token"].values())) * t
            for j, w in enumerate(r["warmup"].values()):
                a["wrows"][j] += w["rows"]
                a["whit"][j] += w["hit_rate"] * w["rows"]
                a["wfetch"][j] += w["fetch_mb_per_token"] * w["rows"]
    out = {"requests": int(len(pick)), "prompt_rows_mean": float(pre_n[pick].mean()),
           "decode_rows_mean": float(dec_n[pick].mean())}
    for v, a in acc.items():
        t = a["tokens"]
        out[v] = {"hit_rate": a["hit"] / t, "fetch_mb_per_token": a["fetch"] / t / 1e6,
                  "prefill_admission_mb_per_prompt": a["padm"] / len(pick) / 1e6,
                  "ms_per_token": {f"pcie{p / 1e9:g}": float(x / t) for p, x in zip(RATES, a["ms"])},
                  "warmup": {f"[{lo},{hi})": {"rows": int(a["wrows"][j]), "hit_rate": a["whit"][j] / a["wrows"][j],
                                              "fetch_mb_per_token": a["wfetch"][j] / a["wrows"][j]}
                             for j, (lo, hi) in enumerate(zip(BUCKETS, BUCKETS[1:] + ("inf",))) if a["wrows"][j]}}
    return n, f"{round(f * 100)}%", out


def check(raws: dict, fulls: dict, train_prof, ekey, dense, E) -> dict:
    """A_m6 / lru_m6 on the whole trace equal M6's decode-only lfu / lru exactly."""
    out = {}
    for n in ("agent", "mixed"):
        dec = m6.decode_steps(raws[n], E)
        for f in FRACTIONS:
            q = m6.quotas(train_prof, dec.L, E, round(f * dec.L * E))
            for v, pol in (("A_m6", "lfu"), ("lru_m6", "lru")):
                ref = m6.simulate(dec, pol, *q, ekey, dense)
                got = simulate(fulls[n], v, *q, ekey, dense)
                d = {"hit_rate": (ref["hit_rate"], got["hit_rate"]),
                     "ms25": (ref["ms_per_token"]["pcie25"], got["ms_per_token"]["pcie25"]),
                     "ms50": (ref["ms_per_token"]["pcie50"], got["ms_per_token"]["pcie50"]),
                     "fetch": (ref["bytes_per_token"]["fetch"], got["bytes_per_token"]["fetch"])}
                ok = all(abs(a - b) <= 1e-9 * max(1.0, abs(a)) for a, b in d.values())
                out[f"{n}/{round(f * 100)}%/{v}"] = {"ok": ok, **{k: [float(a), float(b)] for k, (a, b) in d.items()}}
                print(n, f, v, "OK" if ok else "MISMATCH", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()
    t0 = time.time()
    ebytes, dense, _, E = m6.expert_bytes(args.gguf)
    d = pathlib.Path(args.traces)
    raws = {n: m6.load(d / f"{n}.npz") for n in ("chat", "code", "agent", "mixed")}
    fulls = {n: full_steps(r, E) for n, r in raws.items()}
    ekey = np.repeat(ebytes, E)
    dprof = {n: m6.decode_steps(r, E).counts for n, r in raws.items()}
    out = {"rates": list(RATES), "gpu": GPU, "decay": DECAY, "thr": THR, "buckets": list(BUCKETS),
           "variants": {k: dataclasses.asdict(x) for k, x in VARIANTS.items()},
           "traces": {n: {"steps": int(len(f.tokens)), "decode_steps": int((f.tokens > 0).sum()),
                          "prefill_steps": int((f.prows > 0).sum()), "ring_steps": int((f.prows >= THR).sum()),
                          "mixed_steps": int(((f.tokens > 0) & (f.prows > 0)).sum()),
                          "decode_rows": int(f.tokens.sum()), "prefill_rows": int(f.prows.sum()), "seqs": f.seqs}
                      for n, f in fulls.items()}}
    train = dprof["chat"] + dprof["code"]
    if args.check:
        out["check"] = check(raws, fulls, train, ekey, dense, E)
    G.update(raws={n: raws[n] for n in ("agent", "mixed")}, E=E, fulls=fulls, profs={"chat+code": train, "agent": dprof["agent"]}, ekey=ekey, dense=dense)
    # held-out as M6: profile chat+code -> agent, mixed; reverse: profile agent -> chat, code (batch-1 streams)
    tests = {"agent": "chat+code", "mixed": "chat+code", "chat": "agent", "code": "agent"}
    jobs = [(n, p, f) for n, p in tests.items() for f in FRACTIONS]
    with mp.get_context("fork").Pool(args.jobs) as pool:
        iso = pool.map_async(_iso, [j for j in jobs if j[0] in ("agent", "mixed")], chunksize=1)
        cells = pool.map(_cell, jobs, chunksize=1)
        iso = iso.get()
    out["heldout"] = {n: {"profile": p, "by_fraction": {}} for n, p in tests.items()}
    for n, f, row in cells:
        out["heldout"][n]["by_fraction"][f] = row
        print(n, f, " ".join(f"{v}:{100 * r['hit_rate']:.1f}/{r['ms_per_token']['pcie47.5']:.2f}"
                             for v, r in row.items()), flush=True)
    out["isolated"] = {n: {} for n in ("agent", "mixed")}
    for n, f, row in iso:
        out["isolated"][n][f] = row
        print("isolated", n, f, row["requests"], " ".join(f"{v}:{100 * row[v]['hit_rate']:.1f}/"
                                                          f"{row[v]['ms_per_token']['pcie47.5']:.2f}" for v in ISO))
    out["replay_s"] = round(time.time() - t0, 1)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
