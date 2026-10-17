"""M6 step 3: offline replay of expert placement policies over captured routing traces (CPU only, numpy).

Decode steps only: a step needs the distinct (layer, expert) pairs its decode rows route to (D12: a miss is paid once
per step); prefill rows never change residency (D8 streams or stages them; their coverage is reported apart). Every
policy gets the same per-layer slot quotas and warm start, from a profile learned on a *training* trace: a floor of
`FLOOR` slots per layer, the rest by the profile's global (layer, expert) ranking (Strata X1, Maya G6). Policies:

  static            the profile's residency, frozen (F1)
  profile_decay     Strata X2 (generate.cpp:8829-8987, expert_source.cpp:3717-3719): +1 per routed entry per decode
                    token, a round after every `every` decode steps: per layer, non-resident candidates with count >=
                    `u_min` (descending) pair with residents (ascending) while cand >= victim + `gain`; the global top
                    `swaps` by gain; counts x `decay` after the round; a round is skipped while edits are pending.
                    Edits land under D4: the victim is unmapped at step c+2, the new expert mapped at c+2+d (d = 2)
  lru               F4 (FreeToken, llama.cpp's cache): every miss is fetched into its layer's least recently used
                    resident not needed in this step (logical-step timestamps)
  promote_on_fetch  Maya G1/G3 (glm_fast.cu:3475-3483, glm_fast_path.cu:4628-4652): misses, in route order, land in
                    the layer's `spares` free slots and become resident; the rest go to scratch; each step boundary
                    refills the spares by evicting the layer's least recently used residents
  lfu               decayed counts (x `decay` per step); a miss replaces the coldest resident not needed in this step
                    when its count is higher (else it is only fetched)
  belady            offline optimal per layer with bypass: after each step a layer keeps the slots' worth of experts
                    used soonest again (an upper bound for demand policies)

Bytes per decode token by path: hit (GPU reads of resident experts), fetch (misses over PCIe, F5a), admission (swap
copies over PCIe; fetch-on-miss policies admit the fetched blob in place, so theirs is 0), cpu (0: F5a only), ssd
(misses outside a RAM tier; the RAM tier holds every expert here, so 0 unless --ram-fraction < 1). Modelled time
per token = (dense + hit + fetch bytes) / GPU rate + (fetch + admission) / PCIe rate + ssd / SSD rate, per step,
summed over steps / decode tokens. The rates are PLACEHOLDERS until M1/M3 land.

    python spikes/x0_offload/m6/replay.py --traces local/tmp/x0-m6 --gguf <file.gguf> --out <json>
    python spikes/x0_offload/m6/replay.py --selfcheck
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import time

import numpy as np

FRACTIONS = (0.10, 0.15, 0.20, 0.25, 0.35, 0.47, 0.60)
POLICIES = ("static", "profile_decay", "lru", "promote_on_fetch", "lfu", "belady")
PCIE = (25e9, 50e9)              # placeholder H2D rates (M1 replaces them)
GPU = 900e9                      # placeholder GPU read rate (≈ the RTX 5080's 960 GB/s)
SSD = 7e9                        # placeholder NVMe rate (M7)
FLOOR = 4                        # slots per layer before the global ranking (≥ promote_on_fetch's 3 spares + 1)
BIG = np.int64(1 << 40)          # "never used again"
DEFAULTS = {"profile_decay": {"decay": 0.7, "every": 4, "swaps": 96, "gain": 1.5, "u_min": 2.0, "lag": 2, "d": 2},
            "promote_on_fetch": {"spares": 3}, "lfu": {"decay": 0.7 ** 0.25}}


@dataclasses.dataclass
class Decode:
    """A trace's decode steps as distinct (layer, expert) keys per step, in route order within each layer."""

    name: str
    L: int
    E: int
    ptr: np.ndarray        # [S+1] offsets into key/cnt/nxt
    key: np.ndarray        # [N] l*E + e
    cnt: np.ndarray        # [N] routed entries (rows) of the key in the step
    nxt: np.ndarray        # [N] the next step needing the key (BIG: never)
    tokens: np.ndarray     # [S] decode rows per step
    counts: np.ndarray     # [L*E] routed entries over the trace (a profile)


def load(path: pathlib.Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files} | {"name": path.stem, "info": json.loads(str(z["info"]))}


def decode_steps(raw: dict, E: int) -> Decode:
    dec = raw["decode"]
    ids, step = raw["ids"][dec].astype(np.int64), raw["step"][dec]
    R, L, k = ids.shape
    _, sidx, tokens = np.unique(step, return_inverse=True, return_counts=True)       # compressed decode steps
    st = np.repeat(sidx, L * k)
    lay = np.tile(np.repeat(np.arange(L), k), R)
    key = lay * E + ids.reshape(-1)
    order = np.arange(R * L * k) % k + np.repeat(np.arange(R), L * k) * k            # route order: row, then rank
    srt = np.lexsort((order, lay, st))
    st, key = st[srt], key[srt]
    u, first, cnt = np.unique(st * (L * E) + key, return_index=True, return_counts=True)
    keep = np.argsort(first, kind="stable")                                         # back to (step, layer, route)
    st, key, cnt = st[first[keep]], key[first[keep]], cnt[keep]
    by_key = np.lexsort((st, key))
    nxt = np.full(len(key), BIG)
    same = key[by_key[1:]] == key[by_key[:-1]]
    nxt[by_key[:-1][same]] = st[by_key[1:][same]]
    ptr = np.concatenate([[0], np.cumsum(np.bincount(st, minlength=len(tokens)))])
    counts = np.bincount(key, weights=cnt, minlength=L * E)
    return Decode(raw["name"], L, E, ptr, key, cnt, nxt, tokens, counts)


def quotas(profile: np.ndarray, L: int, E: int, slots: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-layer slot counts, the warm-start residency (FLOOR per layer, then the global ranking) and each key's
    rank within its layer by the profile (ties: lower expert id)."""
    p = (profile + 1e-9 * (1 - np.arange(L * E) % E / E)).reshape(L, E)        # ties: lower id first
    rank = np.argsort(np.argsort(-p, axis=1, kind="stable"), axis=1, kind="stable").reshape(-1)
    seed = rank < FLOOR
    extra = slots - int(seed.sum())
    if extra > 0:
        seed[np.argsort(np.where(seed, np.inf, -p.reshape(-1)), kind="stable")[:extra]] = True
    return seed.reshape(L, E).sum(1), seed, rank


def _first_n_per_layer(keys: np.ndarray, E: int, n: np.ndarray) -> np.ndarray:
    """A mask over `keys` (grouped by layer, in route order): the first n[layer] of each layer."""
    lay = keys // E
    start = np.searchsorted(lay, lay, side="left")
    return np.arange(len(keys)) - start < n[lay]


def _evict_order(score: np.ndarray, L: int, E: int) -> np.ndarray:
    """Flat keys per layer by ascending score [L, E] (ties: lower expert id)."""
    return np.argsort(score.reshape(L, E), axis=1, kind="stable") + (np.arange(L) * E)[:, None]


def simulate(tr: Decode, policy: str, quota: np.ndarray, seed: np.ndarray, rank: np.ndarray, ebytes: np.ndarray,
             dense: float, knobs: dict | None = None, ram: np.ndarray | None = None) -> dict:
    """One replay: per-path byte totals, hits, and modelled time (per decode token) at each PCIe rate."""
    L, E, LE = tr.L, tr.E, tr.L * tr.E
    kn = DEFAULTS.get(policy, {}) | (knobs or {})
    res = seed.copy()
    last = np.full(LE, -1, np.int64)                  # lru / promote_on_fetch timestamps
    u = np.zeros(LE)                                  # profile_decay / lfu counts
    nu = np.full(LE, BIG)                             # belady: next use
    if policy == "belady":
        first = np.full(LE, BIG)
        np.minimum.at(first, tr.key, np.repeat(np.arange(len(tr.tokens)), np.diff(tr.ptr)))
        nu = first
    spares = np.where(quota < E, np.minimum(kn.get("spares", 0), quota - 1), 0)   # a complete layer needs none
    if policy == "promote_on_fetch":                  # the spares start empty: each layer's coldest seeds out
        res &= rank < np.repeat(quota - spares, E)
    pending: list[tuple[int, np.ndarray, bool]] = []  # profile_decay: (step, keys, map?) applied at launch
    hres = res.copy()                                 # the host's view (profile_decay)
    tot = {"hit": 0.0, "fetch": 0.0, "admission": 0.0, "ssd": 0.0, "dense": 0.0}
    t_serial = np.zeros(len(PCIE))
    t_hidden = np.zeros(len(PCIE))
    hits = need_n = swaps_done = 0
    need = np.zeros(LE, bool)
    for s in range(len(tr.tokens)):
        if pending:
            due = [p for p in pending if p[0] <= s]
            for _, keys, on in due:
                res[keys] = on
            pending = [p for p in pending if p[0] > s]
        a, b = tr.ptr[s], tr.ptr[s + 1]
        keys = tr.key[a:b]
        hit = res[keys]
        miss = keys[~hit]
        hb, fb = ebytes[keys[hit]].sum(), ebytes[miss].sum()
        sb = ebytes[miss[~ram[miss]]].sum() if ram is not None else 0.0
        ab = 0.0
        hits += int(hit.sum())
        need_n += len(keys)
        if policy == "lru" and len(miss):
            last[keys] = s
            need[keys] = True
            n_miss = np.bincount(miss // E, minlength=L)
            evictable = (res & ~need).reshape(L, E).sum(1)
            n_ev = np.minimum(n_miss, evictable)
            order = _evict_order(np.where(res & ~need, last, BIG), L, E)
            res[order[np.arange(E)[None, :] < n_ev[:, None]]] = False
            res[miss[_first_n_per_layer(miss, E, n_ev)]] = True
            need[keys] = False
        elif policy == "lru":
            last[keys] = s
        elif policy == "promote_on_fetch":
            last[keys] = s
            prom = miss[_first_n_per_layer(miss, E, spares)]   # every boundary refilled the spares
            res[prom] = True
            n_prom = np.bincount(prom // E, minlength=L)
            if len(prom):                             # the boundary: evict the LRU residents, one per promotion
                order = _evict_order(np.where(res, last, BIG), L, E)
                res[order[np.arange(E)[None, :] < n_prom[:, None]]] = False
        elif policy == "lfu":
            u *= kn["decay"]
            np.add.at(u, keys, tr.cnt[a:b])
            if len(miss):
                need[keys] = True
                _pair_swap(res, need, u, miss, L, E, gain=0.0, u_min=-1.0, limit=None, apply=True)
                need[keys] = False
        elif policy == "profile_decay":
            np.add.at(u, keys, tr.cnt[a:b])
            if (s + 1) % kn["every"] == 0:
                if not pending:
                    cin, cout = _pair_swap(hres, None, u, None, L, E, gain=kn["gain"], u_min=kn["u_min"],
                                           limit=kn["swaps"], apply=False)
                    if len(cin):
                        hres[cout], hres[cin] = False, True
                        pending.append((s + kn["lag"], cout, False))
                        pending.append((s + kn["lag"] + kn["d"], cin, True))
                        ab = ebytes[cin].sum()
                        swaps_done += len(cin)
                u *= kn["decay"]
        elif policy == "belady":
            nu[keys] = tr.nxt[a:b]
            cand = res.copy()
            cand[keys] = True
            order = _evict_order(np.where(cand, nu, BIG + 1), L, E)
            res[:] = False
            res[order[np.arange(E)[None, :] < quota[:, None]]] = True
        tot["hit"] += hb
        tot["fetch"] += fb
        tot["admission"] += ab
        tot["ssd"] += sb
        tot["dense"] += dense
        gpu = (dense + hb + fb) / GPU + sb / SSD
        for i, p in enumerate(PCIE):
            t_serial[i] += gpu + (fb + ab) / p
            t_hidden[i] += gpu + fb / p
    ntok = int(tr.tokens.sum())
    out = {"hit_rate": hits / need_n, "steps": len(tr.tokens), "tokens": ntok, "swaps": swaps_done,
           "bytes_per_token": {k: v / ntok for k, v in tot.items()} | {"cpu": 0.0},
           "ms_per_token": {f"pcie{int(p / 1e9)}": 1e3 * t_serial[i] / ntok for i, p in enumerate(PCIE)},
           "ms_per_token_admission_hidden": {f"pcie{int(p / 1e9)}": 1e3 * t_hidden[i] / ntok
                                             for i, p in enumerate(PCIE)}}
    return out


def _pair_swap(res: np.ndarray, need: np.ndarray | None, u: np.ndarray, miss: np.ndarray | None, L: int, E: int, *,
               gain: float, u_min: float, limit: int | None, apply: bool) -> tuple[np.ndarray, np.ndarray]:
    """Strata's pairing per layer: candidates (non-resident; `miss` if given) by count descending against victims
    (resident, not needed now) ascending, while cand >= victim + gain (lfu: strictly greater); the global top
    `limit` by gain. Returns (incoming, outgoing) keys; `apply` writes them into `res`."""
    cand_mask = np.zeros(L * E, bool)
    if miss is not None:
        cand_mask[miss] = True
    else:
        cand_mask = ~res & (u >= u_min)
    vict_mask = res & (~need if need is not None else True)
    INF = np.inf
    cu = np.where(cand_mask, -u, INF).reshape(L, E)
    corder = np.argsort(cu, axis=1, kind="stable")
    cvals = -np.take_along_axis(cu, corder, 1)                     # descending counts; -inf where absent
    vu = np.where(vict_mask, u, INF).reshape(L, E)
    vorder = np.argsort(vu, axis=1, kind="stable")
    vvals = np.take_along_axis(vu, vorder, 1)
    ok = (cvals > vvals + gain) if gain == 0.0 else (cvals >= vvals + gain)
    ok &= np.isfinite(cvals) & np.isfinite(vvals)
    ok = np.cumprod(ok, axis=1).astype(bool)                       # break at the first failing pair
    base = (np.arange(L) * E)[:, None]
    cin, cout, g = (corder + base)[ok], (vorder + base)[ok], (cvals - vvals)[ok]
    if limit is not None and len(g) > limit:
        top = np.argsort(-g, kind="stable")[:limit]
        cin, cout = cin[top], cout[top]
    if apply:
        res[cout], res[cin] = False, True
    return cin, cout


# ---------------------------------------------------------------- the other tables


def expert_bytes(gguf: str) -> tuple[np.ndarray, float, dict, int]:
    """Per-layer bytes of one expert (gate + up + down) and the dense bytes a decode step reads (every non-expert
    tensor but the token embedding), from the file's tensor table."""
    from holon_std.weights.gguf.reader import read
    fields, infos = read(gguf)
    L = fields[f"{fields['general.architecture']}.block_count"]
    per, types, dense, E = np.zeros(L), {}, 0, 0
    for t in infos:
        nbytes = t.rows * t.row_bytes
        if "_exps" in t.name:
            layer, role = int(t.name.split(".")[1]), t.name.split(".")[2]
            per[layer] += nbytes / t.shape[0]
            E = t.shape[0]
            types.setdefault(role, {}).setdefault(t.type.name, 0)
            types[role][t.type.name] += 1
        elif t.name != "token_embd.weight":
            dense += nbytes
    return per, float(dense), types, E


def windows(raw: dict, L: int, E: int, sizes=(1, 2, 3, 4)) -> dict:
    """Distinct experts per layer: per decode step by its decode rows, and per window of w consecutive decode
    tokens of one sequence (verify-window-like groups)."""
    dec = raw["decode"]
    ids, step, seq = raw["ids"][dec].astype(np.int64), raw["step"][dec], raw["seq"][dec]
    k = ids.shape[-1]
    out = {"per_step": {}, "per_window": {}}
    _, sidx, rows = np.unique(step, return_inverse=True, return_counts=True)
    keys = (np.repeat(sidx, L * k) * L + np.tile(np.repeat(np.arange(L), k), len(ids))) * E + ids.reshape(-1)
    per_step = np.bincount(np.unique(keys) // (L * E), minlength=len(rows)) / L
    for b in np.unique(rows):
        out["per_step"][int(b)] = {"steps": int((rows == b).sum()),
                                   "distinct_per_layer": float(per_step[rows == b].mean())}
    for w in sizes:
        tot = n = 0
        for s in np.unique(seq):
            x = ids[seq == s]
            m = len(x) // w
            if not m:
                continue
            x = x[:m * w].reshape(m, w, L, k).transpose(0, 2, 1, 3).reshape(m * L, w * k)
            x = np.sort(x, axis=1)
            tot += int((1 + (np.diff(x, axis=1) != 0).sum(1)).sum())
            n += m * L
        out["per_window"][w] = tot / n if n else None
    return out


def coverage(raw: dict, L: int, E: int, ebytes: np.ndarray, residency: dict, chunks=(64, 256, 1024, 4096)) -> dict:
    """Prefill: per chunk of C consecutive prompt tokens of one request, the share of a layer's experts it routes to;
    and D8's inputs per layer: routed non-resident bytes (staging) against all non-resident bytes (the stream)."""
    pre = ~raw["decode"]
    ids, seq = raw["ids"][pre].astype(np.int64), raw["seq"][pre]
    k = ids.shape[-1]
    out = {}
    for C in chunks:
        covs, routed = [], {f: [] for f in residency}
        for s in np.unique(seq):
            x = ids[seq == s]
            m = len(x) // C
            if not m:
                continue
            x = x[:m * C].reshape(m, C, L, k).transpose(0, 2, 1, 3).reshape(m, L, C * k)
            hit = np.zeros((m, L, E), bool)
            np.put_along_axis(hit, x, True, axis=2)
            covs.append(hit.sum(2).reshape(-1) / E)
            for f, res in residency.items():
                nonres = hit & ~res.reshape(L, E)[None]
                routed[f].append((nonres.sum(2) * ebytes[None]).sum(1) / L)      # mean bytes per layer per chunk
        if not covs:
            out[C] = None
            continue
        c = np.concatenate(covs)
        out[C] = {"chunks": int(len(c) // L),
                  "coverage_mean": float(c.mean()), "coverage_p10": float(np.quantile(c, 0.1)),
                  "coverage_p90": float(np.quantile(c, 0.9)),
                  "routed_nonresident_mb_per_layer": {f: float(np.concatenate(r).mean() / 1e6)
                                                      for f, r in routed.items()},
                  "all_nonresident_mb_per_layer": {f: float(((~res.reshape(L, E)).sum(1) * ebytes).mean() / 1e6)
                                                   for f, res in residency.items()}}
    return out


def skew(counts: np.ndarray, L: int, E: int) -> dict:
    p = counts.reshape(L, E) / counts.reshape(L, E).sum(1, keepdims=True)
    top = np.sort(p, axis=1)[:, ::-1][:, :max(1, round(0.1 * E))].sum(1)
    ent = -(np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0)).sum(1) / np.log(E)
    stats = lambda a: {"mean": float(a.mean()), "min": float(a.min()), "max": float(a.max())}  # noqa: E731
    return {"top10pct_share": stats(top), "entropy_norm": stats(ent),
            "per_layer_top10pct_share": [round(float(x), 4) for x in top]}


def recall(raw: dict, L: int, E: int, k: int, residency: dict) -> dict | None:
    """D11: layer l+1's router on layer l's MoE input vs layer l+1's true top-k, over the sampled decode rows; and
    for a prefetch of the predicted non-resident experts, the share of true misses it covers and its useless bytes."""
    rr, pred = raw["rec_rows"], raw["rec_pred"].astype(np.int64)
    if not len(rr):
        return None
    true = raw["ids"][rr][:, 1:, :].astype(np.int64)                     # [r, L-1, k]
    ink = (pred[..., :k, None] == true[..., None, :]).any(-1).sum(-1)    # predicted top-k ∩ true
    inx = (pred[..., :, None] == true[..., None, :]).any(-1).sum(-1)     # predicted top-k+4 ∩ true
    out = {"rows": int(len(rr)), "recall_topk": float(ink.mean() / k), "recall_topk4": float(inx.mean() / k),
           "precision_topk4": float(inx.mean() / pred.shape[-1]),
           "recall_topk_per_layer": [round(float(x), 4) for x in ink.mean(0) / k],
           "recall_topk4_per_layer": [round(float(x), 4) for x in inx.mean(0) / k]}
    lay = np.arange(1, L)[None, :, None]
    for f, res in residency.items():
        miss = ~res[lay * E + true]                                        # true non-resident routes
        covered = (pred[..., :, None] == true[..., None, :]).any(-2)       # [r, L-1, k] true id predicted
        pmiss = ~res[lay * E + pred]                                       # predicted non-resident
        useful = (pred[..., :, None] == true[..., None, :]).any(-1)
        out[f"prefetch_at_{f}"] = {"miss_coverage": float((miss & covered).sum() / max(1, miss.sum())),
                                   "useless_share_of_prefetch": float((pmiss & ~useful).sum() / max(1, pmiss.sum()))}
    return out


# ---------------------------------------------------------------- driver


def evaluate(test: Decode, profile: np.ndarray, ebytes_key: np.ndarray, dense: float, policies=POLICIES,
             fractions=FRACTIONS, knobs: dict | None = None, ram_fraction: float = 1.0) -> dict:
    """Every policy at every cached fraction. `ram_fraction` < 1: a RAM tier holding that share of the experts (the
    profile's top), the rest read from SSD on a miss (the hook for F9; 1: RAM holds every expert)."""
    ram = None
    if ram_fraction < 1:
        ram = np.zeros(len(profile), bool)
        ram[np.argsort(-profile, kind="stable")[:round(ram_fraction * len(profile))]] = True
    out = {}
    for f in fractions:
        q = quotas(profile, test.L, test.E, round(f * test.L * test.E))
        out[f"{round(f * 100)}%"] = {p: simulate(test, p, *q, ebytes_key, dense, (knobs or {}).get(p), ram)
                                     for p in policies}
    return out


SWEEP = {"profile_decay": {"decay": (0.5, 0.9), "every": (1, 16), "swaps": (32, 384), "gain": (0.5, 4.0),
                           "u_min": (1.0, 4.0)},
         "promote_on_fetch": {"spares": (1, 8)}, "lfu": {"decay": (0.9, 0.99)}}


def sweep(pairs: list[tuple[Decode, np.ndarray]], ebytes_key, dense, fractions=(0.25, 0.47)) -> dict:
    """One knob at a time around the defaults, scored by mean modelled time (PCIe 25) over the tuning pairs."""
    out = {}
    for policy, grid in SWEEP.items():
        rows = []
        for knob, vals in [(None, (None,))] + list(grid.items()):
            for v in vals:
                kn = {} if knob is None else {knob: v}
                ms = []
                for test, prof in pairs:
                    for f in fractions:
                        q = quotas(prof, test.L, test.E, round(f * test.L * test.E))
                        ms.append(simulate(test, policy, *q, ebytes_key, dense, kn)["ms_per_token"]["pcie25"])
                rows.append({"knobs": DEFAULTS[policy] | kn, "ms_per_token_pcie25_mean": float(np.mean(ms))})
        out[policy] = rows
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", help="directory of capture.py's .npz files")
    ap.add_argument("--gguf")
    ap.add_argument("--train", nargs="+", default=["chat", "code"])
    ap.add_argument("--test", nargs="+", default=["agent", "mixed"])
    ap.add_argument("--ptime", help="capture.py's ptime record (prefill compute vs chunk size)")
    ap.add_argument("--ram-fraction", type=float, default=1.0, help="share of experts a RAM tier holds (SSD hook)")
    ap.add_argument("--out")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--report", nargs="+", help="replay JSONs (forward first): print the README's tables")
    args = ap.parse_args()
    if args.selfcheck:
        return selfcheck()
    if args.report:
        return print(report([json.loads(pathlib.Path(f).read_text()) for f in args.report]))
    t0 = time.time()
    ebytes, dense, types, E = expert_bytes(args.gguf)
    raws = {p.stem: load(p) for p in sorted(pathlib.Path(args.traces).glob("*.npz"))}
    dec = {n: decode_steps(r, E) for n, r in raws.items()}
    any_ = next(iter(dec.values()))
    L = any_.L
    k = raws[any_.name]["ids"].shape[-1]
    ekey = np.repeat(ebytes, E)
    prof = {n: d.counts for n, d in dec.items()}
    train = sum(prof[n] for n in args.train)
    others = [n for n in dec if n not in args.train]
    out = {"model": {"layers": L, "experts": E, "top_k": k, "expert_bytes_per_layer": ebytes.tolist(),
                     "expert_bytes_mean": float(ebytes.mean()), "expert_tensor_types": types,
                     "dense_bytes_per_step": dense, "experts_total_bytes": float(ebytes.sum() * E)},
           "rates_placeholder": {"pcie": list(PCIE), "gpu": GPU, "ssd": SSD,
                                 "note": "placeholders until M1 (PCIe) and M3 land; GPU ≈ the RTX 5080"},
           "traces": {n: {"info": r["info"], "rows": int(len(r["decode"])), "decode_rows": int(r["decode"].sum()),
                          "decode_steps": int(len(dec[n].tokens)), "prefill_rows": int((~r["decode"]).sum())}
                      for n, r in raws.items()},
           "train": args.train, "test": args.test, "floor_slots_per_layer": FLOOR, "defaults": DEFAULTS,
           "ram_fraction": args.ram_fraction}
    # held-out: profile from the training traces, each policy on every other trace
    out["heldout"] = {n: evaluate(dec[n], train, ekey, dense, ram_fraction=args.ram_fraction) for n in others}
    # profile transfer: static with the trace's own profile against the training profile
    out["profile_transfer"] = {}
    for n in dec:
        row = {}
        for src_name, src in [("own", prof[n])] + [(m, prof[m]) for m in dec if m != n] + \
                ([("train", train)] if n not in args.train else []):
            row[src_name] = {f"{round(f * 100)}%": simulate(dec[n], "static", *quotas(src, L, E, round(f * L * E)),
                                                             ekey, dense)["hit_rate"] for f in FRACTIONS}
        out["profile_transfer"][n] = row
    # knobs tuned on the training traces (profile of one, scored on the other), then held-out at the tuned knobs
    pairs = [(dec[a], prof[b]) for a in args.train for b in args.train if a != b]
    out["sweep"] = sweep(pairs, ekey, dense) if pairs else {}
    tuned = {p: min(rows, key=lambda r: r["ms_per_token_pcie25_mean"])["knobs"] for p, rows in out["sweep"].items()}
    out["tuned_knobs"] = tuned
    out["heldout_tuned"] = {n: evaluate(dec[n], train, ekey, dense, policies=tuple(tuned), knobs=tuned)
                            for n in args.test if n in dec}
    # the other tables
    res = {f"{round(f * 100)}%": quotas(train, L, E, round(f * L * E))[1] for f in (0.25, 0.47)}
    out["distinct"] = {n: windows(r, L, E) for n, r in raws.items()}
    out["prefill_coverage"] = {n: coverage(r, L, E, ebytes, res) for n, r in raws.items() if (~r["decode"]).any()}
    out["skew"] = {n: skew(d.counts, L, E) for n, d in dec.items()}
    out["recall"] = {n: recall(r, L, E, k, res) for n, r in raws.items()}
    if args.ptime:
        out["prefill_compute_a5000"] = json.loads(pathlib.Path(args.ptime).read_text())
    out["replay_s"] = round(time.time() - t0, 1)
    text = json.dumps(out, indent=1)
    if args.out:
        pathlib.Path(args.out).write_text(text)
    print(summary(out))


def summary(out: dict) -> str:
    lines = []
    for n, by_f in out["heldout"].items():
        lines.append(f"\n## held-out {n} (profile: {'+'.join(out['train'])}) hit% / ms per token @PCIe25 / @50")
        lines.append("frac  " + "  ".join(f"{p:>22}" for p in POLICIES))
        for f, by_p in by_f.items():
            lines.append(f"{f:>4}  " + "  ".join(
                f"{100 * by_p[p]['hit_rate']:5.1f} {by_p[p]['ms_per_token']['pcie25']:6.2f} "
                f"{by_p[p]['ms_per_token']['pcie50']:6.2f}  " for p in POLICIES))
    return "\n".join(lines)


def d8(cov: dict, ptime: dict, L: int) -> list[dict]:
    """D8's choice per chunk size, fraction and PCIe rate: stream every non-resident expert of the next layer during
    this layer if its time hides under the layer's compute or its exposed rest beats routed-only staging's (taken as
    fully exposed). The layer compute is the A5000's prefill step / layers: a placeholder for the bar card's."""
    rows = []
    for C, c in cov.items():
        if c is None or str(C) not in ptime:
            continue
        comp = ptime[str(C)]["step_ms"] / L
        for f, allb in c["all_nonresident_mb_per_layer"].items():
            routed = c["routed_nonresident_mb_per_layer"][f]
            for p in PCIE:
                stream, staged = allb * 1e6 / p * 1e3, routed * 1e6 / p * 1e3
                exposed = max(0.0, stream - comp)
                rows.append({"chunk": int(C), "cached": f, "pcie": int(p / 1e9), "coverage": c["coverage_mean"],
                             "layer_compute_ms": comp, "stream_ms": stream, "staged_ms": staged,
                             "choice": "stream" if exposed <= staged else "staged"})
    return rows


def report(outs: list[dict]) -> str:
    """The README's tables from the forward (and reverse) replay records."""
    fw = outs[0]
    md = []

    def table(head: list[str], rows: list[list]) -> None:
        md.append("| " + " | ".join(head) + " |")
        md.append("|" + "---|" * len(head))
        md.extend("| " + " | ".join(str(x) for x in r) + " |" for r in rows)
        md.append("")

    for n, by_f in fw["heldout"].items():
        if n not in fw["test"]:
            continue
        md.append(f"Held-out **{n}**, profile from {'+'.join(fw['train'])}: hit % / ms per token at PCIe 25 / 50 "
                  "(placeholder rates)\n")
        cell = lambda r: f"{100 * r['hit_rate']:.1f} / {r['ms_per_token']['pcie25']:.2f} / "  # noqa: E731
        table(["cached"] + list(POLICIES), [[f] + [cell(by_p[p]) + f"{by_p[p]['ms_per_token']['pcie50']:.2f}"
                                                   for p in POLICIES] for f, by_p in by_f.items()])
    md.append("Every held-out cell, ms per token at PCIe 25 (best non-Belady in bold)\n")
    rows, wins = [], {}
    for o in outs:
        for n, by_f in o["heldout"].items():
            for f in ("10%", "25%", "47%"):
                by_p = by_f[f]
                ms = {p: by_p[p]["ms_per_token"]["pcie25"] for p in POLICIES}
                best = min((p for p in POLICIES if p != "belady"), key=ms.get)
                for rate in ("pcie25", "pcie50"):
                    b = min((p for p in POLICIES if p != "belady"), key=lambda p: by_p[p]["ms_per_token"][rate])
                    wins[b] = wins.get(b, 0) + 1
                rows.append([n, "+".join(o["train"]), f] + [f"**{ms[p]:.2f}**" if p == best else f"{ms[p]:.2f}"
                                                         for p in POLICIES])
    table(["test", "profile", "cached"] + list(POLICIES), rows)
    md.append(f"Least modelled time (non-Belady), cells won over both rates: {wins}\n")
    md.append("Bytes per decode token by path (MB), held-out agent\n")
    rows = []
    for f in ("25%", "47%"):
        for p in POLICIES:
            b = fw["heldout"]["agent"][f][p]["bytes_per_token"]
            rows.append([f, p] + [f"{b[k] / 1e6:.1f}" for k in ("hit", "fetch", "admission", "cpu", "ssd", "dense")]
                        + [f"{fw['heldout']['agent'][f][p]['ms_per_token_admission_hidden']['pcie25']:.2f}"])
    table(["cached", "policy", "hit", "fetch", "admission", "cpu", "ssd", "dense", "ms @25, admission hidden"], rows)
    md.append("Knob sweep (tuning: profile of chat scored on code and back, 25% and 47%), mean ms per token @25\n")
    rows = []
    for p, rs in fw["sweep"].items():
        base = rs[0]["ms_per_token_pcie25_mean"]
        for r in rs:
            diff = {k: v for k, v in r["knobs"].items() if DEFAULTS[p].get(k) != v}
            rows.append([p, json.dumps(diff) if diff else "defaults", f"{r['ms_per_token_pcie25_mean']:.2f}",
                         f"{100 * (r['ms_per_token_pcie25_mean'] / base - 1):+.1f}%"])
    table(["policy", "knob", "ms", "vs defaults"], rows)
    md.append("Tuned knobs on the held-out traces, ms per token @25 (defaults → tuned)\n")
    rows = []
    for n, by_f in fw["heldout_tuned"].items():
        for f in ("25%", "47%"):
            rows.append([n, f] + [f"{fw['heldout'][n][f][p]['ms_per_token']['pcie25']:.2f} → "
                                  f"{by_f[f][p]['ms_per_token']['pcie25']:.2f}" for p in by_f[f]])
    table(["test", "cached"] + list(next(iter(fw["heldout_tuned"].values()))["25%"]), rows)
    md.append("Profile transfer: static hit % at 25 / 47% cached, by the trace the profile came from\n")
    srcs = ["own", "chat", "code", "agent", "mixed", "train"]
    table(["trace"] + srcs, [[n] + [f"{100 * row[s]['25%']:.1f} / {100 * row[s]['47%']:.1f}" if s in row else "—"
                                    for s in srcs] for n, row in fw["profile_transfer"].items()])
    md.append("Distinct experts per layer: per decode step (by decode rows) and per window of w tokens of one "
              "sequence\n")
    table(["trace", "per step (rows: distinct)", "w=1", "w=2", "w=3", "w=4"],
          [[n, ", ".join(f"{b}: {v['distinct_per_layer']:.1f}" for b, v in d["per_step"].items())]
           + [f"{d['per_window'][str(w)]:.1f}" for w in (1, 2, 3, 4)] for n, d in fw["distinct"].items()])
    md.append("Per-layer skew of decode routing (mean [min, max] over layers)\n")
    mmm = lambda a: f"{a['mean']:.3f} [{a['min']:.3f}, {a['max']:.3f}]"  # noqa: E731
    table(["trace", "top-10% experts' share", "normalized entropy"],
          [[n, mmm(s["top10pct_share"]), mmm(s["entropy_norm"])] for n, s in fw["skew"].items()])
    md.append("Next-layer recall (D11): layer l+1's router on layer l's MoE input, decode rows\n")
    rows = []
    for n, r in fw["recall"].items():
        pl = r["recall_topk_per_layer"]
        rows.append([n, r["rows"], f"{100 * r['recall_topk']:.1f}", f"{100 * min(pl):.1f} / {100 * max(pl):.1f}",
                     f"{100 * r['recall_topk4']:.1f}", f"{100 * r['precision_topk4']:.1f}",
                     f"{100 * r['prefetch_at_25%']['miss_coverage']:.1f}",
                     f"{100 * r['prefetch_at_25%']['useless_share_of_prefetch']:.1f}"])
    table(["trace", "rows", "recall top-8", "per layer min / max", "recall top-12", "precision top-12",
           "25%: misses covered by top-12", "25%: prefetched non-resident unused"], rows)
    md.append("Prefill coverage and D8's inputs (MB per layer; layer compute = A5000 step / 48, a placeholder)\n")
    rows = []
    ptime = fw.get("prefill_compute_a5000", {}).get("result", {})
    for n, cov in fw["prefill_coverage"].items():
        if n not in ("agent", "mixed"):
            continue
        for r in d8(cov, ptime, fw["model"]["layers"]):
            rows.append([n, r["chunk"], f"{100 * r['coverage']:.0f}%", r["cached"], r["pcie"],
                         f"{r['layer_compute_ms']:.2f}", f"{r['stream_ms']:.2f}", f"{r['staged_ms']:.2f}", r["choice"]])
    table(["trace", "chunk", "coverage", "cached", "PCIe GB/s", "layer compute ms", "stream all ms", "routed-only ms",
           "D8"], rows)
    return "\n".join(md)


def selfcheck() -> None:
    """Synthetic traces: Belady bounds every demand policy; LRU and static agree on a trace the seed covers."""
    rng = np.random.default_rng(1)
    L, E, k, R = 3, 16, 2, 400
    hot = rng.permutation(E)[:6]
    ids = np.where(rng.random((R, L, k)) < 0.8, rng.choice(hot, (R, L, k)), rng.integers(0, E, (R, L, k)))
    ids[..., 1] = (ids[..., 0] + 1 + rng.integers(0, E - 1, (R, L))) % E       # unique per row
    raw = {"ids": ids.astype(np.uint8), "step": np.arange(R) // 2, "decode": np.ones(R, bool), "name": "syn"}
    tr = decode_steps(raw, E)
    assert tr.tokens.sum() == R and (np.diff(tr.ptr) <= 2 * k * L).all()
    eb = np.ones(L * E)
    for frac in (0.25, 0.5):
        q = quotas(tr.counts, L, E, round(frac * L * E))
        r = {p: simulate(tr, p, *q, eb, 0.0)["hit_rate"] for p in POLICIES}
        assert all(r["belady"] >= v - 1e-12 for v in r.values()), r
        print(frac, {p: round(v, 3) for p, v in r.items()})
    full = quotas(tr.counts, L, E, L * E)
    assert all(simulate(tr, p, *full, eb, 0.0)["hit_rate"] == 1.0 for p in POLICIES)
    print("selfcheck ok")


if __name__ == "__main__":
    main()
