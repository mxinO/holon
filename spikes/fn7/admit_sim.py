"""FN step 7 (budgeted admission, predict first, as X's M6 did): replays `admit_trace.py`'s decode and verify steps
under a placement policy and a CPU share, on the CPU. Per step and layer: the routed experts not resident are its
misses; the CPU takes ⌊s·n⌋ of them, the lowest-ranked (best routing rank, then expert id, as F5b's resolve); the
GPU reads the rest over PCIe (the critical path). Admissions, from decayed counts as F11's (× 0.915 per step, + 1
per entry): - `all` (today's F11): every call admits its candidates (its misses but the CPU's) into its coldest
unrouted residents while the candidate's score is strictly greater; - `budget`: per step, the misses (the CPU's too)
paired in each layer with its coldest unrouted residents while the candidate beats its victim by `gain`, then the B
best pairs by score gain across layers (Strata's adaptive swaps). An admitted miss the GPU read costs no more than
its read; one the CPU served is copied on the link besides. Reports per step: critical-path pieces (experts the GPU
read), link pieces (those plus the CPU-served admissions), the hit rate (routed entries resident), admissions.
Usage: admit_sim.py <trace.npz> [--out json]"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np

DECAY = np.float32(0.915)


def simulate(tr: dict, share: float, policy: str, budget: int, gain: float, warm: int) -> dict:
    ids, res, score = tr["ids"], tr["resident"].copy(), tr["score"].astype(np.float32).copy()
    steps, groups, t, k = ids.shape
    tot = dict.fromkeys(("critical", "link", "entries", "hits", "admitted", "cpu"), 0)
    for i in range(steps):
        pairs, extra = [], 0
        cpu_sets = []
        for g in range(groups):
            x = ids[i, g]
            live = x >= 0
            e_all, ranks = x[live], np.broadcast_to(np.arange(k), x.shape)[live]
            counts = np.bincount(e_all, minlength=res.shape[1]).astype(np.float32)
            routed = counts > 0
            if i >= warm:
                tot["entries"] += int(live.sum())
                tot["hits"] += int(res[g, e_all].sum())
            score[g] = score[g] * DECAY + counts                         # f32, one add per entry (counts are small)
            misses = np.flatnonzero(routed & ~res[g])
            best = {int(e): int(ranks[e_all == e].min()) for e in misses}
            c = int(share * len(misses))
            cpu = {e for _, e in sorted((best[e], e) for e in misses)[len(misses) - c:]} if c else set()
            cpu_sets.append(cpu)
            if i >= warm:
                tot["critical"] += len(misses) - len(cpu)
                tot["cpu"] += len(cpu)
            cands = sorted((e for e in misses if policy == "budget" or e not in cpu), key=lambda e: (-score[g, e], e))
            victims = sorted(np.flatnonzero(res[g] & ~routed), key=lambda e: (score[g, e], e))
            for a, b in zip(cands, victims, strict=False):
                if not score[g, a] > score[g, b] + np.float32(gain):
                    break
                pairs.append((float(score[g, a] - score[g, b]), g, int(a), int(b)))
        if policy == "budget":
            pairs = sorted(pairs, key=lambda p: (-p[0], p[1], p[2]))[:budget]
        for _, g, a, b in pairs:
            res[g, a], res[g, b] = True, False
            extra += a in cpu_sets[g]
        if i >= warm:
            tot["admitted"] += len(pairs)
            tot["link"] += extra
    n = steps - warm
    return {"share": share, "policy": policy, "budget": budget, "gain": gain,
            "critical_per_step": round(tot["critical"] / n, 1),
            "link_per_step": round((tot["critical"] + tot["link"]) / n, 1),
            "cpu_per_step": round(tot["cpu"] / n, 1), "admitted_per_step": round(tot["admitted"] / n, 1),
            "hit_rate": round(tot["hits"] / tot["entries"], 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--warm", type=int, default=20, help="steps replayed before counting")
    ap.add_argument("--out")
    a = ap.parse_args()
    tr = dict(np.load(a.trace))
    rows = [simulate(tr, 0.0, "all", 0, 0.0, a.warm)]
    rows += [simulate(tr, s, "all", 0, 0.0, a.warm) for s in (0.3, 0.45, 0.55)]
    rows += [simulate(tr, s, "budget", b, g, a.warm)
             for s, b, g in itertools.product((0.0, 0.3, 0.45, 0.55), (16, 32, 64, 128), (0.0, 1.5))]
    base = rows[0]
    for r in rows:
        r["critical_vs_base"] = round(r["critical_per_step"] / base["critical_per_step"] - 1, 3)
        r["hit_points_vs_base"] = round(100 * (r["hit_rate"] - base["hit_rate"]), 2)
    print(json.dumps({"steps": int(tr["ids"].shape[0]), "rows": rows}, indent=1))
    if a.out:
        open(a.out, "w").write(json.dumps({"steps": int(tr["ids"].shape[0]), "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
