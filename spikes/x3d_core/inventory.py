"""The per-step kernel and copy inventory of a `profile.py` report (nsys SQLite export): every kernel and memcpy of a
graphed batch-1 decode step, in stream order, named by the step-loop phase that issued it.

Each step is an NVTX range "launch" (graphed steps inside "decode", eager ones inside "eager"); its work is what the
CUDA runtime calls inside that range enqueued (by correlation id; a graph launch's kernels carry its id). The eager
steps attribute: the innermost NVTX range around each call names its phase, and the graphed step's i-th kernel takes
the phase of the eager step's i-th kernel (the replay is the eager launches, recorded; names are checked to match).
Durations are medians over the graphed steps; `span` is a phase's first start to its last end (gaps included).

    python spikes/x3d_core/inventory.py <report.sqlite> [--json out] [--phases prepare store logits ...]
"""
from __future__ import annotations

import argparse
import bisect
import collections
import json
import sqlite3
import statistics

CORE = ("prepare", "logits", "store", "upload", "launch", "observe", "writes")   # phases whose own work is core's


def ranges(db, strings, name):
    rows = db.execute("select start, end, text, textId, globalTid from NVTX_EVENTS where end is not null")
    return sorted((a, b, tid) for a, b, t, ti, tid in rows if (t or strings.get(ti)) == name)


def steps(db, strings) -> dict[str, list[list[dict]]]:
    """Per window ("eager", "decode"): per step, its kernels and copies in start order, each with its phase."""
    nvtx = [(a, b, t or strings.get(ti), tid) for a, b, t, ti, tid in
            db.execute("select start, end, text, textId, globalTid from NVTX_EVENTS where end is not null")]
    calls = list(db.execute("select start, end, correlationId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME"))
    work = collections.defaultdict(list)          # correlation id -> its kernels / copies
    for a, b, s, d, corr in db.execute("select start, end, shortName, demangledName, correlationId "
                                       "from CUPTI_ACTIVITY_KIND_KERNEL"):
        work[corr].append({"start": a, "end": b, "name": strings[s], "long": strings.get(d, "")[:200]})
    kinds = {1: "memcpy H2D", 2: "memcpy D2H", 8: "memcpy D2D", 10: "memcpy P2P"}
    for a, b, n, k, corr in db.execute("select start, end, bytes, copyKind, correlationId "
                                       "from CUPTI_ACTIVITY_KIND_MEMCPY"):
        work[corr].append({"start": a, "end": b, "name": f"{kinds.get(k, k)} {n} B", "long": ""})
    for a, b, k, corr in db.execute("select start, end, 'memset', correlationId from CUPTI_ACTIVITY_KIND_MEMSET"):
        work[corr].append({"start": a, "end": b, "name": "memset", "long": ""})
    calls.sort()
    starts = [c[0] for c in calls]
    out = {}
    for window in ("eager", "decode"):
        (w0, w1, _), = ranges(db, strings, window)
        launches = [(a, b) for a, b, _ in ranges(db, strings, "launch") if w0 <= a < w1]
        out[window] = []
        for a, b in launches:
            inner = [(x, y, n) for x, y, n, _ in nvtx if a <= x and y <= b and n != "launch"]
            items = []
            for c0, c1, corr, _ in calls[bisect.bisect_left(starts, a): bisect.bisect_right(starts, b)]:
                around = [r for r in inner if r[0] <= c0 and c1 <= r[1]]
                phase = min(around, key=lambda r: r[1] - r[0])[2] if around else "launch"
                items += [dict(w, phase=phase) for w in work.get(corr, [])]
            out[window].append(sorted(items, key=lambda w: w["start"]))
    return out


def inventory(db, strings) -> dict:
    s = steps(db, strings)
    eager, graphed = s["eager"][-1], s["decode"]
    # Steps that sync a block-table row (a new page every `page` tokens) add its copy and kernel: the modal step
    # length is the inventory, the others are reported.
    lengths = collections.Counter(len(g) for g in graphed)
    n = lengths.most_common(1)[0][0]
    modal = [g for g in graphed if len(g) == n]
    names = collections.Counter(w["name"] for w in modal[0])
    odd = [list((collections.Counter(w["name"] for w in g) - names).elements()) for g in graphed if len(g) != n]
    graphed = modal
    # The graphed step = [launch-time copies before] + the replay + [copies after]; align the replay's kernels with
    # the eager step's kernels in order (copies are matched by name and kept in place).
    ek = [w for w in eager if not w["name"].startswith(("memcpy", "memset"))]
    rows, k = [], 0
    for i, w in enumerate(graphed[0]):
        dur = statistics.median(g[i]["end"] - g[i]["start"] for g in graphed) / 1e3
        gap = statistics.median(g[i]["start"] - g[i - 1]["end"] for g in graphed) / 1e3 if i else 0.0
        if w["name"].startswith(("memcpy", "memset")):
            phase = next((e["phase"] for e in eager if e["name"] == w["name"]), "launch")
        else:
            if k >= len(ek) or ek[k]["name"] != w["name"]:
                raise SystemExit(f"kernel {k}: graphed {w['name']} vs eager {ek[k]['name'] if k < len(ek) else None}")
            phase = ek[k]["phase"]
            k += 1
        rows.append({"i": i, "phase": phase, "name": w["name"], "long": w["long"], "us": round(dur, 2),
                     "gap_us": round(gap, 2)})
    steps_us = [(g[-1]["end"] - g[0]["start"]) / 1e3 for g in graphed]
    per_phase = collections.defaultdict(lambda: {"kernels": 0, "copies": 0, "us": 0.0, "span_us": 0.0})
    for r in rows:
        p = per_phase[r["phase"]]
        p["copies" if r["name"].startswith(("memcpy", "memset")) else "kernels"] += 1
        p["us"] = round(p["us"] + r["us"], 2)
    for phase in per_phase:                   # a phase's span: gaps before its kernels included, over contiguous runs
        per_phase[phase]["span_us"] = round(sum(r["us"] + r["gap_us"] for r in rows if r["phase"] == phase), 2)
    return {"graphed_steps": len(graphed), "other_steps": dict(lengths), "other_steps_extra": odd[:2], "step_gpu_us": round(statistics.median(steps_us), 1),
            "phases": dict(per_phase), "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sqlite")
    ap.add_argument("--json")
    ap.add_argument("--phases", nargs="*", default=None, help="rows to print (default: core's phases)")
    a = ap.parse_args()
    db = sqlite3.connect(a.sqlite)
    strings = dict(db.execute("select id, value from StringIds"))
    inv = inventory(db, strings)
    show = set(a.phases) if a.phases is not None else set(CORE)
    print(f"graphed steps: {inv['graphed_steps']}, step GPU span {inv['step_gpu_us']} us")
    for phase, p in sorted(inv["phases"].items(), key=lambda kv: -kv[1]["span_us"]):
        print(f"  {phase:28s} kernels {p['kernels']:4d} copies {p['copies']:3d}  {p['us']:9.1f} us  "
              f"span {p['span_us']:9.1f} us")
    for r in inv["rows"]:
        if r["phase"] in show or r["phase"].startswith(tuple(show)):
            print(f"  {r['i']:4d} {r['phase']:24s} {r['us']:6.2f} +{r['gap_us']:5.2f}  {r['name'][:60]}  "
                  f"{r['long'][:90]}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(inv, f, indent=1)


if __name__ == "__main__":
    main()
