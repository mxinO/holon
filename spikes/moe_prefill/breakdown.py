"""Prefill GPU time by component from an nsys SQLite export (`nsys export --type sqlite`) of `prefill.py --profile`.

Each profiled request is the NVTX range `req:<length>`; a kernel belongs to the request whose range holds its launch
(the runtime call with its correlation id) and to the innermost NVTX region (`experts`, `dense`, `router`, `combine`)
open on the launching thread at that call. Inside a region kernels are split by name (GEMM, activation quantization,
id helper, SwiGLU, combine); kernels outside every region are named by x4_gap's rules (attention, norms, glue, ...).
`busy` is the union of kernel and copy intervals; `wall` the request's range (time to the first token).

    python spikes/moe_prefill/breakdown.py <report.sqlite> [--top 12]
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sqlite3
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "x4_gap"))
import nsys_breakdown  # noqa: E402  (x4_gap's name rules and interval union)

REGIONS = ("experts", "dense", "router", "combine")
INSIDE = [("GEMM", r"mul_mat_q|stream_k_fixup|mul_mat_vec"), ("act quant (q8_1)", r"quantize"),
          ("id helper", r"mm_ids|ids_helper"), ("SwiGLU / act", r"swiglu|act_and_mul|silu|glu"),
          ("combine", r"moe_weighted|weighted_sum|moe_combine|moe_sum"), ("router", r"topk|router|gemv|gemm|dot")]


def component(region: str | None, short: str, demangled: str, grid: tuple) -> str:
    if region is None:
        cat = nsys_breakdown.category(short, demangled, grid)
        return "other: " + cat
    if region in ("router", "combine"):
        return region
    sub = next((name for name, rule in INSIDE if re.search(rule, short, re.I) or re.search(rule, demangled[:200],
                                                                                            re.I)), "glue")
    return f"{region}: {sub}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sqlite")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    db = sqlite3.connect(a.sqlite)
    strings = dict(db.execute("select id, value from StringIds"))
    ranges = [(s, e, t or strings.get(ti), tid) for s, e, t, ti, tid in db.execute(
        "select start, end, text, textId, globalTid from NVTX_EVENTS where end is not null")]
    requests = sorted((s, e, name[4:]) for s, e, name, _ in ranges if name and name.startswith("req:"))
    # (time, order, kind, payload) per thread: range opens and closes, launches; a stack gives the innermost region
    events = collections.defaultdict(list)
    for s, e, name, tid in ranges:
        if name in REGIONS:
            events[tid] += [(s, 0, "open", name), (e, 2, "close", name)]
    for s, cid, tid in db.execute("select start, correlationId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME"):
        events[tid].append((s, 1, "launch", cid))
    region_of, launch_at = {}, {}
    for evs in events.values():
        stack = []
        for t, _, kind, x in sorted(evs):
            if kind == "open":
                stack.append(x)
            elif kind == "close":
                stack.remove(x) if x in stack else None
            else:
                region_of[x], launch_at[x] = (stack[-1] if stack else None), t
    kernels = list(db.execute("select start, end, correlationId, shortName, demangledName, gridX, gridY, gridZ "
                              "from CUPTI_ACTIVITY_KIND_KERNEL"))
    copies = list(db.execute("select start, end, correlationId from CUPTI_ACTIVITY_KIND_MEMCPY"))
    out = {}
    for s0, e0, length in requests:
        inside = [k for k in kernels if s0 <= launch_at.get(k[2], -1) <= e0]
        cps = [c for c in copies if s0 <= launch_at.get(c[2], -1) <= e0]
        comp, count, names = collections.Counter(), collections.Counter(), collections.Counter()
        for s, e, cid, sh, de, gx, gy, gz in inside:
            short, dem = strings[sh], strings.get(de, "")
            c = component(region_of.get(cid), short, dem, (gx, gy, gz))
            comp[c] += e - s
            count[c] += 1
            names[(c, short[:60])] += e - s
        busy = nsys_breakdown.union([(s, e) for s, e, *_ in inside] + [(s, e) for s, e, _ in cps])
        ms = 1e-6
        out[length] = {"wall_ms": round((e0 - s0) * ms, 1), "busy_ms": round(busy * ms, 1),
                       "kernel_ms": round(sum(comp.values()) * ms, 1), "memcpy_ms": round(sum(e - s for s, e, _ in cps)
                                                                                         * ms, 1),
                       "components_ms": {c: [round(v * ms, 1), count[c]] for c, v in comp.most_common()},
                       "top_kernels_ms": [[c, k, round(v * ms, 1)] for (c, k), v in names.most_common(a.top)]}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
