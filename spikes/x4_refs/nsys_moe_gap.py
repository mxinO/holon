"""llama.cpp `--moe-cache-mib`'s host round trip per MoE layer, from an nsys profile (exported to SQLite) of a
llama-server run whose last request is a batch-1 decode (bench/offload_ctx.py worker ... --wrap "nsys profile -t cuda
--cuda-graph-trace=graph ...": a CUDA graph counts as GPU work from its start to its end; with =node, each kernel).

Per MoE layer the cache's copy callback reads the selected expert ids to the host (a 4 x n_expert_used-byte D2H copy),
synchronizes the stream, plans the LRU on the host, uploads the misses and the layer's slot map (H2D), and the
scheduler then launches the layer's expert split. Decode steps are delimited by the logits copy (D2H of n_vocab x 4
bytes); the last `--tokens` steps are read. For each ids copy:

  gap      = start of the first kernel after the ids copy - end of the last kernel before it (GPU compute idle)
  upload   = time the GPU spends in H2D copies of expert bytes inside the gap (the misses; the part a device-side
             resolve also pays as fetch)
  host     = gap - upload: sync wake-up, id read, LRU plan, upload and slot-map issue, the next split's launch

Per token: step time, kernel busy time (union of kernel intervals), the sums of the layers' gap / upload / host.

    python spikes/x4_refs/nsys_moe_gap.py <profile.sqlite> [--tokens 200] [--n-vocab 151936] [--top-k 8]
"""
from __future__ import annotations

import argparse
import bisect
import json
import sqlite3
import statistics

H2D, D2H = 1, 2


def q(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]


def busy(intervals: list[tuple[int, int]], lo: int, hi: int) -> int:
    """Length of the union of `intervals` (sorted by start) clipped to [lo, hi]."""
    total, cur_s, cur_e = 0, None, None
    for s, e in intervals:
        s, e = max(s, lo), min(e, hi)
        if e <= s:
            continue
        if cur_e is None or s > cur_e:
            total += (cur_e - cur_s) if cur_e is not None else 0
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return total + ((cur_e - cur_s) if cur_e is not None else 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sqlite")
    ap.add_argument("--tokens", type=int, default=200, help="decode steps read, the last ones of the profile")
    ap.add_argument("--n-vocab", type=int, default=151936)
    ap.add_argument("--top-k", type=int, default=8)
    a = ap.parse_args()
    db = sqlite3.connect(a.sqlite)
    kernels = list(db.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL"))
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" in tables:                      # --cuda-graph-trace=graph: whole graphs
        kernels += db.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE")
    kernels.sort()
    kstart = [s for s, _ in kernels]
    kend_sorted = sorted(e for _, e in kernels)
    copies = list(db.execute("SELECT start, end, bytes, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start"))
    logits = [e for s, e, b, k in copies if k == D2H and b == a.n_vocab * 4]
    ids = [(s, e) for s, e, b, k in copies if k == D2H and b == a.top_k * 4]
    uploads = [(s, e, b) for s, e, b, k in copies if k == H2D and b >= 65536]    # expert bytes, not slot maps

    steps = list(zip(logits[-a.tokens - 1:-1], logits[-a.tokens:], strict=True))
    per_layer, per_token = [], []
    for lo, hi in steps:
        layers = [(s, e) for s, e in ids if lo < s < hi]
        gaps = []
        for s, e in layers:
            i = bisect.bisect_right(kend_sorted, s) - 1                 # last kernel ending before the ids copy
            prev = kend_sorted[i]
            j = bisect.bisect_left(kstart, e)                           # first kernel starting after it
            nxt = kstart[j]
            up = busy(sorted((us, ue) for us, ue, _ in uploads if prev < us < nxt), prev, nxt)
            nbytes = sum(b for us, _, b in uploads if prev < us < nxt)
            gaps.append({"gap_us": (nxt - prev) / 1e3, "upload_us": up / 1e3, "host_us": (nxt - prev - up) / 1e3,
                         "upload_mb": nbytes / 1e6})
        per_layer += gaps
        per_token.append({"step_ms": (hi - lo) / 1e6, "kernel_ms": busy(kernels, lo, hi) / 1e6, "layers": len(layers),
                          "gap_ms": sum(g["gap_us"] for g in gaps) / 1e3,
                          "upload_ms": sum(g["upload_us"] for g in gaps) / 1e3,
                          "host_ms": sum(g["host_us"] for g in gaps) / 1e3,
                          "upload_mb": sum(g["upload_mb"] for g in gaps)})

    def dist(rows: list[dict], key: str) -> dict:
        xs = [r[key] for r in rows]
        return {"median": round(statistics.median(xs), 3), "mean": round(statistics.mean(xs), 3),
                "p10": round(q(xs, 0.1), 3), "p90": round(q(xs, 0.9), 3)}

    hit = [g for g in per_layer if g["upload_mb"] == 0]
    miss = [g for g in per_layer if g["upload_mb"] > 0]
    out = {"tokens": len(per_token), "moe_layers_per_token": statistics.median(t["layers"] for t in per_token),
           "per_token": {k: dist(per_token, k) for k in ("step_ms", "kernel_ms", "gap_ms", "upload_ms", "host_ms",
                                                         "upload_mb")},
           "per_layer_all": {k: dist(per_layer, k) for k in ("gap_us", "upload_us", "host_us")},
           "per_layer_no_miss": {"share": round(len(hit) / len(per_layer), 3),
                                 **{k: dist(hit, k) for k in ("gap_us", "host_us")}},
           "per_layer_with_miss": {"share": round(len(miss) / len(per_layer), 3),
                                   **{k: dist(miss, k) for k in ("gap_us", "upload_us", "host_us", "upload_mb")}}}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
