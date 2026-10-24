"""FN's verify pass, Holon against Strata, by component (the lead's call after §23.7): one graphed MTP verify pass
(1 + 3 rows, the head's three draft sub-steps) as served, its GPU time split into the same buckets on both engines.

`run` (Holon, under `nsys profile --cuda-graph-trace=node --trace=cuda,nvtx`, the whole process): the deployment
serves one bench prompt (a key of a bench plan's requests.json), then `--steps` verify steps, each in an NVTX `step`
range. `breakdown.instrument`'s component ranges are installed before the graphs are captured, so every graph node
is created inside its component's range; the trace's node-creation events carry that time and thread.

`holon` reads the trace: each kernel, memcpy and memset of the measured steps goes to its step (its launch: the graph
launch or the eager call) and to the component open where it was made (the node's creation, or the eager launch).
`moe` is split by kernel into the miss fetch (`k_fetch*`), router and resolve (`holon_moe_router`, `k_resolve`, the
top-k) and the routed experts (mmvq, quantize, activation, combine). Per step: each bucket's GPU ms, the step's
period (first kernel to the next step's first), its busy time (the union of its operations) and the idle rest
(host and launch gaps), and the fetched bytes (the store's tap) over the fetch's ms.

`strata` reads a trace of Strata's `--serve` (`nsys profile --cuda-graph-trace=node`): its kernels between the
first and last decode window, bucketed by name (`STRATA_BUCKETS`, from its source files), per window (the windows
from its `strata decode timing` log line).

    nsys profile --cuda-graph-trace=node --trace=cuda,nvtx --sample=none -o <rep> python spikes/fn7/verify_split.py \\
        run --model <gguf> --flags '<MTP deployment>' --requests <requests.json> --key 4096/0 --out <json>
    python spikes/fn7/verify_split.py holon --rep <rep>.nsys-rep --run <json> --out <json>
    python spikes/fn7/verify_split.py strata --rep <rep>.nsys-rep --log <server log> --out <json>"""
from __future__ import annotations

import argparse
import bisect
import collections
import json
import pathlib
import re
import sqlite3
import statistics
import subprocess
import sys
import time
import types

ROOT = pathlib.Path(__file__).resolve().parents[2]
COMPONENTS = ("hc.mix", "hc.combine", "gdn", "qsa", "moe", "shared", "ple", "lm_head")
ROUTE = ("holon_moe_router", "k_resolve", "DeviceScan", "topk", "sort")


def run(a) -> dict:
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT / "spikes/fn6"))
    sys.path.insert(0, str(ROOT / "spikes/fn7"))
    import breakdown
    from _engine import engine
    from holon.api import SamplingParams
    from holon.core import executor

    capture = executor.Executor.capture_graphs

    def labelled(self):                      # the component ranges open while the graphs' nodes are created
        breakdown.instrument(types.SimpleNamespace(executor=self))
        capture(self)
    executor.Executor.capture_graphs = labelled
    plan = json.loads(pathlib.Path(a.requests).read_text())
    ids = next(r["ids"] for r in plan["requests"] if r["key"] == a.key)
    e = engine(a.model, a.flags, max_seqs=1, max_model_len=33792)
    e.generate([plan["warmup"]["ids"]], SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True))
    e.add(ids, SamplingParams(temperature=0.0, max_tokens=a.steps * 4 + 64, ignore_eos=True))
    while e.has_work() and not any(d.token_ids for d in e.step()):
        pass
    for _ in range(4):
        e.step()
    store = e.ctx.shared_objects.get("std.moe.offload")
    missed = {"units": 0, "bytes": 0, "steps": 0, "on": False}
    if store is not None:                    # the measured steps' fetched misses (not resident before the step)
        unit = np.array([[c.unit_bytes for _, c in sorted(store.groups[g].items())] for g in store._order])
        prev = {"map": store.residency.host.copy()}

        def tap(step, record) -> None:
            if missed["on"]:
                gone = record["routed"] & ~(prev["map"] >= 0).all(-1) & ~record["cpu"]
                missed["units"] += int(gone.sum())
                missed["bytes"] += int((gone[..., None] * unit[:, None, :]).sum())
                missed["steps"] += 1
            prev["map"] = store.residency.host.copy()
        store.taps.append(tap)
    torch.cuda.synchronize()
    missed["on"], walls, tokens = True, [], 0
    for _ in range(a.steps):
        t0 = time.perf_counter()
        torch.cuda.nvtx.range_push("step")
        out = e.step()
        torch.cuda.nvtx.range_pop()
        walls.append(time.perf_counter() - t0)
        tokens += sum(len(d.token_ids) for d in out)
    torch.cuda.synchronize()
    missed["on"] = False
    stats = {k: v for k, v in e.stats().items() if k.startswith("spec")}
    e.close()
    return {"key": a.key, "steps": len(walls), "host_ms": [round(w * 1e3, 3) for w in walls], "tokens": tokens,
            "spec": stats, "missed": {k: v for k, v in missed.items() if k != "on"}, "flags": a.flags}


def _db(rep: str) -> sqlite3.Connection:
    rep = pathlib.Path(rep)
    db = rep.with_suffix(".sqlite")
    if not db.exists() or db.stat().st_mtime < rep.stat().st_mtime:      # (a stale export of an earlier report)
        subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite", "true", "-o", str(db), str(rep)],
                       check=True, capture_output=True)
    return sqlite3.connect(db)


def _ops(c: sqlite3.Connection) -> list[tuple]:
    """(start, end, correlationId, graphNodeId, name) of every kernel, memcpy and memset."""
    ops = list(c.execute("SELECT k.start, k.end, k.correlationId, k.graphNodeId, s.value FROM "
                         "CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.shortName"))
    for table, kind in (("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"), ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")):
        try:
            ops += [(*r, kind) for r in c.execute(f"SELECT start, end, correlationId, graphNodeId FROM {table}")]
        except sqlite3.OperationalError:
            pass
    return sorted(ops)


def _busy(spans: list[tuple[int, int]]) -> int:
    total, cur = 0, None
    for s, e in sorted(spans):
        if cur is None or s > cur:
            total, cur = total + e - s, e
        elif e > cur:
            total, cur = total + e - cur, e
    return total


def holon(a) -> dict:
    c = _db(a.rep)
    names = dict(c.execute("SELECT id, value FROM StringIds"))
    ranges = collections.defaultdict(list)                        # tid -> [(start, end, text)]
    for start, end, text, tid, text_id in c.execute(
            "SELECT start, end, text, globalTid, textId FROM NVTX_EVENTS WHERE eventType = 59 AND end IS NOT NULL"):
        ranges[tid].append((start, end, text if text is not None else names.get(text_id, "")))
    for v in ranges.values():
        v.sort()

    def open_at(tid, t):
        rs = ranges.get(tid, [])
        return [r for r in rs[:bisect.bisect_right(rs, (t, float("inf"), ""))] if r[1] >= t]

    def component(tid, t) -> str:
        rs = open_at(tid, t)
        inner = [r for r in rs if r[2] in COMPONENTS or r[2] == "mtp" or r[2].startswith("mtp.")]
        comp = max(inner)[2] if inner else "other"
        if any(r[2] == "mtp" for r in rs):
            comp = comp if comp.startswith("mtp") else "mtp"
        return comp
    created, original = {}, {}                                    # node -> its creation; a clone -> its original
    for start, tid, node, orig in c.execute(
            "SELECT start, globalTid, graphNodeId, originalGraphNodeId FROM CUDA_GRAPH_NODE_EVENTS ORDER BY start"):
        if orig is None:
            created.setdefault(node, (tid, start))
        else:
            original[node] = orig

    def made(node):                    # an executable graph's node -> the captured node it was cloned from
        for _ in range(8):
            if node not in original:
                break
            node = original[node]
        return created.get(node, (None, None))
    launch = {corr: (tid, t) for corr, tid, t in c.execute(
        "SELECT correlationId, globalTid, start FROM CUPTI_ACTIVITY_KIND_RUNTIME")}
    steps = sorted((s, e) for v in ranges.values() for s, e, text in v if text == "step")
    per = [collections.Counter() for _ in steps]
    spans = [[] for _ in steps]
    first = [None] * len(steps)
    top = collections.defaultdict(collections.Counter)
    for start, end, corr, node, name in _ops(c):
        tid, t = launch.get(corr, (None, None))
        if t is None:
            continue
        i = bisect.bisect_right(steps, (t, float("inf"))) - 1
        if i < 0 or t > steps[i][1]:
            continue
        where = made(node) if node is not None else (tid, t)
        comp = component(*where) if where[0] is not None else "other"
        if comp in ("moe", "mtp.moe"):
            pre = comp[:-3]
            comp = (pre + "fetch" if name.startswith("k_fetch") else
                    pre + "route" if any(name.startswith(r) or r in name for r in ROUTE) else pre + "experts")
        per[i][comp] += (end - start) / 1e6
        spans[i].append((start, end))
        first[i] = start if first[i] is None else min(first[i], start)
        top[comp][name] += (end - start) / 1e6
    n = len(steps)
    keep = [i for i in range(n - 1) if first[i] is not None and first[i + 1] is not None]
    med = lambda xs: round(statistics.median(xs), 4)              # noqa: E731
    comps = sorted({k for p in per for k in p})
    info = json.loads(pathlib.Path(a.run).read_text())
    period = [(first[i + 1] - first[i]) / 1e6 for i in keep]
    busy = [_busy(spans[i]) / 1e6 for i in keep]
    out = {"steps": len(keep), "period_ms": med(period), "busy_ms": med(busy),
           "idle_ms": med([p - b for p, b in zip(period, busy, strict=True)]),
           "components_ms": {k: round(statistics.mean(per[i][k] for i in keep), 4) for k in comps},
           "top": {k: {nm: round(v / len(keep), 4) for nm, v in top[k].most_common(8)} for k in comps}}
    m = info.get("missed") or {}
    fetch = out["components_ms"].get("fetch")
    if m.get("steps") and fetch:
        out["fetch"] = {"mib_per_step": round(m["bytes"] / m["steps"] / 2**20, 2),
                        "units_per_step": round(m["units"] / m["steps"], 2),
                        "gbs": round(m["bytes"] / m["steps"] / (fetch / 1e3) / 1e9, 1)}
    return info | out


# Strata's kernels by source (src/kernels/cuda/*.cu and src/prefill/*.cu), first match wins
STRATA_BUCKETS = (
    ("fetch", r"copy_from_mapped|copy_rows_from_mapped|copy_i32_from_mapped|fetch_blobs|copy16|memcpy"),
    ("route", r"route|router_top10|topk_reg|route_prior"),
    ("experts", r"mmvq|mul_mat_vec|quantize_q8_1|swiglu|s2_|s_gemv|group_resident|hit_select|gu_kernel|down_kernel|"
                r"gu_pair|down_pair|add_hits|activation_correction|combine|reduce_experts|dequant_gu|dequant_flat"),
    ("shared", r"shared|scalar_gate|sigmoid_scale|scalar_sigmoid"),
    ("qsa", r"qsa|attend|indexer|append|block_scores|block_topk|score|kv_append|kv_gather|norm_rope|rope|gate_attn|"
            r"select"),
    ("gdn", r"gdn|conv_silu|beta_sigmoid|gate_softplus|l2_norm|out_norm|conv_commit"),
    ("hc", r"^gr_|weighted_rms_norm|post|pre_gated|down_silu|broadcast_streams|add_streams"),
    ("ple", r"ple|history|conv_residual|rms_rep|bcast|gnorm"),
    ("lm_head", r"argmax|bf16_gemv|bf16_f32_mmvf|q6_k|q5_k|sampler"),
)


def strata(a) -> dict:
    c = _db(a.rep)
    log = pathlib.Path(a.log).read_text(errors="replace")
    timing = re.findall(r"strata decode timing: (\d+) windows.*?([\d.]+) ms/window = verify ([\d.]+).*?draft ([\d.]+)",
                        log)
    ops = _ops(c)
    by, total, names = collections.Counter(), 0.0, collections.Counter()
    for start, end, _, _, name in ops:
        b = next((k for k, rx in STRATA_BUCKETS if re.search(rx, name)), "other")
        by[b] += (end - start) / 1e6
        names[(b, name)] += (end - start) / 1e6
        total += (end - start) / 1e6
    return {"timing": [dict(zip(("windows", "ms_window", "verify", "draft"), map(float, t), strict=True))
                       for t in timing],
            "gpu_ms_total": round(total, 3), "buckets_ms_total": {k: round(v, 3) for k, v in by.most_common()},
            "kernels_ms_total": {f"{b}:{n}": round(v, 3) for (b, n), v in names.most_common(60)}}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--flags", required=True)
    r.add_argument("--requests", required=True)
    r.add_argument("--key", required=True)
    r.add_argument("--steps", type=int, default=24)
    r.add_argument("--out", required=True)
    h = sub.add_parser("holon")
    h.add_argument("--rep", required=True)
    h.add_argument("--run", required=True)
    h.add_argument("--out", required=True)
    s = sub.add_parser("strata")
    s.add_argument("--rep", required=True)
    s.add_argument("--log", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {"run": run, "holon": holon, "strata": strata}[a.cmd](a)
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
