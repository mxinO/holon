"""Markdown tables from bench.py's evidence JSONs (the README's results).

    python report.py ../../../evidence/x0_bench_rtx-5080_<date>.json ../../../evidence/x0_bench_rtx-5090_<date>.json
"""

import json
import sys
from pathlib import Path

EXPERTS = 128


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def f(x, nd=1):
    return "—" if x is None else f"{x:.{nd}f}"


def m1(d):
    m = d["m1"]
    copy = {r["mib"]: r["gbps"] for r in m["copy"]}
    print("pinned cudaMemcpyAsync, GB/s: " + ", ".join(f"{k} MiB {v:.1f}" for k, v in copy.items()))
    rows = []
    for label, b in m["blobs"].items():
        for r in b["rows"]:
            if r["n"] in (1, 8, 64):
                rows.append([label, r["n"], f(r["memcpy_each_gbps"]), f(r["memcpy_batch_gbps"]),
                             f(r["gather_best_gbps"]), r["gather_best"], r["gather_saturating_ctas"],
                             f(r["gather_over_dma"], 2)])
        rows.append([label, "8, in a graph", "", "", f(b["graph_gather_n8_gbps"]), b["graph_gather_config"], "", ""])
    print(table(["blob", "N", "memcpy ×N", "BatchAsync", "gather best", "config (CTAs×threads, unroll)",
                 "CTAs ≥ 95%", "gather / best DMA"], rows))


def m3(d, key):
    m = d[key]
    rows = []
    for name, r in m.items():
        tab = {(x["threads"], x["tokens"]): x["us_per_expert"] for x in r["rows"] if x["jobs"] == 1}
        j8 = {x["threads"]: x["us_per_expert"] for x in r["rows"] if x["jobs"] == 8}
        ts = sorted({t for t, _ in tab})
        best = min(ts, key=lambda t: tab[t, 1])
        rows.append([name, f"{r['expert_bytes'] / 1e6:.2f}"] + [f(tab[t, 1]) for t in ts]
                    + [f(j8[best]), f"{r['expert_bytes'] / tab[best, 1] / 1e3:.0f}",
                       " / ".join(f(tab[ts[-1], k] / k) for k in (2, 4, 8)),
                       f"{r['bitwise']['mismatches']} of {r['bitwise']['checks']}"])
    print(table(["expert (gate·up + down)", "MB"] + [f"{t} thr" for t in ts]
                + ["8 per batch", "GB/s best", f"µs/token at 2/4/8 tokens, {ts[-1]} thr", "bitwise mismatches"],
                rows))


def m2(d):
    m = d["m2"]
    rows = [[r["threads"], r["mode"], f(r["cpu_us_per_expert"]), f(r["gpu_gbps"])] for r in m["steady"]]
    print(f"expert {m['expert_bytes'] / 1e6:.2f} MB, gather {m['gather_config']}")
    print(table(["CPU threads", "GPU traffic", "CPU µs/expert", "GPU GB/s"], rows))
    for nt, s in m["split"].items():
        print(f"\nsplit at {nt} threads:")
        rows = [[n, p["k"], f(p["fetch_only_us"]), f(p["path_us"]), f"{100 * p['gain']:.0f}%"]
                for n, p in s["plan"].items()]
        print(table(["misses n", "plan[n] (CPU)", "fetch-only µs", "split µs", "gain"], rows))


def m5(d):
    m = d["m5"]
    rows = []
    for r in m["rows"]:
        rows.append([r["rows"], r["k"], f(r["device"]["cpu_us_per_layer"]), f(r["device"]["overhead_us_per_layer"]),
                     f(r["host"]["overhead_us_per_layer"]), f(r["device_work"]["exposed_us_per_layer"]),
                     f(r["host_work"]["exposed_us_per_layer"]),
                     "yes" if r["device"]["exact"] and r["host"]["exact"] else "NO"])
    print(f"{m['threads']} compute threads; stranded at stop: {m['stranded']}")
    print(table(["rows", "jobs k", "CPU µs/layer", "overhead device-planned", "overhead host-planned",
                 "exposed with GPU work, device", "exposed, host", "bitwise vs ggml-cpu"], rows))


def m7(d):
    m = d["m7"]
    if "skipped" in m:
        print("M7 skipped:", m["skipped"])
        return
    rows = []
    for blob in ("1.46MB", "2.8MB", "7.1MB"):
        rs = [r for r in m["rows"] if r["blob"] == blob]
        qd1 = next(r for r in rs if r["mode"] == "aio" and r["qd"] == 1)
        best = max(rs, key=lambda r: r["gbps"])
        pr = {r["workers"]: r["gbps"] for r in rs if r["mode"] == "pread"}
        rows.append([blob, f(qd1["gbps"], 2), f(qd1["lat_p50_us"], 0), f(best["gbps"], 2),
                     f"{best['mode']} w={best['workers']} qd={best['qd']}",
                     " / ".join(f(pr[w], 2) for w in sorted(pr))])
    print(f"test file {m['file_bytes'] >> 30} GiB, written at {m['fill_gbps']:.1f} GB/s, removed: "
          f"{m['test_file_removed']}")
    print(table(["read", "QD 1 GB/s", "QD 1 p50 µs", "best GB/s", "at", "pread 1/2/4/8 workers GB/s"], rows))


def prefill(d):
    rate = d["m1"]["blobs"]["2.8MB"]["rows"][3]["gather_best_gbps"] if "m1" in d else 48.0
    rows = []
    for label, p in d["prefill"].items():
        for r in p["rows"]:
            stream = {fit: EXPERTS * (1 - fit) * p["expert_bytes"] / rate / 1e6 for fit in (0.47, 0.25)}
            rows.append([label, r["tokens"], f(r["ms"], 3), f(stream[0.47], 2), f(stream[0.25], 2)])
    print(f"streaming time = non-resident experts' bytes / {rate:.1f} GB/s (M1 gather)")
    print(table(["layer", "chunk tokens", "expert GEMMs ms", "stream at 47% resident ms",
                 "stream at 25% resident ms"], rows))


def main():
    for path in sys.argv[1:]:
        d = json.loads(Path(path).read_text())
        print(f"\n## {d['tag']} ({d['env']['gpu']}, CPU {d['env']['isa_tier']})\n")
        for name, fn in (("m1", m1), ("m3_native", lambda d: m3(d, "m3_native")),
                         ("m3_avx2", lambda d: m3(d, "m3_avx2")), ("m2", m2), ("m5", m5), ("m7", m7),
                         ("prefill", prefill)):
            if name in d:
                print(f"\n### {name}\n")
                fn(d)


if __name__ == "__main__":
    main()
