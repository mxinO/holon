"""Assemble milestone X's reference record (evidence/x4_refs_<gpu>_<date>.json) from the runs in bench/.out/: the
interleaved rounds (`bench/offload_ctx.py run --name x4_refs`), the n-cpu-moe and --fit sweeps, the nsys analyses of
llama.cpp's MoE cache (`nsys_moe_gap.py`), and the recomputed roofline band of acceptance 3's natural-fit
`--moe-cache-mib` row (docs/dev/v1-offload.md §13).

    python spikes/x4_refs/record.py --out evidence/x4_refs_rtx-5080_<date>.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import re
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench" / ".out"
N_LAYERS = 48

# The nsys runs (bench/offload_ctx.py worker, 1K context, the natural-fit cache arm) and their decode rate under the
# profiler, to set against the unprofiled 1K rate of the same arm in the rounds.
NSYS = {
    "graph_pinned": {"flags": "-cmoe --moe-cache-mib 7900 --load-mode none -lv 4", "trace": "--cuda-graph-trace=graph",
                     "decode_tps_profiled": 175.7},
    "graph_mmap": {"flags": "-cmoe --moe-cache-mib 7900 --load-mode mmap -lv 4", "trace": "--cuda-graph-trace=graph",
                   "decode_tps_profiled": 148.2},
    "node_pinned": {"flags": "-cmoe --moe-cache-mib 7900 --load-mode none -lv 4", "trace": "--cuda-graph-trace=node",
                    "decode_tps_profiled": 156.9},
    "node_mmap": {"flags": "-cmoe --moe-cache-mib 7900 --load-mode mmap -lv 4", "trace": "--cuda-graph-trace=node",
                  "decode_tps_profiled": 133.0},
}
NSYS_CMD = ("python bench/offload_ctx.py worker --engine llama --server bench/.envs/llama.cpp-offload/bin/llama-server "
            "--model <file.gguf> --flags '<flags>' --contexts 1024 --wrap 'nsys profile -t cuda <trace> --sample=none "
            "--cpuctxsw=none -o <report>'; nsys export --type sqlite; "
            "python spikes/x4_refs/nsys_moe_gap.py <report>.sqlite")

# FreeToken with no sizing flag (--moe-cache-auto, one run after the rounds): the cache it picks and its KV floor.
FT_DEFAULT = {"flags": "", "moe_cache_size": 4464, "expert_share": round(4464 / 6144, 3), "kv_tokens": 8206,
              "results": {"1024": {"prefill_tps": 2730.3, "decode_tps": 226.2},
                          "4096": {"prefill_tps": 10737.5, "decode_tps": 207.3}},
              "note": "its 8,206-token KV cannot hold the 32K rows, so the rounds size the cache with --moe-cache-rate"}

# Roofline parameters of §13 (fixed before step 4) and the ranges the band spans.
HOLON_FETCH_GBPS = 47.5                 # M1: the in-graph gather
HOLON_OVERHEAD_MS = (0.45, 0.55)        # step 2's measured 0.20 ms + F11's pick and decay 0.25-0.35 ms
HOLON_KERNEL_SPEED = (0.93, 1.0)        # Holon's all-resident speed over llama.cpp's on sm_120 (release), to parity
HOLON_BYTES_SHARE = (0.77, 1.0)         # lfu's miss bytes over lru's (M6: lru 10-30% more), to equal


def scrub(x):
    """No absolute paths or terminal colours in a published record: a path becomes its last component."""
    if isinstance(x, str):
        x = re.sub(r"\x1b\[[0-9;]*m", "", x)                         # terminal colours
        return re.sub(r"/(?:home|scratch|tmp|root)[^\s'\",)]*/([^\s'\",)/]+)", r"\1", x)
    if isinstance(x, list):
        return [scrub(v) for v in x]
    if isinstance(x, dict):
        return {k: scrub(v) for k, v in x.items()}
    return x


def cache_stats(rows: list[dict]) -> dict:
    """llama.cpp's MoE cache counters per arm (its exit log: warm-up plus the three requests), medians over rounds."""
    out = {}
    for r in rows:
        for line in r.get("log", []):
            if m := re.search(r"ubatch <= 8: hits = (\d+), misses = (\d+), hit rate = ([\d.]+)%, "
                              r"uploaded = ([\d.]+) MiB", line):
                out.setdefault(r["arm"], []).append({"hits": int(m[1]), "misses": int(m[2]), "hit_rate": float(m[3]),
                                                     "uploaded_mib": float(m[4])})
    return {arm: {k: statistics.median(x[k] for x in xs) for k in xs[0]} | {"rounds": len(xs)}
            for arm, xs in out.items()}


def band(decode_tps: float, ref_tps: float, host_ms: float, bytes_mb: tuple[float, float], upload_gbps: float) -> dict:
    """Holon F5a over a llama.cpp cache arm at one context. llama.cpp's GPU work G comes from the pinned arm's measured
    rate (`ref_tps` = G + host + B / r_l); Holon = G / k + overhead + s B / r_h; the ratio = the arm's measured time /
    Holon's, over the parameter box."""
    t_arm, t_ref = 1e3 / decode_tps, 1e3 / ref_tps

    def ratio(b: float, k: float, o: float, s: float) -> float:
        g = t_ref - host_ms - b / upload_gbps
        return t_arm / (g / k + o + s * b / HOLON_FETCH_GBPS)

    box = list(itertools.product(bytes_mb, HOLON_KERNEL_SPEED, HOLON_OVERHEAD_MS, HOLON_BYTES_SHARE))
    at_093 = [ratio(*p) for p in box if p[1] == HOLON_KERNEL_SPEED[0]]
    central = ratio(statistics.mean(bytes_mb), HOLON_KERNEL_SPEED[0], statistics.mean(HOLON_OVERHEAD_MS),
                    statistics.mean(HOLON_BYTES_SHARE))
    return {"ref_ms": round(t_arm, 3), "band": [round(min(map(lambda p: ratio(*p), box)), 3),
                                                  round(max(map(lambda p: ratio(*p), box)), 3)],
            "band_at_k_0.93": [round(min(at_093), 3), round(max(at_093), 3)], "central": round(central, 3)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main_run = json.loads((OUT / "x4_refs" / "record.json").read_text())
    sweeps = {name: json.loads((OUT / name / "record.json").read_text())
              for name in ("x4_sweep", "x4_sweep2", "x4_sweep3")}
    nsys = {k: v | {"analysis": json.loads((OUT / "x4_nsys" / f"{k}.json").read_text())} for k, v in NSYS.items()}

    g = nsys["graph_pinned"]["analysis"]
    host_layer_us = g["per_layer_all"]["host_us"]["mean"]
    host_ms = host_layer_us * N_LAYERS / 1e3
    with_miss = g["per_layer_with_miss"]
    upload_gbps = with_miss["upload_mb"]["median"] / with_miss["upload_us"]["median"] * 1e3   # MB/us -> GB/s
    stats = cache_stats(main_run["rows"])
    nat = stats["nat_cache_pinned"]
    expert_mb = 17553.16224 / (N_LAYERS * 128)       # the file's expert bytes per expert (gate + up + down)
    fill = 7900 * 2**20 / (expert_mb * 1e6)          # slots filled once from empty
    tokens = 64 + 3 * 256                            # decode tokens of one arm-round (warm-up + three requests)
    run_bytes = (nat["misses"] - fill) / tokens * expert_mb
    bytes_mb = (round(g["per_token"]["upload_mb"]["mean"], 1), round(run_bytes, 1))
    summary = main_run["summary"]
    tps = {arm: {c: summary[arm][c]["decode_tps"]["median"] for c in ("1024", "4096", "32768")}
           for arm in ("nat_cache_pinned", "nat_cache", "nat_ft_offload")}
    bands = {arm: {c: band(t[c], tps["nat_cache_pinned"][c], host_ms, bytes_mb, upload_gbps) for c in t}
             for arm, t in tps.items()}
    record = main_run | {
        "pins": {"llama.cpp": "f74e2fa6a31cc33ab4ee4ae516641cfe62df3924 (2026-10-16; build 11706, CUDA sm_120a)",
                 "FreeToken": "21d9578bde22840798b6063824416264fbdcdc25 (2026-10-16; torch 2.14.1+cu130, [accel])",
                 "setup": "LLAMA_ARCHS=120a bash bench/setup_baselines.sh offload"},
        "command": main_run["command"].replace("main.json", "spikes/x4_refs/arms.json"),
        "sweeps": {k: {"command": v["command"], "arms": v["arms"], "summary": v["summary"],
                       "rows": [{x: r[x] for x in ("arm", "results", "error", "peak_vram_mib") if x in r}
                                for r in v["rows"]]} for k, v in sweeps.items()},
        "moe_cache_stats": stats,
        "nsys": {"command": NSYS_CMD, "runs": nsys,
                 "unprofiled_1k_decode_tps": summary["nat_cache_pinned"]["1024"]["decode_tps"]["median"]},
        "roofline_band": {
            "row": "natural fit, batch 1: Holon F5a vs llama.cpp --moe-cache-mib (bar 1.1x) and FreeToken (1.0x)",
            "model": "llama = G + H + B / r_l, with G from the pinned cache arm's measured decode rate; "
                     "Holon = G / k + O + s B / r_h; ratio = the arm's measured time per token / Holon's",
            "H_ms_per_token": round(host_ms, 3), "H_us_per_layer": round(host_layer_us, 1),
            "r_l_gbps": round(upload_gbps, 1), "r_h_gbps": HOLON_FETCH_GBPS, "B_mb_per_token": bytes_mb,
            "k": HOLON_KERNEL_SPEED, "O_ms": HOLON_OVERHEAD_MS, "s": HOLON_BYTES_SHARE,
            "vs": bands,
            "freetoken_default_sizing": FT_DEFAULT},
    }
    pathlib.Path(a.out).write_text(json.dumps(scrub(record), indent=1) + "\n")
    print(json.dumps(record["roofline_band"], indent=1))


if __name__ == "__main__":
    main()
