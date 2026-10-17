"""Per-token GPU time by category from an nsys SQLite export (`nsys export --type sqlite`), for x4_gap's engines.

Windows:
- `nvtx` (Holon, FreeToken; `ctx.py --nsys-context`): decode = the NVTX range "decode" (first token to last: `--tokens`
  steps, 255 for 256 generated tokens); prefill = from the capture's first kernel to that range's start.
- `logits:N` (llama-server, unmarked): the last N decode steps, delimited by the logits' device-to-host copies (the
  largest D2H copies, one per step).

Every kernel and memcpy starting in a window is counted in one category (first matching rule); `idle` is the window
less the union of busy intervals (any stream), so overlapped copies are not double-counted in it.

    python spikes/x4_gap/nsys_breakdown.py <report.sqlite> [--window nvtx|logits:200] [--tokens 255] [--top 25]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sqlite3

VOCAB_ROWS = 151936                           # the vehicle's lm_head rows (a grid of one row per block)

# (category, rule on the short name or the demangled name), first match wins; mmvq/mmq are split by grid below
RULES = [
    ("fetch (expert gather / copy)", r"k_fetch|fast_index_copy|copy_missing|batch_copy|gather_experts|expert_copy"),
    ("resolve / cache bookkeeping", r"k_resolve|ensure_experts|_lru_|slot_for|_prefill_hit|hit_compact|remap"),
    ("router (gate GEMV + top-k)", r"topk_moe|gemvx|gemv|cublasDot|cublasGemv|reduce_1Block|moe_topk|_router|argsort"),
    ("MoE combine", r"moe_weighted|moe_sum|weighted_sum"),
    ("embedding / glue", r"get_rows"),
    ("attention (incl. q/k norm, RoPE, KV write)",
     r"paged_attn|merge_splits|MergeStates|qk_rope|QKRMSNorm|flash_attn|BatchDecode|BatchPrefill|Rotary|rope|"
     r"kv_write|store_kv|set_rows|attn|fattn"),
    ("norms", r"rms_?norm|RMSNorm|layer_norm|l2_norm"),
    ("activation quant (q8_1)", r"quantize_q8_1|quantize_mmq|quantize_"),
    ("sampling / logits", r"ArgMax|argmax|sampl|top_k_top_p|reduce_kernel"),
    ("activation (silu / glu)", r"silu|swiglu|glu|act_and_mul"),
]


def category(short: str, demangled: str, grid: tuple[int, int, int]) -> str:
    if short == "mul_mat_vec_f":                 # the vehicle's only f32 matrices are the routers (llama.cpp)
        return "router (gate GEMV + top-k)"
    if short.startswith("mul_mat_vec"):
        if grid[0] == VOCAB_ROWS:
            return "lm_head GEMV"
        return "MoE expert GEMV (mmvq, ids)" if grid[1] > 1 else "dense linear GEMV (mmvq)"
    if short.startswith("mul_mat_q") or "mul_mat_q<" in demangled:
        return "mmq (prefill GEMM)"
    if re.search(r"gemm|cutlass.*gemm|sm\d+_xmma|ampere_|Kernel2|nvjet", short + demangled, re.I) and \
            not re.search(r"norm", short, re.I):
        return "cuBLAS / CUTLASS GEMM"
    for name, rule in RULES:
        if re.search(rule, short) or re.search(rule, demangled[:300]):
            return name
    return "embedding / glue"


def union(intervals: list[tuple[int, int]]) -> int:
    busy, end = 0, None
    for a, b in sorted(intervals):
        if end is None or a > end:
            busy, end = busy + (b - a), b
        elif b > end:
            busy, end = busy + (b - end), b
    return busy


def windows(db: sqlite3.Connection, spec: str, strings: dict) -> dict[str, tuple[int, int]]:
    if spec == "nvtx":
        rows = list(db.execute("select start, end, text, textId from NVTX_EVENTS where end is not null"))
        decode = [(a, b) for a, b, t, ti in rows if (t or strings.get(ti)) == "decode"]
        if not decode:
            raise SystemExit("no NVTX range 'decode'")
        start = db.execute("select min(start) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
        return {"decode": decode[-1], "prefill": (start, decode[-1][0])}
    n = int(spec.split(":")[1])
    d2h = list(db.execute("select start, end, bytes from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind = 2 "
                          "order by start"))
    big = max(b for *_, b in d2h)
    marks = [a for a, _, b in d2h if b >= big * 0.9]
    if len(marks) < n + 1:
        raise SystemExit(f"{len(marks)} logits copies, need {n + 1}")
    return {"decode": (marks[-n - 1], marks[-1])}


def breakdown(db: sqlite3.Connection, window: tuple[int, int], tokens: int, strings: dict, top: int) -> dict:
    t0, t1 = window
    kernels = list(db.execute("select start, end, shortName, demangledName, gridX, gridY, gridZ "
                              "from CUPTI_ACTIVITY_KIND_KERNEL where start >= ? and start < ?", (t0, t1)))
    copies = list(db.execute("select start, end, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY "
                             "where start >= ? and start < ?", (t0, t1)))
    cats, counts, names = collections.Counter(), collections.Counter(), collections.Counter()
    for a, b, s, d, gx, gy, gz in kernels:
        short, dem = strings[s], strings.get(d, "")
        c = category(short, dem, (gx, gy, gz))
        cats[c] += b - a
        counts[c] += 1
        gemv = "GEMV" in c
        names[(c, short[:80], gx if gemv else 0, gy if gemv else 0)] += b - a
    kinds = {1: "H2D", 2: "D2H", 8: "D2D", 10: "P2P"}
    for a, b, _, kind in copies:
        c = f"memcpy {kinds.get(kind, kind)}"
        cats[c] += b - a
        counts[c] += 1
    span = t1 - t0
    busy = union([(a, b) for a, b, *_ in kernels] + [(a, b) for a, b, *_ in copies])
    us = 1e-3 / tokens
    out = {"tokens": tokens, "window_us_per_token": round(span * us, 1), "busy_us_per_token": round(busy * us, 1),
           "idle_us_per_token": round((span - busy) * us, 1),
           "categories": {c: {"us_per_token": round(v * us, 1), "launches_per_token": round(counts[c] / tokens, 2)}
                          for c, v in cats.most_common()},
           "memcpy_bytes_per_token": {kinds.get(k, str(k)): round(sum(n for _, _, n, kk in copies if kk == k) / tokens)
                                      for k in {kk for *_, kk in copies}},
           "top_kernels": [{"category": c, "kernel": k, "grid": [gx, gy], "us_per_token": round(v * us, 1)}
                           for (c, k, gx, gy), v in names.most_common(top)]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sqlite")
    ap.add_argument("--window", default="nvtx")
    ap.add_argument("--tokens", type=int, default=255, help="decode steps in the NVTX window")
    ap.add_argument("--prompt", type=int, default=0, help="prompt tokens (nvtx: also break down the prefill)")
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    db = sqlite3.connect(a.sqlite)
    strings = dict(db.execute("select id, value from StringIds"))
    w = windows(db, a.window, strings)
    tokens = a.tokens if a.window == "nvtx" else int(a.window.split(":")[1])
    out = {"decode": breakdown(db, w["decode"], tokens, strings, a.top)}
    if a.prompt and "prefill" in w:
        out["prefill"] = breakdown(db, w["prefill"], a.prompt, strings, a.top)
        out["prefill"]["window_ms"] = round((w["prefill"][1] - w["prefill"][0]) / 1e6, 2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
