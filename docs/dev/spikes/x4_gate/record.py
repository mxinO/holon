"""Milestone X step 4's record (evidence/x4_gate_<gpu>_<date>.json): the natural fit, the offload lock and its
verdicts, the tuning rounds (`moe.ring_copy`, the profile warm start, `lfu` against `lru`), the gate's interleaved
rounds with every acceptance-3 row's ratio against its bar and §13's predicted band, the capped rows (recorded, judged
after step 6), and the exactness snapshots. Prints the README's tables.

    python spikes/x4_gate/record.py --runs <dir: tune/ gate_nat/ gate_cap/ step8k/ exact/ fit.json tune_lock.log> \\
        --lock <the lock> --date <date> --commit <measured commit> --out evidence/x4_gate_rtx-5080_<date>.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONTEXTS = ("1024", "4096", "32768")
MODEL = "unsloth/Qwen3-30B-A3B-GGUF@d5b1d57bd0b5/Qwen3-30B-A3B-Q4_K_M.gguf"
ENGINES = {"llama.cpp": "f74e2fa6a31c (llama-server, --moe-cache-mib)", "FreeToken": "21d9578bde22"}

# §13's predicted ratios of Holon F5a + lfu over each reference at 47% / 25% (spikes/x4_gap/README_32k.md: written
# before step 3c), as (low, central, high) per context; a single value where §13 gives no band. Holon's predicted
# rate (central, tok/s) converts them to after steps 3c and 3d, which took ~0.25 ms off every decode token (§14).
BANDS = {
    "natural": {
        "ft": [(0.95, 0.97, 1.00), (0.92, 0.95, 0.98), (0.97, 0.99, 1.01)],
        "cache_pinned": [(1.07, 1.10, 1.13), (1.09, 1.13, 1.17), (1.06, 1.08, 1.11)],
        "cache": [(1.30,) * 3, (1.52,) * 3, (1.32, 1.35, 1.38)],
        "fit": [(2.65,) * 3, (2.18,) * 3, (1.76, 1.80, 1.84)],
        "ncmoe": [(2.35,) * 3, (2.00,) * 3, (1.61, 1.64, 1.68)],
        "rate": (207.4, 166.5, 110.5)},
    "capped": {
        "ft": [(0.99, 1.05, 1.11), (0.89, 0.94, 1.00), (0.97, 1.02, 1.06)],
        "cache_pinned": [(1.15, 1.21, 1.28), (1.12, 1.19, 1.26), (1.06, 1.10, 1.15)],
        "cache": [(2.22,) * 3, (2.20,) * 3, (1.75, 1.83, 1.91)],
        "fit": [(2.61,) * 3, (2.26,) * 3, (1.86, 1.95, 2.04)],
        "ncmoe": [(2.56,) * 3, (2.27,) * 3, (1.83, 1.91, 2.00)],
        "rate": (121.4, 102.9, 77.7)},
}
SAVED_MS = 0.25                    # steps 3c (0.21 ms) and 3d (0.04 ms) per decode token, after the bands were written
# Acceptance 3's rows: (row, reference kind, decode bar, prefill bar)
ROWS = [("FreeToken (auto)", "ft", 1.0, 1.0), ("llama.cpp static split, --fit", "fit", 1.5, 1.0),
        ("llama.cpp static split, tuned --n-cpu-moe", "ncmoe", 1.5, 1.0),
        ("llama.cpp --moe-cache-mib, default (mmap)", "cache", 1.1, 1.0),
        ("llama.cpp --moe-cache-mib, pinned (--load-mode none)", "cache_pinned", 1.1, 1.0)]
ARMS = {  # fit -> holon arm, reference kind -> arm
    "natural": ("holon_nat", {"ft": "ft_nat", "fit": "llama_fit", "ncmoe": "llama_ncmoe16", "cache": "cache_nat",
                              "cache_pinned": "cache_nat_pinned"}),
    "47%": ("holon_47", {"ft": "ft_47", "fit": "llama_fit", "ncmoe": "llama_ncmoe16", "cache": "cache_47",
                         "cache_pinned": "cache_47_pinned"}),
    "capped": ("holon_25", {"ft": "cap_ft", "fit": "cap_fit", "ncmoe": "cap_ncmoe36", "cache": "cap_cache",
                            "cache_pinned": "cap_cache_pinned"}),
}


def scrub(x):
    """No absolute paths in a published record: a path becomes its last component."""
    if isinstance(x, str):
        return re.sub(r"/(?:home|scratch|tmp|root)[^\s'\",)]*/([^\s'\",)/]+)", r"\1", x)
    if isinstance(x, list):
        return [scrub(v) for v in x]
    if isinstance(x, dict):
        return {k: scrub(v) for k, v in x.items()}
    return x


def rows(d: pathlib.Path) -> list[dict]:
    return [json.loads(x) for x in (d / "raw.jsonl").read_text().splitlines()]


def stats(rs: list[dict]) -> dict:
    """arm -> context -> metric -> {median, mean, sd, min, max, n} over the error-free rounds."""
    out: dict = {}
    for r in rs:
        for c, m in r.get("results", {}).items():
            for k, v in m.items():
                out.setdefault(r["arm"], {}).setdefault(c, {}).setdefault(k, []).append(v)
    return {a: {c: {k: {"median": round(statistics.median(v), 4), "mean": round(statistics.fmean(v), 4),
                        "sd": round(statistics.stdev(v), 4) if len(v) > 1 else 0.0, "min": min(v), "max": max(v),
                        "n": len(v)} for k, v in m.items()} for c, m in cs.items()} for a, cs in out.items()}


def band(fit: str, kind: str, i: int) -> dict | None:
    if fit == "47%":
        fit = "natural"
    b = BANDS.get(fit, {}).get(kind)
    if b is None:
        return None
    t = 1000 / BANDS[fit]["rate"][i]
    shift = t / (t - SAVED_MS)
    lo, c, hi = b[i]
    return {"written": [lo, c, hi], "after_3c_3d": [round(lo * shift, 3), round(c * shift, 3), round(hi * shift, 3)]}


def gate(st: dict, fit: str) -> list[dict]:
    holon, refs = ARMS[fit]
    out = []
    for row, kind, dbar, pbar in ROWS:
        ref = refs[kind]
        if ref not in st or holon not in st:
            continue
        for i, c in enumerate(CONTEXTS):
            h, r = st[holon].get(c), st[ref].get(c)
            if h is None or r is None:
                continue
            for metric, bar in (("decode_tps", dbar), ("prefill_tps", pbar)):
                ratio = h[metric]["median"] / r[metric]["median"]
                b = band(fit, kind, i) if metric == "decode_tps" else None
                lo = h[metric]["min"] / r[metric]["max"]
                hi = h[metric]["max"] / r[metric]["min"]
                rec = {"row": row, "reference_arm": ref, "context": int(c), "metric": metric,
                       "holon": h[metric]["median"], "reference": r[metric]["median"], "ratio": round(ratio, 3),
                       "ratio_range": [round(lo, 3), round(hi, 3)], "bar": bar,
                       "verdict": "pass" if ratio >= bar else "miss", "predicted": b}
                if b is not None:
                    blo, _, bhi = b["after_3c_3d"]
                    rec["vs_band"] = "below" if ratio < blo else "above" if ratio > bhi else "inside"
                out.append(rec)
    return out


def table(g: list[dict], metric: str) -> str:
    lines = ["| Row (bar) | 1K | 4K | 32K |", "|---|---|---|---|"]
    for row, _, dbar, pbar in ROWS:
        cells = []
        for c in CONTEXTS:
            r = next((x for x in g if x["row"] == row and x["metric"] == metric and str(x["context"]) == c), None)
            if r is None:
                cells.append("—")
                continue
            cell = f"{r['ratio']:.2f} {'pass' if r['verdict'] == 'pass' else '**miss**'}"
            if r["predicted"]:
                lo, cen, hi = r["predicted"]["after_3c_3d"]
                cell += f" [{lo:.2f}–{hi:.2f}]" if lo != hi else f" [{cen:.2f}]"
            cells.append(cell)
        lines.append(f"| {row} ({dbar if metric == 'decode_tps' else pbar}×) | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def abba(st: dict, a: str, b: str, metric: str) -> dict:
    out = {}
    for c in CONTEXTS:
        x, y = st[a][c][metric], st[b][c][metric]
        sd = max(x["sd"], y["sd"])
        out[c] = {a: x["median"], b: y["median"], "gain": round(y["median"] / x["median"] - 1, 4), "sd": sd,
                  "beyond_2sd": abs(y["median"] - x["median"]) > 2 * sd, "n": [x["n"], y["n"]]}
    return out


def exactness(d: pathlib.Path) -> dict:
    import sys
    sys.path.insert(0, str(ROOT / "judge"))
    from holon_judge.snapshot import Snapshot, diff
    out = {}
    lock = d / "lock_snapshot.json"
    if lock.exists():
        rep = json.loads(lock.read_text())
        out["lock_snapshot"] = {"runs": rep["runs"], "run_identical": rep["run_identical"], "steps": rep["steps"],
                                "diffs": rep["diffs"]}
    snaps = sorted((d / "x3").glob("*_[0-9].json"))
    if snaps:
        first = Snapshot.load(snaps[0])
        metas = {p.stem: json.loads(p.with_suffix(".meta.json").read_text()) for p in snaps}
        out["placements"] = {
            p.stem: {"identical_to": snaps[0].stem if not diff(first, Snapshot.load(p)) else None,
                     "diff": diff(first, Snapshot.load(p)) or None,
                     **{k: metas[p.stem].get(k) for k in ("cache", "policy", "admissions", "swap_rounds",
                                                         "cached_per_class", "borrowed", "peak_misses")},
                     "parity_first_differing_step": (metas[p.stem].get("parity") or {}).get("first_differing_step")}
            for p in snaps}
        out["all_identical"] = all(v["diff"] is None for v in out["placements"].values())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=pathlib.Path, required=True)
    ap.add_argument("--lock", type=pathlib.Path, required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--commit", required=True, help="the measured commit")
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()
    tune = stats(rows(a.runs / "tune"))
    nat = stats(rows(a.runs / "gate_nat"))
    cap = stats(rows(a.runs / "gate_cap")) if (a.runs / "gate_cap" / "raw.jsonl").exists() else {}
    lock_log = (a.runs / "tune_lock.log").read_text(errors="replace")
    l2 = [x for x in lock_log.splitlines() if re.match(r"(overall|floor|vs floor|budget|chat|code|recall|agent|PASS|"
                                                       r"FAIL|verdict|attention\.)", x.strip())]
    record = {
        "name": "x4_gate", "date": a.date, "gpu": "NVIDIA GeForce RTX 5080", "model": MODEL, "commit": a.commit,
        "references": ENGINES,
        "method": ("spikes/x4_gap/ctx.py (bench/offload_ctx.py's method with a Holon worker): every arm in a fresh "
                   "process, an untimed 512 + 64 warm-up, one request per context of natural-text prompt ids "
                   "(1,024 / 4,096 / 32,768 tokens) generating 256 tokens, batch 1, greedy, ignore_eos; every engine "
                   "allocates a 33,792-token context; arms interleaved, reversed on odd rounds; medians"),
        "fit": json.loads((a.runs / "fit.json").read_text()),
        "lock": {"text": a.lock.read_text(), "verdict_log": l2},
        "tuning": {"summary": tune,
                   "ring_copy_prefill": abba(tune, "holon_sm", "holon_dma", "prefill_tps"),
                   "ring_copy_decode": abba(tune, "holon_sm", "holon_dma", "decode_tps"),
                   "profile_decode": abba(tune, "holon_sm", "holon_sm_prof", "decode_tps"),
                   "lru_over_lfu_decode": abba(tune, "holon_sm", "holon_lru", "decode_tps")},
        "gate": {"natural": {"summary": nat, "rows": gate(nat, "natural")},
                 "47%": {"rows": gate(nat, "47%")},
                 "capped": {"summary": cap, "rows": gate(cap, "capped")} if cap else None},
        "step8192_probe": abba(stats(rows(a.runs / "step8k")), "holon_nat", "holon_s8k", "prefill_tps")
        | {"decode": abba(stats(rows(a.runs / "step8k")), "holon_nat", "holon_s8k", "decode_tps")},
        "exactness": exactness(a.runs / "exact"),
        "followups": json.loads((a.runs / "followups.json").read_text()) if (a.runs / "followups.json").exists()
        else None,
        "arms": {n: json.loads((HERE / f"arms_{n}.json").read_text())
                 for n in ("tune", "natural", "capped", "step8k")},
    }
    record = scrub(record)
    if a.out:
        a.out.write_text(json.dumps(record, indent=1) + "\n")
    for fit in ("natural", "47%", "capped"):
        g = (record["gate"].get(fit) or {}).get("rows")
        if g:
            print(f"\n## {fit}: decode\n\n{table(g, 'decode_tps')}\n\n## {fit}: prefill\n\n{table(g, 'prefill_tps')}")


if __name__ == "__main__":
    main()
