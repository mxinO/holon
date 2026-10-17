"""Step 3c's evidence record for one GPU class: per change its saving on each row (ab.py's ABBA rounds), its class and
M0.7's verdict (kept where the saving is ≥ max(0.5%, 3 sd) on some row), the end-to-end holon_tg rounds (base against
final, one process per run), the nsys breakdowns before and after, and the judge's results.

    python spikes/x3c_kernels/record.py --gpu rtx-a5000 --date 2026-10-17 --commit <measured> --out <record.json> \\
        --ab resident=<ab.json> ... --tg resident=<tg dir> ... --nsys resident/base=<breakdown.json> ... \\
        [--facts <judge results .json>] [--clock <clock_check .json>]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics

MODEL = "unsloth/Qwen3-30B-A3B-GGUF@d5b1d57bd0b5/Qwen3-30B-A3B-Q4_K_M.gguf"
CLASS = {"residual": "bitwise", "router": "bounded", "prepare": "bitwise"}
WHAT = {"residual": "the layer's residual add in the MoE combine's kernel (moe_combine@1's add)",
        "router": "moe_router@1: the router GEMV from bf16 and the softmax top-k in one launch (decode / verify)",
        "prepare": "the router's launch writes the gate/up input's q8_1 (route_prepare): no quantize launch"}
CATEGORIES = ("router (gate GEMV + top-k)", "embedding / glue", "activation quant (q8_1)", "MoE combine",
              "MoE expert GEMV (mmvq, ids)", "dense linear GEMV (mmvq)", "norms")


def pairs(items: list[str]) -> dict[str, pathlib.Path]:
    return {k: pathlib.Path(v) for k, v in (i.split("=", 1) for i in items)}


def verdict(rows: dict) -> dict:
    """M0.7 §1.1: kept if some decode row saves ≥ max(0.5%, 3 sd); the rows that pass."""
    passing = [r for r, s in rows.items() if "/decode" in r and s["sd_pct"] is not None
               and s["pct"] >= max(0.5, 3 * s["sd_pct"])]
    return {"passes_on": passing, "pays": bool(passing)}


def tg(d: pathlib.Path) -> dict:
    runs: dict = {}
    for f in sorted(d.glob("*_*.json")):
        arm, r = re.fullmatch(r"(base|final)_(\d+)\.json", f.name).groups()
        for k, v in json.loads(f.read_text())["results"].items():
            runs.setdefault(k, {}).setdefault(arm, {})[int(r)] = v["mean"]
    out = {}
    for k, arms in runs.items():
        rounds = sorted(set(arms["base"]) & set(arms["final"]))
        b, f = [arms["base"][r] for r in rounds], [arms["final"][r] for r in rounds]
        ms = [1e3 / x - 1e3 / y for x, y in zip(b, f, strict=True)]
        out[k] = {"tok_s": {"base": b, "final": f}, "median_tok_s": {"base": statistics.median(b),
                                                                    "final": statistics.median(f)},
                  "speedup": round(statistics.median(f) / statistics.median(b), 4),
                  "ms_per_token_saved": round(statistics.mean(ms), 4),
                  "sd_ms": round(statistics.stdev(ms), 4) if len(ms) > 1 else None}
    return out


def nsys(p: pathlib.Path) -> dict:
    d = json.loads(p.read_text())["decode"]
    return {"window_us_per_token": d["window_us_per_token"], "busy_us_per_token": d["busy_us_per_token"],
            "idle_us_per_token": d["idle_us_per_token"],
            "categories": {c: d["categories"][c] for c in CATEGORIES if c in d["categories"]}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ab", nargs="*", default=[])
    ap.add_argument("--tg", nargs="*", default=[])
    ap.add_argument("--nsys", nargs="*", default=[])
    ap.add_argument("--facts")
    ap.add_argument("--clock")
    a = ap.parse_args()
    ab = {k: json.loads(p.read_text()) for k, p in pairs(a.ab).items()}
    changes = {}
    for change in CLASS:
        rows = {f"{deploy}/{row}": s for deploy, r in ab.items() for row, s in r["savings"].get(change, {}).items()}
        changes[change] = {"what": WHAT[change], "class": CLASS[change], "rows": rows, "verdict": verdict(rows)}
    record = {
        "gpu": a.gpu, "date": a.date, "model": MODEL, "commit": a.commit,
        "method": "spikes/x3c_kernels/ab.py: one engine per deployment, the changes switched in process (each arm's "
                  "graphs captured again), graphed decode at batch 1 by per-step wall after a prompt of 1,024 and "
                  "4,096 tokens (and eager prefill of one prompt); a warm-up round, then ABBA rounds; a change's "
                  "saving against the previous arm, its sd that of the per-round differences",
        "ab": {k: {"engine_args": r["engine_args"], "rounds": r["rounds"], "steps": r["steps"],
                   "times_ms": r["times_ms"]} for k, r in ab.items()},
        "changes": changes,
        "end_to_end": {k: tg(p) for k, p in pairs(a.tg).items()},
        "nsys": {k: nsys(p) for k, p in pairs(a.nsys).items()},
    }
    if a.facts:
        record["judge"] = json.loads(pathlib.Path(a.facts).read_text())
    if a.clock:
        record["clock"] = json.loads(pathlib.Path(a.clock).read_text())
    pathlib.Path(a.out).write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
