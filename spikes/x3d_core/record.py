"""Step 3d's evidence record for one GPU class: the per-step kernel inventory before and after (`inventory.py` on
`profile.py`'s reports), per change its saving on each row (ab.py's ABBA rounds), its class and M0.7's verdict (kept
where the saving is ≥ max(0.5%, 3 sd) on some decode row), the end-to-end holon_tg rounds (base against final, one
process per run, ABBA) and the snapshot facts.

    python spikes/x3d_core/record.py --gpu rtx-a5000 --date 2026-10-17 --commit <measured> --out <record.json> \\
        [--ab resident=<ab.json> ...] [--tg resident=<tg dir> ...] [--inventory resident=<before.json>,<after.json> ...]
        [--facts <facts .json>]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics

MODELS = {"qwen3-30b-a3b": "unsloth/Qwen3-30B-A3B-GGUF@d5b1d57bd0b5/Qwen3-30B-A3B-Q4_K_M.gguf",
          "qwen3-8b": "unsloth/Qwen3-8B-GGUF@a6adef130ffb/Qwen3-8B-Q4_K_M.gguf"}
CLASS = {"prep": "bitwise", "upload": "bitwise", "telemetry": "bitwise"}
WHAT = {"prep": "without a drafter the host stages positions, page lengths, write indices and the store's slots and "
                "lengths; the device moves only its last tokens and writes the SeqTable; no logits gather when every "
                "token is a logits row",
        "upload": "steps of at most a decode / verify step's tokens stage into an arena of that size, fields smallest "
                  "first: one host-to-device copy",
        "telemetry": "the shared objects' telemetry gathered as bytes on the device and copied to the host once"}
CORE = ("upload", "prepare", "store", "logits", "launch", "observe", "writes")   # phases whose own work is the core's


def pairs(items: list[str]) -> dict[str, str]:
    return dict(i.split("=", 1) for i in items)


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
        out[k] = {"tok_s": {"base": b, "final": f},
                  "median_tok_s": {"base": statistics.median(b), "final": statistics.median(f)},
                  "median_ms_per_token": {"base": round(1e3 / statistics.median(b), 4),
                                          "final": round(1e3 / statistics.median(f), 4)},
                  "speedup": round(statistics.median(f) / statistics.median(b), 4),
                  "ms_per_token_saved": round(statistics.mean(ms), 4),
                  "sd_ms": round(statistics.stdev(ms), 4) if len(ms) > 1 else None}
    return out


def inventory(p: str) -> dict:
    d = json.loads(pathlib.Path(p).read_text())
    return {"step_gpu_us_profiled": d["step_gpu_us"],
            "phases": {k: v for k, v in d["phases"].items() if k != "trunk"},
            "rows": [{k: r[k] for k in ("phase", "name", "us", "gap_us")} for r in d["rows"]
                     if r["phase"] != "trunk" and not (r["phase"] == "logits" and r["us"] > 50)]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ab", nargs="*", default=[])
    ap.add_argument("--tg", nargs="*", default=[])
    ap.add_argument("--inventory", nargs="*", default=[])
    ap.add_argument("--facts")
    a = ap.parse_args()
    ab = {k: json.loads(pathlib.Path(p).read_text()) for k, p in pairs(a.ab).items()}
    changes = {}
    for change in CLASS:
        rows = {f"{deploy}/{row}": s for deploy, r in ab.items() for row, s in r["savings"].get(change, {}).items()}
        if rows:
            changes[change] = {"what": WHAT[change], "class": CLASS[change], "rows": rows, "verdict": verdict(rows)}
    record = {
        "gpu": a.gpu, "date": a.date, "models": MODELS, "commit": a.commit,
        "method": {
            "inventory": "spikes/x3d_core/profile.py under nsys (--cuda-graph-trace=node), batch-1 decode after a "
                         "1,024-token prompt: eager steps with NVTX ranges per step-loop phase name each kernel; a "
                         "graphed step's kernels take the eager step's phases by position (inventory.py). Durations "
                         "are profiled (node tracing inflates small kernels); `gap_us` is the idle time before each",
            "ab": "spikes/x3d_core/ab.py: one engine per deployment, the changes switched in process (each arm's "
                  "graphs captured again; base = bdd8422's executor methods and uploader, spikes/x3d_core/base.py), "
                  "graphed decode at batch 1 by per-step wall after a prompt of 1,024 and 4,096 tokens and eager "
                  "prefill of one prompt; a warm-up round, then ABBA rounds; a change's saving against the previous "
                  "arm, its sd that of the per-round differences",
            "end_to_end": "bench/holon_tg.py -n 128 -r 3 --depths 1024 4096 --engine-args '--max-seqs 1 …', one "
                          "process per run, base bdd8422 against the final commit, ABBA rounds"},
        "inventory": {k: {"before": inventory(v.split(",")[0]), "after": inventory(v.split(",")[1])}
                      for k, v in pairs(a.inventory).items()},
        "ab": {k: {"model": r["model"], "engine_args": r["engine_args"], "rounds": r["rounds"], "steps": r["steps"],
                   "times_ms": r["times_ms"]} for k, r in ab.items()},
        "changes": changes,
        "end_to_end": {k: tg(pathlib.Path(p)) for k, p in pairs(a.tg).items()},
    }
    if a.facts:
        record |= json.loads(pathlib.Path(a.facts).read_text())
    pathlib.Path(a.out).write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
