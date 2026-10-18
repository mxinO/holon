"""moe_prefill's evidence record: the timed arms (`prefill.py run`'s raw.jsonl: medians over repetitions × rounds,
ratios against FreeToken's default and gated arms), the nsys breakdowns (`breakdown.py`'s JSON per arm), the per-call
GEMM benches (`gemm.py --json`) and the startup memory runs (`prefill.py run` over a `kv_*.json` arms file), with the
commands and engine pins. Every input is optional but the GPU, date and commit.

    python spikes/moe_prefill/record.py [--timed <dir>] [--nsys <dir>] [--gemm <json>...] [--kv <dir>...]
        --name moe_prefill --gpu "RTX A5000" --date 2026-10-18 --commit <sha> > evidence/<name>_<class>_<date>.json
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics

ENGINES = {"holon": "this repository at the record's commit (std.moe@1; the store and kernels per arm: arms_*.json)",
           "freetoken": "FreeToken 21d9578bde22 (bench/setup_baselines.sh offload: torch 2.14.1+cu130, [accel]); "
                        "--moe-strategy fused (all experts resident) or --moe-cache-rate (offload); the gated arm adds "
                        "--max-extend-length 4096, its default chunk is 8,192",
           "model": "Qwen3-30B-A3B Q4_K_M (unsloth/Qwen3-30B-A3B-GGUF); the 16 GB lock's: UD-IQ3_XXS"}
COMMANDS = [
    "python spikes/moe_prefill/prefill.py run --arms spikes/moe_prefill/arms_<card>.json --model <gguf> --out <dir>/timed "
    "--rounds 2 --reps 5 5 3 --holon-python <holon venv python> --freetoken-python <freetoken venv python>",
    "python spikes/moe_prefill/prefill.py run --arms spikes/moe_prefill/arms_<card>_nsys.json --model <gguf> "
    "--out <dir>/nsys --rounds 1 --reps 1 ...",
    "nsys export --type sqlite -o <report>.sqlite <report>.nsys-rep; "
    "python spikes/moe_prefill/breakdown.py <report>.sqlite > <report>.json",
    "python spikes/moe_prefill/gemm.py --model <gguf> --layer 0|6 --tokens 16 24 32 40 48 56 64 512 1024 2048 4096 8192 "
    "[--freetoken [--f32]] --json <out>",
    "python spikes/moe_prefill/prefill.py run --arms spikes/moe_prefill/kv_<lock>.json --model <gguf> --out <dir>/kv "
    "--rounds 1 --reps 1 --lengths <a prompt below the lock's context>",
]


def rows(path: pathlib.Path) -> list[dict]:
    """A raw.jsonl's rows, an error kept as its last line (the exception, no paths)."""
    out = [json.loads(x) for x in path.read_text().splitlines() if x]
    return [r | ({"error": r["error"].strip().splitlines()[-1]} if "error" in r else {}) for r in out]


def timed(raw: list[dict]) -> dict:
    times = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in raw:
        for length, v in r.items():
            if isinstance(v, dict) and "s" in v:
                times[r["arm"]][length] += v["s"]
    summary = {arm: {length: {"median_s": round(statistics.median(ts), 4), "min_s": min(ts), "max_s": max(ts),
                              "n": len(ts)} for length, ts in by.items()} for arm, by in times.items()}
    out = {"summary": summary}
    for kind, pick in (("default", lambda a: "g4096" not in a), ("gated_4096", lambda a: "g4096" in a)):
        ft = next((a for a in summary if a.startswith("ft") and pick(a)), None)
        if ft:
            out[f"holon_over_freetoken_{kind}"] = {
                arm: {n: round(summary[ft][n]["median_s"] / v["median_s"], 3) for n, v in by.items() if n in summary[ft]}
                for arm, by in summary.items() if not arm.startswith("ft")}
            out[f"freetoken_{kind}_arm"] = ft
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timed")
    ap.add_argument("--nsys")
    ap.add_argument("--gemm", nargs="*", default=[])
    ap.add_argument("--kv", nargs="*", default=[])
    ap.add_argument("--name", default="moe_prefill")
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--commit", required=True)
    a = ap.parse_args()
    record = {"name": a.name, "gpu": a.gpu, "date": a.date, "commit": a.commit, "engines": ENGINES,
              "method": "spikes/moe_prefill/README.md; docs/v1-moe-prefill.md §2.1", "commands": COMMANDS}
    if a.timed:
        raw = rows(pathlib.Path(a.timed, "raw.jsonl"))
        record |= {"timed": timed([r for r in raw if "error" not in r]), "timed_raw": raw,
                   "memory": {r["arm"]: r["memory"] for r in raw if r.get("memory")}}
    if a.nsys:
        record["nsys_breakdown_ms"] = {p.stem: json.loads(p.read_text())
                                       for p in sorted(pathlib.Path(a.nsys).glob("nsys_*.json"))}
    if a.gemm:
        record["gemm_per_call"] = [json.loads(pathlib.Path(p).read_text()) for p in a.gemm]
    if a.kv:
        record["startup_memory"] = {pathlib.Path(d).name: rows(pathlib.Path(d, "raw.jsonl")) for d in a.kv}
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
