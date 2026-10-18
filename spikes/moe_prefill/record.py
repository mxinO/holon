"""moe_prefill's evidence record: the timed arms (`prefill.py run`'s raw.jsonl: medians over repetitions × rounds,
ratios against FreeToken), the nsys breakdowns (`breakdown.py`'s JSON per arm) and the per-call GEMM benches
(`gemm.py --json`), with the commands and engine pins.

    python spikes/moe_prefill/record.py --timed <dir> --nsys <dir> --gemm <json>... --gpu "RTX A5000" \\
        --date 2026-10-18 --commit <sha> > evidence/moe_prefill_rtx-a5000_<date>.json
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics

ENGINES = {"holon": "this repository at the record's commit (all-resident: std.moe@1 with the default resident store)",
           "freetoken": "FreeToken 21d9578bde22 (bench/setup_baselines.sh offload: torch 2.14.1+cu130, [accel]), "
                        "--moe-strategy fused (all experts resident)",
           "model": "Qwen3-30B-A3B Q4_K_M (unsloth/Qwen3-30B-A3B-GGUF)"}
COMMANDS = [
    "python spikes/moe_prefill/prefill.py run --arms spikes/moe_prefill/arms.json --model <gguf> --out <dir>/timed "
    "--rounds 2 --holon-python <holon venv python> --freetoken-python <freetoken venv python>",
    "python spikes/moe_prefill/prefill.py run --arms spikes/moe_prefill/arms_nsys.json --model <gguf> --out <dir>/nsys "
    "--rounds 1 ...",
    "nsys export --type sqlite -o <report>.sqlite <report>.nsys-rep; "
    "python spikes/moe_prefill/breakdown.py <report>.sqlite > <report>.json",
    "python spikes/moe_prefill/gemm.py --model <gguf> --layer 0|6 [--freetoken | --j-check] --json <out>",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timed", required=True)
    ap.add_argument("--nsys", required=True)
    ap.add_argument("--gemm", nargs="+", default=[])
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--commit", required=True)
    a = ap.parse_args()
    raw = [json.loads(x) for x in pathlib.Path(a.timed, "raw.jsonl").read_text().splitlines() if x]
    times = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in raw:
        for length, v in r.items():
            if isinstance(v, dict) and "s" in v:
                times[r["arm"]][length] += v["s"]
    summary = {arm: {length: {"median_s": round(statistics.median(ts), 4), "tok_s": round(int(length) /
                                                                                    statistics.median(ts), 1),
                              "min_s": min(ts), "max_s": max(ts), "n": len(ts)} for length, ts in by.items()}
               for arm, by in times.items()}
    ft = summary.get("ft_fused", {})
    ratios = {arm: {length: round(ft[length]["median_s"] / v["median_s"], 3)
                    for length, v in by.items() if length in ft} for arm, by in summary.items() if arm != "ft_fused"}
    nsys = {p.stem: json.loads(p.read_text()) for p in sorted(pathlib.Path(a.nsys).glob("nsys_*.json"))}
    gemm = [json.loads(pathlib.Path(p).read_text()) for p in a.gemm]
    print(json.dumps({"name": "moe_prefill", "gpu": a.gpu, "date": a.date, "commit": a.commit, "engines": ENGINES,
                      "method": "spikes/moe_prefill/README.md", "commands": COMMANDS,
                      "timed_summary": summary, "holon_over_freetoken_speed": ratios, "timed_raw": raw,
                      "nsys_breakdown_ms": nsys, "gemm_per_call": gemm}, indent=1))


if __name__ == "__main__":
    main()
