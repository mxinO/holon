"""docs/m07-perf.md §1.3's mixed-step bars, from a `bench/run.py` record whose first backend is today's schedule
(`holon`) and whose other arms set `chunk_with_decode`: per trace and arm,

- the per-request maximum inter-token gap at p50, p99 and max, and its improvement (today's / the arm's; the bar:
  ≥ 2× at p99 and max);
- the paired TTFT (run.py's median over (round, session, turn) of the arm's TTFT over today's), against
  1 + max(15%, 2 sd), the sd that of today's per-round TTFT medians relative to their mean;
- TTFT p99 against today's (the long prompt's bar: at most +25%);
- throughput against today's, against max(5%, 2 sd of today's per-round throughput, relative);
- TPOT p99 (recorded), and the mixed steps counted.

`--aa`: the A/A of the paired TTFT instead — today's arm against itself, round against round, per (session, turn)
(run.py's pairing), each unordered pair of rounds once (both orders would pool r with 1/r: a median of 1 by
construction): the per-pair medians, their sd and the one farthest from 1.

    python bench/mixed_bars.py <record.json> [--aa] [--json out]
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics

import run


def _rel_sd(values: list[float]) -> float:
    return statistics.stdev(values) / statistics.mean(values) if len(values) > 1 else 0.0


def bars(record: dict) -> dict:
    base_name = record["order"][0] if isinstance(record["order"], list) else "holon"
    out = {}
    for trace, by_backend in record["summary"].items():
        base = by_backend[base_name]
        rounds = sorted({r["round"] for r in record["runs"] if r["trace"] == trace and r["backend"] == base_name})
        ttft_rounds = [r["summary"]["ttft_ms"]["p50"] for r in record["runs"]
                       if r["trace"] == trace and r["backend"] == base_name]
        ttft_bar = 1 + max(0.15, 2 * _rel_sd(ttft_rounds))
        tput_bar = max(0.05, 2 * _rel_sd(base["throughput_per_round"]))
        rows = {}
        for name, s in by_backend.items():
            gap = s["max_gap_ms"]
            row = {"rounds": len(s["throughput_per_round"]), "gap_ms": gap, "ttft_ms": s["ttft_ms"],
                   "tpot_p99_ms": s["tpot_ms"]["p99"], "throughput_tok_s": s["throughput_tok_s"],
                   "mixed_steps": s.get("stats", {}).get("mixed_steps")}
            if name != base_name:
                b = base["max_gap_ms"]
                row["gap_x"] = {q: round(b[q] / gap[q], 2) if gap[q] else None for q in ("p50", "p99", "max")}
                row["paired_ttft"] = s.get("paired_ttft", {}).get("median")
                row["ttft_p99_change"] = round(s["ttft_ms"]["p99"] / base["ttft_ms"]["p99"] - 1, 4)
                row["throughput_change"] = round(s["throughput_tok_s"] / base["throughput_tok_s"] - 1, 4)
                row["passes"] = {
                    "gap ≥ 2× (p99, max)": all(row["gap_x"][q] is not None and row["gap_x"][q] >= 2
                                              for q in ("p99", "max")),
                    f"paired TTFT ≤ {ttft_bar:.3f}": row["paired_ttft"] is not None and row["paired_ttft"] <= ttft_bar,
                    "TTFT p99 ≤ +25%": row["ttft_p99_change"] <= 0.25,
                    f"throughput within {tput_bar:.1%}": row["throughput_change"] >= -tput_bar,
                    "mixed steps > 0": bool(row["mixed_steps"])}
            rows[name] = row
        out[trace] = {"baseline": base_name, "baseline_rounds": rounds, "ttft_bar": round(ttft_bar, 4),
                      "throughput_bar": round(tput_bar, 4), "arms": rows}
    return out


def aa(record: dict) -> dict:
    """Per trace: today's paired TTFT against itself (`run.paired_ttft`), each unordered pair of rounds as a round."""
    base_name = record["order"][0] if isinstance(record["order"], list) else "holon"
    out = {}
    for trace in record["summary"]:
        runs = [r for r in record["runs"] if r["trace"] == trace and r["backend"] == base_name]
        pairs = list(itertools.combinations(sorted(runs, key=lambda r: r["round"]), 2))
        paired = run.paired_ttft([dict(a, round=n) for n, (a, _) in enumerate(pairs)],
                                 [dict(b, round=n) for n, (_, b) in enumerate(pairs)])
        medians = list(paired["per_round"].values())
        out[trace] = {"rounds": len(runs), "pairs": len(pairs), "per_pair": medians,
                      "farthest": max(medians, key=lambda m: abs(m - 1)),
                      "sd": round(statistics.stdev(medians), 4) if len(medians) > 1 else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("record")
    ap.add_argument("--aa", action="store_true", help="today's arm against itself (paired TTFT)")
    ap.add_argument("--json")
    args = ap.parse_args()
    if args.aa:
        result = aa(json.load(open(args.record)))
        for trace, t in result.items():
            print(f"{trace}: A/A paired TTFT over {t['pairs']} pairs of {t['rounds']} rounds: per-pair medians "
                  f"{t['per_pair']}, sd {t['sd']}, farthest from 1 {t['farthest']}")
        if args.json:
            with open(args.json, "w") as f:
                json.dump({"source": args.record.rsplit("/", 1)[-1], "aa": result}, f, indent=1)
        return
    result = bars(json.load(open(args.record)))
    for trace, t in result.items():
        print(f"## {trace} (paired TTFT bar {t['ttft_bar']}, throughput bar {t['throughput_bar']:.1%})")
        for name, r in t["arms"].items():
            g = r["gap_ms"]
            line = (f"{name:12s} gap p50/p99/max {g['p50']:.0f} / {g['p99']:.0f} / {g['max']:.0f} ms; TTFT p50/p99 "
                    f"{r['ttft_ms']['p50']:.0f} / {r['ttft_ms']['p99']:.0f}; TPOT p99 {r['tpot_p99_ms']}; "
                    f"{r['throughput_tok_s']} tok/s; mixed {r['mixed_steps']}")
            if "passes" in r:
                line += (f"; gap x {r['gap_x']['p99']} / {r['gap_x']['max']}; paired {r['paired_ttft']}; TTFT p99 "
                         f"{r['ttft_p99_change']:+.1%}; tput {r['throughput_change']:+.1%}; "
                         + ", ".join(f"{k}: {'ok' if v else 'FAIL'}" for k, v in r["passes"].items()))
            print(line)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"source": args.record.rsplit("/", 1)[-1], "bars": result}, f, indent=1)


if __name__ == "__main__":
    main()
