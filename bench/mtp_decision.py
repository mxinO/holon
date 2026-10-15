"""Release design D3's MTP default rule, from `bench/run.py` records of a plain lock (`holon`) against MTP over it
(`holon@mtp`) on the Claude-Code-shaped trace, ABBA rounds (several records' rounds are concatenated). Per round:

- the median TPOT of each arm, over requests with more than one token; the ratio MTP / plain;
- the p95 of the per-request maximum inter-token gap, and the p95 TTFT; the ratios MTP / plain.

Over the n rounds (mean and standard error of the per-round ratios):

- **on** needs the TPOT to improve, 1 − mean, by ≥ 5% and by ≥ 2 SE, and for the gap and the TTFT the one-sided 90%
  upper bound, mean + t(0.90, n − 1) · SE, ≤ 1.10;
- **power** (after ≥ 4 rounds): with the mean and the sd held, the fewest rounds at which every criterion would pass;
  none when a mean already fails (more rounds would not change it), else run on to it if it is ≤ 8;
- **plain** when a mean fails, or no round count up to 8 would pass, or 8 rounds are run without passing.

The pooled p99s of the gap and the TTFT are reported as point estimates.

    python bench/mtp_decision.py <record.json> [<record.json> ...] [--trace claude_code] [--json out]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics

from run import pct
from scipy import stats

TPOT_GAIN, TAIL_BOUND, MAX_ROUNDS, POWER_AFTER = 0.05, 1.10, 8, 4


def _rows(run: dict) -> list[dict]:
    return [dict(zip(run["columns"], r, strict=True)) for r in run["requests"]]


def _metrics(rows: list[dict]) -> dict:
    """An arm's round: median TPOT, p95 max gap and p95 TTFT (seconds), and the lists the pooled p99s come from."""
    many = [r for r in rows if r["generated"] > 1]
    tpot = [(r["last"] - r["first"]) / (r["generated"] - 1) for r in many]
    gap, ttft = [r["max_gap"] for r in many], [r["first"] - r["release"] for r in rows]
    return {"tpot_p50": statistics.median(tpot), "gap_p95": pct(gap, 95), "ttft_p95": pct(ttft, 95),
            "_gap": gap, "_ttft": ttft, "requests": len(rows)}


def rounds(records: list[dict], trace: str, plain: str, mtp: str) -> list[dict]:
    """Each round's metrics per arm, the records' rounds in order (a round missing an arm is skipped)."""
    out = []
    for record in records:
        by_round: dict[int, dict] = {}
        for run in record["runs"]:
            if run.get("trace") == trace and run["backend"] in (plain, mtp):
                by_round.setdefault(run["round"], {})[run["backend"]] = _metrics(_rows(run))
        out += [arms for _, arms in sorted(by_round.items()) if len(arms) == 2]
    return out


def _t(n: int) -> float:
    return float(stats.t.ppf(0.90, n - 1))


def passes(mean: dict, sd: dict, n: int) -> dict:
    """The rule at n rounds for these per-round ratio means and sds."""
    se = {k: sd[k] / math.sqrt(n) for k in sd}
    gain = 1 - mean["tpot_p50"]
    return {"tpot": gain >= TPOT_GAIN and gain >= 2 * se["tpot_p50"],
            **{k: mean[k] + _t(n) * se[k] <= TAIL_BOUND for k in ("gap_p95", "ttft_p95")}}


def decide(per_round: list[dict], plain: str, mtp: str) -> dict:
    n = len(per_round)
    keys = ("tpot_p50", "gap_p95", "ttft_p95")
    ratios = {k: [r[mtp][k] / r[plain][k] for r in per_round] for k in keys}
    mean = {k: statistics.mean(v) for k, v in ratios.items()}
    sd = {k: statistics.stdev(v) if n > 1 else math.inf for k, v in ratios.items()}
    se = {k: sd[k] / math.sqrt(n) for k in keys}
    now = passes(mean, sd, n) if n > 1 else {"tpot": False, "gap_p95": False, "ttft_p95": False}
    means_fail = 1 - mean["tpot_p50"] < TPOT_GAIN or mean["gap_p95"] > TAIL_BOUND or mean["ttft_p95"] > TAIL_BOUND
    needed = None
    if not means_fail and n > 1:
        needed = next((m for m in range(n, MAX_ROUNDS + 1) if all(passes(mean, sd, m).values())), None)
    if all(now.values()):
        verdict = "on"
    elif n < POWER_AFTER:
        verdict = f"run on to {POWER_AFTER} rounds"
    elif means_fail:
        verdict = "plain (a mean fails the rule)"
    elif needed is not None and needed > n:
        verdict = f"run on to {needed} rounds"
    else:
        verdict = f"plain (undecided at {n} rounds)"
    pooled = {arm: {"tpot_p50_ms": round(statistics.median([r[arm]["tpot_p50"] for r in per_round]) * 1e3, 2),
                    "gap_p99_ms": round(pct([g for r in per_round for g in r[arm]["_gap"]], 99) * 1e3, 1),
                    "ttft_p99_ms": round(pct([t for r in per_round for t in r[arm]["_ttft"]], 99) * 1e3, 1)}
              for arm in (plain, mtp)}
    return {
        "rounds": n, "requests_per_round": [r[plain]["requests"] for r in per_round],
        "per_round": [{arm: {k: round(r[arm][k] * 1e3, 2) for k in keys} for arm in (plain, mtp)} for r in per_round],
        "ratios": {k: [round(x, 4) for x in v] for k, v in ratios.items()},
        "mean": {k: round(v, 4) for k, v in mean.items()}, "se": {k: round(v, 4) for k, v in se.items()},
        "tpot_gain": round(1 - mean["tpot_p50"], 4),
        "upper_90": {k: round(mean[k] + _t(n) * se[k], 4) if n > 1 else None for k in ("gap_p95", "ttft_p95")},
        "passes": now, "rounds_needed": needed, "verdict": verdict, "pooled_point_estimates": pooled}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records", nargs="+")
    ap.add_argument("--trace", default="claude_code")
    ap.add_argument("--plain", default="holon")
    ap.add_argument("--mtp", default="holon@mtp")
    ap.add_argument("--json")
    args = ap.parse_args()
    per_round = rounds([json.load(open(p)) for p in args.records], args.trace, args.plain, args.mtp)
    result = {"sources": [p.rsplit("/", 1)[-1] for p in args.records], "trace": args.trace,
              **decide(per_round, args.plain, args.mtp)}
    print(json.dumps({k: v for k, v in result.items() if k != "per_round"}, indent=1))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
