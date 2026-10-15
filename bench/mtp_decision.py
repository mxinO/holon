"""Release design D3's MTP default rule, from `bench/run.py` records of a plain lock (`holon`) against MTP over it
(`holon@mtp`) on the Claude-Code-shaped trace, ABBA rounds; a run-on's records follow the first's, in order. Per
round:

- the median TPOT of each arm, over requests with more than one token; the ratio MTP / plain;
- the p95 of the per-request maximum inter-token gap, and the p95 TTFT; the ratios MTP / plain.

Over the n rounds (mean and standard error of the per-round ratios):

- **on** needs the TPOT to improve, 1 − mean, by ≥ 5% and by ≥ 2 SE, and for the gap and the TTFT the one-sided 90%
  upper bound, mean + t(0.90, n − 1) · SE, ≤ 1.10;
- **one power estimate**, from the first 4 rounds: with their mean and sd held, the fewest rounds (≤ 8) at which every
  criterion would pass. None when a mean fails (more rounds would not change it) or no count up to 8 passes: plain.
  Otherwise the run goes on to that count, and the rule is applied once there: on, or plain;
- no verdict before 4 rounds or between 4 and the target; none past 8 rounds; never on if a round lost an arm or
  ran fewer than 200 requests.

The records must agree (model, class, lock plan, trace, corpus, arms) and sample at temperature 1.0. Also reported:
the pooled p99s as point estimates, each arm's KV tokens, prefix tokens saved and preemptions, and MTP's accepted
drafts per k = 3 verify row (M0.5's Claude Code capture: 1.78).

    python bench/mtp_decision.py <record.json> [<record.json> ...] [--trace claude_code] [--json out]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics

from run import pct
from scipy import stats

TPOT_GAIN, TAIL_BOUND, POWER_AFTER, MAX_ROUNDS, TEMPERATURE, MIN_REQUESTS = 0.05, 1.10, 4, 8, 1.0, 200
KEYS = ("tpot_p50", "gap_p95", "ttft_p95")
SAME = ("model", "gpu_class", "traces_sha256", "corpus", "arms")


def _metrics(run: dict) -> dict:
    """An arm's round: median TPOT, p95 max gap and p95 TTFT (seconds), its counters, and the pooled p99s' lists."""
    rows = [dict(zip(run["columns"], r, strict=True)) for r in run["requests"]]
    many = [r for r in rows if r["generated"] > 1]
    tpot = [(r["last"] - r["first"]) / (r["generated"] - 1) for r in many]
    gap, ttft = [r["max_gap"] for r in many], [r["first"] - r["release"] for r in rows]
    return {"tpot_p50": statistics.median(tpot), "gap_p95": pct(gap, 95), "ttft_p95": pct(ttft, 95),
            "requests": len(rows), "stats": run.get("stats", {}), "_gap": gap, "_ttft": ttft}


def rounds(records: list[dict], trace: str, plain: str, mtp: str) -> tuple[list[dict], list[str]]:
    """Each complete round's metrics per arm, the records' rounds in order; and every arm's error, by round."""
    out, errors = [], []
    for k, record in enumerate(records):
        by_round: dict[int, dict] = {}
        for run in record["runs"]:
            if run["backend"] not in (plain, mtp):
                continue
            if "error" in run:
                errors.append(f"record {k} round {run['round']} {run['backend']}: {run['error']}")
            elif run["trace"] == trace:
                by_round.setdefault(run["round"], {})[run["backend"]] = _metrics(run)
        for r, arms in sorted(by_round.items()):
            if len(arms) < 2:
                errors.append(f"record {k} round {r}: only {', '.join(arms)}")
            elif min(a["requests"] for a in arms.values()) < MIN_REQUESTS:
                errors.append(f"record {k} round {r}: fewer than {MIN_REQUESTS} requests")
            else:
                out.append(arms)
    return out, errors


def check(records: list[dict], paths: list[str]) -> None:
    """The records are distinct and of one experiment, sampled at D3's temperature."""
    if len({pathlib.Path(p).resolve() for p in paths}) < len(paths) or \
            len({json.dumps(r["runs"], sort_keys=True) for r in records}) < len(records):
        raise SystemExit("a record is given twice")
    for key in SAME:
        if len({json.dumps(r.get(key), sort_keys=True) for r in records}) > 1:
            raise SystemExit(f"the records differ in {key}")
    if len({r["matched"]["lock"]["plan_id"] for r in records}) > 1:
        raise SystemExit("the records ran different locks")
    if any(r["matched"]["temperature"] != TEMPERATURE for r in records):
        raise SystemExit(f"D3 samples at temperature {TEMPERATURE}")


def _t(n: int) -> float:
    return float(stats.t.ppf(0.90, n - 1))


def summary(ratios: dict, n: int) -> tuple[dict, dict]:
    """Means and sds of the first n per-round ratios (rounded: a ratio of exactly 1.10 is 1.10)."""
    mean = {k: round(statistics.mean(v[:n]), 6) for k, v in ratios.items()}
    sd = {k: round(statistics.stdev(v[:n]), 6) for k, v in ratios.items()}
    return mean, sd


def passes(mean: dict, sd: dict, n: int) -> dict:
    """The rule at n rounds for these per-round ratio means and sds."""
    se = {k: sd[k] / math.sqrt(n) for k in KEYS}
    gain = round(1 - mean["tpot_p50"], 6)
    return {"tpot": gain >= TPOT_GAIN and gain >= 2 * se["tpot_p50"],
            **{k: round(mean[k] + _t(n) * se[k], 6) <= TAIL_BOUND for k in ("gap_p95", "ttft_p95")}}


def target(mean: dict, sd: dict) -> int | None:
    """D3's one power estimate, from the first 4 rounds: the round count to run to, or None (plain)."""
    if 1 - mean["tpot_p50"] < TPOT_GAIN or mean["gap_p95"] > TAIL_BOUND or mean["ttft_p95"] > TAIL_BOUND:
        return None
    return next((m for m in range(POWER_AFTER, MAX_ROUNDS + 1) if all(passes(mean, sd, m).values())), None)


def decide(per_round: list[dict], errors: list[str], plain: str, mtp: str) -> dict:
    n = len(per_round)
    if n < POWER_AFTER:
        return {"rounds": n, "errors": errors, "verdict": f"run on to {POWER_AFTER} rounds"}
    if n > MAX_ROUNDS:
        raise SystemExit(f"{n} rounds: D3 runs at most {MAX_ROUNDS}")
    ratios = {k: [r[mtp][k] / r[plain][k] for r in per_round] for k in KEYS}
    goal = target(*summary(ratios, POWER_AFTER))
    mean, sd = summary(ratios, n)
    now = passes(mean, sd, n)
    if goal is None:
        verdict = "plain (the power estimate at 4 rounds: a mean fails, or no count up to 8 passes)"
    elif n < goal:
        verdict = f"run on to {goal} rounds"
    elif not all(now.values()):
        verdict = f"plain (the rule fails at {n} rounds)"
    else:
        verdict = "plain (a round lost an arm or ran short)" if errors else "on"
    se = {k: sd[k] / math.sqrt(n) for k in KEYS}
    arms = {}
    for arm in (plain, mtp):
        counters = {k: sum(r[arm]["stats"].get(k, 0) for r in per_round)
                    for k in ("prefix_tokens_saved", "preemptions")}
        k3 = {int(k.rsplit("_", 1)[1]): sum(r[arm]["stats"].get(k, 0) for r in per_round)
              for k in {k for r in per_round for k in r[arm]["stats"] if k.startswith("spec_rows_k3_accepting_")}}
        arms[arm] = {
            "tpot_p50_ms_median_of_rounds": round(statistics.median(r[arm]["tpot_p50"] for r in per_round) * 1e3, 2),
            "gap_p99_ms_pooled": round(pct([g for r in per_round for g in r[arm]["_gap"]], 99) * 1e3, 1),
            "ttft_p99_ms_pooled": round(pct([t for r in per_round for t in r[arm]["_ttft"]], 99) * 1e3, 1),
            "requests_per_round": [r[arm]["requests"] for r in per_round], **counters,
            **({"accepted_per_k3_row": round(sum(a * c for a, c in k3.items()) / sum(k3.values()), 3)}
               if sum(k3.values()) else {})}
    return {
        "rounds": n, "errors": errors, "power_target": goal,
        "per_round_ms": [{arm: {k: round(r[arm][k] * 1e3, 2) for k in KEYS} for arm in (plain, mtp)}
                         for r in per_round],
        "ratios": {k: [round(x, 4) for x in v] for k, v in ratios.items()},
        "mean": mean, "se": {k: round(v, 4) for k, v in se.items()}, "tpot_gain": round(1 - mean["tpot_p50"], 4),
        "upper_90": {k: round(mean[k] + _t(n) * se[k], 4) for k in ("gap_p95", "ttft_p95")},
        "passes": now, "verdict": verdict, "arms": arms}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records", nargs="+", help="run.py records in round order (a run-on's after the first's)")
    ap.add_argument("--trace", default="claude_code")
    ap.add_argument("--plain", default="holon")
    ap.add_argument("--mtp", default="holon@mtp")
    ap.add_argument("--json")
    args = ap.parse_args()
    records = [json.load(open(p)) for p in args.records]
    check(records, args.records)
    per_round, errors = rounds(records, args.trace, args.plain, args.mtp)
    result = {"sources": [p.rsplit("/", 1)[-1] for p in args.records], "trace": args.trace,
              "kv_tokens": {b: i.get("kv_tokens") for b, i in records[0]["backends"].items()},
              **decide(per_round, errors, args.plain, args.mtp)}
    print(json.dumps({k: v for k, v in result.items() if k != "per_round_ms"}, indent=1))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
