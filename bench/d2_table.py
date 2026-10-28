"""M0.7 step 1 (docs/dev/m07-perf.md D2, D9): the per-shape crossover table from repeated sweeps of the candidate
shapes,
and D9 at the entries that ship.

A shape gets an entry — the smallest limit in {3, 6, 7} below llama.cpp's for which every graphed row it moves to mmq
(4, 7, 8) passes there — or keeps llama.cpp's rule. A row passes when, over the repeated sweeps (paired by sweep),
mmq's mean time per call is at least 3% under mmvq's and the mean difference exceeds 3 sd of the per-sweep
differences. The entries ship where their saving sums to at least 1% of some lock's decode or verify step (step 0's
profiles: calls per step by shape and row count; each call priced at the repeats' mean). Entries are chosen on
today's mmq (with its f32 -> bf16 cast, `mmq_cast`); `--variant mmq_f32` derives the whole table — entries, savings,
what ships — from the no-cast column instead (D4: the re-derivation if D4 is kept). D9: the target's 4-row / 1-row
matmul time over its decode step at the shipped entries (with the cast), each call's time the mean over every sweep
that has its shape.

    python bench/d2_table.py --class rtx-5080 --repeats <sweep.json> ... [--full <sweep.json> ...] [--dir evidence]
                             [--variant mmq_cast|mmq_f32] [--json out]

`--repeats`: the candidate sweeps (`qmm_sweep.py --shapes`); `--full`: further sweeps of the target's whole lock
(D9's other shapes), besides step 0's records in `--dir`.
"""
from __future__ import annotations

import argparse
import json
import statistics

from step0_report import DROP, GRAPHED_ROWS, MARGIN, d9, load, shapes_of

LIMITS = (3, 6, 7)
SD = 3.0                       # an entry's difference must exceed this many sd of the per-sweep differences


def row_test(runs: list[dict], rows: int, variant: str) -> dict:
    """mmvq against `variant` at `rows` over the sweeps `runs` (one shape's records, paired by sweep)."""
    vq = [r["us"][str(rows)]["mmvq"] for r in runs]
    q = [r["us"][str(rows)][variant] for r in runs]
    d = [a - b for a, b in zip(vq, q, strict=True)]
    sd = statistics.stdev(d) if len(d) > 1 else float("inf")
    mean_vq, mean_q = statistics.mean(vq), statistics.mean(q)
    return {"mmvq_us": round(mean_vq, 3), "mmq_us": round(mean_q, 3), "margin": round(1 - mean_q / mean_vq, 4),
            "diff_us": round(statistics.mean(d), 3), "sd_us": round(sd, 3),
            "passes": mean_q <= (1 - MARGIN) * mean_vq and statistics.mean(d) > SD * sd}


def entry(runs: list[dict], variant: str) -> tuple[int | None, dict]:
    """The shape's limit (None: llama.cpp's rule) and each graphed row's test."""
    llama = runs[0]["llama_mmvq_rows"]
    tests = {r: row_test(runs, r, variant) for r in GRAPHED_ROWS if r <= llama}
    for limit in LIMITS:
        moved = [r for r in GRAPHED_ROWS if limit < r <= llama]
        if limit < llama and moved and all(tests[r]["passes"] for r in moved):
            return limit, tests
    return None, tests


def table(repeats: list[dict], variant: str) -> dict:
    """Per candidate shape: its entry (`limit`, on `variant`) and, for the record, with and without the cast, and the
    row tests behind them."""
    out = {}
    for key in shapes_of(repeats[0]):
        runs = [shapes_of(sw)[key] for sw in repeats if key in shapes_of(sw)]
        with_cast, tests = entry(runs, "mmq_cast")
        no_cast, tests_f32 = entry(runs, "mmq_f32")
        out["{}/{}x{}/{}".format(*key)] = {"llama": runs[0]["llama_mmvq_rows"], "with_cast": with_cast,
                                          "no_cast": no_cast, "sweeps": len(runs), "rows": tests,
                                          "rows_no_cast": tests_f32,
                                          "limit": with_cast if variant == "mmq_cast" else no_cast}
    return out


def savings(entries: dict, repeats: list[dict], profiles: dict[str, dict], variant: str) -> list[dict]:
    """Each profiled decode / verify step's saving under the entries against llama.cpp's rule (calls priced at the
    repeats' means; only the candidates' calls move)."""
    mean = {}
    for key in entries:
        runs = [shapes_of(sw)[tuple(_key(key))] for sw in repeats if tuple(_key(key)) in shapes_of(sw)]
        mean[key] = {r: {v: statistics.mean(x["us"][r][v] for x in runs) for v in ("mmvq", variant)}
                     for r in runs[0]["us"]}
    out = []
    for name, rec in profiles.items():
        for rk, row in rec["rows"].items():
            if row["kind"] == "prefill":
                continue
            by_entry = {}
            for c in row.get("gemm", {}).get("calls", []):
                key = f"{c['type']}/{c['in']}x{c['out']}/{c['mode']}"
                e, rows = entries.get(key), str(c["rows"])
                if not e or not e["limit"] or rows not in mean[key] or not e["limit"] < c["rows"] <= e["llama"]:
                    continue
                u = mean[key][rows]
                by_entry[key] = by_entry.get(key, 0.0) + c["count"] * (u["mmvq"] - u[variant]) / 1e3
            step = row["graphed_wall_ms"]
            out.append({"model": name, "row": rk, "step_ms": step, "saving_ms": round(sum(by_entry.values()), 3),
                        "share": round(sum(by_entry.values()) / step, 4),
                        "by_entry": {k: round(v / step, 4) for k, v in sorted(by_entry.items())}})
    return out


def _key(key: str) -> list:
    name, shape, mode = key.split("/")
    k, out = shape.split("x")
    return [name, int(k), int(out), mode]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--class", dest="cls", required=True)
    ap.add_argument("--repeats", nargs="+", required=True)
    ap.add_argument("--full", nargs="*", default=[])
    ap.add_argument("--dir", default="docs/dev/evidence")
    ap.add_argument("--variant", choices=("mmq_cast", "mmq_f32"), default="mmq_cast")
    ap.add_argument("--json")
    args = ap.parse_args()
    repeats = [json.load(open(p)) for p in args.repeats]
    full = [json.load(open(p)) for p in args.full]
    profiles = load(args.dir, args.cls, "profile")
    entries = table(repeats, args.variant)
    steps = savings(entries, repeats, profiles, args.variant)
    shipped = sorted({k for s in steps if s["share"] >= DROP for k in s["by_entry"]})
    out = {"class": args.cls, "gpu": repeats[0]["gpu"], "variant": args.variant, "sweeps": len(repeats),
           "entries": entries, "steps": steps, "shipped": {k: entries[k]["limit"] for k in shipped}}
    step0 = [*load(args.dir, args.cls, "qmm-sweep", exact=True).values(),
             *next(iter(load(args.dir, args.cls, "qmm-sweep-d9-repeat", exact=True).values()), {}).get("repeats", [])]
    target = profiles.get("27b-plain")
    if target and args.variant == "mmq_cast":               # D9 is defined with today's cast
        out["d9"] = d9([*step0, *full, *repeats], target, {k: {"with_cast": v} for k, v in out["shipped"].items()})
    out["crossover"] = [f'("{args.cls}", "{n}", {k}, {o}, "{m}"): {v},' for n, k, o, m, v in
                        (_key(key) + [lim] for key, lim in out["shipped"].items())]
    print(json.dumps({k: v for k, v in out.items() if k != "entries"}, indent=1))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
