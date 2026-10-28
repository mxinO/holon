"""Milestone X's close (docs/dev/v1-offload.md §14, "Step 8, part 2"): the record evidence/x8_close_rtx-5080_<date>.json —
F5b's tune at the capped fit, the capped-fit and F1 rows (interleaved rounds, spikes/x4_gap/ctx.py), the new offload
locks with their verdicts and snapshots, F5b's bounded L2 against the path-aware floor, and the one propagation
(D15) of every lock in plans/, class by class. Prints the README's tables.

    python spikes/x8_close/record.py --runs <dir: <class>/{verdict,prop}-<name>.log, snap-<name>.json, rtx-5080/...> \\
        --date <date> --commit <measured commit> --out evidence/x8_close_rtx-5080_<date>.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "x4_gate"))
from record import CONTEXTS, ENGINES, MODEL, ROWS, abba, rows, scrub, stats  # noqa: E402  (x4_gate's)

CLASSES = ("rtx-5080", "rtx-4090", "rtx-5090", "rtx-3090")
# Holon arm -> reference arms (acceptance 3's rows at the capped fit)
CAPPED_REFS = {"ft": "cap_ft", "fit": "cap_fit", "ncmoe": "cap_ncmoe36", "cache": "cap_cache",
               "cache_pinned": "cap_cache_pinned"}
NATURAL_REFS = {"ft": "ft_nat", "fit": "llama_fit", "ncmoe": "llama_ncmoe16", "cache": "cache_nat",
                "cache_pinned": "cache_nat_pinned"}
# The F1 row (bar 0.95x decode): Holon static against llama.cpp's static split at each fit
F1 = {"capped": ({"f5a": "f1_25", "f5b": "f1_25_f5b"}, {"--fit (-fitt 7150)": "cap_fit",
                                                        "--n-cpu-moe 36": "cap_ncmoe36"}),
      "natural": ({"f5a": "f1_nat", "f5b": "f1_nat_f5b"}, {"--fit (default)": "llama_fit",
                                                           "--n-cpu-moe 16": "llama_ncmoe16"})}
L2_LINE = re.compile(r"^\s*(verdict|snapshot trace|trace covered|overall|floor|vs floor|budget|PASS|FAIL|wrote|unit:|"
                     r"site|engine|commit|uv_lock|attention\.|F5b|moe|exit)")


def load(path: pathlib.Path):
    return json.loads(path.read_text()) if path.exists() else None


def fit_of(path: pathlib.Path) -> dict | None:
    """fit.py's line (`FIT {…}`) without its pools."""
    if not path.exists():
        return None
    t = path.read_text()
    out = json.loads(t[t.index("{"):])
    return {k: v for k, v in out.items() if k != "pools"}


def ratios(st: dict, holon: str, refs: dict, bars: dict | None = None, metrics=("decode_tps", "prefill_tps")) -> list:
    out = []
    for kind, ref in refs.items():
        if ref not in st or holon not in st:
            continue
        for c in CONTEXTS:
            h, r = st[holon].get(c), st[ref].get(c)
            if h is None or r is None:
                continue
            for metric in metrics:
                bar = (bars or {}).get((kind, metric))
                ratio = h[metric]["median"] / r[metric]["median"]
                out.append({"holon_arm": holon, "reference": kind, "reference_arm": ref, "context": int(c),
                            "metric": metric, "holon": h[metric]["median"], "reference_rate": r[metric]["median"],
                            "ratio": round(ratio, 3), "bar": bar,
                            "ratio_range": [round(h[metric]["min"] / r[metric]["max"], 3),
                                            round(h[metric]["max"] / r[metric]["min"], 3)],
                            "verdict": None if bar is None else "pass" if ratio >= bar else "miss"})
    return out


def report(path: pathlib.Path) -> list[str] | None:
    if not path.exists():
        return None
    return [x.rstrip() for x in path.read_text(errors="replace").splitlines() if L2_LINE.match(x)]


def snap(path: pathlib.Path) -> dict | None:
    if not path.exists():
        return None
    s = json.loads(path.read_text())
    return {k: s[k] for k in ("runs", "trace", "steps", "run_identical", "diffs")}


def propagation(runs: pathlib.Path) -> list[dict]:
    out = []
    for cls in CLASSES:
        d = runs / cls
        for v in sorted(d.glob("verdict-*.log")):
            name = v.stem.removeprefix("verdict-")
            rep = report(v) or []
            text = "\n".join(rep)
            prop = report(d / f"prop-{name}.log") or []
            wrote = next((x.split("wrote ", 1)[1] for x in prop if x.startswith("wrote ")), None)
            out.append({"class": cls, "lock": name,
                        "verdict": next((x.split(": ", 1)[1] for x in rep if x.startswith("verdict:")), None),
                        "trace": ("identical" if "byte-identical" in text else
                                  "differs" if "differs from the lock" in text else None),
                        "l2": "PASS" if "PASS" in rep else "FAIL" if "FAIL" in rep else None,
                        "verdict_report": rep, "approved": wrote is not None, "written": wrote,
                        "approve_report": prop or None, "snapshot_x3": snap(d / f"snap-{name}.json")})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=pathlib.Path, required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()
    d = a.runs / "rtx-5080"
    st = stats(rows(d / "runs" / "close"))
    nat_rows = [r for n in ("natural", "f1nat") if (d / "runs" / n / "raw.jsonl").exists()
                for r in rows(d / "runs" / n)]
    nst = stats([r for r in nat_rows if not r.get("error")]) if nat_rows else {}
    bars = {(k, "decode_tps"): b for _, k, b, _ in ROWS} | {(k, "prefill_tps"): p for _, k, _, p in ROWS}
    capped = {arm: ratios(st, arm, CAPPED_REFS, bars) for arm in ("holon_25", "holon_25_lc0", "holon_25_f5b")
              if arm in st}
    f1 = {fit: {split: ratios(src, arm, refs, {(k, "decode_tps"): 0.95 for k in refs}, ("decode_tps",))
                for split, arm in arms.items() if arm in src}
          for fit, (arms, refs) in F1.items() for src in [st if fit == "capped" else nst]}
    natural = ratios(nst, "holon_nat", NATURAL_REFS, bars) if "holon_nat" in nst else None
    record = {
        "name": "x8_close", "date": a.date, "gpu": "NVIDIA GeForce RTX 5080", "model": MODEL, "commit": a.commit,
        "references": ENGINES,
        "method": ("spikes/x4_gap/ctx.py: every arm in a fresh process, an untimed 512 + 64 warm-up, one request "
                   "per context of natural-text prompt ids (1,024 / 4,096 / 32,768 tokens) generating 256 tokens, "
                   "batch 1, greedy, ignore_eos; every engine allocates a 33,792-token context; arms interleaved, "
                   "reversed on odd rounds (ABBA), 3 rounds, one desktop, one day; medians"),
        "fits": {n: fit_of(d / f"{f}.json") for n, f in (("lfu", "fit"), ("static", "fit_static"),
                                                       ("static_f5b", "fit_static_f5b"), ("lfu_f5b", "fit_f5b"))},
        "f5b_tune": {"capped": load(d / "tune_split.json"), "capped_log": report(d / "tune_split.log"),
                     "natural": load(d / "tune_split_nat.json"), "natural_log": report(d / "tune_split_nat.log")},
        "f5b_natural_rows": ({"summary": (fst := stats(rows(d / "runs" / "natf5b"))),
                              "f5b_over_f5a_same_pin": abba(fst, "holon_nat_lc0_f5bpin", "holon_nat_f5b", "decode_tps"),
                              "f5b_over_the_lock": abba(fst, "holon_nat", "holon_nat_f5b", "decode_tps")}
                             if (d / "runs" / "natf5b" / "raw.jsonl").exists() else None),
        "rows": {"summary": st, "capped": capped, "f1": f1,
                 "natural_at_close": {"summary": nst, "rows": natural,
                                      "errors": [{k: r.get(k) for k in ("round", "arm", "error")}
                                                 for r in nat_rows if r.get("error")]},
                 "f5b_over_f5a_decode": abba(st, "holon_25_lc0", "holon_25_f5b", "decode_tps")
                 if {"holon_25_lc0", "holon_25_f5b"} <= st.keys() else None,
                 "f5b_over_f5a_prefill": abba(st, "holon_25_lc0", "holon_25_f5b", "prefill_tps")
                 if {"holon_25_lc0", "holon_25_f5b"} <= st.keys() else None,
                 "loop_core_decode": abba(st, "holon_25", "holon_25_lc0", "decode_tps")
                 if {"holon_25", "holon_25_lc0"} <= st.keys() else None},
        "l0_ggml_cpu_avx512": report(d / "split2.log"),
        "offload_locks": {
            "natural": {"verdict": report(d / "nat-verdict.log"), "approve": report(d / "nat-approve.log"),
                        "snapshot_x3": snap(d / "snap-nat-lock.json")},
            "capped_f5b": {"written": False,
                           "propagate_verdict": report(d / "f5b-verdict.log"),
                           "snapshot_x3": snap(d / "snap-f5b-capped.json"),
                           "bounded_l2": load(d / "f5b_compare.json"),
                           "candidate": (d / "f5b_l2.log").read_text().strip().splitlines()[-1:]
                           if (d / "f5b_l2.log").exists() else None,
                           "split": load(d / "f5b_split.json")}},
        "propagation_lost_desktop": [{"class": "rtx-3090", "lock": v.stem.removeprefix("verdict-"), "report": report(v)}
                                     for v in sorted((a.runs / "rtx-3090-dead").glob("verdict-*.log"))],
        "propagation": propagation(a.runs),
        "arms": {n: json.loads((HERE / f).read_text()) for n, f in (("capped", "arms.json"),
                                                                    ("natural", "arms_natural.json"))},
    }
    record = scrub(record)
    if a.out:
        a.out.write_text(json.dumps(record, indent=1) + "\n")
    for arm, g in list(capped.items()) + [("natural at the close: holon_nat", natural or [])]:
        print(f"\n## {arm}")
        for metric in ("decode_tps", "prefill_tps"):
            print(f"\n| {metric} | 1K | 4K | 32K |\n|---|---|---|---|")
            for _, kind, dbar, pbar in ROWS:
                cells = [next((f"{x['ratio']:.2f} {x['verdict']}" for x in g if x["reference"] == kind
                               and x["metric"] == metric and str(x["context"]) == c), "—") for c in CONTEXTS]
                print(f"| {kind} ({dbar if metric == 'decode_tps' else pbar}x) | " + " | ".join(cells) + " |")
    for fit, by in f1.items():
        for split, g in by.items():
            print(f"\n## F1 {fit} {split}: " + "; ".join(f"{x['reference']} {x['context']} {x['ratio']:.2f}"
                                                      for x in g))
    print("\n## absolute (decode / prefill)")
    for arm, cs in list(st.items()) + list(nst.items()):
        print(f"| {arm} | " + " | ".join(f"{cs[c]['decode_tps']['median']:.1f} / {cs[c]['prefill_tps']['median']:.0f}"
                                          for c in CONTEXTS if c in cs) + " |")
    print("\n## propagation")
    for p in record["propagation"]:
        print(f"| {p['class']} | {p['lock']} | {p['trace']} | {p['l2']} | {p['approved']} | "
              f"{(p['snapshot_x3'] or {}).get('run_identical')} |")


if __name__ == "__main__":
    main()
