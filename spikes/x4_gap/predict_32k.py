"""Acceptance 3's decode rows predicted at 1K / 4K / 32K (README_32k.md), x4_gap's way: Holon's decode time =
the bytes-to-time line through two measured placements of the 47% cache (its intercept is everything but the misses,
the offload store's own +0.2 ms included) + F11's measured cost + the line's slope × the replayed `lfu` miss bytes;
ratios against x4_refs' medians. 1K / 4K come from x4_gap's record as is; 32K from the 32K runs (`ctx.py` with
`--contexts 1024 4096 32768` on the same arms), whose 1K / 4K rows check that the reruns reproduce x4_gap. The same
model at FreeToken's own decode miss bytes (its counters, one run per fit) is the comparison column.

    python spikes/x4_gap/replay.py <out>/x4_32k/r0_holon_{static,prof,prof25}.npz > <out>/x4_32k/replay.json
    python spikes/x4_gap/predict_32k.py --runs <out> --date <date> --commit <commit>   # <out>: x4_32k, x4_32k_ft
"""
from __future__ import annotations

import argparse
import json
import pathlib

from record import EXPERT_MB, LAYERS, ROOT, medians, rows

F11_MS = {"natural": 0.08, "capped": 0.11}   # F11's pick and decay with candidates, per token (§13, after step 3)
F11_BAND = 0.04                               # ± ms
BYTES_BAND = 0.10                             # ± on the replayed miss bytes (replay against the engine)
REFS = {  # acceptance 3's decode rows: (x4_refs arm, bar) per fit
    "natural": {"FreeToken": ("nat_ft_auto", 1.0), "llama.cpp cache, pinned": ("nat_cache_pinned", 1.1),
                "llama.cpp cache, default (mmap)": ("nat_cache", 1.1), "llama.cpp static, --fit": ("nat_fit", 1.5),
                "llama.cpp static, tuned --n-cpu-moe 16": ("nat_ncmoe16", 1.5)},
    "capped": {"FreeToken": ("cap_ft_auto", 1.0), "llama.cpp cache, pinned": ("cap_cache_pinned", 1.1),
               "llama.cpp cache, default (mmap)": ("cap_cache", 1.1), "llama.cpp static, --fit": ("cap_fit", 1.5),
               "llama.cpp static, --n-cpu-moe 36": ("cap_ncmoe36", 1.5)},
}


def line(med: dict, c: str) -> tuple[float, float]:
    """(intercept ms, ms per MB) through the static (id-order) and profiled placements at 47%."""
    (x0, y0), (x1, y1) = [(med[a][c]["miss_mb_per_token"]["median"], 1e3 / med[a][c]["decode_tps"]["median"])
                          for a in ("holon_static", "holon_prof")]
    slope = (y1 - y0) / (x1 - x0)
    return y1 - slope * x1, slope


def predict(base: float, slope: float, mb: dict, refs: dict) -> dict:
    out = {}
    for fit, mb_fit in mb.items():
        f11 = F11_MS[fit]
        ms = base + f11 + slope * mb_fit
        lo = base + f11 + F11_BAND + slope * mb_fit * (1 + BYTES_BAND)      # slow edge
        hi = base + f11 - F11_BAND + slope * mb_fit * (1 - BYTES_BAND)      # fast edge
        rows_ = {}
        for name, (arm, bar) in REFS[fit].items():
            t = refs[arm]
            r = [round(1e3 / x / t, 3) for x in (ms, lo, hi)]
            rows_[name] = {"ref_tps": t, "bar": bar, "ratio": r[0], "band": r[1:],
                           "verdict": "pass" if r[1] >= bar else "miss" if r[2] < bar else "straddles"}
        out[fit] = {"miss_mb_per_token": mb_fit, "ms_per_token": round(ms, 3), "tps": round(1e3 / ms, 1),
                    "rows": rows_}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--commit", required=True)
    a = ap.parse_args()
    runs, ft_runs = pathlib.Path(a.runs) / "x4_32k", pathlib.Path(a.runs) / "x4_32k_ft"
    gap = json.loads((ROOT / "evidence" / "x4_gap_rtx-5080_2026-10-17.json").read_text())
    refs_rec = json.loads((ROOT / "evidence" / "x4_refs_rtx-5080_2026-10-17.json").read_text())
    refs = {c: {k: v[c]["decode_tps"]["median"] for k, v in refs_rec["summary"].items()} for c in ("1024", "4096",
                                                                                                    "32768")}
    med = medians(rows(runs))
    ft = medians(rows(ft_runs))
    replay = json.loads((runs / "replay.json").read_text())

    def lfu(rec: dict, arm: str, c: str) -> float:
        return rec[f"r0_{arm}@{c}"]["policies"]["lfu"]["fetch_mb_per_token"]

    def ft_mb(c: str) -> dict:
        return {fit: round(ft[arm][c]["ft_missing_per_layer"]["median"] * LAYERS * EXPERT_MB, 1)
                for fit, arm in (("natural", "ft_nat"), ("capped", "ft_cap"))}

    pred, at_ft, fits = {}, {}, {}
    for c in ("1024", "4096"):                                  # x4_gap's record, as the §13 re-band used it
        f = gap["decode_fit"][c]
        base, slope = f["ms_per_token_at_0_mb"], f["ms_per_mb"]
        pred[c] = predict(base, slope, {"natural": lfu(gap["replay"], "holon_prof", c),
                                        "capped": lfu(gap["replay"], "holon_prof25", c)}, refs[c])
        at_ft[c] = predict(base, slope, {"natural": f["freetoken_miss_mb_per_token"],
                                         "capped": f["freetoken_capped_miss_mb_per_token"]}, refs[c])
    for c in ("1024", "4096", "32768"):                         # the reruns (1K / 4K: the reproduction check)
        base, slope = line(med, c)
        p25 = med["holon_prof25"][c]
        fits[c] = {"ms_per_token_at_0_mb": round(base, 3), "ms_per_mb": round(slope, 5),
                   "effective_gather_gb_s": round(1 / slope, 1),
                   "prof25_measured_ms": round(1e3 / p25["decode_tps"]["median"], 3),
                   "prof25_on_the_line_ms": round(base + slope * p25["miss_mb_per_token"]["median"], 3),
                   "lfu_mb_per_token": {"natural": lfu(replay, "holon_prof", c),
                                        "capped": lfu(replay, "holon_prof25", c)}}
    f = fits["32768"]
    pred["32768"] = predict(f["ms_per_token_at_0_mb"], f["ms_per_mb"], f["lfu_mb_per_token"], refs["32768"])
    at_ft["32768"] = predict(f["ms_per_token_at_0_mb"], f["ms_per_mb"], ft_mb("32768"), refs["32768"])
    rerun = {c: predict(fits[c]["ms_per_token_at_0_mb"], fits[c]["ms_per_mb"], fits[c]["lfu_mb_per_token"], refs[c])
             for c in ("1024", "4096")}
    arms = [json.loads((d / "arms.json").read_text()) for d in (runs, ft_runs)]
    record = {
        "name": "x4_32k", "gpu": "rtx-5080", "date": a.date, "commit": a.commit,
        "model": gap["model"], "engines": {"Holon": f"the v1-offload branch at {a.commit} (step 2's offload store, "
                                                     "static placement, F5a; the same code x4_gap measured)",
                                           "references": "x4_refs' medians (llama.cpp f74e2fa, FreeToken 21d9578)"},
        "method": "spikes/x4_gap/README_32k.md: ctx.py (x4_gap's driver) with contexts 1024 4096 32768 on x4_gap's "
                  "Holon arms (static id-order and profiled seeds at 47%, profiled at 25%), interleaved rounds, "
                  "medians; replay.py on the round-0 routing traces; predicted time = line intercept + F11 + slope × "
                  "lfu replay MB; bands: F11 ± 0.04 ms and miss bytes ± 10%; FreeToken's decode miss bytes from "
                  "its counters (moe_collect_stats) × 2.857 MB, one round per fit",
        "commands": ["python spikes/x4_gap/ctx.py run --name x4_32k --arms <arms.json> --rounds 2 --model <gguf> "
                     "--contexts 1024 4096 32768 --ctx 33792",
                     "python spikes/x4_gap/replay.py <out>/x4_32k/r0_holon_{static,prof,prof25}.npz "
                     "> <out>/x4_32k/replay.json",
                     "python spikes/x4_gap/ctx.py run --name x4_32k_ft --arms <ft arms> --rounds 1 --model <gguf> "
                     "--contexts 1024 4096 32768 --ctx 33792 --python <freetoken venv python>",
                     "python spikes/x4_gap/predict_32k.py --runs <out> --date <date> --commit <commit>"],
        "assumptions": {"f11_ms": F11_MS, "f11_band_ms": F11_BAND, "miss_bytes_band": BYTES_BAND,
                        "capped_line": "the 47% line applied to the 25% replay bytes (x4_gap's method); "
                                       "prof25's measured point checks it"},
        "arms": arms, "summary": med, "freetoken_summary": ft, "replay": replay, "decode_fit": fits,
        "freetoken_miss_mb_per_token": {c: ft_mb(c) for c in ("1024", "4096", "32768")},
        "predictions": pred, "at_freetoken_miss_bytes": at_ft, "predictions_from_reruns_1k_4k": rerun,
    }
    out = ROOT / "evidence" / f"x4_32k_rtx-5080_{a.date}.json"
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(out)
    for name, table in (("lfu", pred), ("ft bytes", at_ft), ("rerun", rerun)):
        for c, p in table.items():
            for fit, v in p.items():
                print(name, c, fit, v["miss_mb_per_token"], v["tps"],
                      {k: (r["ratio"], r["band"], r["verdict"]) for k, r in v["rows"].items()})
    print(json.dumps(fits))


if __name__ == "__main__":
    main()
