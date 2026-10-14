"""M0.7 step 0 (docs/m07-perf.md D1, §11): the tables of §11 derived from one class's step-0 records — glue share per
row, the re-stated §5 candidates (saving as a share of their step), D1's timing ratios, D8's f, the crossover table
(D2) and D9's trigger. Every number is computed here from the records; the method of each estimate is in §11.

    python bench/step0_report.py --class rtx-5090 [--dir evidence] [--json out]

Records read (`<dir>/m07-step0_<what>_<class>_<date>.json`): `profile-<name>` (step_profile.py), `qmm-sweep`
(qmm_sweep.py); the shared-quantization counts come from the profiles' `gemm.inputs`.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os

GROUPS = {                     # report column -> step_profile categories
    "GEMM": ["gemm"],
    "attention": ["attention"],
    "act quant": ["act quant (q8_1 / fp4)"],
    "cast f32→bf16": ["f32 -> bf16 cast"],
    "copy bf16→f32": ["bf16 -> f32 copy"],
    "KV store": ["kv store"],
    "q/k norm + rope": ["q/k norm", "rope"],
    "norms": ["add+rmsnorm (fused)", "rmsnorm"],
    "silu·mul": ["silu*mul"],
    "residual add": ["residual add"],
    "gates": ["gates (sigmoid, mul, softplus, exp)"],
    "GDN core + conv": ["gdn core", "gdn conv"],
}
DROP = 0.01                    # a candidate under 1% of its step is dropped (D1)
# Decode steps (one token each: valid for decode rows only) a row ran before the middle of its graphed timing window
# (step_profile.py: eager warm-up 2, counter 1, timed 32, traced 32, graphed warm-up 3, half of the timed 32): the
# context of its graphed wall, for records made before `graphed_ctx` was recorded.
GRAPHED_AT = 86
MMVQ_MAX = 8                   # mmvq's most rows (llama.cpp's MMVQ_MAX_BATCH_SIZE)


def graphed_ctx(row: dict) -> float:
    return row.get("graphed_ctx", row["ctx"] + GRAPHED_AT)


def load(directory: str, cls: str, what: str, exact: bool = False) -> dict[str, dict]:
    """`{name: record}` of `m07-step0_<what>-<name>_<cls>_<date>.json` (`exact`: of `m07-step0_<what>_...` only)."""
    out = {}
    for path in sorted(glob.glob(os.path.join(directory, f"m07-step0_{what}*_{cls}_*.json"))):
        stem = os.path.basename(path).removeprefix("m07-step0_").rsplit("_", 2)[0]
        if stem != what if exact else not stem.startswith(what + "-"):
            continue
        with open(path) as f:
            out[stem.removeprefix(what).lstrip("-") or what] = json.load(f)
    return out


def redundancy(directory: str) -> dict[str, float]:
    """Model id -> the share of a decode step's quantizations whose input was already quantized (D6's shared
    quantization), from any class's profile that counted inputs (the count is the model's, not the class's)."""
    out = {}
    for path in glob.glob(os.path.join(directory, "m07-step0_profile-*.json")):
        with open(path) as f:
            rec = json.load(f)
        g = rec["rows"].get("decode_b1", {}).get("gemm", {})
        calls = sum(c["count"] for c in g.get("calls", []))
        if calls and g.get("inputs"):
            out[rec["model"]] = 1 - g["inputs"] / calls
    return out


def step_view(row: dict) -> tuple[float, dict, float]:
    """(step ms, category ms scaled to unprofiled time, profile inflation) of a row: graphed for decode and verify
    (category ms / the profiled-over-wall ratio), eager for prefill (its wall includes the host's idle)."""
    if row["kind"] == "prefill":
        return row["wall_ms"], {c: v[1] for c, v in row["eager"]["cat"].items()}, 1.0
    r = row["ratio_graphed_profiled_over_wall"]
    return row["graphed_wall_ms"], {c: v[1] / r for c, v in row["graphed"]["cat"].items()}, r


def glue(profiles: dict[str, dict]) -> list[dict]:
    rows = []
    for name, rec in profiles.items():
        for key, row in rec["rows"].items():
            step, cat, _ = step_view(row)
            shares = {g: sum(cat.get(c, 0.0) for c in cs) / step for g, cs in GROUPS.items()}
            other = 1 - sum(shares.values())
            rows.append({"model": name, "row": key, "step_ms": round(step, 3), **{g: round(v, 4) for g, v in
                                                                                   shares.items()},
                         "other": round(other, 4)})
    return rows


def candidates(profiles: dict[str, dict], shared: dict[str, float]) -> list[dict]:
    """Per (model, row): each fusion candidate's estimated saving (ms, share of the step). Estimates:
    - kv_write: the KV store's time beyond one launch per attention layer (bf16 KV: two index_puts become one);
    - qk_rope_kv (D3, on top of kv_write): q/k norm + rope;
    - mmq bf16 out (D4): the f32 -> bf16 casts, and NVFP4's f32 x scale2 -> bf16 multiplies (the "gates" of a model
      without GDN) where its matmuls run mmq; where they run mmvq above one row the multiplies are a new candidate;
      on GDN prefill rows the casts include some outside mmq (an upper bound);
    - fp4 activations read as bf16 (not in §5): native W4A4's bf16 -> f32 activation copies;
    - gate/up + SwiGLU (D6, one column): silu·mul on decode B1;
    - shared q8_1 (D6): act quant x the share of the step's quantizations whose input was already quantized
      (decode / verify; the model's share, from whichever class's profile counted it);
    - GDN gating (D6, prefill): the gates' time on a GDN model's prefill rows;
    - rewind glue (D5): the rewind region's time other than the delta rule and conv (eager verify trace, scaled
      to the graphed step)."""
    out = []
    for name, rec in profiles.items():
        for key, row in rec["rows"].items():
            step, cat, _ = step_view(row)
            src = row["eager"] if row["kind"] == "prefill" else row["graphed"]
            kv_n = src["cat"].get("kv store", [0, 0])[0]
            attn_layers = src["cat"].get("q/k norm", [0, 0])[0] / 2 or None
            c = {}
            kv = cat.get("kv store", 0.0)
            bf16_kv = "kv.format" not in rec["engine"].get("plugin_options", {})     # q8_0 KV: one launch already
            if bf16_kv and attn_layers and kv_n > attn_layers:
                c["kv_write (bf16)"] = kv * (1 - attn_layers / kv_n)
            c["qk_rope_kv (D3)"] = cat.get("q/k norm", 0.0) + cat.get("rope", 0.0)
            c["mmq bf16 out (D4)"] = cat.get("f32 -> bf16 cast", 0.0)
            scale2 = 0.0 if cat.get("gdn core") else cat.get("gates (sigmoid, mul, softplus, exp)", 0.0)
            nvfp4_rows = [x["rows"] for x in row.get("gemm", {}).get("calls", []) if x["type"] == "nvfp4"]
            if scale2 and nvfp4_rows:                      # no GDN: the gates are NVFP4's f32 x scale2 -> bf16 writes
                if row["kind"] == "prefill" or max(nvfp4_rows, default=0) > MMVQ_MAX:
                    c["mmq bf16 out (D4)"] += scale2       # mmq's write-back
                else:                                      # mmvq above one row: D4 does not reach it
                    c["NVFP4 scale2 in mmvq above one row (new)"] = scale2
            if cat.get("bf16 -> f32 copy"):
                c["fp4 activations read as bf16 (new)"] = cat["bf16 -> f32 copy"]
            if row["kind"] != "prefill":
                if row.get("batch") == 1 and row["kind"] == "decode":
                    c["gate/up + SwiGLU (D6)"] = cat.get("silu*mul", 0.0)
                if rec["model"] in shared:
                    c["shared q8_1 (D6)"] = cat.get("act quant (q8_1 / fp4)", 0.0) * shared[rec["model"]]
            elif cat.get("gdn core"):
                c["GDN gating (D6)"] = cat.get("gates (sigmoid, mul, softplus, exp)", 0.0)
            if "rewind" in row.get("regions", {}):
                rw = row["regions"]["rewind"]
                glue_ms = rw["ms"] - rw["cat"].get("gdn core", 0) - rw["cat"].get("gdn conv", 0)
                c["rewind glue (D5)"] = glue_ms * row["graphed"]["kernel_ms"] / row["eager"]["kernel_ms"] / \
                    row["ratio_graphed_profiled_over_wall"]
            for cand, ms in c.items():
                if ms > 0:
                    out.append({"model": name, "row": key, "candidate": cand, "ms": round(ms, 4),
                                "share": round(ms / step, 4)})
    return out


def ratios(profiles: dict[str, dict]) -> list[dict]:
    out = []
    for name, rec in profiles.items():
        for key, row in rec["rows"].items():
            if row["kind"] == "prefill":
                continue
            clocks = row.get("clocks", {})
            out.append({"model": name, "row": key, "graphed_wall_ms": row["graphed_wall_ms"],
                        "eager_kernel_ms": row["eager"]["kernel_ms"],
                        "r1_graphed_wall_over_eager_kernels": row["ratio_graphed_wall_over_eager_kernels"],
                        "r2_graphed_profiled_over_wall": row["ratio_graphed_profiled_over_wall"],
                        "sm_mhz_graphed": clocks.get("graphed", {}).get("sm_mhz", {}).get("p50"),
                        "sm_mhz_eager": clocks.get("eager", {}).get("sm_mhz", {}).get("p50"),
                        "power_w_graphed": clocks.get("graphed", {}).get("power_w"),
                        "power_w_eager": clocks.get("eager", {}).get("power_w"),
                        "slowdown_graphed": clocks.get("graphed", {}).get("slowdown_share")})
    return out


def shapes_of(sweep: dict) -> dict[tuple, dict]:
    return {(x["type"], x["in"], x["out"], x["mode"]): x for x in sweep["shapes"]}


def d8(profile: dict, sweep: dict | None) -> dict:
    """D8's f on decode B1: bytes = the matmuls' weight bytes + the KV read (at the graphed window's context); B =
    weight bytes / the step's GEMV time: its calls priced from the per-call sweep (graph-timed, unprofiled), less the
    activation quantization each sweep call includes (the step's profiled q8_1 quantize time), so B is the GEMV
    kernels' alone; f = 1 - bytes / (B x step). The profile's GEMV time is kept beside it: profiled kernels overlap
    under PDL (r2 > 1 is overlap, not inflation), so the profile alone can't separate a GEMV from its neighbours."""
    row = profile["rows"]["decode_b1"]
    weights = row["gemm"]["weight_bytes"]
    kv = profile["kv_bytes_per_token"] * graphed_ctx(row)
    shapes = shapes_of(sweep) if sweep else {}
    calls = row["gemm"]["calls"]
    priced = [(c, shapes.get((c["type"], c["in"], c["out"], c["mode"]))) for c in calls]
    quantize_ms = row["graphed"]["cat"].get("act quant (q8_1 / fp4)", [0, 0.0])[1]
    gemv_ms = sum(c["count"] * s["us"][str(c["rows"])]["llama"] for c, s in priced if s) / 1e3 - quantize_ms
    out = {"step_ms": row["graphed_wall_ms"], "weights_gb": round(weights / 1e9, 4), "kv_gb": round(kv / 1e9, 5),
           "gemv_ms_profiled": round(row["graphed"]["cat"]["gemm"][1], 3),
           "quantize_ms_profiled": round(quantize_ms, 3),
           "stream_gb_s": sweep["stream"]["gb_s_best"] if sweep else None}
    if not sweep or not all(s for _, s in priced):
        return out | {"f": None}
    bw = weights / (gemv_ms / 1e3)
    f = 1 - (weights + kv) / (bw * row["graphed_wall_ms"] / 1e3)
    return out | {"gemv_ms_sweep": round(gemv_ms, 3), "gemv_gb_s": round(bw / 1e9, 1), "f": round(f, 4),
                  "projection": {c: round(1 / (1 - f * c), 3) for c in (0.4, 0.55)},
                  "gate": "stop" if 1 / (1 - f * 0.55) < 1.15 else "prototype"}


GRAPHED_ROWS = (4, 7, 8)       # graphed row counts where mmvq and mmq compete: buckets 4 and 8, the 7-row draft
MARGIN = 0.03                  # an entry needs mmq faster than mmvq by at least this, per call


def entry(shape: dict, variant: str) -> int | None:
    """D2's per-shape limit (mmvq up to it, mmq above), or None (llama.cpp's rule): the smallest limit in {3, 6, 7}
    below llama.cpp's for which every graphed row it moves to mmq is at least MARGIN faster there."""
    llama = shape["llama_mmvq_rows"]
    for limit in (3, 6, 7):
        moved = [r for r in GRAPHED_ROWS if limit < r <= llama]
        if limit < llama and moved and all(
                shape["us"][str(r)][variant] <= (1 - MARGIN) * shape["us"][str(r)]["mmvq"] for r in moved):
            return limit
    return None


def crossover(sweep: dict, profiles: dict[str, dict]) -> dict:
    """D2 keyed per shape (type, in, out, mode): each shape's entry (`entry`, with today's cast and without it, D4),
    then each profiled decode / verify step's saving under the entries against llama.cpp's rule, per entry (calls of
    more than 9 rows are mmq either way: `partial` marks rows that had some)."""
    shapes = shapes_of(sweep)
    table = {f"{k[0]}/{k[1]}x{k[2]}/{k[3]}": {"llama": s["llama_mmvq_rows"], "with_cast": entry(s, "mmq_cast"),
                                              "no_cast": entry(s, "mmq_f32")} for k, s in shapes.items()}
    table = {k: v for k, v in table.items() if v["with_cast"] is not None or v["no_cast"] is not None}

    def call_us(s: dict, rows: int, limit: int, variant: str) -> float:
        u = s["us"][str(rows)]
        return u["mmvq"] if rows <= limit else u[variant]

    steps = []
    for name, rec in profiles.items():
        for rk, row in rec["rows"].items():
            if row["kind"] == "prefill":
                continue
            today = d2 = d4 = d2d4 = 0.0
            skipped = 0
            by_entry: collections.Counter = collections.Counter()
            for c in row.get("gemm", {}).get("calls", []):
                key = (c["type"], c["in"], c["out"], c["mode"])
                s = shapes.get(key)
                if s is None or str(c["rows"]) not in s["us"]:
                    skipped += c["count"]
                    continue
                e = table.get(f"{key[0]}/{key[1]}x{key[2]}/{key[3]}", {})
                lim, lim4 = e.get("with_cast") or s["llama_mmvq_rows"], e.get("no_cast") or s["llama_mmvq_rows"]
                a = call_us(s, c["rows"], s["llama_mmvq_rows"], "mmq_cast")
                b = call_us(s, c["rows"], lim, "mmq_cast")
                f = call_us(s, c["rows"], s["llama_mmvq_rows"], "mmq_f32")
                d = call_us(s, c["rows"], lim4, "mmq_f32")
                today, d2, d4, d2d4 = (today + c["count"] * a, d2 + c["count"] * b, d4 + c["count"] * f,
                                       d2d4 + c["count"] * d)
                if a > b:
                    by_entry[f"{key[0]}/{key[1]}x{key[2]}"] += c["count"] * (a - b)
            if today:
                step = row["graphed_wall_ms"]
                steps.append({"model": name, "row": rk, "step_ms": step, "d2_saving_ms": round((today - d2) / 1e3, 3),
                              "d2_share": round((today - d2) / 1e3 / step, 4),
                              "d2_after_d4_share": round((d4 - d2d4) / 1e3 / step, 4), "partial": skipped > 0,
                              "d2_share_by_entry": {k: round(v / 1e3 / step, 4) for k, v in by_entry.items()}})
    shipped = sorted({k for x in steps if x["d2_share"] >= DROP for k in x["d2_share_by_entry"]})
    return {"table": table, "steps": steps, "shipped": shipped}


def _mean(found: list[dict], rows: str, variant: str) -> float:
    return sum(x["us"][rows][variant] for x in found) / len(found)


def d9(sweeps: list[dict], profile: dict, table: dict) -> dict:
    """The target's 4-row / 1-row matmul time over its decode step's calls at the D2 entries that ship (with today's
    cast), each call's time the mean over the sweeps given; and the upside in ms: what its 4-row calls cost beyond
    their 1-row time (what a small-batch matmul could recover at most)."""
    one = four = 0.0
    maps = [shapes_of(sw) for sw in sweeps]
    for c in profile["rows"]["decode_b1"]["gemm"]["calls"]:
        key = (c["type"], c["in"], c["out"], c["mode"])
        found = [m[key] for m in maps if key in m]
        if not found:
            continue
        e = table.get(f"{key[0]}/{key[1]}x{key[2]}/{key[3]}", {})
        limit = e.get("with_cast") or found[0]["llama_mmvq_rows"]
        one += c["count"] * _mean(found, "1", "mmvq")
        four += c["count"] * _mean(found, "4", "mmvq" if 4 <= limit else "mmq_cast")
    return {"ratio_4_over_1": round(four / one, 4), "trigger": four / one > 1.25, "sweeps": len(sweeps),
            "upside_ms": round((four - one) / 1e3, 2)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--class", dest="cls", required=True)
    ap.add_argument("--dir", default="evidence")
    ap.add_argument("--json")
    args = ap.parse_args()
    profiles = load(args.dir, args.cls, "profile")
    sweep = next(iter(load(args.dir, args.cls, "qmm-sweep", exact=True).values()), None)
    out = {"class": args.cls, "glue": glue(profiles), "candidates": candidates(profiles, redundancy(args.dir)),
           "ratios": ratios(profiles)}
    if "8b-q4_k_m" in profiles:
        out["d8"] = d8(profiles["8b-q4_k_m"], sweep)
    if sweep:
        out["crossover"] = crossover(sweep, profiles)
        target = next((p for n, p in profiles.items() if n == "27b-plain"), None)
        repeats = next(iter(load(args.dir, args.cls, "qmm-sweep-d9-repeat", exact=True).values()), {})
        if target:
            shipped = set(out["crossover"]["shipped"])          # D9 at the entries that ship, not every candidate
            table = {k: v for k, v in out["crossover"]["table"].items() if k.rsplit("/", 1)[0] in shipped}
            out["d9"] = d9([sweep, *repeats.get("repeats", [])], target, table)
    print(json.dumps(out, indent=1))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
