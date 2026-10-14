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


def load(directory: str, cls: str, what: str) -> dict[str, dict]:
    out = {}
    for path in sorted(glob.glob(os.path.join(directory, f"m07-step0_{what}*_{cls}_*.json"))):
        name = os.path.basename(path).split("_")[1].removeprefix(what).lstrip("-") or what
        with open(path) as f:
            out[name] = json.load(f)
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


def candidates(profiles: dict[str, dict]) -> list[dict]:
    """Per (model, row): each fusion candidate's estimated saving (ms, share of the step). Estimates:
    kv_write = the KV store's time beyond one launch per attention layer (its two index_puts become one);
    qk_rope_kv (D3, on top) = q/k norm + rope; mmq bf16 (D4) = the f32 -> bf16 casts; gate/up + SwiGLU (D6, one
    column) = silu·mul at one row; shared q8_1 (D6) = act quant x the share of quantizations of an input already
    quantized (decode / verify steps); GDN gating (D6, prefill) = the gates' time on prefill rows of a GDN model;
    rewind glue (D5) = the rewind region's time other than the delta rule and conv kernels (eager verify trace,
    scaled by the graphed / eager kernel time)."""
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
            if row["kind"] != "prefill":
                if row.get("batch") == 1 and row["kind"] == "decode":
                    c["gate/up + SwiGLU (D6)"] = cat.get("silu*mul", 0.0)
                g = row.get("gemm", {})
                calls = sum(x["count"] for x in g.get("calls", []))
                if calls and g.get("inputs"):
                    c["shared q8_1 (D6)"] = cat.get("act quant (q8_1 / fp4)", 0.0) * (1 - g["inputs"] / calls)
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


def d8(profile: dict, sweep: dict | None) -> dict:
    """D8's f on decode B1: bytes = the matmuls' weight bytes + the KV read; B = weight bytes / unprofiled GEMV time
    (the profiled time over the profiled-over-wall ratio); f = 1 - bytes / (B x step)."""
    row = profile["rows"]["decode_b1"]
    r2 = row["ratio_graphed_profiled_over_wall"]
    gemv_ms = row["graphed"]["cat"]["gemm"][1] / r2
    weights = row["gemm"]["weight_bytes"]
    ctx = (row["ctx"] + row.get("ctx_end", row["ctx"])) / 2
    kv = profile["kv_bytes_per_token"] * ctx
    bw = weights / (gemv_ms / 1e3)
    f = 1 - (weights + kv) / (bw * row["graphed_wall_ms"] / 1e3)
    return {"step_ms": row["graphed_wall_ms"], "weights_gb": round(weights / 1e9, 4), "kv_gb": round(kv / 1e9, 5),
            "gemv_ms_unprofiled": round(gemv_ms, 3), "gemv_gb_s": round(bw / 1e9, 1),
            "stream_gb_s": sweep["stream"]["gb_s_best"] if sweep else None, "f": round(f, 4),
            "projection": {c: round(1 / (1 - f * c), 3) for c in (0.4, 0.55)},
            "gate": "stop" if 1 / (1 - f * 0.55) < 1.15 else "prototype"}


def crossover(sweep: dict, profiles: dict[str, dict]) -> dict:
    """Per (type, mode): llama.cpp's mmvq row limit here, and the limit a per-class table would take — mmvq for every
    n up to it — from the shapes' per-call times summed with their per-step counts (today's cast; and without it,
    D4), rows 1-8. Then each profiled decode / verify step's saving under that table (rows at most 9)."""
    weight = collections.Counter()
    for rec in profiles.values():
        for row in rec["rows"].values():
            if row["kind"] != "prefill":
                for c in row.get("gemm", {}).get("calls", []):
                    key = (c["type"], c["in"], c["out"], c["mode"])
                    weight[key] = max(weight[key], c["count"])
    shapes = {(s["type"], s["in"], s["out"], s["mode"]): s for s in sweep["shapes"]}
    table = {}
    for t, mode in sorted({(s["type"], s["mode"]) for s in sweep["shapes"]}):
        group = [s for k, s in shapes.items() if k[0] == t and k[3] == mode]
        w = {id(s): weight.get((s["type"], s["in"], s["out"], s["mode"]), 1) for s in group}
        entry = {"llama": group[0]["llama_mmvq_rows"], "shapes": len(group)}
        for variant in ("mmq_cast", "mmq_f32"):
            limit = 0
            for n in range(1, 9):
                vq = sum(w[id(s)] * s["us"][str(n)]["mmvq"] for s in group)
                mq = sum(w[id(s)] * s["us"][str(n)][variant] for s in group)
                if vq > mq:
                    break
                limit = n
            entry["best_with_cast" if variant == "mmq_cast" else "best_no_cast"] = limit
        table[f"{t}/{mode}"] = entry

    def call_us(key, rows, limit, variant):
        s = shapes.get(key)
        if s is None or str(rows) not in s["us"]:
            return None
        u = s["us"][str(rows)]
        return u["mmvq"] if rows <= limit else u[variant]

    steps = []
    for name, rec in profiles.items():
        for rk, row in rec["rows"].items():
            if row["kind"] == "prefill":
                continue
            today = d2 = d2d4 = 0.0
            for c in row.get("gemm", {}).get("calls", []):
                key, e = (c["type"], c["in"], c["out"], c["mode"]), table.get(f"{c['type']}/{c['mode']}")
                if e is None:
                    continue
                a = call_us(key, c["rows"], e["llama"], "mmq_cast")
                b = call_us(key, c["rows"], e["best_with_cast"], "mmq_cast")
                d = call_us(key, c["rows"], e["best_no_cast"], "mmq_f32")
                if None in (a, b, d):
                    continue
                today, d2, d2d4 = today + c["count"] * a, d2 + c["count"] * b, d2d4 + c["count"] * d
            if today:
                step = row["graphed_wall_ms"]
                steps.append({"model": name, "row": rk, "step_ms": step,
                              "gemm_today_ms": round(today / 1e3, 3), "d2_saving_ms": round((today - d2) / 1e3, 3),
                              "d2_share": round((today - d2) / 1e3 / step, 4),
                              "d2_after_d4_share": round((today - d2d4) / 1e3 / step, 4)})
    return {"table": table, "steps": steps}


def d9(sweep: dict, profile: dict, table: dict) -> dict:
    """The target's 4-row / 1-row matmul time over its decode step's calls, at D2's table (with today's cast)."""
    one = four = 0.0
    shapes = {(s["type"], s["in"], s["out"], s["mode"]): s for s in sweep["shapes"]}
    for c in profile["rows"]["decode_b1"]["gemm"]["calls"]:
        s = shapes.get((c["type"], c["in"], c["out"], c["mode"]))
        if s is None:
            continue
        limit = table[f"{c['type']}/{c['mode']}"]["best_with_cast"]
        one += c["count"] * s["us"]["1"]["mmvq"]
        four += c["count"] * (s["us"]["4"]["mmvq"] if 4 <= limit else s["us"]["4"]["mmq_cast"])
    return {"ratio_4_over_1": round(four / one, 3), "trigger": four / one > 1.25}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--class", dest="cls", required=True)
    ap.add_argument("--dir", default="evidence")
    ap.add_argument("--json")
    args = ap.parse_args()
    profiles = load(args.dir, args.cls, "profile")
    sweep = next(iter(load(args.dir, args.cls, "qmm-sweep").values()), None)
    out = {"class": args.cls, "glue": glue(profiles), "candidates": candidates(profiles), "ratios": ratios(profiles)}
    if "8b-q4_k_m" in profiles:
        out["d8"] = d8(profiles["8b-q4_k_m"], sweep)
    if sweep:
        out["crossover"] = crossover(sweep, profiles)
        target = next((p for n, p in profiles.items() if n == "27b-plain"), None)
        if target:
            out["d9"] = d9(sweep, target, out["crossover"]["table"])
    print(json.dumps(out, indent=1))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
