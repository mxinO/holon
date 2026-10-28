"""M0.7 step 5 (docs/dev/m07-perf.md D8, §16): the megakernel gate's f on one class, profile-free, from paired rounds of
records — `step_profile.py` (8B Q4_K_M, decode B1, graphed), `qmm_sweep.py` (its locks' shapes, with the isolated q8_1
quantize per `in`) and `glu_sweep.py` (the fused gate/up call).

- bytes = the weights the step's matmuls read (the fused gate/up's whole stack included; the head's too) + the
  embedding rows it gathers + the KV read at the graphed window's context;
- GEMV time = each call of the step priced from the per-call sweep (graph-timed, unprofiled: mmvq up to the kernel's
  `mmvq_rows`, else mmq; the fused gate/up from the glu sweep) less the isolated, graph-timed q8_1 quantize of its
  `in` and rows, which every swept call includes: the GEMV kernels' time alone;
- B = weight bytes / GEMV time; f = 1 - bytes / (B x step), step the unprofiled graphed wall;
- projection 1 / (1 - f c) at c = 0.4 and 0.55; the gate stops under 1.15x at c = 0.55 (f < 24%).

f is computed per round (the i-th records of each list) and over the rounds' means.

    python bench/d8_gate.py --profiles p1.json ... --sweeps s1.json ... --glu g1.json ... [--json out]
"""
from __future__ import annotations

import argparse
import json
import statistics

from step0_report import graphed_ctx, shapes_of

C = (0.4, 0.55)
GATE = 1.15                    # the projection at c = 0.55 below which D8 stops


def quantize_us(sweep: dict) -> dict[tuple[int, int], float]:
    return {(q["in"], int(r)): us for q in sweep["quantize"] for r, us in q["us"].items()}


def call_us(call: dict, shapes: dict, glu: dict) -> float:
    """A step's call as swept (its activation quantize included)."""
    if call.get("glu"):
        return glu[(call["type"], call["in"], call["out"])]["us"]["fused"]
    s, rows = shapes[(call["type"], call["in"], call["out"], call["mode"])], call["rows"]
    return s["us"][str(rows)]["mmvq" if rows <= s["mmvq_rows"] else "mmq_cast"]


def one(profile: dict, sweep: dict, glu_sweep: dict) -> dict:
    row = profile["rows"]["decode_b1"]
    shapes, glu, quant = shapes_of(sweep), {(g["type"], g["in"], g["out"]): g for g in glu_sweep["shapes"]}, \
        quantize_us(sweep)
    calls = row["gemm"]["calls"]
    swept = sum(c["count"] * call_us(c, shapes, glu) for c in calls) / 1e3
    quantize = sum(c["count"] * quant[(c["in"], c["rows"])] for c in calls) / 1e3
    weights, embedding = row["gemm"]["weight_bytes"], row["gemm"]["embedding_bytes"]
    kv = profile["kv_bytes_per_token"] * graphed_ctx(row)
    return {"step_ms": row["graphed_wall_ms"], "graphed_ctx": graphed_ctx(row), "weights_gb": weights / 1e9,
            "embedding_bytes": embedding, "kv_gb": kv / 1e9, "swept_ms": swept, "quantize_ms": quantize,
            "gemv_ms": swept - quantize, "gemv_ms_profiled": row["graphed"]["cat"]["gemm"][1],
            "quantize_ms_profiled": row["graphed"]["cat"].get("act quant (q8_1 / fp4)", [0, 0.0])[1],
            "calls": sum(c["count"] for c in calls), "stream_gb_s": sweep["stream"]["gb_s_best"]}


def gate(r: dict) -> dict:
    bw = r["weights_gb"] / (r["gemv_ms"] / 1e3)
    f = 1 - (r["weights_gb"] + r["embedding_bytes"] / 1e9 + r["kv_gb"]) / (bw * r["step_ms"] / 1e3)
    proj = {c: 1 / (1 - f * c) for c in C}
    return {"gemv_gb_s": round(bw, 1), "f": round(f, 4), "projection": {c: round(p, 3) for c, p in proj.items()},
            "gate": "stop" if proj[0.55] < GATE else "prototype"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profiles", nargs="+", required=True)
    ap.add_argument("--sweeps", nargs="+", required=True)
    ap.add_argument("--glu", nargs="+", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    if not len(a.profiles) == len(a.sweeps) == len(a.glu):
        ap.error("one profile, sweep and glu sweep per round")
    load = lambda p: json.load(open(p))                          # noqa: E731
    rounds = [one(load(p), load(s), load(g)) for p, s, g in zip(a.profiles, a.sweeps, a.glu, strict=True)]
    mean = {k: statistics.mean(r[k] for r in rounds) for k in rounds[0]}
    out = {"gpu": load(a.profiles[0])["gpu"], "rounds": [r | gate(r) for r in rounds], "mean": mean | gate(mean),
           "f_sd": round(statistics.stdev(gate(r)["f"] for r in rounds), 4) if len(rounds) > 1 else None}
    print(json.dumps(out, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
