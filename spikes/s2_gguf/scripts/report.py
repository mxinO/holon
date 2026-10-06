"""Join llama.cpp and our per-kernel numbers -> results/compare_sm*.json + markdown tables on stdout.

usage: python scripts/report.py <sm>
Ratios are llama_us / ours_us (>= 1: we are faster; the M0.1/M0.2 bar is >= 0.8).
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
TYPES = ["IQ3_XXS", "IQ3_S", "IQ4_XS", "Q3_K", "Q4_K", "Q8_0"]
NS = [1, 8, 32, 512]
# per-layer linear ops: (kind, m, k, count)
LAYERS = {
    "Qwen3-8B": [("dense", 4096, 4096, 2), ("dense", 1024, 4096, 2), ("dense", 12288, 4096, 2),
                 ("dense", 4096, 12288, 1)],
    "Qwen3-30B-A3B": [("dense", 4096, 2048, 1), ("dense", 512, 2048, 2), ("dense", 2048, 4096, 1),
                      ("moe_gate_up", 768, 2048, 2), ("moe_down", 2048, 768, 1)],
}


def load_llama(sm: str) -> dict:
    out = {}
    for line in (HERE / "results" / f"llamacpp_ops_sm{sm}.sql").read_text().splitlines():
        if not line.startswith("INSERT"):
            continue
        vals = re.findall(r"'([^']*)'", line.split("VALUES", 1)[1])
        op, params, time_us = vals[3], vals[4], float(vals[9])
        p = dict(re.findall(r"(\w+)=([^,\[]+|\[[^\]]*\])", params))
        typ = p["type_a"].upper()
        n, m, k = int(p["n"]), int(p["m"]), int(p["k"])
        if op == "MUL_MAT":
            kind = "dense"
        else:
            kind = "moe_gate_up" if p["b"] == "1" else "moe_down"
        out[(typ, kind, m, k, n)] = time_us
    return out


def main() -> None:
    sm = sys.argv[1]
    llama = load_llama(sm)
    ours = json.loads((HERE / "results" / f"ours_ops_sm{sm}.json").read_text())
    paths = defaultdict(dict)
    for r in ours["rows"]:
        paths[(r["type"], r["kind"], r["m"], r["k"], r["n"])][r["path"]] = r["us"]

    cases = []
    for key, by_path in sorted(paths.items()):
        best = min(by_path, key=by_path.get)
        cases.append({"type": key[0], "kind": key[1], "m": key[2], "k": key[3], "n": key[4],
                      "llama_us": llama.get(key), "ours_us": by_path, "best_path": best,
                      "ratio_best": llama[key] / by_path[best] if key in llama else None})

    # per-layer sums (best path per op), per type and n
    layer = {}
    for model, ops in LAYERS.items():
        for t in TYPES:
            for n in NS:
                l_us = sum(c * llama[(t, kind, m, k, n)] for kind, m, k, c in ops)
                per_path = defaultdict(float)
                o_best = 0.0
                for kind, m, k, c in ops:
                    bp = paths[(t, kind, m, k, n)]
                    o_best += c * min(bp.values())
                    for p, us in bp.items():
                        per_path[p] += c * us
                layer[f"{model}|{t}|{n}"] = {"llama_us": l_us, "ours_best_us": o_best, "ratio": l_us / o_best}

    # per-path sums: Qwen3-8B dense layer, and Qwen3-30B-A3B expert ops only (gate+up+down)
    groups = {"8B dense layer": LAYERS["Qwen3-8B"], "30B-A3B experts": LAYERS["Qwen3-30B-A3B"][3:]}
    per_path = {}
    for g, ops in groups.items():
        for t in TYPES:
            for n in NS:
                l_us = sum(c * llama[(t, kind, m, k, n)] for kind, m, k, c in ops)
                common = set.intersection(*(set(paths[(t, kind, m, k, n)]) for kind, m, k, _ in ops))
                for pth in common:
                    o_us = sum(c * paths[(t, kind, m, k, n)][pth] for kind, m, k, c in ops)
                    per_path[f"{g}|{t}|{n}|{pth}"] = {"llama_us": l_us, "ours_us": o_us, "ratio": l_us / o_us}

    out = {"device": ours["device"], "cases": cases, "layer": layer, "per_path": per_path}
    (HERE / "results" / f"compare_sm{sm}.json").write_text(json.dumps(out, indent=1))

    print(f"### Per-layer linear time, sm_{sm} ({ours['device']}): llama.cpp us / ours (best path) us = ratio\n")
    for model in LAYERS:
        print(f"{model} | " + " | ".join(f"n={n}" for n in NS))
        print("---|" + "---|" * len(NS))
        for t in TYPES:
            cells = []
            for n in NS:
                v = layer[f"{model}|{t}|{n}"]
                cells.append(f"{v['llama_us']:.0f} / {v['ours_best_us']:.0f} = **{v['ratio']:.2f}**")
            print(f"{t} | " + " | ".join(cells))
        print()

    print(f"### Per path, sm_{sm}: ratio llama.cpp / ours (>= 0.8 meets the bar)\n")
    path_names = ["mmvq", "mmq", "dequant_gemm", "moe_vec", "moe_mmq"]
    for g in groups:
        print(f"{g} | path | " + " | ".join(f"n={n}" for n in NS))
        print("---|---|" + "---|" * len(NS))
        for t in TYPES:
            for pth in path_names:
                cells = [per_path.get(f"{g}|{t}|{n}|{pth}") for n in NS]
                if any(cells):
                    print(f"{t} | {pth} | " + " | ".join(f"{c['ratio']:.2f}" if c else "-" for c in cells))
        print()

    print(f"### Per-op detail, sm_{sm}: llama.cpp us | ours us per path (ratio llama/ours)\n")
    print("type | op (m x k) | n | llama.cpp | mmvq | mmq | dequant+GEMM | moe_vec | moe_mmq")
    print("---|---|---|---|---|---|---|---|---")
    for c in cases:
        o = c["ours_us"]

        def cell(p):
            return f"{o[p]:.1f} ({c['llama_us'] / o[p]:.2f})" if p in o and c["llama_us"] else "-"

        print(f"{c['type']} | {c['kind']} {c['m']}x{c['k']} | {c['n']} | {c['llama_us']:.1f} | "
              + " | ".join(cell(p) for p in ("mmvq", "mmq", "dequant_gemm", "moe_vec", "moe_mmq")))


if __name__ == "__main__":
    main()
