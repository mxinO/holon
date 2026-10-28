"""Render results/noise_floor_*.json as the markdown tables used in README.md.

    python tables.py results/noise_floor_qwen3-8b.json > results/tables_qwen3-8b.md
"""

import json
import sys

COLS = ["positions", "top-1", "top-1 (gap≥0.1)", "KL mean", "KL p50", "KL p99", "KL max", "ΔPPL ± SE", "imputed"]


def table(title: str, cols: list[str], rows: list[list]) -> None:
    print(f"\n{title}\n\n| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")


def metrics(m: dict) -> list:
    return [m["n"], f"{m['top1']:.3%}", f"{m['top1_confident']:.3%}", f"{m['kl_mean']:.2e}", f"{m['kl_p50']:.2e}",
            f"{m['kl_p99']:.2e}", f"{m['kl_max']:.3f}", f"{m['dppl']:+.3%} ± {m['dppl_se']:.3%}", f"{m['missing']:.2%}"]


def main(path: str) -> None:
    r = json.load(open(path))
    g = r["golden"]
    print(f"golden: {g['model']}@{g['revision'][:8]} fp32 {g['device']} ({g['threads']} threads), "
          f"PPL {r['vs_fp32']['bf16']['ppl_golden']:.3f}, {g['n_positions']} positions, wall {g['wall_s']} s, "
          f"top-k part {r['golden_topk_bytes'] / 1e6:.2f} MB")
    table("vs fp32 golden", ["run", *COLS], [[n, *metrics(m)] for n, m in r["vs_fp32"].items()])
    table("bootstrap SE over sequences (fp32 vs run)", ["run", "top-1", "KL p50", "KL p99", "ΔNLL"],
          [[n, f"{s['top1']:.3%}", f"{s['kl_p50']:.1e}", f"{s['kl_p99']:.1e}", f"{s['dnll']:.1e}"]
           for n, s in ((n, m["bootstrap_se"]) for n, m in r["vs_fp32"].items())])
    table("bf16 vs bf16", ["pair", *COLS, "bitwise"],
          [[n.replace("|", " vs "), *metrics(m), m["bitwise_identical"]] for n, m in r["pairs"].items()])
    for run, b in r["breakdown"].items():
        table(f"breakdown fp32 vs {run}", ["subset", *COLS], [[n, *metrics(m)] for n, m in b.items()])
    b = r["budgets"]["proposed"]
    table("proposed budgets (floor = bf16) and verdicts", ["run", "verdict", "failures"],
          [[n, "PASS" if v["passed"] else "FAIL", "; ".join(v["failures"])]
           for n, v in r["budgets"]["verdicts"].items()])
    print(f"\nbudgets: top-1 ≥ {b['top1_min']:.3%}, KL p50 ≤ {b['kl_p50_max']:.2e}, "
          f"KL p99 ≤ {b['kl_p99_max']:.2e}, |ΔPPL| ≤ {b['dppl_max']:.3%}")
    rows = []
    for run, v in r["kl_validation"].items():
        e = v["exact"]
        for k in (5, 10, 20, 50, 100):
            a = v[f"k{k}"]
            rows.append([run, k, f"{e['p50']:.2e}", f"{a['p50']:.2e}", f"{e['p99']:.2e}", f"{a['p99']:.2e}",
                         f"{a['ratio_p50']:.2f}", f"{a['ratio_p10']:.2f}",
                         f"{a['tail_mass_p50']:.1e} / {a['tail_mass_p99']:.2f}"])
    table(f"KL approximation: bucketed top-k + tail vs exact full-vocab ({v['n']} positions, every 32nd, "
          "sequence starts excluded)", ["run", "k", "exact p50", "approx p50", "exact p99", "approx p99",
                                         "ratio p50", "ratio p10", "golden tail mass p50 / p99"], rows)


if __name__ == "__main__":
    main(sys.argv[1])
