"""Print markdown tables from results/*.json (used to fill README.md).

  python summarize.py results/bench_bf16_a5000.json [...]
"""

import json
import sys
from collections import defaultdict


def fmt(x):
    return "—" if x is None else (f"{x:.1f}" if x < 1000 else f"{x:.0f}")


def latency_tables(res: dict):
    rows = res["latency"]
    order = ["fi_decode_cudacore", "fi_decode_tc", "fi_paged_prefill", "fi_decode_tc_devmeta",
             "fi_paged_prefill_devmeta", "triton", "xqa"]
    kernels = [k for k in order if any(r["kernel"] == k for r in rows)]
    by = defaultdict(dict)
    for r in rows:
        by[(r["nq"], r["geom"], r["page"], r["batch"], r["ctx"])][r["kernel"]] = r.get("us")
    for nq in sorted({k[0] for k in by}):
        ks = [k for k in kernels if any(k in v for kk, v in by.items() if kk[0] == nq)]
        best_fi = [k for k in ks if k in ("fi_decode_tc", "fi_decode_cudacore", "fi_paged_prefill")]
        print(f"\n#### nq = {nq} ({'decode' if nq == 1 else f'verify, k = {nq - 1}'}) — µs per layer call\n")
        print("| geom | page | B | ctx | " + " | ".join(ks) + " | triton / best host-plan FI |")
        print("|" + "---|" * (len(ks) + 5))
        for key in sorted(k for k in by if k[0] == nq):
            v = by[key]
            fi = [v[k] for k in best_fi if v.get(k)]
            ratio = f"{v['triton'] / min(fi):.2f}" if v.get("triton") and fi else "—"
            print(f"| {key[1]} | {key[2]} | {key[3]} | {key[4]} | " + " | ".join(fmt(v.get(k)) for k in ks)
                  + f" | {ratio} |")


def correctness_table(res: dict):
    worst = defaultdict(lambda: [0.0, 0, set()])
    for r in res["correctness"]:
        key = (r["kernel"], r.get("nq", "-"))
        if "error" in r:
            worst[key][2].add(r["error"][:90])
        else:
            worst[key][0] = max(worst[key][0], r["max_abs"])
            worst[key][1] += 1
    print("\n| kernel | nq | configs ok | worst max_abs vs fp32 | errors |\n|---|---|---|---|---|")
    for (k, nq), (m, n, e) in sorted(worst.items(), key=lambda x: str(x[0])):
        print(f"| {k} | {nq} | {n} | {m:.4f} | {'; '.join(sorted(e)) or ''} |")


def prefill_table(res: dict):
    if not res.get("prefill"):
        return
    print("\n| geom | B | ctx | ragged µs | ragged TFLOPS | paged µs | paged TFLOPS |\n|---|---|---|---|---|---|---|")
    by = defaultdict(dict)
    for r in res["prefill"]:
        by[(r["geom"], r["batch"], r["ctx"])][r["kernel"]] = r
    for key, v in sorted(by.items()):
        rg, pg = v.get("fi_ragged_prefill", {}), v.get("fi_paged_prefill", {})
        print(f"| {key[0]} | {key[1]} | {key[2]} | {fmt(rg.get('us'))} | {fmt(rg.get('tflops'))} | "
              f"{fmt(pg.get('us'))} | {fmt(pg.get('tflops'))} |")


def main():
    for path in sys.argv[1:]:
        res = json.load(open(path))
        print(f"\n### {path}  ({res['fingerprint']['gpu']}, sm_{res['fingerprint']['sm']})")
        if "correctness" in res:
            correctness_table(res)
            latency_tables(res)
            prefill_table(res)


if __name__ == "__main__":
    main()
