"""A1's human-gate figures comparable to Strata's (docs/v1-offload.md §6): tail skip against the no-skip candidate at
the same frozen placement — KL(no-skip ‖ skip) per position (the judge's top-20 bucketed KL, a lower bound on the full
vocabulary's), its mean, p50, p99, and top-1 agreement — at F11-representative placements (`placement.py`), beside
each candidate against the fp32 golden (and the mmvq floor's figures):

    python spikes/x7_heldout/gate.py <dir> <golden> <floor> <out.json> <fit> ... [--extra <name>=<skip.safetensors>]

`<dir>` holds `noskip<fit>.safetensors` and `skip<fit>.safetensors` per fit (l2.py, one token per step). The no-skip
candidates of every fit must be bitwise equal (F5a is exact at any placement), so any of them is the reference for an
extra skip candidate (the step-7 run at F2's warm start)."""
import json
import pathlib
import sys

from holon_judge.compare import metrics
from holon_judge.recording import Recording


def row(m) -> dict:
    return {"n": m.n, "top1": m.top1, "kl_mean": m.kl_mean, "kl_p50": m.kl_p50, "kl_p99": m.kl_p99,
            "kl_max": m.kl_max, "dppl": m.dppl, "dppl_se": m.dppl_se}


def figures(golden: Recording, cand: Recording) -> dict:
    overall, by_set = metrics(golden, cand)
    return row(overall) | {"by_set": {name: row(m) for name, m in by_set.items()}}


def main() -> None:
    args = sys.argv[1:]
    extra = dict(a.split("=", 1) for a in args[args.index("--extra") + 1:]) if "--extra" in args else {}
    args = args[:args.index("--extra")] if "--extra" in args else args
    d, golden, floor, out, fits = pathlib.Path(args[0]), Recording.load(args[1]), Recording.load(args[2]), \
        pathlib.Path(args[3]), args[4:]
    result = {"floor_vs_golden": figures(golden, floor), "fits": {}}
    noskip = {f: Recording.load(d / f"noskip{f}.safetensors") for f in fits}
    result["noskip_bitwise_equal_across_fits"] = all(noskip[fits[0]].same_tensors(n) for n in noskip.values())
    for f in fits:
        skip = Recording.load(d / f"skip{f}.safetensors")
        result["fits"][f] = {"skip_vs_noskip": figures(noskip[f], skip), "skip_vs_golden": figures(golden, skip),
                             "noskip_vs_golden": figures(golden, noskip[f])}
    for name, path in extra.items():
        skip = Recording.load(path)
        result["fits"][name] = {"skip_vs_noskip": figures(noskip[fits[0]], skip),
                                "skip_vs_golden": figures(golden, skip)}
    out.write_text(json.dumps(result, indent=1) + "\n")
    for f, r in result["fits"].items():
        s = r["skip_vs_noskip"]
        print(f"{f}: skip vs no-skip KL mean {s['kl_mean']:.3e} p50 {s['kl_p50']:.3e} p99 {s['kl_p99']:.3e} top-1 "
              f"{100 * s['top1']:.2f}%; vs golden top-1 {100 * r['skip_vs_golden']['top1']:.2f}% KL mean "
              f"{r['skip_vs_golden']['kl_mean']:.3e}")
    print("no-skip equal across fits:", result["noskip_bitwise_equal_across_fits"])


if __name__ == "__main__":
    main()
