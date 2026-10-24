"""Per-change perf smoke (docs/release.md §22): batch-1 decode of this tree against a base tree (main's build) on one
GPU, in a few minutes (2.3 min on an RTX A5000 with Qwen3-8B Q4_K_M). Each round runs both arms, in ABBA order, each
as a fresh `holon_tg.py` process from its own tree with that tree first on PYTHONPATH; an arm that imported Holon from
anywhere else is an error. nvidia-smi samples the GPU through every arm: the clock screen (`clock_check.py`'s rule,
p10 ≥ 0.9 × p50 of the SM clock while the GPU is busy) must hold in every arm, or the result is void, as it is when
no sample shows the GPU busy (or nvidia-smi gave nothing usable); a busy GPU whose SM clock reads `[N/A]` has no screen,
which the verdict says.

    python bench/perf_smoke.py --base <checkout of main> --model <checkpoint | file.gguf> [--gpu 0] [--rounds 3]
                               [-n 128] [--depth 1024] [-r 5] [--engine-args "<holon serve flags>"] [--json out]

The base checkout needs its own GGUF kernel build if its kernels' sources differ (`holon prebuild`). Both arms find
their plugins through this environment's entry points (the installed distributions' metadata), importing each
plugin's module from the arm's own tree: a change that adds, removes or moves a plugin between the trees cannot be
compared this way (the arm fails to import it), only one that changes plugins' code.

Verdict and exit status: "slower" (1) or "faster" (0) when every round's candidate / base ratio lies beyond `--tol`
on the same side, else "within noise" (0); "void" (3) when a clock screen failed.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys
import tempfile

from clock_check import SLOWDOWN, STEADY

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUSY = 50                                       # GPU utilization (%) of a sample counted in the clock screen
PROBE = ("import runpy, sys, holon, holon_std, holon_plugins as p; print('imported', *(m.__file__ for m in "
         "(holon, holon_std, p)), flush=True); sys.argv = sys.argv[1:]; "
         "runpy.run_path(sys.argv[0], run_name='__main__')")


def arm(tree: pathlib.Path, a: argparse.Namespace, out: pathlib.Path) -> dict:
    """One fresh holon_tg process from `tree`: its tokens/s and clock screen."""
    env = os.environ | {"PYTHONPATH": os.pathsep.join(str(tree / d) for d in ("core", "std", "plugins")),
                        "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(a.gpu)}
    cmd = [sys.executable, "-c", PROBE, str(tree / "bench" / "holon_tg.py"), "--model", a.model, "-n", str(a.n),
           "--depths", str(a.depth), "-r", str(a.r), "--engine-args", a.engine_args, "--json", str(out)]
    with tempfile.TemporaryFile("w+") as clocks:
        smi = subprocess.Popen(["nvidia-smi", "-i", str(a.gpu), "--query-gpu=clocks.sm,utilization.gpu,"
                                "clocks_event_reasons.active", "--format=csv,noheader,nounits", "--loop-ms=100"],
                               stdout=clocks, text=True)
        run = subprocess.run(cmd, env=env, capture_output=True, text=True)
        smi.terminate()
        smi.wait()
        clocks.seek(0)
        samples = [line.split(", ") for line in clocks.read().splitlines()]
    if run.returncode:
        raise SystemExit(f"{tree}: holon_tg failed:\n{run.stdout[-2000:]}{run.stderr[-4000:]}")
    paths = next(line.split()[1:] for line in run.stdout.splitlines() if line.startswith("imported "))
    own = [tree / "core" / "holon", tree / "std" / "holon_std", tree / "plugins" / "holon_plugins"]
    if stray := [p for p, d in zip(paths, own, strict=True) if pathlib.Path(p).resolve().parent != d]:
        raise SystemExit(f"{tree}: the arm imported Holon from elsewhere: {stray}")
    loaded = [(sm, reasons) for sm, util, reasons in samples if _number(util) is not None and _number(util) >= BUSY]
    busy = sorted(x for sm, _ in loaded if (x := _number(sm)) is not None)
    slow = sum(int(r, 16) & SLOWDOWN != 0 for _, r in loaded if r.strip().startswith("0x"))
    p10, p50 = (busy[int(q * len(busy))] if busy else None for q in (0.1, 0.5))
    (result,) = json.loads(out.read_text())["results"].values()
    return {"tok_s": result["mean"], "runs": result["runs"], "sm_mhz_p10": p10, "sm_mhz_p50": p50,
            "slowdown_samples": slow,
            # no busy sample at all (or no usable output) voids the round; busy samples without a clock leave no screen
            "steady": False if not loaded else None if not busy else p10 >= STEADY * p50}


def _number(field: str) -> float | None:
    """An nvidia-smi field's value; None for `[N/A]` and the like."""
    try:
        return float(field)
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=pathlib.Path, required=True, help="the tree to compare against (main's checkout)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", type=int, default=0, help="nvidia-smi index of the GPU")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("-n", type=int, default=128, help="decoded tokens per repetition")
    ap.add_argument("--depth", type=int, default=1024, help="prompt tokens")
    ap.add_argument("-r", type=int, default=5, help="repetitions per arm (after a warm-up)")
    ap.add_argument("--engine-args", default="")
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--json")
    a = ap.parse_args()
    trees = {"candidate": ROOT, "base": a.base.resolve()}
    rounds = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(a.rounds):
            order = ["candidate", "base"] if i % 2 == 0 else ["base", "candidate"]
            rounds.append({name: arm(trees[name], a, pathlib.Path(tmp, f"{i}-{name}.json")) for name in order})
            r = rounds[-1]
            print(f"round {i}: candidate {r['candidate']['tok_s']:.2f}, base {r['base']['tok_s']:.2f} tok/s, ratio "
                  f"{r['candidate']['tok_s'] / r['base']['tok_s']:.4f}", flush=True)
    ratios = [r["candidate"]["tok_s"] / r["base"]["tok_s"] for r in rounds]
    screens = [x["steady"] for r in rounds for x in r.values()]
    verdict = ("void (a clock screen failed: rerun on an idle GPU)" if False in screens else
               "slower" if max(ratios) < 1 - a.tol else "faster" if min(ratios) > 1 + a.tol else "within noise")
    if None in screens:
        verdict += " (no clock screen: the GPU reports no SM clock)"
    record = {"model": a.model.rstrip("/").rsplit("/", 1)[-1], "batch": 1, "depth": a.depth, "n": a.n, "r": a.r,
              "engine_args": a.engine_args, "tol": a.tol, "rounds": rounds, "ratios": [round(x, 4) for x in ratios],
              "median_ratio": round(statistics.median(ratios), 4), "verdict": verdict}
    print(f"candidate / base: median {record['median_ratio']}, rounds {record['ratios']}; {verdict}")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(record, indent=1) + "\n")
    return 3 if verdict.startswith("void") else 1 if verdict.startswith("slower") else 0


if __name__ == "__main__":
    sys.exit(main())
