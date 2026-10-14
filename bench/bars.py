"""The release gate's bars (release design §5.3, plan §9) on one GGUF file and card: Holon's in-process pp512 and tg128
at depth 0 and 8192 (`llama_bench.py`) against `llama-bench` at the gate pin and at the newest upstream
(`setup_baselines.sh`), every engine at the lock's settings — Holon its engine settings, llama-bench the lock's KV
format and step budget (`-ctk/-ctv`, `-b`) with `-ub 512`. ABAB: each round runs every engine once, each in a fresh
process. A round's value per metric is the median of repetitions 2-5 (of 5, after a warm-up); the gate statistic is
the median over rounds; the gate is Holon >= 0.8x the pin on every metric. The record names the ggml commit Holon ran
(`kernels.ggml_cuda`'s vendored tree) and each llama.cpp build's.

    python bench/bars.py --name <cell> --gpu 0 --model <file.gguf> --lock <lock file> [--rounds 3]
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import re
import shlex
import statistics
import subprocess
import sys
import tomllib

from run import BENCH, ENVS, ROOT, git, gpu_class, lock_flags, model_id, public, read_lock

GATE = 0.8
METRICS = {(512, 0, 0): "pp512", (512, 0, 8192): "pp512 @ d8192", (0, 128, 0): "tg128", (0, 128, 8192): "tg128 @ d8192"}
LLAMA_KV = {None: "f16", "q8_0": "q8_0"}               # Holon's kv.format -> llama.cpp's cache type (bf16 KV ~ f16)
GGML = ROOT / "plugins" / "holon_plugins" / "kernels" / "ggml_cuda" / "plugin.toml"
LLAMA_KEEP = ("build_commit", "n_batch", "n_ubatch", "type_k", "type_v", "flash_attn", "n_gpu_layers")  # no host info


def holon(args: argparse.Namespace, engine_args: list[str], out: pathlib.Path) -> tuple[dict, dict]:
    cmd = [sys.executable, str(BENCH / "llama_bench.py"), "--model", args.model, "-p", "512", "-n", "128",
           "--depths", "0", "8192", "--pp-depths", "8192", "-r", "5", "--blocking-sync",
           "--engine-args", shlex.join(engine_args), "--json", str(out)]
    env = args.env | {"PYTHONPATH": os.pathsep.join(str(ROOT / d) for d in ("core", "std", "plugins"))}
    with out.with_suffix(".log").open("w") as log:
        subprocess.run(cmd, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    record = json.loads(out.read_text())
    return {k: v["runs"] for k, v in record["results"].items()}, record


def llama(binary: pathlib.Path, args: argparse.Namespace, flags: list[str], out: pathlib.Path) -> tuple[dict, dict]:
    with out.with_suffix(".log").open("w") as log:
        rows = json.loads(subprocess.run([str(binary), "-m", args.model, *flags], env=args.env, stdout=subprocess.PIPE,
                                         stderr=log, check=True, text=True).stdout)
    return ({METRICS[(r["n_prompt"], r["n_gen"], r["n_depth"])]: r["samples_ts"] for r in rows},
            {k: rows[0].get(k) for k in LLAMA_KEEP})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="the cell, for the evidence file")
    ap.add_argument("--gpu", type=int, required=True, help="nvidia-smi index to pin")
    ap.add_argument("--model", required=True, help="a .gguf file")
    ap.add_argument("--lock", required=True, help="the lock whose settings every engine runs (borrowed if another "
                                                  "class's)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--llama", nargs="+", default=["gate", "upstream"], help="llama.cpp builds (bench/.envs)")
    ap.add_argument("--evidence-dir", default=str(ROOT / "evidence"))
    args = ap.parse_args()
    args.env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(args.gpu))

    lock = read_lock(args.lock)
    engine = lock.pop("engine")
    engine_args = lock_flags(engine, skip=("max_model_len",))     # llama_bench.py sizes the context to its tests
    kv = LLAMA_KV[engine["plugin_options"].get("kv.format")]
    llama_flags = ["-p", "512", "-n", "128", "-d", "0,8192", "-ngl", "99", "-fa", "on", "-ctk", kv, "-ctv", kv,
                   "-b", str(engine["max_step_tokens"]), "-ub", "512", "-r", "5", "-o", "json"]
    gpu = gpu_class(args.gpu)
    raw = BENCH / ".out" / f"bars_{args.name}_{datetime.datetime.now():%Y%m%d-%H%M%S}"
    raw.mkdir(parents=True)

    engines = ["holon", *args.llama]
    rounds: list[dict] = []
    builds: dict[str, dict] = {}
    holon_record: dict = {}
    for r in range(args.rounds):
        rounds.append({})
        for e in engines:
            print(f"round {r} {e} ...", flush=True)
            out = raw / f"r{r}-{e}.json"
            try:
                if e == "holon":
                    rounds[r][e], holon_record = holon(args, engine_args, out)
                else:
                    rounds[r][e], builds[e] = llama(ENVS / f"llama.cpp-{e}" / "bin" / "llama-bench", args,
                                                    llama_flags, out)
            except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as err:
                print(f"  failed: {err}; log {out.with_suffix('.log').relative_to(ROOT)}", flush=True)
                rounds[r][e] = {"error": str(err).replace(str(ROOT) + "/", "")}

    results: dict[str, dict] = {}
    for metric in METRICS.values():
        res: dict = {}
        for e in engines:
            per_round = [round(statistics.median(rd[e][metric][1:5]), 2) for rd in rounds if metric in rd[e]]
            res[e] = {"statistic": round(statistics.median(per_round), 2) if per_round else None,
                      "per_round": per_round}
        for e in args.llama:
            if res["holon"]["statistic"] and res[e]["statistic"]:
                res[f"ratio_{e}"] = round(res["holon"]["statistic"] / res[e]["statistic"], 3)
        results[metric] = res

    dirty = git("status", "--porcelain", "--untracked-files=no", "--", ".", ":!evidence").splitlines()
    date = datetime.date.today().isoformat()
    record = {
        "cell": args.name, "date": date, "gpu_class": gpu, "gpu": holon_record.get("gpu"),
        "model": model_id(args.model), "command": shlex.join(["python", "bench/bars.py",
                                                              *(public(a, args.model) for a in sys.argv[1:])]),
        "commit": git("rev-parse", "HEAD") + ("-dirty" if dirty else ""),
        "ggml": tomllib.loads(GGML.read_text())["provenance"]["commit"],
        "lock": lock | {"borrowed": not lock["fingerprint_class"].startswith(gpu + "/")},
        "holon": {"engine_args": shlex.join(engine_args), "engine": holon_record.get("engine")},
        "llama_cpp": {e: {"flags": shlex.join(llama_flags), **builds.get(e, {})} for e in args.llama},
        "method": "per round the median of repetitions 2-5 (of 5 after a warm-up); statistic = median over rounds; "
                  f"gate: Holon >= {GATE}x llama.cpp's gate build on every metric",
        "results": results,
        "below_gate": [m for m, res in results.items() if res.get("ratio_gate", GATE) < GATE],
        "incomplete": [m for m, res in results.items() if any(res[e]["statistic"] is None for e in engines)],
        "rounds": rounds,
    }
    out = pathlib.Path(args.evidence_dir) / f"bars_release-{args.name}_{gpu}_{date}.json"
    text = re.sub(r"\[\s+([^\[\]{}]*?)\s+\]", lambda m: "[" + re.sub(r"\s+", " ", m[1]) + "]",
                  json.dumps(record, indent=1))                    # number lists on one line, as run.py's records
    out.write_text(text + "\n")
    for metric, res in results.items():
        print(metric, {e: res[e]["statistic"] for e in engines}, {k: v for k, v in res.items() if "ratio" in k})
    print(f"wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")


if __name__ == "__main__":
    main()
