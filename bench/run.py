"""Compare engines on the same GPU and model; writes an evidence record.

Interleaved ABAB: every round runs each backend once, in order, each in a fresh process with its own venv
(`worker.py`), so slow drift of the machine (clocks, temperature, neighbours) hits every backend alike. Each
process loads the model once and replays every trace after a warmup (`driver.warmup_sessions`).

Usage (from the repository root, inside Holon's .venv):
  python bench/run.py --name decode --gpu 5 --model <checkpoint dir> --backends holon minisgl \\
      --traces bench/traces/decode_b1.jsonl bench/traces/decode_b8.jsonl bench/traces/decode_b64.jsonl
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BENCH = ROOT / "bench"
ENVS = BENCH / ".envs"                                   # baseline venvs and sources (setup_baselines.sh)
SOURCES = {"minisgl": "mini-sglang", "nanovllm": "nano-vllm"}
METHOD = (
    "In-process engines driven by bench/driver.py, one scheduling step per call; client-side timestamps taken when "
    "a step returns a token. TTFT = first token - release (arrival, or previous turn's last token + gap); TPOT = "
    "(last - first) / (tokens - 1) per request; throughput = output tokens / (last token - first release). Decode "
    "steps = steps at peak batch that return no first token; decode tok/s = their tokens / their wall time. CPU "
    "ms/step = time.process_time() delta around step() on those steps, measured with --blocking-sync (CUDA "
    "primary context in blocking-sync mode and torch events created blocking, so waiting for the GPU sleeps and "
    "is not counted). Greedy, ignore_eos, fixed output lengths; warmup on the same shapes with other prompt "
    "tokens, excluded.")


def git(*args: str, cwd: pathlib.Path = ROOT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def gpu_class(index: int) -> str:
    """Holon's fingerprint GPU class (product name without vendor words), e.g. "rtx-a5000"."""
    name = subprocess.run(["nvidia-smi", "-i", str(index), "--query-gpu=name", "--format=csv,noheader"],
                          capture_output=True, text=True, check=True).stdout.strip()
    return "-".join(w for w in re.split(r"[\s_]+", name.lower()) if w not in ("nvidia", "geforce"))


def model_id(path: str) -> str:
    """"Org/Name@snapshot" for a Hugging Face cache snapshot, else the directory name (no absolute paths)."""
    m = re.search(r"models--([^/]+)--([^/]+)/snapshots/([0-9a-f]+)", path)
    return f"{m[1]}/{m[2]}@{m[3][:12]}" if m else pathlib.Path(path).name


def public(arg: str, model: str) -> str:
    """A command-line argument fit for a published record: the model as its id, other paths repo-relative."""
    if arg == model:
        return model_id(model)
    if arg.startswith("--") and "=" in arg:
        key, value = arg.split("=", 1)
        return f"{key}={public(value, model)}"
    path = pathlib.Path(arg)
    if not path.is_absolute():
        return arg
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name


def pct(xs: list[float], q: float) -> float | None:
    """Linear-interpolated percentile (q in [0, 100])."""
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q / 100
    lo = int(k)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (k - lo)


def _dist_ms(xs: list[float]) -> dict:
    return {f"p{q}": None if (v := pct(xs, q)) is None else round(v * 1e3, 2) for q in (50, 90, 99)}


def summarize(cases: list[dict], columns: list[str], cpu_time: bool) -> dict:
    """Request percentiles pooled over `cases` (one per round); throughputs averaged over them. CPU time per step
    only when measured (`cpu_time`): with spinning CUDA waits it is just wall time."""
    ttft, tpot, tput, wall, cpu = [], [], [], [], []
    batch = short = dec_steps = dec_tok = 0
    for case in cases:
        rows = [dict(zip(columns, r, strict=True)) for r in case["requests"]]
        ttft += [r["first"] - r["release"] for r in rows]
        tpot += [(r["last"] - r["first"]) / (r["generated"] - 1) for r in rows if r["generated"] > 1]
        short += sum(r["generated"] != r["output"] for r in rows)
        span = max(r["last"] for r in rows) - min(r["release"] for r in rows)
        tput.append(sum(r["generated"] for r in rows) / span)
        peak = max(s[2] for s in case["steps"])
        batch = max(batch, peak)
        steady = [s for s in case["steps"] if s[2] == peak and s[3] == 0]
        dec_steps += len(steady)
        dec_tok += sum(s[2] for s in steady)
        wall += [s[0] for s in steady]
        cpu += [s[1] for s in steady] if cpu_time else []
    return {"requests": sum(len(c["requests"]) for c in cases), "ttft_ms": _dist_ms(ttft), "tpot_ms": _dist_ms(tpot),
            "throughput_tok_s": round(sum(tput) / len(tput), 1), "throughput_per_round": [round(t, 1) for t in tput],
            "decode": {"batch": batch, "steps": dec_steps, "tok_s": round(dec_tok / sum(wall), 1) if wall else None,
                       "wall_ms_p50": None if not wall else round(pct(wall, 50) * 1e3, 3),
                       "cpu_ms_p50": None if not cpu else round(pct(cpu, 50) * 1e3, 3),
                       "cpu_ms_mean": None if not cpu else round(sum(cpu) / len(cpu) * 1e3, 3)},
            "wrong_length": short}


def run_worker(backend: str, args: argparse.Namespace, out: pathlib.Path) -> tuple[dict | None, str]:
    # PCI order: --gpu is the nvidia-smi index, which also names the GPU class in the evidence.
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(args.gpu), HF_HUB_OFFLINE="1")
    if backend == "holon":   # measure this checkout, whatever copy of Holon the venv has installed
        env["PYTHONPATH"] = os.pathsep.join(str(ROOT / d) for d in ("core", "std", "plugins"))
        python = sys.executable
    else:
        env.pop("PYTHONPATH", None)
        python = str(ENVS / backend / "bin" / "python")
    cmd = [python, str(BENCH / "worker.py"), "--backend", backend, "--model", args.model, "--traces", *args.traces,
           "--max-batch", str(args.max_batch), "--max-step-tokens", str(args.max_step_tokens), "--out", str(out)]
    if args.blocking_sync:
        cmd.append("--blocking-sync")
    log = out.with_suffix(".log")
    with log.open("w") as f:
        rc = subprocess.run(cmd, env=env, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc:
        return None, f"exit code {rc}, log {log.relative_to(ROOT)}"
    return json.loads(out.read_text()), ""


def table(summary: dict) -> str:
    lines = ["| trace | backend | TTFT p50 / p99 ms | TPOT p50 / p99 ms | out tok/s | decode tok/s (batch) | "
             "decode step: wall / CPU ms (p50) |", "|---|---|---|---|---|---|---|"]
    for trace, by_backend in summary.items():
        for b, s in by_backend.items():
            d = s["decode"]
            lines.append(f"| {trace} | {b} | {s['ttft_ms']['p50']} / {s['ttft_ms']['p99']} | {s['tpot_ms']['p50']} / "
                         f"{s['tpot_ms']['p99']} | {s['throughput_tok_s']} | {d['tok_s']} ({d['batch']}) | "
                         f"{d['wall_ms_p50']} / {"-" if d['cpu_ms_p50'] is None else d['cpu_ms_p50']} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="workload name for the evidence file")
    ap.add_argument("--gpu", type=int, required=True, help="CUDA device index to pin")
    ap.add_argument("--model", required=True, help="checkpoint directory")
    ap.add_argument("--traces", nargs="+", required=True, help="JSONL specs from workloads.py")
    ap.add_argument("--backends", nargs="+", default=["holon", "minisgl"],
                    choices=["holon", "minisgl", "nanovllm"])
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--max-batch", type=int, default=64, help="max running sequences (same for every backend)")
    ap.add_argument("--max-step-tokens", type=int, default=8192, help="max tokens per prefill step")
    ap.add_argument("--blocking-sync", action="store_true", help="CUDA waits sleep (for CPU time per step)")
    ap.add_argument("--evidence-dir", default=str(ROOT / "evidence"))
    args = ap.parse_args()

    date = datetime.date.today().isoformat()
    gpu = gpu_class(args.gpu)
    raw = BENCH / ".out" / f"{args.name}_{datetime.datetime.now():%Y%m%d-%H%M%S}"
    raw.mkdir(parents=True)
    runs, infos = [], {}
    for r in range(args.rounds):
        for b in args.backends:
            print(f"round {r} {b} ...", flush=True)
            res, err = run_worker(b, args, raw / f"r{r}-{b}.json")
            if res is None:
                print(f"  failed:\n{err}", flush=True)
                runs.append({"round": r, "backend": b, "error": err})
                continue
            if b not in infos:
                infos[b] = res["info"] | ({"commit": git("rev-parse", "HEAD", cwd=ENVS / "src" / SOURCES[b])}
                                          if b in SOURCES else {})
            for case in res["cases"]:
                runs.append({"round": r, "backend": b, "trace": case["trace"], "columns": res["columns"],
                             "summary": summarize([case], res["columns"], args.blocking_sync), **case})

    ok = [run for run in runs if "error" not in run]
    grouped: dict[str, dict[str, list[dict]]] = {}
    for run in ok:
        grouped.setdefault(run["trace"], {}).setdefault(run["backend"], []).append(run)
    summary = {trace: {b: summarize(rs, rs[0]["columns"], args.blocking_sync) for b, rs in by_backend.items()}
               for trace, by_backend in grouped.items()}
    for run in ok:
        del run["steps"]                                    # raw step timings stay in bench/.out
    argv = [public(a, args.model) for a in sys.argv[1:]]
    traces = {pathlib.Path(p).stem: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]
              for p in args.traces}
    # Records written by earlier runs (evidence/) do not make the code dirty; anything else is listed.
    dirty = git("status", "--porcelain", "--untracked-files=no", "--", ".", ":!evidence").splitlines()
    commit = git("rev-parse", "HEAD") + ("-dirty" if dirty else "")
    record = {"workload": args.name, "date": date, "command": " ".join(["python bench/run.py", *argv]),
              "commit": commit, **({"dirty": dirty} if dirty else {}), "gpu_class": gpu,
              "model": model_id(args.model), "rounds": args.rounds,
              "order": args.backends, "method": METHOD, "traces_sha256": traces, "backends": infos,
              "summary": summary, "runs": runs}
    out = pathlib.Path(args.evidence_dir) / f"bench_{args.name}_{gpu}_{date}.json"
    text = json.dumps(record, indent=1, default=str)
    text = re.sub(r"\[\s+([^\[\]{}]*?)\s+\]", lambda m: "[" + re.sub(r"\s+", " ", m[1]) + "]", text)  # rows on one line
    out.write_text(text + "\n")
    print(table(summary))
    print(f"wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")


if __name__ == "__main__":
    main()
