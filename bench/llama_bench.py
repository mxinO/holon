"""Holon's llama-bench (M0.1 design §4.8), in process, batch 1: pp<P> = one P-token prefill (time to its first
token), tg<N> at depth D = N single-token decode steps after a prompt of D tokens (llama-bench's `-d`; time from
the first token to the last). Greedy, `ignore_eos`, random prompts with the prefix cache off; warm-up runs
excluded; mean ± sd over `-r` repetitions, like llama-bench. With `--blocking-sync` (bench/worker.py) CUDA waits
sleep, so the host CPU time per decode step is host work only.

    python bench/llama_bench.py --model <file.gguf | checkpoint dir> [-p 512] [-n 128] [-d 0 256] [-r 5]
                                [--engine-args "--memory-fraction 0.9"] [--blocking-sync] [--json <file>]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import shlex
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _stats(xs: list[float]) -> dict:
    return {"mean": round(statistics.mean(xs), 2), "sd": round(statistics.stdev(xs), 2) if len(xs) > 1 else 0.0,
            "runs": [round(x, 2) for x in xs]}


def _timed(engine, prompt: list[int], max_tokens: int) -> tuple[list[float], list[float]]:
    """One request alone: the wall time each token arrived at (from submission) and the CPU time of each step."""
    from holon.api import SamplingParams

    engine.add(prompt, SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True))
    arrivals, cpu, t0 = [], [], time.perf_counter()
    while engine.has_work():
        c0 = time.process_time()
        deltas = engine.step()
        cpu.append(time.process_time() - c0)
        now = time.perf_counter() - t0
        arrivals += [now] * sum(len(d.token_ids) for d in deltas)
    return arrivals, cpu


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("-p", type=int, default=512, help="prompt tokens of the pp test")
    ap.add_argument("-n", type=int, default=128, help="generated tokens of the tg test")
    ap.add_argument("-d", "--depths", dest="d", type=int, nargs="+", default=[0, 256],
                    help="depths of the tg test (use --depths where FlashInfer loads NVIDIA's CuTe DSL, whose own "
                         "argv parser takes -d for its -diagnostic)")
    ap.add_argument("-r", type=int, default=5, help="repetitions")
    ap.add_argument("--engine-args", default="", help="EngineConfig flags, as for `holon serve`")
    ap.add_argument("--blocking-sync", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.blocking_sync:
        from worker import blocking_sync

        blocking_sync()

    import torch
    from holon.core import platform
    from holon.core.cli import engine_config, engine_flags
    from holon.core.engine import Engine

    flags = argparse.ArgumentParser()
    engine_flags(flags)
    cfg = engine_config(a.model, flags.parse_args([
        "--max-seqs", "1", "--max-model-len", str(max(a.p, max(a.d) + a.n) + 8), "--max-step-tokens",
        str(max(a.p, *a.d, 1)), "--no-prefix-cache", *shlex.split(a.engine_args)]))
    engine = Engine(cfg)
    rng = random.Random(0)
    vocab = engine.source.config()["vocab_size"]

    def prompt(n: int) -> list[int]:                       # ordinary tokens (ids past the special ones' start)
        return [rng.randrange(1000, vocab // 2) for _ in range(n)]

    results: dict[str, dict] = {}
    pp, tg = [], {d: [] for d in a.d}
    cpu = {d: [] for d in a.d}
    for rep in range(a.r + 1):                              # repetition 0: warm-up (graphs, autotuning), excluded
        arrivals, _ = _timed(engine, prompt(a.p), 1)
        if rep:
            pp.append(a.p / arrivals[0])
        for d in a.d:
            arrivals, steps = _timed(engine, prompt(max(d, 1)), a.n + 1)
            if rep:
                tg[d].append(a.n / (arrivals[-1] - arrivals[0]))
                cpu[d] += steps[-a.n:]
    results[f"pp{a.p}"] = _stats(pp)
    for d in a.d:
        results[f"tg{a.n}" + (f" @ d{d}" if d else "")] = _stats(tg[d]) | (
            {"cpu_ms_per_step_p50": round(statistics.median(cpu[d]) * 1e3, 3)} if a.blocking_sync else {})
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    record = {"model": engine.source.identity(), "gpu_class": platform.fingerprint(engine.device).cls,
              "gpu": torch.cuda.get_device_name(engine.device), "commit": commit,
              "command": "python bench/llama_bench.py " + shlex.join(
                  pathlib.Path(x).name if x.startswith("/") else x for x in sys.argv[1:]),
              "engine": {k: str(v) for k, v in vars(cfg).items() if k != "model"}, "results": results}
    print(json.dumps(record["results"], indent=1))
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
