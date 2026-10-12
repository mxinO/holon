"""Holon's llama-bench (M0.1 design §4.8), in process, batch 1: pp<P> = one P-token prefill (time to its first
token), tg<N> at depth D = N single-token decode steps after a prompt of D tokens (llama-bench's `-d`; time from
the first token to the last). Greedy, `ignore_eos`, random prompts with the prefix cache off; warm-up runs
excluded; mean ± sd over `-r` repetitions, like llama-bench. With `--blocking-sync` (bench/worker.py) CUDA waits
sleep, so the host CPU time per decode step is host work only.

`-p` takes several sizes. `--pp-depths D…` also times pp<P> at depth D, as llama-bench's `-d` does for pp: P new
tokens after a D-token context the prefix cache holds (an agent turn's prefill), so the cache is on then.
`--profile` adds one profiled run per pp case: its summed GPU kernel time, and idle = 1 − kernels / wall, the share
of the prefill the GPU waits for the host (what prefill graphs could remove).

    python bench/llama_bench.py --model <file.gguf | checkpoint dir> [-p 512 ...] [-n 128] [--depths 0 256] [-r 5]
                                [--pp-depths 8192 ...] [--profile] [--engine-args "--memory-fraction 0.9"]
                                [--blocking-sync] [--json <file>]
"""
from __future__ import annotations

import argparse
import contextlib
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
    ap.add_argument("-p", type=int, nargs="+", default=[512], help="prompt tokens of the pp tests")
    ap.add_argument("-n", type=int, default=128, help="generated tokens of the tg test")
    ap.add_argument("-d", "--depths", dest="d", type=int, nargs="+", default=[0, 256],
                    help="depths of the tg test (use --depths where FlashInfer loads NVIDIA's CuTe DSL, whose own "
                         "argv parser takes -d for its -diagnostic)")
    ap.add_argument("--pp-depths", type=int, nargs="+", default=[], help="cached context before the pp tests")
    ap.add_argument("--profile", action="store_true", help="GPU kernel time and idle share of each pp test")
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
    cache = [] if a.pp_depths else ["--no-prefix-cache"]          # pp at depth resumes a cached context
    cfg = engine_config(a.model, flags.parse_args([
        "--max-seqs", "1", "--max-model-len", str(max(max(a.p) + max(a.pp_depths, default=0), max(a.d) + a.n) + 8),
        "--max-step-tokens", str(max(*a.p, *a.d, 1)), *cache, *shlex.split(a.engine_args)]))
    engine = Engine(cfg)
    rng = random.Random(0)
    vocab = engine.spec.vocab

    def prompt(n: int) -> list[int]:                       # ordinary tokens (ids past the special ones' start)
        return [rng.randrange(1000, vocab // 2) for _ in range(n)]

    def pp_case(p: int, depth: int, profiler=None) -> float:
        """Seconds to the first token of p new tokens after `depth` cached ones (`profiler`: around those only)."""
        context = prompt(depth)
        if depth:
            _timed(engine, context, 1)                          # fill and cache the context (untimed)
            saved = engine.stats()["prefix_tokens_saved"]
        with profiler or contextlib.nullcontext():
            arrivals, _ = _timed(engine, context + prompt(p), 1)
        if depth and engine.stats()["prefix_tokens_saved"] - saved != depth:
            raise RuntimeError(f"pp{p} @ d{depth}: the context was not reused from the prefix cache")
        return arrivals[0]

    results: dict[str, dict] = {}
    cases = [(p, depth) for depth in (0, *a.pp_depths) for p in a.p]
    pp, tg = {c: [] for c in cases}, {d: [] for d in a.d}
    cpu = {d: [] for d in a.d}
    for rep in range(a.r + 1):                              # repetition 0: warm-up (graphs, autotuning), excluded
        for c in cases:
            wall = pp_case(*c)
            if rep:
                pp[c].append(wall)
        for d in a.d:
            arrivals, steps = _timed(engine, prompt(max(d, 1)), a.n + 1)
            if rep:
                tg[d].append(a.n / (arrivals[-1] - arrivals[0]))
                cpu[d] += steps[-a.n:]
    for p, depth in cases:
        key = f"pp{p}" + (f" @ d{depth}" if depth else "")
        results[key] = _stats([p / w for w in pp[(p, depth)]])
        if a.profile:
            prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
            pp_case(p, depth, prof)
            kernels = sum(e.device_time_total for e in prof.key_averages()) / 1e6
            wall = statistics.median(pp[(p, depth)])
            results[key] |= {"wall_ms": round(wall * 1e3, 2), "kernel_ms": round(kernels * 1e3, 2),
                             "idle": round(1 - kernels / wall, 3)}
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
