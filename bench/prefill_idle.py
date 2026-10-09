"""GPU-idle fraction of eager prefill (M0.3 design §4.6): is graph-capturing prefill worth it at agent-turn sizes?

An agent turn resends its history, which hits the prefix cache, plus `n` new tokens: its prefill computes those `n`
tokens (in two steps when the recurrent policy cuts the prompt at its last aligned block). Per size, the turn's wall
time over its prefill steps (synchronous: overlap off, so a step's wall is host work and GPU work as they overlap)
is measured unprofiled; the summed kernel time of the same turn comes from a profiled run. Idle = 1 − kernels / wall:
the share of the prefill the GPU spends waiting for the host, which prefill graphs would remove.

    python bench/prefill_idle.py --model <file.gguf> --engine-args '<holon serve flags>' [--sizes ...] [--json out]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import shlex
import statistics
import time

import torch
from holon.api import SamplingParams
from holon.core.cli import engine_config, engine_flags
from holon.core.engine import Engine


def turn(engine: Engine, history: list[int], new: list[int], profile: bool) -> tuple[float, float, int]:
    """Prefill `history + new` after `history` is cached: (wall s, kernel s, steps) over its prefill steps."""
    engine.generate([history], SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True))
    engine.add(history + new, SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True))
    prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) if profile else None
    wall = steps = 0
    with prof or torch.autograd.profiler.record_function("turn"):
        while engine.has_work():
            torch.cuda.synchronize()
            t = time.perf_counter()
            engine.step()
            torch.cuda.synchronize()
            wall, steps = wall + time.perf_counter() - t, steps + 1
    kernels = sum(e.device_time_total for e in prof.key_averages()) / 1e6 if prof else 0.0
    return wall, kernels, steps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--engine-args", default="", help="EngineConfig flags, as for `holon serve`")
    ap.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048], help="new tokens")
    ap.add_argument("--history", type=int, default=4096, help="cached tokens before each turn")
    ap.add_argument("-r", type=int, default=5, help="unprofiled repetitions per size (the first is dropped)")
    ap.add_argument("--json")
    args = ap.parse_args()
    flags = argparse.ArgumentParser()
    engine_flags(flags)
    cfg = engine_config(args.model, flags.parse_args(shlex.split(args.engine_args)))
    engine = Engine(dataclasses.replace(cfg, overlap=False))
    rng = random.Random(0)
    tokens = lambda n: [rng.randrange(1000, 50000) for _ in range(n)]     # noqa: E731  (fresh: no cross-turn hits)
    rows = []
    for n in args.sizes:
        walls = [turn(engine, tokens(args.history), tokens(n), False) for _ in range(args.r)][1:]
        _, kernels, steps = turn(engine, tokens(args.history), tokens(n), True)
        wall = statistics.median(w for w, _, _ in walls)
        rows.append({"new_tokens": n, "steps": steps, "wall_ms": round(wall * 1e3, 2),
                     "kernel_ms": round(kernels * 1e3, 2), "idle": round(1 - kernels / wall, 3)})
        print(rows[-1], flush=True)
    out = {"model": args.model.rsplit("/", 1)[-1], "gpu": torch.cuda.get_device_name(), "engine_args": args.engine_args,
           "history": args.history, "rows": rows}
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
