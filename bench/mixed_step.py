"""A mixed step's wall time by GQA's path for its decode rows (docs/dev/m07-perf.md D7, §10 correctness m3): the prefill
kernel over both slices (one call) against the decode kernel for the decode rows beside the prefill kernel for the
chunk (two calls), on one engine.

Requests decode at the given contexts; then a long prompt arrives and prefills in chunks of `chunk_with_decode`
beside them. Each mixed step is timed synchronously (overlap off: a step's wall is its host and GPU work); the arms
alternate per round (ABBA) by flipping every attention planner's `prefill_mixes`, a std.mixers.gqa internal.

    python bench/mixed_step.py --model <file.gguf> --engine-args '<holon serve flags>' --chunk 512 [--json out]
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


def mixed_walls(engine: Engine, contexts: list[int], prompt: int, rng: random.Random) -> tuple[list[float], float]:
    """Walls (s) of the mixed steps while a `prompt`-token request prefills beside requests decoding at `contexts`,
    and the median wall of the decode steps before it."""
    tokens = lambda n: [rng.randrange(1000, 50000) for _ in range(n)]     # noqa: E731  (fresh: no prefix hits)
    params = SamplingParams(temperature=0.0, max_tokens=4096, ignore_eos=True)
    decoders = [engine.add(tokens(c), params) for c in contexts]
    while engine.sched.waiting:
        engine.step()

    def timed() -> float:
        torch.cuda.synchronize()
        t = time.perf_counter()
        engine.step()
        torch.cuda.synchronize()
        return time.perf_counter() - t
    decode = statistics.median(timed() for _ in range(8))
    long = engine.add(tokens(prompt), SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True))
    walls = []
    while any(s.rid == long for s in engine.sched.waiting):
        before = engine.mixed_steps
        wall = timed()
        if engine.mixed_steps > before:
            walls.append(wall)
    for rid in decoders:
        engine.abort(rid)
    while engine.has_work():
        engine.step()
    return walls, decode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--engine-args", default="", help="EngineConfig flags, as for `holon serve`")
    ap.add_argument("--chunk", type=int, default=512, help="chunk_with_decode")
    ap.add_argument("--contexts", type=int, nargs="+", default=[2048, 4096, 6144], help="the decoding requests'")
    ap.add_argument("--prompt", type=int, default=8192, help="the long prompt's tokens")
    ap.add_argument("--rounds", type=int, default=4, help="rounds per arm, ABBA")
    ap.add_argument("--json")
    args = ap.parse_args()
    flags = argparse.ArgumentParser()
    engine_flags(flags)
    cfg = engine_config(args.model, flags.parse_args(shlex.split(args.engine_args)))
    engine = Engine(dataclasses.replace(cfg, overlap=False, chunk_with_decode=args.chunk))
    planners = [p for p in engine.executor.planners.values() if hasattr(p, "prefill_mixes")]
    declared = {p.prefill_mixes for p in planners}
    rng = random.Random(0)
    mixed_walls(engine, args.contexts, args.prompt, rng)                     # warmup (Triton compiles, plans)
    arms = {"one call": True, "two calls": False}
    rows = {name: [] for name in arms}
    for r in range(args.rounds):
        for name in (list(arms) if r % 2 == 0 else list(arms)[::-1]):
            for p in planners:
                p.prefill_mixes = arms[name]
            walls, decode = mixed_walls(engine, args.contexts, args.prompt, rng)
            rows[name].append({"round": r, "mixed_ms_p50": round(statistics.median(walls) * 1e3, 3),
                               "mixed_steps": len(walls), "decode_ms_p50": round(decode * 1e3, 3)})
            print(name, rows[name][-1], flush=True)
    summary = {name: {"mixed_ms": round(statistics.mean(r["mixed_ms_p50"] for r in runs), 3),
                      "sd": round(statistics.stdev(r["mixed_ms_p50"] for r in runs), 3) if len(runs) > 1 else None}
               for name, runs in rows.items()}
    print(summary)
    out = {"model": args.model.rsplit("/", 1)[-1], "gpu": torch.cuda.get_device_name(), "engine_args": args.engine_args,
           "chunk": args.chunk, "contexts": args.contexts, "prompt": args.prompt, "declared_prefill_mixes":
           sorted(declared), "rounds": rows, "summary": summary}
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
