"""Holon's decode measured as bench/freetoken_pp.py measures FreeToken's, for a like-for-like comparison: tg<N> at
depth D and batch B = B prompts of D tokens decoding N tokens each, timed as `generate` of N + 1 tokens minus `generate`
of 1, in tokens per second over the batch; random prompts, the prefix cache off. Mean over `-r` repetitions after a
warm-up.

    python bench/holon_tg.py --model <checkpoint | file.gguf> [-n 128] [--depths 0 8192] [--batch 1 8]
                             [--engine-args "<holon serve flags>"] [-r 5] [--llama-crossover] [--json out]

`--llama-crossover`: kernels.ggml_cuda without its per-shape entries (llama.cpp's rule; D2's baseline).
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("-n", type=int, default=128)
    ap.add_argument("--depths", type=int, nargs="+", default=[0])
    ap.add_argument("--batch", type=int, nargs="+", default=[1])
    ap.add_argument("--engine-args", default="")
    ap.add_argument("-r", type=int, default=5)
    ap.add_argument("--llama-crossover", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.llama_crossover:
        from holon_plugins.kernels.ggml_cuda import plugin as ggml
        ggml.CROSSOVER.clear()
    flags = argparse.ArgumentParser()
    engine_flags(flags)
    cfg = engine_config(a.model, flags.parse_args(shlex.split(a.engine_args)))
    engine = Engine(dataclasses.replace(cfg, max_seqs=max(a.batch), max_model_len=max(a.depths) + a.n + 16,
                                        prefix_cache=False))
    rng = random.Random(0)

    def generate(batch: int, depth: int, tokens: int) -> float:
        prompts = [[rng.randrange(1000, engine.spec.vocab // 2) for _ in range(max(depth, 1))] for _ in range(batch)]
        torch.cuda.synchronize()
        t = time.perf_counter()
        engine.generate(prompts, SamplingParams(temperature=0.0, max_tokens=tokens, ignore_eos=True))
        return time.perf_counter() - t

    tg = {(b, d): [] for b in a.batch for d in a.depths}
    for rep in range(a.r + 1):                              # repetition 0: warm-up (graphs, autotuning), excluded
        for b, d in tg:
            first, total = generate(b, d, 1), generate(b, d, a.n + 1)
            if rep:
                tg[(b, d)].append(b * a.n / (total - first))
    results = {f"tg{a.n}" + (f" @ d{d}" if d else "") + (f" b{b}" if b > 1 else ""):
               {"mean": round(statistics.mean(xs), 2), "runs": [round(x, 2) for x in xs]} for (b, d), xs in tg.items()}
    print(json.dumps(results, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"engine": "holon", "model": a.model.rstrip("/").rsplit("/", 1)[-1],
                       "gpu": torch.cuda.get_device_name(), "llama_crossover": a.llama_crossover, "results": results},
                      f, indent=1)


if __name__ == "__main__":
    main()
