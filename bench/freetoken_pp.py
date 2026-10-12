"""FreeToken's prefill in bench/llama_bench.py's terms, run in FreeToken's own environment: pp<P> = one P-token
prefill, timed to its first token (offline `LLM.generate` with `max_tokens=1`), random prompts; pp<P> at depth D =
P new tokens after a D-token context its radix cache holds (generated once before, untimed). Mean ± sd over `-r`
repetitions after a warm-up, like llama-bench.

    python bench/freetoken_pp.py --model <hf checkpoint dir> [-p 512 ...] [--pp-depths 8192 ...] [-r 5] [--json out]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time

import torch
from freetoken.core import SamplingParams
from freetoken.llm import LLM


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("-p", type=int, nargs="+", default=[512])
    ap.add_argument("--pp-depths", type=int, nargs="+", default=[])
    ap.add_argument("-r", type=int, default=5)
    ap.add_argument("--memory-ratio", type=float, default=0.9)
    ap.add_argument("--json")
    a = ap.parse_args()
    longest = max(a.p) + max(a.pp_depths, default=0) + 8
    llm = LLM(a.model, dtype=torch.bfloat16, max_running_req=1, max_extend_tokens=max(a.p),
              memory_ratio=a.memory_ratio, max_seq_len_override=longest)
    rng = random.Random(0)
    vocab = len(llm.tokenizer)
    one = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)

    def prompt(n: int) -> list[int]:
        return [rng.randrange(1000, vocab // 2) for _ in range(n)]

    def pp_case(p: int, depth: int) -> float:
        context = prompt(depth)
        if depth:
            llm.generate([context], one)                        # fill and cache the context (untimed)
        torch.cuda.synchronize()
        t = time.perf_counter()
        llm.generate([context + prompt(p)], one)
        return time.perf_counter() - t

    cases = [(p, depth) for depth in (0, *a.pp_depths) for p in a.p]
    walls = {c: [] for c in cases}
    for rep in range(a.r + 1):                                  # repetition 0: warm-up, excluded
        for c in cases:
            wall = pp_case(*c)
            if rep:
                walls[c].append(wall)
    results = {}
    for (p, depth), ws in walls.items():
        tps = [p / w for w in ws]
        results[f"pp{p}" + (f" @ d{depth}" if depth else "")] = {
            "mean": round(statistics.mean(tps), 2), "sd": round(statistics.stdev(tps), 2) if len(tps) > 1 else 0.0,
            "runs": [round(x, 2) for x in tps], "wall_ms": round(statistics.median(ws) * 1e3, 2)}
    record = {"engine": "freetoken", "model": a.model.rstrip("/").rsplit("/", 1)[-1],
              "gpu": torch.cuda.get_device_name(), "results": results}
    print(json.dumps(results, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(record, f, indent=1)


if __name__ == "__main__":
    main()
