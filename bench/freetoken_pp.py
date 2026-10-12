"""FreeToken's prefill in bench/llama_bench.py's terms, run in FreeToken's own environment: pp<P> = one P-token
prefill, timed to its first token (offline `LLM.generate` with `max_tokens=1`), random prompts; pp<P> at depth D =
P new tokens after a D-token context its radix cache holds (generated once before, untimed). Mean ± sd over `-r`
repetitions after a warm-up, like llama-bench. tg<N> at depth D and batch B (`-n`, `--depths`, `--batch`): B prompts of
D tokens decoding N tokens each, timed as generating N + 1 minus generating 1 (FreeToken's offline API returns no
per-token times), in tokens per second over the batch.

    python bench/freetoken_pp.py --model <hf checkpoint dir> [-p 512 ...] [--pp-depths 8192 ...] [-n 128]
                                 [--depths 0 8192] [--batch 1 8] [-r 5] [--json out]
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
    ap.add_argument("-n", type=int, default=0, help="decode tokens of the tg tests (0: none)")
    ap.add_argument("--depths", type=int, nargs="+", default=[0], help="prompt tokens before the tg tests")
    ap.add_argument("--batch", type=int, nargs="+", default=[1], help="sequences decoding together")
    ap.add_argument("-r", type=int, default=5)
    ap.add_argument("--memory-ratio", type=float, default=0.9)
    ap.add_argument("--json")
    a = ap.parse_args()
    longest = max(max(a.p) + max(a.pp_depths, default=0), max(a.depths) + a.n + 1) + 8
    llm = LLM(a.model, dtype=torch.bfloat16, max_running_req=max(a.batch), max_extend_tokens=max(*a.p, *a.depths, 1),
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

    def generate(batch: int, depth: int, tokens: int) -> float:
        prompts = [prompt(max(depth, 1)) for _ in range(batch)]
        torch.cuda.synchronize()
        t = time.perf_counter()
        llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=tokens, ignore_eos=True))
        return time.perf_counter() - t

    tg = {(b, d): [] for b in a.batch for d in a.depths} if a.n else {}
    for rep in range(a.r + 1 if a.n else 0):
        for b, d in tg:
            first, total = generate(b, d, 1), generate(b, d, a.n + 1)
            if rep:
                tg[(b, d)].append(b * a.n / (total - first))
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
    for (b, d), xs in tg.items():
        results[f"tg{a.n}" + (f" @ d{d}" if d else "") + (f" b{b}" if b > 1 else "")] = {
            "mean": round(statistics.mean(xs), 2), "runs": [round(x, 2) for x in xs]}
    record = {"engine": "freetoken", "model": a.model.rstrip("/").rsplit("/", 1)[-1],
              "gpu": torch.cuda.get_device_name(), "results": results}
    print(json.dumps(results, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(record, f, indent=1)


if __name__ == "__main__":
    main()
