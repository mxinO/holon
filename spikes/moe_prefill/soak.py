"""Step 4's 16 GB soak (docs/dev/v1-moe-prefill.md §7, the panel's C6): a lock's engine (its flags, `--ctx`, `--max-seqs`),
then `--rounds` × (one long prompt of `--long` tokens, then `--seats` short prompts decoding together), every request
greedy. Per round: the caching allocator's `num_alloc_retries` / `num_device_alloc` / `num_device_free` deltas, the
allocation peak, and the least free device memory seen after any step (`torch.cuda.mem_get_info`). An OOM fails the
run (it raises). Prints `RESULT {...}`.

    python spikes/moe_prefill/soak.py --model <gguf> --flags '<engine flags>' [--ctx 16384] [--max-seqs 32] \\
        [--long 16000] [--seats 32] [--short 512] [--decode 128] [--rounds 4]
"""
import argparse
import dataclasses
import json
import pathlib
import shlex
import sys

import torch
from holon.api import SamplingParams
from holon.core.cli import engine_config, engine_flags
from holon.core.engine import Engine

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
import offload_ctx  # noqa: E402
import workloads  # noqa: E402

COUNTERS = ("num_alloc_retries", "num_device_alloc", "num_device_free")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--flags", default="")
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--max-seqs", type=int, default=32)
    ap.add_argument("--long", type=int, default=16000)
    ap.add_argument("--seats", type=int, default=32)
    ap.add_argument("--short", type=int, default=512)
    ap.add_argument("--decode", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=4)
    a = ap.parse_args()
    flags = argparse.ArgumentParser()
    engine_flags(flags)
    cfg = engine_config(a.model, flags.parse_args(shlex.split(a.flags)))
    engine = Engine(dataclasses.replace(cfg, max_seqs=a.max_seqs, max_model_len=a.ctx, prefix_cache=False))
    pools = [p for p in engine.ledger.pools.values() if p.tokens_per_unit]
    kv = min((p.capacity - p.has_null) * p.tokens_per_unit for p in pools)
    corpus = workloads.corpus(offload_ctx.CORPUS)
    rounds = []
    for r in range(a.rounds):
        torch.cuda.synchronize()
        before, least = torch.cuda.memory_stats(), torch.cuda.mem_get_info()[0]
        torch.cuda.reset_peak_memory_stats()

        def run(prompts: list[list[int]], tokens: int) -> None:
            nonlocal least
            params = SamplingParams(temperature=0.0, max_tokens=tokens, ignore_eos=True)
            for ids in prompts:
                engine.add(ids, params)
            while engine.has_work():
                engine.step()
                least = min(least, torch.cuda.mem_get_info()[0])
        start = 7919 * r
        run([[corpus[(start + i) % len(corpus)] for i in range(a.long)]], 1)
        run([[corpus[(start + 100003 * (s + 1) + i) % len(corpus)] for i in range(a.short)] for s in range(a.seats)],
            a.decode)
        torch.cuda.synchronize()
        after = torch.cuda.memory_stats()
        rounds.append({k: after.get(k, 0) - before.get(k, 0) for k in COUNTERS}
                      | {"least_free_mib": least >> 20, "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3)})
        print(json.dumps({"round": r} | rounds[-1]), flush=True)
    engine.close()
    print("RESULT " + json.dumps({"gpu": torch.cuda.get_device_name(), "kv_tokens": kv, "ctx": a.ctx,
                                  "max_seqs": a.max_seqs, "long": a.long, "seats": a.seats, "short": a.short,
                                  "decode": a.decode, "rounds": rounds}), flush=True)


if __name__ == "__main__":
    main()
