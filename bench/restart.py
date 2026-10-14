"""Supervisor restart time: kill the engine mid-stream and measure how long until it serves again (evidence record).

A greedy stream runs under the supervisor (default engine config: overlap, CUDA graphs); every `--every` tokens the
engine process is killed. Recorded per kill: the supervisor's restart time (engine process launch → ready: weights,
pool sizing, graph capture) and the stall the client sees (last token before the kill → first token after it). The
stream must complete: its whole token budget arrives. (That it continues exactly as an uninterrupted stream would, with
no token duplicated or lost, is tested in fp32 on the CPU; in bf16 a restart recomputes the streamed tokens' state by
prefill, which may flip a later greedy near-tie.)

Usage (from the repository root, inside Holon's .venv):
  python bench/restart.py --gpu 6 --model <checkpoint dir> --kills 5
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import multiprocessing
import os
import statistics
import time

from run import ROOT, git, gpu_class, model_id


async def measure(model: str, kills: int, every: int) -> dict:
    from holon.api import SamplingParams
    from holon.core.engine import EngineConfig
    from holon.core.supervisor import Policy, supervised

    params = SamplingParams(temperature=0.0, max_tokens=every * (kills + 1), ignore_eos=True)
    prompt = list(range(1000, 1064))
    # Deliberate kills: the stream must survive them all, and a lone stream killed again and again is not poison.
    policy = Policy(max_crashes=kills + 1, poison_crashes=kills + 1)
    async with supervised([(EngineConfig(model=model), None)], policy) as sup:
        tokens, stalls, restarts, killed_at = [], [], [], None
        async for d in sup.generate(prompt, params):
            now = time.perf_counter()
            if killed_at is not None and d.token_ids:
                stalls.append(now - killed_at)
                restarts.append((await sup.stats())["restart_s"])
                killed_at = None
            tokens += d.token_ids
            if len(tokens) % every == 0 and len(stalls) < kills and killed_at is None and d.finish_reason is None:
                for proc in multiprocessing.active_children():
                    proc.kill()
                killed_at = time.perf_counter()
        stats = await sup.stats()
    return {"start_s": round(stats["start_s"], 2), "restart_s": [round(s, 2) for s in restarts],
            "stall_s": [round(s, 2) for s in stalls], "restart_median_s": round(statistics.median(restarts), 2),
            "restart_max_s": round(max(restarts), 2),
            "crashes": stats["crashes"], "tokens": len(tokens), "complete": len(tokens) == params.max_tokens}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--model", required=True, help="checkpoint directory")
    parser.add_argument("--kills", type=int, default=5)
    parser.add_argument("--every", type=int, default=64, help="tokens between kills")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    result = asyncio.run(measure(args.model, args.kills, args.every))
    date = datetime.date.today().isoformat()
    gpu = gpu_class(args.gpu)
    record = {"workload": "restart", "date": date, "gpu_class": gpu, "model": model_id(args.model),
              "command": f"python bench/restart.py --gpu {args.gpu} --model {model_id(args.model)} --kills {args.kills}"
                         f" --every {args.every}",
              "commit": git("rev-parse", "HEAD"), **result}
    name = model_id(args.model).split("@")[0].split("/")[-1].lower()
    out = ROOT / "evidence" / f"restart_{name}_{gpu}_{date}.json"
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
