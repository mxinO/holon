"""Decode step time with each attention.paged_decode kernel (engine, eager): prompts of --ctx tokens, batch B."""
import argparse
import json
import pathlib
import statistics
import time

import torch
from holon.api import SamplingParams
from holon.core.engine import Engine, EngineConfig

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--batches", default="1,8,64")
ap.add_argument("--ctx", type=int, default=1024)
ap.add_argument("--steps", type=int, default=32)
ap.add_argument("--out", default=None)
args = ap.parse_args()

results = []
for provider in ("kernels.triton_attn", "kernels.flashinfer"):
    eng = Engine(EngineConfig(model=args.model, max_model_len=args.ctx + args.steps + 8, max_seqs=64,
                              max_step_tokens=8192, kernels={"attention.paged_decode": provider}))
    for b in map(int, args.batches.split(",")):
        g = torch.Generator().manual_seed(b)
        for _ in range(b):
            eng.add(torch.randint(100, 50000, (args.ctx,), generator=g).tolist(),
                    SamplingParams(temperature=0.0, max_tokens=args.steps, ignore_eos=True))
        times = []
        while eng.has_work():
            decode = not eng.sched.waiting
            torch.cuda.synchronize()
            t = time.perf_counter()
            eng.step()
            torch.cuda.synchronize()
            if decode:
                times.append(time.perf_counter() - t)
        ms = statistics.median(times[2:]) * 1e3
        results.append({"kernel": provider, "batch": b, "ctx": args.ctx, "step_ms": round(ms, 3),
                        "tok_s": round(b / ms * 1e3, 1)})
        print(results[-1], flush=True)
    del eng
if args.out:
    pathlib.Path(args.out).write_text(json.dumps({"gpu": torch.cuda.get_device_name(), "results": results}, indent=1))
