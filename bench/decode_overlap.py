"""Decode wall time per step, overlap (depth 1) vs depth 0, eager engine: prompts of --ctx tokens, batch B."""
import argparse
import json
import pathlib
import time

import torch
from holon.api import SamplingParams
from holon.core.engine import Engine, EngineConfig

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--batches", default="1,8,64")
ap.add_argument("--ctx", type=int, default=1024)
ap.add_argument("--steps", type=int, default=64)
ap.add_argument("--out", default=None)
args = ap.parse_args()

results = []
for overlap in (False, True):
    eng = Engine(EngineConfig(model=args.model, max_model_len=args.ctx + args.steps + 8, max_seqs=64,
                              max_step_tokens=8192, overlap=overlap))
    for b in map(int, args.batches.split(",")):
        g = torch.Generator().manual_seed(b)
        for _ in range(b):
            eng.add(torch.randint(100, 50000, (args.ctx,), generator=g).tolist(),
                    SamplingParams(temperature=0.0, max_tokens=args.steps, ignore_eos=True))
        while eng.sched.waiting:                              # prefill (not timed)
            eng.step()
        torch.cuda.synchronize()
        t, cpu, n = time.perf_counter(), time.process_time(), 0
        while eng.has_work():
            eng.step()
            n += 1
        torch.cuda.synchronize()
        wall, cpu = time.perf_counter() - t, time.process_time() - cpu
        results.append({"overlap": overlap, "batch": b, "step_ms": round(wall / n * 1e3, 3),
                        "cpu_ms_per_step": round(cpu / n * 1e3, 3), "tok_s": round(b * n / wall, 1)})
        print(results[-1], flush=True)
    eng.close()
if args.out:
    command = f"bench/decode_overlap.py --ctx {args.ctx} --steps {args.steps}"
    record = {"gpu": torch.cuda.get_device_name(), "command": command, "results": results}
    pathlib.Path(args.out).write_text(json.dumps(record, indent=1))
