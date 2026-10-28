"""F5b's decode gain on the development vehicle (docs/dev/v1-offload.md §13, step 6): decode tok/s of arms of the offload
store — F5a and F5b with a split table and worker count — on natural prompts (the first `--batch` prompts of
judge/prompts/v1.jsonl, cut to `--depth` tokens), timed as bench/holon_tg.py times it (generate N + 1 tokens minus
generate 1), `--rounds` per arm after a warm-up round (one engine at a time: name an arm twice, A B A B, to interleave),
with each arm's CPU share from the route taps.

    python spikes/x6_cpu/decode.py <model.gguf> <out.json> "<name>|<plugin options k=v;k=v>" ... [--batch 1]
           (`host_memory=<bytes>` among an arm's options sets its engine's: a RAM-capped arm)
           [--tokens 128] [--depth 512] [--rounds 3] [--engine-args "<holon serve flags>"]
"""
import argparse
import dataclasses
import gc
import json
import shlex
import statistics
import time

import numpy as np
import torch
from holon.api import SamplingParams
from holon.core.cli import engine_config, engine_flags
from holon.core.engine import Engine
from holon_judge.recording import PromptSet
from holon_std.moe.offload import KEY


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--depth", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--engine-args", default="")
    a = ap.parse_args()
    flags = argparse.ArgumentParser()
    engine_flags(flags)
    base = engine_config(a.model, flags.parse_args(shlex.split(a.engine_args)))
    prompts = [p.ids[:a.depth] for p in PromptSet.load("judge/prompts/v1.jsonl").prompts][:a.batch]
    params = SamplingParams(temperature=0.0, max_tokens=a.tokens + 1, ignore_eos=True)
    first = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    results = {}
    for arm in a.arms:                                   # one engine at a time: measure all rounds, keep the numbers
        name, _, opts = arm.partition("|")
        seen = [k for k in results if k.split("#")[0] == name]
        name += f"#{len(seen)}" if seen else ""
        extra = dict(kv.split("=", 1) for kv in opts.split(";") if kv)
        host = int(extra.pop("host_memory", base.host_memory))     # (an engine setting: a RAM-capped arm's)
        cfg = dataclasses.replace(base, max_seqs=a.batch, prefix_cache=False, host_memory=host,
                                  plugin_options=base.plugin_options | extra)
        t0 = time.time()
        engine = Engine(cfg)
        built = time.time() - t0
        store = engine.ctx.shared_objects.get(KEY)
        steps = []
        if store is not None:
            store.taps.append(lambda step, record, steps=steps: steps.append(
                (record["routed"].sum(), record["cpu"].sum(), int(record["counts"].sum()))))
        rates = []
        for r in range(a.rounds + 1):                    # round 0: warm-up
            torch.cuda.synchronize()
            t = time.perf_counter()
            engine.generate(prompts, first)
            one = time.perf_counter() - t
            steps.clear()
            t = time.perf_counter()
            engine.generate(prompts, params)
            total = time.perf_counter() - t
            if r:
                rates.append(a.batch * a.tokens / (total - one))
        resident = None if store is None else int((store.residency.host >= 0).all(-1).sum())
        decode = [s for s in steps if s[2]]
        io = engine.ctx.shared_objects.get("cpu_task.io")
        results[name] = {"options": extra, "host_memory": host, "tok_s": rates,
                         "io_served": io.native.served() if io else None, "median": statistics.median(rates),
                         "build_s": round(built),
                         "resident_experts": resident,
                         "cpu_experts_per_token": float(np.sum([s[1] for s in decode]) / max(1, a.batch * len(decode))),
                         "worker": None if "cpu_task.compute" not in engine.ctx.shared_objects else
                         {"served": engine.ctx.shared_objects["cpu_task.compute"].native.served(),
                          "flag": engine.ctx.shared_objects["cpu_task.compute"].native.flag(),
                          "cores": engine.ctx.shared_objects["cpu_task.compute"].cores}}
        print(json.dumps({name: results[name]}), flush=True)
        engine.close()
        del engine, store
        gc.collect()
        torch.cuda.empty_cache()
    with open(a.out, "w") as f:
        json.dump({"args": vars(a), "gpu": torch.cuda.get_device_name(), "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
