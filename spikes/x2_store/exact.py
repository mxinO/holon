"""Snapshot traces (judge `snapshot.trace` over judge/prompts/v1.jsonl) of the vehicle on fresh engines: `std.moe@1`
all-resident, and the offload store at given cache fits (docs/v1-offload.md §6, step 2):

    python spikes/x2_store/exact.py <model.gguf> <out dir> "resident@16384;*=1.0@16384;@16384;*=0.25@16384" [runs]

A fit is `resident` or a `moe.cache` value (empty: fill), after `@` the paged-state floor in tokens; every
engine keeps enough KV that the trace never preempts, so the schedule is the trace's. Compare with
`python -m holon_judge snapshot-diff` or `record.py`.
"""
import gc
import json
import pathlib
import sys
import time

import torch
from holon.core.engine import Engine, EngineConfig
from holon_judge import snapshot
from holon_judge.recording import PromptSet

ENGINE = dict(dtype=torch.bfloat16, page_size=16, max_seqs=64, max_step_tokens=4096, max_model_len=8192,
              cuda_graphs=True, overlap=True)


def main() -> None:
    model, out, fits = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3].split(";")
    runs = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    trace = snapshot.trace(PromptSet.load("judge/prompts/v1.jsonl"))
    for spec in fits:
        fit, _, floor = spec.partition("@")
        for run in range(runs):
            kw = {} if fit == "resident" else dict(host_memory=20 << 30, plugin_options={
                "moe.store": "std.moe.offload@1", "moe.cache": fit})
            t = time.time()
            engine = Engine(EngineConfig(model=model, paged_min_tokens=int(floor or 32768), **ENGINE, **kw))
            built = time.time() - t
            snap = snapshot.record(engine, trace)
            store = engine.ctx.shared_objects.get("std.moe.offload")
            snap.meta.update(fit=fit, paged_min_tokens=int(floor or 32768), run=run, build_s=round(built, 1),
                             run_s=round(time.time() - t - built, 1), peak_misses=getattr(store, "peak_misses", None),
                             pools={n: p.capacity for n, p in engine.ledger.pools.items()})
            name = "resident" if fit == "resident" else "offload_" + (fit.replace("*=", "") or "fill")
            snap.save(out / f"{name}_{run}.json")
            print(json.dumps({k: snap.meta[k] for k in ("fit", "run", "build_s", "run_s", "peak_misses", "pools")}),
                  flush=True)
            engine.close()
            del engine
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
