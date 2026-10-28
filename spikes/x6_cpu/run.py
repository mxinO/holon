"""Snapshot traces (judge `snapshot.trace` over judge/prompts/v1.jsonl) of the development vehicle on fresh engines,
as spikes/x3_policies/run.py records them (the same engine settings), for step 6's checks (docs/dev/v1-offload.md §6,
§9): F5a configurations byte-identical to all-resident, F5b run-identical with adaptive placement on, the F5b plants.

    python spikes/x6_cpu/run.py <model.gguf> <out dir> "<name>|<runs>|<plant>|<plugin options k=v;k=v>" ...

Empty options: all-resident (`std.moe@1`); `host_memory=<bytes>` among them sets the engine's (default 21 GiB). Each
run writes `<name>_<run>.json` (its snapshot) and `.meta.json` (the store's and the CPU worker's counts, F11's L1
parity under `placement.lfu@1`, the error that stopped the engine if one did)."""
import contextlib
import gc
import json
import pathlib
import sys
import time

import torch
from holon.core.engine import Engine, EngineConfig
from holon_judge import placement, snapshot
from holon_judge.planted import PLANTS
from holon_judge.recording import PromptSet
from holon_std.moe.offload import CPU, KEY

ENGINE = dict(dtype=torch.bfloat16, page_size=16, max_seqs=64, max_step_tokens=4096, max_model_len=8192,
              cuda_graphs=True, overlap=True, paged_min_tokens=16384)


def main() -> None:
    model, out = sys.argv[1], pathlib.Path(sys.argv[2])
    trace = snapshot.trace(PromptSet.load("judge/prompts/v1.jsonl"))
    out.mkdir(parents=True, exist_ok=True)
    for config in sys.argv[3:]:
        name, runs, plant, opts = config.split("|")
        options = dict(kv.split("=", 1) for kv in opts.split(";") if kv)
        host = int(options.pop("host_memory", 21 << 30))          # (an engine setting: a RAM-capped run's)
        for run in range(int(runs)):
            kw = dict(host_memory=host, plugin_options=options) if options else {}
            meta = {"name": name, "plant": plant, "options": options, "run": run}
            t = time.time()
            with PLANTS[plant]() if plant else contextlib.nullcontext():
                engine = Engine(EngineConfig(model=model, **(ENGINE | kw)))
                meta["build_s"] = round(time.time() - t, 1)
                store = engine.ctx.shared_objects.get(KEY)
                dump = placement.route_dump(engine) if store is not None else contextlib.nullcontext()
                try:
                    with dump as routes:
                        snap = snapshot.record(engine, trace)
                    snap.save(out / f"{name}_{run}.json")
                except Exception as e:                       # a plant that stops the engine
                    meta["error"] = f"{type(e).__name__}: {e}"[:400]
                meta["run_s"] = round(time.time() - t - meta["build_s"], 1)
            if store is not None:
                meta |= {"steps": len(routes.steps), "admissions": routes.admissions, "peak_misses": store.peak_misses,
                         "cpu_experts": int(sum(s["cpu"].sum() for s in routes.steps)),
                         "cached_per_class": {c.name: c.size - c.null - c.staging for c in store.classes.values()}}
                if CPU in engine.ctx.shared_objects:
                    worker = engine.ctx.shared_objects["cpu_task.compute"]
                    meta |= {"worker": {"served": worker.native.served(), "flag": worker.native.flag(),
                                        "threads": len(worker.cores)}}
                if store.policy_ref == "placement.lfu@1" and "error" not in meta:
                    meta["parity"] = placement.parity(routes)
            (out / f"{name}_{run}.meta.json").write_text(json.dumps(meta, indent=1) + "\n")
            print(json.dumps(meta), flush=True)
            engine.close()
            del engine, store
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
