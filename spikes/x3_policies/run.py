"""Snapshot traces (judge `snapshot.trace` over judge/prompts/v1.jsonl) of the vehicle on fresh engines, with adaptive
placement on (docs/v1-offload.md §6, §9 step 3): `std.moe@1` all-resident, and the offload store at a fit under a
placement policy, optionally with a planted bug or the n-gram drafter:

    python spikes/x3_policies/run.py <model.gguf> <out dir> <runs> "<name>|<cache>|<policy>|<plant>|<spec>[|<t>[|<o>]]"

`<cache>` is `resident` or a `moe.cache` value; `<policy>` a `moe.placement` Ref (empty: the default, F11);
`<plant>` a judge plant; `<spec>` `ngram` for the n-gram drafter (3 drafts, the rejection verifier); `<t>` `decode`: the
decode-heavy parity trace instead of the snapshot trace (the first 8 prompts, 256 greedy tokens each); `<o>`:
`seqs=<n>` (max_seqs), `ring=<n>` (moe.ring_tokens), `chunk=<n>` (chunk_with_decode: mixed steps) and `step=<n>`
(max_step_tokens), comma-separated. Every
engine keeps enough KV that the trace never preempts. Each run writes its snapshot and a summary (admissions and delta
records from the route dump, the policy's swap rounds, F11's L1 parity against its replay oracle, the error that
stopped the engine if one did); the first run of each F11 configuration also writes its route trace (`.npz`).
"""
import contextlib
import gc
import json
import pathlib
import sys
import time

import numpy as np
import torch
from holon.api import SamplingParams
from holon.core.engine import Engine, EngineConfig
from holon_judge import placement, snapshot
from holon_judge.planted import PLANTS
from holon_judge.recording import PromptSet
from holon_std.moe.offload import KEY

ENGINE = dict(dtype=torch.bfloat16, page_size=16, max_seqs=64, max_step_tokens=4096, max_model_len=8192,
              cuda_graphs=True, overlap=True, paged_min_tokens=16384)
SPEC = dict(max_draft_tokens=3, slots={"drafter": "drafters.ngram", "draft_policy": "std.spec.fixed_k",
                                       "verifier": "std.spec.rejection"})


def main() -> None:
    model, out, runs = sys.argv[1], pathlib.Path(sys.argv[2]), int(sys.argv[3])
    prompts = PromptSet.load("judge/prompts/v1.jsonl")
    traces = {"": snapshot.trace(prompts), "decode": [
        snapshot.Request(p.ids, SamplingParams(temperature=0.0, max_tokens=256, ignore_eos=True))
        for p in prompts.prompts[:8]]}
    out.mkdir(parents=True, exist_ok=True)
    for config in sys.argv[4:]:
        name, cache, policy, plant, spec, *kind = config.split("|")
        trace = traces[kind[0] if kind else ""]
        extra = dict(o.split("=") for o in (kind[1].split(",") if len(kind) > 1 and kind[1] else []))
        for run in range(runs):
            kw = dict(SPEC, plugin_options={"spec.k": "3", "spec.fallback": 2}) if spec else {}
            if cache != "resident":
                kw = kw | dict(host_memory=20 << 30, plugin_options=kw.get("plugin_options", {}) | {
                    "moe.store": "std.moe.offload@1", "moe.cache": cache, "moe.placement": policy} | (
                    {"moe.ring_tokens": int(extra["ring"])} if "ring" in extra else {}) | (
                    {"moe.ring_copy": extra["copy"]} if "copy" in extra else {}))
            for key, field in (("seqs", "max_seqs"), ("chunk", "chunk_with_decode"), ("step", "max_step_tokens")):
                if key in extra:
                    kw[field] = int(extra[key])
            meta = {"name": name, "cache": cache, "policy": policy or "placement.lfu@1", "plant": plant,
                    "spec": spec, "trace": kind[0] if kind and kind[0] else "snapshot", "run": run, "options": extra,
                    "reference": "resident" + "_ngram" * bool(spec) + "_decode" * bool(kind and kind[0])
                    + "".join(f"_{k}{v}" for k, v in sorted(extra.items()) if k not in ("ring", "copy"))}
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
                except Exception as e:                      # a plant that stops the engine
                    meta["error"] = f"{type(e).__name__}: {e}"[:400]
                meta["run_s"] = round(time.time() - t - meta["build_s"], 1)
            if store is not None:
                meta |= {"peak_misses": store.peak_misses, "steps": len(routes.steps), "staging": store.staging,
                         "borrowed": dict(store.borrowed),
                         "admissions": routes.admissions,
                         "records": sum(len(s["records"]) for s in routes.steps),
                         "swap_rounds": getattr(store.policy, "rounds", None),
                         "cached_per_class": {c.name: c.size - c.null - c.staging for c in store.classes.values()}}
                if meta["policy"] == "placement.lfu@1" and "error" not in meta:
                    meta["parity"] = placement.parity(routes)
                    if run == 0:
                        np.savez_compressed(out / f"{name}_routes.npz", seed=routes.seed,
                                            counts=np.stack([s["counts"] for s in routes.steps]),
                                            needed=np.stack([s["needed"] for s in routes.steps]),
                                            resident=np.stack([s["resident"] for s in routes.steps]))
            (out / f"{name}_{run}.meta.json").write_text(json.dumps(meta, indent=1) + "\n")
            print(json.dumps(meta), flush=True)
            engine.close()
            del engine, store
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
