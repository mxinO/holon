"""Step 7's held-out plugins on the development vehicle (docs/v1-offload.md acceptance 1, §9 step 7): snapshot traces
(judge `snapshot.trace` over judge/prompts/v1.jsonl, the step-3 engine settings) on fresh engines, and F10's L1 parity
on a decode trace:

    python spikes/x7_heldout/run.py <model.gguf> <out dir> "<name>|<runs>|<trace>|<plugin options k=v;k=v>" ...

`<trace>`: `snapshot`, or `parity` (the first 8 prompts, 256 greedy tokens each, eager and synchronous — the routes are
read at the resolves —: F10's residency and empty slots after every step against placement.promote's replay oracle,
and the host mirror the device's map but for the spares' owners). Empty options: all-resident (`std.moe@1`);
`host_memory=<bytes>` among them sets the engine's (default 20 GiB). Each run writes `<name>_<run>.json` (its
snapshot) and `.meta.json` (the store's counts; `diff` against the all-resident run of its trace, `resident_0.json` or
`resident-parity_0.json`: null = byte-identical; the parity verdict)."""
import contextlib
import gc
import json
import pathlib
import sys
import time
from unittest import mock

import numpy as np
import torch
from holon.api import SamplingParams
from holon.core.engine import Engine, EngineConfig
from holon_judge import snapshot
from holon_judge.recording import PromptSet
from holon_plugins.placement.promote import reference
from holon_std.moe.offload import KEY
from holon_std.tiered.resolver import DeviceResolver

ENGINE = dict(dtype=torch.bfloat16, page_size=16, max_seqs=64, max_step_tokens=4096, max_model_len=8192,
              cuda_graphs=True, overlap=True, paged_min_tokens=16384)
DECODE = 1


@contextlib.contextmanager
def routes(engine):
    """Per committed step (synchronous path): each layer's decode routes in the order of their first entry, the step's
    `lend` needs, the device's residency (whole experts) and empty slots after it, and its promotions (records)."""
    store = engine.ctx.shared_objects[KEY]
    trace = {"seed": (store.residency.host >= 0).all(-1), "steps": []}
    calls, call = [[] for _ in store.layers], DeviceResolver.__call__

    def resolve(self, ids, first, *args, **kw):
        decode = ids.cpu()[store.kinds[first:first + len(ids)].cpu() == DECODE]
        calls[self.group] += [e for e in decode.view(-1).tolist() if e >= 0]
        return call(self, ids, first, *args, **kw)

    def tap(step, record) -> None:
        trace["steps"].append({"order": [list(dict.fromkeys(c)) for c in calls], "lends": record["lends"],
                               "resident": (store.residency.device >= 0).all(-1).cpu().numpy(),
                               "empty": store.n_empty.cpu().numpy().copy(),
                               "records": record["records"].copy()})
        for c in calls:
            c.clear()
    store.taps.append(tap)
    try:
        with mock.patch.object(DeviceResolver, "__call__", resolve):
            yield trace
    finally:
        store.taps.remove(tap)


def parity(store, trace) -> dict:
    want = reference.replay(trace["seed"], trace["steps"], store.class_table)
    device, host = store.residency.device.cpu().numpy(), store.residency.host.copy()
    owners_mapped = True
    for g, rows in enumerate(store.state["spare"].cpu().numpy()):
        for o, *slots in rows[rows[:, 0] > 0]:
            owners_mapped &= bool((host[g, o - 1] == slots).all())
            host[g, o - 1] = -1
    differs = next((n for n, (w, s) in enumerate(zip(want, trace["steps"], strict=True))
                    if not (np.array_equal(w["resident"], s["resident"]) and np.array_equal(w["empty"], s["empty"]))),
                   None)
    records = np.concatenate([s["records"] for s in trace["steps"]])
    return {"steps": len(trace["steps"]), "decode_steps": sum(any(map(len, s["order"])) for s in trace["steps"]),
            "borrowing_steps": sum(s["lends"] is not None for s in trace["steps"]),
            "promotions_logged": int((records[:, 1] >= 0).sum()) if len(records) else 0,
            "first_differing_step": differs,
            "mirror_is_device_but_owners": owners_mapped and bool(np.array_equal(np.maximum(device, -1),
                                                                                 np.maximum(host, -1)))}


def main() -> None:
    model, out = sys.argv[1], pathlib.Path(sys.argv[2])
    prompts = PromptSet.load("judge/prompts/v1.jsonl")
    traces = {"snapshot": snapshot.trace(prompts), "parity": [
        snapshot.Request(p.ids, SamplingParams(temperature=0.0, max_tokens=256, ignore_eos=True))
        for p in prompts.prompts[:8]]}
    out.mkdir(parents=True, exist_ok=True)
    for config in sys.argv[3:]:
        name, runs, kind, opts = config.split("|")
        options = dict(kv.split("=", 1) for kv in opts.split(";") if kv)
        host = int(options.pop("host_memory", 20 << 30))
        for run in range(int(runs)):
            kw = dict(host_memory=host, plugin_options=options) if options else {}
            if kind == "parity":
                kw |= dict(cuda_graphs=False, overlap=False)
            meta = {"name": name, "trace": kind, "options": options, "run": run, "host_memory": host if options else 0}
            t = time.time()
            engine = Engine(EngineConfig(model=model, **(ENGINE | kw)))
            meta["build_s"] = round(time.time() - t, 1)
            store = engine.ctx.shared_objects.get(KEY)
            records = []
            if store is not None:
                store.taps.append(lambda step, record, out=records: out.append(record["records"].copy()))
            dump = routes(engine) if kind == "parity" and store is not None else contextlib.nullcontext()
            try:
                with dump as trace:
                    snap = snapshot.record(engine, traces[kind])
                snap.save(out / f"{name}_{run}.json")
            except Exception as e:                       # an engine that stopped
                meta["error"] = f"{type(e).__name__}: {e}"[:400]
            meta["run_s"] = round(time.time() - t - meta["build_s"], 1)
            reference_snap = out / ("resident_0.json" if kind == "snapshot" else f"resident-{kind}_0.json")
            if "error" not in meta and reference_snap.exists() and not name.startswith("resident"):
                meta["diff"] = snapshot.diff(snapshot.Snapshot.load(reference_snap), snap)
            if store is not None:
                logged = np.concatenate(records) if records else np.zeros((0, 3), dtype=np.int32)
                meta |= {"steps": len(records), "admissions_logged": int((logged[:, 1] >= 0).sum()),
                         "peak_misses": store.peak_misses, "borrowed": dict(store.borrowed),
                         "exclusive": store.exclusive, "tiers": [type(t).__name__ for t in store.tiers],
                         "cached_per_class": {c.name: c.size - c.null - c.staging for c in store.classes.values()},
                         "ram_per_class": {c.name: c.held for c in store.classes.values()}}
                if kind == "parity" and "error" not in meta:
                    meta["parity"] = parity(store, trace)
            (out / f"{name}_{run}.meta.json").write_text(json.dumps(meta, indent=1) + "\n")
            print(json.dumps(meta), flush=True)
            engine.close()
            del engine, store
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
