"""An F11-representative frozen placement for A1's human gate (docs/v1-offload.md §6): F11 (`placement.lfu@1`) runs the
parity decode trace (the first 8 prompts, 256 greedy tokens each, the step-3 engine settings) at a cache fit, and the
experts it holds at the end become a routing profile `std.tiered.static@1` seeds from (each resident expert counts 1 +
its F11 score, every other 0), written beside a link to the checkpoint (a profile is named relative to the
checkpoint's directory); a static engine built from it then reports how much of F11's placement it holds:

    python spikes/x7_heldout/placement.py <model.gguf> <out dir> <name> <moe.cache>

Writes `<out dir>/<model file>` (a hard link: same filesystem), `<out dir>/<name>.json` (the profile) and
`<name>.meta.json`."""
import gc
import json
import os
import pathlib
import sys

import numpy as np
import torch
from holon.api import SamplingParams
from holon.core.engine import Engine, EngineConfig
from holon_judge import snapshot
from holon_judge.recording import PromptSet
from holon_std.moe.offload import KEY

ENGINE = dict(dtype=torch.bfloat16, page_size=16, max_seqs=64, max_step_tokens=4096, max_model_len=8192,
              cuda_graphs=True, overlap=True, paged_min_tokens=16384, host_memory=20 << 30)


def resident(engine) -> tuple[np.ndarray, object]:
    store = engine.ctx.shared_objects[KEY]
    return (store.residency.device >= 0).all(-1).cpu().numpy(), store


def main() -> None:
    model, out, name, cache = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3], sys.argv[4]
    out.mkdir(parents=True, exist_ok=True)
    link = out / model.name                    # a hard link: the weight source resolves symbolic ones
    if not link.exists():
        os.link(model.resolve(), link)
    prompts = PromptSet.load("judge/prompts/v1.jsonl")
    trace = [snapshot.Request(p.ids, SamplingParams(temperature=0.0, max_tokens=256, ignore_eos=True))
             for p in prompts.prompts[:8]]
    engine = Engine(EngineConfig(model=str(link), **ENGINE, plugin_options={
        "moe.store": "std.moe.offload@1", "moe.cache": cache}))
    snapshot.record(engine, trace)
    held, store = resident(engine)
    score = store.state["score"].cpu().numpy()
    layers = store.layers
    engine.close()
    del engine, store
    gc.collect()
    torch.cuda.empty_cache()
    counts = np.where(held, 1.0 + score, 0.0)
    (out / f"{name}.json").write_text(json.dumps({"layers": {str(layer): counts[g].tolist()
                                                             for g, layer in enumerate(layers)}}) + "\n")
    engine = Engine(EngineConfig(model=str(link), **ENGINE, plugin_options={
        "moe.store": "std.moe.offload@1", "moe.cache": cache, "moe.placement": "std.tiered.static@1",
        "moe.placement.profile": f"{name}.json"}))
    frozen, _ = resident(engine)
    engine.close()
    meta = {"name": name, "cache": cache, "f11_resident": int(held.sum()), "static_resident": int(frozen.sum()),
            "common": int((held & frozen).sum()), "per_layer_f11": held.sum(1).tolist()}
    (out / f"{name}.meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
