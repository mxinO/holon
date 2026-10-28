"""GLM step 4's launch inventory (docs/dev/v1-glm.md acceptance 3, FN's method): kernel launches per decode token on a
4-block slice of the real file (`holon_testing.gguf_files.slice_blocks`: KDA with the dense FFN at blocks 0–2, DSA with
the routed experts and the shared expert at block 3), eager, after a prompt long enough that DSA selects sparsely
(> 2,051 tokens). Counted by the profiler as kernel-launch API calls inside each block's mixer, FFN and the rest of
the block (its mHC mixes and norms), plus what runs outside the blocks (embedding, final norm, head, sampling);
several decode steps, the median per step.

    CUDA_VISIBLE_DEVICES=0 python spikes/glm4/launches.py <slice.gguf> [--prompt 3000] [--steps 8] [--out <json>]
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics

import torch
from holon.api import SamplingParams
from holon.core.engine import Engine, EngineConfig
from torch.profiler import ProfilerActivity, profile, record_function

LAUNCH = ("LaunchKernel", "cuLaunch")                # the kernel-launch API calls (runtime and driver)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--prompt", type=int, default=3000)              # < 4,096: one step
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--out")
    a = ap.parse_args()
    engine = Engine(EngineConfig(a.model, device="cuda", dtype=torch.bfloat16, cuda_graphs=False, page_size=16,
                                 max_seqs=1, max_step_tokens=4096, max_model_len=a.prompt + a.steps + 64,
                                 memory_fraction=0.9))
    layers = engine.executor.model.layers
    for i, layer in enumerate(layers):                 # each block's, mixer's and FFN's forward as a named range
        for name, m in ((f"block{i}", layer), (f"block{i}.mixer", layer.mixer), (f"block{i}.ffn", layer.ffn)):
            forward = m.forward

            def ranged(*args, name=name, forward=forward, **kw):
                with record_function(name):
                    return forward(*args, **kw)
            m.forward = ranged
    gen = torch.Generator().manual_seed(0)
    engine.add(torch.randint(0, 150000, (a.prompt,), generator=gen).tolist(),
               SamplingParams(temperature=0.0, max_tokens=a.prompt + a.steps, ignore_eos=True))
    engine.step()                                      # the prompt (one step: it fits `max_step_tokens`)
    engine.step()                                      # a warm decode step
    per_step = []
    for _ in range(a.steps):
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            engine.step()
            torch.cuda.synchronize()
        per_step.append(_count(prof))
    engine.close()
    keys = sorted({k for s in per_step for k in s})
    result = {k: statistics.median(s.get(k, 0) for s in per_step) for k in keys}
    result["total"] = sum(v for k, v in result.items() if k != "total")
    kinds = {f"block{i}": type(layers[i].mixer).__name__ + "+" + type(layers[i].ffn).__name__
             for i in range(len(layers))}
    print(json.dumps({"launches": result, "blocks": kinds}, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"launches": result, "blocks": kinds, "prompt": a.prompt, "steps": a.steps}, f, indent=1)


def _count(prof) -> dict:
    """Launch API calls by the block range enclosing them (else "outside")."""
    events = prof.events()
    ranges = [(e.time_range.start, e.time_range.end, e.name) for e in events if e.name.startswith("block")]
    counts = collections.Counter()
    for e in events:
        if any(k in e.name for k in LAUNCH) and e.device_type.name == "CPU":
            t = e.time_range.start
            inside = [(end - s, n) for s, end, n in ranges if s <= t <= end]
            counts[min(inside)[1] if inside else "outside"] += 1           # the innermost range
    return dict(counts)


if __name__ == "__main__":
    main()
