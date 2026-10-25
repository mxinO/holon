"""GLM step 5 (docs/v1-glm.md §9): Holon's own routes on the full model, through the offload store's taps
(`holon_judge.placement.route_dump`), on held-out prompts — FN's 72-item task set's first 60 items (D8's seed: not
the bench corpus), rendered with GLM's chat template, greedy, `--tokens` each, one request at a time.

- **The seed profile** (D8, FN D12f): the decode rows' routes per (layer, expert), written as `holon_tools profile`
  writes one (`<checkpoint stem>.profile.json` beside the checkpoint's files), for the bundle and the locks.
- **N2's re-check** (D7): on Holon's decode routes, the distinct experts a layer of 1 / 2 / 3 / 4 consecutive tokens
  route (Maya's traces: 8 / 13.9 / 19.1 / 23.9), the verify window's cost behind MTP's deferral.

    python spikes/glm5/routes.py <maya-s -00001 shard> '<engine flags>' [--tokens 128] [--items 60] --out <json>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "spikes" / "fn7" / "tasks"))


def main() -> None:
    import run as fn_tasks  # FN's task runner: its items and its flag parser
    from holon.api import SamplingParams
    from holon.core.cli import layered
    from holon.core.engine import Engine, EngineConfig
    from holon_judge.placement import route_dump
    from holon_std.moe.offload import KEY
    from holon_tools import profile

    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("flags")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--items", type=int, default=60)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bundle = fn_tasks.bundle(a.model)
    prompts = [bundle.render_chat(r["messages"]) for r in fn_tasks.items()[:a.items]]
    e = Engine(EngineConfig(a.model, **layered({"max_model_len": 40960, "max_seqs": 1},
                                               fn_tasks.flags(a.flags))))
    store = e.ctx.shared_objects[KEY]
    groups, items = len(store.layers), store.residency.host.shape[1]
    total = np.zeros((groups, items), dtype=np.int64)
    windows = {n: [] for n in (1, 2, 3, 4)}                  # distinct experts a layer, per window of n tokens
    params = SamplingParams(temperature=0.0, max_tokens=a.tokens, ignore_eos=True)
    for ids in prompts:
        with route_dump(e) as trace:
            e.add(ids, params)
            while e.has_work():
                e.step()
        decode = [s["counts"] > 0 for s in trace.steps if s["counts"].sum(1).max() <= store.top_k]   # 1-token steps
        for s in trace.steps:
            if s["counts"].sum(1).max() <= store.top_k:
                total += s["counts"].astype(np.int64)
        for n in windows:
            for i in range(len(decode) - n + 1):
                windows[n].append(np.logical_or.reduce(decode[i:i + n]).sum(1).mean())
    counts = {"checkpoint": store.checkpoint,
              "layers": {str(layer): total[g].tolist() for g, layer in enumerate(store.layers)}}
    written = profile.write(store.profile_dir, profile.default_name(pathlib.Path(a.model)), counts)
    e.close()
    out = {"prompts": len(prompts), "tokens": a.tokens, "flags": shlex.split(a.flags), "profile": written.name,
           "decode_tokens": int(total.sum() // max(1, store.top_k) // groups),
           "distinct_experts_a_layer": {n: round(float(np.mean(v)), 2) for n, v in windows.items() if v},
           "maya_n2": {1: 8, 2: 13.9, 3: 19.1, 4: 23.9}}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
