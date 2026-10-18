"""Why Holon misses more at the capped fit than the replay said (README "Capped-fit miss bytes"): the gate's 4K request
at `*=0.25` (ctx.py's warm-up, then one 4,096-token prompt generating 256 tokens, batch 1, the lock's settings) under
the store's route dump, and per decode step:

- `true`: the step's routed experts not resident before it (the map after the previous commit) — what F5a fetched;
- `end`: ctx.py's count, against the residency at the request's end (right for a frozen placement only);
- the M6 replay's `lfu` (spikes/x0_offload/m6, as §13's ≈ 200 MB came from) on the same routing, from the residency
  the first decode step started with, and from the profile seed (before the warm-up);
- the empty slots the prompt's ring step left (lent, returned empty: D4) and the admissions per step.

    python spikes/x4_gate/capped.py <model.gguf> [--cache '*=0.25'] [--profile <name>] [--context 4096] [--json out]
"""
import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np
from holon.api import SamplingParams
from holon.core.engine import Engine, EngineConfig
from holon_judge.placement import route_dump
from holon_std.moe.offload import KEY

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
import offload_ctx  # noqa: E402
import workloads  # noqa: E402


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


m6 = _load("m6_replay", ROOT / "spikes" / "x0_offload" / "m6" / "replay.py")
sys.modules["replay"] = m6                               # x4_gap's replay imports M6's as `replay`
x4_replay = _load("x4_gap_replay", ROOT / "spikes" / "x4_gap" / "replay.py")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--cache", default="*=0.25")
    ap.add_argument("--profile", default="profile_x4_gate.json")
    ap.add_argument("--placement", default="placement.lfu@1")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--json")
    a = ap.parse_args()
    opts = {"moe.store": "std.moe.offload@1", "moe.placement": a.placement, "moe.cache": a.cache,
            "moe.ring_copy": "dma"} | ({"moe.placement.profile": a.profile} if a.profile else {})
    engine = Engine(EngineConfig(model=a.model, max_seqs=1, max_model_len=33792, prefix_cache=False,
                                 host_memory=20 << 30, plugin_options=opts))
    store = engine.ctx.shared_objects[KEY]
    seeded = (store.residency.host >= 0).all(-1).copy()
    corpus = workloads.corpus(offload_ctx.CORPUS)
    params = SamplingParams(temperature=0.0, max_tokens=offload_ctx.OUTPUT, ignore_eos=True)
    off, p, n = offload_ctx.WARMUP
    engine.generate([[corpus[(off + i) % len(corpus)] for i in range(p)]],
                    SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True))
    before = (store.residency.host >= 0).all(-1).copy()
    with route_dump(engine) as trace:
        engine.generate([offload_ctx.prompt(corpus, a.context)], params)
    L, E = before.shape
    unit = np.array([sum(store.groups[g][r].unit_bytes for r in range(store.roles)) for g in sorted(store.groups)])
    ebytes = np.repeat(unit, E).astype(float)
    steps = [s for s in trace.steps if s["counts"].any()]              # decode steps (batch 1: one row each)
    first = trace.steps.index(steps[0])
    prev = trace.steps[first - 1]["resident"] if first else before
    prefill = trace.steps[:first]
    keys = [np.flatnonzero(s["counts"].reshape(-1)) for s in steps]
    end = steps[-1]["resident"].reshape(-1)
    true_mb, end_mb, adm = [], [], []
    res = prev.reshape(-1)
    for s, k in zip(steps, keys, strict=True):
        true_mb.append(ebytes[k[~res[k]]].sum() / 1e6)
        end_mb.append(ebytes[k[~end[k]]].sum() / 1e6)
        adm.append(int((s["records"][:, 1] >= 0).sum()) if len(s["records"]) else 0)
        res = s["resident"].reshape(-1)
    ptr = np.cumsum([0, *map(len, keys)])
    tr = x4_replay.decode(np.concatenate(keys), ptr, L, E)
    quota = before.sum(1)

    def replay(start: np.ndarray) -> dict:
        r = m6.simulate(tr, "lfu", start.sum(1), start.reshape(-1), np.zeros(L * E, np.int64), ebytes, 0.0)
        return {"hit_rate": round(r["hit_rate"], 4), "miss_mb_per_token": round(r["bytes_per_token"]["fetch"] / 1e6, 1)}

    out = {
        "cache": a.cache, "profile": a.profile, "placement": a.placement, "context": a.context,
        "cached_experts": {"seed": int(seeded.sum()), "before_prompt": int(before.sum()),
                           "after_prompt": int(prev.sum()), "quota": int(quota.sum())},
        "prompt_steps": len(prefill),
        "empty_after_prompt": int(np.asarray(prefill[-1]["empty"]).sum()) if prefill else 0,
        "lent_of_before": int((before & ~prev).sum()),
        "decode_steps": len(steps), "routed_per_token": float(np.mean([len(k) for k in keys])),
        "true_miss_mb_per_token": round(float(np.mean(true_mb)), 1),
        "end_residency_miss_mb_per_token": round(float(np.mean(end_mb)), 1),
        "true_by_quarter": [round(float(np.mean(x)), 1) for x in np.array_split(true_mb, 4)],
        "admissions_per_token": round(float(np.mean(adm)), 1),
        "replay_lfu_from_decode_start": replay(prev),
        "replay_lfu_from_before_prompt": replay(before),
        "replay_lfu_from_seed": replay(seeded),
    }
    print(json.dumps(out, indent=1))
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    engine.close()


if __name__ == "__main__":
    main()
