"""GLM step 5, lending's effect on h (docs/dev/v1-glm.md §9 step 5, §7.4): the decode hit rate — the share of the decode
rows' routed experts the VRAM cache held when routed, as `bench/offload_ctx.py` taps it — over the 256 tokens after
each prompt, per configuration. A prompt step that lends cache slots (the ring, the tail, the activation loan)
displaces cached experts; the decode after it reads them back. Prompts are spans of natural text (a GLM-tokenized
prompt set), each a fresh request, prefix cache off; a warm-up request first. Per configuration: h per prompt, its
mean per context, the cache's experts (VRAM slots per class) and the experts the step lent.

Usage: lend_h.py <model> '<base flags>' --prompts <set.jsonl> --config NAME='<flags>' [--config ...]
       [--contexts 4096 32000] [--spans 3] [--decode 256] [--out <json>]"""
from __future__ import annotations

import argparse
import gc
import json
import pathlib
import statistics

import torch

ROOT = pathlib.Path(__file__).resolve().parents[4]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("flags")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--config", action="append", default=[], help="NAME='<flags>' (over the base flags)")
    ap.add_argument("--contexts", type=int, nargs="+", default=[4096, 32000])
    ap.add_argument("--spans", type=int, default=3)
    ap.add_argument("--decode", type=int, default=256)
    ap.add_argument("--out")
    a = ap.parse_args()
    from holon.api import SamplingParams
    from holon.core.cli import layered
    from holon.core.engine import Engine, EngineConfig
    from holon_judge.engine import flag_settings
    from holon_judge.recording import PromptSet

    stream = [t for p in PromptSet.load(a.prompts).prompts for t in p.ids]
    reqs = [(c, stream[(k * len(stream) // (a.spans + 1) + 977 * c) % (len(stream) - c):][:c])
            for c in a.contexts for k in range(a.spans)]
    out = {}
    for name, _, extra in (c.partition("=") for c in a.config):
        cfg = layered(layered(flag_settings(a.flags), flag_settings(extra)), {"prefix_cache": False})
        e = Engine(EngineConfig(a.model, **cfg))
        store = e.ctx.shared_objects["std.moe.offload"]
        seen, prev = {"hits": 0, "routed": 0}, {"map": store.residency.host.copy()}

        def tap(step, record, seen=seen, prev=prev, store=store) -> None:
            counts = record["counts"]                        # decode rows' routes, [groups, experts]
            seen["routed"] += int(counts.sum())
            seen["hits"] += int(counts[(prev["map"] >= 0).all(-1)].sum())
            prev["map"] = store.residency.host.copy()
        store.taps.append(tap)
        params = SamplingParams(temperature=0.0, max_tokens=a.decode, ignore_eos=True)
        e.generate([stream[:2048]], SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True))   # warm-up
        rows = []
        for c, ids in reqs:
            seen.update(hits=0, routed=0)
            e.generate([ids], params)
            rows.append({"context": c, "h": round(seen["hits"] / max(seen["routed"], 1), 4)})
            print(name, json.dumps(rows[-1]), flush=True)
        cache = {n: int(c.size) for n, c in store.classes.items()}
        out[name] = {"rows": rows, "cache_slots": cache,
                     "h_mean": {c: round(statistics.mean(r["h"] for r in rows if r["context"] == c), 4)
                                for c in a.contexts}}
        print(name, json.dumps({k: v for k, v in out[name].items() if k != "rows"}), flush=True)
        e.close()
        del e, store
        gc.collect()
        torch.cuda.empty_cache()
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
