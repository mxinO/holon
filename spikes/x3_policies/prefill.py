"""Prefill throughput of the vehicle through the offload store, as built before D8 (512-token expert passes, routed
misses staged; `moe.ring_tokens=0`) and with D8 (one expert pass per chunk of up to 4,096 tokens streaming every
layer's missing experts through lent ring slots on a copy engine; `moe.ring_tokens=512`), and all-resident: one
request of N prompt tokens of natural text (judge/prompts/v1.jsonl, concatenated) per run, the step budget N (one
chunk), the prefix cache off, generating one token; tok/s = N / the request's wall time, median of R runs after a
warm-up, arms interleaved by round:

    python spikes/x3_policies/prefill.py <model.gguf> <out.json> <rounds> <N>... -- "<name>|<cache>|<ring_tokens>" ...
"""
import gc
import json
import statistics
import sys
import time

import torch
from holon.api import SamplingParams
from holon.core.engine import Engine, EngineConfig
from holon_judge.recording import PromptSet

ONE = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)


def main() -> None:
    model, out, rounds = sys.argv[1], sys.argv[2], int(sys.argv[3])
    split = sys.argv.index("--")
    sizes, arms = [int(n) for n in sys.argv[4:split]], [a.split("|") for a in sys.argv[split + 1:]]
    stream = [t for p in PromptSet.load("judge/prompts/v1.jsonl").prompts for t in p.ids]
    results = {name: {n: [] for n in sizes} for name, *_ in arms}
    for _ in range(rounds):
        for name, cache, ring in arms:
            kw = {} if cache == "resident" else dict(host_memory=20 << 30, plugin_options={
                "moe.store": "std.moe.offload@1", "moe.cache": cache, "moe.ring_tokens": int(ring)})
            engine = Engine(EngineConfig(model=model, dtype=torch.bfloat16, max_seqs=1, max_step_tokens=max(sizes),
                                         max_model_len=max(sizes) + 64, prefix_cache=False, **kw))
            engine.generate([stream[:512]], ONE)                                  # warm-up
            for n in sizes:
                for i in range(3):
                    prompt = stream[(i + 1) * 997:(i + 1) * 997 + n]
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    engine.generate([prompt], ONE)
                    results[name][n].append(n / (time.perf_counter() - t))
            store = engine.ctx.shared_objects.get("std.moe.offload")
            print(name, {n: round(statistics.median(v), 1) for n, v in results[name].items()},
                  getattr(store, "borrowed", None), flush=True)
            engine.close()
            del engine
            gc.collect()
            torch.cuda.empty_cache()
    summary = {name: {n: {"median_tok_s": round(statistics.median(v), 1), "runs": [round(x, 1) for x in v]}
                      for n, v in by.items()} for name, by in results.items()}
    with open(out, "w") as f:
        json.dump({"arms": [{"name": a[0], "cache": a[1], "ring_tokens": a[2]} for a in arms], "rounds": rounds,
                   "results": summary}, f, indent=1)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
