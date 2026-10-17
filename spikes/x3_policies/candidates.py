"""How often F11's resolve has candidates to pair (a decode-routed expert not resident before the step): batch-1
decode, 128 tokens after a 1K prompt of natural text, per cache fraction; the share of decode layer calls with
candidates prices F11's per-token cost from bench_resolve.py's per-call figures:

    python spikes/x3_policies/candidates.py <model.gguf> "*=0.47" "*=0.25"
"""
import sys

import torch
from holon.api import SamplingParams
from holon.core.engine import Engine, EngineConfig
from holon_judge.placement import route_dump
from holon_judge.recording import PromptSet

model = sys.argv[1]
stream = [t for p in PromptSet.load("judge/prompts/v1.jsonl").prompts for t in p.ids]
for cache in sys.argv[2:]:
    engine = Engine(EngineConfig(model=model, dtype=torch.bfloat16, max_seqs=1, max_step_tokens=2048,
                                 max_model_len=2048, host_memory=20 << 30, plugin_options={
                                     "moe.store": "std.moe.offload@1", "moe.cache": cache}))
    engine.generate([stream[:1024]], SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True))
    with route_dump(engine) as trace:
        engine.generate([stream[3000:4024]], SamplingParams(temperature=0.0, max_tokens=128, ignore_eos=True))
    before, calls, cands = trace.seed, 0, 0
    for s in trace.steps:
        if s["counts"].any():
            c = (s["counts"] > 0) & ~before
            calls += c.shape[0]
            cands += int(c.any(1).sum())
        before = s["resident"]
    print(cache, "decode layer calls", calls, "with candidates", cands, round(cands / max(calls, 1), 3), flush=True)
    engine.close()
