"""GLM step 7, mode (ii) (a): the bench's workload (its warm-up, then 8 × 1K, 8 × 4K, 8 × 32K prompts, 256 tokens
each, one at a time) through Holon's mode (ii) deployment, recording for the replay of the after-prompt levers:
per request the tier of every (layer, expert) before the prompt and after it (VRAM: every role a slot
RAM: every
role a unit
SSD: else), RAM's heat after it, the lent / parked counts, and per decode step the routed experts, the
tier map after the step and the SSD bytes read
the I/O worker's busy time a token.
    python spikes/glm7/after_record.py <model> '<flags>' <requests.json> <out.npz>"""
import argparse
import dataclasses
import json
import shlex
import sys
import time

import numpy as np
from holon.api import SamplingParams
from holon.core.cli import engine_config, engine_flags
from holon.core.engine import Engine

model, flags, reqs, out = sys.argv[1:5]
p = argparse.ArgumentParser()
engine_flags(p)
cfg = dataclasses.replace(engine_config(model, p.parse_args(shlex.split(flags))), max_seqs=1, max_model_len=33792,
                          prefix_cache=False)
e = Engine(cfg)
s = e.ctx.shared_objects["std.moe.offload"]
io = s.io.service.native
R = json.load(open(reqs))

def tiers() -> np.ndarray:                     # [G, E] int8: 0 VRAM, 1 RAM, 2 SSD
    res, host = s.residency.host, s.host_units.cpu().numpy()
    v = (res >= 0).all(-1)
    r = ~v & (host >= 0).all(-1)
    return np.where(v, 0, np.where(r, 1, 2)).astype(np.int8)

steps = []
s.taps.append(lambda step, rec: steps.append((rec["routed"].copy(), tiers(), io.io_stats())))   # any row's routes
e.generate([R["warmup"]["ids"]], SamplingParams(temperature=0.0, max_tokens=R["warmup"]["output"], ignore_eos=True))
dump = {}
for _i, r in enumerate(R["requests"]):
    steps.clear()
    before, p0, heat0 = tiers(), s.parked, None
    t0, st0 = time.perf_counter(), io.io_stats()
    out_ = e.generate([r["ids"]], SamplingParams(temperature=0.0, max_tokens=256, ignore_eos=True))
    t1 = time.perf_counter()
    # the prompt's steps are those before the first decode step: the request's first step(s) route its prompt
    n_prompt = -(-len(r["ids"]) // max(cfg.max_step_tokens, 1))
    routed = np.stack([x[0] for x in steps])
    maps = np.stack([x[1] for x in steps])
    b = np.array([x[2]["bytes"] for x in steps])
    busy = np.array([x[2]["busy_ns"] for x in steps])
    k = r["key"].replace("/", "_")
    dump[f"{k}.before"] = before
    dump[f"{k}.routed"] = np.packbits(routed, axis=-1)
    dump[f"{k}.maps"] = maps
    dump[f"{k}.bytes"] = b - st0["bytes"]
    dump[f"{k}.busy"] = busy - st0["busy_ns"]
    dump[f"{k}.meta"] = np.array([n_prompt, s.parked - p0, len(steps), t1 - t0, routed.shape[-1]])
    heat = getattr(s, "ram_heat", None)
    if heat is not None:
        dump[f"{k}.heat_end"] = heat.cpu().numpy()
    after = maps[n_prompt - 1]
    print(k, "steps", len(steps), "parked", s.parked - p0, "tiers before", np.bincount(before.ravel(), minlength=3),
          "after prompt", np.bincount(after.ravel(), minlength=3),
          "displaced (VRAM/RAM -> SSD)", int(((before < 2) & (after == 2)).sum()),
          "prompt routed (all / SSD-only before)", int(routed[:n_prompt].any(0).sum()),
          int((routed[:n_prompt].any(0) & (before == 2)).sum()),
          "prompt GiB %.2f" % ((b[n_prompt - 1] - st0["bytes"]) / 2**30),
          "decode MB/tok 1-32 %.1f" % ((b[n_prompt + 31] - b[n_prompt - 1]) / 32e6),
          "33-256 %.1f" % ((b[-1] - b[n_prompt + 31]) / (len(b) - n_prompt - 32) / 1e6),
          flush=True)
np.savez_compressed(out, **dump)
e.close()
print("DIAG-AFTER-DONE")
