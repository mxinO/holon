"""GLM step 7 (C's L1): the fixed bytes a mode (ii) prompt step's activations take past their tokens' share of the
loan, with and without prompt logprobs — the loan's overflow at `moe.loan_fixed_mib` 0 (the slack only), per prompt
size, from the store's own counter (`loan_overflow_most`).
    python spikes/glm7/loan_need.py <model> '<flags>' <requests.json>"""
import argparse
import dataclasses
import json
import shlex
import sys

from holon.api import SamplingParams
from holon.core.cli import engine_config, engine_flags
from holon.core.engine import Engine

model, flags, reqs = sys.argv[1:4]
p = argparse.ArgumentParser()
engine_flags(p)
cfg = dataclasses.replace(engine_config(model, p.parse_args(shlex.split(flags))), max_seqs=1, max_model_len=33792,
                          prefix_cache=False)
e = Engine(cfg)
s = e.ctx.shared_objects["std.moe.offload"]
R = json.load(open(reqs))["requests"]
for size in ("1024", "4096", "32768"):
    ids = next(r for r in R if r["key"].startswith(size))["ids"]
    for logprobs in (None, 1):
        s.loan_overflow_most = 0
        e.generate([ids], SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True, prompt_logprobs=logprobs))
        print(f"{size}-token prompt, prompt logprobs {logprobs}: loan overflow {s.loan_overflow_most / 2**20:.0f} MiB "
              f"(fixed {s._loan_fixed >> 20} MiB)", flush=True)
e.close()
print("LOAN-NEED-DONE")
