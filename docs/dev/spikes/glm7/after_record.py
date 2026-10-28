"""GLM step 7, mode (ii) (a): the bench's workload (its warm-up, then 8 × 1K, 8 × 4K, 8 × 32K prompts, 256 tokens
each, one at a time) through Holon's mode (ii) deployment, recorded for the after-prompt levers' replay: per request
the tier of every piece — (layer, expert, role): 0 a VRAM slot, 1 a RAM unit, 2 the SSD only — before the prompt and
after every step, every step's routed experts, the SSD bytes and the I/O worker's busy time after every step, RAM's
heat after the request, the parked count, the wall time.
    python docs/dev/spikes/glm7/after_record.py <model> '<flags>' <requests.json> <out.npz>"""
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
                          prefix_cache=False, overlap=False)      # (a tap sees its own step's map and bytes only)
def drive() -> int:
    """The NVMe's sectors read so far (the hygiene wrap's counter), in bytes; 0 where there is none."""
    try:
        return int(open("/sys/block/nvme0n1/stat").read().split()[2]) * 512
    except OSError:
        return 0


d0 = drive()
e = Engine(cfg)
s = e.ctx.shared_objects["std.moe.offload"]
io = s.io.service.native
R = json.load(open(reqs))
print(f"bytes at ready: drive {(drive() - d0) / 2**30:.2f} GiB, I/O worker {io.io_stats()['bytes'] / 2**30:.2f} GiB",
      flush=True)


def tiers() -> np.ndarray:
    """[G, E, R] int8: each piece's tier (0 VRAM, 1 RAM, 2 SSD only), from the host mirror (the device's map as of
    the last commit) and the RAM units."""
    res, host = s.residency.host, s.host_units.cpu().numpy()
    return np.where(res >= 0, 0, np.where(host >= 0, 1, 2)).astype(np.int8)


steps = []
s.taps.append(lambda step, rec: steps.append((rec["routed"].copy(), tiers(), io.io_stats())))   # any row's routes
e.generate([R["warmup"]["ids"]], SamplingParams(temperature=0.0, max_tokens=R["warmup"]["output"], ignore_eos=True))
print(f"bytes after the warm-up: drive {(drive() - d0) / 2**30:.2f} GiB, I/O worker "
      f"{io.io_stats()['bytes'] / 2**30:.2f} GiB", flush=True)
dump = {"piece_mib": np.array([[sum(x.nbytes for x in s.groups[g][r].pieces[s._piece_index(s.groups[g][r], i, 0, r)][3])
                                / 2**20 for r in range(s.roles)] for i, g in enumerate(s.layers)])}   # a piece's bytes
for r in R["requests"]:
    steps.clear()
    before, p0 = tiers(), s.parked
    t0, st0 = time.perf_counter(), io.io_stats()
    e.generate([r["ids"]], SamplingParams(temperature=0.0, max_tokens=256, ignore_eos=True))
    t1 = time.perf_counter()
    n_prompt = -(-len(r["ids"]) // max(cfg.max_step_tokens, 1))           # the request's prompt steps come first
    routed = np.stack([x[0] for x in steps])
    k = r["key"].replace("/", "_")
    dump[f"{k}.before"] = before
    dump[f"{k}.routed"] = np.packbits(routed, axis=-1)
    dump[f"{k}.maps"] = np.stack([x[1] for x in steps])
    dump[f"{k}.bytes"] = np.array([x[2]["bytes"] for x in steps]) - st0["bytes"]
    dump[f"{k}.busy"] = np.array([x[2]["busy_ns"] for x in steps]) - st0["busy_ns"]
    dump[f"{k}.meta"] = np.array([n_prompt, s.parked - p0, len(steps), t1 - t0, routed.shape[-1]])
    if getattr(s, "ram_heat", None) is not None:
        dump[f"{k}.heat_end"] = s.ram_heat.cpu().numpy()
    print(k, "prompt GiB %.2f" % (dump[f"{k}.bytes"][n_prompt - 1] / 2**30), "wall %.2f s" % (t1 - t0), flush=True)
e.generate([[1, 2, 3]], SamplingParams(temperature=0.0, max_tokens=1))     # (the last request's loan counted)
print(f"loan overflows {s.loan_overflows}, most {s.loan_overflow_most / 2**20:.0f} MiB; bytes at the end: drive "
      f"{(drive() - d0) / 2**30:.2f} GiB", flush=True)
np.savez_compressed(out, **dump)
e.close()
print("DIAG-AFTER-DONE")
