"""FN §21.12: the step-peak loan (`moe.lend_peak`) on a deployment: the loan's bytes and the profile's peaks, then per lending step the pool's
segments, live bytes and the overflow past the range (the bench's warm-up, then prompts of `LENS` tokens). Usage: loan_probe.py <model> '<flags>'
<requests.json> [max_model_len]. Debug only: it reads the pool's `MemPool.snapshot()`, which on torch 2.11 corrupts the host heap
(kernels.expert_io's `ActivationLoan` docstring); never call it on a serving path."""
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "spikes/fn6"))
os.chdir(ROOT)
import torch
from holon.core import platform

peaks, m = [], platform.peak_memory
def pm(d, fn):
    peaks.append(m(d, fn)); return peaks[-1]
platform.peak_memory = pm
from _engine import STORE, engine
from holon_std.tiered.store import TieredStore

_lend = TieredStore.lend_activations
measured = {}
def lend(self, peak):
    dev = self.ctx.device
    top = torch.cuda.max_memory_reserved(dev)
    torch.cuda.empty_cache()
    measured.update(peak_alloc=peak >> 20, reserved_peak=top >> 20, reserved_now=torch.cuda.memory_reserved(dev) >> 20,
                    growth=(top - torch.cuda.memory_reserved(dev)) >> 20)
    return _lend(self, peak)
TieredStore.lend_activations = lend
from holon.api import SamplingParams

model, flags, reqfile = sys.argv[1], sys.argv[2], sys.argv[3]
p = json.loads(pathlib.Path(reqfile).read_text())
e = engine(model, flags, max_seqs=1, max_model_len=int(sys.argv[4]) if len(sys.argv) > 4 else 33792, prefix_cache=False)
s = e.ctx.shared_objects[STORE]
print("measured", measured)
print("profile peaks MiB", [x >> 20 for x in peaks], "loan MiB", s._loan_bytes >> 20, "tail", s.tail,
      "free MiB", torch.cuda.mem_get_info()[0] >> 20, flush=True)
use, rows = s._loan.use, []
def spy(r):
    cm = use(r)
    class W:
        def __enter__(self):
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            W.base = torch.cuda.memory_allocated()
            return cm.__enter__()
        def __exit__(self, *a):
            out = cm.__exit__(*a)
            snap = s._loan.pool.snapshot()
            torch.cuda.synchronize()
            rows.append({"alloc_peak_mib": (torch.cuda.max_memory_allocated() - W.base) >> 20, "range_mib": r.numel() >> 20, "reserved_mib": sum(x["total_size"] for x in snap) >> 20,
                         "segments": len(snap), "biggest": sorted((x["total_size"] >> 20 for x in snap), reverse=True)[:6],
                         "overflow_mib": s._loan.lib.loan_overflow() >> 20, "tokens": s.step_tokens,
                         "by_stream": {str(k): v >> 20 for k, v in __import__("collections").Counter(
                             {x["stream"]: 0 for x in snap}).items()}})
            by = __import__("collections").defaultdict(int)
            for x in snap:
                by[x["stream"]] += x["total_size"]
            rows[-1]["by_stream"] = {str(k): v >> 20 for k, v in by.items()}
            sizes = __import__("collections").Counter()
            for x in snap:
                for b in x["blocks"]:
                    sizes[(b["size"] >> 20)] += 1
            rows[-1]["block_sizes_mib"] = sorted(sizes.items(), reverse=True)[:12]
            print(rows[-1], flush=True)
            return out
    return W()
s._loan.use = spy
try:
    e.generate([p["warmup"]["ids"]], SamplingParams(temperature=0.0, max_tokens=p["warmup"]["output"], ignore_eos=True))
    longest = max((r["ids"] for r in p["requests"]), key=len)
    for n in [int(x) for x in os.environ.get("LENS", "1024 4096 6144 32768").split()]:
        e.generate([longest[:n]], SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True))
        print("done", n, flush=True)
except torch.OutOfMemoryError as x:
    print("OOM", str(x)[:400], flush=True)
    snap = s._loan.pool.snapshot() if s._loan.pool is not None else []
    print("pool reserved MiB", sum(x["total_size"] for x in snap) >> 20, "alloc MiB",
          sum(b["size"] for x in snap for b in x["blocks"] if b["state"] == "active_allocated") >> 20,
          "segments", len(snap), "overflow MiB", s._loan.lib.loan_overflow() >> 20, flush=True)
