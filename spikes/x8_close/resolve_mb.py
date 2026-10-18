"""The X close's bisect: `moe_resolve.lfu@1` (kernels.expert_io) alone at one decode token (E 128, k 8, two roles, 25%
of the experts resident, a zero CPU plan: F5a), 48 calls per CUDA graph (one per layer, each its own map and state);
prints µs per graph (median and min of 7 × 200 replays). Takes the signature before step 6 (no plan / jobs) too, so a
source tree of any commit can be timed by putting it first on PYTHONPATH:

    python spikes/x8_close/resolve_mb.py <label>
"""
import inspect
import sys
import time

import torch
from holon.core.build import BuildContext
from holon.core.plugins import PluginSet, discover

E, K, R, L, STAGING = 128, 8, 2, 48, 64
dev = torch.device("cuda")
ctx = BuildContext(PluginSet(discover()), dev)
facts = dict(experts=E, top_k=K, roles=R, staging=STAGING)
spec = ctx.select_kernel("moe_resolve.lfu@1", site="mb", **facts)
op = spec.impl(ctx, **facts)
new = "plan" in inspect.signature(op.__call__).parameters
g = torch.Generator().manual_seed(0)
layers = []
for _ in range(L):
    res = torch.full((E, R), -1, dtype=torch.int32)
    cached = torch.randperm(E, generator=g)[:E // 4]
    for r in range(R):
        res[cached, r] = (K + STAGING + torch.arange(len(cached))).int()
    t = {"residency": res, "host": torch.stack([torch.arange(E)] * R, 1).int(),
         "limits": torch.full((R,), K + STAGING + E, dtype=torch.int32), "counts": torch.zeros(E, dtype=torch.int32),
         "routed": torch.zeros(E, dtype=torch.int32), "ring": torch.full((E, R), -1, dtype=torch.int32),
         "stage": (K + torch.arange(STAGING)).expand(R, STAGING).int().contiguous(),
         "empty": torch.zeros(R, 8, dtype=torch.int32), "n_empty": torch.zeros(1, dtype=torch.int32),
         "misses": torch.zeros(1, dtype=torch.int32), "errors": torch.zeros(1, dtype=torch.int32),
         "delta": torch.zeros(1 << 16, 3, dtype=torch.int32), "n_delta": torch.zeros(1, dtype=torch.int32),
         "score": torch.rand(E, generator=g)}
    layers.append({k: v.to(dev) for k, v in t.items()})
ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(L)]).int().to(dev)
kinds = torch.ones(1, dtype=torch.uint8, device=dev)
plan = torch.zeros(E + 1, dtype=torch.int32, device=dev)
jobs = torch.zeros(1, 4, dtype=torch.int32, device=dev)


def run():
    for i, t in enumerate(layers):
        t["n_delta"].zero_()
        args = [ids[i:i + 1], kinds, t["residency"], t["host"], t["limits"], t["counts"], t["routed"], t["ring"],
                t["stage"], t["empty"], t["n_empty"], t["misses"], t["errors"], t["delta"], t["n_delta"], i]
        if new:
            args += [plan, jobs]
        op(*args, score=t["score"])


run(); torch.cuda.synchronize()
graph = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    run()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph, stream=s):
        run()
torch.cuda.synchronize()
out = []
for rep in range(7):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(200):
        graph.replay()
    torch.cuda.synchronize(); out.append((time.perf_counter() - t0) / 200 * 1e6)
out.sort()
print(sys.argv[1] if len(sys.argv) > 1 else "", "graph of 48 resolves + 48 zero_: us/graph median %.1f min %.1f" % (out[3], out[0]))
