"""k_resolve (moe_resolve.lfu@1, kernels.expert_io) at Flash-Next's scale: 512 experts, top-10, two roles, a verify
call's 4 decode rows, ~47% of experts resident whole; device time per call (CUDA events), the map reset between calls."""
import sys

import torch
from holon.core.build import BuildContext
from holon.core.plugins import PluginSet, discover

E, K, ROLES, T = 512, 10, 2, int(sys.argv[1]) if len(sys.argv) > 1 else 4
ctx = BuildContext(PluginSet(discover()), torch.device("cuda"))
spec = ctx.select_kernel("moe_resolve.lfu@1", site="b", experts=E, top_k=K, roles=ROLES, staging=E)
resolve = spec.impl(ctx, experts=E, top_k=K, roles=ROLES, staging=E)
g = torch.Generator().manual_seed(0)
resident = torch.rand(E, generator=g) < 0.47
res0 = torch.full((E, ROLES), -1, dtype=torch.int32)
slots = (K + E + torch.randperm(4000, generator=g)[:int(resident.sum())]).int()
for r in range(ROLES):
    res0[resident, r] = slots + r * 5000
dev = lambda t: t.cuda()
host = dev(torch.stack([torch.randperm(E, generator=g) for _ in range(ROLES)], 1).int())
limits = dev(torch.full((ROLES,), 20000, dtype=torch.int32))
counts, routed = dev(torch.zeros(E, dtype=torch.int32)), dev(torch.zeros(E, dtype=torch.int32))
ring = dev(torch.full((E, ROLES), -1, dtype=torch.int32))
stage = dev((K + torch.arange(E)).expand(ROLES, E).int().contiguous())
empty = dev(torch.zeros(ROLES, 8, dtype=torch.int32))
score0 = torch.rand(E, generator=g) * 10
plan, jobs = dev(torch.zeros(E + 1, dtype=torch.int32)), dev(torch.zeros(1, 4, dtype=torch.int32))
kinds = dev(torch.full((T,), 2, dtype=torch.uint8))   # DECODE
from holon_std.moe.kernels import DECODE

kinds.fill_(DECODE)
res, score = dev(res0.clone()), dev(score0.clone())
n_empty, misses, errors = (dev(torch.zeros(1, dtype=torch.int32)) for _ in range(3))
delta, n_delta = dev(torch.zeros(4 * E, 3, dtype=torch.int32)), dev(torch.zeros(1, dtype=torch.int32))
times = []
prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
for it in range(300):
    if it == 50:
        prof.__enter__()
    ids = dev(torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(T)]).int())
    res.copy_(dev(res0)); score.copy_(dev(score0)); n_delta.zero_()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    out = resolve(ids, kinds, res, host, limits, counts, routed, ring, stage, empty, n_empty, misses, errors, delta,
                  n_delta, 0, plan, jobs, score=score)
    b.record()
    torch.cuda.synchronize()
    if it >= 50:
        times.append(a.elapsed_time(b) * 1e3)
prof.__exit__(None, None, None)
ks = sorted(e.device_time for e in prof.events() if "k_resolve" in e.name)
print(f"tokens {T}: k_resolve device p50 {ks[len(ks)//2]:.1f} us, p10 {ks[len(ks)//10]:.1f}, n {len(ks)}, admitted {int(n_delta[0])}")
