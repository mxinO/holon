"""The HC mix alone (kernels.hc_triton's decode/verify path) at Flash-Next's shapes — 4 streams of 2,560, rank 320 (+4
inject), bf16 weights — at 1 and 4 rows, each kernel's device time (torch.profiler), against a read of the same
weight bytes (a sum over them: the bandwidth floor)."""
import sys, torch
from holon_plugins.kernels.hc_triton.kernel import hc_gr_mix

S, H, RANK = 4, 2560, 320
dev = "cuda"
g = torch.Generator(device=dev).manual_seed(0)
norm = torch.randn(S * H, device=dev, generator=g) * 0.1
down = (torch.randn(RANK, S * H, device=dev, generator=g) * 0.02).bfloat16()
up = (torch.randn(S * H, RANK, device=dev, generator=g) * 0.02).bfloat16()
inject = (torch.randn(S, S * H, device=dev, generator=g) * 0.02).bfloat16()
wbytes = (down.numel() + up.numel() + inject.numel()) * 2
flat = torch.cat([down.flatten(), up.flatten(), inject.flatten()])
COPIES = 12                      # every call reads another copy: past the L2 (96 MB on a 5090), as in the model
sets = [(norm.clone(), down.clone(), up.clone(), inject.clone(), flat.clone()) for _ in range(COPIES)]
for T in (1, 4):
    r = torch.randn(T, S * H, device=dev, generator=g)
    y = torch.randn(T, H, device=dev, generator=g).bfloat16()
    wp = torch.rand(T, S, device=dev, generator=g)
    for i in range(20):
        nm, dn, u, ij, fl = sets[i % COPIES]
        hc_gr_mix(r, nm, dn, u, ij, y, wp, S, 1e-6, torch.bfloat16)
        torch.sum(fl, dtype=torch.float32)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for i in range(100):
            nm, dn, u, ij, fl = sets[i % COPIES]
            hc_gr_mix(r, nm, dn, u, ij, y, wp, S, 1e-6, torch.bfloat16)
            nm, dn, u, ij, fl = sets[(i + COPIES // 2) % COPIES]
            torch.sum(fl, dtype=torch.float32)
        torch.cuda.synchronize()
    per = {}
    for e in prof.events():
        if e.device_time > 0:
            per.setdefault(e.name[:40], []).append(e.device_time)
    mix = {k: sorted(v)[len(v) // 2] for k, v in per.items() if k.startswith("_")}
    read = [sorted(v)[len(v) // 2] for k, v in per.items() if "reduce" in k]
    tot = sum(mix.values())
    print(f"T={T}: mix {tot:.1f} us {({k: round(v, 1) for k, v in mix.items()})}; read of {wbytes / 1e6:.1f} MB "
          f"{min(read):.1f} us ({wbytes / min(read) / 1e3:.0f} GB/s); mix/read {tot / min(read):.2f}")
