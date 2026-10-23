"""Prototype: `_mix_rows` split over K — the up projection's partial sums per (K chunk of BKC, column tile) in one
launch, then per column the chunks summed in order and the mix's epilogue (the streams in order) in a second; float32
products, a fixed order (not today's bits); device time and the largest difference from a float64 reference."""
import sys

import torch
import triton
import triton.language as tl
from holon_plugins.kernels.hc_triton import kernel as K


@triton.jit
def _up_part(a_ptr, up_ptr, part_ptr, T, SH, RANK, BJ: tl.constexpr, BKC: tl.constexpr):
    j = tl.program_id(0) * BJ + tl.arange(0, BJ)
    c = tl.program_id(1)
    kk = c * BKC + tl.arange(0, BKC)
    jm, km = j < SH, kk < RANK
    w = tl.load(up_ptr + j[:, None].to(tl.int64) * RANK + kk[None, :], mask=jm[:, None] & km[None, :],
                other=0.0).to(tl.float32)
    for t in range(T):
        a = tl.load(a_ptr + t * RANK + kk, mask=km, other=0.0)
        tl.store(part_ptr + (c * T + t) * SH + j, tl.sum(w * a[None, :], 1), mask=jm)

@triton.jit
def _mix_epi(part_ptr, rstd_ptr, r_ptr, rn_ptr, norm_ptr, x_ptr, T, H, KC: tl.constexpr, S: tl.constexpr,
             PENDING: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(1)
    j = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    jm = j < H
    out = tl.zeros((BLOCK,), tl.float32)
    for s in tl.static_range(S):
        col = s * H + j
        up = tl.zeros((BLOCK,), tl.float32)
        for c in tl.static_range(KC):
            up += tl.load(part_ptr + (c * T + t) * S * H + col, mask=jm, other=0.0)
        if PENDING:
            r_s = tl.load(rn_ptr + t * S * H + col, mask=jm, other=0.0)
            tl.store(r_ptr + t * S * H + col, r_s, mask=jm)
        else:
            r_s = tl.load(r_ptr + t * S * H + col, mask=jm, other=0.0)
        scale = 1.0 + tl.load(norm_ptr + col, mask=jm, other=0.0).to(tl.float32)
        out += tl.sigmoid(up) * (r_s * (tl.load(rstd_ptr + t * S + s) * scale))
    tl.store(x_ptr + t * H + j, (out / S).to(x_ptr.dtype.element_ty), mask=jm)

S, H, RANK = 4, 2560, 320
dev = "cuda"
g = torch.Generator(device=dev).manual_seed(0)
up = (torch.randn(S * H, RANK, device=dev, generator=g) * 0.02).bfloat16()
norm = torch.randn(S * H, device=dev, generator=g) * 0.1
bj, bkc, warps = [int(x) for x in sys.argv[1:4]]
COPIES = 12                      # each call reads another copy of the up weights: past the L2, as in the model
ups = [up.clone() for _ in range(COPIES)]
kc = triton.cdiv(RANK, bkc)
for T in (1, 2, 4, 8):
    a = torch.randn(T, RANK, device=dev, generator=g)
    rstd = torch.rand(T, S, device=dev, generator=g) + 0.5
    r = torch.randn(T, S * H, device=dev, generator=g)
    rn = torch.randn(T, S * H, device=dev, generator=g)
    x0, r0 = torch.empty(T, H, dtype=torch.bfloat16, device=dev), r.clone()
    it = [0]
    def old():
        it[0] += 1
        up = ups[it[0] % COPIES]
        K._mix_rows[(triton.cdiv(H, K.MIX_ROWS[0]),)](a, rstd, up, r0, rn, norm, x0, T, H, RANK, S, True, K.MIX_ROWS[0],
                                                       512, K.GEMV_ROWS, num_warps=K.MIX_ROWS[1])
    x1, r1, part = torch.empty_like(x0), r.clone(), torch.empty(kc, T, S * H, device=dev)
    def new():
        up = ups[(it[0] + COPIES // 2) % COPIES]
        _up_part[(triton.cdiv(S * H, bj), kc)](a, up, part, T, S * H, RANK, bj, bkc, num_warps=warps)
        _mix_epi[(triton.cdiv(H, 512), T)](part, rstd, r1, rn, norm, x1, T, H, kc, S, True, 512, num_warps=4)
    for _ in range(10):
        old(); new()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(50):
            old(); new()
        torch.cuda.synchronize()
    per = {}
    for e in prof.events():
        if e.device_time > 0:
            per.setdefault(e.name[:20], []).append(e.device_time)
    med = {k: sorted(v)[len(v) // 2] for k, v in per.items()}
    ref = (torch.sigmoid(a.double() @ up.double().view(S, H, RANK).transpose(1, 2)) * (rn.double().view(T, S, H).transpose(0, 1) * rstd.double().t()[:, :, None] * (1 + norm.double().view(S, 1, H)))).sum(0) / S
    d_old = (x0.double() - ref).abs().max().item(); d_new = (x1.double() - ref).abs().max().item()
    o, n1, n2 = med['_mix_rows'], med['_up_part'], med['_mix_epi']
    print(f"T={T} bj={bj} bkc={bkc} w={warps}: old {o:.1f} us, split-K {n1:.1f} + {n2:.1f} = {n1 + n2:.1f} us "
          f"({o / (n1 + n2):.2f}x); |x - ref| old {d_old:.1e} new {d_new:.1e}; R same {torch.equal(r0, r1)}")
