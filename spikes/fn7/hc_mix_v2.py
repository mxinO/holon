"""FN §21.13, the HC mix's gate: the whole GEMV-shaped mix as today's `_down_gemv` and `_down_sum`, a split-K up
projection (partial sums per K chunk) and a fixed-order epilogue (the chunks, then the streams, in order) — at 1, 4 and
8 rows; device time against today's mix (12 copies of the weights, every call another: past the L2), `l0.check`
against the oracle. Usage: hc_mix_v2.py [bnv kchunk warps bj bkc warps block]"""
import sys, torch, triton
import triton.language as tl
from holon_judge import l0
from holon_plugins.kernels.hc_triton import kernel as K
from holon_std.hyper import reference as hc

@triton.jit
def _up_part(a_ptr, up_ptr, upp_ptr, T, SH, RANK, BJ: tl.constexpr, BKC: tl.constexpr):
    j = tl.program_id(0) * BJ + tl.arange(0, BJ)
    c = tl.program_id(1)
    kk = c * BKC + tl.arange(0, BKC)
    jm, km = j < SH, kk < RANK
    w = tl.load(up_ptr + j[:, None].to(tl.int64) * RANK + kk[None, :], mask=jm[:, None] & km[None, :],
                other=0.0).to(tl.float32)
    for t in range(T):
        a = tl.load(a_ptr + t * RANK + kk, mask=km, other=0.0)
        tl.store(upp_ptr + (c * T + t) * SH + j, tl.sum(w * a[None, :], 1), mask=jm)

@triton.jit
def _epi(upp_ptr, sq_ptr, r_ptr, rn_ptr, norm_ptr, x_ptr, T, H, eps, S: tl.constexpr, KS: tl.constexpr,
         KC: tl.constexpr, PENDING: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(1)
    j = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    jm = j < H
    out = tl.zeros((BLOCK,), tl.float32)
    for s in tl.static_range(S):
        sq = 0.0
        for c in tl.static_range(KS):
            sq += tl.load(sq_ptr + (s * KS + c) * T + t)
        rstd = tl.rsqrt(sq / H + eps)
        col = s * H + j
        up = tl.zeros((BLOCK,), tl.float32)
        for c in tl.static_range(KC):
            up += tl.load(upp_ptr + (c * T + t) * S * H + col, mask=jm, other=0.0)
        if PENDING:
            r_s = tl.load(rn_ptr + t * S * H + col, mask=jm, other=0.0)
            tl.store(r_ptr + t * S * H + col, r_s, mask=jm)
        else:
            r_s = tl.load(r_ptr + t * S * H + col, mask=jm, other=0.0)
        scale = rstd * (1.0 + tl.load(norm_ptr + col, mask=jm, other=0.0).to(tl.float32))
        out += tl.sigmoid(up) * (r_s * scale)
    tl.store(x_ptr + t * H + j, (out / S).to(x_ptr.dtype.element_ty), mask=jm)

def mix_v2(r, norm, down, up, inject, y_prev, w_prev, streams, eps, dtype, cfg):
    (bnv, kchunk, dwarps), (bj, bkc, uwarps), block = cfg
    t, width = r.shape
    hidden, rank = width // streams, down.shape[0]
    x = torch.empty(t, hidden, dtype=dtype, device=r.device)
    w = None if inject is None else torch.empty(t, streams, dtype=torch.float32, device=r.device)
    pending = y_prev is not None
    rstd = torch.empty(t, streams, dtype=torch.float32, device=r.device)
    sink = w if w is not None else rstd
    y, wp = (y_prev, w_prev) if pending else (rstd, rstd)
    inj = inject if inject is not None else down
    nout = rank + (streams if inject is not None else 0)
    ks = hidden // kchunk
    part = torch.empty(streams * ks, t, nout, dtype=torch.float32, device=r.device)
    sq = torch.empty(streams * ks, t, dtype=torch.float32, device=r.device)
    rn = torch.empty_like(r) if pending else r
    K._down_gemv[(triton.cdiv(nout, bnv), streams * ks)](r, y, wp, norm, down, inj, part, sq, rn, t, hidden, rank,
                                                         nout, streams, pending, inject is not None, bnv, kchunk,
                                                         num_warps=dwarps)
    a = torch.empty(t, rank, dtype=torch.float32, device=r.device)
    K._down_sum[(triton.cdiv(t, K.BM), triton.cdiv(nout, K.BN))](part, sq, a, sink, rstd, t, hidden, rank, nout, eps,
                                                                 streams, ks, K.BM, K.BN, num_warps=K.WARPS)
    kc = triton.cdiv(rank, bkc)
    upp = torch.empty(kc, t, width, dtype=torch.float32, device=r.device)
    _up_part[(triton.cdiv(width, bj), kc)](a, up, upp, t, width, rank, bj, bkc, num_warps=uwarps)
    _epi[(triton.cdiv(hidden, block), t)](upp, sq, r, rn, norm, x, t, hidden, eps, streams, ks, kc, pending, block,
                                          num_warps=4)
    return x, w

S, H, RANK, EPS = 4, 2560, 320, 1e-6
dev = "cuda"
g = torch.Generator().manual_seed(0)
width = S * H
norm = (torch.randn(width, generator=g) * 0.1).bfloat16()
down = (torch.randn(RANK, width, generator=g) / width ** 0.5).bfloat16()
up = (torch.randn(width, RANK, generator=g) / RANK ** 0.5).bfloat16()
gate = (torch.randn(S, width, generator=g) / width ** 0.5).bfloat16()
COPIES = 12
sets = [tuple(x.cuda().clone() for x in (norm, down, up, gate)) for _ in range(COPIES)]
args = sys.argv[1:]
cfg = ((int(args[0]), int(args[1]), int(args[2])), (int(args[3]), int(args[4]), int(args[5])), int(args[6])) if args \
    else ((8, 512, 1), (64, 64, 4), 256)
for T in (1, 4, 8):
    r = torch.randn(T, width, generator=g) * torch.linspace(0.5, 4.0, S).repeat_interleave(H)
    y = torch.randn(T, H, generator=g).bfloat16()
    wp = 2 * torch.rand(T, S, generator=g)
    rc, yc, wpc = r.cuda(), y.cuda(), wp.cuda()
    ok = True
    for pending in (False, True):
        ex = (yc, wpc) if pending else (None, None)
        ours = rc.clone()
        x, w = mix_v2(ours, *sets[0], *ex, S, EPS, torch.bfloat16, cfg)
        r_ref = r.clone()
        x_ref, w_ref = hc.hc_gr_mix(r_ref, norm, down, up, gate, *((y, wp) if pending else (None, None)), S, EPS,
                                    torch.bfloat16)
        ok &= l0.check(x, x_ref, torch.bfloat16, width).passed and l0.check(w, w_ref, torch.float32, width).passed
        ok &= l0.check(ours, r_ref, torch.float32, 1).passed
    it = [0]
    def old():
        it[0] += 1
        K.hc_gr_mix(rc.clone() if False else rc, *sets[it[0] % COPIES], yc, wpc, S, EPS, torch.bfloat16)
    def new():
        it[0] += 1
        mix_v2(rc, *sets[it[0] % COPIES], yc, wpc, S, EPS, torch.bfloat16, cfg)
    for _ in range(10):
        old(); new()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(60):
            old()
        torch.cuda.synchronize()
    o = sum(e.device_time for e in prof.events() if e.device_time > 0 and e.name.startswith("_")) / 60
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(60):
            new()
        torch.cuda.synchronize()
    per = {}
    for e in prof.events():
        if e.device_time > 0 and e.name.startswith("_"):
            per[e.name[:12]] = per.get(e.name[:12], 0) + e.device_time / 60
    n = sum(per.values())
    print(f"T={T} cfg={cfg}: today {o:.1f} us, v2 {n:.1f} us {({k: round(v, 1) for k, v in per.items()})} "
          f"({o / n:.2f}x); l0 {ok}")
