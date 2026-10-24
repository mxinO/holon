"""FN step 7 item 7: the HC prefill projections as bf16x3 — an fp32 activation split exactly into three bf16 terms,
each times the (exactly bf16) weights on tensor cores with fp32 output, summed — against the float32 SGEMMs (D10):
time per 4,096-row pass and the error against a float64 product, at FN's shapes (4 streams × 2,560, rank 320 + 4
inject rows). Usage: hc_bf16x3.py"""
from __future__ import annotations

import functools
import json

import torch
import triton
import triton.language as tl


@triton.jit
def _mm3(a_ptr, b_ptr, c_ptr, M, N, K, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """C[b] = A[b] (float32) · B[b] (bf16) as three bf16 products of A's exact split on tensor cores, f32 sums."""
    pid_m, pid_n, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rm, rn, rk = pid_m * BM + tl.arange(0, BM), pid_n * BN + tl.arange(0, BN), tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(a_ptr + b * sab + rm[:, None] * sam + (k0 + rk)[None, :] * sak,
                    mask=(rm[:, None] < M) & ((k0 + rk)[None, :] < K), other=0.0)
        w = tl.load(b_ptr + b * sbb + (k0 + rk)[:, None] * sbk + rn[None, :] * sbn,
                    mask=((k0 + rk)[:, None] < K) & (rn[None, :] < N), other=0.0)
        hi = a.to(tl.bfloat16)
        r = a - hi.to(tl.float32)
        mid = r.to(tl.bfloat16)
        lo = (r - mid.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(hi, w, acc)
        acc = tl.dot(mid, w, acc)
        acc = tl.dot(lo, w, acc)
    tl.store(c_ptr + b * scb + rm[:, None] * scm + rn[None, :] * scn, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def fused_bmm(x: torch.Tensor, w: torch.Tensor, bm: int = 128, bn: int = 64, bk: int = 32) -> torch.Tensor:
    b, m, k = x.shape
    n = w.shape[2]
    c = torch.empty(b, m, n, dtype=torch.float32, device=x.device)
    _mm3[(triton.cdiv(m, bm), triton.cdiv(n, bn), b)](x, w, c, m, n, k, *x.stride(), *w.stride(), *c.stride(),
                                                      bm, bn, bk, num_warps=4, num_stages=3)
    return c


def split3(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hi = x.bfloat16()
    r = x - hi.float()
    mid = r.bfloat16()
    return hi, mid, (r - mid.float()).bfloat16()


def bf16x3_bmm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    hi, mid, lo = split3(x)
    p = torch.bmm(hi, w, out_dtype=torch.float32)
    p += torch.bmm(mid, w, out_dtype=torch.float32)
    p += torch.bmm(lo, w, out_dtype=torch.float32)
    return p


def timed(fn, reps: int = 20) -> float:
    fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    g = torch.Generator(device="cuda").manual_seed(0)
    s, n, h, nout, rank = 4, 4096, 2560, 324, 320
    xn = torch.randn(s, n, h, device="cuda", generator=g)
    wd = (torch.randn(s, h, nout, device="cuda", generator=g) * h ** -0.5).bfloat16()
    a = torch.randn(1, n, rank, device="cuda", generator=g)
    wu = (torch.randn(1, rank, s * h, device="cuda", generator=g) * rank ** -0.5).bfloat16()
    out = {"gpu": torch.cuda.get_device_name()}
    for name, x, w in (("down", xn, wd), ("up", a, wu)):
        ref = torch.bmm(x.double(), w.double())
        f32 = torch.bmm(x, w.float())
        x3 = bf16x3_bmm(x, w)
        fx = fused_bmm(x, w)
        scale = ref.abs().max().item()
        out[name] = {"sgemm_ms": round(timed(lambda x=x, w=w: torch.bmm(x, w.float())), 3),
                     "sgemm_cast_w_ms": round(timed(functools.partial(torch.bmm, x, w.float())), 3),
                     "bf16x3_ms": round(timed(lambda x=x, w=w: bf16x3_bmm(x, w)), 3),
                     "fused_ms": {f"{bm}x{bn}x{bk}": round(timed(functools.partial(fused_bmm, x, w, bm, bn, bk)), 3)
                                  for bm, bn, bk in ((128, 64, 32), (128, 128, 32), (64, 128, 32), (128, 64, 64))},
                     "err_fused": (fx.double() - ref).abs().max().item() / scale,
                     "err_sgemm": (f32.double() - ref).abs().max().item() / scale,
                     "err_bf16x3": (x3.double() - ref).abs().max().item() / scale}
        print(name, out[name], flush=True)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
