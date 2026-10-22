"""FN step 7 item 7: the HC prefill projections as bf16x3 — an fp32 activation split exactly into three bf16 terms,
each times the (exactly bf16) weights on tensor cores with fp32 output, summed — against the float32 SGEMMs (D10):
time per 4,096-row pass and the error against a float64 product, at FN's shapes (4 streams × 2,560, rank 320 + 4
inject rows). Usage: hc_bf16x3.py"""
from __future__ import annotations

import json

import torch


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
        scale = ref.abs().max().item()
        out[name] = {"sgemm_ms": round(timed(lambda x=x, w=w: torch.bmm(x, w.float())), 3),
                     "sgemm_cast_w_ms": round(timed(lambda x=x, wf=w.float(): torch.bmm(x, wf)), 3),
                     "bf16x3_ms": round(timed(lambda x=x, w=w: bf16x3_bmm(x, w)), 3),
                     "err_sgemm": (f32.double() - ref).abs().max().item() / scale,
                     "err_bf16x3": (x3.double() - ref).abs().max().item() / scale}
        print(name, out[name], flush=True)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
