"""Correctness and latency of the attention paths (questions 1, 3, 4).

  CUDA_VISIBLE_DEVICES=0 python bench.py --kv bf16            # full matrix
  CUDA_VISIBLE_DEVICES=0 python bench.py --kv fp8 --modes decode verify

Latency = median per-call time of the kernel captured in a CUDA graph (plan done before capture), averaged
over enough KV replicas (one per "layer", sharing the plan) to exceed 3x L2.
"""

import argparse
import itertools
import traceback

import torch

from common import GEOMS, Geom, err, graph_time_us, make_paged_kv, reference, replicas_for, save
from fi_kernels import XQA, FIDecode, FIPagedPrefill, FIRaggedPrefill, sm_scale
from triton_attn import TritonOnPagedKV, pick_splits

FP8 = torch.float8_e4m3fn
NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
HAS_XQA = torch.cuda.get_device_capability(0)[0] in (9, 10, 12)


def kernels_for(mode: str, g: Geom, kv, nq: int, max_len: int, kv_dtype) -> dict:
    """name -> zero-arg factory returning an object with run(q, kv). Host plan() is done here."""
    B = kv.lens.numel()

    def fi_decode(use_tc, meta="host"):
        k = FIDecode(g, kv, use_tc, meta)
        k.plan()
        return k

    def fi_paged(meta="host"):
        k = FIPagedPrefill(g, kv, nq, meta)
        k.plan()
        return k

    def triton_k():
        return TritonOnPagedKV(g.hq, g.hkv, g.d, kv.page_size, B, nq, pick_splits(B, g.hkv, max_len, NUM_SMS))

    ks = {}
    if mode == "decode":
        ks["fi_decode_cudacore"] = lambda: fi_decode(False)
        ks["fi_decode_tc"] = lambda: fi_decode(True)
        ks["fi_decode_tc_devmeta"] = lambda: fi_decode(True, "device")
    else:
        ks["fi_paged_prefill"] = lambda: fi_paged()
        ks["fi_paged_prefill_devmeta"] = lambda: fi_paged("device")
    if kv_dtype == torch.bfloat16:
        ks["triton"] = triton_k
    if HAS_XQA:
        ks["xqa"] = lambda: XQA(g, kv, nq)
    return ks


def correctness(g: Geom, page: int, kv_dtype, nqs: list[int]) -> list[dict]:
    """Ragged lengths, odd sizes, vs the fp32 torch reference."""
    rows = []
    lens = [1, 17, 1000, 4097, 513, 2 * page, 8191, 64]
    kv = make_paged_kv(g, lens, page, dtype=kv_dtype, seed=1)[0]
    for nq in nqs:
        mode = "decode" if nq == 1 else "verify"
        lens_ok = [max(n, nq) for n in lens]
        kv.lens.copy_(torch.tensor(lens_ok, dtype=torch.int32))
        q = torch.randn(len(lens) * nq, g.hq, g.d, device="cuda", dtype=torch.bfloat16)
        ref = reference(q, kv, nq, sm_scale(g))
        for name, mk in kernels_for(mode, g, kv, nq, max(lens), kv_dtype).items():
            row = {"geom": g.name, "page": page, "nq": nq, "kernel": name}
            try:
                row.update(err(mk().run(q, kv), ref))
            except Exception as e:  # unsupported config: record why
                row["error"] = str(e).splitlines()[0][:300]
            rows.append(row)
            print("corr", row, flush=True)
    return rows


def latency(g: Geom, page: int, batch: int, ctx: int, kv_dtype, nqs: list[int]) -> list[dict]:
    rows = []
    bytes_per_layer = batch * ctx * g.hkv * g.d * 2 * torch.finfo(kv_dtype).bits // 8
    layers = replicas_for(bytes_per_layer)
    kvs = make_paged_kv(g, [ctx] * batch, page, dtype=kv_dtype, layers=layers)
    for nq in nqs:
        mode = "decode" if nq == 1 else "verify"
        q = torch.randn(batch * nq, g.hq, g.d, device="cuda", dtype=torch.bfloat16)
        outs = {}
        for name, mk in kernels_for(mode, g, kvs[0], nq, ctx, kv_dtype).items():
            row = {"geom": g.name, "page": page, "batch": batch, "ctx": ctx, "nq": nq, "kernel": name,
                   "layers": layers}
            k = None
            try:
                k = mk()
                outs[name] = k.run(q, kvs[0]).clone()
                row["us"] = graph_time_us([lambda kv=kv: k.run(q, kv) for kv in kvs])
                if name == "triton":
                    row["splits"] = k.splits
            except Exception as e:
                row["error"] = str(e).splitlines()[0][:300]
            rows.append(row)
            print("lat", row, flush=True)
            del k
        base = outs.get("fi_decode_tc", outs.get("fi_paged_prefill"))
        for r in rows:
            if r["nq"] == nq and r["kernel"] in outs and base is not None:
                r["maxdiff_vs_fi"] = (outs[r["kernel"]].float() - base.float()).abs().max().item()
    del kvs
    torch.cuda.empty_cache()
    return rows


def prefill_latency(g: Geom, batch: int, ctx: int, page: int) -> list[dict]:
    """First chunk: ragged (K/V from the projection) and paged (K/V read back from the cache)."""
    rows = []
    q = torch.randn(batch * ctx, g.hq, g.d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch * ctx, g.hkv, g.d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    kv = make_paged_kv(g, [ctx] * batch, page)[0]
    for name in ["fi_ragged_prefill", "fi_paged_prefill"]:
        row = {"geom": g.name, "page": page, "batch": batch, "ctx": ctx, "kernel": name}
        try:
            if name == "fi_ragged_prefill":
                r = FIRaggedPrefill(g, [ctx] * batch)
                fn = lambda: r.run(q, k, v)
            else:
                r = FIPagedPrefill(g, kv, ctx, graph=False)   # prefill runs eager (v0): no graph buffers
                r.plan()
                fn = lambda: r.run(q, kv)
            row["us"] = graph_time_us([fn], warmup=2, reps=5)
            flops = 4 * batch * g.hq * g.d * ctx * ctx / 2                  # causal
            row["tflops"] = flops / row["us"] / 1e6
        except Exception as e:
            row["error"] = str(e).splitlines()[0][:300]
        rows.append(row)
        print("prefill", row, flush=True)
    return rows


def prefill_correctness(g: Geom) -> list[dict]:
    lens = [1, 300, 2048, 77]
    q = torch.randn(sum(lens), g.hq, g.d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(sum(lens), g.hkv, g.d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    out = FIRaggedPrefill(g, lens).run(q, k, v)
    worst = {"max_abs": 0.0, "mean_abs": 0.0, "nan": False}
    off = 0
    for n in lens:   # reference: dense causal per sequence
        kk = k[off:off + n].float().repeat_interleave(g.group, 1).transpose(0, 1)
        vv = v[off:off + n].float().repeat_interleave(g.group, 1).transpose(0, 1)
        s = q[off:off + n].float().transpose(0, 1) @ kk.transpose(1, 2) * sm_scale(g)
        s.masked_fill_(torch.ones(n, n, device="cuda", dtype=torch.bool).triu(1), float("-inf"))
        e = err(out[off:off + n], (s.softmax(-1) @ vv).transpose(0, 1))
        worst = {"max_abs": max(worst["max_abs"], e["max_abs"]), "mean_abs": max(worst["mean_abs"], e["mean_abs"]),
                 "nan": worst["nan"] or e["nan"]}
        off += n
    row = {"geom": g.name, "kernel": "fi_ragged_prefill", **worst}
    print("corr", row, flush=True)
    return [row]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv", default="bf16", choices=["bf16", "fp8"])
    ap.add_argument("--geoms", nargs="+", default=list(GEOMS))
    ap.add_argument("--pages", nargs="+", type=int, default=[16, 64])
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 8, 64])
    ap.add_argument("--ctx", nargs="+", type=int, default=[1024, 8192, 32768])
    ap.add_argument("--nqs", nargs="+", type=int, default=[1, 2, 3, 5])
    ap.add_argument("--modes", nargs="+", default=["decode", "verify", "prefill"])
    ap.add_argument("--max-prefill-tokens", type=int, default=65536)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    kv_dtype = torch.bfloat16 if a.kv == "bf16" else FP8
    nqs = [n for n in a.nqs if ("decode" in a.modes and n == 1) or ("verify" in a.modes and n > 1)]
    res = {"args": vars(a), "correctness": [], "latency": [], "prefill": []}
    name = f"bench_{a.kv}{a.tag}"
    for gname in a.geoms:
        g = GEOMS[gname]
        for page in a.pages:
            res["correctness"] += correctness(g, page, kv_dtype, nqs)
            for batch, ctx in itertools.product(a.batches, a.ctx):
                try:
                    res["latency"] += latency(g, page, batch, ctx, kv_dtype, nqs)
                except Exception:
                    traceback.print_exc()
            save(name, res)
        if "prefill" in a.modes and kv_dtype == torch.bfloat16:
            res["correctness"] += prefill_correctness(g)
            for batch, ctx in itertools.product(a.batches, a.ctx):
                if batch * ctx <= a.max_prefill_tokens:
                    res["prefill"] += prefill_latency(g, batch, ctx, 16)
        print("saved", save(name, res), flush=True)


if __name__ == "__main__":
    main()
