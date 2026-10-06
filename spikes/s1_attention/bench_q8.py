"""Can the Triton prototype carry quantized KV? q8_0 KV (int8 + fp16 scale per 32) vs bf16, decode and verify.

Reports error vs the fp32 reference on the dequantized cache (kernel error only) and latency next to
FlashInfer's fp8-e4m3 tensor-core path (the other half-size KV layout).

  CUDA_VISIBLE_DEVICES=0 python bench_q8.py
"""

import torch

from common import GEOMS, PagedKV, err, graph_time_us, make_paged_kv, reference, replicas_for, save
from fi_kernels import FIDecode, FIPagedPrefill, sm_scale
from triton_attn import TritonPagedAttention, pick_splits

NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count


def quantize_q8_0(x: torch.Tensor):
    """[..., D] bf16 -> int8 [..., D], fp16 scales [..., D/32] (GGUF q8_0: scale = absmax / 127)."""
    blocks = x.float().unflatten(-1, (-1, 32))
    scale = blocks.abs().amax(-1) / 127
    q = torch.round(blocks / scale.clamp_min(1e-12)[..., None]).to(torch.int8)
    return q.flatten(-2), scale.half()


def dequant(q, s):
    return (q.float().unflatten(-1, (-1, 32)) * s.float()[..., None]).flatten(-2)


def main():
    rows = []
    for gname in ["qwen3_8b", "qwen3_8_27b"]:
        g = GEOMS[gname]
        for B, ctx in [(1, 1024), (1, 32768), (8, 8192), (64, 8192)]:
            layers = replicas_for(B * ctx * g.hkv * g.d * 4)
            kvs = make_paged_kv(g, [ctx] * B, 16, layers=layers)
            q8 = [(*quantize_q8_0(kv.k), *quantize_q8_0(kv.v)) for kv in kvs]
            fp8 = [PagedKV(kv.k.to(torch.float8_e4m3fn), kv.v.to(torch.float8_e4m3fn), kv.page_table, kv.lens, 16)
                   for kv in kvs]
            for nq in (1, 3):
                q = torch.randn(B * nq, g.hq, g.d, device="cuda", dtype=torch.bfloat16)
                t = TritonPagedAttention(g.hq, g.hkv, g.d, 16, B, nq, pick_splits(B, g.hkv, ctx, NUM_SMS))
                kv0 = kvs[0]
                row = {"geom": gname, "batch": B, "ctx": ctx, "nq": nq, "layers": layers}
                row["triton_bf16_us"] = graph_time_us(
                    [lambda kv=kv: t.run(q, kv.k, kv.v, kv.page_table, kv.lens) for kv in kvs])
                row["triton_q8_0_us"] = graph_time_us(
                    [lambda kv=kv, z=z: t.run(q, z[0], z[2], kv.page_table, kv.lens, z[1], z[3])
                     for kv, z in zip(kvs, q8)])
                fi = FIDecode(g, fp8[0], True) if nq == 1 else FIPagedPrefill(g, fp8[0], nq)
                fi.plan()
                row["fi_fp8_us"] = graph_time_us([lambda kv=kv: fi.run(q, kv) for kv in fp8])
                if B * ctx <= 8 * 8192:   # error vs fp32 reference over the dequantized q8_0 cache
                    z = q8[0]
                    deq = PagedKV(dequant(z[0], z[1]), dequant(z[2], z[3]), kv0.page_table, kv0.lens, 16)
                    out = t.run(q, z[0], z[2], kv0.page_table, kv0.lens, z[1], z[3])
                    row["q8_0_err"] = err(out, reference(q, deq, nq, sm_scale(g)))
                    row["q8_0_vs_bf16_cache_err"] = err(out, reference(q, kv0, nq, sm_scale(g)))
                rows.append(row)
                print(row, flush=True)
            del kvs, q8, fp8
            torch.cuda.empty_cache()
    print("saved", save("bench_q8", {"rows": rows}))


if __name__ == "__main__":
    main()
