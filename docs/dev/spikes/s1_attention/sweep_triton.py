"""Offline sweep to pin the Triton prototype's launch config per head_dim (no autotune at run time).

  CUDA_VISIBLE_DEVICES=0 python sweep_triton.py
"""

import itertools

import torch

from common import GEOMS, graph_time_us, make_paged_kv, replicas_for, save
from triton_attn import TritonPagedAttention, pick_splits

NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
SHAPES = [(1, 1024), (1, 32768), (8, 8192), (64, 8192)]
CONFIGS = [{"BLOCK_N": n, "num_warps": w, "num_stages": s}
           for n, w, s in itertools.product([16, 32, 64, 128], [2, 4, 8], [1, 2, 3])]


def time_cfg(g, kvs, nq, splits, cfg):
    B = kvs[0].lens.numel()
    t = TritonPagedAttention(g.hq, g.hkv, g.d, kvs[0].page_size, B, nq, splits, cfg)
    q = torch.randn(B * nq, g.hq, g.d, device="cuda", dtype=torch.bfloat16)
    return graph_time_us([lambda kv=kv: t.run(q, kv.k, kv.v, kv.page_table, kv.lens) for kv in kvs])


def main():
    rows = []
    for gname, nq in itertools.product(["qwen3_8b", "qwen3_8_27b"], [1, 5]):
        g = GEOMS[gname]
        for B, ctx in SHAPES:
            kvs = make_paged_kv(g, [ctx] * B, 16, layers=replicas_for(B * ctx * g.hkv * g.d * 4))
            splits = pick_splits(B, g.hkv, ctx, NUM_SMS, min_tokens=128)
            for cfg in CONFIGS:
                try:
                    us = time_cfg(g, kvs, nq, splits, cfg)
                except Exception:   # e.g. out of shared memory
                    us = None
                rows.append({"geom": gname, "nq": nq, "batch": B, "ctx": ctx, "splits": splits, **cfg, "us": us})
            best = min((r for r in rows if r["geom"] == gname and r["nq"] == nq and r["batch"] == B
                        and r["ctx"] == ctx and r["us"]), key=lambda r: r["us"])
            print("best", best, flush=True)
            # Split granularity for the best config.
            for mt in [64, 128, 256, 512, 1024]:
                cfg = {k: best[k] for k in ("BLOCK_N", "num_warps", "num_stages")}
                s = pick_splits(B, g.hkv, ctx, NUM_SMS, min_tokens=mt)
                rows.append({"geom": gname, "nq": nq, "batch": B, "ctx": ctx, "splits": s, "min_tokens": mt,
                             **cfg, "us": time_cfg(g, kvs, nq, s, cfg)})
                print("split", rows[-1], flush=True)
            del kvs
            torch.cuda.empty_cache()
    print("saved", save("sweep_triton", {"rows": rows}))


if __name__ == "__main__":
    main()
