"""Search vLLM's original autotune space for every chunk-prefill kernel and record the winners.

The winners become `fla_ops.utils.PINNED` (design §5.6 rule 7: no run-time autotuning in plug-serve).
Usage: CUDA_VISIBLE_DEVICES=0 python tune.py results/tune_sm86.json
"""

import json
import os
import sys

os.environ["S3_FLA_AUTOTUNE"] = "1"  # must precede the fla_ops import (inside gdn)

import torch  # noqa: E402
import triton  # noqa: E402

import gdn  # noqa: E402
from fla_ops import chunk_delta_h, chunk_o, chunk_scaled_dot_kkt, cumsum, solve_tril, wy_fast  # noqa: E402

KERNELS = {
    "chunk_local_cumsum_scalar_kernel": cumsum.chunk_local_cumsum_scalar_kernel,
    "chunk_scaled_dot_kkt_fwd_kernel": chunk_scaled_dot_kkt.chunk_scaled_dot_kkt_fwd_kernel,
    "merge_16x16_to_64x64_inverse_kernel": solve_tril.merge_16x16_to_64x64_inverse_kernel,
    "recompute_w_u_fwd_kernel": wy_fast.recompute_w_u_fwd_kernel,
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64": chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
    "chunk_fwd_kernel_o": chunk_o.chunk_fwd_kernel_o,
}


def autotuner(fn):
    while not isinstance(fn, triton.runtime.Autotuner):
        fn = fn.fn
    return fn


def inputs(shape, T, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    Hg, H, K, V = shape["Hg"], shape["H"], shape["K"], shape["V"]
    q = torch.randn(1, T, Hg, K, generator=g, device="cuda").bfloat16()
    k = torch.randn(1, T, Hg, K, generator=g, device="cuda").bfloat16()
    v = torch.randn(1, T, H, V, generator=g, device="cuda").bfloat16()
    gate = -torch.rand(1, T, H, generator=g, device="cuda") * 2  # log-decay in [-2, 0]
    beta = torch.rand(1, T, H, generator=g, device="cuda").bfloat16()
    h0 = torch.zeros(1, H, V, K, device="cuda")
    return q, k, v, gate, beta, h0


def main(out_path):
    result = dict(device=gdn.device_info(), shapes={})
    for model in ("9b", "27b"):
        shape = gdn.gdn_shape(model)
        q, k, v, gate, beta, h0 = inputs(shape, 8192)
        gdn.kernel_chunk(q, k, v, gate, beta, h0, gdn.cu_seqlens_of([8192]))  # triggers the search
        torch.cuda.synchronize()
        per_kernel = {}
        for name, fn in KERNELS.items():
            tuner = autotuner(fn)
            per_kernel[name] = {
                str(key): dict(kwargs=cfg.kwargs, num_warps=cfg.num_warps, num_stages=cfg.num_stages)
                for key, cfg in tuner.cache.items()
            }
            tuner.cache.clear()  # next model shape re-tunes
        result["shapes"][model] = dict(shape=shape, T=8192, best=per_kernel)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
