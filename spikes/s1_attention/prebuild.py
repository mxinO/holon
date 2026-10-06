"""Pre-build every FlashInfer JIT module the spike (and a v0 attention plugin) uses, in one parallel ninja run.

The cache lands in $FLASHINFER_WORKSPACE_BASE/.cache/flashinfer/<version>/<arch>/cached_ops; set
FLASHINFER_CUDA_ARCH_LIST (e.g. "8.6 12.0") to build for other GPUs than the visible one, MAX_JOBS to cap jobs.

  FLASHINFER_WORKSPACE_BASE=$PWD/out/fi_ws CUDA_VISIBLE_DEVICES=0 python prebuild.py
"""

import itertools
import time

import torch
from flashinfer.jit import build_jit_specs
from flashinfer.jit.attention.modules import gen_batch_decode_module, gen_batch_prefill_module
from flashinfer.jit.xqa import gen_xqa_module

BF16, FP8, I32 = torch.bfloat16, torch.float8_e4m3fn, torch.int32


def specs():
    out = []
    for kv, d in itertools.product((BF16, FP8), (128, 256)):
        out.append(gen_batch_decode_module(BF16, kv, BF16, I32, d, d, 0, False, False))
        out.append(gen_batch_prefill_module("fa2", BF16, kv, BF16, I32, d, d, 0, False, False, False))
    if torch.cuda.get_device_capability(0)[0] in (9, 10, 12):
        for kv, page, (d, g), nq in itertools.product((BF16, FP8), (16, 64), ((128, 4), (256, 4), (256, 6)),
                                                      (1, 2, 3, 5)):
            out.append(gen_xqa_module(BF16, kv, page, d, g, False, BF16, nq))
    return out


if __name__ == "__main__":
    s = specs()
    t = time.perf_counter()
    build_jit_specs(s)
    print(f"built {len(s)} modules in {time.perf_counter() - t:.1f}s")
