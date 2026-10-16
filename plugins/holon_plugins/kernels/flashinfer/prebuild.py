"""Build the FlashInfer JIT modules this plugin runs, ahead of time, for one SM (`holon prebuild`; release design D1):

    python -m holon_plugins.kernels.flashinfer.prebuild --sm 86

FlashInfer keeps them in its cache, one directory per target architecture (`$FLASHINFER_WORKSPACE_BASE/.cache/
flashinfer/<version>/<arch>/cached_ops`, default base: the home directory); an engine on a GPU of that SM loads them
and compiles nothing. Every module the plugin's kernels can select is built: paged prefill and tensor-core decode
(FlashInfer's FA2 prefill module) for each supported q dtype, KV dtype and head dim, rope and silu_and_mul.

Each module is built by its own ninja, in its own directory: FlashInfer runs that build before it loads a module, and
ninja recompiles a module whose directory has no build log — which a combined build (`build_jit_specs`, one log for
all) leaves, so every module compiled again at first use.
"""
from __future__ import annotations

import argparse
import itertools
import os
import time

ARCHS = {86: "8.6", 89: "8.9", 120: "12.0f"}   # FlashInfer's arch list entries (12.0f: the family target)
HEAD_DIMS = (64, 128, 256)                      # plugin._supports
FP8_KV_SM = 89                                  # plugin.register: e4m3 KV from sm_89 on


def specs(sm: int) -> list:
    import torch
    from flashinfer.jit import gen_act_and_mul_module, gen_batch_prefill_module
    from flashinfer.jit.rope import gen_rope_module

    out = [gen_rope_module(), gen_act_and_mul_module("silu")]
    for q, d in itertools.product((torch.bfloat16, torch.float16), HEAD_DIMS):
        for kv in (q, torch.float8_e4m3fn) if sm >= FP8_KV_SM else (q,):
            out.append(gen_batch_prefill_module("fa2", q, kv, q, torch.int32, d, d, 0, False, False, False))
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m holon_plugins.kernels.flashinfer.prebuild", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sm", type=int, required=True, choices=sorted(ARCHS))
    sm = p.parse_args(argv).sm
    os.environ["FLASHINFER_CUDA_ARCH_LIST"] = ARCHS[sm]      # read when flashinfer is imported: this process only
    from flashinfer.jit import env

    built, t = specs(sm), time.perf_counter()
    for spec in built:
        spec.build(verbose=False)
    print(f"sm {sm}: {len(built)} FlashInfer modules in {env.FLASHINFER_JIT_DIR} ({time.perf_counter() - t:.0f} s)",
          flush=True)


if __name__ == "__main__":
    main()
