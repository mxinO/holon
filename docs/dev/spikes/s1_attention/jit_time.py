"""FlashInfer JIT cost (question 6): first-use compile per module, warm load, and a parallel pre-build.

Every measurement uses a fresh FLASHINFER_WORKSPACE_BASE under out/ so the user's ~/.cache is untouched.

  CUDA_VISIBLE_DEVICES=0 python jit_time.py
"""

import os
import shutil
import subprocess
import sys
import time

import torch

from common import OUT, class_name, save

# (name, python expression returning a JitSpec); evaluated in a child with flashinfer imported.
BF16, FP8, I32 = "torch.bfloat16", "torch.float8_e4m3fn", "torch.int32"
SPECS = {
    f"decode_cudacore_{kv}_hd{d}":
        f"gen_batch_decode_module({BF16}, {dt}, {BF16}, {I32}, {d}, {d}, 0, False, False)"
    for kv, dt in [("bf16", BF16), ("fp8", FP8)] for d in (128, 256)
} | {
    f"prefill_fa2_{kv}_hd{d}":   # also serves tensor-core decode, verify rows and ragged prefill
        f"gen_batch_prefill_module('fa2', {BF16}, {dt}, {BF16}, {I32}, {d}, {d}, 0, False, False, False)"
    for kv, dt in [("bf16", BF16), ("fp8", FP8)] for d in (128, 256)
}
if torch.cuda.get_device_capability(0)[0] in (9, 10, 12):
    SPECS |= {
        f"xqa_{kv}_p{p}_hd{d}_g{g}_q{nq}":
            f"gen_xqa_module({BF16}, {dt}, {p}, {d}, {g}, False, {BF16}, {nq})"
        for kv, dt in [("bf16", BF16), ("fp8", FP8)] for p, d, g in [(16, 128, 4), (16, 256, 6)] for nq in (1, 3)
    }

CHILD = """
import sys, time, torch
from flashinfer.jit import build_jit_specs
from flashinfer.jit.attention.modules import gen_batch_decode_module, gen_batch_prefill_module
from flashinfer.jit.xqa import gen_xqa_module
specs = [eval(e) for e in sys.argv[2:]]
t = time.perf_counter()
if sys.argv[1] == "build":
    build_jit_specs(specs)
else:
    for s in specs:
        s.build_and_load()
print("SECONDS", time.perf_counter() - t)
"""


def child(base: str, mode: str, exprs: list[str], jobs: str | None = None) -> float:
    env = dict(os.environ, FLASHINFER_WORKSPACE_BASE=base)
    if jobs:
        env["MAX_JOBS"] = jobs
    p = subprocess.run([sys.executable, "-c", CHILD, mode, *exprs], env=env, capture_output=True, text=True)
    for line in p.stdout.splitlines():
        if line.startswith("SECONDS"):
            return float(line.split()[1])
    raise RuntimeError(p.stderr[-2000:])


def du(path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main():
    root = OUT / f"jit_cold_{class_name()}"   # per class: several hosts may share this directory
    shutil.rmtree(root, ignore_errors=True)
    res = {"per_module": [], "prebuild": {}}
    for name, expr in SPECS.items():
        base = root / name
        cold = child(str(base), "load", [expr])          # compile + load, sequential, in a fresh cache
        warm = child(str(base), "load", [expr])          # already compiled: load only
        res["per_module"].append({"module": name, "cold_s": cold, "warm_s": warm,
                                  "cache_bytes": du(base)})
        print(res["per_module"][-1], flush=True)
    base = root / "_all"
    t0 = time.perf_counter()
    build = child(str(base), "build", list(SPECS.values()))
    res["prebuild"] = {"modules": len(SPECS), "build_jit_specs_s": build, "wall_s": time.perf_counter() - t0,
                       "cache_bytes": du(base), "layout": sorted(str(p.relative_to(base)) for p in
                                                                 base.glob(".cache/flashinfer/*/*/cached_ops/*"))}
    print(res["prebuild"], flush=True)
    print("saved", save("jit_time", res))


if __name__ == "__main__":
    main()
