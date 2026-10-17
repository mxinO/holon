"""One request's decode under nsys: a prompt of `--ctx` random tokens, then `--tokens` graphed decode steps at batch
1 inside an NVTX range "decode", bracketed by cudaProfilerStart/Stop (nsys `--capture-range=cudaProfilerApi`).

    nsys profile -t cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop \
        -o <report> python spikes/x3c_kernels/profile.py --model <gguf> [--ctx 1024] [--engine-args '<flags>'] \
        [--arm base|residual|router|prepare]
    python spikes/x4_gap/nsys_breakdown.py <report>.sqlite --tokens <tokens>
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import os
import random
import shlex
import sys

import torch
from holon.api import SamplingParams
from holon.core.cli import engine_config, engine_flags
from holon.core.engine import Engine

sys.path.insert(0, os.path.dirname(__file__))
import ab  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--engine-args", default="")
    ap.add_argument("--arm", default=ab.ARMS[-1], choices=ab.ARMS, help="the changes up to this one on (ab.py's)")
    a = ap.parse_args()
    flags = argparse.ArgumentParser()
    engine_flags(flags)
    cfg = engine_config(a.model, flags.parse_args(shlex.split(a.engine_args)))
    engine = Engine(dataclasses.replace(cfg, max_seqs=1, max_model_len=a.ctx + a.tokens + 64, prefix_cache=False))
    stack = contextlib.ExitStack()
    stack.enter_context(ab.arm(engine, a.arm))
    rng = random.Random(0)
    engine.add([rng.randrange(1000, 30000) for _ in range(a.ctx)],
               SamplingParams(temperature=0.0, max_tokens=a.tokens + 32, ignore_eos=True))
    while engine.sched.waiting or any(s.num_computed < s.prompt_len for s in engine.sched.running):
        engine.step()
    for _ in range(16):                              # eager, capture, replays
        engine.step()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("decode")
    for _ in range(a.tokens):
        engine.step()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
