"""One request's batch-1 decode under nsys, with every kernel attributable to the step-loop phase that issued it.

After a prompt of `--ctx` random tokens and warm-up steps, `--eager` decode steps run with graphs off and NVTX ranges
around the executor's phases (`prepare` and inside it each planner's `prepare_device`, each planner's `plan_bounds`, `trunk`, `logits`, `sample`,
`store`, the uploader's copy, the launch's result copies), then `--tokens` graphed decode steps run inside a range
"decode", each step in a range "launch". A graph replays the same launches as an eager step, in the same order, so
`inventory.py` names each graphed kernel by the eager step's phase at its index.

    nsys profile -t cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop \
        -o <report> python spikes/x3d_core/profile.py --model <gguf> [--ctx 1024] [--engine-args '<flags>'] [--arm base]
    nsys export --type sqlite -o <report>.sqlite <report>.nsys-rep
    python spikes/x3d_core/inventory.py <report>.sqlite
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import functools
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


def ranged(obj, attr: str, name: str, stack: contextlib.ExitStack) -> None:
    """Wrap `obj.attr` in an NVTX range `name` (undone by `stack`)."""
    fn = getattr(obj, attr)

    @functools.wraps(fn)
    def wrapper(*args, **kw):
        torch.cuda.nvtx.range_push(name)
        try:
            return fn(*args, **kw)
        finally:
            torch.cuda.nvtx.range_pop()
    setattr(obj, attr, wrapper)
    stack.callback(setattr, obj, attr, fn)


def instrument(engine: Engine, stack: contextlib.ExitStack) -> None:
    ex = engine.executor
    for attr, name in (("_prepare", "prepare"), ("_logits", "logits"), ("_store", "store"), ("_observe", "observe")):
        ranged(ex, attr, name, stack)
    ranged(ex.model, "trunk", "trunk", stack)
    ranged(ex.sampler, "sample", "sample", stack)
    for up in {id(u): u for u in (ex.uploader, ex.small_uploader)}.values():
        ranged(up, "end", "upload", stack)
    ranged(ex.writes, "run", "writes", stack)
    for group, planner in ex.planners.items():
        ranged(planner, "prepare_device", f"prepare_device:{group}", stack)
        ranged(planner, "plan_bounds", f"plan_bounds:{group}", stack)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--eager", type=int, default=4)
    ap.add_argument("--engine-args", default="")
    ap.add_argument("--arm", default=ab.ARMS[-1], choices=ab.ARMS, help="the changes up to this one on (ab.py's)")
    a = ap.parse_args()
    flags = argparse.ArgumentParser()
    engine_flags(flags)
    cfg = engine_config(a.model, flags.parse_args(shlex.split(a.engine_args)))
    engine = Engine(dataclasses.replace(cfg, max_seqs=1, max_model_len=a.ctx + a.eager + a.tokens + 64,
                                        prefix_cache=False))
    armed = contextlib.ExitStack()                  # kept alive to the end: dropping it would undo the arm
    armed.enter_context(ab.arm(engine, a.arm))
    rng = random.Random(0)
    engine.add([rng.randrange(1000, 30000) for _ in range(a.ctx)],
               SamplingParams(temperature=0.0, max_tokens=a.eager + a.tokens + 48, ignore_eos=True))
    while engine.sched.waiting or any(s.num_computed < s.prompt_len for s in engine.sched.running):
        engine.step()
    for _ in range(16):                              # eager, capture, replays
        engine.step()
    torch.cuda.synchronize()
    with contextlib.ExitStack() as stack:
        instrument(engine, stack)
        ranged(engine.executor, "launch", "launch", stack)
        graphs, engine.executor.graphs = engine.executor.graphs, None
        for _ in range(2):
            engine.step()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push("eager")
        for _ in range(a.eager):
            engine.step()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        engine.executor.graphs = graphs
        for _ in range(4):
            engine.step()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("decode")
        for _ in range(a.tokens):
            engine.step()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()
    armed.close()


if __name__ == "__main__":
    main()
