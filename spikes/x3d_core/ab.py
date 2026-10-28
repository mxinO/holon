"""Step 3d's keep-or-revert measurement (docs/dev/v1-offload.md §14, docs/dev/m07-perf.md §1.1's rule): each change to the
step loop's per-step device work, measured incrementally on one engine — graphed decode at batch 1 by per-step wall,
after a prompt of each context (and, optionally, eager prefill of single prompts by wall).

The arms add the changes in order (`ARMS`), each on top of the previous: `base` (bdd8422's executor methods and
uploader layout, `base.py`), then each change. A change is switched off by restoring the code it replaced, in this
process; every switch drops the captured graphs, so each arm replays its own path. Rounds visit the arms in ABBA
order (forward, then backward); each row's per-round difference against the previous arm gives the mean saving and
its sd (kept if ≥ max(0.5%, 3 sd)). Spike x3c_kernels' `ab.py`, with its arms swapped for these.

    python spikes/x3d_core/ab.py --model <gguf> [--engine-args '<flags>'] [--ctx 1024 4096] [--rounds 8] [--steps 256] [--prefill 512 4096]
        [--arms base prep ...] [--name label] [--json out]
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import random
import shlex
import statistics
import sys
import time
from unittest import mock

import torch
from holon.api import SamplingParams
from holon.core import platform
from holon.core.cli import engine_config, engine_flags
from holon.core.engine import Engine

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bench"))
import base  # noqa: E402
from run import public  # noqa: E402

ARMS = ("base", "prep", "upload")
_OLD: dict[tuple[int, str], object] = {}      # (executor id, arm) -> its uploader (one per engine: pinned arenas)


@contextlib.contextmanager
def arm(engine: Engine, name: str):
    """The engine with the changes up to `name` on (ARMS order), the rest off; its graphs dropped on entry."""
    ex, on = engine.executor, set(ARMS[1:ARMS.index(name) + 1])
    with contextlib.ExitStack() as stack:
        if "prep" not in on:                         # bdd8422's device prep, store, logits gather and layout
            old = _OLD.setdefault((id(ex), "base"), base.uploader(ex))
            stack.enter_context(mock.patch.multiple(ex, uploader=old, small_uploader=old, **base.methods(ex)))
        elif "upload" not in on:                     # one arena for every step, fields in the order given
            old = _OLD.setdefault((id(ex), "prep"), base.prep_uploader(ex))
            stack.enter_context(mock.patch.multiple(ex, uploader=old, small_uploader=old))
        if ex.graphs is not None:
            ex.graphs = platform.Graphs(ex.device)
        yield


def timed(fn, steps: int) -> float:
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(steps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / steps * 1e3


def rows(engine: Engine, contexts: list[int], steps: int, prefill: list[int]) -> dict[str, float]:
    """ms per graphed decode step at batch 1 after a prompt of each context (random ids, the same every arm); ms per
    prompt of each `prefill` length alone (eager, median of 3)."""
    out = {}
    for p in prefill:
        rng = random.Random(p)
        prompt = [rng.randrange(1000, 30000) for _ in range(p)]

        def one():
            engine.add(prompt, SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True))
            while engine.has_work():
                engine.step()
        one()
        out[f"prefill_{p}"] = statistics.median(timed(one, 1) for _ in range(3))
    for ctx in contexts:
        rng = random.Random(ctx)
        rid = engine.add([rng.randrange(1000, 30000) for _ in range(ctx)],
                         SamplingParams(temperature=0.0, max_tokens=1 << 20, ignore_eos=True))
        while engine.sched.waiting or any(s.num_computed < s.prompt_len for s in engine.sched.running):
            engine.step()
        timed(engine.step, 3)                       # eager, capture, replay
        out[f"decode_{ctx}"] = timed(engine.step, steps)
        engine.abort(rid)
        while engine.has_work():
            engine.step()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--engine-args", default="")
    ap.add_argument("--ctx", type=int, nargs="+", default=[1024, 4096])
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=1, help="rounds of every arm run first and not counted")
    ap.add_argument("--prefill", type=int, nargs="*", default=[])
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    ap.add_argument("--name", default="")
    ap.add_argument("--json")
    a = ap.parse_args()
    flags = argparse.ArgumentParser()
    engine_flags(flags)
    cfg = engine_config(a.model, flags.parse_args(shlex.split(a.engine_args)))
    engine = Engine(dataclasses.replace(cfg, prefix_cache=False, max_seqs=1, max_model_len=max([*a.ctx, *a.prefill]) + a.steps + 64,
                                        max_step_tokens=max([cfg.max_step_tokens, *a.prefill])))
    arms = [n for n in ARMS if n in a.arms]
    times: dict[str, list[dict[str, float]]] = {n: [] for n in arms}
    for r in range(-a.warmup, a.rounds):
        for name in arms if r % 2 == 0 else arms[::-1]:
            with arm(engine, name):
                got = rows(engine, a.ctx, a.steps, a.prefill)
            if r >= 0:
                times[name].append(got)
            print(f"round {r} {name}", {k: round(v, 4) for k, v in got.items()}, flush=True)
    result = {"name": a.name, "gpu": torch.cuda.get_device_name(), "model": public(a.model, a.model),
              "engine_args": " ".join(public(x, a.model) for x in shlex.split(a.engine_args)),
              "rounds": a.rounds, "warmup_rounds": a.warmup, "steps": a.steps, "arms": arms, "times_ms": times, "savings": {}}
    for prev, name in zip(arms, arms[1:], strict=False):
        result["savings"][name] = {}
        for row in times[name][0]:
            ms = [p[row] - c[row] for p, c in zip(times[prev], times[name], strict=True)]
            pct = [(p[row] - c[row]) / p[row] * 100 for p, c in zip(times[prev], times[name], strict=True)]
            result["savings"][name][row] = {
                "ms": round(statistics.mean(ms), 4), "pct": round(statistics.mean(pct), 3),
                "sd_pct": round(statistics.stdev(pct), 3) if len(pct) > 1 else None,
                "median_ms": {"before": round(statistics.median(p[row] for p in times[prev]), 4),
                              "after": round(statistics.median(c[row] for c in times[name]), 4)}}
        print(name, json.dumps(result["savings"][name]), flush=True)
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
