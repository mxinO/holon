"""Step 3b's perf check (docs/v1-moe-prefill.md §7, D11): the offload store with `moe.ring_lend` `tail` against
`coldest`, batch 1, one engine process per mode, at each context:

- the prompt's borrowing steps by use (`borrowed`: `ring` = a ring step with `ring_ok`, else `stage`); a `SHORT` step
  raises in `after_commit`, so a finished run had none;
- the time to the first token over `--reps` prompts (after an untimed warm-up over the same prompts);
- over the first `--decode` tokens of one more prompt, the miss bytes per token: each decode step's routed experts
  not resident before it (x4_gate's `capped.py` "true" count), and the tail (class, slots).

    python spikes/moe_prefill/tail.py --model <gguf> --config natural [--cache '<moe.cache>'] [--mode tail]
        [--contexts 4096 32768] [--reps 3] [--decode 64] [--json out]
"""
import argparse
import dataclasses
import json
import pathlib
import shlex
import statistics
import sys
import time

import numpy as np
import torch
from holon.api import SamplingParams
from holon.core.cli import engine_config, engine_flags
from holon.core.engine import Engine
from holon_judge.placement import route_dump

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
import offload_ctx  # noqa: E402
import workloads  # noqa: E402

STRIDE = 7919


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default="natural", help="a configuration of identity_configs.json")
    ap.add_argument("--cache", help="moe.cache in place of the configuration's")
    ap.add_argument("--mode", default="coldest", choices=("coldest", "tail"))
    ap.add_argument("--contexts", type=int, nargs="+", default=[4096, 32768])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--decode", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=33792)
    ap.add_argument("--json")
    a = ap.parse_args()
    flags = argparse.ArgumentParser()
    engine_flags(flags)
    text = json.loads((pathlib.Path(__file__).parent / "identity_configs.json").read_text())[a.config]
    cfg = engine_config(a.model, flags.parse_args(shlex.split(text)))
    opts = dict(cfg.plugin_options) | {"moe.ring_lend": a.mode} | ({"moe.cache": a.cache} if a.cache else {})
    engine = Engine(dataclasses.replace(cfg, max_seqs=1, max_model_len=a.ctx, prefix_cache=False,
                                        plugin_options=opts))
    store = engine.ctx.shared_objects["std.moe.offload"]
    corpus = workloads.corpus(offload_ctx.CORPUS)

    def prompt(n: int, rep: int) -> list[int]:
        start = offload_ctx.OFFSETS.get(n, 0) + rep * STRIDE
        return [corpus[(start + i) % len(corpus)] for i in range(n)]

    def ttft(ids: list[int]) -> float:
        rid = engine.add(ids, SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True))
        t0, done = time.perf_counter(), False
        while not done:
            done = any(d.rid == rid and d.finish_reason is not None for d in engine.step())
        return time.perf_counter() - t0

    L, E = store.residency.host.shape[:2]
    unit = np.array([sum(store.groups[g][r].unit_bytes for r in range(store.roles)) for g in sorted(store.groups)])
    ebytes = np.repeat(unit, E).astype(float)
    out = {"mode": a.mode, "config": a.config, "cache": opts.get("moe.cache"), "tail": getattr(store, "tail", None),
           "gpu": torch.cuda.get_device_name(), "contexts": {}}
    for n in a.contexts:
        for r in range(a.reps):                                   # untimed warm-up over the prompt set
            ttft(prompt(n, r))
        before = dict(store.borrowed)
        times = [ttft(prompt(n, r)) for r in range(a.reps)]
        borrowed = {k: store.borrowed[k] - before.get(k, 0) for k in store.borrowed}
        start = (store.residency.host >= 0).all(-1).copy()
        with route_dump(engine) as trace:
            engine.generate([prompt(n, a.reps)], SamplingParams(temperature=0.0, max_tokens=a.decode + 1,
                                                                ignore_eos=True))
        steps = [s for s in trace.steps if s["counts"].any()]        # decode steps (batch 1: one row each)
        first = trace.steps.index(steps[0])
        res = (trace.steps[first - 1]["resident"] if first else start).reshape(-1)
        miss = []
        for s in steps[:a.decode]:
            k = np.flatnonzero(s["counts"].reshape(-1))
            miss.append(ebytes[k[~res[k]]].sum())
            res = s["resident"].reshape(-1)
        out["contexts"][str(n)] = {
            "ttft_s": [round(t, 4) for t in times], "ttft_median_s": round(statistics.median(times), 4),
            "borrowed": borrowed, "ring_ok": borrowed.get("stage", 0) == 0 and borrowed.get("ring", 0) > 0,
            "decode_tokens": len(miss), "miss_mb_per_token": round(float(np.mean(miss)) / 1e6, 2),
            "miss_mb_by_quarter": [round(float(np.mean(x)) / 1e6, 2) for x in np.array_split(miss, 4)]}
        print(json.dumps({n: out["contexts"][str(n)]}), flush=True)
    out["peak_misses"] = store.peak_misses
    engine.close()
    print("RESULT " + json.dumps(out), flush=True)
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
