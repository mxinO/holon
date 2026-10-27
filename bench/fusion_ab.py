"""M0.7 step 2's keep-or-revert measurement (docs/dev/m07-perf.md §1.1): each fusion's saving, measured incrementally on
one engine, per release row — graphed decode B1/B4/B8 (or verify steps when the deployment drafts) by per-step wall,
eager prefill by wall.

The arms add the fusions in order, each on top of the previous: `base` (all off: the step-1 code paths), then
`kv_write` (the model-dtype layout's Triton copy), `qk_rope_kv` (D3, on top of the copy), `mmq_bf16` (D4), `scale2`
(D11), `glu` (D6's gate/up + SwiGLU) and `q8_1` (D6's shared quantization). An arm's saving is its row time against
the previous arm's. A fusion is switched off by restoring the code it replaced, in this process: the index_put
write, the composed norms / rope / write, mmq into an f32 tensor and its cast (and scale2's multiply), mmvq into f32
then scale2's multiply above one row, the composed gate/up and SwiGLU, each linear quantizing its own input. Every
switch drops the captured graphs (`Graphs` captures again on the next steps), so graphed rows replay the arm's own
path. Rounds visit the arms in ABBA order (forward, then backward) against drift; each row's per-round difference
gives the mean saving and its sd (§1.1: kept if ≥ max(0.5%, 3 sd) on some release row).

    python bench/fusion_ab.py --model <file.gguf | dir> [--lock <lock>] [--engine-args '<flags>'] [--rounds 4]
        [--prefill 512 2048] [--batches 1 4 8] [--ctx 512] [--arms base kv_write ...] [--name label] [--json out]
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
import time
from unittest import mock

import torch
from holon.api import SamplingParams
from holon.core import platform
from holon.core.cli import engine_config, engine_flags
from holon.core.engine import Engine
from holon_plugins.kernels.ggml_cuda import plugin as ggml
from holon_plugins.kernels.triton_attn import plugin as triton_attn
from holon_std.mixers.gqa import units as gqa
from run import lock_flags, public, read_lock

ARMS = ("base", "kv_write", "qk_rope_kv", "mmq_bf16", "scale2", "glu", "q8_1")
STEPS = 64


def _index_put_write(self, k, v, kv, unit, off) -> None:
    kv[unit, 0, :, off], kv[unit, 1, :, off] = k, v              # the step-1 model-dtype write


def _qlinear_call(on: set[str]):
    """`QLinear.__call__` with D4 (mmq writes out_dtype) and D11 (scale2 in mmvq above one row) as `on` says; off,
    the step-1 code: an f32 tensor, then the cast or scale2's multiply."""
    def call(self, x, parts):
        xq = None
        if isinstance(x, ggml.Prepared):
            xq, x = self._prepared(x), x.x
        x, rows, blocks, scale2 = x.contiguous(), len(x), parts["blocks"], parts.get("scale2")
        y = torch.empty(rows, self.out, dtype=self.out_dtype, device=x.device)
        if rows == 0:
            return y
        if rows <= self.mmvq_rows:
            self._count("mmvq")
            if scale2 is None or rows == 1 or "scale2" in on:
                self.lib.mmvq(self.type, blocks, x, y, scale2, self.workspace, xq=xq)
            else:
                f32 = y if y.dtype == torch.float32 else torch.empty(rows, self.out, device=x.device)
                self.lib.mmvq(self.type, blocks, x, f32, None, self.workspace, xq=xq)
                torch.mul(f32, scale2, out=y)
            return y
        self._count("mmq")
        blocks = self._mmq_blocks(parts)
        if "mmq_bf16" in on:
            for start in range(0, rows, ggml.CHUNK):
                self.lib.mmq(self.type, self.prec, blocks, self._mmq_input(x[start:start + ggml.CHUNK], self.f32),
                             y[start:start + ggml.CHUNK], scale2, self.workspace)
            return y
        f32 = None if y.dtype == torch.float32 else torch.empty(min(rows, ggml.CHUNK), self.out, device=x.device)
        for start in range(0, rows, ggml.CHUNK):
            xs, ys = x[start:start + ggml.CHUNK], y[start:start + ggml.CHUNK]
            out = ys if f32 is None else f32[:len(xs)]
            self.lib.mmq(self.type, self.prec, blocks, self._mmq_input(xs, self.f32), out, None, self.workspace)
            if scale2 is not None:
                torch.mul(out, scale2, out=ys)
            elif f32 is not None:
                ys.copy_(out)
        return y
    return call


@contextlib.contextmanager
def arm(engine: Engine, name: str):
    """The engine with the fusions up to `name` on (ARMS order), the rest off; its graphs dropped on entry."""
    on = set(ARMS[1:ARMS.index(name) + 1])
    roots = [engine.executor.model, *(m for _, m in (engine.ctx.aux.modules if engine.ctx.aux else ()))]
    mixers = [m for r in roots for m in r.modules() if isinstance(m, gqa.GQA)]     # the drafter's too
    fused = [m._fused for m in mixers]
    with contextlib.ExitStack() as stack:
        if "kv_write" not in on:
            stack.enter_context(mock.patch.object(triton_attn.KvWrite, "__call__", _index_put_write))
        if "qk_rope_kv" not in on:
            for m in mixers:
                m._fused = None
        stack.enter_context(mock.patch.object(ggml.QLinear, "__call__", _qlinear_call(on)))
        if "glu" not in on:
            stack.enter_context(mock.patch.object(ggml.QLinearGlu, "__call__", lambda self, x, parts: None))
        if "q8_1" not in on:
            stack.enter_context(mock.patch.object(ggml.QLinear, "prepare_input", lambda self, x: ggml.Prepared(x)))
        if engine.executor.graphs is not None:
            engine.executor.graphs = platform.Graphs(engine.executor.device)
        try:
            yield
        finally:
            for m, f in zip(mixers, fused, strict=True):
                m._fused = f


def timed(fn, steps: int) -> float:
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(steps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / steps * 1e3


def rows(engine: Engine, prefill: list[int], batches: list[int], ctx: int, kind: str) -> dict[str, float]:
    """ms per row: each prefill alone (eager, median of 3), and graphed decode / verify steps at each batch."""
    rng = random.Random(0)
    prompt = lambda n: [rng.randrange(1000, 30000) for _ in range(n)]       # noqa: E731
    out = {}
    for p in prefill:
        def one(p=p):
            engine.add(prompt(p), SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True))
            while engine.has_work():
                engine.step()
        one()
        out[f"prefill_{p}"] = statistics.median(timed(one, 1) for _ in range(3))
    for b in batches:
        rids = [engine.add(prompt(ctx), SamplingParams(temperature=0.0, max_tokens=1 << 20, ignore_eos=True))
                for _ in range(b)]
        while engine.sched.waiting or any(s.num_computed < s.prompt_len for s in engine.sched.running):
            engine.step()
        timed(engine.step, 3)                       # eager, capture, replay
        out[f"{kind}_b{b}"] = timed(engine.step, STEPS)
        for rid in rids:
            engine.abort(rid)
        while engine.has_work():
            engine.step()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lock", help="a lock: its [engine] settings (then --engine-args over them)")
    ap.add_argument("--engine-args", default="")
    ap.add_argument("--prefill", type=int, nargs="*", default=[512, 2048])
    ap.add_argument("--batches", type=int, nargs="*", default=[1, 4, 8])
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    ap.add_argument("--name", default="")
    ap.add_argument("--json")
    a = ap.parse_args()
    flags = argparse.ArgumentParser()
    engine_flags(flags)
    given = lock_flags(read_lock(a.lock)["engine"], (), a.engine_args) if a.lock else shlex.split(a.engine_args)
    cfg = engine_config(a.model, flags.parse_args(given))
    cfg = dataclasses.replace(cfg, prefix_cache=False, max_seqs=max([cfg.max_seqs, *a.batches]),
                              max_step_tokens=max([cfg.max_step_tokens, *a.prefill]),
                              max_model_len=max([*a.prefill, a.ctx]) + 4096)
    engine = Engine(cfg)
    kind = "verify" if cfg.max_draft_tokens > 0 else "decode"
    arms = [n for n in ARMS if n in a.arms]
    times: dict[str, list[dict[str, float]]] = {n: [] for n in arms}
    for r in range(a.rounds):
        for name in arms if r % 2 == 0 else arms[::-1]:
            with arm(engine, name):
                times[name].append(rows(engine, a.prefill, a.batches, a.ctx, kind))
            print(f"round {r} {name}", {k: round(v, 3) for k, v in times[name][-1].items()}, flush=True)
    result = {"name": a.name, "gpu": torch.cuda.get_device_name(), "model": public(a.model, a.model),
              "lock": public(a.lock, "") if a.lock else None,
              "engine_args": " ".join(public(x, a.model) for x in shlex.split(a.engine_args)),
              "rounds": a.rounds, "steps": STEPS, "arms": arms, "times_ms": times, "savings": {}}
    for prev, name in zip(arms, arms[1:], strict=False):
        result["savings"][name] = {}
        for row in times[name][0]:
            diffs = [(p[row] - c[row]) / p[row] * 100 for p, c in zip(times[prev], times[name], strict=True)]
            result["savings"][name][row] = {"pct": round(statistics.mean(diffs), 3),
                                            "sd": round(statistics.stdev(diffs), 3) if len(diffs) > 1 else None}
        print(name, result["savings"][name], flush=True)
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
