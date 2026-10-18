"""moe_prefill's driver: all-resident batch-1 prefill (time to the first token) of one GGUF at several prompt lengths,
Holon against FreeToken on one GPU, with probes of Holon's expert-path pass and mmq rows per call, and an nsys pass
whose kernels are attributed to regions by NVTX ranges pushed around each engine's Python calls.

- Prompts: spans of natural text (`bench/offload_ctx.py`'s corpus), a distinct start per (repetition, length), so no
  prompt is a prefix of another (FreeToken's radix cache stays cold; Holon runs with its prefix cache off).
- Each arm runs in a fresh process: an untimed 512-token warm-up, then `--reps` rounds over the lengths (one-token
  requests, greedy; one count for every length, or one per length: the fixed prompt set of docs/v1-moe-prefill.md
  §5.2 is `--reps 5 5 3` at 1K / 4K / 32K), then with `--profile` one more request per length between
  cudaProfilerStart/Stop, each inside an NVTX range `req:<length>` (for `nsys --capture-range=cudaProfilerApi`).
- Memory (Holon): the KV tokens the paged pools hold at startup, the startup profile's peak (the worst step's
  allocation peak plus the graphs' pool, what `state_memory` subtracts), and over the timed requests the allocation
  peak and the reserved bytes beyond what was allocated after startup. `--ctx` / `--max-seqs` set the engine's
  context and seats (a lock's, for its startup KV tokens).
- Holon probes, set in the worker process only (nothing shipped changes): `--pass-tokens` std.moe's resident expert
  pass (512 as shipped: `Resident.pass_tokens`), `--expert-rows` kernels.ggml_cuda's mmq rows per `linear_experts`
  call (its `CHUNK` while the instance is built), `--dense-rows` its `qlinear` mmq rows per call (`CHUNK`).
- Regions (NVTX, pushed in this process only): Holon `experts` (`LinearExperts.__call__`), `dense` (`QLinear`),
  `router` (`MoE._route`), `combine`; FreeToken `experts` (`GgmlMoEKernel.apply`: GEMMs, activation, weighted sum),
  `dense` (`GgmlLinearKernel.apply`), `router` (`fused_topk`).

    python spikes/moe_prefill/prefill.py worker --engine holon|freetoken --model <gguf> [--lengths 1024 4096 32768]
        [--reps 3 | --reps 5 5 3] [--profile] [--pass-tokens N] [--expert-rows N] [--dense-rows N]
        [--ctx 33792] [--max-seqs 1] [--flags '<engine flags>']
    python spikes/moe_prefill/prefill.py run --arms <arms.json> --model <gguf> --out <dir> [--rounds 2] \\
        [--holon-python <venv python>] [--freetoken-python <venv python>]
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import json
import os
import pathlib
import shlex
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
import offload_ctx  # noqa: E402  (the corpus)
import workloads  # noqa: E402

CTX = 33792                       # every engine allocates this context (x4_refs' 32K rows)
STRIDE = 7919                     # repetition r of length c starts at OFFSETS[c] + r · STRIDE


def prompt(corpus: list[int], length: int, rep: int) -> list[int]:
    start = offload_ctx.OFFSETS.get(length, 0) + rep * STRIDE
    return [corpus[(start + i) % len(corpus)] for i in range(length)]


def ranged(obj, attr: str, label: str) -> None:
    """Wrap `obj.attr` in an NVTX range `label`."""
    import torch
    f = getattr(obj, attr)

    @functools.wraps(f)
    def g(*a, **k):
        torch.cuda.nvtx.range_push(label)
        try:
            return f(*a, **k)
        finally:
            torch.cuda.nvtx.range_pop()
    setattr(obj, attr, g)


def holon_engine(a: argparse.Namespace):
    import dataclasses

    from holon.core import platform
    from holon.core.cli import engine_config, engine_flags
    from holon.core.engine import Engine
    from holon_plugins.kernels.ggml_cuda import plugin as ggml
    from holon_std.moe import units

    profiled = []                                   # the startup profile's terms (`Executor.state_memory`)
    def recorded(f):
        def g(device, fn):
            profiled.append(f(device, fn))
            return profiled[-1]
        return g
    for name in ("peak_memory", "reserved_growth"):
        setattr(platform, name, recorded(getattr(platform, name)))

    if a.pass_tokens:
        units.Resident.pass_tokens = staticmethod(lambda tokens: a.pass_tokens)
    if a.dense_rows:
        ggml.CHUNK = a.dense_rows
    if a.expert_rows:
        init = ggml.LinearExperts.__init__

        def built(self, *args, **kw):
            chunk, ggml.CHUNK = ggml.CHUNK, a.expert_rows
            try:
                init(self, *args, **kw)
            finally:
                ggml.CHUNK = chunk
        ggml.LinearExperts.__init__ = built
    for obj, attr, label in ((ggml.LinearExperts, "__call__", "experts"), (ggml.QLinear, "__call__", "dense"),
                             (units.MoE, "_route", "router"), (ggml.MoeCombine, "__call__", "combine"),
                             (ggml.MoeCombine, "add", "combine")):
        ranged(obj, attr, label)
    flags = argparse.ArgumentParser()
    engine_flags(flags)
    cfg = engine_config(a.model, flags.parse_args(shlex.split(a.flags)))
    engine = Engine(dataclasses.replace(cfg, max_seqs=a.max_seqs, max_model_len=a.ctx, prefix_cache=False))
    pools = [p for p in engine.ledger.pools.values() if p.tokens_per_unit]
    memory = {"kv_tokens": min((p.capacity - p.has_null) * p.tokens_per_unit for p in pools),
              "profiled_peak_gib": round(sum(profiled) / 2**30, 3), "max_model_len": engine.cfg.max_model_len}

    def request(ids: list[int]) -> float:
        from holon.api import SamplingParams
        rid = engine.add(ids, SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True))
        t0, done = time.perf_counter(), False
        while not done:
            for d in engine.step():
                done |= d.rid == rid and d.finish_reason is not None
        return time.perf_counter() - t0
    return request, engine.close, memory


def freetoken_engine(a: argparse.Namespace):
    import dataclasses

    from freetoken.core import SamplingParams
    from freetoken.layers.quantization.linear import gguf as linear_gguf
    from freetoken.layers.quantization.moe import gguf as moe_gguf
    from freetoken.llm import LLM
    from freetoken.moe import fused
    from freetoken.scheduler import SchedulerConfig
    from freetoken.server.args import parse_args

    for obj, attr, label in ((moe_gguf.GgmlMoEKernel, "apply", "experts"), (linear_gguf.GgmlLinearKernel, "apply",
                                                                             "dense"), (fused, "fused_topk", "router")):
        ranged(obj, attr, label)
    cfg, _ = parse_args(["--model-path", a.model, "--max-seq-len-override", str(CTX), *shlex.split(a.flags)])
    own = ("model_path", "tp_info", "dtype", "offline_mode", "_unique_suffix")
    kwargs = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(SchedulerConfig) if f.name not in own}
    llm = LLM(a.model, dtype=cfg.dtype, **kwargs)

    def request(ids: list[int]) -> float:
        t0 = time.perf_counter()
        llm.generate([ids], SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True))
        return time.perf_counter() - t0
    return request, lambda: None, {}


def worker(a: argparse.Namespace) -> None:
    import torch
    corpus = workloads.corpus(offload_ctx.CORPUS)
    request, close, memory = (holon_engine if a.engine == "holon" else freetoken_engine)(a)
    reps = dict(zip(a.lengths, a.reps * len(a.lengths) if len(a.reps) == 1 else a.reps, strict=True))
    request([corpus[i] for i in range(512)])                         # warm-up
    torch.cuda.synchronize()
    base, startup_peak = torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    times = {str(c): [] for c in a.lengths}
    for rep in range(max(reps.values())):
        for c in a.lengths:
            if rep < reps[c]:
                torch.cuda.synchronize()
                times[str(c)].append(request(prompt(corpus, c, rep)))
    torch.cuda.synchronize()
    if memory:
        memory |= {"allocated_after_start_gib": round(base / 2**30, 3),
                   "run_peak_over_start_gib": round((torch.cuda.max_memory_allocated() - base) / 2**30, 3),
                   "reserved_over_start_gib": round((torch.cuda.memory_reserved() - base) / 2**30, 3)}
    if a.profile:
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        for c in a.lengths:
            torch.cuda.nvtx.range_push(f"req:{c}")
            request(prompt(corpus, c, reps[c]))
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
        torch.cuda.profiler.stop()
    out = {c: {"s": [round(t, 4) for t in ts], "tok_s": [round(int(c) / t, 1) for t in ts]} for c, ts in times.items()}
    out["peak_alloc_gib"] = round(max(startup_peak, torch.cuda.max_memory_allocated()) / 2**30, 2)
    out["memory"] = memory
    close()
    print("RESULT " + json.dumps(out), flush=True)


def orchestrate(a: argparse.Namespace) -> None:
    """Fresh worker processes, arms interleaved per round (reversed on odd rounds), raw rows appended to
    <out>/raw.jsonl.
    An arm: {"name", "engine", "args"} (+ "nsys": the nsys command line to wrap it in, `{out}` its report's stem;
    "rounds")."""
    python = {"holon": a.holon_python, "freetoken": a.freetoken_python}
    arms = json.loads(pathlib.Path(a.arms).read_text())
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for rnd in range(a.rounds):
        for arm in arms if rnd % 2 == 0 else arms[::-1]:
            if rnd >= arm.get("rounds", a.rounds):
                continue
            log = out / f"r{rnd}_{arm['name']}.log"
            nsys = shlex.split(arm.get("nsys", "").replace("{out}", str(out / f"{arm['name']}")))
            cmd = [*nsys, python[arm["engine"]], str(pathlib.Path(__file__)), "worker", "--engine", arm["engine"],
                   "--model", a.model, "--lengths", *map(str, a.lengths), "--reps", *map(str, a.reps),
                   *shlex.split(arm.get("args", ""))]
            print(f"round {rnd} {arm['name']}", flush=True)
            with open(log, "w") as f:
                p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ))
            text = log.read_text(errors="replace")
            line = next((x for x in text.splitlines() if x.startswith("RESULT ")), None)
            row = {"round": rnd, "arm": arm["name"]}
            row |= json.loads(line[7:]) if p.returncode == 0 and line else {"error": text.strip()[-400:]}
            print("  " + json.dumps(row), flush=True)
            with open(out / "raw.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--engine", choices=("holon", "freetoken"), required=True)
    w.add_argument("--flags", default="")
    w.add_argument("--profile", action="store_true")
    w.add_argument("--pass-tokens", type=int, default=0)
    w.add_argument("--expert-rows", type=int, default=0)
    w.add_argument("--dense-rows", type=int, default=0)
    w.add_argument("--ctx", type=int, default=CTX)
    w.add_argument("--max-seqs", type=int, default=1)
    o = sub.add_parser("run")
    o.add_argument("--arms", required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--rounds", type=int, default=2)
    o.add_argument("--holon-python", default=sys.executable)
    o.add_argument("--freetoken-python", default=str(ROOT / "bench" / ".envs" / "freetoken-offload" / "bin" / "python"))
    for p in (w, o):
        p.add_argument("--model", required=True)
        p.add_argument("--lengths", type=int, nargs="+", default=[1024, 4096, 32768])
        p.add_argument("--reps", type=int, nargs="+", default=[3], help="per length, or one for every length")
    a = ap.parse_args()
    with contextlib.suppress(KeyboardInterrupt):
        worker(a) if a.cmd == "worker" else orchestrate(a)


if __name__ == "__main__":
    main()
