"""M0.7 step 0 (docs/dev/m07-perf.md D1, D2, D8): where a step's GPU time goes, per row kind, for one deployment.

Rows: prefill of P tokens (one request alone: one step), decode at batch B (B requests of --ctx tokens), and verify
steps at batch B when the deployment drafts; at the deployment's overlap (a window's wall per step is the step's
cadence).
- Each row is timed unprofiled and traced with torch.profiler (CUPTI records the kernels inside graph replays):
  graphed and eager for decode and verify, eager for prefill (never graphed). Kernels are bucketed by name
  (`category`); an eager verify step is also split by region (trunk, head, verifier, rewind, drafter, ...).
- D1's two timing ratios per decode / verify row: the unprofiled graphed wall over the eager kernel sum, and the
  profiled graphed kernel sum over the unprofiled graphed wall; the SM clock, power and throttle reasons sampled
  (nvidia-smi, 100 ms) over a graphed and an eager window of about 2 s each.
- One eager step per row counts the ggml matmul calls per (type, in, out, mode, rows) and the weight bytes they read
  (D2's step weights; D8's bytes, beside the KV bytes per token of the token pools).

    python bench/step_profile.py --model <file.gguf | dir> [--lock <lock>] [--engine-args '<flags>']
        [--prefill 512 2048] [--batches 1 4 8] [--ctx 512] [--name label] [--json out]
"""
from __future__ import annotations

import argparse
import bisect
import collections
import dataclasses
import functools
import json
import os
import random
import re
import shlex
import statistics
import sys
import tempfile
import time

import torch
from clock_check import Sampler
from holon.api import SamplingParams
from holon.core.cli import engine_config, engine_flags
from holon.core.engine import Engine
from holon_plugins.kernels.ggml_cuda import plugin as ggml
from run import lock_flags, public, read_lock

STEPS = 32                     # steps per timed or traced window
CLOCK_S = 2.0                  # seconds per clock window

# (category, name pattern), first match wins; names lower-cased. "glue" is everything but gemm and attention. Copies
# are told apart by their op's dtypes (`Classifier`): an f32 -> bf16 cast (mmq's output), a bf16 -> f32 copy (native
# FP4's activations), or a plain copy.
CATEGORIES = [
    ("add+rmsnorm (fused)", r"fusedaddrmsnorm|fused_add_rmsnorm"),
    ("q/k norm", r"(rmsnorm|rms_norm).*align\d+o(64|128|256)1"),
    ("rmsnorm", r"rmsnorm|rms_norm"),
    ("gemm", r"mul_mat_vec_q|mul_mat_q|stream_k_fixup|gemm|cutlass|cublas|xmma|s16816|splitkreduce|gemv"),
    ("act quant (q8_1 / fp4)", r"quantize_q8_1|quantize_mmq|quantize_row|quantize_mxfp4|quantize_nvfp4"),
    ("attention", r"_paged_attn|_merge_splits|_paged_prefill|batchprefill|batchdecode|mergestate|flash_fwd|"
                  r"variablelengthmergestates|prefillwithkvcache|_attn_fwd|_attention"),
    ("kv store", r"index_put|_write_q8_0|_kv_write|scatter_kernel.*kv"),
    ("rope", r"rotary|rope"),
    ("silu*mul", r"act_and_mul|silu|swiglu"),
    ("gdn core", r"fused_sigmoid|chunk_|solve_tril|cumsum|wy_fast|recompute_w|l2norm|kkt|fused_recurrent|merge_16x16|"
                 r"delta|gated_delta"),
    ("gdn conv", r"conv"),
    ("sampling / head", r"argmax|softmax|topk|sort|sampl|exponential|gumbel"),
    ("embedding", r"get_rows|embedding"),
    ("residual add", r"cudafunctor_add|functor_add|add_kernel"),
    ("memcpy / memset", r"^memcpy|^memset"),
    ("copy / cat / fill", r"copy_kernel|fillfunctor|cat_|catarray"),
    ("index / scalar ops", r"<long|<int|long,|index_kernel|index_copy|where|compare|bool|masked|scatter|gather|"
                           r"index_select|arange|clamp|reduce_kernel"),
    ("gates (sigmoid, mul, softplus, exp)", r"sigmoid|softplus|exp_kernel|neg_kernel|binaryfunctor|mulfunctor"),
    ("other elementwise", r"elementwise"),
]
CAST, TO_F32 = "f32 -> bf16 cast", "bf16 -> f32 copy"


def category(name: str) -> str:
    low = name.lower()
    return next((cat for cat, pattern in CATEGORIES if re.search(pattern, low)), "other")


class Classifier:
    """`category`, with copy kernels split by their op's dtypes where the trace has ops (eager steps, traced with
    shapes); graph replays carry no ops, so a kernel name seen on an eager step keeps what it was there."""

    def __init__(self) -> None:
        self.learned: dict[str, str] = {}
        self.unknown: collections.Counter = collections.Counter()

    def __call__(self, e: dict, ops: dict) -> str:
        name = e["name"]
        if name in self.learned:
            return self.learned[name]
        cat = category(name)
        op = ops.get(e.get("args", {}).get("External id"))
        if op is not None and "copy_kernel" in name:
            dst, src = (op[1] or ["", ""])[:2]
            cat = CAST if (dst, src) == ("c10::BFloat16", "float") else TO_F32 if (dst, src) == (
                "float", "c10::BFloat16") else cat
            self.learned[name] = cat
        if cat in ("other", "other elementwise"):
            self.unknown[name[:120]] += 1
        return cat


def trace(fn) -> list[dict]:
    """The trace events of `fn()`: kernels, memcpys and memsets, their launches, ops (with input types) and
    annotations."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA,
                                            torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
        fn()
        torch.cuda.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        prof.export_chrome_trace(f.name)
        with open(f.name) as g:
            return json.load(g)["traceEvents"]


def kernels(events: list[dict]) -> list[dict]:
    return sorted((e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")), key=lambda e: e["ts"])


def ops(events: list[dict]) -> dict:
    """External id -> (the first op's name, its input types)."""
    out = {}
    for e in events:
        if e.get("cat") == "cpu_op" and "External id" in e.get("args", {}):
            out.setdefault(e["args"]["External id"], (e["name"], e["args"].get("Input type")))
    return out


def breakdown(events: list[dict], steps: int, classify: Classifier) -> dict:
    """Per step: kernel count, kernel ms, span ms (first start to last end), and (count, ms) per category."""
    ks, by_id = kernels(events), ops(events)
    count, ms = collections.Counter(), collections.Counter()
    for e in ks:
        cat = classify(e, by_id)
        count[cat] += 1
        ms[cat] += e["dur"]
    return {"kernels": round(len(ks) / steps, 1), "kernel_ms": round(sum(e["dur"] for e in ks) / steps / 1e3, 4),
            "span_ms": round((ks[-1]["ts"] + ks[-1]["dur"] - ks[0]["ts"]) / steps / 1e3, 4) if ks else 0.0,
            "cat": {c: [round(count[c] / steps, 1), round(ms[c] / steps / 1e3, 4)] for c in sorted(ms, key=ms.get,
                                                                                                    reverse=True)}}


def regions(events: list[dict], steps: int, classify: Classifier) -> dict:
    """Per step and region (the innermost "R:" annotation around the kernel's launch): kernels, ms, ms per category."""
    spans = sorted((e["ts"], e["ts"] + e["dur"], e["name"][2:]) for e in events
                   if e.get("cat") == "user_annotation" and e["name"].startswith("R:"))
    starts = [s[0] for s in spans]
    launch = {e["args"]["correlation"]: e["ts"] for e in events
              if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}

    def region(ts: float | None) -> str:
        if ts is None:
            return "?"
        for a, b, name in reversed(spans[:bisect.bisect_right(starts, ts)]):
            if a <= ts <= b:
                return name
        return "outside"

    out: dict[str, dict] = {}
    by_id = ops(events)
    for e in kernels(events):
        r = out.setdefault(region(launch.get(e["args"].get("correlation"))), {"kernels": 0, "ms": 0.0, "cat": {}})
        r["kernels"] += 1
        r["ms"] += e["dur"] / 1e3
        cat = classify(e, by_id)
        r["cat"][cat] = r["cat"].get(cat, 0.0) + e["dur"] / 1e3
    return {name: {"kernels": round(r["kernels"] / steps, 1), "ms": round(r["ms"] / steps, 4),
                   "cat": {c: round(v / steps, 4) for c, v in sorted(r["cat"].items(), key=lambda x: -x[1])}}
            for name, r in sorted(out.items(), key=lambda x: -x[1]["ms"])}


class GemmCounter:
    """Counts the ggml qlinear calls (type, in, out, mode, rows; `glu` for mmvq's fused gate/up + SwiGLU, D6), the
    weight bytes they read, the embedding rows' bytes the step gathers and, on decode and verify steps, the distinct
    inputs they quantize (the same tensor, unchanged, quantized by several calls: D6's shared quantization; the inputs
    are held for the step so no address is reused), while active."""

    def __init__(self) -> None:
        self.calls: collections.Counter = collections.Counter()
        self.bytes = self.embedding_bytes = 0
        self.inputs: dict[tuple, torch.Tensor] = {}
        self.active = False

        def counting(original, glu: bool):
            def counted(kernel, x, parts):
                y = original(kernel, x, parts)
                if self.active and not (glu and y is None):             # a glu call that composes is counted there
                    rows = x if isinstance(x, torch.Tensor) else x.x    # a prepared input's (D6)
                    self.calls[(kernel.name, kernel.k, kernel.out, kernel.act["mmq"], len(rows), glu)] += 1
                    self.bytes += parts["blocks"].numel() * parts["blocks"].element_size()
                    if len(rows) <= 64:
                        self.inputs[(id(rows), rows._version)] = rows
                return y
            return counted

        def embedded(original):
            def counted(kernel, ids, blocks):
                if self.active:                                         # blocks: [vocab, row bytes]
                    self.embedding_bytes += len(ids) * blocks.nbytes // len(blocks)
                return original(kernel, ids, blocks)
            return counted

        ggml.QLinear.__call__ = counting(ggml.QLinear.__call__, False)
        ggml.QLinearGlu.__call__ = counting(ggml.QLinearGlu.__call__, True)
        ggml.QEmbedding.__call__ = embedded(ggml.QEmbedding.__call__)

    def __call__(self, fn) -> dict:
        self.calls.clear()
        self.inputs.clear()
        self.bytes = self.embedding_bytes = 0
        self.active = True
        try:
            fn()
            torch.cuda.synchronize()
        finally:
            self.active = False
        inputs, self.inputs = len(self.inputs), {}
        return {"weight_bytes": self.bytes, "embedding_bytes": self.embedding_bytes, "inputs": inputs or None,
                "calls": [{"type": t, "in": k, "out": n, "mode": m, "rows": r, "count": c}
                          | ({"glu": True} if g else {}) for (t, k, n, m, r, g), c in sorted(self.calls.items())]}


def annotate(engine: Engine) -> None:
    """Wrap the executor's calls in "R:<region>" annotations (they run on eager steps; graphs replay without them)."""
    ex = engine.executor

    def wrap(obj, attr: str, label: str) -> None:
        f = getattr(obj, attr)

        @functools.wraps(f)
        def g(*a, **k):
            with torch.profiler.record_function("R:" + label):
                return f(*a, **k)
        setattr(obj, attr, g)

    wrap(ex, "_prepare", "prepare")
    wrap(ex, "_store", "store")
    wrap(ex, "_run_drafter", "drafter")
    wrap(ex.model, "trunk", "trunk")
    wrap(ex.model, "head", "head")
    if ex.verifier is not None:
        wrap(ex.verifier, "verify", "verifier")
    wrap(ex.sampler, "sample", "sampler")
    ex.rewinders = [functools.partial(lambda ctx, r: _annotated("rewind", r, ctx), r=r) for r in ex.rewinders]


def _annotated(label: str, fn, *args):
    with torch.profiler.record_function("R:" + label):
        return fn(*args)


def timed(fn, steps: int) -> float:
    """ms per call of `fn`, `steps` calls."""
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(steps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / steps * 1e3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lock", help="a lock: its [engine] settings (then --engine-args over them)")
    ap.add_argument("--engine-args", default="", help="engine flags, as for `holon serve`")
    ap.add_argument("--prefill", type=int, nargs="*", default=[512, 2048])
    ap.add_argument("--batches", type=int, nargs="*", default=[1, 4, 8], help="decode (or verify) batch sizes")
    ap.add_argument("--ctx", type=int, default=512, help="prompt tokens of each decoding request")
    ap.add_argument("--name", default="")
    ap.add_argument("--llama-crossover", action="store_true", help="kernels.ggml_cuda without its per-shape entries "
                    "(llama.cpp's rule: D2's baseline, docs/dev/m07-perf.md)")
    ap.add_argument("--json")
    args = ap.parse_args()
    if args.llama_crossover:
        from holon_plugins.kernels.ggml_cuda import plugin as ggml
        ggml.CROSSOVER.clear()
    flags = argparse.ArgumentParser()
    engine_flags(flags)
    given = (lock_flags(read_lock(args.lock)["engine"], (), args.engine_args) if args.lock
             else shlex.split(args.engine_args))
    cfg = engine_config(args.model, flags.parse_args(given))
    cfg = dataclasses.replace(cfg, prefix_cache=False,
                              max_seqs=max([cfg.max_seqs, *args.batches]),
                              max_step_tokens=max([cfg.max_step_tokens, *args.prefill]),
                              max_model_len=max([*args.prefill, args.ctx]) + 4096)
    engine = Engine(cfg)
    ex, counter, classify = engine.executor, GemmCounter(), Classifier()
    annotate(engine)
    graphs = ex.graphs
    rng = random.Random(0)
    prompt = lambda n: [rng.randrange(1000, 30000) for _ in range(n)]   # noqa: E731
    pools = [p for p in engine.ledger.pools.values() if p.tokens_per_unit]
    result = {"name": args.name, "gpu": torch.cuda.get_device_name(), "model": public(args.model, args.model),
              "lock": public(args.lock, "") if args.lock else None,
              "engine_args": " ".join(public(a, args.model) for a in shlex.split(args.engine_args)),
              "engine": {k: {o: public(str(x), args.model) for o, x in v.items()} if isinstance(v, dict) else v
                         for k, v in dataclasses.asdict(cfg).items() if k != "model"},
              "kv_bytes_per_token": sum(p.unit_bytes / p.tokens_per_unit for p in pools),
              "drafts": cfg.max_draft_tokens > 0, "llama_crossover": args.llama_crossover, "rows": {}}
    sampler = Sampler()

    def one(params: SamplingParams, n: int) -> None:
        engine.add(prompt(n), params)
        while engine.has_work():
            engine.step()

    for p in args.prefill:                            # prefill is eager; the request alone
        params = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
        gemm = counter(lambda p=p, params=params: one(params, p))
        walls = [timed(lambda p=p, params=params: one(params, p), 1) for _ in range(3)]
        events = trace(lambda p=p, params=params: one(params, p))
        row = {"kind": "prefill", "tokens": p, "wall_ms": round(statistics.median(walls), 3),
               "eager": breakdown(events, 1, classify), "gemm": gemm}
        row["idle"] = round(1 - row["eager"]["kernel_ms"] / row["wall_ms"], 4)
        result["rows"][f"prefill_{p}"] = row
        print(f"prefill {p}", {k: v for k, v in row.items() if k not in ("gemm", "eager")},
              row["eager"]["kernel_ms"], flush=True)

    kind = "verify" if result["drafts"] else "decode"
    for b in args.batches:
        rids = [engine.add(prompt(args.ctx), SamplingParams(temperature=0.0, max_tokens=1 << 20, ignore_eos=True))
                for _ in range(b)]
        while engine.sched.waiting or any(s.num_computed < s.prompt_len for s in engine.sched.running):
            engine.step()
        stats0 = engine.stats()
        row = {"kind": kind, "batch": b, "ctx": args.ctx}
        ex.graphs = None                              # eager first: graph replays reuse its kernel names' classes
        timed(engine.step, 2)
        row["gemm"] = counter(engine.step)
        row["eager_wall_ms"] = round(timed(engine.step, STEPS), 4)
        events = trace(lambda: [engine.step() for _ in range(STEPS)])
        row["eager"] = breakdown(events, STEPS, classify)
        if kind == "verify":
            row["regions"] = regions(events, STEPS, classify)
        ex.graphs = graphs
        timed(engine.step, 3)
        ctx0 = statistics.mean(s.num_computed for s in engine.sched.running)
        row["graphed_wall_ms"] = round(timed(engine.step, STEPS), 4)
        row["graphed_ctx"] = round((ctx0 + statistics.mean(s.num_computed for s in engine.sched.running)) / 2)
        row["graphed"] = breakdown(trace(lambda: [engine.step() for _ in range(STEPS)]), STEPS, classify)
        row["ratio_graphed_wall_over_eager_kernels"] = round(row["graphed_wall_ms"] / row["eager"]["kernel_ms"], 4)
        row["ratio_graphed_profiled_over_wall"] = round(row["graphed"]["kernel_ms"] / row["graphed_wall_ms"], 4)
        row["clocks"] = {}
        for mode in ("graphed", "eager"):
            ex.graphs = graphs if mode == "graphed" else None
            start = time.time()
            while time.time() - start < CLOCK_S:
                engine.step()
            torch.cuda.synchronize()
            row["clocks"][mode] = sampler.summary(start + 0.2, time.time())
        ex.graphs = graphs
        stats = engine.stats()
        row["spec"] = {k: v - stats0.get(k, 0) for k, v in stats.items()
                       if k.startswith("spec_") and v != stats0.get(k)}
        row["ctx_end"] = int(statistics.mean(s.num_computed for s in engine.sched.running))
        for rid in rids:
            engine.abort(rid)
        while engine.has_work():
            engine.step()
        result["rows"][f"{kind}_b{b}"] = row
        print(f"{kind} b{b}", {k: row[k] for k in ("graphed_wall_ms", "eager_wall_ms",
                                                   "ratio_graphed_wall_over_eager_kernels",
                                                   "ratio_graphed_profiled_over_wall")},
              row["graphed"]["kernel_ms"], row["eager"]["kernel_ms"], row["clocks"], flush=True)
    sampler.close()
    result["unclassified"] = dict(classify.unknown.most_common(40))
    if classify.unknown:
        print("unclassified:", classify.unknown.most_common(15), file=sys.stderr)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1, default=str)


if __name__ == "__main__":
    main()
