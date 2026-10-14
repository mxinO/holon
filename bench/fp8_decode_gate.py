"""M0.6 acceptance 11, the FP8 decode gate (docs/m06-lowp.md §1, B1): `kernels.triton_fp8`'s W8A16 split-K decode
kernel timed per projection shape of a dense FP8 checkpoint (as the engine binds them: q/k/v and gate/up merged),
M = 1 … 16, turned into an end-to-end tg prediction at batch 1:

    step = layers · Σ_shapes t(shape, M = 1) + t(bf16 LM head, M = 1) + non-GEMM time

The non-GEMM time comes from Holon decoding the llama.cpp Q8_0 conversion of the same checkpoint on this card: the
median step time of its tg run less the summed time of its quantized matmul kernels per step (mmvq and its q8_1
activation quantizer, the Q8_0 head included). The prediction is compared with llama-bench's tg128 of that Q8_0 file
(its JSON, measured in the same job): ≥ 0.85× passes.

Every timing replays a CUDA graph over enough weight copies to stream from DRAM, not L2 (as each layer's weights do
in a decode step). `sweep` times every launch configuration per shape at M = 1 and M = 16 (the two row kinds of the
kernel; the fastest are pinned into the plugin's table).

    python bench/fp8_decode_gate.py sweep --model <FP8 checkpoint> [--json out.json]
    python bench/fp8_decode_gate.py gate --model <FP8 checkpoint> --gguf <Q8_0 conversion> --llama-bench <json>
                                         [--json out.json]
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import random
import shlex
import statistics
import subprocess
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
GEMM_KERNELS = ("mul_mat_vec_q", "quantize_q8_1")      # ggml's mmvq and its activation quantizer
STREAM_BYTES = 512 << 20                              # weight copies per timing: well past any consumer card's L2


def shapes(model: str) -> tuple[int, list[tuple[str, int, int]], tuple[int, int]]:
    """Layers, the (name, out, in) projections of one decoder layer as merged, and the head's (vocab, hidden)."""
    c = json.loads((pathlib.Path(model) / "config.json").read_text())
    h, d = c["hidden_size"], c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
    q, kv, inter = c["num_attention_heads"] * d, c["num_key_value_heads"] * d, c["intermediate_size"]
    return c["num_hidden_layers"], [("qkv", q + 2 * kv, h), ("o", h, q), ("gate_up", 2 * inter, h),
                                    ("down", h, inter)], (c["vocab_size"], h)


def graph_us(fn, args: list[tuple], reps: int = 21) -> float:
    """Median µs per call of `fn(*a)` over `args`, replayed as one CUDA graph."""
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        for a in args:
            fn(*a)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for a in args:
            fn(*a)
    g.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(reps):
        start.record()
        g.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / len(args))
    return statistics.median(times)


def fp8_weights(n: int, k: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    copies = max(2, STREAM_BYTES // (n * k))
    gen = torch.Generator(device="cuda").manual_seed(n * k)
    return [(torch.randint(0, 0x7F, (n, k), dtype=torch.uint8, device="cuda", generator=gen),
             torch.rand(n // 128, k // 128, device="cuda", generator=gen) * 1e-2) for _ in range(copies)]


def time_decode(n: int, k: int, m: int, cfg, weights=None) -> float:
    from holon_plugins.kernels.triton_fp8 import kernel

    x = torch.randn(m, k, device="cuda").bfloat16()
    return graph_us(lambda w, s: kernel.decode(x, w, s, torch.bfloat16, cfg), weights or fp8_weights(n, k))


def sweep(a) -> dict:
    from holon_plugins.kernels.triton_fp8 import kernel

    _, layer, _ = shapes(a.model)
    out = {}
    for name, n, k in layer:
        weights, kb = fp8_weights(n, k), k // kernel.BLOCK
        for m in (1, kernel.DECODE_ROWS):
            results = []
            for bn, sp, nw, ns in itertools.product((16, 32, 64, 128), [s for s in (1, 2, 4, 8, 16) if kb % s == 0],
                                                    (2, 4, 8), (2, 3, 4)):
                cfg = kernel.Decode(bn, sp, nw, ns)
                results.append((time_decode(n, k, m, cfg, weights), cfg))
            results.sort(key=lambda r: r[0])
            out[f"{n}x{k} M={m}"] = [{"us": round(t, 2), "GB/s": round(n * k / t / 1e3), "config": cfg.__dict__}
                                     for t, cfg in results[:5]]
            print(name, m, out[f"{n}x{k} M={m}"][0], flush=True)
        del weights
    return out


def head_us(vocab: int, hidden: int) -> float:
    w = torch.randn(vocab, hidden, device="cuda").bfloat16()
    x = torch.randn(1, hidden, device="cuda").bfloat16()
    return graph_us(lambda: torch.nn.functional.linear(x, w), [()])


def q8_0_profile(gguf: str, n: int = 128, reps: int = 3) -> dict:
    """Holon's tg<n> on the Q8_0 file, batch 1 at depth 1 (bench/llama_bench.py's engine flags): median step ms over
    `reps` runs, and from one profiled run the quantized matmul kernels' time per step."""
    sys.path.insert(0, str(ROOT / "bench"))
    from holon.core.cli import engine_config, engine_flags
    from holon.core.engine import Engine
    from llama_bench import _timed

    flags = argparse.ArgumentParser()
    engine_flags(flags)
    engine = Engine(engine_config(gguf, flags.parse_args(["--max-seqs", "1", "--max-model-len", str(n + 16),
                                                          "--max-step-tokens", "512", "--no-prefix-cache"])))
    rng = random.Random(0)
    steps = []
    for rep in range(reps + 1):
        arrivals, _ = _timed(engine, [rng.randrange(1000, 50000)], n + 1)
        if rep:
            steps.append((arrivals[-1] - arrivals[0]) / n)
    prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
    with prof:
        _timed(engine, [rng.randrange(1000, 50000)], n + 1)
    gemm = launches = other = 0.0
    for e in prof.key_averages():
        if any(s in e.key for s in GEMM_KERNELS):
            gemm += e.device_time_total
            launches += e.count
        else:
            other += e.device_time_total
    engine.close()
    step_ms = statistics.median(steps) * 1e3
    gemm_ms = gemm / 1e3 / (n + 1)
    return {"tg_tok_s": round(1e3 / step_ms, 2), "step_ms": round(step_ms, 4), "gemm_ms_per_step": round(gemm_ms, 4),
            "gemm_launches_per_step": round(launches / (n + 1), 1),
            "other_kernels_ms_per_step": round(other / 1e3 / (n + 1), 4),
            "non_gemm_ms_per_step": round(step_ms - gemm_ms, 4), "runs_step_ms": [round(s * 1e3, 4) for s in steps]}


def llama_tg(path: str, n: int = 128) -> dict:
    """llama-bench's tg<n> at depth 0: mean and the median of repetitions 2–5 (M0.1's rule)."""
    for r in json.loads(pathlib.Path(path).read_text()):
        if r["n_prompt"] == 0 and r["n_gen"] == n and not r.get("n_depth"):
            runs = [n / (ns / 1e9) for ns in r["samples_ns"]]
            return {"mean": round(r["avg_ts"], 2), "runs": [round(x, 2) for x in runs],
                    "median_reps_2_5": round(statistics.median(runs[1:5]), 2), "build": r.get("build_commit")}
    raise ValueError(f"no tg{n} at depth 0 in {path}")


def gate(a) -> dict:
    from holon.core import platform
    from holon_plugins.kernels.triton_fp8 import kernel

    layers, layer, (vocab, hidden) = shapes(a.model)
    gpu = platform.fingerprint(torch.device("cuda")).gpu
    per_shape, gemm_us = {}, 0.0
    for name, n, k in layer:
        cfgs = {r: kernel.decode_config(gpu, n, k, r) for r in (1, kernel.DECODE_ROWS)}
        weights = fp8_weights(n, k)
        times = {m: round(time_decode(n, k, m, cfgs[1 if m == 1 else kernel.DECODE_ROWS], weights), 2)
                 for m in range(1, kernel.DECODE_ROWS + 1)}
        per_shape[name] = {"out": n, "in": k, "configs": {r: c.__dict__ for r, c in cfgs.items()}, "us": times,
                           "GB/s at M=1": round(n * k / times[1] / 1e3)}
        gemm_us += times[1]
        print(name, per_shape[name]["GB/s at M=1"], "GB/s", times, flush=True)
        del weights
    head = head_us(vocab, hidden)
    q8 = q8_0_profile(a.gguf)
    step_ms = layers * gemm_us / 1e3 + head / 1e3 + q8["non_gemm_ms_per_step"]
    llama = llama_tg(a.llama_bench)
    predicted = 1e3 / step_ms
    ratio = predicted / llama["median_reps_2_5"]
    return {"layers": layers, "shapes": per_shape, "fp8_gemm_ms_per_step": round(layers * gemm_us / 1e3, 4),
            "bf16_head_us": round(head, 2), "q8_0_holon": q8, "predicted_step_ms": round(step_ms, 4),
            "predicted_tg128": round(predicted, 2), "llama_bench_q8_0_tg128": llama, "ratio": round(ratio, 3),
            "gate": "pass (>= 0.85)" if ratio >= 0.85 else "fail (< 0.85)"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("sweep", "gate"))
    ap.add_argument("--model", required=True, help="the FP8 checkpoint (its config.json gives the shapes)")
    ap.add_argument("--gguf", help="its llama.cpp Q8_0 conversion (gate)")
    ap.add_argument("--llama-bench", help="llama-bench's JSON for that file (gate)")
    ap.add_argument("--json")
    a = ap.parse_args()
    from holon.core import platform

    result = sweep(a) if a.what == "sweep" else gate(a)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    record = {"gpu_class": platform.fingerprint(torch.device("cuda")).cls, "gpu": torch.cuda.get_device_name(),
              "commit": commit, "command": "python bench/fp8_decode_gate.py " + shlex.join(
                  pathlib.Path(x).name if x.startswith("/") else x for x in sys.argv[1:]), "results": result}
    print(json.dumps(result, indent=1))
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
