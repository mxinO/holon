"""S3 performance: chunked prefill throughput and recurrent decode / verify latency of the vendored GDN kernels at
Qwen3.5-9B and Qwen3.8-27B layer shapes (kernel level: core GDN incl. in-kernel q/k l2norm; projections excluded).

Usage: CUDA_VISIBLE_DEVICES=0 python bench.py results/perf_sm86.json [autotune]
`autotune` searches vLLM's original config space instead of the pinned configs (prefill only).
"""

import json
import math
import os
import statistics
import sys

if sys.argv[2:] == ["autotune"]:
    os.environ["S3_FLA_AUTOTUNE"] = "1"  # must precede the fla_ops import (inside gdn)

import torch  # noqa: E402

import gdn  # noqa: E402
import fla_ops  # noqa: E402
from fla_ops.index import prepare_chunk_indices, prepare_chunk_offsets  # noqa: E402

PREFILL = [[64], [512], [2048], [8192], [32768], [2048] * 4]
DECODE_N = [1, 4, 16, 64]
VERIFY_N = [1, 4, 16]
K_DRAFT = 3


def cuda_time_ms(fn, iters=20, warmup=3):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    return statistics.median(times)


def rand_inputs(shape, n_tok, gen):
    Hg, H, K, V = shape["Hg"], shape["H"], shape["K"], shape["V"]
    r = lambda *s: torch.randn(*s, generator=gen, device="cuda")  # noqa: E731
    return dict(
        q=r(1, n_tok, Hg, K).bfloat16(), k=r(1, n_tok, Hg, K).bfloat16(), v=r(1, n_tok, H, V).bfloat16(),
        a=r(1, n_tok, H).bfloat16(), b=r(1, n_tok, H).bfloat16(),
        g=-torch.rand(1, n_tok, H, generator=gen, device="cuda") * 2, beta=torch.rand(1, n_tok, H, generator=gen, device="cuda").bfloat16(),
    )


def bench_prefill(shape, gen, autotune):
    rows = []
    # vLLM's autotune keys exclude T: the first call's T picks the configs for every T. Tune at a representative T.
    for lens in ([[8192]] if autotune else []) + PREFILL:
        n = sum(lens)
        x = rand_inputs(shape, n, gen)
        cu = gdn.cu_seqlens_of(lens)
        ci, co = prepare_chunk_indices(cu, 64), prepare_chunk_offsets(cu, 64)  # host-side "plan", outside timing
        h0 = torch.zeros(len(lens), shape["H"], shape["V"], shape["K"], device="cuda")
        fn = lambda: fla_ops.chunk_gated_delta_rule(  # noqa: E731
            x["q"], x["k"], x["v"], x["g"], x["beta"], initial_state=h0, output_final_state=True, cu_seqlens=cu,
            chunk_indices=ci, chunk_offsets=co, use_qk_l2norm_in_kernel=True)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        ms = cuda_time_ms(fn)
        workspace_mib = (torch.cuda.max_memory_allocated() - base) / 2**20  # internal temporaries + outputs
        graph = torch.cuda.CUDAGraph()  # GPU time without host launch overhead (v0 runs prefill eager)
        with torch.cuda.graph(graph):
            fn()
        graph_ms = cuda_time_ms(graph.replay)
        rows.append(dict(lens=lens, tokens=n, ms_per_layer=ms, graph_ms_per_layer=graph_ms, workspace_mib=workspace_mib,
                         tok_s_per_layer=n / ms * 1e3,
                         tok_s_all_gdn_layers=n / (ms * shape["gdn_layers"]) * 1e3))
        print("prefill", shape["H"], lens if len(lens) == 1 else f"{len(lens)}x{lens[0]}", f"{ms:.3f} ms eager",
              f"{graph_ms:.3f} ms graph", flush=True)
        del graph
    return rows[1:] if autotune else rows


def decode_call(kind, x, pool, idx, A_log, dt_bias, n, t_per_seq=1, n_acc=None):
    """One layer's decode (t_per_seq=1) or verify (t_per_seq=k+1, idx [n, k+1]) call; static tensors only."""
    if kind == "fused_recurrent":
        return lambda: fla_ops.fused_recurrent_gated_delta_rule(
            x["q"].view(n, 1, *x["q"].shape[2:]), x["k"].view(n, 1, *x["k"].shape[2:]), x["v"].view(n, 1, *x["v"].shape[2:]),
            x["g"].view(n, 1, -1), x["beta"].view(n, 1, -1), initial_state=pool, inplace_final_state=True,
            ssm_state_indices=idx, use_qk_l2norm_in_kernel=True)
    if kind == "packed_decode":
        mixed = torch.cat([x["q"].view(n, -1), x["k"].view(n, -1), x["v"].view(n, -1)], dim=-1)
        out = torch.empty(n, 1, *x["v"].shape[2:], device="cuda", dtype=torch.bfloat16)
        return lambda: fla_ops.fused_recurrent_gated_delta_rule_packed_decode(
            mixed, x["a"].view(n, -1), x["b"].view(n, -1), A_log, dt_bias, 128 ** -0.5, pool, out, idx,
            use_qk_l2norm_in_kernel=True)
    cu = torch.arange(0, n * t_per_seq + 1, t_per_seq, device="cuda", dtype=torch.int32)
    return lambda: fla_ops.fused_sigmoid_gating_delta_rule_update(
        A_log, x["a"], x["b"], dt_bias, x["q"], x["k"], x["v"], initial_state=pool, inplace_final_state=True,
        cu_seqlens=cu, ssm_state_indices=idx, num_accepted_tokens=n_acc, use_qk_l2norm_in_kernel=True)


def bench_recurrent(shape, gen, kinds, ns, t_per_seq):
    """Graph-captured: one call per pool, pools rotated so the states do not stay in L2 (as in a real model,
    where weights stream through L2 between GDN layers). Reports per-layer GPU time and eager per-call time."""
    H, K, V = shape["H"], shape["K"], shape["V"]
    A_log, dt_bias = torch.randn(H, device="cuda"), torch.randn(H, device="cuda")
    rows = []
    for n in ns:
        slots_per_seq = t_per_seq
        pool_bytes = (1 + n * slots_per_seq) * H * V * K * 4
        n_pools = min(shape["gdn_layers"], max(2, math.ceil(512 * 2**20 / pool_bytes)))
        pools = [torch.zeros(1 + n * slots_per_seq, H, V, K, device="cuda") for _ in range(n_pools)]
        x = rand_inputs(shape, n * t_per_seq, gen)
        if t_per_seq == 1:
            idx, n_acc = torch.arange(1, n + 1, device="cuda", dtype=torch.int32), None
        else:
            idx = torch.arange(1, 1 + n * t_per_seq, device="cuda", dtype=torch.int32).view(n, t_per_seq)
            n_acc = torch.randint(1, t_per_seq + 1, (n,), device="cuda", dtype=torch.int32)
        for kind in kinds:
            calls = [decode_call(kind, x, p, idx, A_log, dt_bias, n, t_per_seq, n_acc) for p in pools]
            for c in calls:
                c()  # compile outside capture
            eager_ms = cuda_time_ms(lambda: [c() for c in calls]) / n_pools
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for c in calls:
                    c()
            graph_ms = cuda_time_ms(graph.replay) / n_pools
            state_bytes = 2 * n * H * V * K * 4  # read + write one state per Seq
            rows.append(dict(kind=kind, n_seqs=n, tokens_per_seq=t_per_seq, n_pools=n_pools,
                             graph_us_per_layer=graph_ms * 1e3, eager_us_per_layer=eager_ms * 1e3,
                             graph_us_all_gdn_layers=graph_ms * 1e3 * shape["gdn_layers"],
                             state_GBps=state_bytes / (graph_ms * 1e-3) / 1e9))
            print(kind, H, n, t_per_seq, f"{graph_ms * 1e3:.1f} us graph, {eager_ms * 1e3:.1f} us eager", flush=True)
        del pools
        torch.cuda.empty_cache()
    return rows


def main(out_path, autotune):
    gen = torch.Generator(device="cuda").manual_seed(0)
    result = dict(device=gdn.device_info(), autotune=autotune, models={})
    for model in ("9b", "27b"):
        shape = gdn.gdn_shape(model)
        r = dict(shape=shape, prefill=bench_prefill(shape, gen, autotune))
        if not autotune:
            r["decode"] = bench_recurrent(shape, gen, ("fused_recurrent", "fused_sigmoid_gating", "packed_decode"),
                                          DECODE_N, 1)
            r["verify"] = bench_recurrent(shape, gen, ("fused_sigmoid_gating",), VERIFY_N, K_DRAFT + 1)
        result["models"][model] = r
    result["max_memory_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] == ["autotune"])
