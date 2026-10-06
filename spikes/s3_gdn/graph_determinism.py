"""S3 graph capture + determinism.

1. Recurrent decode / verify captured in a CUDA graph with slot indices, accepted counts and inputs in static device
   tensors; replays with new contents vs eager (bitwise).
2. Chunked prefill captured with host-precomputed chunk indices (informational: v0 runs prefill eager).
3. Run-to-run byte identity: 10 runs in-process, and SHA-256 of outputs across fresh processes (pinned configs, and
   once with vLLM's autotuning to show what pinning buys).
Usage: CUDA_VISIBLE_DEVICES=3 python graph_determinism.py results/graph_det_sm86.json
"""

import hashlib
import json
import os
import subprocess
import sys

import torch

import gdn
import fla_ops
from fla_ops.index import prepare_chunk_indices, prepare_chunk_offsets

SHAPE = gdn.gdn_shape("9b")
H, Hg, D = SHAPE["H"], SHAPE["Hg"], SHAPE["K"]
A_LOG = torch.linspace(-1, 1, H, device="cuda")
DT_BIAS = torch.linspace(-0.5, 0.5, H, device="cuda")


def rand(gen, n_tok):
    r = lambda *s: torch.randn(*s, generator=gen, device="cuda")  # noqa: E731
    return dict(q=r(1, n_tok, Hg, D).bfloat16(), k=r(1, n_tok, Hg, D).bfloat16(), v=r(1, n_tok, H, D).bfloat16(),
                a=r(1, n_tok, H).bfloat16(), b=r(1, n_tok, H).bfloat16(),
                g=-torch.rand(1, n_tok, H, generator=gen, device="cuda") * 2,
                beta=torch.rand(1, n_tok, H, generator=gen, device="cuda").bfloat16())


def sigmoid_step(x, pool, idx, cu, n_acc=None):
    return fla_ops.fused_sigmoid_gating_delta_rule_update(
        A_LOG, x["a"], x["b"], DT_BIAS, x["q"], x["k"], x["v"], initial_state=pool, inplace_final_state=True,
        cu_seqlens=cu, ssm_state_indices=idx, num_accepted_tokens=n_acc, use_qk_l2norm_in_kernel=True)[0]


def recurrent_step(x, pool, idx, cu, n_acc=None):
    return fla_ops.fused_recurrent_gated_delta_rule(
        x["q"], x["k"], x["v"], x["g"], x["beta"], initial_state=pool, inplace_final_state=True, cu_seqlens=cu,
        ssm_state_indices=idx, num_accepted_tokens=n_acc, use_qk_l2norm_in_kernel=True)[0]


def graph_recurrent(step, n_seqs=8, t_per_seq=1, replays=5):
    """Capture once; each replay writes new inputs, a new slot assignment (incl. NULL padding) and new accepted
    counts into the static buffers. Graph output and pool must equal an eager run on a pool copy, bitwise."""
    gen = torch.Generator(device="cuda").manual_seed(1)
    static = rand(gen, n_seqs * t_per_seq)
    n_slots = 1 + 2 * n_seqs * t_per_seq
    pool = torch.randn(n_slots, H, D, D, generator=gen, device="cuda") * 0.1
    pool[0] = 0
    shape = (n_seqs,) if t_per_seq == 1 else (n_seqs, t_per_seq)
    idx = torch.zeros(shape, device="cuda", dtype=torch.int32)
    n_acc = torch.ones(n_seqs, device="cuda", dtype=torch.int32) if t_per_seq > 1 else None
    cu = torch.arange(0, n_seqs * t_per_seq + 1, t_per_seq, device="cuda", dtype=torch.int32)

    def fill(r):
        g = torch.Generator(device="cuda").manual_seed(100 + r)
        for k, v in rand(g, n_seqs * t_per_seq).items():
            static[k].copy_(v)
        perm = torch.randperm(n_slots - 1, generator=g, device="cuda")[: idx.numel()] + 1
        perm[: r % 3] = 0  # some padded Seqs (NULL slot)
        idx.copy_(perm.view(shape).to(torch.int32))
        if n_acc is not None:
            n_acc.copy_(torch.randint(1, t_per_seq + 1, (n_seqs,), generator=g, device="cuda", dtype=torch.int32))

    fill(0)
    step(static, pool.clone(), idx, cu, n_acc)  # compile outside capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step(static, pool, idx, cu, n_acc)
    results = []
    for r in range(1, replays + 1):
        fill(r)
        ref_pool = pool.clone()
        ref_out = step(static, ref_pool, idx, cu, n_acc)
        graph.replay()
        live = (idx.view(n_seqs, -1)[:, 0] > 0) if n_acc is None else (idx.gather(1, (n_acc - 1).long()[:, None])[:, 0] > 0)
        live_tok = live.repeat_interleave(t_per_seq)
        results.append(dict(
            output_bitwise_equal=bool(torch.equal(out[:, live_tok], ref_out[:, live_tok])),
            pool_bitwise_equal=bool(torch.equal(pool, ref_pool)),
            padded_seqs=int((~live).sum()),
        ))
    return dict(n_seqs=n_seqs, tokens_per_seq=t_per_seq, replays=results)


def graph_chunk(lens=(300, 1000, 64, 2048)):
    gen = torch.Generator(device="cuda").manual_seed(2)
    n = sum(lens)
    static = rand(gen, n)
    cu = gdn.cu_seqlens_of(list(lens))
    ci, co = prepare_chunk_indices(cu, 64), prepare_chunk_offsets(cu, 64)
    h0 = torch.randn(len(lens), H, D, D, generator=gen, device="cuda") * 0.1

    def run():
        return fla_ops.chunk_gated_delta_rule(
            static["q"], static["k"], static["v"], static["g"], static["beta"], initial_state=h0,
            output_final_state=True, cu_seqlens=cu, chunk_indices=ci, chunk_offsets=co, use_qk_l2norm_in_kernel=True)

    run()
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            o, s = run()
    except Exception as e:  # noqa: BLE001 — record why capture fails
        return dict(captured=False, error=repr(e))
    g2 = torch.Generator(device="cuda").manual_seed(3)
    for k, v in rand(g2, n).items():
        static[k].copy_(v)
    graph.replay()
    o_ref, s_ref = run()
    return dict(captured=True, lens=list(lens), output_bitwise_equal=bool(torch.equal(o, o_ref)),
                state_bitwise_equal=bool(torch.equal(s, s_ref)))


def outputs() -> dict[str, list[torch.Tensor]]:
    """The tensors whose bytes are compared: chunk (one-shot and 4-Seq varlen) and decode/verify kernels."""
    gen = torch.Generator(device="cuda").manual_seed(0)
    res = {}
    for name, lens in (("chunk_T8192", [8192]), ("chunk_varlen", [1, 63, 777, 2048])):
        x = rand(gen, sum(lens))
        h0 = torch.randn(len(lens), H, D, D, generator=gen, device="cuda") * 0.1
        o, s = gdn.kernel_chunk(x["q"], x["k"], x["v"], x["g"], x["beta"], h0, gdn.cu_seqlens_of(lens))
        res[name] = [o, s]
    pool0 = torch.randn(17, H, D, D, generator=gen, device="cuda") * 0.1
    x = rand(gen, 16)
    idx = torch.arange(1, 17, device="cuda", dtype=torch.int32)
    cu = torch.arange(0, 17, device="cuda", dtype=torch.int32)
    for name, step in (("fused_recurrent_decode", recurrent_step), ("fused_sigmoid_decode", sigmoid_step)):
        pool = pool0.clone()
        res[name] = [step(x, pool, idx, cu), pool]
    pool = pool0.clone()
    cu4 = torch.arange(0, 17, 4, device="cuda", dtype=torch.int32)
    n_acc = torch.tensor([1, 2, 3, 4], device="cuda", dtype=torch.int32)
    res["fused_sigmoid_verify_k3"] = [sigmoid_step(x, pool, idx.view(4, 4), cu4, n_acc), pool]
    return res


def digest(ts):
    h = hashlib.sha256()
    for t in ts:
        h.update(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def in_process_identical(runs=10):
    first = {k: digest(v) for k, v in outputs().items()}
    same = {k: True for k in first}
    for _ in range(runs - 1):
        for k, v in outputs().items():
            same[k] &= digest(v) == first[k]
    return same


def fresh_process_hashes(autotune):
    env = dict(os.environ, S3_FLA_AUTOTUNE="1" if autotune else "0")
    p = subprocess.run([sys.executable, __file__, "--hashes"], env=env, capture_output=True, text=True, check=True)
    return json.loads(p.stdout.strip().splitlines()[-1])


def main(out_path):
    result = dict(device=gdn.device_info())
    result["graph_decode_fused_recurrent"] = graph_recurrent(recurrent_step)
    result["graph_decode_fused_sigmoid"] = graph_recurrent(sigmoid_step)
    result["graph_verify_k3_fused_sigmoid"] = graph_recurrent(sigmoid_step, t_per_seq=4)
    result["graph_chunk_prefill"] = graph_chunk()
    result["in_process_10_runs_identical"] = in_process_identical()
    runs = [fresh_process_hashes(False) for _ in range(2)]
    tuned = fresh_process_hashes(True)
    result["process_hashes_pinned"] = runs[0]
    result["two_processes_identical"] = runs[0] == runs[1]
    result["autotuned_process_equal_to_pinned"] = {k: tuned[k] == runs[0][k] for k in runs[0]}
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    if sys.argv[1:] == ["--hashes"]:
        print(json.dumps({k: digest(v) for k, v in outputs().items()}))
    else:
        main(sys.argv[1])
