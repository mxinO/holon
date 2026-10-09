"""S3 correctness: vendored GDN kernels vs the HF pure-torch reference, on real Qwen3.5-4B / Qwen3.8-27B layer-0
weights with random inputs.

Reference ("golden"): HF `torch_recurrent_gated_delta_rule` in fp32 from the same bf16 q/k/v/g/beta. Noise floor
("hf_bf16"): what HF itself computes in a bf16 model (its chunk prefill path; l2norm in bf16, bf16 output).
Usage: CUDA_VISIBLE_DEVICES=0 python correctness.py results/correctness_sm86.json
"""

import json
import sys

import torch

import gdn  # puts third_party/ on sys.path
import fla_ops  # noqa: E402
from gdn import errors, to_fla, to_hf

T_LIST = [1, 16, 64, 512, 2048, 8192]
BATCH_LENS = [1, 63, 777, 2048]  # batch 4, different lengths, packed varlen
RESUME_SPLITS = {"unaligned": [0, 37, 600, 1500, 2048], "aligned_64": [0, 64, 1024, 1536, 2048]}
DECODE_PREFIX = [5, 100, 512, 1000]
DECODE_STEPS = 32
SLOTS = [3, 1, 6, 2]  # pool slots of the 4 decode Seqs (slot 0 is the kernels' NULL slot)
K_DRAFT = 3
N_ACCEPT = [1, 2, 3, 4]  # tokens accepted per Seq out of k+1 = 4 (includes the always-accepted first token)


def capture(mixer, norm, length, seed):
    """Core inputs (+ raw a/b gate pre-activations for the fused-gating kernels) for one random sequence."""
    x = gdn.random_hidden(norm, 1, length, seed)
    d = gdn.capture_core_inputs(mixer, x)
    with torch.no_grad():
        d["a"] = mixer.in_proj_a(x).contiguous()
        d["b"] = mixer.in_proj_b(x).contiguous()
    return d


def sl(d, lo, hi):
    return {k: v[:, lo:hi] for k, v in d.items()}


def cat(ds):
    return {k: torch.cat([d[k] for d in ds], dim=1) for k in ds[0]}


def core(d):
    return {k: d[k] for k in ("q", "k", "v", "g", "beta")}


def no_nan(*ts):
    return all(bool(torch.isfinite(t).all()) for t in ts)


def prefill_one_shot(mixer, norm, H):
    rows = []
    for T in T_LIST:
        d = core(capture(mixer, norm, T, seed=T))
        o_ref, s_ref = gdn.ref_recurrent(**d)
        o_hf, s_hf = gdn.ref_chunk(**d, golden=False)
        o_chunk32, s_chunk32 = gdn.ref_chunk(**d)
        o, s = gdn.kernel_chunk(**d, h0_fla=torch.zeros(1, H, 128, 128, device="cuda"), cu_seqlens=gdn.cu_seqlens_of([T]))
        rows.append(dict(
            T=T,
            kernel_o=errors(o, o_ref), kernel_state=errors(to_hf(s), s_ref),
            hf_bf16_o=errors(o_hf, o_ref), hf_bf16_state=errors(s_hf, s_ref),
            hf_fp32_chunk_o=errors(o_chunk32, o_ref), hf_fp32_chunk_state=errors(s_chunk32, s_ref),
        ))
        print("prefill", rows[-1]["T"], rows[-1]["kernel_o"]["fro_rel"], rows[-1]["hf_bf16_o"]["fro_rel"], flush=True)
    return rows


def prefill_batch(mixer, norm, H):
    """4 Seqs of different lengths in one varlen launch; Seqs 2 and 3 start from a non-zero state."""
    seqs, h0s, refs = [], [], []
    for i, T in enumerate(BATCH_LENS):
        warm = 300 if i >= 2 else 0
        d = core(capture(mixer, norm, warm + T, seed=100 + i))
        h0 = gdn.ref_recurrent(**sl(d, 0, warm))[1] if warm else torch.zeros(1, H, 128, 128, device="cuda")
        seqs.append(sl(d, warm, warm + T))
        h0s.append(h0)
        refs.append(gdn.ref_recurrent(**seqs[-1], h0_hf=h0))
    o, s = gdn.kernel_chunk(**cat(seqs), h0_fla=to_fla(torch.cat(h0s)), cu_seqlens=gdn.cu_seqlens_of(BATCH_LENS))
    offs = gdn.cu_seqlens_of(BATCH_LENS).tolist()
    return [
        dict(T=T, h0_nonzero=i >= 2, o=errors(o[:, offs[i]:offs[i + 1]], refs[i][0]), state=errors(to_hf(s[i:i + 1]), refs[i][1]))
        for i, T in enumerate(BATCH_LENS)
    ]


def prefill_resume(mixer, norm, H):
    """Chunked prefill resuming from the fp32 state at arbitrary offsets (prefix-cache hit) vs one shot."""
    T = RESUME_SPLITS["unaligned"][-1]
    d = core(capture(mixer, norm, T, seed=7))
    o_ref, s_ref = gdn.ref_recurrent(**d)
    zero = torch.zeros(1, H, 128, 128, device="cuda")
    o_one, s_one = gdn.kernel_chunk(**d, h0_fla=zero, cu_seqlens=gdn.cu_seqlens_of([T]))
    out = {}
    for name, cuts in RESUME_SPLITS.items():
        outs, state = [], zero
        for lo, hi in zip(cuts, cuts[1:]):
            o, state = gdn.kernel_chunk(**sl(d, lo, hi), h0_fla=state, cu_seqlens=gdn.cu_seqlens_of([hi - lo]))
            outs.append(o)
        o_res = torch.cat(outs, dim=1)
        out[name] = dict(
            cuts=cuts,
            resumed_vs_one_shot_o=errors(o_res, o_one), resumed_vs_one_shot_state=errors(state, s_one),
            bitwise_equal=bool(torch.equal(o_res, o_one) and torch.equal(state, s_one)),
            resumed_o=errors(o_res, o_ref), resumed_state=errors(to_hf(state), s_ref),
            one_shot_o=errors(o_one, o_ref), one_shot_state=errors(to_hf(s_one), s_ref),
        )
    return out


def decode_inputs(mixer, norm, extra):
    """Per Seq: full captured stream of DECODE_PREFIX[i] + extra tokens."""
    return [capture(mixer, norm, L + extra, seed=200 + i) for i, L in enumerate(DECODE_PREFIX)]


def prefill_into_pool(streams, H, pool, slots):
    pre = [core(sl(d, 0, L)) for d, L in zip(streams, DECODE_PREFIX)]
    _, s = gdn.kernel_chunk(**cat(pre), h0_fla=torch.zeros(len(pre), H, 128, 128, device="cuda"),
                            cu_seqlens=gdn.cu_seqlens_of(DECODE_PREFIX))
    pool[slots] = s


def step_tokens(streams, pos):
    """Token pos[i] of every Seq, stacked as a batch of N Seqs x 1 token: [N, 1, ...]."""
    return {k: torch.cat([d[k][:, p:p + 1] for d, p in zip(streams, pos)], dim=0) for k in streams[0]}


def decode_step(kind, t, pool, idx, mixer, v_head_tiled=False):
    """One decode token per Seq through one of the three vendored recurrent kernels; updates pool in place."""
    if kind == "fused_recurrent":
        o, _ = fla_ops.fused_recurrent_gated_delta_rule(
            t["q"], t["k"], t["v"], t["g"], t["beta"], initial_state=pool, inplace_final_state=True,
            ssm_state_indices=idx, use_qk_l2norm_in_kernel=True, v_head_tiled=v_head_tiled)
    elif kind == "fused_sigmoid_gating":
        o, _ = fla_ops.fused_sigmoid_gating_delta_rule_update(
            mixer.A_log, t["a"], t["b"], mixer.dt_bias, t["q"], t["k"], t["v"], initial_state=pool,
            inplace_final_state=True, ssm_state_indices=idx, use_qk_l2norm_in_kernel=True, v_head_tiled=v_head_tiled)
    else:  # packed_decode: q/k/v as the post-conv fused row, gating fused
        N = t["q"].shape[0]
        mixed = torch.cat([t["q"].reshape(N, -1), t["k"].reshape(N, -1), t["v"].reshape(N, -1)], dim=-1)
        o = torch.empty(N, 1, *t["v"].shape[2:], device="cuda", dtype=t["v"].dtype)
        fla_ops.fused_recurrent_gated_delta_rule_packed_decode(
            mixed, t["a"].reshape(N, -1), t["b"].reshape(N, -1), mixer.A_log, mixer.dt_bias, 128 ** -0.5,
            pool, o, idx, use_qk_l2norm_in_kernel=True, v_head_tiled=v_head_tiled)
    return o


def decode_after_prefill(mixer, norm, H):
    streams = decode_inputs(mixer, norm, DECODE_STEPS)
    refs = [gdn.ref_recurrent(**core(d)) for d in streams]
    idx = torch.tensor(SLOTS, device="cuda", dtype=torch.int32)
    out = {}
    for kind in ("fused_recurrent", "fused_sigmoid_gating", "packed_decode"):
        pool = torch.zeros(8, H, 128, 128, device="cuda")
        prefill_into_pool(streams, H, pool, idx.long())
        outs = [decode_step(kind, step_tokens(streams, [L + j for L in DECODE_PREFIX]), pool, idx, mixer)
                for j in range(DECODE_STEPS)]
        o = torch.cat(outs, dim=1)  # [N, steps, H, V]
        rows = []
        for i, L in enumerate(DECODE_PREFIX):
            rows.append(dict(prefix=L, o=errors(o[i:i + 1], refs[i][0][:, L:]),
                             state=errors(to_hf(pool[SLOTS[i]:SLOTS[i] + 1]), refs[i][1])))
        untouched = [s for s in range(8) if s not in SLOTS]
        out[kind] = dict(rows=rows, other_slots_untouched=bool((pool[untouched] == 0).all()), finite=no_nan(o, pool))
    return out


def verify_k(mixer, norm, H):
    """Spec verify with k=3 drafts vs sequential decode, for checkpoint_slots and replay commit modes."""
    N, k1 = len(DECODE_PREFIX), K_DRAFT + 1
    streams = decode_inputs(mixer, norm, 2 * k1)
    cu = torch.arange(0, N * k1 + 1, k1, device="cuda", dtype=torch.int32)
    acc = torch.tensor(N_ACCEPT, device="cuda", dtype=torch.int32)

    def tokens(starts, n):  # n tokens per Seq from per-Seq start positions, packed [1, N*n, ...]
        return {key: torch.cat([d[key][:, s:s + n] for d, s in zip(streams, starts)], dim=1) for key in streams[0]}

    def run(t, pool, idx, n_acc=None, cu_seqlens=cu):
        return fla_ops.fused_sigmoid_gating_delta_rule_update(
            mixer.A_log, t["a"], t["b"], mixer.dt_bias, t["q"], t["k"], t["v"], initial_state=pool,
            inplace_final_state=True, cu_seqlens=cu_seqlens, ssm_state_indices=idx, num_accepted_tokens=n_acc,
            use_qk_l2norm_in_kernel=True)[0]

    def sequential(starts, n, base_pool, slots):
        """n single-token decode steps; returns outputs [N, n] and the per-token states [n, N, ...]."""
        pool, idx = base_pool.clone(), torch.tensor(slots, device="cuda", dtype=torch.int32)
        outs, states = [], []
        for j in range(n):
            outs.append(decode_step("fused_sigmoid_gating", step_tokens(streams, [s + j for s in starts]), pool, idx, mixer))
            states.append(pool[idx.long()].clone())
        return torch.cat(outs, dim=1), states

    base = torch.zeros(1 + N * k1, H, 128, 128, device="cuda")
    spec = torch.arange(1, 1 + N * k1, device="cuda", dtype=torch.int32).view(N, k1)  # k+1 slots per Seq
    prefill_into_pool(streams, H, base, spec[:, 0].long())
    start = list(DECODE_PREFIX)

    # checkpoint_slots: step A writes the state after every token into its own slot
    pool = base.clone()
    o_a = run(tokens(start, k1), pool, spec, n_acc=torch.ones(N, device="cuda", dtype=torch.int32))
    o_seq, st_seq = sequential(start, k1, base, spec[:, 0].tolist())
    step_a = dict(
        o_vs_sequential=errors(o_a.view(N, k1, H, 128), o_seq),
        o_bitwise_equal_sequential=bool(torch.equal(o_a.view(N, k1, H, 128), o_seq)),
        slot_states_bitwise_equal_sequential=all(torch.equal(pool[spec[:, j].long()], st_seq[j]) for j in range(k1)),
    )
    # step B reads slot n_accept-1 and continues the accepted stream
    start_b = [s + a for s, a in zip(start, N_ACCEPT)]
    o_b = run(tokens(start_b, k1), pool, spec, n_acc=acc)
    refs = [gdn.ref_recurrent(**core(sl(d, 0, s + k1))) for d, s in zip(streams, start_b)]
    step_b = dict(rows=[dict(n_accept=a, o=errors(o_b.view(N, k1, H, 128)[i:i + 1], refs[i][0][:, -k1:]),
                             state=errors(to_hf(pool[spec[i, -1]].unsqueeze(0)), refs[i][1]))
                        for i, a in enumerate(N_ACCEPT)])
    accepted = base.clone()  # sequential decode restarted from the state after each Seq's accepted prefix
    for i, a in enumerate(N_ACCEPT):
        accepted[spec[i, 0]] = st_seq[a - 1][i]
    o_seq_b, _ = sequential(start_b, k1, accepted, spec[:, 0].tolist())
    step_b["o_bitwise_equal_sequential"] = bool(torch.equal(o_b.view(N, k1, H, 128), o_seq_b))

    # replay: verify stores only the state after token 0; commit re-runs the accepted drafts' cached inputs
    pool_r = base.clone()
    one_slot = torch.zeros(N, k1, device="cuda", dtype=torch.int32)
    one_slot[:, 0] = spec[:, 0]
    o_r = run(tokens(start, k1), pool_r, one_slot)
    replay_n = [a - 1 for a in N_ACCEPT]
    replay_idx = torch.zeros(N, K_DRAFT, device="cuda", dtype=torch.int32)
    for i, n in enumerate(replay_n):
        if n:
            replay_idx[i, n - 1] = spec[i, 0]
    t_rep = {key: torch.cat([d[key][:, s + 1:s + 1 + n] for d, s, n in zip(streams, start, replay_n)], dim=1)
             for key in streams[0]}
    run(t_rep, pool_r, replay_idx, n_acc=torch.tensor([max(n, 1) for n in replay_n], device="cuda", dtype=torch.int32),
        cu_seqlens=gdn.cu_seqlens_of(replay_n))
    refs_r = [gdn.ref_recurrent(**core(sl(d, 0, s + a))) for d, s, a in zip(streams, start, N_ACCEPT)]
    replay = dict(
        verify_o_bitwise_equal_checkpoint_slots=bool(torch.equal(o_r, o_a)),
        committed_state_bitwise_equal_sequential=all(
            torch.equal(pool_r[spec[i, 0]], st_seq[a - 1][i]) for i, a in enumerate(N_ACCEPT)),
        rows=[dict(n_accept=a, state=errors(to_hf(pool_r[spec[i, 0]].unsqueeze(0)), refs_r[i][1]))
              for i, a in enumerate(N_ACCEPT)],
    )
    return dict(k=K_DRAFT, n_accept=N_ACCEPT, checkpoint_slots=dict(step_a=step_a, step_b=step_b), replay=replay,
                finite=no_nan(o_a, o_b, o_r, pool, pool_r))


class KernelPath:
    """Swaps the HF mixer's core GDN calls for the vendored kernels. The fp32 state lives in `pool` (slot 0 is
    NULL); HF's cache would round it to bf16, so the state HF hands back in is ignored."""

    def __init__(self, mixer, v_head_tiled=False):
        self.mixer, self.pool, self.tiled, self.core_out = mixer, None, v_head_tiled, []
        mixer.chunk_gated_delta_rule, mixer.recurrent_gated_delta_rule = self.chunk, self.recurrent

    def _qk(self, query, key):
        rep = self.mixer.num_v_heads // self.mixer.num_k_heads
        return query[:, :, ::rep].contiguous(), key[:, :, ::rep].contiguous()

    def chunk(self, query, key, value, g, beta, **_):
        q, k = self._qk(query, key)
        B, T, H, V = value.shape
        flat = [x.reshape(1, B * T, *x.shape[2:]) for x in (q, k, value, g, beta)]
        cu = torch.arange(0, B * T + 1, T, device="cuda", dtype=torch.int32)
        o, s = gdn.kernel_chunk(*flat, h0_fla=torch.zeros(B, H, V, k.shape[-1], device="cuda"), cu_seqlens=cu,
                                v_head_tiled=self.tiled)
        self.pool = torch.cat([torch.zeros_like(s[:1]), s])
        self.core_out.append(o.view(B, T, H, V))
        return o.view(B, T, H, V), to_hf(s)

    def recurrent(self, query, key, value, g, beta, **_):
        q, k = self._qk(query, key)
        idx = torch.arange(1, value.shape[0] + 1, device="cuda", dtype=torch.int32)
        o, _ = fla_ops.fused_recurrent_gated_delta_rule(
            q, k, value, g, beta, initial_state=self.pool, inplace_final_state=True, ssm_state_indices=idx,
            use_qk_l2norm_in_kernel=True, v_head_tiled=self.tiled)
        self.core_out.append(o)
        return o, to_hf(self.pool[1:])


@torch.no_grad()
def layer_level(model, decode_steps=16):
    """Whole GDN mixer (projections, conv, gating, core, gated RMSNorm, out_proj): ours (bf16 weights + kernels,
    fp32 state) and HF bf16 (torch path, HF caches the state in bf16) vs HF fp32 (torch path)."""
    from transformers import DynamicCache

    cfg = gdn.text_config(model)
    gold, norm = gdn.load_layer(model, dtype=torch.float32)
    hf_bf16, _ = gdn.load_layer(model)
    ours, _ = gdn.load_layer(model)
    path = KernelPath(ours)
    rows = []
    for T in (16, 512, 2048, 8192):
        x = gdn.random_hidden(norm, 1, T + decode_steps, seed=300 + T).bfloat16()
        ys = {}
        for name, m in (("gold", gold), ("hf_bf16", hf_bf16), ("ours", ours)):
            cache = DynamicCache(config=cfg)
            xin = x.float() if name == "gold" else x
            y = [m(xin[:, :T], cache_params=cache)]
            y += [m(xin[:, T + j:T + j + 1], cache_params=cache) for j in range(decode_steps)]
            ys[name] = (y[0].float(), torch.cat(y[1:], dim=1).float())
        row = dict(T=T, decode_steps=decode_steps)
        for phase, i in (("prefill", 0), ("decode", 1)):
            e_ours, e_hf = errors(ys["ours"][i], ys["gold"][i]), errors(ys["hf_bf16"][i], ys["gold"][i])
            row[phase] = dict(ours=e_ours, hf_bf16=e_hf, fro_ratio_ours_over_hf_bf16=e_ours["fro_rel"] / e_hf["fro_rel"])
        rows.append(row)
        print("layer", model, T, row["prefill"]["fro_ratio_ours_over_hf_bf16"], row["decode"]["fro_ratio_ours_over_hf_bf16"],
              flush=True)
    del path
    return rows


def main(out_path, models=("4b", "27b")):
    result = dict(device=gdn.device_info(), models={})
    for model in models:
        mixer, norm = gdn.load_layer(model)
        H = gdn.gdn_shape(model)["H"]
        r = dict(shape=gdn.gdn_shape(model))
        r["prefill_one_shot"] = prefill_one_shot(mixer, norm, H)
        r["prefill_batch4_varlen"] = prefill_batch(mixer, norm, H)
        r["prefill_resume"] = prefill_resume(mixer, norm, H)
        r["decode_after_prefill"] = decode_after_prefill(mixer, norm, H)
        r["verify_k"] = verify_k(mixer, norm, H)
        del mixer
        torch.cuda.empty_cache()
        r["layer_level"] = layer_level(model)
        result["models"][model] = r
        torch.cuda.empty_cache()
    result["max_memory_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:] and [tuple(sys.argv[2:])]))
