"""S3 GGUF tiled value-head order: can the kernels consume llama.cpp's tiled v-head layout with a `v_head_order`
knob (q/k head = v head % Hg, i.e. `repeat` instead of `repeat_interleave`) instead of repacking weights?

The GGUF layout is simulated from real HF weights with the inverse of transformers' `TiledToGroupedRows`
(src/transformers/integrations/gguf/gguf_conversion_mapping.py, `_qwen35`): tiled is `v0k0 v0k1 v1k0 v1k1`,
grouped (HF) is `v0k0 v1k0 v0k1 v1k1`. Permuted tensors, as listed there: in_proj_qkv (value block rows),
conv1d (value block rows), in_proj_z (rows), in_proj_a / in_proj_b / A_log / dt_bias (per head), out_proj (input
columns). Usage: CUDA_VISIBLE_DEVICES=3 python tiled_order.py results/tiled_order_sm86.json
"""

import json
import sys

import torch
from transformers import DynamicCache

import gdn
from correctness import KernelPath, capture, decode_step, step_tokens


def tiled_from_grouped(num_k_heads, heads_per_k, head_dim=1):
    """Index map with tiled[i] = grouped[map[i]] (transformers' TiledToGroupedRows applies its argsort)."""
    idx = torch.arange(num_k_heads * heads_per_k * head_dim)
    return idx.reshape(num_k_heads, heads_per_k, head_dim).transpose(0, 1).reshape(-1)


@torch.no_grad()
def to_gguf_order(mixer):
    """Rewrite a grouped (HF) mixer's weights into the GGUF tiled value-head order, in place."""
    nk, rep, dv = mixer.num_k_heads, mixer.num_v_heads // mixer.num_k_heads, mixer.head_v_dim
    per_value, per_head = tiled_from_grouped(nk, rep, dv).cuda(), tiled_from_grouped(nk, rep).cuda()
    qk = 2 * mixer.key_dim
    for w in (mixer.in_proj_qkv.weight, mixer.conv1d.weight):
        w[qk:] = w[qk:][per_value].clone()
    mixer.in_proj_z.weight.copy_(mixer.in_proj_z.weight[per_value])
    for w in (mixer.in_proj_a.weight, mixer.in_proj_b.weight, mixer.A_log, mixer.dt_bias):
        w.copy_(w[per_head])
    mixer.out_proj.weight.copy_(mixer.out_proj.weight[:, per_value])
    return per_head


@torch.no_grad()
def run(mixer, x, T, steps):
    """Prefill T tokens then decode `steps`; returns (mixer output, the out_proj inputs)."""
    seen = []
    hook = mixer.out_proj.register_forward_pre_hook(lambda _, args: seen.append(args[0]))
    cache = DynamicCache(config=mixer.config)
    ys = [mixer(x[:, :T], cache_params=cache)] + [mixer(x[:, T + j:T + j + 1], cache_params=cache) for j in range(steps)]
    hook.remove()
    return torch.cat(ys, dim=1), torch.cat(seen, dim=1)


def check(model, T=1000, steps=8):
    grouped, norm = gdn.load_layer(model)
    grouped.config = gdn.text_config(model)
    ref_path = KernelPath(grouped)
    x = gdn.random_hidden(norm, 1, T + steps, seed=11)
    y_ref, z_ref = run(grouped, x, T, steps)
    pool_ref = ref_path.pool.clone()

    tiled, _ = gdn.load_layer(model)
    tiled.config = grouped.config
    per_head = to_gguf_order(tiled)
    per_value_inv = torch.argsort(tiled_from_grouped(tiled.num_k_heads, len(per_head) // tiled.num_k_heads, tiled.head_v_dim))
    out = dict(model=model, T=T, decode_steps=steps)
    for knob in (True, False):  # False = the negative control: tiled weights through the grouped kernels
        path = KernelPath(tiled, v_head_tiled=knob)
        y, z = run(tiled, x, T, steps)
        core = [o[:, :, torch.argsort(per_head)] for o in path.core_out]  # back to grouped head order
        state = path.pool[:, torch.argsort(per_head)]
        out[f"v_head_tiled={knob}"] = dict(
            core_outputs_bitwise_equal=all(torch.equal(a, b) for a, b in zip(core, ref_path.core_out)),
            final_state_bitwise_equal=bool(torch.equal(state, pool_ref)),
            out_proj_input_bitwise_equal=bool(torch.equal(z[..., per_value_inv.cuda()], z_ref)),
            layer_output=gdn.errors(y.float(), y_ref.float()),
        )
    return out


@torch.no_grad()
def recurrent_kernels(model, n_seqs=4, steps=4):
    """The three decode kernels on tiled-order inputs/state with the knob vs grouped order, bitwise."""
    grouped, norm = gdn.load_layer(model)
    tiled, _ = gdn.load_layer(model)
    per_head = to_gguf_order(tiled)  # also permutes A_log / dt_bias
    back = torch.argsort(per_head)
    streams = [capture(grouped, norm, steps, seed=40 + i) for i in range(n_seqs)]
    idx = torch.arange(1, n_seqs + 1, device="cuda", dtype=torch.int32)
    pool0 = torch.randn(n_seqs + 1, grouped.num_v_heads, 128, 128, device="cuda") * 0.1
    out = {}
    for kind in ("fused_recurrent", "fused_sigmoid_gating", "packed_decode"):
        pool_g, pool_t = pool0.clone(), pool0[:, per_head].clone()
        same = True
        for j in range(steps):
            t = step_tokens(streams, [j] * n_seqs)
            t_tiled = {k: (v[:, :, per_head] if k in ("v", "g", "beta", "a", "b") else v) for k, v in t.items()}
            o_g = decode_step(kind, t, pool_g, idx, grouped)
            o_t = decode_step(kind, t_tiled, pool_t, idx, tiled, v_head_tiled=True)
            same &= torch.equal(o_t[:, :, back], o_g)
        out[kind] = dict(outputs_bitwise_equal=bool(same), state_bitwise_equal=bool(torch.equal(pool_t[:, back], pool_g)))
    return out


def main(out_path):
    result = dict(device=gdn.device_info(), models=[check(m) for m in ("4b", "27b")],
                  recurrent_kernels={m: recurrent_kernels(m) for m in ("4b", "27b")})
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
