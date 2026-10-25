"""GLM-5.3-Flash (`glm5_next`; llama.cpp's `glm5-next` GGUF): a pure recipe (imports only holon.api), written fresh
(docs/v1-glm.md §5: it shares no names, hyper-connections or rules with `qwen4_exp`).

The trunk carries R, four float32 streams of the hidden size (`ModelSpec.residual`, D3): `std.hyper.expand@1` around
the embedding (the token table on the host, `std.tiered.rows@1`), a `std.hyper.mhc_block@1` per layer, the stream mean
then the RMSNorm as the norm (`std.hyper.mean@1`). Every layer the GGUF gives a KV head (`attention.head_count_kv`
read raw per layer: 1 for DSA, 0 for KDA) is DSA (`std.mixers.dsa_mla@1`, D5), the rest KDA (`std.mixers.kda@1`,
D4); the first `leading_dense_block_count` FFNs are dense SwiGLU, the rest the routed experts (sigmoid scores, the
selection bias, scaled; D6) plus the ungated shared expert (`std.ffn.sum@1`), every SwiGLU clamped at
`swiglu_clamp_exp`. The mixers and the dense / shared FFN take `rows` (D9; 0: off).

llama.cpp's converter (27c7a76, conversion/glm.py) writes `ssm_a = −exp(A_log)` (undone by `glm5_next.a_log`), each
KDA conv as [1, C, 1, K] (`glm5_next.conv`: [C, K]), MLA's `kv_b_proj` split into `attn_k_b` [H, kv_lora, nope] and
`attn_v_b` [H, v, kv_lora] (held as such), and the mHC tensors under `hc_{attn,ffn}_{fn,base,scale}`. Norms are
stored as their weights (no offset). The MTP block (`blk.<n>` past the trunk) is skipped: MTP is deferred (D7).
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from holon.api import LayerDecl, ModelSpec, Ref, Residual, WeightMap, WeightRule

_KDA = (("attn_q", "q_proj"), ("attn_k", "k_proj"), ("attn_v", "v_proj"), ("ssm_f_a", "f_a_proj"),
        ("ssm_f_b", "f_b_proj"), ("ssm_beta", "b_proj"), ("ssm_g_a", "g_a_proj"), ("ssm_g_b", "g_b_proj"))
_DSA = (("attn_q_a", "q_a"), ("attn_q_a_norm", "q_a_norm"), ("attn_q_b", "q_b"), ("attn_kv_a_mqa", "kv_a"),
        ("attn_kv_a_norm", "kv_a_norm"), ("indexer.attn_q_b", "wq_b"), ("indexer.attn_k", "wk"),
        ("indexer.proj", "weights_proj"), ("indexer_compressor_gate", "compress_gate"))
RULES = (
    WeightRule("token_embd.weight", "embed.inner.weight"),
    WeightRule("output.weight", "lm_head.weight"),
    WeightRule("output_norm.weight", "norm.inner.weight"),
    *(WeightRule(f"blk.{{i}}.hc_{half}_{p}.weight", f"layers.{{i}}.{half}_hc.{p}")
      for half in ("attn", "ffn") for p in ("fn", "base", "scale")),
    WeightRule("blk.{i}.attn_norm.weight", "layers.{i}.input_norm.weight"),
    WeightRule("blk.{i}.ffn_norm.weight", "layers.{i}.post_norm.weight"),
    WeightRule("blk.{i}.attn_output.weight", "layers.{i}.mixer.o_proj.weight"),          # KDA's and DSA's
    *(WeightRule(f"blk.{{i}}.{g}.weight", f"layers.{{i}}.mixer.{h}.weight") for g, h in _KDA + _DSA),
    *(WeightRule(f"blk.{{i}}.ssm_conv1d_{x}.weight", f"layers.{{i}}.mixer.conv_{x}", "glm5_next.conv") for x in "qkv"),
    WeightRule("blk.{i}.ssm_a", "layers.{i}.mixer.A_log", "glm5_next.a_log"),
    WeightRule("blk.{i}.ssm_dt.bias", "layers.{i}.mixer.dt_bias"),
    WeightRule("blk.{i}.ssm_norm.weight", "layers.{i}.mixer.norm_weight"),
    WeightRule("blk.{i}.attn_k_b.weight", "layers.{i}.mixer.k_b"),
    WeightRule("blk.{i}.attn_v_b.weight", "layers.{i}.mixer.v_b"),
    WeightRule("blk.{i}.indexer.k_norm.weight", "layers.{i}.mixer.k_norm_weight"),
    WeightRule("blk.{i}.indexer.k_norm.bias", "layers.{i}.mixer.k_norm_bias"),
    WeightRule("blk.{i}.indexer_compressor_ape.weight", "layers.{i}.mixer.ape"),
    *(WeightRule(f"blk.{{i}}.ffn_{g}.weight", f"layers.{{i}}.ffn.{g}_proj.weight") for g in ("gate", "up", "down")),
    WeightRule("blk.{i}.ffn_gate_inp.weight", "layers.{i}.ffn.parts.0.router"),
    WeightRule("blk.{i}.exp_probs_b.bias", "layers.{i}.ffn.parts.0.bias"),
    *(WeightRule(f"blk.{{i}}.ffn_{g}_exps.weight", f"layers.{{i}}.ffn.parts.0.{g}.weight")
      for g in ("gate", "up", "down")),
    *(WeightRule(f"blk.{{i}}.ffn_{g}_shexp.weight", f"layers.{{i}}.ffn.parts.1.{g}_proj.weight")
      for g in ("gate", "up", "down")),
)
ROWS = 2048           # D9: KDA's and the dense / shared FFN's sub-batch (N4: ≈ 113–119 KiB a token)
DSA_ROWS = 1024       # DSA's query path: its f32 q̃ and context are ≈ 0.4 MiB a row (step 4: ≤ 126 KiB a token at 16K)


def facts(c: Mapping[str, Any]) -> dict[str, Any]:
    """What the recipe reads from a `glm5-next` GGUF's keys (`c["gguf"]`)."""
    g = c["gguf"]
    n = c["num_hidden_layers"] - g.get("nextn_predict_layers", 0)          # block_count includes MTP blocks
    kv = g["attention.head_count_kv"]
    kv = list(kv) if isinstance(kv, (list, tuple)) else [kv] * n
    limit = g.get("swiglu_clamp_exp", 0.0)
    return {"layers": n, "dsa": [bool(k) for k in kv[:n]], "dense": g["leading_dense_block_count"],
            "streams": g["hyper_connection.count"], "iters": g["hyper_connection.sinkhorn_iterations"],
            "hc_eps": g["hyper_connection.epsilon"], "heads": g["attention.head_count"],
            "head_dim": g["kda.head_dim"], "conv_kernel": g["ssm.conv_kernel"],
            "lower_bound": g.get("kda.gate_lower_bound", -5.0), "q_lora": g["attention.q_lora_rank"],
            "kv_lora": g["attention.kv_lora_rank"], "nope": g["attention.key_length_mla"],
            "v": g["attention.value_length_mla"], "index_heads": g["attention.indexer.head_count"],
            "index_head_dim": g["attention.indexer.key_length"], "budget": g["attention.indexer.top_k"],
            "ratio": g["attention.indexer.kpool"], "index_eps": g.get("attention.layer_norm_epsilon", 1e-6),
            "experts": g["expert_count"], "top_k": g["expert_used_count"], "inter": g["expert_feed_forward_length"],
            "shared": g.get("expert_shared_count", 1), "scale": g.get("expert_weights_scale", 1.0),
            "renormalize": bool(g.get("expert_weights_norm", True)), "dense_inter": g["feed_forward_length"],
            "limit": float(limit[0] if isinstance(limit, (list, tuple)) else limit),
            "skip": tuple(f"blk.{i}." for i in range(n, c["num_hidden_layers"]))}


def recipe(c: Mapping[str, Any], source: str) -> ModelSpec:
    if source != "gguf":
        raise ValueError(f"glm5_next: GGUF files only (D1), not {source!r}")
    f, eps, hidden, rows = facts(c), c["rms_norm_eps"], c["hidden_size"], ROWS
    norm = Ref("std.ops.rmsnorm@1", eps=eps)
    kda = Ref("std.mixers.kda@1", heads=f["heads"], head_dim=f["head_dim"], conv_kernel=f["conv_kernel"],
              lower_bound=f["lower_bound"], norm_eps=eps, rows=rows)
    dsa = Ref("std.mixers.dsa_mla@1", heads=f["heads"], q_lora=f["q_lora"], kv_lora=f["kv_lora"], nope=f["nope"],
              v=f["v"], index_heads=f["index_heads"], index_head_dim=f["index_head_dim"], ratio=f["ratio"],
              budget=f["budget"], norm=norm, index_eps=f["index_eps"], rows=DSA_ROWS)
    swiglu = Ref("std.ffn.swiglu@1", limit=f["limit"], rows=rows)
    moe = Ref("std.moe@1", experts=f["experts"], top_k=f["top_k"], inter=f["inter"], renormalize=f["renormalize"],
              scoring="sigmoid", bias=True, scale=f["scale"], limit=f["limit"])
    routed = Ref("std.ffn.sum@1", parts=(moe, swiglu.with_(inter=f["inter"] * f["shared"])))
    dense = swiglu.with_(inter=f["dense_inter"])
    layers = tuple(LayerDecl(dsa if f["dsa"][i] else kda, dense if i < f["dense"] else routed,
                             "indexed_attention" if f["dsa"][i] else "linear_attention") for i in range(f["layers"]))
    n = f["streams"]
    return ModelSpec(
        hidden=hidden, vocab=c["vocab_size"], layers=layers,
        block=Ref("std.hyper.mhc_block@1", n=n, norms=norm, eps=eps, hc_eps=f["hc_eps"], iters=f["iters"]),
        embed=Ref("std.hyper.expand@1", inner=Ref("std.tiered.rows@1", name="token_embd"), n=n),
        norm=Ref("std.hyper.mean@1", inner=norm, n=n),
        lm_head=Ref("std.ops.lm_head@1", tied=False), residual=Residual(n * hidden, "float32"),
        weights=WeightMap("gguf", rules=RULES, skip=f["skip"]))
