"""Qwen3 MoE: a pure recipe (imports only holon.api)."""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from holon.api import LayerDecl, ModelSpec, Ref, WeightMap, WeightRule

HF_WEIGHTS = WeightMap("hf_safetensors", rules=(  # explicit MoE rules: `mlp.{rest}` would send the router to `gate`
    WeightRule("model.embed_tokens.weight", "embed.weight"),
    WeightRule("model.norm.weight", "norm.weight"),
    WeightRule("lm_head.weight", "lm_head.weight"),
    WeightRule("model.layers.{i}.input_layernorm.{rest}", "layers.{i}.input_norm.{rest}"),
    WeightRule("model.layers.{i}.post_attention_layernorm.{rest}", "layers.{i}.post_norm.{rest}"),
    WeightRule("model.layers.{i}.self_attn.{rest}", "layers.{i}.mixer.{rest}"),
    WeightRule("model.layers.{i}.mlp.gate.weight", "layers.{i}.ffn.router"),
    *(WeightRule(f"model.layers.{{i}}.mlp.experts.{{e}}.{g}_proj.weight", f"layers.{{i}}.ffn.{g}.weight", stack="e")
      for g in ("gate", "up", "down")),
))
GGUF_WEIGHTS = WeightMap("gguf", rules=(        # llama.cpp names; its Qwen3 converter keeps HF's tensor orientation
    WeightRule("token_embd.weight", "embed.weight"),
    WeightRule("output_norm.weight", "norm.weight"),
    WeightRule("output.weight", "lm_head.weight"),
    WeightRule("blk.{i}.attn_norm.weight", "layers.{i}.input_norm.weight"),
    WeightRule("blk.{i}.ffn_norm.weight", "layers.{i}.post_norm.weight"),
    *(WeightRule(f"blk.{{i}}.attn_{g}.weight", f"layers.{{i}}.mixer.{h}.weight")
      for g, h in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"), ("output", "o_proj"),
                   ("q_norm", "q_norm"), ("k_norm", "k_norm"))),
    WeightRule("blk.{i}.ffn_gate_inp.weight", "layers.{i}.ffn.router"),
    *(WeightRule(f"blk.{{i}}.ffn_{g}_exps.weight", f"layers.{{i}}.ffn.{g}.weight") for g in ("gate", "up", "down")),
))


def rope_theta(c: Mapping[str, Any]) -> float:
    """Plain RoPE only; accepts both config styles (`rope_theta` + `rope_scaling`, or `rope_parameters`)."""
    params = c.get("rope_parameters") or {"rope_theta": c["rope_theta"], **(c.get("rope_scaling") or {})}
    kind = params.get("rope_type", params.get("type", "default"))
    if kind != "default":
        raise NotImplementedError(f"qwen3_moe: rope type {kind!r} not supported yet")
    return params["rope_theta"]


def moe_facts(c: Mapping[str, Any], source: str) -> dict[str, Any]:
    """The MoE ffn's facts: a GGUF file's raw `qwen3moe.*` keys (llama.cpp hard-codes renormalisation for qwen3moe,
    and so does this), or an HF config's (transformers 5 writes `num_experts` as `num_local_experts`)."""
    if source == "gguf":
        g = c["gguf"]
        return {"experts": g["expert_count"], "top_k": g["expert_used_count"],
                "inter": g["expert_feed_forward_length"], "renormalize": True}
    if c.get("mlp_only_layers") or c.get("decoder_sparse_step", 1) != 1:
        raise NotImplementedError(f"qwen3_moe: dense layers among the MoE ones are not supported yet "
                                  f"(mlp_only_layers={c.get('mlp_only_layers')}, "
                                  f"decoder_sparse_step={c.get('decoder_sparse_step')})")
    return {"experts": c.get("num_experts") or c["num_local_experts"], "top_k": c["num_experts_per_tok"],
            "inter": c["moe_intermediate_size"], "renormalize": c["norm_topk_prob"]}


def recipe(c: Mapping[str, Any], source: str) -> ModelSpec:
    if source not in ("hf_safetensors", "gguf"):
        raise ValueError(f"qwen3_moe: no weight map for source {source!r}")
    unsupported = {k: c.get(k) for k in ("attention_bias", "use_sliding_window") if c.get(k)}
    if unsupported:
        raise NotImplementedError(f"qwen3_moe: config options not supported yet: {unsupported}")
    norm = Ref("std.ops.rmsnorm@1", eps=c["rms_norm_eps"])
    attn = Ref("std.mixers.gqa@1", heads=c["num_attention_heads"], kv_heads=c["num_key_value_heads"],
               head_dim=c["head_dim"], qk_norm=norm, rope=Ref("std.ops.rope@1", theta=rope_theta(c)))
    mlp = Ref("std.moe@1", **moe_facts(c, source))
    tied = c.get("tie_word_embeddings", False)
    return ModelSpec(
        hidden=c["hidden_size"], vocab=c["vocab_size"],
        layers=tuple(LayerDecl(attn, mlp, "full_attention") for _ in range(c["num_hidden_layers"])),
        block=Ref("std.blocks.prenorm@1"), embed=Ref("std.ops.embedding@1"), norm=norm,
        lm_head=Ref("std.ops.lm_head@1", tied=tied),
        weights=(GGUF_WEIGHTS if source == "gguf" else
                 dataclasses.replace(HF_WEIGHTS, skip=("lm_head.",)) if tied else HF_WEIGHTS),
    )
