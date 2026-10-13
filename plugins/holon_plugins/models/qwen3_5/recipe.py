"""Qwen3.5 / Qwen3.8 text model: a pure recipe (imports only holon.api).

Every `full_attention_interval`-th layer is full attention (output-gated, partial rotary), the rest Gated DeltaNet
(docs/m03-qwen38.md §2, §4.4). The HF checkpoint stores zero-centred norms (`x̂·(1 + w)`); llama.cpp's converter writes
`w + 1` in f32 and `ssm_a = −exp(A_log)`, both undone here by value transforms, and orders value heads tiled, which
the GDN kernels read as a fact. The vision tower is skipped; the MTP block is too, except in the draft head
`draft_head("mtp")` returns (docs/m04-spec.md D2): the checkpoint's multi-token-prediction layer as a
`std.spec.mtp_head@1` over the trunk's own attention and MLP Refs, built only when a drafter asks for it.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from holon.api import AuxSpec, LayerDecl, ModelSpec, Ref, WeightMap, WeightRule

_P = "model.language_model."
HF_WEIGHTS = WeightMap("hf_safetensors", skip=("model.visual.", "mtp."), rules=(
    WeightRule(_P + "embed_tokens.weight", "embed.weight"),
    WeightRule(_P + "norm.weight", "norm.weight"),
    WeightRule("lm_head.weight", "lm_head.weight"),
    WeightRule(_P + "layers.{i}.input_layernorm.weight", "layers.{i}.input_norm.weight"),
    WeightRule(_P + "layers.{i}.post_attention_layernorm.weight", "layers.{i}.post_norm.weight"),
    WeightRule(_P + "layers.{i}.self_attn.{rest}", "layers.{i}.mixer.{rest}"),
    WeightRule(_P + "layers.{i}.linear_attn.conv1d.weight", "layers.{i}.mixer.conv_weight", "qwen3_5.squeeze"),
    WeightRule(_P + "layers.{i}.linear_attn.norm.weight", "layers.{i}.mixer.norm_weight"),
    WeightRule(_P + "layers.{i}.linear_attn.{rest}", "layers.{i}.mixer.{rest}"),
    WeightRule(_P + "layers.{i}.mlp.{rest}", "layers.{i}.ffn.{rest}"),
))
_NORMS = (("attn_norm", "input_norm"), ("post_attention_norm", "post_norm"))
_ATTN = (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"), ("output", "o_proj"))
_GDN = (("attn_qkv", "in_proj_qkv"), ("attn_gate", "in_proj_z"), ("ssm_beta", "in_proj_b"), ("ssm_alpha", "in_proj_a"),
        ("ssm_out", "out_proj"))
GGUF_RULES = (
    WeightRule("token_embd.weight", "embed.weight"),
    WeightRule("output_norm.weight", "norm.weight", "qwen3_5.minus_one"),
    WeightRule("output.weight", "lm_head.weight"),
    *(WeightRule(f"blk.{{i}}.{g}.weight", f"layers.{{i}}.{h}.weight", "qwen3_5.minus_one") for g, h in _NORMS),
    *(WeightRule(f"blk.{{i}}.attn_{g}.weight", f"layers.{{i}}.mixer.{h}.weight") for g, h in _ATTN),
    *(WeightRule(f"blk.{{i}}.attn_{g}_norm.weight", f"layers.{{i}}.mixer.{g}_norm.weight", "qwen3_5.minus_one")
      for g in ("q", "k")),
    *(WeightRule(f"blk.{{i}}.{g}.weight", f"layers.{{i}}.mixer.{h}.weight") for g, h in _GDN),
    WeightRule("blk.{i}.ssm_conv1d.weight", "layers.{i}.mixer.conv_weight"),
    WeightRule("blk.{i}.ssm_norm.weight", "layers.{i}.mixer.norm_weight"),
    WeightRule("blk.{i}.ssm_a", "layers.{i}.mixer.A_log", "qwen3_5.a_log"),
    WeightRule("blk.{i}.ssm_dt.bias", "layers.{i}.mixer.dt_bias"),
    *(WeightRule(f"blk.{{i}}.ffn_{g}.weight", f"layers.{{i}}.ffn.{g}_proj.weight") for g in ("gate", "up", "down")),
)


def _facts(c: Mapping[str, Any], source: str) -> dict[str, Any]:
    """Everything the recipe needs, read from an HF text config or a GGUF file's `qwen35.*` keys."""
    if source == "gguf":
        g = c["gguf"]
        n = c["num_hidden_layers"] - g.get("nextn_predict_layers", 0)        # block_count includes MTP blocks
        recurrent = g.get("attention.recurrent_layers")        # per layer, when the converter wrote it
        full = ([not r for r in recurrent[:n]] if recurrent is not None else
                [(i + 1) % g["full_attention_interval"] == 0 for i in range(n)])
        hv = g["ssm.time_step_rank"]
        return {"layers": n, "full": full, "heads_k": g["ssm.group_count"],
                "heads_v": hv, "head_k_dim": g["ssm.state_size"], "head_v_dim": g["ssm.inner_size"] // hv,
                "conv_kernel": g["ssm.conv_kernel"], "rotary_dim": g["rope.dimension_count"],
                "sections": list(g.get("rope.dimension_sections", []))[:3], "theta": c["rope_theta"],
                "order": "tiled", "skip": tuple(f"blk.{i}." for i in range(n, c["num_hidden_layers"]))}
    t = c.get("text_config", c)
    if t.get("output_gate_type", "swish") not in ("swish", "silu") or not t.get("attn_output_gate", True):
        raise NotImplementedError(f"qwen3_5: output gates {t.get('output_gate_type')!r} / "
                                  f"{t.get('attn_output_gate')!r} not supported")
    rope = t["rope_parameters"]
    if rope.get("rope_type", "default") != "default":
        raise NotImplementedError(f"qwen3_5: rope type {rope.get('rope_type')!r} not supported yet")
    return {"layers": t["num_hidden_layers"], "full": [k == "full_attention" for k in t["layer_types"]],
            "heads_k": t["linear_num_key_heads"], "heads_v": t["linear_num_value_heads"],
            "head_k_dim": t["linear_key_head_dim"], "head_v_dim": t["linear_value_head_dim"],
            "conv_kernel": t["linear_conv_kernel_dim"],
            "rotary_dim": int(t["head_dim"] * rope["partial_rotary_factor"]),
            "sections": rope.get("mrope_section", []), "theta": rope["rope_theta"], "order": "grouped", "skip": ()}


def _refs(c: Mapping[str, Any], source: str) -> dict[str, Any]:
    """The recipe's facts and the Refs the trunk and the draft head share."""
    t = c if source == "gguf" else c.get("text_config", c)
    f = _facts(c, source)
    if f["sections"] and sum(f["sections"]) * 2 != f["rotary_dim"]:
        raise NotImplementedError(f"qwen3_5: mrope sections {f['sections']} do not cover the rotary dims")
    # mrope over (T, H, W) with T = H = W = position, as for text, is plain 1-D rope over the rotary dims.
    norm = Ref("std.ops.rmsnorm@1", eps=t["rms_norm_eps"], offset=1.0)       # zero-centred: x̂·(1 + w)
    attn = Ref("std.mixers.gqa@1", heads=t["num_attention_heads"], kv_heads=t["num_key_value_heads"],
               head_dim=t["head_dim"], qk_norm=norm, output_gate="sigmoid",
               rope=Ref("std.ops.rope@1", theta=f["theta"], rotary_dim=f["rotary_dim"]))
    gdn = Ref("std.mixers.gdn@1", heads_k=f["heads_k"], heads_v=f["heads_v"], head_k_dim=f["head_k_dim"],
              head_v_dim=f["head_v_dim"], conv_kernel=f["conv_kernel"], v_head_order=f["order"],
              norm_eps=t["rms_norm_eps"])
    mlp = Ref("std.ffn.swiglu@1", inter=t["intermediate_size"])
    return {"t": t, "f": f, "norm": norm, "attn": attn, "gdn": gdn, "mlp": mlp}


def recipe(c: Mapping[str, Any], source: str) -> ModelSpec:
    if source not in ("hf_safetensors", "gguf"):
        raise ValueError(f"qwen3_5: no weight map for source {source!r}")
    r = _refs(c, source)
    t, f = r["t"], r["f"]
    tied = t.get("tie_word_embeddings", c.get("tie_word_embeddings", False))
    if source == "gguf":
        weights = WeightMap("gguf", rules=GGUF_RULES, skip=f["skip"])
    else:
        weights = WeightMap("hf_safetensors", rules=HF_WEIGHTS.rules,
                            skip=HF_WEIGHTS.skip + (("lm_head.",) if tied else ()))
    return ModelSpec(
        hidden=t["hidden_size"], vocab=t["vocab_size"],
        layers=tuple(LayerDecl(r["attn"] if full else r["gdn"], r["mlp"],
                               "full_attention" if full else "linear_attention") for full in f["full"]),
        block=Ref("std.blocks.prenorm@1"), embed=Ref("std.ops.embedding@1"), norm=r["norm"],
        lm_head=Ref("std.ops.lm_head@1", tied=tied), weights=weights,
    )


# The MTP block: HF `mtp.*`; GGUF `blk.<n>.*` after the trunk's n blocks (`nextn.*` for the head's own tensors).
_MTP_HF = (
    WeightRule("mtp.fc.weight", "fc.weight"),
    WeightRule("mtp.pre_fc_norm_embedding.weight", "enorm.weight"),
    WeightRule("mtp.pre_fc_norm_hidden.weight", "hnorm.weight"),
    WeightRule("mtp.norm.weight", "norm.weight"),
    WeightRule("mtp.layers.{i}.input_layernorm.weight", "layers.{i}.input_norm.weight"),
    WeightRule("mtp.layers.{i}.post_attention_layernorm.weight", "layers.{i}.post_norm.weight"),
    WeightRule("mtp.layers.{i}.self_attn.{rest}", "layers.{i}.mixer.{rest}"),
    WeightRule("mtp.layers.{i}.mlp.{rest}", "layers.{i}.ffn.{rest}"),
)


def _mtp_gguf(n: int) -> tuple[WeightRule, ...]:
    """Block n (the first after the trunk) as layer 0 of the head; norms stored as w + 1."""
    b, minus = f"blk.{n}.", "qwen3_5.minus_one"
    return (
        WeightRule(b + "nextn.eh_proj.weight", "fc.weight"),
        WeightRule(b + "nextn.enorm.weight", "enorm.weight", minus),
        WeightRule(b + "nextn.hnorm.weight", "hnorm.weight", minus),
        WeightRule(b + "nextn.shared_head_norm.weight", "norm.weight", minus),
        *(WeightRule(f"{b}{g}.weight", f"layers.0.{h}.weight", minus) for g, h in _NORMS),
        *(WeightRule(f"{b}attn_{g}.weight", f"layers.0.mixer.{h}.weight") for g, h in _ATTN),
        *(WeightRule(f"{b}attn_{g}_norm.weight", f"layers.0.mixer.{g}_norm.weight", minus) for g in ("q", "k")),
        *(WeightRule(f"{b}ffn_{g}.weight", f"layers.0.ffn.{g}_proj.weight") for g in ("gate", "up", "down")),
    )


def draft_head(c: Mapping[str, Any], source: str, name: str) -> AuxSpec | None:
    """The checkpoint's MTP layer as a draft head (None if it has none, or for any other name)."""
    if name != "mtp" or source not in ("hf_safetensors", "gguf"):
        return None
    r = _refs(c, source)
    t, f = r["t"], r["f"]
    if source == "gguf":
        nextn = c["gguf"].get("nextn_predict_layers", 0)
        if nextn != 1:
            return None
        n = f["layers"]
        weights = WeightMap("gguf", rules=_mtp_gguf(n), only=(f"blk.{n}.",),
                            skip=(f"blk.{n}.nextn.embed_tokens.", f"blk.{n}.nextn.shared_head_head."))
    else:
        if t.get("mtp_num_hidden_layers", 0) != 1:
            return None
        weights = WeightMap("hf_safetensors", rules=_MTP_HF, only=("mtp.",))
    head = Ref("std.spec.mtp_head@1", hidden=t["hidden_size"], block=Ref("std.blocks.prenorm@1"), norm=r["norm"],
               order="embed_first")
    return AuxSpec(head, (LayerDecl(r["attn"], r["mlp"], "full_attention"),), weights)
