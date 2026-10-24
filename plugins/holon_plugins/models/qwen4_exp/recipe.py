"""Qwen3.8-Flash-Next (`qwen4_exp`; llama.cpp's `qwen4exp` GGUF): a pure recipe (imports only holon.api).

The trunk carries R, four float32 streams of the hidden size (`ModelSpec.residual`, D3): `std.hyper.expand@1` around
the embedding, a `std.hyper.gr_block@1` per layer, `std.hyper.gr_mix@1` as the norm (the output mixer). Every
`full_attention_interval`-th layer is QSA (`std.mixers.qsa@1`, D6), the rest Gated DeltaNet with the sigmoid output
gate (D8); every FFN the routed experts plus the gated shared expert (`std.ffn.sum@1`, D7). PLE (`qwen4_exp.ple@1`,
D4) is the block's `pre` at its layer, its state on GDN's 64-token grid.

GGUF only (D1: the deployed files). llama.cpp's converter writes every zero-centred norm as w + 1 in f32 (the HC,
PLE, indexer and q / k norms; not GDN's gated norm) and `ssm_a = −exp(A_log)`, both undone by value transforms
(`plugin.py`), and orders GDN's value heads tiled. The shared expert's gate is [H] in the deployed files and [1, H]
in the tiny one (converter versions differ): a row either way (`qwen4_exp.row`). The MTP block (`blk.<n>` after the
trunk's n) is skipped here.

`draft_head("mtp")` (D9) is `std.spec.mtp_hc_head@1` over one QSA + MoE + shared-expert layer, its weights mapped
from Strata's MTP pack (`mtp-q2_0.gguf`, the drafter's `mtp.path`): HF names under `mtp.`, norms stored as w, the
indexer's q / k projection and each expert's gate / up rows fused as transformers stores them.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from holon.api import AuxSpec, LayerDecl, ModelSpec, Ref, Residual, WeightMap, WeightRule

MINUS = "qwen4_exp.minus_one"
_HC = (("norm", "norm", MINUS), ("down", "down", None), ("up", "up", None), ("inject", "inject", None))
_GDN = (("attn_qkv", "in_proj_qkv"), ("attn_gate", "in_proj_z"), ("ssm_beta", "in_proj_b"),
        ("ssm_alpha", "in_proj_a"), ("ssm_out", "out_proj"))
_ATTN = (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"), ("output", "o_proj"))
RULES = (
    WeightRule("token_embd.weight", "embed.inner.weight"),
    WeightRule("output.weight", "lm_head.weight"),
    *(WeightRule(f"output_hc_{g}.weight", f"norm.hc.{h}", t) for g, h, t in _HC[:3]),
    *(WeightRule(f"blk.{{i}}.hc_{half}_{g}.weight", f"layers.{{i}}.{half}_hc.{h}", t)
      for half in ("attn", "ffn") for g, h, t in _HC),
    *(WeightRule(f"blk.{{i}}.{g}.weight", f"layers.{{i}}.mixer.{h}.weight") for g, h in _GDN),
    WeightRule("blk.{i}.ssm_conv1d.weight", "layers.{i}.mixer.conv_weight"),
    WeightRule("blk.{i}.ssm_norm.weight", "layers.{i}.mixer.norm_weight"),
    WeightRule("blk.{i}.ssm_a", "layers.{i}.mixer.A_log", "qwen4_exp.a_log"),
    WeightRule("blk.{i}.ssm_dt.bias", "layers.{i}.mixer.dt_bias"),
    *(WeightRule(f"blk.{{i}}.attn_{g}.weight", f"layers.{{i}}.mixer.attn.{h}.weight") for g, h in _ATTN),
    *(WeightRule(f"blk.{{i}}.attn_{g}_norm.weight", f"layers.{{i}}.mixer.attn.{g}_norm.weight", MINUS)
      for g in ("q", "k")),
    *(WeightRule(f"blk.{{i}}.indexer.{g}_proj.weight", f"layers.{{i}}.mixer.index_{g}_proj.weight") for g in "qk"),
    WeightRule("blk.{i}.indexer.q_norm.weight", "layers.{i}.mixer.index_q_norm.weight", MINUS),
    WeightRule("blk.{i}.indexer.k_norm.weight", "layers.{i}.mixer.index_k_norm", MINUS),
    WeightRule("blk.{i}.ffn_gate_inp.weight", "layers.{i}.ffn.parts.0.router"),
    *(WeightRule(f"blk.{{i}}.ffn_{g}_exps.weight", f"layers.{{i}}.ffn.parts.0.{g}.weight")
      for g in ("gate", "up", "down")),
    *(WeightRule(f"blk.{{i}}.ffn_{g}_shexp.weight", f"layers.{{i}}.ffn.parts.1.{g}_proj.weight")
      for g in ("gate", "up", "down")),
    WeightRule("blk.{i}.ffn_gate_inp_shexp.weight", "layers.{i}.ffn.parts.1.gate.weight", "qwen4_exp.row"),
    WeightRule("blk.{i}.ple_key.weight", "layers.{i}.pre.key_proj.weight"),
    WeightRule("blk.{i}.ple_value.weight", "layers.{i}.pre.value_proj.weight"),
    *(WeightRule(f"blk.{{i}}.ple_norm_{g}.weight", f"layers.{{i}}.pre.norm_{g}", MINUS)
      for g in ("key", "query", "conv")),
    WeightRule("blk.{i}.ple_conv1d.weight", "layers.{i}.pre.conv"),
)


def facts(c: Mapping[str, Any]) -> dict[str, Any]:
    """What the recipe reads from a `qwen4exp` GGUF's keys (`c["gguf"]`) and their mapped names."""
    g = c["gguf"]
    n = c["num_hidden_layers"] - g.get("nextn_predict_layers", 0)          # block_count includes MTP blocks
    recurrent = g.get("attention.recurrent_layers")
    full = ([not r for r in recurrent[:n]] if recurrent is not None else
            [(i + 1) % g["full_attention_interval"] == 0 for i in range(n)])
    hv = g["ssm.time_step_rank"]
    ple_heads = (g["ple.ngram_size"] - 1) * g["ple.heads_per_ngram"]
    return {"layers": n, "full": full, "streams": g["hyper_connection.count"], "rank": g["hyper_connection.low_rank"],
            "heads_k": g["ssm.group_count"], "heads_v": hv, "head_k_dim": g["ssm.state_size"],
            "head_v_dim": g["ssm.inner_size"] // hv, "conv_kernel": g["ssm.conv_kernel"],
            "rotary_dim": g["rope.dimension_count"], "index_heads": g["attention.indexer.head_count"],
            "index_head_dim": g["attention.indexer.key_length"], "budget": g["attention.indexer.top_k"],
            "ratio": max(g.get("attention.compress_ratios") or [4]) or 4, "experts": g["expert_count"],
            "top_k": g["expert_used_count"], "inter": g["expert_feed_forward_length"],
            "shared_inter": g["expert_shared_feed_forward_length"], "ple_layers": tuple(g["ple.layers"]),
            "ngram": g["ple.ngram_size"], "heads_per_ngram": g["ple.heads_per_ngram"], "ple_heads": ple_heads,
            "ple_dim": g["embedding_length_per_layer_input"], "ple_kernel": g["ple.conv_kernel"],
            "eos": g["ple.eos_token_id"], "multipliers": tuple(g["ple.layer_multipliers"]),
            "sizes": tuple(g["ple.head_vocab_sizes"]), "offsets": tuple(g["ple.head_offsets"]),
            "skip": tuple(f"blk.{i}." for i in range(n, c["num_hidden_layers"]))}


PAD = 128      # HF's `make_ngram_vocab_size_divisible_by` (Qwen3.8-Flash-Next's): the converter keeps the padded table
_PACK_HC = (("hc_norm", "norm"), ("input_mix_weight_down", "down"), ("input_mix_weight_up", "up"),
            ("block_inject_weight", "inject"))
MTP_RULES = (           # Strata's pack (St tools/mtp_pack.py): HF names, under the WeightMap's root `mtp.`
    *(WeightRule(f"{g}.weight", f"{h}.weight") for g, h in (
        ("fc_embedding", "fc_embedding"), ("fc_hidden", "fc_hidden"), ("pre_fc_norm_embedding", "enorm"),
        ("pre_fc_norm_hidden", "hnorm"))),
    *(WeightRule(f"hyper_connection_mixer.{g}.weight", f"out.hc.{h}") for g, h in _PACK_HC[:3]),
    *(WeightRule(f"layers.{{i}}.{half}_hyper_connection.{g}.weight", f"layers.{{i}}.{ours}_hc.{h}")
      for half, ours in (("attn", "attn"), ("mlp", "ffn")) for g, h in _PACK_HC),
    *(WeightRule(f"layers.{{i}}.self_attn.{g}.weight", f"layers.{{i}}.mixer.attn.{g}.weight")
      for g in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")),
    WeightRule("layers.{i}.self_attn.indexer.index_qk_proj.weight", "layers.{i}.mixer.index_qk_proj.weight"),
    WeightRule("layers.{i}.self_attn.indexer.q_layernorm.weight", "layers.{i}.mixer.index_q_norm.weight"),
    WeightRule("layers.{i}.self_attn.indexer.k_layernorm.weight", "layers.{i}.mixer.index_k_norm"),
    WeightRule("layers.{i}.mlp.gate.weight", "layers.{i}.ffn.parts.0.router"),
    WeightRule("layers.{i}.mlp.experts.gate_up_proj", "layers.{i}.ffn.parts.0.gate_up.weight"),
    WeightRule("layers.{i}.mlp.experts.down_proj", "layers.{i}.ffn.parts.0.down.weight"),
    *(WeightRule(f"layers.{{i}}.mlp.shared_expert.{g}_proj.weight", f"layers.{{i}}.ffn.parts.1.{g}_proj.weight")
      for g in ("gate", "up", "down")),
    WeightRule("layers.{i}.mlp.shared_expert_gate.weight", "layers.{i}.ffn.parts.1.gate.weight"),
)


def _refs(c: Mapping[str, Any], f: dict[str, Any], packed: bool = False) -> dict[str, Ref]:
    """The trunk's norm, QSA and FFN; `packed`: as the MTP pack stores them (fused indexer q / k, fused gate / up)."""
    eps = c["rms_norm_eps"]
    norm = Ref("std.ops.rmsnorm@1", eps=eps, offset=1.0)                       # zero-centred: x̂·(1 + w)
    rope = Ref("std.ops.rope@1", theta=c["rope_theta"], rotary_dim=f["rotary_dim"])
    qsa = Ref("std.mixers.qsa@1", heads=c["num_attention_heads"], kv_heads=c["num_key_value_heads"],
              head_dim=c["head_dim"], qk_norm=norm, output_gate="sigmoid", rope=rope, index_heads=f["index_heads"],
              index_head_dim=f["index_head_dim"], ratio=f["ratio"], budget=f["budget"], eps=eps)
    moe = Ref("std.moe@1", experts=f["experts"], top_k=f["top_k"], inter=f["inter"], renormalize=True)
    if packed:
        qsa, moe = qsa.with_(fused_index_qk=True), moe.with_(fused_gate_up=True)
    ffn = Ref("std.ffn.sum@1", parts=(moe, Ref("std.ffn.gated_swiglu@1", inter=f["shared_inter"])))
    return {"norm": norm, "qsa": qsa, "ffn": ffn}


def draft_head(c: Mapping[str, Any], source: str, name: str) -> AuxSpec | None:
    """The MTP head (D9), its weights Strata's pack's (None for any other name)."""
    if name != "mtp" or source != "gguf":
        return None
    f, eps = facts(c), c["rms_norm_eps"]
    r, gr = _refs(c, f, packed=True), {"n": f["streams"], "rank": f["rank"], "eps": eps}
    head = Ref("std.spec.mtp_hc_head@1", hidden=c["hidden_size"], n=f["streams"], norm=r["norm"],
               block=Ref("std.hyper.gr_block@1", **gr), out=Ref("std.hyper.gr_mix@1", **gr))
    return AuxSpec(head, (LayerDecl(r["qsa"], r["ffn"], "indexed_attention"),),
                   WeightMap("gguf", rules=MTP_RULES, root="mtp."))


def recipe(c: Mapping[str, Any], source: str) -> ModelSpec:
    if source != "gguf":
        raise ValueError(f"qwen4_exp: GGUF files only (D1), not {source!r}")
    f, eps, hidden = facts(c), c["rms_norm_eps"], c["hidden_size"]
    n, rank = f["streams"], f["rank"]
    r = _refs(c, f)
    qsa, ffn = r["qsa"], r["ffn"]
    gdn = Ref("std.mixers.gdn@1", heads_k=f["heads_k"], heads_v=f["heads_v"], head_k_dim=f["head_k_dim"],
              head_v_dim=f["head_v_dim"], conv_kernel=f["conv_kernel"], v_head_order="tiled", norm_eps=eps,
              out_gate="sigmoid", merge_ab=True)
    ple = Ref("qwen4_exp.ple@1", n=n, width=n * hidden, embed_dim=f["ple_heads"] * f["ple_dim"], ngram=f["ngram"],
              heads_per_ngram=f["heads_per_ngram"], eos=f["eos"], multipliers=f["multipliers"], sizes=f["sizes"],
              offsets=f["offsets"], table_rows=-(-sum(f["sizes"]) // PAD) * PAD, dilation=f["ngram"],
              kernel=f["ple_kernel"], eps=eps, alignment=64, in_step=64)        # GDN's chunk (D4)
    rules = RULES + tuple(WeightRule("per_layer_token_embd.weight", f"layers.{i}.pre.table.weight")
                          for i in f["ple_layers"][:1])
    return ModelSpec(
        hidden=hidden, vocab=c["vocab_size"],
        layers=tuple(LayerDecl(qsa if full else gdn, ffn, "indexed_attention" if full else "linear_attention")
                     for full in f["full"]),
        block=Ref("std.hyper.gr_block@1", n=n, rank=rank, eps=eps, pre=ple, pre_at=f["ple_layers"]),
        embed=Ref("std.hyper.expand@1", inner=Ref("std.tiered.rows@1", name="token_embd"), n=n),
        norm=Ref("std.hyper.gr_mix@1", n=n, rank=rank, eps=eps),
        lm_head=Ref("std.ops.lm_head@1", tied=False), residual=Residual(n * hidden, "float32"),
        weights=WeightMap("gguf", rules=rules, skip=f["skip"]))
