"""Tiny GGUF files for tests: an HF Qwen3 or Qwen3-MoE checkpoint written with gguf-py's writer, as llama.cpp's
converter would (its tensor names, NEOX order kept, F32 norms and router, expert tensors stacked [E, out, in])."""
from __future__ import annotations

import json
import pathlib
import re

import gguf
import numpy as np
from safetensors.torch import load_file

NAMES = {"model.embed_tokens.weight": "token_embd.weight", "lm_head.weight": "output.weight",  # HF -> llama.cpp
         "model.norm.weight": "output_norm.weight"}
LAYER = {"input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm", "self_attn.q_proj": "attn_q",
         "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
         "self_attn.q_norm": "attn_q_norm", "self_attn.k_norm": "attn_k_norm", "mlp.gate_proj": "ffn_gate",
         "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down"}


EXPERT = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight")


def gguf_name(hf: str) -> str:
    if hf in NAMES:
        return NAMES[hf]
    _, _, i, rest = hf.split(".", 3)
    if rest == "mlp.gate.weight":
        return f"blk.{i}.ffn_gate_inp.weight"
    return f"blk.{i}.{LAYER[rest.removesuffix('.weight')]}.weight"


def _tensors(checkpoint: pathlib.Path) -> dict[str, tuple[np.ndarray, list[str]]]:
    """The checkpoint's tensors by GGUF name, with the HF names they come from: per-expert tensors stacked into
    llama.cpp's [E, out, in] (HF names in expert order)."""
    out, experts = {}, {}
    for name, t in load_file(checkpoint / "model.safetensors").items():
        m = EXPERT.fullmatch(name)
        if m:
            experts.setdefault(f"blk.{m[1]}.ffn_{m[3]}_exps.weight", {})[int(m[2])] = (name, t.float().numpy())
        else:
            out[gguf_name(name)] = (t.float().numpy(), [name])
    for name, by_e in experts.items():
        ordered = [by_e[e] for e in range(len(by_e))]
        out[name] = (np.stack([x for _, x in ordered]), [n for n, _ in ordered])
    return out


def write(checkpoint: pathlib.Path, path: pathlib.Path, qtype: gguf.GGMLQuantizationType,
          quantize_norms: bool = False, types: dict[str, gguf.GGMLQuantizationType] | None = None,
          split: int = 0) -> dict[str, np.ndarray]:
    """The checkpoint as a GGUF file (matrices and expert stacks in `qtype`, or `types[<GGUF name part>]`, e.g.
    `{"ffn_up_exps": Q4_0}`; norms and the router F32 unless `quantize_norms`; with `split`, as files of that many
    tensors, `<stem>-00001-of-0000N.gguf` …); returns the dequantized tensors by HF name."""
    cfg = json.loads((checkpoint / "config.json").read_text())
    moe = cfg["model_type"] == "qwen3_moe"
    w = gguf.GGUFWriter(path, "qwen3moe" if moe else "qwen3", split_max_tensors=split)
    w.add_block_count(cfg["num_hidden_layers"])
    w.add_context_length(cfg["max_position_embeddings"])
    w.add_embedding_length(cfg["hidden_size"])
    w.add_feed_forward_length(cfg["intermediate_size"])
    w.add_head_count(cfg["num_attention_heads"])
    w.add_head_count_kv(cfg["num_key_value_heads"])
    w.add_key_length(cfg["head_dim"])
    w.add_value_length(cfg["head_dim"])
    w.add_rope_freq_base(cfg.get("rope_theta") or cfg["rope_parameters"]["rope_theta"])
    w.add_layer_norm_rms_eps(cfg["rms_norm_eps"])
    if moe:
        w.add_expert_count(cfg.get("num_experts") or cfg["num_local_experts"])   # transformers 5 renamed it
        w.add_expert_used_count(cfg["num_experts_per_tok"])
        w.add_expert_feed_forward_length(cfg["moe_intermediate_size"])
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("qwen2")
    w.add_token_list([f"t{i}" for i in range(cfg["vocab_size"])])
    w.add_token_types([1] * cfg["vocab_size"])
    w.add_token_merges([])
    dequantized = {}
    for name, (x, hf) in _tensors(checkpoint).items():
        if (x.ndim >= 2 and not name.endswith("ffn_gate_inp.weight")) or (quantize_norms and x.shape[-1] % 32 == 0):
            kind = next((t for part, t in (types or {}).items() if f".{part}." in name), qtype)
            blocks = gguf.quants.quantize(x.reshape(-1, x.shape[-1]), kind)
            w.add_tensor(name, blocks.reshape(*x.shape[:-1], blocks.shape[-1]), raw_dtype=kind)
            x = gguf.quants.dequantize(blocks, kind).reshape(x.shape)
        else:
            w.add_tensor(name, x)
        dequantized |= dict(zip(hf, x if len(hf) > 1 else [x], strict=True))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return dequantized
