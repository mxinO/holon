"""Tiny GGUF files for tests: an HF Qwen3 checkpoint written with gguf-py's writer, as llama.cpp's converter would
(its tensor names, NEOX order kept, F32 norms)."""
from __future__ import annotations

import json
import pathlib

import gguf
import numpy as np
from safetensors.torch import load_file

NAMES = {"model.embed_tokens.weight": "token_embd.weight", "lm_head.weight": "output.weight",  # HF -> llama.cpp
         "model.norm.weight": "output_norm.weight"}
LAYER = {"input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm", "self_attn.q_proj": "attn_q",
         "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
         "self_attn.q_norm": "attn_q_norm", "self_attn.k_norm": "attn_k_norm", "mlp.gate_proj": "ffn_gate",
         "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down"}


def gguf_name(hf: str) -> str:
    if hf in NAMES:
        return NAMES[hf]
    _, _, i, rest = hf.split(".", 3)
    return f"blk.{i}.{LAYER[rest.removesuffix('.weight')]}.weight"


def write(checkpoint: pathlib.Path, path: pathlib.Path, qtype: gguf.GGMLQuantizationType,
          quantize_norms: bool = False) -> dict[str, np.ndarray]:
    """The checkpoint as a GGUF file (matrices in `qtype`, norms F32 unless `quantize_norms`); returns the
    dequantized tensors by HF name."""
    cfg = json.loads((checkpoint / "config.json").read_text())
    w = gguf.GGUFWriter(path, "qwen3")
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
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("qwen2")
    w.add_token_list([f"t{i}" for i in range(cfg["vocab_size"])])
    w.add_token_types([1] * cfg["vocab_size"])
    w.add_token_merges([])
    dequantized = {}
    for name, t in load_file(checkpoint / "model.safetensors").items():
        x = t.float().numpy()
        if x.ndim == 2 or (quantize_norms and x.shape[-1] % 32 == 0):
            blocks = gguf.quants.quantize(x.reshape(-1, x.shape[-1]), qtype)
            w.add_tensor(gguf_name(name), blocks, raw_dtype=qtype)
            x = gguf.quants.dequantize(blocks, qtype).reshape(x.shape)
        else:
            w.add_tensor(gguf_name(name), x)
        dequantized[name] = x
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return dequantized
