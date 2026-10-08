"""GGUF files on the judge's side (M0.1 design §4.6): the dense HF model a file stands for, built with gguf-py's own
dequantization through the judge's own GGUF → HF name map (never the engine's loader or weight map), and the
per-tensor gate that checks what both maps believe (orientation, no Q/K permutation).
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re

import gguf
import numpy as np
import torch
import transformers

GATE = 0.5          # NMSE(dequantized, HF bf16) per tensor: a correct quant explains at least half the energy
# llama.cpp's tensor names (src/llama-arch.cpp) for the dense-transformer tensors -> HF Qwen3 names.
_GLOBAL = {"token_embd.weight": "model.embed_tokens.weight", "output_norm.weight": "model.norm.weight",
           "output.weight": "lm_head.weight"}
_LAYER = {"attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm", "attn_q": "self_attn.q_proj",
          "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
          "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm", "ffn_gate": "mlp.gate_proj",
          "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}
NAME_MAP_HASH = hashlib.sha256(json.dumps([_GLOBAL, _LAYER], sort_keys=True).encode()).hexdigest()[:16]


def hf_name(name: str) -> str:
    if name in _GLOBAL:
        return _GLOBAL[name]
    m = re.fullmatch(r"blk\.(\d+)\.(\w+)\.weight", name)
    if m is None or m[2] not in _LAYER:
        raise KeyError(f"no HF name for GGUF tensor {name!r}")
    return f"model.layers.{m[1]}.{_LAYER[m[2]]}.weight"


def dequantized(path: str | pathlib.Path) -> dict[str, torch.Tensor]:
    """Every tensor of the file, dequantized by gguf-py (fp32), by HF name."""
    out = {}
    for t in gguf.GGUFReader(path).tensors:
        x = gguf.quants.dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32, copy=False)
        out[hf_name(t.name)] = torch.from_numpy(x.reshape([int(n) for n in reversed(t.shape)]).copy())
    return out


def hf_model(path: str | pathlib.Path, config_dir: str | pathlib.Path, dtype: torch.dtype = torch.float32,
             device: str = "cpu") -> torch.nn.Module:
    """The HF model of `config_dir`'s config with the file's dequantized weights (the dense model the file means)."""
    config = transformers.AutoConfig.from_pretrained(config_dir)
    with torch.device("meta"):
        model = transformers.AutoModelForCausalLM.from_config(config, attn_implementation="sdpa")
    weights = dequantized(path)
    if config.tie_word_embeddings:
        weights.pop("lm_head.weight", None)
    model.load_state_dict({k: v.to(dtype) for k, v in weights.items()}, strict=not config.tie_word_embeddings,
                          assign=True)
    if config.tie_word_embeddings:
        model.tie_weights()
    model.model.rotary_emb = type(model.model.rotary_emb)(config=config)    # its buffers are not weights: built here
    return model.to(device).eval()


def gate(path: str | pathlib.Path, hf_dir: str | pathlib.Path) -> dict[str, float]:
    """NMSE of every dequantized tensor against the HF checkpoint's (bf16) tensor of the same name."""
    from safetensors import safe_open

    reference = {}
    for file in sorted(pathlib.Path(hf_dir).glob("*.safetensors")):
        with safe_open(file, "pt") as f:
            reference |= {k: f.get_tensor(k) for k in f.keys()}
    out = {}
    for name, x in dequantized(path).items():
        ref = reference[name].float()
        out[name] = ((x - ref).pow(2).sum() / ref.pow(2).sum()).item()
    return out


def file_meta(path: str | pathlib.Path) -> dict:
    """Provenance without local paths: the file (hub repo and stem for a hub-cache file), its sha256 (the blob name
    for a hub-cache symlink), the gguf-py that dequantized it and the name map's hash."""
    from importlib.metadata import version

    p = pathlib.Path(path)
    hub = [part for part in p.parts if part.startswith("models--")]
    model = f"{hub[-1].removeprefix('models--').replace('--', '/')}/{p.stem}" if hub else p.stem
    if p.is_symlink() and re.fullmatch(r"[0-9a-f]{64}", p.resolve().name):
        digest = p.resolve().name
    else:
        with open(p, "rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
    return {"model": model, "gguf_sha256": digest, "gguf_py": version("gguf"), "name_map": NAME_MAP_HASH}
