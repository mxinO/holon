"""Tiny DFlash2 draft checkpoints for tests: random weights under the names and `config.json` layout of z-lab's DFlash2
checkpoints (z-lab/Qwen3.8-27B-DFlash2), for a given target's hidden size and vocabulary; and their GGUF as llama.cpp's
own converter writes it."""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import torch
from safetensors.torch import save_file

from .qwen3_5_files import LLAMA, can_convert  # noqa: F401 — re-exported


def config(hidden: int, vocab: int, taps: tuple[int, ...], mask: int, layers: int = 2, window: int = 6,
           heads: tuple[int, int, int] = (4, 2, 16), inter: int = 96, rank: int = 8, top_k: int = 4) -> dict:
    """`heads`: query heads, kv heads, head dim."""
    return {"architectures": ["DFlash2DraftModel"], "model_type": "qwen3", "hidden_size": hidden,
            "intermediate_size": inter, "num_hidden_layers": layers, "num_attention_heads": heads[0],
            "num_key_value_heads": heads[1], "head_dim": heads[2], "rms_norm_eps": 1e-6, "vocab_size": vocab,
            "is_causal": False, "use_sliding_window": True, "sliding_window": window,
            "layer_types": ["sliding_attention"] * layers, "max_window_layers": layers,
            "rope_parameters": {"rope_theta": 1e7, "rope_type": "default"}, "tie_word_embeddings": False,
            "dflash_config": {"block_size": 8, "target_layer_ids": list(taps), "mask_token_id": mask,
                              "conv_kernel_size": 2, "conv_group_size": 16, "selector_rank": rank,
                              "selector_top_k": top_k}}


def save(path: pathlib.Path, hidden: int, vocab: int, taps: tuple[int, ...], mask: int, seed: int = 0,
         **overrides) -> pathlib.Path:
    """A random DFlash2 checkpoint in `path`: weights at scales that keep activations O(1), norms and the conv's base
    kernel near their trained shapes (identity tap 0) with noise, so no part is inert."""
    c = config(hidden, vocab, taps, mask, **overrides)
    d = c["dflash_config"]
    g = torch.Generator().manual_seed(seed)
    rand = lambda *s, scale: torch.randn(*s, generator=g) * scale                       # noqa: E731
    norm = lambda n: 1 + rand(n, scale=0.2)                                               # noqa: E731
    heads, kv, dim, inter = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["intermediate_size"]
    groups = hidden // d["conv_group_size"]
    t = {"fc.weight": rand(hidden, len(taps) * hidden, scale=(len(taps) * hidden) ** -0.5),
         "hidden_norm.weight": norm(hidden), "norm.weight": norm(hidden),
         "candidate_selector.predecessor_codebook": rand(vocab, d["selector_rank"], scale=1.0),
         "candidate_selector.successor_codebook": rand(vocab, d["selector_rank"], scale=1.0),
         "candidate_selector.hidden_projection.weight": rand(d["selector_rank"], hidden, scale=hidden ** -0.5)}
    for i in range(c["num_hidden_layers"]):
        p = f"layers.{i}."
        t |= {p + "input_layernorm.weight": norm(hidden), p + "post_attention_layernorm.weight": norm(hidden),
              p + "self_attn.q_proj.weight": rand(heads * dim, hidden, scale=hidden ** -0.5),
              p + "self_attn.k_proj.weight": rand(kv * dim, hidden, scale=hidden ** -0.5),
              p + "self_attn.v_proj.weight": rand(kv * dim, hidden, scale=hidden ** -0.5),
              p + "self_attn.o_proj.weight": rand(hidden, heads * dim, scale=(heads * dim) ** -0.5),
              p + "self_attn.q_norm.weight": norm(dim), p + "self_attn.k_norm.weight": norm(dim),
              p + "mlp.gate_proj.weight": rand(inter, hidden, scale=hidden ** -0.5),
              p + "mlp.up_proj.weight": rand(inter, hidden, scale=hidden ** -0.5),
              p + "mlp.down_proj.weight": rand(hidden, inter, scale=inter ** -0.5)}
        for conv in ("attention_conv", "mlp_conv"):
            base = rand(2, d["conv_kernel_size"], hidden, scale=0.2)
            base[:, 0] += 1
            t |= {f"{p}{conv}.base_kernel": base,
                  f"{p}{conv}.kernel_projection.weight": rand(2 * d["conv_kernel_size"] * groups, hidden, scale=0.05)}
    path.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in t.items()}, path / "model.safetensors")
    (path / "config.json").write_text(json.dumps(c))
    return path


def convert(path: pathlib.Path, target: pathlib.Path, outtype: str = "q8_0") -> pathlib.Path:
    """`path` converted by llama.cpp's own converter (its DFlash support) to a GGUF of `outtype`, with the tokenizer
    of `target` (an HF checkpoint directory with the draft's vocabulary)."""
    out = path / f"draft-{outtype}.gguf"
    subprocess.run([sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), str(path), "--target-model-dir",
                    str(target), "--outtype", outtype, "--outfile", str(out)],
                   env=os.environ | {"PYTHONPATH": str(LLAMA / "gguf-py")}, check=True, capture_output=True)
    return out
