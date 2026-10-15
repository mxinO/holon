"""Tiny Qwen3.5/3.8-architecture checkpoints for CPU tests: random weights in transformers' `Qwen3_5ForCausalLM`,
saved as `Qwen3_5ForConditionalGeneration` stores them, and their GGUF as llama.cpp's own converter writes it (an
oracle that does not share the engine's or the judge's beliefs about the conversion)."""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import torch
from safetensors.torch import save_file
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

ROOT = pathlib.Path(__file__).resolve().parents[2]
LLAMA = ROOT / "spikes" / "s2_gguf" / "third_party" / "llama.cpp"
TOKENIZER = sorted(pathlib.Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*"))


def tiny_config(heads_v: int, vocab: int = 320, dim: int = 16) -> Qwen3_5TextConfig:
    """2 key heads (so tiled and grouped value-head orders differ), `heads_v` value heads of `dim` (64 for the GPU
    kernels; attention heads of max(32, dim)), layers alternating linear and full attention, partial rotary (8 dims)
    with mrope sections."""
    head = max(32, dim)
    return Qwen3_5TextConfig(
        vocab_size=vocab, hidden_size=64, intermediate_size=96, num_hidden_layers=4, layer_types=None,
        full_attention_interval=2, num_attention_heads=4, num_key_value_heads=2, head_dim=head,
        linear_num_key_heads=2, linear_num_value_heads=heads_v, linear_key_head_dim=dim, linear_value_head_dim=dim,
        linear_conv_kernel_dim=4, tie_word_embeddings=False, attn_implementation="eager", eos_token_id=None,
        rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 8 / head,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True})


def tiny_hf(heads_v: int, seed: int = 0, vocab: int = 320, dim: int = 16) -> torch.nn.Module:
    """A random model whose norms, A_log and dt_bias are randomised too (their initial values would hide mistakes)."""
    torch.manual_seed(seed)
    model = hf.Qwen3_5ForCausalLM(tiny_config(heads_v, vocab, dim)).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(("norm.weight", "A_log", "dt_bias")):
                p.copy_(torch.randn_like(p) * 0.5)
    return model


def checkpoint(model) -> list[tuple[str, torch.Tensor]]:
    """The state dict as `Qwen3_5ForConditionalGeneration` stores it (language model under `model.language_model.`)."""
    return [(k.replace("model.", "model.language_model.", 1), v) for k, v in model.state_dict().items()]


def save(model, path: pathlib.Path, mtp: bool = False) -> pathlib.Path:
    """`model` as an HF checkpoint directory (with a random MTP block when `mtp`, as the real checkpoints have)."""
    tensors = dict(checkpoint(model))
    cfg = model.config.to_dict()
    if mtp:
        g = torch.Generator().manual_seed(3)
        dtype = next(model.parameters()).dtype
        rand = lambda *s: (torch.randn(*s, generator=g) * 0.05).to(dtype)            # noqa: E731
        h = model.config.hidden_size
        tensors |= {"mtp.fc.weight": rand(h, 2 * h), "mtp.norm.weight": rand(h),
                    "mtp.pre_fc_norm_embedding.weight": rand(h), "mtp.pre_fc_norm_hidden.weight": rand(h)}
        full = next(i for i, k in enumerate(model.config.layer_types) if k == "full_attention")
        tensors |= {f"mtp.layers.0.{k.removeprefix(f'model.layers.{full}.')}": rand(*v.shape)
                    for k, v in model.state_dict().items() if k.startswith(f"model.layers.{full}.")}
        cfg |= {"mtp_num_hidden_layers": 1}
    save_file({k: v.contiguous() for k, v in tensors.items()}, path / "model.safetensors")
    (path / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"],
                                                  "model_type": "qwen3_5", "tie_word_embeddings": False,
                                                  "text_config": cfg}))
    return path


def convert(path: pathlib.Path) -> pathlib.Path:
    """`path` (an HF checkpoint directory) converted by llama.cpp's `convert_hf_to_gguf.py` to f32, with the Qwen3.8
    tokenizer the converter needs (the model's vocabulary must be its 248320)."""
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (path / name).write_bytes((TOKENIZER[0] / name).read_bytes())
    out = path / "tiny.gguf"
    subprocess.run([sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), str(path), "--outtype", "f32",
                    "--outfile", str(out)], env=os.environ | {"PYTHONPATH": str(LLAMA / "gguf-py")}, check=True,
                   capture_output=True)
    return out


def can_convert() -> bool:
    return (LLAMA / "convert_hf_to_gguf.py").is_file() and bool(TOKENIZER)
