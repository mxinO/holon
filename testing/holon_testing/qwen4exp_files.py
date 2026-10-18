"""The tiny Qwen3.8-Flash-Next (`qwen4_exp`) for CPU tests (docs/v1-flash-next.md D2, §6.1): random weights in
transformers' `Qwen4ExpForCausalLM` (5.19.0; run in the golden venv, `local/envs/golden-qwen4`), saved as
`Qwen4ExpForConditionalGeneration` stores them (the n-gram table in `split_ngram_parts` shards, a random MTP block
under `mtp.`); its GGUF as llama.cpp's own converter writes it (an oracle for the conversion's conventions that does
not share the engine's or the judge's beliefs: norm + 1, `ssm_a`, the conv layouts, the HC reshapes); and its MTP
block as Strata's tools see it: `tools/mtp_fetch.py`'s layout (one raw bf16 file per tensor and `mtp-manifest.json`),
packed by `tools/mtp_pack.py`.

    python -m holon_testing.qwen4exp_files --out <dir> --llama <llama.cpp 27c7a76> [--strata <Strata 296b59d>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys

import torch
from safetensors.torch import save_file
from transformers import Qwen4ExpTextConfig
from transformers.models.qwen4_exp import modeling_qwen4_exp as hf

TOKENIZER = ("Qwen/Qwen3.8-Flash-Next", "de4b8e4d43b917e7706784d8bb445c9af86a3540")
EOS = 248044


def tiny_config() -> Qwen4ExpTextConfig:
    """§6.1: 8 layers (QSA at 3 and 7, PLE at layer 1), H = 64, four streams of rank 16, the real vocabulary; attention
    4 / 2 heads x 64; indexer 2 x 32, ratio 4, budget 16 (so selection is sparse past 16 + 3 tokens); GDN 2 / 6 heads x
    32; 8 experts top-2 with I = 64 and a shared expert of 64 (Q2_0's 64-value blocks); PLE 4 n-gram heads x 16 over
    1,000-row hash tables; one MTP layer."""
    return Qwen4ExpTextConfig(
        vocab_size=248320, hidden_size=64, num_hidden_layers=8, layer_types=None, full_attention_interval=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64, linear_num_key_heads=2, linear_num_value_heads=6,
        linear_key_head_dim=32, linear_value_head_dim=32, linear_conv_kernel_dim=4, moe_intermediate_size=64,
        shared_expert_intermediate_size=64, num_experts=8, num_experts_per_tok=2, hc_count=4, hc_lowrank=16,
        ple_layer_ids=[2], ple_embed_dim=64, heads_per_ngram=2, ngram_size=3, ngram_vocab_size_base=1000,
        split_ngram_parts=4, indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=32, indexer_budget=16,
        indexer_compress_ratio=4, output_gate_type="sigmoid", eos_token_id=EOS, bos_token_id=EOS,
        tie_word_embeddings=False, mtp_num_hidden_layers=1,
        rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                         "mrope_section": [3, 3, 2], "mrope_interleaved": True})


def tiny_hf(seed: int = 0) -> torch.nn.Module:
    """The random model; every norm, the PLE conv, `A_log` and `dt_bias` randomised too (HF zero-initialises the norms
    and the PLE conv, and zeros would hide a wrong convention)."""
    torch.manual_seed(seed)
    model = hf.Qwen4ExpForCausalLM(tiny_config()).eval()
    norms = (hf.Qwen4ExpTextRMSNorm, hf.Qwen4ExpTextRMSNormGated)
    picked = [m.weight for m in model.modules() if isinstance(m, norms)] + [
        p for name, p in model.named_parameters() if name.endswith(("A_log", "dt_bias", "ple.conv1d.weight"))]
    with torch.no_grad():
        for p in picked:
            p.copy_(torch.randn_like(p) * 0.5)
        for p in model.parameters():                     # bf16 values, as the real checkpoint holds: the converter's
            p.copy_(p.bfloat16().float())                # w + 1 and the judge's - 1 are then exact in f32
    assert all(p.abs().min() > 0 for p in picked)
    return model


def mtp_tensors(model: torch.nn.Module, seed: int = 3) -> dict[str, torch.Tensor]:
    """A random MTP block (bf16) under the real checkpoint's names: the projections from e and h, the pre-fc norms,
    one QSA + MoE layer shaped like the trunk's (layer 3's) and its output mixer."""
    g = torch.Generator().manual_seed(seed)
    rand = lambda *s: (torch.randn(*s, generator=g) * 0.05).to(torch.bfloat16)            # noqa: E731
    c = model.config
    h, n = c.hidden_size, c.hc_count * c.hidden_size
    out = {"mtp.fc_embedding.weight": rand(h, h), "mtp.fc_hidden.weight": rand(h, h),
           "mtp.pre_fc_norm_embedding.weight": rand(h), "mtp.pre_fc_norm_hidden.weight": rand(n)}
    qsa = next(i for i, t in enumerate(c.layer_types) if t == "indexed_attention")
    for k, v in model.model.layers[qsa].state_dict().items():
        out[f"mtp.layers.0.{k}"] = rand(*v.shape)
    for k, v in model.model.hyper_connection_mixer.state_dict().items():
        out[f"mtp.hyper_connection_mixer.{k}"] = rand(*v.shape)
    return out


def checkpoint(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """The state dict as `Qwen4ExpForConditionalGeneration` stores it: bf16, the language model under
    `model.language_model.`, the n-gram table split row-wise into `split_ngram_parts` shards, the int64 hash buffers."""
    out = {}
    parts = model.config.split_ngram_parts
    for k, v in model.state_dict().items():
        k = k.replace("model.", "model.language_model.", 1)
        v = v.bfloat16() if v.is_floating_point() else v
        if k.endswith("ngram_embedding.weight"):
            for i, shard in enumerate(v.chunk(parts)):
                out[k.replace("ngram_embedding.weight", f"ngram_embedding.shard_{i}.weight")] = shard.contiguous()
        else:
            out[k] = v.contiguous()
    return out


def save(model: torch.nn.Module, path: pathlib.Path) -> pathlib.Path:
    """`model` (with its random MTP block) as an HF checkpoint directory."""
    path.mkdir(parents=True, exist_ok=True)
    save_file(checkpoint(model) | mtp_tensors(model), str(path / "model.safetensors"))
    text = model.config.to_dict() | {"model_type": "qwen4_exp_text"}
    (path / "config.json").write_text(json.dumps({"architectures": ["Qwen4ExpForConditionalGeneration"],
                                                  "model_type": "qwen4_exp", "tie_word_embeddings": False,
                                                  "language_model_only": True, "text_config": text}))
    return path


def tokenizer_dir() -> pathlib.Path:
    hub = pathlib.Path(os.environ.get("HF_HUB_CACHE") or pathlib.Path(os.environ.get("HF_HOME") or
                                                                       "~/.cache/huggingface").expanduser() / "hub")
    return hub / f"models--{TOKENIZER[0].replace('/', '--')}" / "snapshots" / TOKENIZER[1]


def convert(path: pathlib.Path, llama: pathlib.Path) -> pathlib.Path:
    """`path` (the checkpoint directory) converted by llama.cpp's `convert_hf_to_gguf.py` to f32, with the model's
    tokenizer (the converter needs the full 248,320-token vocabulary)."""
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        (path / name).write_bytes((tokenizer_dir() / name).read_bytes())
    out = path / "tiny-qwen4exp-f32.gguf"
    subprocess.run([sys.executable, str(llama / "convert_hf_to_gguf.py"), str(path), "--outtype", "f32",
                    "--outfile", str(out)], env=os.environ | {"PYTHONPATH": str(llama / "gguf-py")}, check=True)
    return out


def mtp_pack(model: torch.nn.Module, path: pathlib.Path, strata: pathlib.Path, llama: pathlib.Path) -> pathlib.Path:
    """The MTP block in `tools/mtp_fetch.py`'s layout (St tools/mtp_fetch.py:1-12), then Strata's own
    `tools/mtp_pack.py --experts q2_0` (HF names, fused gate_up_proj, BF16 dense parts, F32 norms, MSE-scaled Q2_0)."""
    src = path / "mtp"
    (src / "tensors").mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, t in sorted(mtp_tensors(model).items()):
        raw = t.contiguous().view(torch.int16).numpy().tobytes()
        (src / "tensors" / f"{name}.bin").write_bytes(raw)
        manifest.append({"name": name, "dtype": "BF16", "shape": list(t.shape), "file": f"tensors/{name}.bin",
                         "sha256": hashlib.sha256(raw).hexdigest()})
    (src / "mtp-manifest.json").write_text(json.dumps(manifest, indent=1))
    out = path / "tiny-qwen4exp-mtp-q2_0.gguf"
    subprocess.run([sys.executable, str(strata / "tools" / "mtp_pack.py"), "--src", str(src), "--experts", "q2_0",
                    "--out", str(out)], env=os.environ | {"STRATA_GGUF_PY": str(llama / "gguf-py")}, check=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--llama", required=True, type=pathlib.Path, help="a llama.cpp checkout at 27c7a76")
    ap.add_argument("--strata", type=pathlib.Path, help="a Strata checkout at 296b59d (the MTP pack)")
    a = ap.parse_args()
    model = tiny_hf()
    path = save(model, a.out)
    files = [convert(path, a.llama)] + ([mtp_pack(model, path, a.strata, a.llama)] if a.strata else [])
    for f in files:
        print(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}")


if __name__ == "__main__":
    main()
