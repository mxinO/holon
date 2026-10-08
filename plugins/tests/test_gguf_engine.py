"""GGUF models end to end on the CPU (the test-only reference kernels): a tiny random Qwen3 written as a GGUF with
quantized weights serves exactly what the same model serves from its dequantized weights (M0.1 step 1)."""
import json
import re

import gguf
import numpy as np
import pytest
import torch
from holon.core.engine import Engine, EngineConfig
from holon.core.plugins import PluginError
from safetensors.torch import load_file, save_file

from .test_engine import GREEDY, tiny_case, tiny_checkpoint  # noqa: F401 — fixtures

Q = gguf.GGMLQuantizationType
NAMES = {"model.embed_tokens.weight": "token_embd.weight", "lm_head.weight": "output.weight",  # HF -> llama.cpp
         "model.norm.weight": "output_norm.weight"}
LAYER = {"input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm", "self_attn.q_proj": "attn_q",
         "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
         "self_attn.q_norm": "attn_q_norm", "self_attn.k_norm": "attn_k_norm", "mlp.gate_proj": "ffn_gate",
         "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down"}


def _gguf_name(hf: str) -> str:
    if hf in NAMES:
        return NAMES[hf]
    _, _, i, rest = hf.split(".", 3)
    return f"blk.{i}.{LAYER[rest.removesuffix('.weight')]}.weight"


def _write(checkpoint, path, qtype, quantize_norms=False) -> dict[str, np.ndarray]:
    """The checkpoint as a GGUF file (matrices in `qtype`, norms F32); returns the dequantized tensors by HF name."""
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
            w.add_tensor(_gguf_name(name), blocks, raw_dtype=qtype)
            x = gguf.quants.dequantize(blocks, qtype).reshape(x.shape)
        else:
            w.add_tensor(_gguf_name(name), x)
        dequantized[name] = x
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return dequantized


def _engine(model, **kw) -> Engine:
    cfg = dict(model=str(model), device="cpu", dtype=torch.float32, page_size=4, max_seqs=3, max_step_tokens=10,
               max_model_len=40, state_memory=1 << 20)
    return Engine(EngineConfig(**cfg | kw))


@pytest.mark.parametrize("qtype", [Q.Q8_0, Q.Q4_0])
def test_a_quantized_gguf_serves_what_its_dequantized_weights_serve(tiny_checkpoint, tiny_case, tmp_path, qtype):  # noqa: F811
    prompts, _ = tiny_case
    file = tmp_path / f"tiny-{qtype.name}.gguf"
    dense = tmp_path / "dense"
    dense.mkdir()
    (dense / "config.json").write_text((tiny_checkpoint / "config.json").read_text())
    save_file({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in _write(tiny_checkpoint, file, qtype).items()},
              dense / "model.safetensors")
    ours = _engine(file)
    assert ours.source.kind == "gguf" and ours.source.arch == "gguf.qwen3"
    assert ours.generate(prompts, GREEDY) == _engine(dense).generate(prompts, GREEDY)
    # One quantized matmul per merged group (q/k/v, gate/up) plus o, down and the head; the merged-away
    # projections bound no kernel, so they leave no site in a lock.
    linears = [k for k in ours.ctx.kernels if k.endswith("/qlinear@1")]
    assert any(k.endswith("/qembedding@1") for k in ours.ctx.kernels)
    outs = sorted(int(re.search(r",out=(\d+)", k)[1]) for k in linears)
    assert outs == [64, 64, 64 + 2 * 32, 2 * 128, 512], outs         # o, down, qkv, gate+up, lm_head


def test_a_unit_that_cannot_read_a_layout_fails_at_load(tiny_checkpoint, tmp_path):  # noqa: F811
    file = tmp_path / "norms-quantized.gguf"
    _write(tiny_checkpoint, file, Q.Q8_0, quantize_norms=True)          # RMSNorm has no quantized form
    with pytest.raises(PluginError, match="cannot read layout"):
        _engine(file)
