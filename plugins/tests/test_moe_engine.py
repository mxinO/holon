"""Qwen3-MoE end to end on the CPU (test-only reference kernels) against an independent oracle: a tiny random HF
`Qwen3MoeForCausalLM`, written as a GGUF with quantized expert stacks, must serve what transformers computes on the
same (dequantized) weights (docs/m02-moe.md, step 1)."""
import json

import gguf
import numpy as np
import pytest
import torch
from holon.api import SamplingParams
from holon.core.engine import Engine, EngineConfig
from holon_testing.gguf_files import write
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM

from .test_engine import _hf_greedy

Q = gguf.GGMLQuantizationType
GREEDY = SamplingParams(temperature=0.0, max_tokens=12, ignore_eos=True)


@pytest.fixture(scope="module")
def tiny_moe(tmp_path_factory):
    """A random two-layer Qwen3-MoE (8 experts, top-2), and its prompts."""
    torch.manual_seed(0)
    cfg = Qwen3MoeConfig(vocab_size=512, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         num_experts=8, num_experts_per_tok=2, norm_topk_prob=True, tie_word_embeddings=False,
                         eos_token_id=None)
    path = tmp_path_factory.mktemp("tiny_qwen3_moe")
    model = Qwen3MoeForCausalLM(cfg)
    with torch.no_grad():                                  # a router sharp enough that routing matters
        for layer in model.model.layers:
            layer.mlp.gate.weight.mul_(8)
    model.save_pretrained(path)
    prompts = [torch.randint(0, 512, (n,)).tolist() for n in (3, 17, 8, 25)]
    return path, prompts


def _gguf_and_dense(checkpoint, tmp_path, qtype, **kw):
    """The checkpoint as a GGUF file, and an HF checkpoint of the file's dequantized weights (the oracle's)."""
    file, dense = tmp_path / f"moe-{qtype.name}.gguf", tmp_path / "dense"
    dequantized = write(checkpoint, file, qtype, **kw)
    dense.mkdir()
    (dense / "config.json").write_text((checkpoint / "config.json").read_text())
    save_file({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in dequantized.items()},
              dense / "model.safetensors")
    return file, dense


def _engine(model, **kw) -> Engine:
    cfg = dict(model=str(model), device="cpu", dtype=torch.float32, page_size=4, max_seqs=3, max_step_tokens=10,
               max_model_len=40, state_memory=1 << 20)
    return Engine(EngineConfig(**cfg | kw))


@pytest.mark.parametrize("qtype", [Q.Q8_0, Q.Q4_0])
def test_a_moe_gguf_serves_what_transformers_computes(tiny_moe, tmp_path, qtype):
    checkpoint, prompts = tiny_moe
    file, dense = _gguf_and_dense(checkpoint, tmp_path, qtype)
    hf = AutoModelForCausalLM.from_pretrained(dense, dtype=torch.float32).eval()
    engine = _engine(file)
    assert engine.source.arch == "gguf.qwen3moe"
    rids = [engine.add(p, SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=3)) for p in prompts]
    got = {}
    while engine.has_work():
        got |= {d.rid: d.prompt_logprobs for d in engine.step() if d.prompt_logprobs is not None}
    for p, r in zip(prompts, rids, strict=True):
        with torch.no_grad():
            lp = hf(torch.tensor([p])).logits[0, :-1].log_softmax(-1)
        want = lp.gather(-1, torch.tensor(p[1:])[:, None])[:, 0]
        torch.testing.assert_close(torch.tensor(got[r].token), want, atol=1e-4, rtol=1e-4)
        assert got[r].top_ids == lp.topk(3).indices.tolist()
    assert _engine(file).generate(prompts, GREEDY) == _hf_greedy(hf, prompts, GREEDY.max_tokens)


def test_the_gate_up_merge_puts_gate_first_and_binds_one_kernel(tiny_moe, tmp_path):
    checkpoint, _ = tiny_moe
    file, _ = _gguf_and_dense(checkpoint, tmp_path, Q.Q8_0)
    engine = _engine(file)
    sites = [k for k in engine.ctx.kernels if k.endswith("/linear_experts@1")]
    cfg = json.loads((checkpoint / "config.json").read_text())
    merged = [k for k in sites if "glu=swiglu" in k and f"out={2 * cfg['moe_intermediate_size']}" in k]
    assert len(merged) == 1 and len(sites) == 2, sites                 # the merged gate/up stack and down


def test_gate_and_up_of_different_layouts_fail_at_startup(tiny_moe, tmp_path):
    checkpoint, _ = tiny_moe
    file, _ = _gguf_and_dense(checkpoint, tmp_path, Q.Q8_0, types={"ffn_up_exps": Q.Q4_0})
    with pytest.raises(ValueError, match="gate and up experts must share a layout"):
        _engine(file)
