"""The judge on a tiny Qwen3-MoE written as a Q8_0 GGUF (docs/m02-moe.md §4.6): the file's golden is HF's model of
its dequantized weights, the gate checks the judge's expert assembly against transformers' own loading, an MoE floor
runs the D3-mirroring block, and Tier A — the engine on the production source with reference kernels — sits on the
golden while each planted MoE bug moves it."""
import gguf
import numpy as np
import pytest
import torch
from holon.core.engine import Engine, EngineConfig
from holon_judge import gguf as judge_gguf
from holon_judge import golden
from holon_judge.compare import compare
from holon_judge.engine import score
from holon_judge.planted import moe_plants
from holon_testing.gguf_files import write
from safetensors.torch import save_file
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

VOCAB = 512                                                   # the tiny prompts' vocabulary (conftest)


@pytest.fixture(scope="module")
def moe_files(tmp_path_factory):
    """A random two-layer Qwen3-MoE (8 experts, top-2, a sharp router), its Q8_0 GGUF and an HF checkpoint of the
    file's dequantized weights."""
    torch.manual_seed(0)
    cfg = Qwen3MoeConfig(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         num_experts=8, num_experts_per_tok=2, norm_topk_prob=True, tie_word_embeddings=False,
                         eos_token_id=None)
    d = tmp_path_factory.mktemp("moe")
    checkpoint, file, dense = d / "hf", d / "tiny-moe-Q8_0.gguf", d / "dense"
    model = Qwen3MoeForCausalLM(cfg)
    with torch.no_grad():
        for layer in model.model.layers:
            layer.mlp.gate.weight.mul_(8)
            layer.mlp.experts.gate_up_proj.mul_(20)          # SiLU out of its linear range
            layer.mlp.experts.down_proj.mul_(5)               # the MoE output matters beside the residual
    model.save_pretrained(checkpoint)
    dequantized = write(checkpoint, file, gguf.GGMLQuantizationType.Q8_0)
    dense.mkdir()
    (dense / "config.json").write_text((checkpoint / "config.json").read_text())
    save_file({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in dequantized.items()},
              dense / "model.safetensors")
    return checkpoint, file, dense


@pytest.fixture(scope="module")
def gold(moe_files, tiny_prompts):
    checkpoint, file, _ = moe_files
    return golden.record(str(file), tiny_prompts, config=str(checkpoint))


def test_the_golden_is_hfs_model_of_the_dequantized_weights(moe_files, tiny_prompts, gold):
    r = compare(gold, golden.record(str(moe_files[2]), tiny_prompts)).overall
    assert r.top1 == 1 and r.kl_p99 < 1e-8 and "moe_block" not in gold.meta    # fp32: HF's own block


def test_the_gate_checks_every_expert_against_transformers_own_loading(moe_files):
    checkpoint, file, _ = moe_files
    nmse = judge_gguf.gate(file, checkpoint)
    assert sum(".mlp.experts.gate_up_proj[" in n for n in nmse) == 2 * 8 and max(nmse.values()) < 1e-3


def test_an_moe_floor_runs_the_d3_block(moe_files, tiny_prompts, gold):
    checkpoint, file, _ = moe_files
    floor = golden.record(str(file), tiny_prompts, dtype=torch.bfloat16, config=str(checkpoint))
    pure = golden.record(str(file), tiny_prompts, dtype=torch.bfloat16, config=str(checkpoint), pure_hf=True)
    assert floor.meta["moe_block"].startswith("d3-reference@") and "moe_block" not in pure.meta
    assert compare(gold, floor).overall.kl_p99 <= compare(gold, pure).overall.kl_p99   # never looser than pure HF


def _tier_a(file, prompts):
    engine = Engine(EngineConfig(model=str(file), device="cpu", dtype=torch.float32, page_size=4, max_seqs=4,
                                 max_model_len=64, max_step_tokens=16, state_memory=1 << 20))
    return score(engine, prompts)


@pytest.fixture(scope="module")
def clean(moe_files, tiny_prompts, gold):
    return compare(gold, _tier_a(moe_files[1], tiny_prompts)).overall


def test_tier_a_scores_the_dequantized_model(clean):
    assert clean.top1 == 1 and clean.kl_p99 < 1e-8


@pytest.mark.parametrize("plant", sorted(moe_plants(layer=1)))
def test_a_planted_moe_bug_moves_the_scores(moe_files, tiny_prompts, gold, clean, plant):
    with moe_plants(layer=1)[plant]():
        r = compare(gold, _tier_a(moe_files[1], tiny_prompts))
    assert r.overall.kl_p99 > 100 * clean.kl_p99, r.table()
