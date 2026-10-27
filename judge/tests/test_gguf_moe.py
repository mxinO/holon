"""The judge on a tiny Qwen3-MoE written as a Q8_0 GGUF (docs/dev/m02-moe.md §4.6): the file's golden is HF's model of
its dequantized weights, the gate checks the judge's expert assembly against the checkpoint's tensors, an MoE floor
runs the D3-mirroring block, and Tier A — the engine on the production source with reference kernels — sits on the
golden while each planted MoE bug moves it."""
from unittest import mock

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
from holon_testing.ref_kernels import kernels as ref_kernels
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
    """Bit for bit: the file's golden runs one layer at a time on the judge's expert assembly, transformers' runs
    the whole model it loaded (its own fusion of the per-expert tensors); so does the dense checkpoint streamed by the
    judge's HF reader (`hf_quant.state`), which the gates read the originals with."""
    dense = golden.record(str(moe_files[2]), tiny_prompts)
    assert gold.same_tensors(dense) and "moe_block" not in gold.meta    # fp32: HF's own block
    assert golden.record(str(moe_files[2]), tiny_prompts, stream=True).same_tensors(dense)


def test_the_gate_checks_every_expert_against_the_checkpoint(moe_files):
    checkpoint, file, _ = moe_files
    nmse = judge_gguf.gate(file, checkpoint)
    assert sum(".mlp.experts.gate_up_proj[" in n for n in nmse) == 2 * 8 and max(nmse.values()) < 1e-3


def test_an_moe_floor_runs_the_d3_block(moe_files, tiny_prompts, gold):
    checkpoint, file, _ = moe_files
    floor = golden.record(str(file), tiny_prompts, dtype=torch.bfloat16, config=str(checkpoint))
    pure = golden.record(str(file), tiny_prompts, dtype=torch.bfloat16, config=str(checkpoint), pure_hf=True)
    assert floor.meta["moe_block"].startswith("d3-reference@") and "moe_block" not in pure.meta
    # never looser than pure HF beyond float noise: both sit near 1e-7 on the tiny model, where the CPU's BLAS (threads,
    # instruction set) moves the last bits
    assert compare(gold, floor).overall.kl_p99 <= compare(gold, pure).overall.kl_p99 + 1e-6


def _tier_a(file, prompts):
    engine = Engine(EngineConfig(model=str(file), device="cpu", dtype=torch.float32, page_size=4, max_seqs=4,
                                 max_model_len=64, max_step_tokens=16, state_memory=1 << 20))
    return score(engine, prompts)


@pytest.fixture(scope="module")
def tier_a(moe_files, tiny_prompts):
    return _tier_a(moe_files[1], tiny_prompts)


@pytest.fixture(scope="module")
def clean(gold, tier_a):
    return compare(gold, tier_a).overall


def test_tier_a_scores_the_dequantized_model(moe_files, tiny_prompts, tier_a, clean):
    assert clean.top1 == 1 and clean.kl_p99 < 1e-8
    with mock.patch.object(ref_kernels, "CACHE_BYTES", 0):     # a 30B's stacks: only the routed experts, per call
        assert _tier_a(moe_files[1], tiny_prompts).same_tensors(tier_a)


@pytest.mark.parametrize("plant", sorted(moe_plants(layer=1)))
def test_a_planted_moe_bug_moves_the_scores(moe_files, tiny_prompts, gold, clean, plant):
    with moe_plants(layer=1)[plant]():
        r = compare(gold, _tier_a(moe_files[1], tiny_prompts))
    assert r.overall.kl_p99 > 100 * clean.kl_p99, r.table()


@pytest.mark.parametrize("kernel_path", ["mmq", "mmvq"])
def test_the_tier_b_floor_emulates_q8_1_in_the_experts(moe_files, tiny_prompts, gold, kernel_path):
    checkpoint, file, _ = moe_files
    plain = golden.record(str(file), tiny_prompts, dtype=torch.bfloat16, config=str(checkpoint))
    floor = golden.record(str(file), tiny_prompts, dtype=torch.bfloat16, config=str(checkpoint), q8_1=kernel_path)
    assert floor.meta["q8_1"] == kernel_path and floor.meta["moe_block"].startswith("d3-reference@")
    assert not floor.same_tensors(plain)                                  # q8_1 rounding reached the experts
    assert compare(gold, floor).overall.kl_p99 < 1e-2


def test_a_cpu_split_floor_takes_ggml_cpus_activations_for_the_cpus_experts(moe_files, tiny_prompts, gold):
    """§6's path-aware floor (docs/dev/v1-offload.md F5b): its rule picks, token by token, the experts the contract's
    (std.moe.reference `cpu_share`, one token per call) sends to the CPU of a frozen placement, and those entries take
    ggml-cpu's activations; the floor moves off the plain one and stays near the golden."""
    from holon_std.moe import reference

    checkpoint, file, _ = moe_files
    split = {"plan": [0, 1, 1], "tokens": 1, "layers": {"0": [0, 3, 5], "1": [1, 2, 6]}}
    g = torch.Generator().manual_seed(0)
    ids = torch.stack([torch.randperm(8, generator=g)[:2] for _ in range(60)]).int()
    residency = torch.full((8, 2), -1, dtype=torch.int32)
    residency[split["layers"]["0"]] = 9
    ring, host, kinds = torch.full((8, 2), -1), torch.zeros(8, 2, dtype=torch.int32), torch.ones(1, dtype=torch.uint8)
    for t in range(len(ids)):
        _, cpu, _ = reference.cpu_share(ids[t:t + 1], kinds, residency, ring, host, torch.tensor(split["plan"]))
        assert torch.equal(cpu >= 0, judge_gguf.cpu_entries(split, 0, ids[t:t + 1]))
    plain = golden.record(str(file), tiny_prompts, dtype=torch.bfloat16, config=str(checkpoint), q8_1="mmvq")
    floor = golden.record(str(file), tiny_prompts, dtype=torch.bfloat16, config=str(checkpoint), q8_1="mmvq",
                          cpu_split=split)
    assert floor.meta["cpu_split"] and not floor.same_tensors(plain) and compare(gold, floor).overall.kl_p99 < 1e-2
