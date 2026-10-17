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
            layer.mlp.experts.gate_up_proj.mul_(20)          # SiLU out of its linear range
            layer.mlp.experts.down_proj.mul_(5)               # the MoE output matters beside the residual
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
    sites = [k for k in engine.ctx.kernels if k.endswith("/linear_experts@2")]
    cfg = json.loads((checkpoint / "config.json").read_text())
    merged = [k for k in sites if "glu=swiglu" in k and f"out={2 * cfg['moe_intermediate_size']}" in k]
    assert len(merged) == 1 and len(sites) == 2, sites                 # the merged gate/up stack and down


def test_gate_and_up_of_different_layouts_fail_at_startup(tiny_moe, tmp_path):
    checkpoint, _ = tiny_moe
    file, _ = _gguf_and_dense(checkpoint, tmp_path, Q.Q8_0, types={"ffn_up_exps": Q.Q4_0})
    with pytest.raises(ValueError, match="gate and up experts must share a layout"):
        _engine(file)


def test_an_hf_moe_checkpoint_serves_what_transformers_computes(tiny_moe):
    """The bf16/fp32 HF path (design D1, D7): per-expert tensors stacked by the stacking rule, dense experts."""
    checkpoint, prompts = tiny_moe
    hf = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float32).eval()
    engine = _engine(checkpoint)
    assert engine.source.kind == "hf_safetensors" and engine.source.arch == "Qwen3MoeForCausalLM"
    rids = [engine.add(p, SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=3)) for p in prompts]
    got = {}
    while engine.has_work():
        got |= {d.rid: d.prompt_logprobs for d in engine.step() if d.prompt_logprobs is not None}
    for p, r in zip(prompts, rids, strict=True):
        with torch.no_grad():
            lp = hf(torch.tensor([p])).logits[0, :-1].log_softmax(-1)
        torch.testing.assert_close(torch.tensor(got[r].token), lp.gather(-1, torch.tensor(p[1:])[:, None])[:, 0],
                                   atol=1e-4, rtol=1e-4)


def test_every_stacked_expert_is_the_checkpoints_tensor(tiny_moe):
    """The structural test (design §4.6): each slice of the loaded stacks equals the checkpoint's tensor, bit for
    bit, whatever the routing would have touched — the merged gate/up included."""
    from safetensors.torch import load_file

    checkpoint, _ = tiny_moe
    tensors = load_file(checkpoint / "model.safetensors")
    engine = _engine(checkpoint)
    cfg = json.loads((checkpoint / "config.json").read_text())
    inter = cfg["moe_intermediate_size"]
    for i, layer in enumerate(engine.executor.model.layers):
        moe, mlp = layer.ffn, f"model.layers.{i}.mlp"
        assert torch.equal(moe.router, tensors[f"{mlp}.gate.weight"])
        for e in range(moe.down.experts):
            gate_up = moe._gate_up.weight[e]
            assert torch.equal(gate_up[:inter], tensors[f"{mlp}.experts.{e}.gate_proj.weight"])
            assert torch.equal(gate_up[inter:], tensors[f"{mlp}.experts.{e}.up_proj.weight"])
            assert torch.equal(moe.down.weight[e], tensors[f"{mlp}.experts.{e}.down_proj.weight"])


def test_the_hf_facts_and_a_missing_expert(tiny_moe, tmp_path):
    from holon_plugins.models.qwen3_moe.recipe import moe_facts
    from safetensors.torch import load_file

    checkpoint, _ = tiny_moe
    cfg = json.loads((checkpoint / "config.json").read_text())
    assert moe_facts(cfg, "hf_safetensors") == {"experts": 8, "top_k": 2, "inter": 32, "renormalize": True}
    with pytest.raises(NotImplementedError, match="dense layers"):
        moe_facts(cfg | {"mlp_only_layers": [1]}, "hf_safetensors")
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "config.json").write_text((checkpoint / "config.json").read_text())
    tensors = load_file(checkpoint / "model.safetensors")
    del tensors["model.layers.1.mlp.experts.5.up_proj.weight"]
    save_file(tensors, broken / "model.safetensors")
    with pytest.raises(Exception, match=r"layers\.1\.ffn\.up\.weight: stack slices not loaded: \[5\]"):
        _engine(broken)


@pytest.mark.parametrize("plant", ["hf-stack-off-by-one", "hf-gate-up-swapped", "hf-two-experts-swapped"])
def test_the_structural_test_catches_a_misplaced_expert(tiny_moe, plant):
    from holon_judge.planted import hf_moe_plants
    from holon_judge.structure import mismatches

    checkpoint, _ = tiny_moe
    assert mismatches(_engine(checkpoint), checkpoint) == []
    with hf_moe_plants(layer=1, experts=8)[plant]():
        engine = _engine(checkpoint)
    assert mismatches(engine, checkpoint)


def _served(engine, prompts) -> tuple[list, dict]:
    """Greedy streams and prompt logprobs of `prompts`."""
    rids = [engine.add(p, SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=3)) for p in prompts]
    got = {}
    while engine.has_work():
        got |= {d.rid: d.prompt_logprobs for d in engine.step() if d.prompt_logprobs is not None}
    return engine.generate(prompts, GREEDY), [(got[r].token, got[r].top_ids, got[r].top) for r in rids]


def test_the_offload_store_serves_what_the_resident_store_does(tiny_moe, tmp_path):
    """docs/v1-offload.md D4, D5, F1, F5a: with `moe.store = std.moe.offload@1` the experts stay in the file's map and
    in pinned host pools (filled from its extents), classes by (role, type, shape) — here gate/up Q8_0 and down
    Q4_0 —, every miss fetched into staging slots: the outputs are bitwise the resident store's whatever the cache
    holds (every expert by default, half, a profile's pick per class, none). The static policy caches item 0, 1, … of
    every layer, or a profile's hottest; a layer's distinct misses never pass its staging slots, and a step that
    did would stop the engine."""
    import json as json_

    checkpoint, prompts = tiny_moe
    file, _ = _gguf_and_dense(checkpoint, tmp_path, Q.Q8_0, types={"ffn_down_exps": Q.Q4_0})
    want = _served(_engine(file), prompts)
    profile = tmp_path / "profile.json"
    hot = [[3, 1], [6, 5]]                                       # per layer: the hottest experts, then the next
    counts = np.zeros((2, 8))
    for layer, (a, b) in enumerate(hot):
        counts[layer, [a, b]] = [9, 5]
    profile.write_text(json_.dumps({"counts": counts.tolist()}))
    for cache, extra, cached in (("", {}, 8), ("*=0.5", {}, 4), ("gate_up.q8_0.64x64=4,*=2", {
            "moe.placement.profile": str(profile)}, 1), ("*=0", {}, 0)):
        engine = _engine(file, host_memory=1 << 24, plugin_options={
            "moe.store": "std.moe.offload@1", "moe.offload.cache": cache, **extra})
        store = engine.ctx.shared_objects["std.moe.offload"]
        assert _served(engine, prompts) == want, cache
        assert sorted(store.classes) == ["down.q4_0.64x32", "gate_up.q8_0.64x64"]
        assert {c.host.capacity for c in store.classes.values()} == {16}          # 2 layers x 8 experts each
        assert all(c.null == 2 and c.staging == 8 for c in store.classes.values())
        resident = store.residency.host >= 0                                     # [layer, expert, role]
        assert (resident.all(-1) == resident.any(-1)).all()                      # whole experts
        assert resident[..., 0].sum(-1).tolist() == [cached] * 2
        if extra:
            assert [sorted(np.flatnonzero(r).tolist()) for r in resident[..., 0]] == [[3], [6]]
        elif cached:
            assert resident[:, :cached].all()
        assert 0 < store.peak_misses <= store.staging if cached < 8 else store.peak_misses == 0
    store.staging = store.peak_misses - 1                                    # as if staging had been too small
    engine.add(prompts[0], GREEDY)
    with pytest.raises(RuntimeError, match="distinct misses"):
        while engine.has_work():
            engine.step()


@pytest.fixture(scope="module")
def gpu_moe(tmp_path_factory):
    """A random Qwen3-MoE the GGUF kernels take (rows of 256 values), as a GGUF of two expert classes, its prompts
    (decode-length to chunked prefill), and the resident store's outputs on the GPU."""
    torch.manual_seed(1)
    cfg = Qwen3MoeConfig(vocab_size=512, hidden_size=256, intermediate_size=512, moe_intermediate_size=256,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
                         num_experts=16, num_experts_per_tok=4, norm_topk_prob=True, tie_word_embeddings=False,
                         eos_token_id=None)
    path = tmp_path_factory.mktemp("gpu_moe")
    model = Qwen3MoeForCausalLM(cfg)
    with torch.no_grad():
        for layer in model.model.layers:
            layer.mlp.gate.weight.mul_(8)
    model.save_pretrained(path / "hf")
    write(path / "hf", path / "moe.gguf", Q.Q8_0, types={"ffn_down_exps": Q.Q4_0})
    prompts = [torch.randint(0, 512, (n,)).tolist() for n in (1, 7, 40, 300)]
    engine = _gpu_engine(path / "moe.gguf")
    want = _served(engine, prompts)
    engine.close()
    return path / "moe.gguf", prompts, want


def _gpu_engine(model, **kw) -> Engine:
    return Engine(EngineConfig(model=str(model), device="cuda", dtype=torch.bfloat16, page_size=16, max_seqs=4,
                               max_step_tokens=256, max_model_len=512, cuda_graphs=True, overlap=True, **kw))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("cache", ["", "*=0.25", "*=0"])
def test_the_offload_store_on_the_gpu_is_bitwise_the_resident_store(gpu_moe, cache):
    """docs/v1-offload.md §6's exact class on the GPU kernels (kernels.ggml_cuda over class arenas, kernels.expert_io's
    resolve and zero-copy gather from mapped host pools), decode steps replayed from CUDA graphs and prefill chunked
    past the mmvq limit: the offload store's streams and prompt logprobs are the resident store's, bit for bit."""
    file, prompts, want = gpu_moe
    engine = _gpu_engine(file, host_memory=1 << 26, plugin_options={"moe.store": "std.moe.offload@1",
                                                                   "moe.offload.cache": cache})
    store = engine.ctx.shared_objects["std.moe.offload"]
    assert _served(engine, prompts) == want
    assert engine.stats()["graphs"] > 0 and (store.peak_misses > 0) == (cache != "")
    assert {v["provider"] for k, v in engine.ctx.kernels.items() if k.startswith("std.tiered")} == {"kernels.expert_io"}
    engine.close()
