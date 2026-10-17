"""Qwen3-MoE end to end on the CPU (test-only reference kernels) against an independent oracle: a tiny random HF
`Qwen3MoeForCausalLM`, written as a GGUF with quantized expert stacks, must serve what transformers computes on the
same (dequantized) weights (docs/m02-moe.md, step 1)."""
import contextlib
import json
import pathlib
import types
import warnings
from unittest import mock

import gguf
import numpy as np
import pytest
import torch
from holon.api import RowKind, SamplingParams
from holon.core.batching import host_plan
from holon.core.engine import Engine, EngineConfig
from holon_std.moe.offload import KEY, PASS
from holon_std.moe.units import CHUNK
from holon_std.tiered.ring import Lent
from holon_testing.gguf_files import write
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM

from .test_engine import _hf_greedy

Q = gguf.GGMLQuantizationType
GREEDY = SamplingParams(temperature=0.0, max_tokens=12, ignore_eos=True)


@pytest.fixture(scope="module")
def tiny_moe(tmp_path_factory):
    """A random two-layer Qwen3-MoE (8 experts, top-2), and its prompts."""
    return _tiny_moe(tmp_path_factory, 2)


@pytest.fixture(scope="module")
def tiny_moe_deep(tmp_path_factory):
    """Four layers of it (a ring's two halves fit a half-cached layout once lending spreads over the layers)."""
    return _tiny_moe(tmp_path_factory, 4)


def _tiny_moe(tmp_path_factory, layers: int):
    torch.manual_seed(0)
    cfg = Qwen3MoeConfig(vocab_size=512, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
                         num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         num_experts=8, num_experts_per_tok=2, norm_topk_prob=True, tie_word_embeddings=False,
                         eos_token_id=None)
    path = tmp_path_factory.mktemp(f"tiny_qwen3_moe{layers}")
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
    # A prepared handle another provider made is read as the tensor it stands for, `.x` (std.moe.kernels).
    unit = next(m for m in engine.executor.model.modules() if hasattr(m, "_gate_up") and m._gate_up is not None)
    assert engine.ctx.kernels[merged[0]]["provider"] == "testing.ref_kernels"
    x, ids = torch.randn(3, 1, cfg["hidden_size"]), torch.tensor([[0, 5], [7, 2], [3, 4]], dtype=torch.int32)
    assert torch.equal(unit._gate_up(types.SimpleNamespace(x=x), ids), unit._gate_up(x, ids))


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
    holds (every expert by default, as many as a short fill pool's bytes hold, half, a profile's pick per class, none).
    The static policy caches whole experts, F2's warm start: item 0, 1, … of every layer, or a profile's hottest; a
    layer's distinct misses never pass its staging slots, and a step that did would stop the engine. The adaptive
    policies (the default F11, F4, F2–F3's two-phase swaps through the copy stream) serve the same outputs at a
    quarter of the experts while they move them; F11's residency, step by step, is its replay oracle's (L1 parity)."""
    import json as json_

    from holon_judge.placement import parity, route_dump
    from holon_tools import profile as tool  # `holon_tools profile`: decode routes, beside the file

    checkpoint, prompts = tiny_moe
    file, _ = _gguf_and_dense(checkpoint, tmp_path, Q.Q8_0, types={"ffn_down_exps": Q.Q4_0})
    want = _served(_engine(file), prompts)
    profile = tmp_path / "profile.json"
    hot = [[3, 1], [6, 5]]                                       # per layer: the hottest experts, then the next
    counts = np.zeros((2, 8))
    for layer, (a, b) in enumerate(hot):
        counts[layer, [a, b]] = [9, 5]
    profile.write_text(json_.dumps({"layers": {str(layer): row for layer, row in enumerate(counts.tolist())}}))
    static = {"moe.store": "std.moe.offload@1", "moe.placement": "std.tiered.static@1"}
    for cache, extra, floor, cached in (("", {}, None, 8), ("", {}, 1800, None), ("*=0.5", {}, None, 4), (
            "gate_up.q8_0.64x64=4,*=2", {"moe.placement.profile": profile.name}, None, 1), ("*=0", {}, None, 0)):
        engine = _engine(file, host_memory=1 << 24, paged_min_tokens=floor, plugin_options={
            **static, "moe.cache": cache, **extra})
        store = engine.ctx.shared_objects[KEY]
        counts = []                                                    # each committed step's routing counts
        store.policy.after_commit = lambda step, view, counts=counts: counts.append(step.sum()) or []
        assert _served(engine, prompts) == want, cache
        assert set(counts) <= {r * 2 * 2 for r in range(4)} and max(counts) == 12   # decode rows x top-2 x 2 layers
        assert sorted(store.classes) == ["down.q4_0.64x32", "gate_up.q8_0.64x64"]
        assert {c.host.capacity * store.granule // c.unit_bytes for c in store.classes.values()} == {16}  # 2 x 8
        assert all(c.null == 2 and c.staging == 8 for c in store.classes.values())
        resident = store.residency.host >= 0                                     # [layer, expert, role]
        assert (resident.all(-1) == resident.any(-1)).all()                      # whole experts
        per_layer = resident[..., 0].sum(-1).tolist()
        if cached is None:             # a fill pool short of every expert: whole experts as far as its bytes go
            cached = per_layer[0]
            assert 0 < cached < 8 and per_layer[1] in (cached - 1, cached)
            assert len(store.classes["gate_up.q8_0.64x64"].slots()) == 2 + 8 + sum(per_layer)
        assert per_layer == [cached] * 2 or floor
        if extra:
            assert [sorted(np.flatnonzero(r).tolist()) for r in resident[..., 0]] == [[3], [6]]
        elif cached:
            assert resident[:, :per_layer[1]].all()
        assert 0 < store.peak_misses <= store.staging if cached < 8 else store.peak_misses == 0
    with pytest.raises(ValueError, match="relative to the checkpoint's directory"):   # plans record no host path
        _engine(file, host_memory=1 << 24, plugin_options={"moe.store": "std.moe.offload@1",
                                                           "moe.placement.profile": str(profile)})
    store.stage = store.stage[:, :store.peak_misses - 1].contiguous()       # as if staging had been too small
    for p in prompts:
        engine.add([(t + 7) % 512 for t in p], GREEDY)                       # new prompts (no prefix-cache hit)
    with pytest.raises(RuntimeError, match="distinct misses"):
        while engine.has_work():
            engine.step()
    link = tmp_path / "snapshot" / file.name                      # a hub cache's snapshot link to its blob
    link.parent.mkdir()
    link.symlink_to(file)
    routes, where = tool.profile(EngineConfig(model=str(link), device="cpu", dtype=torch.float32, page_size=4,
                                              max_seqs=3, max_step_tokens=10, max_model_len=40, state_memory=1 << 20,
                                              host_memory=1 << 24), prompts, 4)
    assert sorted(routes["layers"]) == ["0", "1"] and sum(map(sum, routes["layers"].values())) == 4 * 3 * 2 * 2
    assert tool.write(where, "routes.json", routes) == tmp_path / "routes.json"      # where the store reads it
    for policy in ("", "placement.lru@1", "placement.profile_decay@1"):
        engine = _engine(link, host_memory=1 << 24, plugin_options={
            "moe.store": "std.moe.offload@1", "moe.cache": "*=0.25", "moe.placement": policy,
            "moe.placement.profile": "routes.json"})
        store = engine.ctx.shared_objects[KEY]
        with route_dump(engine) as trace:
            assert _served(engine, prompts) == want, policy
        resident = store.residency.host >= 0
        assert (resident.all(-1) == resident.any(-1)).all()
        if policy == "placement.profile_decay@1":     # swaps through the copy stream; the last may still be landing
            assert store.policy.rounds and (resident.all(-1) <= store.policy.want).all(), store.policy.rounds
        else:
            assert trace.admissions > 0 and resident[..., 0].sum(-1).tolist() == [2, 2]
        if not policy:
            assert parity(trace)["first_differing_step"] is None and trace.mirrored
            at, picked = store.n_delta.clone(), store.lend(torch.ones(2, dtype=torch.int64))   # a slot per class
            assert picked.sum() == 1 and (store.residency.device[picked] == -1).all()
            (g, e), = np.argwhere(picked.numpy()).tolist()
            assert store.delta[g, at[g]].tolist() == [g, -1, e] and (store.n_delta - at).tolist() == [g == 0, g == 1]


def test_only_decode_steps_count_and_decay(tiny_moe, tmp_path):
    """F11 decays and admits once per step, from its decode rows (docs/v1-offload.md D4): a one-token last prompt
    chunk is a prefill row (the scheduler's `row_kind`), beside other prompts even past the step's first 512-token
    expert pass (prompts of 1, 513 and 1 tokens in one step) and in a step of nothing else (three one-token prompts),
    so the replay oracle still matches the device step by step; a plan without row kinds is refused."""
    from holon_judge.placement import parity, route_dump

    checkpoint, _ = tiny_moe
    file, _ = _gguf_and_dense(checkpoint, tmp_path, Q.Q8_0)
    engine = _engine(file, max_step_tokens=1100, max_model_len=600, host_memory=1 << 24, plugin_options={
        "moe.store": "std.moe.offload@1", "moe.cache": "*=0.25"})
    mixed = [torch.randint(0, 512, (n,), generator=torch.Generator().manual_seed(n)).tolist() for n in (1, 513, 1)]
    for prompts in (mixed, [[5], [6], [7]]):               # the second from the first's residency and scores
        with route_dump(engine) as trace:
            engine.generate(prompts, SamplingParams(temperature=0.0, max_tokens=6, ignore_eos=True))
        report = parity(trace)
        assert report["first_differing_step"] is None and trace.mirrored and (report["admissions"] > 0 or
                                                                              max(map(len, prompts)) == 1), report
        assert not trace.steps[0]["counts"].any()                   # the prefill step: three prompts, none counted
    store = engine.ctx.shared_objects[KEY]
    with pytest.raises(ValueError, match="without row kinds"):            # no guess: every launched plan has them
        store.before_launch(host_plan(0, np.zeros(1, np.int32), np.zeros(1, np.int32), np.ones(1, np.int32),
                                      np.ones(1, np.int32), np.zeros(1, np.int64)))


@pytest.mark.parametrize("policy, copy", [("", "sm"), ("placement.lru@1", "sm"), ("", "dma")])
def test_prefill_steps_borrow_slots_and_give_them_back(tiny_moe_deep, tmp_path, policy, copy):
    """D4's device lending and D8 on the CPU: a step of 10 prompt tokens (its whole budget) streams every layer's
    missing experts through a ring of lent slots (`moe.ring_tokens`), a smaller one stages its misses in lent slots
    (its staging slots hold a graphed step's, 3 rows × top-2); the slots come back empty, each its layer's first
    victim: the outputs are the resident store's, the cache never shrinks (per class, resident pieces + empty slots
    stay the seed's), and F11's residency, empty slots included, is its replay oracle's at every step — mixed steps
    (decode rows and a prompt chunk, `chunk_with_decode`) included."""
    from holon_judge.placement import parity, route_dump

    checkpoint, prompts = tiny_moe_deep
    file, _ = _gguf_and_dense(checkpoint, tmp_path, Q.Q8_0)
    want = _served(_engine(file), prompts)
    engine = _engine(file, host_memory=1 << 24, chunk_with_decode=4, plugin_options={
        "moe.store": "std.moe.offload@1", "moe.cache": "*=0.5", "moe.placement": policy, "moe.ring_tokens": 10,
        "moe.ring_copy": copy})
    store, mixed = engine.ctx.shared_objects[KEY], []
    store.taps.append(lambda step, record: mixed.append(step.n_decode > 0))
    offload = engine.executor.model.layers[0].ffn._store
    assert store.staging == 6 and store.prefill == 8 and store.lending
    assert offload.pass_tokens(10) == PASS and offload.pass_tokens(9) == CHUNK     # the worst step profiles a pass
    held = []

    def size(step, record):           # per class: resident pieces + empty slots
        per = np.zeros(len(store.classes), dtype=np.int64)
        for g, row in enumerate(store.residency.host):
            for r in range(store.roles):
                per[store.class_table[g, r]] += int((row[:, r] >= 0).sum()) + len(store.residency.empty[g])
        held.append(per)
    store.taps.append(size)
    fresh = [[(t + 11) % 512 for t in p] for p in prompts]        # no prefix-cache hit: their prompts mix with decodes
    with route_dump(engine) as trace:
        assert _served(engine, prompts) == want
        assert engine.generate(fresh, GREEDY) == _engine(file).generate(fresh, GREEDY)
    assert store.borrowed["ring"] and store.borrowed["stage"] and any(mixed), store.borrowed
    assert all((h == held[0]).all() for h in held) and held[0].tolist() == [16, 16]
    assert trace.mirrored
    if not policy:
        report = parity(trace)
        assert report["first_differing_step"] is None and report["lends"] and report["admissions"], report


def test_the_startup_profile_runs_a_streaming_pass(tiny_moe, tmp_path):
    """The worst step of the startup profile runs on minimum pools, where a fill cache holds nothing to lend: its
    expert passes are still a streaming step's (sized by the step's tokens, not by the layout). At batch 1 (staging
    for one row, a pass's for prefill) the minimum pool still holds the layout that stages a pass."""
    checkpoint, _ = tiny_moe
    file, _ = _gguf_and_dense(checkpoint, tmp_path, Q.Q8_0)
    engine = _engine(file, max_seqs=1, host_memory=1 << 24, plugin_options={"moe.store": "std.moe.offload@1",
                                                                             "moe.ring_tokens": 10})
    store = engine.ctx.shared_objects[KEY]
    assert store.staging < store.prefill
    staged = sum((c.null + store.prefill) * c.unit_bytes + store._align for c in store.classes.values())
    assert store.arena.min_units * store.arena.unit_bytes >= staged      # what it lays out below, alignment aside
    engine.ledger.materialize(None)                       # the profile's minimum pools
    offload = engine.executor.model.layers[0].ffn._store
    assert not store.lending and offload.pass_tokens(10) == PASS


def test_a_cache_too_small_to_lend_stages_instead(tiny_moe, tmp_path):
    """A layout whose cache cannot lend a prefill pass's staging (a fill pool of 3 experts per class over the paged
    floor, F11 with the ring on) keeps its staging slots at a pass's size and never borrows: it serves as before. One
    that lends a pass's staging but not a ring's two halves (two half-cached layers: every lent expert is a miss of
    one of them) stages its ring steps in lent slots instead — the device's choice — under parity."""
    from holon_judge.placement import parity, route_dump

    checkpoint, prompts = tiny_moe
    file, _ = _gguf_and_dense(checkpoint, tmp_path, Q.Q8_0)
    want = _served(_engine(file), prompts)
    engine = _engine(file, host_memory=1 << 24, paged_min_tokens=1790, plugin_options={
        "moe.store": "std.moe.offload@1", "moe.ring_tokens": 10})
    store = engine.ctx.shared_objects[KEY]
    assert not store.lending and store.stage.shape[1] == store.prefill
    assert _served(engine, prompts) == want and store.borrowed == {"ring": 0, "stage": 0}
    engine = _engine(file, host_memory=1 << 24, plugin_options={
        "moe.store": "std.moe.offload@1", "moe.cache": "*=0.5", "moe.ring_tokens": 10})
    store = engine.ctx.shared_objects[KEY]
    with route_dump(engine) as trace:
        assert _served(engine, prompts) == want
    assert store.lending and store.borrowed["ring"] == 0 and store.borrowed["stage"] > 0
    assert parity(trace)["first_differing_step"] is None and trace.mirrored
    one = np.ones(10, np.int32)
    borrowed, plan = dict(store.borrowed), host_plan(99, np.arange(10, dtype=np.int32), 0 * one, one, one,
                                                    np.zeros(10, np.int64))
    plan.row_kind = np.full(10, RowKind.PREFILL, dtype=np.int8)       # ten one-token last chunks: a graphed step,
    store.before_launch(plan)                                          # whose forward (no Python) borrows nothing
    store.after_commit(plan, {name: t.clone() for name, t in store.telemetry.items()})
    assert store.borrowed == borrowed


def test_the_delta_log_holds_a_steps_admissions_and_lending(tiny_moe, tmp_path):
    """The delta log has a region of 2 × items records per layer (docs/v1-offload.md D4): a decode step of k · rows ≥ E
    routes whose first layer admits and then lends every resident expert of every layer (in the forward, after the
    step's first resolve, as the ring will) logs it all without overflow, the host mirror follows the device, and the
    outputs stay the resident store's."""
    from holon_judge.placement import route_dump
    from holon_std.tiered.resolver import DeviceResolver

    checkpoint, prompts = tiny_moe
    file, _ = _gguf_and_dense(checkpoint, tmp_path, Q.Q8_0)
    want = _served(_engine(file, max_seqs=4), prompts)
    engine = _engine(file, max_seqs=4, host_memory=1 << 24, plugin_options={
        "moe.store": "std.moe.offload@1", "moe.cache": "*=0.5"})
    store, call, lent = engine.ctx.shared_objects[KEY], DeviceResolver.__call__, []

    def resolve(self, ids, first):
        slots = call(self, ids, first)
        if not self.group and not first and not lent and bool((store.kinds == 1).sum() >= 4):   # 4 rows × top-2
            lent.append(int(store.lend(torch.full((2,), store.items * store.roles)).sum()))
        return slots
    with route_dump(engine) as trace, mock.patch.object(DeviceResolver, "__call__", resolve):
        assert _served(engine, prompts) == want
    step = next(s for s in trace.steps if (s["records"][:, 1] < 0).any())
    assert lent == [8] and (step["records"][:, 1] < 0).sum() == 8 and (step["records"][:, 1] >= 0).any()
    assert store.delta.shape[1] == 2 * store.items and trace.mirrored and not store.residency.host.max() >= 0


def test_a_route_transform_drops_routes_end_to_end(tiny_moe, tmp_path):
    """The route-transform call site (docs/v1-offload.md D3, `moe.route_transform`): a transform plugin's contract
    runs between the top-k and the resolve, with the layer's map; a dropped route (here every token's last rank) reads
    its rank's null slot and counts nowhere, so the outputs are those of the top-k with that rank's weight zeroed (to
    rounding on the CPU, whose reference GEMMs round a row by the expert's batch: dropping one changes the others'). A
    resident store refuses the option."""
    from holon.api import Manifest
    from holon.core.plugins import Installed, PluginSet, discover
    from holon_judge.placement import route_dump
    from holon_judge.planted import _moe

    from .fixtures.drop_last.plugin import DropLast

    checkpoint, prompts = tiny_moe
    file, _ = _gguf_and_dense(checkpoint, tmp_path, Q.Q8_0)
    with _moe(topk=lambda weights, ids, experts: (torch.cat([weights[:, :-1], 0 * weights[:, -1:]], 1), ids)):
        want = _served(_engine(file), prompts)
    found = discover()
    fixture = pathlib.Path(__file__).parent / "fixtures" / "drop_last"
    found["tests.drop_last"] = Installed(Manifest.load(fixture / "plugin.toml"), "tests.fixtures.drop_last", fixture)
    options = {"moe.store": "std.moe.offload@1", "moe.cache": "*=0.5", "moe.route_transform": "drop_last"}
    engine = Engine(EngineConfig(model=str(file), device="cpu", dtype=torch.float32, page_size=4, max_seqs=3,
                                 max_step_tokens=10, max_model_len=40, state_memory=1 << 20, host_memory=1 << 24,
                                 plugin_options=options), plugins=PluginSet(found))
    calls = DropLast.calls
    with route_dump(engine) as trace:
        streams, lps = _served(engine, prompts)
    assert streams == want[0]
    for (token, top_ids, _), (token_w, top_ids_w, _) in zip(lps, want[1], strict=True):
        torch.testing.assert_close(torch.tensor(token), torch.tensor(token_w), atol=1e-4, rtol=1e-4)
        assert top_ids == top_ids_w
    counts = np.stack([s["counts"].sum(1) for s in trace.steps])       # per step and layer: one route per decode row
    assert DropLast.calls > calls and (counts[:, 0] == counts[:, 1]).all() and counts.max() == 3
    with pytest.raises(ValueError, match="only an offload store serves"):
        Engine(EngineConfig(model=str(file), device="cpu", dtype=torch.float32, page_size=4, max_seqs=3,
                            max_step_tokens=10, max_model_len=40, state_memory=1 << 20,
                            plugin_options={"moe.route_transform": "drop_last"}), plugins=PluginSet(found))


@pytest.fixture(scope="module")
def gpu_moe(tmp_path_factory):
    """A random Qwen3-MoE as a GGUF of two expert classes, hidden and expert widths 1024 (an mmq tile then runs
    several K iterations: where stream-k engages it splits them), its prompts (decode-length to chunked prefill),
    and the resident store's outputs on the GPU."""
    torch.manual_seed(1)
    cfg = Qwen3MoeConfig(vocab_size=512, hidden_size=1024, intermediate_size=1024, moe_intermediate_size=1024,
                         num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=2, head_dim=128,
                         num_experts=8, num_experts_per_tok=2, norm_topk_prob=True, tie_word_embeddings=False,
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


@contextlib.contextmanager
def _no_host_sync(cls, *methods, mode="error"):
    """`cls`'s methods raise on any operation that waits for the device (a borrowing step's lending and ring), or
    (`mode` "warn") warn of each."""
    def strict(fn):
        def run(*args, **kw):
            torch.cuda.set_sync_debug_mode(mode)
            try:
                return fn(*args, **kw)
            finally:
                torch.cuda.set_sync_debug_mode("default")
        return run
    with contextlib.ExitStack() as stack:
        for name in methods:
            stack.enter_context(mock.patch.object(cls, name, strict(getattr(cls, name))))
        yield


def _gpu_engine(model, **kw) -> Engine:
    return Engine(EngineConfig(model=str(model), device="cuda", dtype=torch.bfloat16, page_size=16, max_seqs=4,
                               max_step_tokens=256, max_model_len=512, cuda_graphs=True, overlap=True, **kw))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("policy, cache", [("", ""), ("", "*=0.25"), ("", "*=0"), ("placement.lru@1", "*=0.25"),
                                           ("placement.profile_decay@1", "*=0.25"), ("std.tiered.static@1", "*=0.25"),
                                           ("", "*=0.5 ring"), ("", "*=0.5 dma")])
def test_the_offload_store_on_the_gpu_is_bitwise_the_resident_store(gpu_moe, policy, cache):
    """docs/v1-offload.md §6's exact class on the GPU kernels (kernels.ggml_cuda over class arenas, kernels.expert_io's
    resolve, its F11 and F4 variants and zero-copy gather from mapped host pools), decode steps replayed from CUDA
    graphs and prefill chunked past the mmvq limit: the offload store's streams and prompt logprobs are the resident
    store's, bit for bit, under every placement policy (arenas of 42, 18, 10 slots: stream-k unpatched). F11's
    residency, step by step, is its replay oracle's (L1 parity), with admissions in graphed and eager steps, the ring's
    copies on SMs or on copy engines; a borrowing step's lending and ring never wait for the device (with copy
    engines, but for one explicit sync to read the copy list). A
    corrupt map entry is flagged by the resolve and stops the engine at the commit; the kernels' id-error word is
    watched too. The arena pool is released when the ledger frees its pools (the startup profile reads free memory
    then)."""
    from holon_judge.placement import parity, route_dump

    file, prompts, want = gpu_moe
    cache, ring = cache.partition(" ")[::2]          # "ring" / "dma": steps of 32 prompt tokens or more stream (D8),
    engine = _gpu_engine(file, host_memory=1 << 27, plugin_options={       # their copies on SMs / copy engines
        "moe.store": "std.moe.offload@1", "moe.cache": cache, "moe.placement": policy,
        "moe.ring_tokens": 32 if ring else 512, "moe.ring_copy": "dma" if ring == "dma" else "sm"})
    store = engine.ctx.shared_objects[KEY]
    commit, routed = store.after_commit, []          # per step: its decode rows' routes, and the counts the step left
    store.after_commit = lambda step, t: routed.append((int(step.n_query.sum()) * 2 * 4 * bool(
        (step.n_query == step.n_logits).all()), int(t["counts"].sum()))) or commit(step, t)   # decode, verify steps
    mode = "warn" if ring == "dma" else "error"                   # "dma": one sync per streaming step, counted
    with (route_dump(engine) as trace, warnings.catch_warnings(record=True) as syncs,
          _no_host_sync(Lent, "begin", "sources", "done", mode=mode)):
        warnings.simplefilter("always")
        assert _served(engine, prompts) == want
    syncs = [w for w in syncs if "called a synchronizing" in str(w.message)]
    assert len(syncs) == (store.borrowed["ring"] if ring == "dma" else 0), (len(syncs), store.borrowed)
    assert engine.stats()["graphs"] > 0 and (store.peak_misses > 0) == (cache != "")
    assert all(a == b for a, b in routed) and max(routed)[0] > 8      # top-2 x 4 layers; graph padding never counts
    variant = {"": "moe_resolve.lfu@1", "placement.lru@1": "moe_resolve.lru@1"}.get(policy, "moe_resolve@1")
    assert {(k.rpartition("/")[2], v["provider"]) for k, v in engine.ctx.kernels.items() if k.startswith("std.tiered")
            } == {(variant, "kernels.expert_io"), ("expert_fetch@1", "kernels.expert_io"),
                  ("expert_prefetch@1", "kernels.expert_io")}
    if not policy:
        report = parity(trace)
        assert report["first_differing_step"] is None and (report["admissions"] > 0) == (cache in ("*=0.25", "*=0.5"))
        assert report["device_is_mirror"] and bool(store.borrowed["ring"]) == bool(ring), (report, store.borrowed)
    elif policy == "placement.profile_decay@1":
        assert store.policy.rounds > 0
    experts = {k.rpartition("/")[2]: v["provider"] for k, v in engine.ctx.kernels.items() if "linear_experts" in k}
    assert experts == {"linear_experts_arena@1": "kernels.ggml_cuda"}            # arenas never bind @2
    words = [t for name, t in store.telemetry.items() if name.startswith("kernel_errors")]
    assert len(words) == 1 and all(m._kernel.errors is words[0] for m in engine.executor.model.modules()
                                   if hasattr(m, "cls"))          # the ggml id-error word, one per engine
    if not cache:                                # a corrupt map entry: null slots read, the commit stops the engine
        store.residency.device[0, :, 0] = 1 << 20
        engine.add(prompts[0], GREEDY)
        with pytest.raises(RuntimeError, match=r"flagged errors \{'errors': 1\}"):
            while engine.has_work():
                engine.step()
    held, nbytes = torch.cuda.memory_allocated(), store.arena.capacity * store.arena.unit_bytes
    store.arena._set(0)                         # as the ledger frees it: no view outlives the arena pool's buffer
    assert held - torch.cuda.memory_allocated() >= nbytes > 0
    engine.close()
