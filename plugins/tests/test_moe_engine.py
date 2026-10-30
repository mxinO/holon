"""Qwen3-MoE end to end on the CPU (test-only reference kernels) against an independent oracle: a tiny random HF
`Qwen3MoeForCausalLM`, written as a GGUF with quantized expert stacks, must serve what transformers computes on the
same (dequantized) weights (docs/dev/m02-moe.md, step 1)."""
import collections
import contextlib
import json
import os
import pathlib
import subprocess
import sys
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
from holon_std.moe.kernels import DECODE, PADDING
from holon_std.moe.offload import KEY
from holon_std.moe.units import CHUNK, MoE
from holon_std.tiered.ring import Lent
from holon_testing.gguf_files import write
from holon_testing.ref_kernels.kernels import LinearExperts
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM

from .test_engine import _hf_greedy

PREBUILT = ("ggml_cuda", "expert_io", "cpu_task")  # CUDA extensions built if missing (conftest)


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


@pytest.mark.parametrize("store", ["resident", "offload"])
def test_a_prefill_step_passes_whole_chunks_then_its_tail(tiny_moe, tmp_path, monkeypatch, store):
    """docs/dev/v1-moe-prefill.md D1 on both stores (offload: a streaming step, the ring on): a step's expert passes
    take
    the smaller of the two stacks' `pass_rows` (the reference provider's: 4,096 within a 4,100-token step budget) in
    whole chunks, and the step's last partial chunk is a pass of its own, so every row's chunk and kernel path are a
    chunk's: calls of 512 then 88 rows at 600 tokens, 1,024 then 3 at 1,027, 4,096 then 4 at 4,100, each for gate/up
    then down in every layer. D10's fallback: each pass allocates one `scratch` of the larger of its kernels'
    `scratch_bytes`, passed to both calls (no `out`: no lent memory); a kernel that does not define `scratch_bytes` is
    called without the keywords. D3′: a router with `pass_rows` takes a pass in one call."""
    checkpoint, _ = tiny_moe
    file, _ = _gguf_and_dense(checkpoint, tmp_path, Q.Q8_0)
    offload = {"host_memory": 1 << 24, "plugin_options": {"moe.store": "std.moe.offload@1"}}
    offload = offload if store == "offload" else {}
    engine = _engine(file, max_step_tokens=4100, max_model_len=4200, state_memory=8 << 20, **offload)
    layers = len(engine.executor.model.layers)
    assert {m._store.pass_tokens(4100) for m in engine.executor.model.modules() if hasattr(m, "_gate_up")} == {4096}
    sizes, call = [], LinearExperts.__call__

    def recorded(self, x, parts, ids, **kw):
        sizes.append(len(ids))
        scratch = kw.get("scratch")
        buffers.append((sorted(kw), None if scratch is None else (scratch.data_ptr(), len(scratch))))
        return call(self, x, parts, ids, **kw)
    for tokens, passes in ((600, [512, 88]), (1027, [1024, 3]), (4100, [4096, 4]), (601, [512, 89])):
        sizes.clear()
        buffers: list = []
        prompt = torch.randint(0, 512, (tokens,), generator=torch.Generator().manual_seed(tokens)).tolist()
        with mock.patch.object(LinearExperts, "__call__", recorded):
            engine.generate([prompt], SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True))
        assert sizes == [n for n in passes for _ in range(2)] * layers, (tokens, sizes)
        if hasattr(LinearExperts, "scratch_bytes"):
            moe = next(m for m in engine.executor.model.modules() if hasattr(m, "_down"))
            want = [max(moe._gate_up.kernel.scratch_bytes(n), moe._down.kernel.scratch_bytes(n)) for n in passes]
            assert [kw for kw, _ in buffers] == [["scratch"]] * len(buffers)
            assert [b[1][1] for b in buffers[::2]] == want * layers and buffers[::2] == buffers[1::2]   # one a pass
            monkeypatch.delattr(LinearExperts, "scratch_bytes")       # the last prompt: a kernel without buffers
        else:
            assert buffers == [([], None)] * len(buffers)
    # D3′: a router declaring `pass_rows` routes a pass of whole CHUNKs in one call, the last partial chunk on its own
    moes = [m for m in engine.executor.model.modules() if hasattr(m, "_route_rows")]
    routed, route = [], MoE._route
    monkeypatch.setattr(MoE, "_route", lambda self, x: routed.append(len(x)) or route(self, x))
    for m in moes:
        m._route_rows = 4096
    prompt = torch.randint(0, 512, (4100,), generator=torch.Generator().manual_seed(1)).tolist()
    engine.generate([prompt], SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True))
    assert routed == [4096, 4] * layers, routed


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


def _served(engine, prompts, mirrored=None) -> tuple[list, dict]:
    """Greedy streams and prompt logprobs of `prompts` (`mirrored`: a store whose host mirror must be its device map,
    tier codes too, once the prompt steps are committed)."""
    rids = [engine.add(p, SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=3)) for p in prompts]
    got = {}
    while engine.has_work():
        got |= {d.rid: d.prompt_logprobs for d in engine.step() if d.prompt_logprobs is not None}
    if mirrored is not None:
        assert np.array_equal(mirrored.residency.device.cpu().numpy(), mirrored.residency.host)
    return engine.generate(prompts, GREEDY), [(got[r].token, got[r].top_ids, got[r].top) for r in rids]


def test_the_offload_store_serves_what_the_resident_store_does(tiny_moe, tmp_path, capsys):
    """docs/dev/v1-offload.md D4, D5, F1, F5a: with `moe.store = std.moe.offload@1` the experts stay in the file's map
    and in pinned host pools (filled from its extents), classes by (role, type, shape) — here gate/up Q8_0 and down
    Q4_0 —, every miss fetched into staging slots: the outputs are bitwise the resident store's whatever the cache
    holds (every expert by default, as many as a short fill pool's bytes hold, half, a profile's pick per class, none).
    The static policy caches whole experts, F2's warm start: item 0, 1, … of every layer, or a profile's hottest; a
    layer's distinct misses never pass its staging slots, and a step that did would stop the engine. The adaptive
    policies (the default F11, F4, F2–F3's two-phase swaps through the copy stream) serve the same outputs at a
    quarter of the experts while they move them; F11's residency, step by step, is its replay oracle's (L1 parity)."""
    import json as json_

    from holon_judge.placement import parity, route_dump
    from holon_tools.__main__ import main  # `holon_tools profile`: decode routes, beside the file

    checkpoint, prompts = tiny_moe
    file, _ = _gguf_and_dense(checkpoint, tmp_path, Q.Q8_0, types={"ffn_down_exps": Q.Q4_0})
    want = _served(_engine(file), prompts)
    profile = tmp_path / "profile.json"
    hot = [[3, 1], [6, 5]]                                       # per layer: the hottest experts, then the next
    counts = np.zeros((2, 8))
    for layer, (a, b) in enumerate(hot):
        counts[layer, [a, b]] = [9, 5]
    profile.write_text(json_.dumps({"checkpoint": [[file.name, file.stat().st_size]],
                                    "layers": {str(layer): row for layer, row in enumerate(counts.tolist())}}))
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
            # Its counts pin exactly (D4: a lock pins the fill's slot counts); a pinned arena that does not get its
            # size — one expert more than fits above the paged floor, or a budget leaving every pool at its minimum —
            # refuses to serve at its first launch (the startup profile's minimum pools never launch), never taking
            # the floor or running without its cache.
            pins = [",".join(f"{c.name}={c.size - c.null - c.staging + more}" for c in store.classes.values())
                    for more in (0, 1)]
            exact = _engine(file, host_memory=1 << 24, paged_min_tokens=floor, plugin_options={
                **static, "moe.cache": pins[0]})
            assert (exact.ctx.shared_objects[KEY].residency.host >= 0).all(-1).sum(-1).tolist() == per_layer
            assert _served(exact, prompts) == want
            least = sum(p.min_units * p.unit_bytes for p in exact.ledger.pools.values() if not p.host)
            for kw, pin in (({"paged_min_tokens": floor}, pins[1]), ({"state_memory": least}, pins[0])):
                short = _engine(file, host_memory=1 << 24, plugin_options={**static, "moe.cache": pin}, **kw)
                with pytest.raises(RuntimeError, match="does not fit above the paged floor"):
                    _served(short, prompts)
        assert per_layer == [cached] * 2 or floor
        if extra:
            assert [sorted(np.flatnonzero(r).tolist()) for r in resident[..., 0]] == [[3], [6]]
        elif cached:
            assert resident[:, :per_layer[1]].all()
        assert 0 < store.peak_misses <= store.staging if cached < 8 else store.peak_misses == 0
    with pytest.raises(ValueError, match="relative to the checkpoint's directory"):   # plans record no host path
        _engine(file, host_memory=1 << 24, plugin_options={"moe.store": "std.moe.offload@1",
                                                           "moe.placement.profile": str(profile)})
    capsys.readouterr()
    cold = _engine(file, host_memory=1 << 24, plugin_options={"moe.store": "std.moe.offload@1", "moe.cache": "*=0.25",
                                                              "moe.placement.profile": "absent.profile.json"})
    assert _served(cold, prompts) == want                    # a lock's profile not written here yet: said, served cold
    assert capsys.readouterr().out.count("no routing profile") == 1
    store.stage = store.stage[:, :store.peak_misses - 1].contiguous()       # as if staging had been too small
    for p in prompts:
        engine.add([(t + 7) % 512 for t in p], GREEDY)                       # new prompts (no prefix-cache hit)
    with pytest.raises(RuntimeError, match="distinct misses"):
        while engine.has_work():
            engine.step()
    # `holon_tools profile` on a hub cache's snapshot link writes `<stem>.profile.json` beside the blob, where the
    # store reads it, recording the file; another file's profile in that directory is refused, not used.
    link = tmp_path / "snapshot" / "moe-a.gguf"
    link.parent.mkdir()
    link.symlink_to(file)
    (tmp_path / "prompts.jsonl").write_text("".join(json_.dumps({"id": str(n), "category": "c", "ids": p}) + "\n"
                                                    for n, p in enumerate(prompts)))
    assert main(["profile", str(link), "--prompts", str(tmp_path / "prompts.jsonl"), "--max-tokens", "4", "--device",
                 "cpu", "--dtype", "float32", "--page-size", "4", "--max-seqs", "3", "--max-step-tokens", "10",
                 "--max-model-len", "40", "--state-memory", str(1 << 20), "--host-memory", str(1 << 24)]) == 0
    routes = json_.loads((tmp_path / "moe-a.profile.json").read_text())
    assert routes["checkpoint"] == [[file.name, file.stat().st_size]]
    assert sorted(routes["layers"]) == ["0", "1"] and sum(map(sum, routes["layers"].values())) == 4 * 3 * 2 * 2
    (tmp_path / "other").mkdir()
    other, _ = _gguf_and_dense(checkpoint, tmp_path / "other", Q.Q4_0)
    other = other.rename(tmp_path / other.name)                     # another quant beside it in the blob directory
    with pytest.raises(ValueError, match="was counted on"):
        _engine(other, host_memory=1 << 24, plugin_options={"moe.store": "std.moe.offload@1",
                                                            "moe.placement.profile": "moe-a.profile.json"})
    for policy in ("", "placement.lru@1", "placement.profile_decay@1"):
        engine = _engine(link, host_memory=1 << 24, plugin_options={
            "moe.store": "std.moe.offload@1", "moe.cache": "*=0.25", "moe.placement": policy,
            "moe.placement.profile": "moe-a.profile.json"})
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
    """F11 decays and admits once per step, from its decode rows (docs/dev/v1-offload.md D4): a one-token last prompt
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
    # the worst step profiles a pass (of the bound kernels' pass_rows: one chunk within this 10-token step budget)
    assert store.large_passes(10) and not store.large_passes(9) and offload.pass_tokens(10) == CHUNK
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


@pytest.fixture(scope="module")
def wide_moe(tmp_path_factory):
    """FN's MoE shape at a tiny width (docs/dev/v1-moe-prefill.md §5.1): 512 experts, top-10, two layers, as a GGUF of
    Q8_0 stacks, its prompts and the resident store's outputs."""
    torch.manual_seed(2)
    cfg = Qwen3MoeConfig(vocab_size=512, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         num_experts=512, num_experts_per_tok=10, norm_topk_prob=True, tie_word_embeddings=False,
                         eos_token_id=None)
    path = tmp_path_factory.mktemp("wide_moe")
    model = Qwen3MoeForCausalLM(cfg)
    with torch.no_grad():
        for layer in model.model.layers:
            layer.mlp.gate.weight.mul_(8)
            layer.mlp.experts.gate_up_proj.mul_(20)
            layer.mlp.experts.down_proj.mul_(5)
    model.save_pretrained(path / "hf")
    write(path / "hf", path / "moe.gguf", Q.Q8_0)
    prompts = [torch.randint(0, 512, (n,)).tolist() for n in (3, 17, 8, 25)]
    return path / "moe.gguf", prompts, _served(_engine(path / "moe.gguf", state_memory=1 << 24), prompts)


@contextlib.contextmanager
def _tail_checked(store):
    """D11's terms, asserted around `Lent` on every step that lends a tail (its tables copied on the device as the step
    goes, checked at its end: no host sync inside the step): call 0 lends n slots of the tail's class (`lends` row 0)
    and evicts, in every role, every expert holding a slot of the range, which the rounds then never pick; the range
    (`tail_view`) is the class's last n slots, contiguous, and holds the bytes its clients' passes in this step take
    (no more slots than those, nor than the tail's); no slot of the
    range is in the pool, the stage lists or the ring, nor mapped — at the step's start and after every layer —; what
    the step holds per class is what its pool holds (the range counted in neither), so a held ring has a slot for every
    missing piece and no step is short; at its end the empty slots it found lie below the lent ones, as they were, and
    the lent ones lie above them whole, by call then expert id. Yields each tail step's (empty slots found, `lends`
    rows, ring held)."""
    begin, done, steps = Lent.begin, Lent.done, []

    def checked_begin(self):
        before, found = store.residency.device.clone(), (store.empty.clone(), store.n_empty.clone())
        begin(self)
        if self.tail is None:
            return
        cls, n = self.tail
        c, classes = list(store.classes.values())[cls], store._class_index
        view = store.tail_view()
        need = max(f(store.step_tokens) for f in store.tail_bytes)       # the step's passes', at most the tail's
        assert n <= store.tail[1] and need <= max(f() for f in store.tail_bytes)
        assert view.is_contiguous() and len(view) == n * c.unit_bytes >= need > 0 and len(view) < need + c.unit_bytes
        assert view.data_ptr() == c.slots()[c.size - n].data_ptr()
        self._snap = {"before": before, "found": found, "maps": [store.residency.device.clone()],
                      "held": self._held(classes), "want": self._want(store.residency.device, classes)} | {
            name: getattr(store, name).clone() for name in ("pool", "stage_tab", "ring", "lent_call", "lend_needs",
                                                            "ring_ok", "errors")}

    def checked_done(self, g):
        done(self, g)
        if self.tail is None:
            return
        snap = self._snap
        snap["maps"].append(store.residency.device.clone())
        if g < len(store.layers) - 1:
            return
        t = {k: v.cpu() if isinstance(v, torch.Tensor) else [x.cpu() for x in v] for k, v in snap.items()}
        (cls, n), roles = self.tail, torch.from_numpy(store.class_table == self.tail[0])        # [G, R]
        lo, size = list(store.classes.values())[cls].size - n, list(store.classes.values())[cls].size

        def inside(slots):
            return (slots >= lo) & (slots < size)
        held = (inside(t["before"]) & roles[:, None, :]).any(-1)                             # [G, E]: the range's
        call, ring_ok = t["lent_call"], bool(t["ring_ok"])
        assert (t["maps"][0][held] == -1).all() and (call[held] == 0).all()
        assert (call != 0).eq(~held).all()                                  # the rounds never pick a holder
        rounds = store.lent_slots.cpu().permute(0, 2, 1)[roles][(call > 0)[roles.any(1)]]
        assert (rounds < lo).all(), "a round lent a slot past the step's lend end (held would count it, the pool not)"
        assert t["lend_needs"][0].tolist() == [n * (k == cls) for k in range(len(store.classes))]
        assert not any(inside(m.permute(0, 2, 1)[roles]).any() for m in t["maps"]), "a piece mapped in the tail"
        assert not inside(t["pool"][cls, :-1]).any(), "the pool holds the tail"              # (past it: the sink)
        assert torch.equal(t["held"], (t["pool"][:, :-1] >= 0).sum(1)), "held counts other slots than the pool's"
        stage, ring = t["stage_tab"][roles], t["ring"].permute(0, 2, 1)[roles]
        assert not inside(stage).any() and not inside(ring).any()
        missing = t["maps"][0].permute(0, 2, 1)[roles] < 0
        assert not ring_ok or ((t["held"] >= t["want"]).all() and (ring[missing] >= 0).all())
        assert not int(t["errors"]) & 4                                                     # never short
        (stack, n_found), empty, lent_slots = t["found"], store.empty.cpu(), store.lent_slots.cpu()
        for layer in range(len(store.layers)):                  # given back: found below, then the lent, whole
            k = int(n_found[layer])
            order = sorted(call[layer].ge(0).nonzero()[:, 0].tolist(), key=lambda e: (int(call[layer, e]), e))
            assert int(store.n_empty[layer]) == k + len(order)
            assert torch.equal(empty[layer, :, :k], stack[layer, :, :k])
            assert torch.equal(empty[layer, :, k:k + len(order)], lent_slots[layer, order].t())
        steps.append((int(n_found.sum()), t["lend_needs"].numpy(), ring_ok))
    with mock.patch.object(Lent, "begin", checked_begin), mock.patch.object(Lent, "done", checked_done):
        yield steps


@pytest.mark.parametrize("stream", [0, 20])
def test_a_tail_step_lends_its_passes_a_contiguous_range(wide_moe, capsys, stream):
    """docs/dev/v1-moe-prefill.md D10, D11 on the CPU at FN's E = 512, top-10, under F11 (`moe.ring_lend = tail`): a
    streaming step first lends the tail — call 0, the last slots of the largest class arena (gate/up's), enough for
    the passes' `out` and `scratch` —, then X's rounds (calls 1…), every step's terms as `_tail_checked` asserts them;
    the first, on a cache with no spare empty slots, stages from the rounds' slots without falling short. The passes'
    buffers are the range's (the reference provider stages each chunk through `scratch`, then `out`), so an expert
    left mapped there would read them as weights: the outputs are the resident store's, mixed steps included, and F11's
    residency, empty slots and their stacks are its replay's step by step (the replay models `lend_tail` from the
    slots). With `moe.stream_tokens` 20, a step of 10 to 19 prompt tokens stages its routed misses in large passes
    instead, under the same terms (its tail lent, its rounds' slots its staging). The plant that leaves the tail's
    experts mapped fires and fails parity. A configuration whose tail class cannot stage a pass outside the range, and
    every policy without `lend_tail` (F11 and F10 have it), are refused at bind."""
    from holon_judge.placement import parity, route_dump
    from holon_judge.planted import tail_unmap_undone
    from holon_plugins.placement.lfu.reference import replay

    file, prompts, want = wide_moe
    tail = {"moe.store": "std.moe.offload@1", "moe.cache": "*=0.75", "moe.ring_tokens": 10, "moe.ring_lend": "tail",
            "moe.stream_tokens": stream}
    engine = _engine(file, state_memory=1 << 24, host_memory=1 << 26, chunk_with_decode=4, plugin_options=tail)
    store = engine.ctx.shared_objects[KEY]
    assert store.tail == (0, 15) and list(store.classes)[0].startswith("gate_up") and store.prefill == 512
    with route_dump(engine) as trace, _tail_checked(store) as steps:
        assert _served(engine, prompts) == want
    # (a staging step lends no rounds where the empty slots it found hold its staging)
    assert steps and steps[0][0] == 0 and all((s[1][0] > 0).sum() == 1 and (s[1][1:].any() or stream and s[0])
                                              for s in steps), steps
    report = parity(trace)
    assert report["first_differing_step"] is None and trace.mirrored and report["admissions"], report
    # parity compared the empty slots' stacks step by step (the host's, in the delta log's order); the device's too
    stacks = replay(trace.seed, trace.steps, trace.score, trace.classes, trace.slots, trace.sizes, trace.stacks)[-1]
    n_empty = store.n_empty.tolist()
    assert n_empty == stacks["empty"].tolist() and all(
        np.array_equal(store.empty[g, :, :n].t().numpy(), stacks["stacks"][g]) for g, n in enumerate(n_empty))
    if stream:
        assert store.borrowed["stage"] and not store.borrowed["ring"] and not any(s[2] for s in steps), store.borrowed
        return
    with tail_unmap_undone():
        planted = _engine(file, state_memory=1 << 24, host_memory=1 << 26, chunk_with_decode=4, plugin_options=tail)
        with route_dump(planted) as trace:
            _served(planted, prompts)
    fired = int(capsys.readouterr().err.split("tail-unmap-undone fired: {'lent': ")[1].split("}")[0])
    assert fired > 0 and parity(trace)["first_differing_step"] is not None
    with pytest.raises(ValueError, match="cannot stage a prefill pass"):
        _engine(file, state_memory=1 << 24, host_memory=1 << 26, plugin_options=tail | {"moe.cache": "*=0.5"})
    for policy in ("placement.lru@1", "placement.profile_decay@1"):      # (F10 lends a tail: docs/dev/v1-glm.md D8a)
        with pytest.raises(ValueError, match="does not lend a tail"):
            _engine(file, state_memory=1 << 24, host_memory=1 << 26, plugin_options=tail | {"moe.placement": policy})


def test_the_startup_profile_runs_a_streaming_pass(tiny_moe, tmp_path):
    """The worst step of the startup profile runs on minimum pools, where a fill cache holds nothing to lend: its
    expert passes are still a streaming step's (sized by the step's tokens, not by the layout). At batch 1 (staging
    for one row, a pass's for prefill) the minimum pool still holds the layout that stages a pass. Under
    `ring_lend = "tail"` the minimum also holds the tail's bytes, so the passes' buffers come from the pool — a lent
    tail, or the staging layout's own — never from the activations the profile measures; it serves the resident
    store's outputs."""
    checkpoint, prompts = tiny_moe
    file, _ = _gguf_and_dense(checkpoint, tmp_path, Q.Q8_0)
    engine = _engine(file, max_seqs=1, host_memory=1 << 24, plugin_options={"moe.store": "std.moe.offload@1",
                                                                             "moe.ring_tokens": 10})
    store = engine.ctx.shared_objects[KEY]
    assert store.staging < store.prefill
    staged = sum((c.null + store.prefill) * c.unit_bytes + store._align for c in store.classes.values())
    assert store.arena.min_units * store.arena.unit_bytes >= staged      # what it lays out below, alignment aside
    engine.ledger.materialize(None)                       # the profile's minimum pools
    offload = engine.executor.model.layers[0].ffn._store
    assert not store.lending and store.large_passes(10)
    assert offload.pass_tokens(10) == min(s.kernel.pass_rows for s in offload._stacks) == CHUNK
    want = _served(_engine(file, max_seqs=1), prompts)
    engine = _engine(file, max_seqs=1, host_memory=1 << 24, plugin_options={
        "moe.store": "std.moe.offload@1", "moe.ring_tokens": 10, "moe.ring_lend": "tail"})
    store = engine.ctx.shared_objects[KEY]
    engine.ledger.materialize(None)
    assert store.lending or store._own_tail is not None
    assert _served(engine, prompts) == want


def test_a_floored_class_is_seeded_its_hottest_items_first():
    """A class the seed leaves short of a pass's staging (FN's IQ1_M gate/up of 3 layers, an MTP head's) is floored:
    its groups' items in F2's order come first, so it can lend; the rest keep F2's order."""
    from holon_std.tiered.store import _Budget

    a, b, c = (types.SimpleNamespace(name=n, unit_bytes=1, cap=16) for n in "abc")
    store = types.SimpleNamespace(classes={"a": a, "b": b, "c": c}, groups={0: {0: a, 1: b}, 1: {0: c, 1: b}},
                                  _order={0: 0, 1: 1})
    profile = np.array([list(range(16, 8, -1)), [1] + [0] * 7])      # group 1 (class c) cold: F2's 4 of it only
    natural = _Budget(store, 18)
    assert [x for x in natural.warm_start(profile) if natural.take(*x)] == _Budget.f2(profile)[:9]
    assert natural.taken["c"] == 4
    floored = _Budget(store, 18, {"c": 6})
    order = floored.warm_start(profile)
    f2 = _Budget.f2(profile)
    assert order[:6] == [(1, i) for i in range(6)] and order[6:] == [x for x in f2 if x not in order[:6]]
    assert len([x for x in order if floored.take(*x)]) == 9 and floored.taken["c"] == 6


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
    """The delta log has a region of 2 × items records per layer (docs/dev/v1-offload.md D4), and one for the returned
    lent
    slots' marker (FN D12a): a decode step of k · rows ≥ E routes whose first layer admits and then lends every
    resident expert of every layer (in the forward, after the step's first resolve, as the ring will) logs it all
    without overflow, the host mirror follows the device, and the outputs stay the resident store's."""
    from holon_judge.placement import route_dump
    from holon_std.tiered.resolver import DeviceResolver

    checkpoint, prompts = tiny_moe
    file, _ = _gguf_and_dense(checkpoint, tmp_path, Q.Q8_0)
    want = _served(_engine(file, max_seqs=4), prompts)
    engine = _engine(file, max_seqs=4, host_memory=1 << 24, plugin_options={
        "moe.store": "std.moe.offload@1", "moe.cache": "*=0.5"})
    store, call, lent = engine.ctx.shared_objects[KEY], DeviceResolver.__call__, []

    def resolve(self, ids, first, *share, **kw):
        resolved = call(self, ids, first, *share, **kw)                 # (slots, cpu)
        if not self.group and not first and not lent and bool((store.kinds == 1).sum() >= 4):   # 4 rows × top-2
            lent.append(int(store.lend(torch.full((2,), store.items * store.roles)).sum()))
        return resolved
    with route_dump(engine) as trace, mock.patch.object(DeviceResolver, "__call__", resolve):
        assert _served(engine, prompts) == want
    step = next(s for s in trace.steps if (s["records"][:, 1] < 0).any())
    assert lent == [8] and (step["records"][:, 1] < 0).sum() == 8 and (step["records"][:, 1] >= 0).any()
    assert store.delta.shape[1] == 2 * store.items + 1 and trace.mirrored and not store.residency.host.max() >= 0


def test_a_route_transform_drops_routes_end_to_end(tiny_moe, tmp_path):
    """The route-transform call site (docs/dev/v1-offload.md D3, `moe.route_transform`): a transform plugin's contract
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
    # Graph padding reaches the transform dropped (every id −1), so a transform ranking a pass's rows (A1's best
    # rank) never counts it: a decode row beside a padding row, as in a padded bucket.
    engine.ctx.shared_objects[KEY].kinds[:2] = torch.tensor([DECODE, PADDING], dtype=torch.uint8)
    offload = engine.executor.model.layers[1].ffn._store
    got, _ = offload.transform(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), torch.ones(2, 2), 0)
    assert got.tolist() == [[1, -1], [-1, -1]]
    with pytest.raises(ValueError, match="only an offload store serves"):
        Engine(EngineConfig(model=str(file), device="cpu", dtype=torch.float32, page_size=4, max_seqs=3,
                            max_step_tokens=10, max_model_len=40, state_memory=1 << 20,
                            plugin_options={"moe.route_transform": "drop_last"}), plugins=PluginSet(found))


def test_the_cpus_share_computes_a_passs_lowest_ranked_misses(tiny_moe, tmp_path):
    """F5b (docs/dev/v1-offload.md D4–D7) on the CPU providers: with a split table (`moe.cpu_plan`; past its end the
    last
    entry's share), a pass of at most the graphed steps' tokens (decode, or a short prompt chunk) gives the CPU
    `plan[n]` of its n misses — computed by
    `cpu_experts@1`'s oracle, the activations quantized as ggml-cpu quantizes them, patched in before the combine —,
    larger passes none: outputs close to the resident store's (bounded), run-identical, F11's residency still its
    replay's step by step (the CPU's misses never admitted). On a host of another CPU tier the split is off (S5) and the
    outputs are the resident store's bit for bit."""
    from holon_judge.placement import parity, route_dump
    from holon_std.moe.cpu import plan_table
    from holon_std.moe.offload import CPU

    assert plan_table("1,1,2", 6) == [0, 1, 1, 2, 2, 3, 4] and plan_table("0,0", 4) is None
    with pytest.raises(ValueError, match="must lie in"):
        plan_table("2", 4)
    checkpoint, prompts = tiny_moe
    file, _ = _gguf_and_dense(checkpoint, tmp_path, Q.Q8_0, types={"ffn_down_exps": Q.Q4_0})
    want = _served(_engine(file), prompts)
    split = {"moe.store": "std.moe.offload@1", "moe.cache": "*=0.5", "moe.cpu_plan": "1,1,2,2"}
    runs = []
    for policy in ("std.tiered.static@1", "std.tiered.static@1", ""):
        engine = _engine(file, host_memory=1 << 24, plugin_options=split | {"moe.placement": policy})
        share, entries = engine.ctx.shared_objects[CPU], []

        def collect(slot, y, cpu, f=share.collect, entries=entries):
            entries.append(int((cpu >= 0).sum()))
            f(slot, y, cpu)
        share.collect = collect
        with route_dump(engine) as trace:
            runs.append(_served(engine, prompts))
        assert sum(entries) > 0 and len(entries) < 2 * len(trace.steps)         # passes of at most 3 tokens
        assert 0 < sum(s["cpu"].sum() for s in trace.steps) <= sum(entries)        # an expert per job, a row per token
        if not policy:
            assert parity(trace)["first_differing_step"] is None and trace.admissions > 0
    assert runs[0] == runs[1]                                                  # run-identical
    close = [np.allclose(a, b, atol=0.05) for (_, _, a), (_, _, b) in zip(runs[0][1], want[1], strict=True)]
    assert all(close)
    engine = _engine(file, host_memory=1 << 24, plugin_options=split | {"moe.cpu_isa": "another-tier"})
    assert CPU not in engine.ctx.shared_objects and _served(engine, prompts) == want
    # `moe.cpu_min_tokens`: passes of fewer tokens fetch all their misses (verify-only at batch 1, by size)
    engine = _engine(file, host_memory=1 << 24, plugin_options=split | {"moe.cpu_min_tokens": 2})
    share, rows = engine.ctx.shared_objects[CPU], []
    share.collect = lambda slot, y, cpu, f=share.collect: (rows.append(len(cpu)), f(slot, y, cpu))[1]
    assert np.allclose(_served(engine, prompts)[1][0][2], want[1][0][2], atol=0.05)
    assert share.takes(2) and not share.takes(1) and rows and min(rows) >= 2


def test_a_second_seed_reads_only_what_ram_did_not_hold(tiny_moe, tmp_path, monkeypatch):
    """Exclusive tiers seeded twice (the startup profile's minimum pools, then the final ones): the second seed reads
    from the checkpoint only the pieces RAM did not hold — a VRAM piece RAM held is copied out of its unit, a RAM piece
    stays in its unit — and leaves what a single seed does: every piece in the same tier and VRAM slot, each RAM unit
    holding the file's bytes of the piece the map names, the resident store's outputs."""
    from holon_std.tiered import files

    checkpoint, prompts = tiny_moe
    file, _ = _gguf_and_dense(checkpoint, tmp_path, Q.Q8_0, types={"ffn_down_exps": Q.Q4_0})
    short = {"moe.store": "std.moe.offload@1", "moe.cache": "*=0.25", "moe.ram": "*=0.25",
             "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"}
    once = _engine(file, host_memory=1 << 24, plugin_options=short)
    engine = _engine(file, host_memory=1 << 24, plugin_options=short)
    store, classes = engine.ctx.shared_objects[KEY], list(engine.ctx.shared_objects[KEY].classes.values())
    engine.ledger.materialize(None)
    ram = [set(store.owner[k][store.owner[k] >= 0].tolist()) for k in range(len(classes))]
    read = []
    real = files.read
    monkeypatch.setattr(files, "read", lambda jobs: read.append(sum(d.nbytes for _, d in jobs)) or real(jobs))
    engine.ledger.materialize(engine.cfg.state_memory)
    res = store.residency.device.cpu().numpy()
    new = sum(sum(e.nbytes for e in c.pieces[store._piece_index(c, g, i, r)][3])
              for g, layer in enumerate(store.layer_classes) for i in range(store.items) for r, c in enumerate(layer)
              if res[g, i, r] >= -1 and g * store.items + i not in ram[classes.index(c)])
    assert 0 < sum(read) == new < sum(sum(e.nbytes for e in p[3]) for c in classes for p in c.pieces)
    assert np.array_equal(res, once.ctx.shared_objects[KEY].residency.device.cpu().numpy())
    assert _bytes_exact(store) == 0
    _held_once(store)
    assert _served(engine, prompts) == _served(_engine(file), prompts)


def test_exclusive_tiers_serve_what_the_resident_store_does(tiny_moe, tmp_path):
    """docs/dev/v1-offload.md D10, F9: with RAM holding a quarter of the pieces (`moe.ram`) and the checkpoint's file
    the
    rest (std.tiered.gguf_direct@1), every expert lives in one of VRAM, RAM and the SSD: SSD misses are read by the
    copy task, F11's victims demoted into RAM (the admitted staged, then moved), decode SSD reads kept — and the
    outputs are the resident store's bit for bit, F11's VRAM residency its replay's, the RAM units owned once each and
    the map's tier codes where they say. Without a slower tier, or under a device policy with no heat (F4) or a host
    policy that edits (F2–F3), the store refuses a short RAM; a file tier's descriptors other than 'pread' extents are
    refused at prepare (the copy task reads that kind only; a page-cache tier is one with O_DIRECT off)."""
    from holon_judge.placement import parity, route_dump

    checkpoint, prompts = tiny_moe
    file, _ = _gguf_and_dense(checkpoint, tmp_path, Q.Q8_0, types={"ffn_down_exps": Q.Q4_0})
    want = _served(_engine(file), prompts)
    short = {"moe.store": "std.moe.offload@1", "moe.cache": "*=0.25", "moe.ram": "*=0.25",
             "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"}
    for policy in ("", "std.tiered.static@1"):
        engine = _engine(file, host_memory=1 << 24, plugin_options=short | {"moe.placement": policy})
        store = engine.ctx.shared_objects[KEY]
        with route_dump(engine) as trace:
            assert _served(engine, prompts) == want, policy
        assert store.exclusive and {c.held for c in store.classes.values()} == {4} and trace.mirrored
        if not policy:
            assert parity(trace)["first_differing_step"] is None and trace.admissions > 0
        _held_once(store)
        engine.close()
    for refused in ({"moe.tiers": "std.tiered.pinned@1"}, {"moe.placement": "placement.lru@1"},
                    {"moe.placement": "placement.profile_decay@1"}):
        with pytest.raises(NotImplementedError, match="fewer pieces"):
            _engine(file, host_memory=1 << 24, plugin_options=short | refused)
    from holon_std.tiered.gguf_direct import GgufDirect

    read = GgufDirect.read                                    # a file tier answering another descriptor kind
    with mock.patch.object(GgufDirect, "read", lambda self, c, n: ("memcpy", read(self, c, n)[1])), \
            pytest.raises(ValueError, match="'pread' extents"):
        _engine(file, host_memory=1 << 24, plugin_options=short)


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


@pytest.fixture(scope="module")
def gpu_moe_wide(tmp_path_factory):
    """`gpu_moe` wide enough for a staging step's rounds (docs/dev/v1-glm.md D8s: a class lends a fixed set of a round's
    RAM slots and the landing rows' past its tail), its down experts two classes on disjoint layers as GLM's (D8w's
    tails, one a class): 128 experts, 8 a token, widths 512, four layers, layers 0-1's down Q5_0, 2-3's Q4_0."""
    torch.manual_seed(2)
    cfg = Qwen3MoeConfig(vocab_size=512, hidden_size=512, intermediate_size=512, moe_intermediate_size=512,
                         num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=128,
                         num_experts=128, num_experts_per_tok=8, norm_topk_prob=True, tie_word_embeddings=False,
                         eos_token_id=None)
    path = tmp_path_factory.mktemp("gpu_moe_wide")
    model = Qwen3MoeForCausalLM(cfg)
    with torch.no_grad():
        for layer in model.model.layers:
            layer.mlp.gate.weight.mul_(8)
    model.save_pretrained(path / "hf")
    write(path / "hf", path / "moe.gguf", Q.Q8_0,
          types={"0.ffn_down_exps": Q.Q5_0, "1.ffn_down_exps": Q.Q5_0, "ffn_down_exps": Q.Q4_0})
    prompts = [torch.randint(0, 512, (n,)).tolist() for n in (1, 40, 300, 450)]
    engine = _gpu_engine(path / "moe.gguf")
    want = _served(engine, prompts)
    engine.close()
    return path / "moe.gguf", prompts, want


def _mode2(**over) -> dict:
    """GLM mode (ii)'s deployment terms on the tiny models (docs/dev/v1-glm.md, the process section: an exclusive-tier
    change's engine tests run under them too): exclusive tiers, F10 (`placement.promote@1`, the spares a tail step
    releases), the tail lent with the step-peak loan required, RAM's own heat, the loan's margin and largest range,
    split calls and gathers beside the compute — the cache and RAM fractions the tiny models'."""
    return {"moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.ring_lend": "tail",
            "moe.lend_peak": "require", "moe.cache": "*=0.75", "moe.ram": "*=0.5",
            "moe.placement": "placement.promote@1", "moe.ram_heat": "0.999", "moe.loan_margin": "1.12",
            "moe.loan_largest": "0.55", "moe.split": "on", "moe.copy_beside": "on",
            "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"} | over


@pytest.fixture
def mode2():
    """`_mode2`, the deployment's terms (overrides by keyword)."""
    return _mode2


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
    return Engine(EngineConfig(**dict(model=str(model), device="cuda", dtype=torch.bfloat16, page_size=16, max_seqs=4,
                                      max_step_tokens=256, max_model_len=512, cuda_graphs=True, overlap=True) | kw))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("policy, cache", [("", ""), ("", "*=0.25"), ("", "*=0"), ("placement.lru@1", "*=0.25"),
                                           ("placement.profile_decay@1", "*=0.25"), ("std.tiered.static@1", "*=0.25"),
                                           ("", "*=0.5 ring"), ("", "*=0.5 dma"), ("", "*=0.75 tail"),
                                           ("", "*=0.25 split"), ("", "*=0.75 pipe"),
                                           ("", "*=0.25 ssd"), ("", "*=0.25 ssd1"), ("", "*=0.25 ssdsplit"),
                                           ("std.tiered.static@1", "*=0.25 ssd"),
                                           ("placement.promote@1", "*=0.25 ssd")])
def test_the_offload_store_on_the_gpu_is_bitwise_the_resident_store(gpu_moe, policy, cache):
    """docs/dev/v1-offload.md §6's exact class on the GPU kernels (kernels.ggml_cuda over class arenas,
    kernels.expert_io's
    resolve, its F11 and F4 variants and zero-copy gather from mapped host pools), decode steps replayed from CUDA
    graphs and prefill chunked past the mmvq limit: the offload store's streams and prompt logprobs are the resident
    store's, bit for bit, under every placement policy (arenas of 42, 18, 10 slots: stream-k unpatched), and with RAM
    holding a quarter of the experts (exclusive tiers, D10: SSD reads on the I/O worker inside decode graphs; at one
    sequence, "ssd1", its prompt steps borrow their staging from the cache, the lent experts parked in RAM over its
    coldest or sent to the SSD, while its own staging slots stay a graphed step's, and its SSD reads land in landing
    rows of two, a call's in rounds, its prompt steps of 8 tokens or more reading the next layers' sets ahead:
    docs/dev/v1-glm.md D8b+e, D8f). F11's
    residency, step by step, is its replay oracle's (L1 parity), with admissions in graphed and eager steps, the ring's
    copies on SMs or on copy engines; a borrowing step's lending and ring never wait for the device (with copy
    engines, but for one explicit sync to read the copy list). A
    corrupt map entry is flagged by the resolve and stops the engine at the commit; the kernels' id-error word is
    watched too. The arena pool is released when the ledger frees its pools (the startup profile reads free memory
    then)."""
    from holon_judge.placement import parity, route_dump

    file, prompts, want = gpu_moe
    cache, ring = cache.partition(" ")[::2]          # "ring" / "dma": steps of 32 prompt tokens or more stream (D8),
    one = ring == "ssd1"                             # (one sequence: a graphed step's staging below a pass's)
    if one:                                          # the resident store at one sequence (GPU bits follow the rows)
        resident = _gpu_engine(file, max_seqs=1)
        want = _served(resident, prompts)
        resident.close()
    ssd, split = ring in ("ssd", "ssd1", "ssdsplit"), ring in ("split", "ssdsplit")   # their copies on SMs / copy
    #                                                engines, "tail" a tail (D11), "ssdsplit" both (D8c)
    ring = "" if ssd or split else ring              # first; "ssd": RAM holds a quarter; "split": the decode passes'
    pipe = ring == "pipe"                            # resident entries first, the misses' fetch on a side stream;
                                                     # "pipe": prompt steps stage, their misses in 4 fetched parts
    from holon_std.tiered import store as tiered_store
    from holon_std.tiered.resolver import DeviceResolver
    tiers, splits = DeviceResolver._tiers, []                # exclusive tiers' calls, split or not (graphs captured
    spy_tiers = mock.patch.object(DeviceResolver, "_tiers",  # at the build included), and what each enqueues first
                                  lambda self, *a: splits.append(a[-1]) or order.append("call") or tiers(self, *a))
    order, rows, gather = [], tiered_store.TieredStore.round_rows, DeviceResolver._gather
    spy_rows = mock.patch.object(tiered_store.TieredStore, "round_rows",
                                 lambda self, r, **kw: order.append(f"rows {r}") or rows(self, r, **kw))
    spy_gather = mock.patch.object(DeviceResolver, "_gather", lambda self, f, n, landed=False, part=0: order.append(
        "ssd" if landed else "ram") or gather(self, f, n, landed=landed, part=part))
    with (spy_tiers, spy_rows, spy_gather,
          mock.patch.object(tiered_store, "ON_DEMAND", 2 if one else tiered_store.ON_DEMAND)):   # ssd1: rounds of 2
        engine = _gpu_engine(
            file, host_memory=1 << 27, **({"max_seqs": 1} if one else {}), plugin_options={
            "moe.store": "std.moe.offload@1", "moe.cache": cache, "moe.placement": policy,
            "moe.ring_tokens": 32 if ring else 512, "moe.ring_copy": "dma" if ring == "dma" else "sm",
            "moe.ring_lend": "tail" if ring in ("tail", "pipe") else "coldest", "moe.split": "on" if split else "off"}
            | ({"moe.stream_tokens": 1 << 20, "moe.pipeline": 4} if pipe else {})
            | ({"moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1", "moe.ram": "*=0.25"} if ssd else {}))
    store = engine.ctx.shared_objects[KEY]
    if ssd:                                          # (prompt steps of 8 tokens or more read ahead; never one with
        store.read_ahead_tokens = 8                  # a decode row: at 4 sequences, mixed steps keep: D8f)
    calls = []                                       # eager calls' SSD misses (round 0's count) and read-ahead hits
    spy_rows = mock.patch.object(tiered_store.TieredStore, "round_rows", lambda self, r, **kw: (
        r or calls.append(("ssd", self.n_ssd[:1].clone())), rows(self, r, **kw))[1])
    spy_hits = mock.patch.object(DeviceResolver, "_gather", lambda self, f, n, landed=False, part=0: (
        landed and part is None and calls.append(("hit", n[:1].clone())), gather(self, f, n, landed=landed,
                                                                                part=part))[1])
    commit, routed = store.after_commit, []          # per step: its decode rows' routes, and the counts the step left
    store.after_commit = lambda step, t: routed.append((int(step.n_query[np.isin(step.row_kind, (    # its decode and
        RowKind.DECODE, RowKind.VERIFY))].sum()) * 2 * 4, int(t["counts"].sum()))) or commit(step, t)   # verify rows
    mode = "warn" if ring == "dma" else "error"                   # "dma": one sync per streaming step, counted
    from holon_std.tiered.resolver import DeviceResolver
    piped, pipelined, rings = [], DeviceResolver._pipelined, []

    def piping(self, *a):                            # (a pipelined pass's misses against its staging ring's width)
        lent = self.store.lent
        if lent is not None and lent.width is not None and self.group < self.store.borrowing_groups:
            rings.append((a[3].max(), lent.width[torch.as_tensor(self.store.class_table[self.group])].min()))
        piped.append(a[-1])
        return pipelined(self, *a)
    spy = mock.patch.object(DeviceResolver, "_pipelined", piping)
    with (spy, route_dump(engine) as trace, warnings.catch_warnings(record=True) as syncs,
          _no_host_sync(Lent, "begin", "sources", "done", mode=mode), _tail_checked(store) as tails, spy_rows,
          spy_hits):
        warnings.simplefilter("always")
        assert _served(engine, prompts, mirrored=store if one else None) == want
    assert bool(tails) == (ring in ("tail", "pipe"))
    syncs = [w for w in syncs if "called a synchronizing" in str(w.message)]
    assert len(syncs) == (store.borrowed["ring"] if ring == "dma" else 0), (len(syncs), store.borrowed)
    assert engine.stats()["graphs"] > 0 and (store.peak_misses > 0) == (cache != "")
    assert all(a == b for a, b in routed) and max(routed)[0] > (0 if one else 8)   # top-2 x 4 layers; no padding
    variant = {"": ("moe_resolve.lfu@1", "kernels.expert_io"),
               "placement.lru@1": ("moe_resolve.lru@1", "kernels.expert_io"),
               "placement.promote@1": ("moe_resolve.promote@1", "kernels.promote")}.get(
        policy, ("moe_resolve@1", "kernels.expert_io"))
    assert {(k.rpartition("/")[2], v["provider"]) for k, v in engine.ctx.kernels.items() if k.startswith("std.tiered")
            } == {variant, ("expert_fetch@1", "kernels.expert_io"),
                  ("expert_prefetch@1", "kernels.expert_io")} | ({("moe_tiers@2", "kernels.expert_io"),
                                                                 ("cpu_task@1", "kernels.cpu_task")} if ssd else set())
    if ssd:                                          # the SSD reads ran on the I/O worker, fail-open untouched
        io = engine.ctx.shared_objects["cpu_task.io"].native
        assert store.exclusive and io.served() > 0 and io.flag() == 0
        assert any(splits) == split              # "ssdsplit": RAM misses beside the hits' passes (D8c)
        starts = [i for i, e in enumerate(order) if e == "call"]  # round 0's reads requested before the RAM gather
        assert starts and all(order[i + 1:i + 3] == ["rows 0", "ram"] for i in starts), order[:8]   # (C S1-M1)
        # ssd1: a call's SSD misses past its landing rows, read in rounds of 2 (docs/dev/v1-glm.md D8f)
        peak = max((int(t) for k, t in calls if k == "ssd"), default=0)
        assert not one or (store.landing_rows == 2 and peak > 2), peak
        assert not one or sum(int(t) for k, t in calls if k == "hit") > 0     # ssd1: prompt steps of 8 tokens or
        #                                                                       more read their sets ahead
    if one:                                          # exclusive tiers lend their prompt steps' staging (GLM D8b+e)
        assert store.staging < store.prefill and store.lending and store.borrowed["stage"] > 0, store.borrowed
        assert int(store.parked) > 0                 # lent experts parked in RAM (D8b), read back by later misses
    if not policy:
        report = parity(trace)
        assert report["first_differing_step"] is None and (report["admissions"] > 0) == (cache != "" and cache != "*=0")
        assert report["device_is_mirror"] and bool(store.borrowed["ring"]) == bool(ring and not pipe), (
            report, store.borrowed)
        assert not pipe or (store.borrowed["stage"] and piped)
        # "pipe": its prompt steps' staging a ring of two parts (FN §21.16), less than a pass's worth: a pass's misses
        # past its ring's slots (reused)
        rings = [(int(n), int(w)) for n, w in rings]                 # (read after serving: no sync between passes)
        assert not pipe or (max(w for _, w in rings) < store.prefill and any(n > w for n, w in rings)), rings
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


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_a_tail_store_lends_its_streaming_steps_their_activations(gpu_moe):
    """The core's ActivationLender under `moe.lend_peak` (FN step 7, the toy proof): the startup profile measures the
    worst step, then reserves only the worst one below `ring_tokens` (smaller); a streaming step's forward allocates
    from the tail's last bytes (`activation_loan@1`: kernels.expert_io's MemPool), a range in proportion to its prompt
    tokens, past the passes' buffers; the outputs are the resident store's bit for bit (CUDA graphs on, overlap on).
    The loan options' defaults lend the bytes the loan had before them (FN's RTX 5080 lock's layout). A buffer kept
    from inside a lending step stops the engine at the next launch."""
    from holon.core import platform
    from holon_std.tiered.store import LOAN_MARGIN, LOAN_SLACK, _profiled_tokens

    file, prompts, want = gpu_moe
    peaks, measure = [], platform.peak_memory

    def peak_memory(device, fn):
        peaks.append(measure(device, fn))
        return peaks[-1]
    opts = {"moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.ring_lend": "tail", "moe.lend_peak": "on"}
    with mock.patch.object(platform, "peak_memory", peak_memory):
        engine = _gpu_engine(file, host_memory=1 << 27, plugin_options=opts)
    store = engine.ctx.shared_objects[KEY]
    assert len(peaks) == 2 and 0 < peaks[1] < peaks[0] and store._loan_bytes >= peaks[0]
    ranges, use = [], store._loan.use
    store._loan.use = lambda r: (ranges.append((r.data_ptr(), r.numel(), store.step_tokens, store._step_loan,
                                                store._step_tail, store._step_gu_loan)), use(r))[1]
    assert _served(engine, prompts) == want
    cls, n = store.tail
    c = list(store.classes.values())[cls]
    end = c.slots()[c.size - 1].data_ptr() + c.unit_bytes
    # the loan options' defaults (moe.loan_margin, moe.loan_taper, moe.loan_largest: provisional, docs/dev/v1-glm.md
    # D8g) lend the bytes the loan had before them, as FN's RTX 5080 lock's layout (RAM holds every expert) takes it
    align, tokens = store._align, _profiled_tokens(engine.ctx.options)
    before = lambda share: -(-(int(share) + LOAN_SLACK) // align) * align                   # noqa: E731
    assert store._loan_bytes == before(store._peak_per_token * tokens * LOAN_MARGIN)
    buffers = lambda at: max(f(at) for f in store.tail_bytes)                               # noqa: E731
    assert store.borrowed["ring"] and ranges                 # each the tail's last bytes (512-byte aligned), its
    for at, nbytes, tokens, loan, tail, gu in ranges:        # step's share of the worst step's loan
        share = store._loan_per_token * tokens
        assert at + nbytes == end and at % 512 == 0 and share < nbytes + 512 and nbytes <= store._loan_bytes
        assert loan == gu == min(store._loan_bytes, before(share))
        assert tail == (cls, min(n, -(-(buffers(tokens) + loan) // c.unit_bytes)))
    assert store._loan.held() == 0 and store._loan.lib.loan_overflow() == 0     # the range held every segment
    kept, real = [], store.activations

    def leaky(step):                                 # the plant: a buffer made in the loan and kept
        loan = real(step)
        if loan is None:
            return None

        @contextlib.contextmanager
        def keeping():
            with loan:
                kept.append(torch.empty(1 << 16, device="cuda"))
                yield
        return keeping()
    store.activations = leaky
    engine.add([(t + 7) % 512 for t in prompts[-1]], GREEDY)              # no prefix hit: a streaming step
    with pytest.raises(RuntimeError, match="outlive"):
        while engine.has_work():
            engine.step()
    assert kept
    kept.clear()
    engine.close()                                  # (the loan's range free for the next engine)


_LENT_STEPS = """
import sys
import torch
from holon.api import SamplingParams
from holon.core.engine import Engine, EngineConfig
engine = Engine(EngineConfig(model=sys.argv[1], device="cuda", dtype=torch.bfloat16, page_size=16, max_seqs=4,
                             max_step_tokens=256, max_model_len=512, host_memory=1 << 27, plugin_options={
                                 "moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.ring_lend": "tail",
                                 "moe.lend_peak": "on"}))
loan, uses = engine.ctx.shared_objects["std.moe.offload"]._loan, []
use = loan.use
loan.use = lambda r: (uses.append(r.numel()), use(r))[1]
g = torch.Generator().manual_seed(0)
for _ in range(int(sys.argv[2])):
    n = int(torch.randint(40, 250, (1,), generator=g))
    engine.generate([torch.randint(0, 512, (n,), generator=g).tolist()], SamplingParams(temperature=0.0, max_tokens=1))
assert len(uses) >= int(sys.argv[2]), len(uses)
"""


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_lent_steps_leave_the_host_heap_whole(gpu_moe, tmp_path):
    """A process serving 600 lent steps under glibc's heap checks (every Python object on glibc's heap, freed memory
    poisoned) exits cleanly. The loan's `held` was torch's `MemPool.snapshot()`, which on torch 2.11 corrupts the host
    heap: a glibc abort in a later, unrelated allocation within a few hundred steps (GLM step 7's engine restarts,
    about one a thousand requests); it is now the native loan's bytes still out after `release`."""
    file, _, _ = gpu_moe
    script = tmp_path / "lent.py"
    script.write_text(_LENT_STEPS)
    env = os.environ | {"PYTHONMALLOC": "malloc", "MALLOC_CHECK_": "3", "MALLOC_PERTURB_": "165",
                        "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    run = subprocess.run([sys.executable, str(script), str(file), "600"], env=env, capture_output=True, text=True,
                         timeout=1200)
    assert run.returncode == 0, run.stderr[-3000:]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_a_lent_steps_loan_holds_its_fixed_bytes(gpu_moe):
    """`moe.loan_fixed_mib` (GLM step 5): a lent step's loan is its share of the worst step's by its prompt tokens plus
    the fixed bytes (a step's fixed costs do not shrink with its tokens: mid-size steps once overflowed their loans
    into the device's own memory and ran out of it), at most the worst step's loan — which holds them too (step 6: FN's
    4K-token steps with prompt logprobs overflowed a worst-step loan without them). The resident store's bits."""
    file, prompts, want = gpu_moe
    opts = {"moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.ring_lend": "tail", "moe.lend_peak": "on",
            "moe.loan_fixed_mib": 32}
    engine = _gpu_engine(file, host_memory=1 << 27, plugin_options=opts)
    store = engine.ctx.shared_objects[KEY]
    loans, use = [], store._loan.use
    store._loan.use = lambda r: (loans.append((store.step_tokens, r.numel())), use(r))[1]
    assert _served(engine, prompts) == want
    assert loans and all(n >= min(store._loan_per_token * t + (32 << 20), store._loan_bytes) - 1024 for t, n in loans)
    assert store._loan_bytes >= store._loan_per_token * engine.cfg.max_step_tokens + (32 << 20)
    engine.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("context", [{}, {"max_seqs": 1, "max_model_len": 128}], ids=["step", "short-context"])
def test_a_loans_share_is_of_the_profiled_steps_tokens(gpu_moe, context):
    """The worst step's loan is its tokens' share plus the fixed bytes, its tokens those the startup profile ran: at a
    context shorter than a step (`holon_tools profile --max-model-len 4096` on GLM's 32K-step deployment) the
    profile's worst step is `max_seqs` rows of `max_model_len` tokens, and a share counted per `max_step_tokens` left a
    1,293-token step 32 MiB short."""
    file, _, _ = gpu_moe
    opts = {"moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.ring_lend": "tail", "moe.lend_peak": "on",
            "moe.loan_fixed_mib": 32}
    engine = _gpu_engine(file, host_memory=1 << 27, plugin_options=opts, **context)
    store, cfg = engine.ctx.shared_objects[KEY], engine.cfg
    worst = min(cfg.max_step_tokens, cfg.max_seqs * cfg.max_model_len)
    assert abs(store._loan_per_token * worst + (32 << 20) - store._loan_bytes) <= store._align
    engine.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_a_loan_overflow_is_counted_and_said_once(gpu_moe, capsys):
    """Step 5's review B m1: a lent step whose activations outgrow its loan (here its per-token share forced to 0 and
    its fixed bytes the 20 MiB floor) takes the device's own memory; the store counts each such step and its most bytes,
    and says so once, naming `moe.loan_fixed_mib`. The resident store's bits."""
    file, prompts, want = gpu_moe
    opts = {"moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.ring_lend": "tail", "moe.lend_peak": "on"}
    engine = _gpu_engine(file, host_memory=1 << 27, plugin_options=opts)
    store = engine.ctx.shared_objects[KEY]
    store._loan_per_token, store._loan_fixed = 0.0, 1 << 20        # (a loan of 1 MiB: every lent step overflows)
    assert _served(engine, prompts) == want
    engine.add([3] * 4, GREEDY)                                     # (a lent step is counted at the next)
    while engine.has_work():
        engine.step()
    said = capsys.readouterr().out
    assert store.loan_overflows >= 1 and store.loan_overflow_most > 0 and said.count("moe.loan_fixed_mib") == 1
    engine.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("cache, kept", [("*=0.72", "own tail"), ("*=0.5", "reserved")])
def test_a_loan_the_layout_cannot_lend_stays_reserved(gpu_moe, capsys, cache, kept):
    # (a cache of 23 a class keeps its own tail at tails of 14-15 slots, this loan's 14: one slot of room for a peak
    # the profile measures larger in a long-running process; 22 a class, at 13-14, ran out at 15 in the full suite)
    """`moe.lend_peak` where the layout cannot lend (decided at build, never refused at a launch): a cache too small to
    lend a prompt step's staging keeps the loan in its own tail (the pool's last bytes, the cache that much smaller),
    each large step's activations allocated at its end; a tail class too small to hold the loan beside a pass's
    staging drops the loan (the engine reserves the worst step's peak). Both say so, and serve the resident store's
    bits."""
    file, prompts, want = gpu_moe
    opts = {"moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.ring_lend": "tail", "moe.lend_peak": "on",
            "moe.cache": cache, "moe.loan_largest": 0.9}
    engine = _gpu_engine(file, host_memory=1 << 27, plugin_options=opts)
    store = engine.ctx.shared_objects[KEY]
    ranges = []
    if store._loan is not None:
        use = store._loan.use
        store._loan.use = lambda r: (ranges.append((r.data_ptr(), r.numel())), use(r))[1]
    assert _served(engine, prompts) == want
    said = capsys.readouterr().out
    if kept == "own tail":
        at, n = store._own_tail
        end = store.arena.view(torch.uint8, (store.arena.capacity * store.granule,))[at:at + n][-1:].data_ptr() + 1
        assert not store.lending and n >= store._loan_bytes and ranges and "own tail" in said
        assert all(a + nbytes == end for a, nbytes in ranges)
    else:
        assert store._loan is None and store._loan_bytes == 0 and "stay reserved" in said
        assert store._largest(256) == 0             # no loan, so no step asks its tail for a range (A's D8k m1)
    engine.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_a_layout_whose_down_tails_leave_no_staging_does_not_lend(gpu_moe_wide, tmp_path):
    """D8w's whole-expert tail lends each down class's last slots too: a layout lends only where every class with a
    tail still stages a prompt pass's misses outside it — decided at build, not flagged by a step's kernels (errors 4:
    a 24 GB card at mode (ii)'s flags, its IQ3_XXS down class 377 slots, a tail of 227, a pass's 288 misses). Here a
    200 MiB budget caches ≈ 160 of each down class's 256 pieces, a tail of 77 in each and a pass of 128 misses: the
    layout stages (its own tail), and serves the resident store's bits."""
    file, prompts, want = gpu_moe_wide              # GLM's shape: gate/up Q4_0, down Q8_0 and Q5_0
    write(file.parent / "hf", tmp_path / "moe.gguf", Q.Q4_0,
          types={"0.ffn_down_exps": Q.Q8_0, "1.ffn_down_exps": Q.Q8_0, "ffn_down_exps": Q.Q5_0})
    one = {"max_seqs": 1, "prefix_cache": False}    # (its KV holds one sequence: the reference schedules alike)
    reference = _gpu_engine(tmp_path / "moe.gguf", **one)
    want = _served(reference, prompts)
    reference.close()
    opts = {"moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.ring_lend": "tail", "moe.lend_peak": "on",
            "moe.ram": "*=0.25", "moe.loan_fixed_mib": 64, "moe.placement": "placement.promote@1",
            "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"}
    engine = _gpu_engine(tmp_path / "moe.gguf", host_memory=1 << 29, plugin_options=opts, **one,
                         memory_fraction=(200 << 20) / torch.cuda.mem_get_info()[0])
    store = engine.ctx.shared_objects[KEY]
    names = list(store.classes)
    short = [c for c, m in store.tail_down.items() if store._lend_room[names[c]] - m < store.prefill]
    assert store.whole_tail and store.tail_down and short, (store._lend_room, store.tail_down, store.prefill)
    assert not store.lending
    assert _served(engine, prompts) == want and _bytes_exact(store) == 0
    engine.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_a_required_loan_without_the_startup_profile_is_refused(gpu_moe):
    """`moe.lend_peak = require` with an explicit state budget (`state_memory`: no startup profile, so nothing is
    measured to lend): refused at the first launch, naming why (it once served on, lending nothing, until a large step
    ran out of memory)."""
    file, prompts, _ = gpu_moe
    opts = {"moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.ring_lend": "tail",
            "moe.lend_peak": "require"}
    engine = _gpu_engine(file, host_memory=1 << 27, state_memory=64 << 20, plugin_options=opts)
    with pytest.raises(RuntimeError, match="startup profile did not run"):
        _served(engine, prompts)
    engine.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("policy", ["", "placement.promote@1"])
def test_a_required_loan_is_lent_under_exclusive_tiers_or_refused(gpu_moe, policy):
    """`moe.lend_peak = require` (docs/dev/v1-glm.md D9): a mode (ii)-shaped layout — RAM holding a quarter of the
    experts, three quarters cached, tail lending — starts, its large steps stage (no ring under exclusive tiers) with
    their activations in the lent tail, and serves the resident store's bits — under F11 and under F10 (its spares in
    the tail released: docs/dev/v1-glm.md D8a); a layout that would lend none refuses to start, naming why."""
    file, prompts, want = gpu_moe
    opts = {"moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.ring_lend": "tail",
            "moe.lend_peak": "require", "moe.cache": "*=0.75", "moe.ram": "*=0.25",
            "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1", "moe.placement": policy}
    engine = _gpu_engine(file, host_memory=1 << 27, plugin_options=opts)
    store = engine.ctx.shared_objects[KEY]
    ranges, use = [], store._loan.use_ranges           # (a whole-expert tail: a range per class with a tail, D8w)
    store._loan.use_ranges = lambda rs: (ranges.append([r.numel() for r in rs]), use(rs))[1]
    assert _served(engine, prompts) == want
    assert store.exclusive and ranges and store.borrowed["stage"] and not store.borrowed["ring"], store.borrowed
    assert store.whole_tail and store.tail_down and max(map(len, ranges)) == 1 + len(store.tail_down), ranges
    held = _slots_held(store)                        # again, F10's spares (refilled by the decodes) in the lent tail:
    assert _served(engine, prompts) == want          # released, their slots return (step 1's review S1-M1)
    assert torch.equal(_slots_held(store), held)
    _held_once(store)                                # (lending, parking and promotions: still one copy each)
    engine.close()
    with pytest.raises(RuntimeError, match="require.*moe.ring_lend is not tail"):
        _gpu_engine(file, host_memory=1 << 27, plugin_options=opts | {"moe.ring_lend": "coldest"})



@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_rams_long_heat_is_its_oracles(gpu_moe):
    """`moe.ram_heat` (docs/dev/v1-glm.md §7.4's lever 1, step 6): an exclusive RAM tier's victims by a long-memory heat
    of its own — each step's decode tokens decay it by the factor a token, then add the step's routed counts — which the
    tiers' placement and the parking rank by in place of the policy's. Against its oracle over the committed steps'
    counts (the last step's are folded at the next step's start), graphed decode steps included; the resident store's
    bits."""
    file, prompts, want = gpu_moe
    opts = {"moe.store": "std.moe.offload@1", "moe.cache": "*=0.5", "moe.ram": "*=0.25", "moe.ram_heat": 0.9,
            "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"}
    engine = _gpu_engine(file, host_memory=1 << 27, plugin_options=opts)
    store, counts = engine.ctx.shared_objects[KEY], []
    assert store.heat is store.ram_heat
    store.taps.append(lambda step, record: counts.append(record["counts"].astype(np.float64)))
    assert _served(engine, prompts) == want
    assert engine.generate([[5, 6, 7]], SamplingParams(temperature=0.0, max_tokens=12, ignore_eos=True))
    heat = np.zeros_like(counts[0])                   # (no profile: seeded at 0)
    for c in counts[:-1]:
        heat = heat * 0.9 ** (c[0].sum() / store.top_k) + c
    assert any(c.sum() for c in counts[:-1]) and np.allclose(store.ram_heat.cpu().numpy(), heat, rtol=1e-5, atol=1e-5)
    engine.close()
    with pytest.raises(ValueError, match="moe.ram_heat ranks an exclusive RAM tier"):
        _gpu_engine(file, host_memory=1 << 27,
                    plugin_options=opts | {"moe.tiers": "std.tiered.pinned@1", "moe.ram": ""})




@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_a_spare_policys_moves_run_beside_and_fence_the_staging_slots(gpu_moe):
    """GLM step 7: under a policy with spares (F10) and exclusive tiers a call's admitted moves and demotions run on
    the store's moves stream — the admissions first, the next call's staging writes after them (`admitted`), the
    demotions beside the next calls, a later call reusing their table or their victims' slots after them (`moved`,
    the step's join) — and the store still serves the resident store's bits, one copy an expert, its demotions late
    (a sleep before each); under `moves_unfenced` (no fence, each call's demotions enqueued after the call two later
    rewrote their table: a deterministic delay, not a race) it does not."""
    from holon_judge.planted import _late_demotions, moves_unfenced

    file, prompts, _ = gpu_moe
    params = SamplingParams(temperature=0.0, max_tokens=48, ignore_eos=True)     # (long enough to read demoted
    resident = _gpu_engine(file)                                                  # experts back from RAM)
    want = resident.generate(prompts, params)
    resident.close()
    opts = {"moe.store": "std.moe.offload@1", "moe.cache": "*=0.25", "moe.ram": "*=0.25", "moe.ring_tokens": 512,
            "moe.placement": "placement.promote@1", "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"}
    for plant in (False, True):
        with moves_unfenced() if plant else _late_demotions(20_000_000) as fired:
            engine = _gpu_engine(file, host_memory=1 << 27, plugin_options=opts)
            store = engine.ctx.shared_objects[KEY]
            assert store.moves_async
            got = engine.generate(prompts, params)
            wrong = _bytes_exact(store)
            if not plant:
                _held_once(store)
            engine.close()
        assert (got == want and not wrong) != plant and (not plant or fired["deferred"]), (wrong, fired)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("cache", ["*=0.5", "*=0.75"])
def test_a_layout_lends_only_past_the_policys_reserved_slots(gpu_moe, cache):
    """F10 under exclusive tiers: its spares (`reserved` a layer in each role's class) are cache slots no step lends.
    A cache whose seeded experts less the spares cannot lend a prefill pass's staging stages it in slots of its own
    (it stopped at the first prompt step with the store's SHORT error when the layout counted the spares as lendable);
    one that can still lends. Either way the resident store's bits."""
    file, prompts, _ = gpu_moe
    resident = _gpu_engine(file, max_seqs=1)               # (one sequence a step: a pass of 8 misses, staging of 2)
    want = _served(resident, prompts)
    resident.close()
    opts = {"moe.store": "std.moe.offload@1", "moe.cache": cache, "moe.ram": "*=0.25",
            "moe.placement": "placement.promote@1", "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"}
    engine = _gpu_engine(file, host_memory=1 << 27, plugin_options=opts, max_seqs=1)
    store = engine.ctx.shared_objects[KEY]
    assert _served(engine, prompts) == want
    held = store._held_out()
    assert all(n == 3 * len(store.layers) for n in held.values())          # 3 spares a layer, one class a role
    lendable = min(c.size - c.null - c.staging - held[n] for n, c in store.classes.items())
    assert store.lending == (lendable >= store.prefill) == (cache == "*=0.75")
    engine.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_a_keep_never_takes_a_unit_whose_demotion_is_pending(gpu_moe):
    """GLM step 7 (A's D8p-M3): a call's demotions run beside the next call, whose keeps the I/O worker writes from the
    host, so `moe_tiers@2`'s stamps keep the units the previous call demoted into out of its displacements. With
    RAM's coldest forced onto those experts at every call and the demotions late, the store serves the resident
    store's bits; with the stamps dropped (`keep_into_pending_demotion`) keeps land on such units and it does not."""
    from holon_judge.planted import keep_into_pending_demotion, pending_victims
    from holon_plugins.placement.promote import plugin as promote

    file, prompts, _ = gpu_moe
    params = SamplingParams(temperature=0.0, max_tokens=48, ignore_eos=True)
    resident = _gpu_engine(file, cuda_graphs=False)
    want = resident.generate(prompts, params)
    resident.close()
    opts = {"moe.store": "std.moe.offload@1", "moe.cache": "*=0.25", "moe.ram": "*=0.25", "moe.ring_tokens": 512,
            "moe.placement": "placement.promote@1", "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"}
    for plant in (pending_victims, keep_into_pending_demotion):          # (admitting every miss: the most demotions)
        with plant() as fired, mock.patch.object(promote, "BUDGET", None):
            engine = _gpu_engine(file, host_memory=1 << 27, plugin_options=opts, cuda_graphs=False)
            got = engine.generate(prompts, params)
            wrong = _bytes_exact(engine.ctx.shared_objects[KEY])
            engine.close()
        stamped = plant is pending_victims
        assert fired["forced"] and (fired["keeps_on_pending"] == 0) == stamped, dict(fired)
        assert (wrong == 0) == stamped and (got == want or not stamped), wrong


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("plant", [None, "demotion_before_gather"])
def test_a_calls_demotions_follow_the_next_calls_gathers(gpu_moe, plant):
    """D8t (docs/dev/v1-glm.md): under F10 a call's demotions are enqueued after the next call's gathers (no longer
    under them), its table's `moved` after them; a call of the same layer next (mixed steps of 1,024 rows in passes
    of 512) enqueues them first, and the step's last call at its join. The store serves the resident store's streams
    and every piece's bytes. Demotions before the call's own gathers (`demotion_before_gather`: a RAM-sourced
    admission staged from its victim's bytes) leave wrong bytes. (A same-layer call after one with demotions cannot
    occur — decode rows never span a policy's passes, D4, and a drafter's sub-steps join at their last layer — so
    the guard is defensive and has no plant.)"""
    from holon_judge import planted

    file, prompts, _ = gpu_moe
    prompts = prompts + [torch.randint(0, 400, (900,), generator=torch.Generator().manual_seed(5)).tolist()]
    params = SamplingParams(temperature=0.0, max_tokens=24, ignore_eos=True)
    base = {"max_step_tokens": 1024, "max_model_len": 2048, "chunk_with_decode": 1, "cuda_graphs": False}
    resident = _gpu_engine(file, **base)
    want = resident.generate(prompts, params)
    resident.close()
    opts = {"moe.store": "std.moe.offload@1", "moe.cache": "*=0.25", "moe.ram": "*=0.25", "ggml_cuda.pass_rows": 512,
            "moe.placement": "placement.promote@1", "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"}
    with getattr(planted, plant)() if plant else contextlib.nullcontext() as fired:
        engine = _gpu_engine(file, host_memory=1 << 27, plugin_options=opts, **base)
        store = engine.ctx.shared_objects[KEY]
        same, tier_op = [], store.resolvers[0].__class__._tier_op

        def counted(self, ids, kinds, fetch, n):  # a same-layer call (its pending table its own)
            pending = self.store.pending_demotions
            same.append(pending is not None and pending[0] == self.group)
            return tier_op(self, ids, kinds, fetch, n)
        rids = {engine.add(p, params): i for i, p in enumerate(prompts)}
        got, wrong, seen = [[] for _ in rids], 0, 0
        with mock.patch.object(store.resolvers[0].__class__, "_tier_op", counted) if plant is None else \
                contextlib.nullcontext():
            while engine.has_work():
                for d in engine.step():
                    got[rids[d.rid]] += d.token_ids
                if plant and sum(fired.values()) > seen:     # the bytes right after a planted step (before a later
                    seen = sum(fired.values())                # eviction hides them)
                    wrong += _bytes_exact(store)
        wrong += _bytes_exact(store)
        engine.close()
    if plant is None:
        assert got == want and wrong == 0 and any(same), (wrong, sum(same))
    else:
        assert seen and wrong, (plant, dict(fired))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("plant", [None, "demotion_table_rewritten"])
def test_a_layers_next_call_enqueues_its_last_calls_demotions_first(gpu_moe, plant):
    """D8t (A's M1, m1): a layer called twice in a row with decode rows (a drafter's sub-steps on an auxiliary group;
    here layer 1's routed experts run twice every decode step, the first result dropped): the first call's
    promotions refill its spares, their demotions deferred; the second call, of the same move table, enqueues them
    before its tier op rewrites the table. The streams are the resident store's and every piece's bytes the file's,
    read right after each step; with that flush dropped (`demotion_table_rewritten`) the first call's demotions copy
    the second call's list and its victims' RAM units are left stale (RAM with free units: the victims demoted)."""
    from holon_judge import planted

    file, prompts, _ = gpu_moe
    params = SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True)
    resident = _gpu_engine(file, cuda_graphs=False)
    want = resident.generate(prompts, params)
    resident.close()
    opts = {"moe.store": "std.moe.offload@1", "moe.cache": "*=0.25", "moe.ram": "*=0.75",
            "moe.placement": "placement.promote@1", "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"}
    with getattr(planted, plant)() if plant else contextlib.nullcontext() as fired:
        engine = _gpu_engine(file, host_memory=1 << 28, plugin_options=opts, cuda_graphs=False)
        store, moe = engine.ctx.shared_objects[KEY], engine.executor.model.layers[1].ffn
        forward, flushed = moe._forward, []

        def twice(x, residual=None, join=None):  # (decode steps: every row a decode row)
            if x.shape[0] <= len(prompts):
                forward(x, None, None)
            return forward(x, residual, join)
        moe._forward = twice
        flush = store.flush_demotions

        def counted():
            flushed.append(store.pending_demotions is not None and store.pending_demotions[0] == 1)
            return flush()
        store.flush_demotions = counted
        rids = {engine.add(p, params): i for i, p in enumerate(prompts)}
        got, wrong = [[] for _ in rids], 0
        while engine.has_work():
            for d in engine.step():
                got[rids[d.rid]] += d.token_ids
            wrong += _bytes_exact(store)                     # right after each step (a later eviction hides them)
        engine.close()
    if plant is None:
        assert got == want and wrong == 0 and sum(flushed) > 1, (wrong, sum(flushed))
    else:
        assert fired["rewritten"] and wrong, (dict(fired), wrong)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_a_stores_fence_orders_a_delayed_producer():
    """The fence std's stores use (`torch.Event` on `torch.accelerator`'s streams; scripts/ci/check_events.py): a
    consumer stream waiting on the event of a producer stream that writes late reads the producer's bytes — the
    pattern item 2's moves and the parking use (a generic stream ignores a CUDA-typed event's wait)."""
    device = torch.device("cuda")
    main = torch.accelerator.current_stream(device)
    producer, consumer = torch.Stream(device=device), torch.Stream(device=device)
    x = torch.zeros(1 << 20, dtype=torch.int32, device=device)
    y = torch.empty_like(x)
    producer.wait_stream(main)
    with producer:
        torch.cuda._sleep(50_000_000)
        x.fill_(7)
        done = torch.Event(device=device)
        done.record()
    consumer.wait_event(done)
    with consumer:
        y.copy_(x)
    main.wait_stream(consumer)
    assert bool((y == 7).all())


def _bytes_exact(store, cache: dict | None = None) -> int:
    """Exclusive tiers: how many pieces' RAM units or VRAM slots do not hold the bytes the file has for the piece the
    map names there (an idle engine; `_ram_units_hold_their_pieces` asserts the RAM half); `cache`: the file's bytes
    kept by piece across calls."""
    from holon_std.tiered import files
    from holon_std.tiered.store import _offsets

    torch.cuda.synchronize()
    res, host, wrong = store.residency.device.cpu().numpy(), store.host_units.cpu().numpy(), 0
    cache = {} if cache is None else cache
    for g, classes in enumerate(store.layer_classes):
        for i in range(store.items):
            for r, c in enumerate(classes):
                if res[g, i, r] < -1:
                    continue
                n = store._piece_index(c, g, i, r)
                if (c.name, n) not in cache:
                    cache[c.name, n] = want = np.empty(c.unit_bytes, dtype=np.uint8)
                    files.read([(e, want[o:o + e.nbytes]) for e, o in _offsets(c.pieces[n][3])])
                want = cache[c.name, n]
                got = c.host_view()[int(host[g, i, r])].numpy() if res[g, i, r] == -1 else \
                    c.slots()[int(res[g, i, r])].cpu().numpy()
                wrong += not np.array_equal(got[:c.unit_bytes], want)
    return wrong


def _held_once(store) -> None:
    """Exclusive tiers: every expert in one tier — a RAM unit's piece exactly where the map says (tier code −1), never
    beside a VRAM copy; each RAM unit owned once, by the piece whose host unit it is (no copy a promotion that raced a
    demotion could leave twice: Maya's duplicate-copy GC, step 7) — or reserved for a parked expert whose D8x move
    has not mapped it yet (`reservations`), which no piece's host unit names (A's D8x2-M1; the expert may have been
    routed into another tier meanwhile: phase 3 frees the unit then)."""
    res, host = store.residency.device.cpu().numpy(), store.host_units.cpu().numpy()
    assert ((res == -1) == (host >= 0)).all() and ((res >= 0).all(-1) == (res >= 0).any(-1)).all()
    reserved = {(k, u): item for k, u, item in store.reservations()}
    for k, c in enumerate(store.classes.values()):        # each RAM unit holds one piece, the one the map says
        owner = store.owner[k].cpu().numpy()
        units = [u for u in np.flatnonzero(owner >= 0) if (k, u) not in reserved]
        g, e = np.divmod(owner[units], store.items)
        role = [next(r for r, x in store.groups[store.layers[gg]].items() if x is c) for gg in g]
        assert len(set(owner[owner >= 0])) == np.count_nonzero(owner >= 0) and (host[g, e, role] == units).all()
        named = {int(host[gg, ee, rr]) for gg in range(len(store.layers)) for ee in range(store.items)
                 for rr, x in store.groups[store.layers[gg]].items() if x is c}
        assert all(owner[u] == item and u not in named for (kk, u), item in reserved.items() if kk == k)

def _slots_held(store) -> torch.Tensor:
    """Each layer's cache slots per role between steps: mapped, empty, a policy's spares' — conserved (none leaks)."""
    spare = store.state.get("spare")
    spares = (spare[..., 0] > 0).sum(1) if spare is not None else 0
    return ((store.residency.device >= 0).sum(1) + (store.n_empty + spares)[:, None]).cpu()


def _spares_lent(store) -> int:
    """D8s (C): the policy's spare slots among the slots this step's non-tail lending took (calls ≥ 1)."""
    spare = store.state.get("spare")
    if spare is None or store.lent is None:
        return 0
    lent = torch.where((store.lent_call >= 1)[..., None], store.lent_slots, -1)        # [G, E, R]
    owned = torch.where((spare[..., 0] > 0)[..., None], spare[..., 1:], -2)          # [G, S, R]
    return int((lent[:, :, None] == owned[:, None]).all(-1).sum())


def test_a_whole_expert_tail_fits_glms_mode_ii_geometry():
    """D8w at GLM's RTX 5090 mode (ii) geometry (no GPU: the store's sizing alone): gate/up IQ2_XXS 4.125 MiB a slot,
    3,265 cached; down IQ3_XXS 3.0625 MiB on 8 of 42 layers, 611 cached; down IQ2_S 2.5625 MiB on 34, 2,654 cached;
    a pass's staging 288; the 16,384-row passes' buffers ≈ 2.05 GiB and the 32K loan 7,948 MiB. The tail holds the
    passes' buffers in gate/up's slots (their views are one range: the startup profile once found it short, the
    buffers allocated beside it, the peak and the loan doubled, and the layout refused to start), the loan in the
    rest and the down classes' tails, each down class keeping a pass's staging; and ≈ 1.6× fewer experts than a
    gate/up-only tail of the same bytes."""
    from holon_std.tiered.store import SlotClass, TieredStore

    mib = 1 << 20
    classes = {"gate_up": SlotClass("gate_up", 4224 * 1024, 1, 32, cache=3265),
               "down.iq3": SlotClass("down.iq3", int(3.0625 * mib), 1, 32, cache=611),
               "down.iq2": SlotClass("down.iq2", int(2.5625 * mib), 1, 32, cache=2654)}
    store = object.__new__(TieredStore)
    store.exclusive, store.roles, store.prefill, store.classes, store._auxiliary = True, 2, 288, classes, []
    store._order = list(range(42))
    store.groups = {g: {0: classes["gate_up"], 1: classes["down.iq3" if g < 8 else "down.iq2"]} for g in range(42)}
    buffers, loan = int(2.05 * 1024 * mib), 7948 * mib
    n = store._split(buffers, loan, 0, lambda c: c.cap)
    m = store._down(n)
    unit = classes["gate_up"].unit_bytes
    assert n * unit >= buffers + 2 * 1024 * mib                         # the buffers, and the 32K R (2 GiB) beside
    assert n * unit + sum(k * list(classes.values())[c].unit_bytes for c, k in m.items()) >= buffers + loan
    assert all(list(classes.values())[c].cap - k >= store.prefill for c, k in m.items()), m
    for small in (0, 256 * mib):                                        # a step's own (the profile's: no loan)
        assert store._split(buffers, small, 0, lambda c: c.cap) * unit >= buffers
    gate_up_only = -(-(buffers + loan) // unit)
    assert 1.3 < gate_up_only / n < 1.7 and sorted(m) == [1, 2], (n, gate_up_only, m)


def _loan_free(store) -> bool:
    """D8w: no piece the map names sits in this step's activation loan — its other classes' tails and its tail's
    class's last bytes (`_in_loan`'s ranges)."""
    res, classes = store.residency.device, list(store.classes.values())
    cls = store._class_index.long()[:, None, :].expand_as(res)
    bad = torch.zeros_like(res, dtype=torch.bool)
    for c, m in store._step_down.items():
        bad |= (cls == c) & (res >= classes[c].size - m)
    if (tail := store._loan_slots()) is not None and store._step_gu_loan:
        bad |= (cls == tail[0]) & (res >= tail[1])
    return not bool(bad.any())


def _slots_once(store) -> bool:
    """Each class's slot is mapped by one piece or on one layer's empty stack, never both nor twice."""
    res, empty, n = (t.cpu().numpy() for t in (store.residency.device, store.empty, store.n_empty))
    classes = store._class_index.cpu().numpy()
    for c in range(len(store.classes)):
        mine = [(g, r) for g in range(len(n)) for r in range(res.shape[2]) if classes[g, r] == c]
        held = [res[g, :, r][res[g, :, r] >= 0] for g, r in mine] + [empty[g, r, :n[g]] for g, r in mine]
        held = np.concatenate(held)
        if len(set(held.tolist())) != len(held):
            return False
    return True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("policy", ["", "placement.promote@1"])
def test_a_whole_expert_tail_lends_its_experts_whole(gpu_moe, policy):
    """D8w (docs/dev/v1-glm.md): under exclusive tiers the tail's experts are lent whole — the tail's class's last n
    slots and the down class's last m, its residents relocated into the slots the lent experts free (copied after their
    pieces are parked, before the forward writes the loan), so the loan's ranges hold no mapped piece; a layer whose
    expert was lent gets the tail slot back at the step's end; each slot is held once; the host mirror (map and empty
    stacks) is the device's; every piece's bytes are the file's and the streams the resident store's, with the
    tail's class holding only part of the loan. Each of the compaction's plants is caught (D8w's list)."""
    from holon_judge import planted

    file, prompts, want = gpu_moe
    opts = {"moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.ring_lend": "tail",
            "moe.lend_peak": "require", "moe.cache": "*=0.75", "moe.ram": "*=0.25", "moe.placement": policy,
            "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"}
    plants = [None, "tail_down_not_compacted", "compaction_map_stale", "compaction_return_wrong_layer",
              "compaction_beside_moves", "loan_range_overlap"]
    for plant in plants if policy else plants[:1]:
        with planted.PLANTS[plant]() if plant else contextlib.nullcontext():
            engine = _gpu_engine(file, host_memory=1 << 27, plugin_options=opts)
            store = engine.ctx.shared_objects[KEY]
            checks, lend_tail = [], store.lend_tail

            def watched(cls, n, store=store, lend_tail=lend_tail, checks=checks):
                before = store.residency.device.clone()
                picked = lend_tail(cls, n)
                kept = (before >= 0) & (store.residency.device >= 0)
                checks.append((_loan_free(store), int((kept & (before != store.residency.device)).sum())))
                return picked
            store.lend_tail = watched
            # whole-expert placement keeps an expert's pieces at aligned slots here (one down class, every layer's);
            # GLM's down classes cover disjoint layers, so its slots do not stay aligned: the down slots shuffled
            torch.cuda.synchronize()
            res, down = store.residency.device, store._class_index[:, 1].long()
            gen = torch.Generator().manual_seed(0)
            for c in store.tail_down:
                mine = (down == c)[:, None] & (res[..., 1] >= 0)
                slots = res[..., 1][mine]
                moved = slots[torch.randperm(len(slots), generator=gen).to(slots.device)]
                arena = list(store.classes.values())[c].slots()
                arena[moved.long()] = arena[slots.long()].clone()
                res[..., 1][mine] = moved
            store.residency.host[...] = res.cpu().numpy()
            try:
                got = _served(engine, prompts)
                for shift in (7, 11):                            # more steps: a returned slot's next admissions
                    engine.generate([[(t + shift) % 512 for t in p] for p in prompts], GREEDY)
                torch.cuda.synchronize()
                ok = (got == want, all(free for free, _ in checks), _slots_once(store), _bytes_exact(store) == 0)
            except RuntimeError as e:                            # (a plant the store or the provider refuses)
                assert plant is not None, e
                ok = ("overlap" in str(e),) if plant == "loan_range_overlap" else (False,)
            if plant is None:
                assert store.whole_tail and store.tail_down and any(moved for _, moved in checks), checks
                assert all(ok), ok
                _held_once(store)
                mirror = store.residency
                assert (mirror.host == store.residency.device.cpu().numpy()).all()
                n_empty, empty = store.n_empty.cpu().numpy(), store.empty.cpu().numpy()
                assert all(len(mirror.empty[g]) == n_empty[g] and all((x == empty[g, :, j]).all() for j, x in
                                                                       enumerate(mirror.empty[g]))
                           for g in range(len(n_empty)))
                unit = list(store.classes.values())[store.tail[0]].unit_bytes
                assert store.tail[1] * unit < store._loan_bytes     # the tail's class holds only part of the loan
            elif plant == "loan_range_overlap":
                assert ok == (True,), ok
            else:
                assert not all(ok), (plant, ok)
            engine.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_a_loan_takes_its_margin_its_tapered_fixed_bytes_and_a_range_its_largest_allocation_fits(gpu_moe_wide):
    """The 32K borrow (docs/dev/v1-glm.md §13, step 7): `moe.loan_margin` scales the lent peak, `moe.loan_taper` takes
    the fixed bytes whole up to its first token count and down to LOAN_SLACK at its second, and under a whole-expert
    tail `moe.loan_largest` sizes the tail so one loan range holds that share of a step's peak — the gate/up range is
    all the tail's gate/up slots past the passes' buffers. Lent steps serve the resident store's outputs."""
    from holon_std.tiered.store import LOAN_SLACK, _profiled_tokens

    file, prompts, want = gpu_moe_wide
    opts = {"moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.ring_lend": "tail",
            "moe.lend_peak": "require", "moe.cache": "*=0.75", "moe.ram": "*=0.25",
            "moe.placement": "placement.promote@1", "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1",
            "moe.loan_fixed_mib": 64, "moe.loan_margin": 1.2, "moe.loan_taper": "64,192", "moe.loan_largest": 0.9}
    engine = _gpu_engine(file, host_memory=1 << 29, plugin_options=opts)
    store = engine.ctx.shared_objects[KEY]
    tokens, align = _profiled_tokens(engine.ctx.options), store._align
    peak = store._peak_per_token * tokens
    assert store._loan_bytes == -(-(int(peak * 1.2) + LOAN_SLACK) // align) * align       # (256 tokens: past 192)
    share = [store._loan_per_token * at for at in (64, 128, 192)]
    assert [store._step_loan_bytes(at) for at in (64, 128, 192)] == [
        -(-(int(s) + f) // align) * align for s, f in zip(share, (64 << 20, 32 << 20, LOAN_SLACK), strict=True)]
    cls, _ = store.tail                          # a largest allocation past every range the loan's sizing gives
    classes, cap = list(store.classes.values()), (lambda c: c.cap)
    buffers, loan = max(f() for f in store.tail_bytes), store._loan_bytes
    plain, n = store._split(buffers, loan, cls, cap), store._split(buffers, loan, cls, cap, 2 * loan)
    ranges = [n * classes[cls].unit_bytes - buffers, *(m * classes[c].unit_bytes for c, m in store._down(n).items())]
    assert n > plain and max(ranges) >= 2 * loan, (plain, n, ranges, loan)
    steps, use = [], store._loan.use_ranges

    def recorded(ranges):
        cls, n = store._step_tail
        unit = list(store.classes.values())[cls].unit_bytes
        buffers = max(f(store.step_tokens) for f in store.tail_bytes)
        steps.append((max(len(r) for r in ranges), store._largest(store.step_tokens),
                      store._step_gu_loan == n * unit - buffers))
        return use(ranges)
    store._loan.use_ranges = recorded
    assert _served(engine, prompts) == want
    assert steps and all(big >= need > 0 and whole for big, need, whole in steps), steps
    assert store.whole_tail and store.loan_overflows == 0
    engine.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("pass_rows", [512, 2048, 4096, 8192, 16384])
def test_a_tail_holds_a_loan_range_for_the_largest_allocation_at_every_pass_size(gpu_moe_wide, tmp_path, pass_rows):
    """D8k (docs/dev/v1-glm.md): the tail the startup profile sizes holds one loan range of a step's largest
    allocation (`moe.loan_largest` of the profiled peak) whatever its passes' buffers, at every `ggml_cuda.pass_rows`
    choice (2 KiB a row here: 1-32 MiB beside a 30 MiB peak), and so does a lent step's share of it — on GLM's shape,
    down pieces near a gate/up piece's size. The tail was sized before the profile's peak per token was set, its
    largest range there by chance: GLM's 32K step at 8,192-row passes found none for its 2 GiB residual streams."""
    from holon_std.tiered.store import _profiled_tokens

    file, _, _ = gpu_moe_wide                    # gate/up Q4_0, down Q8_0 and Q5_0 (GLM: IQ2_XXS, IQ3_XXS and IQ2_S)
    write(file.parent / "hf", tmp_path / "moe.gguf", Q.Q4_0,
          types={"0.ffn_down_exps": Q.Q8_0, "1.ffn_down_exps": Q.Q8_0, "ffn_down_exps": Q.Q5_0})
    opts = {"moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.ring_lend": "tail",
            "moe.lend_peak": "require", "moe.cache": "*=0.75", "moe.ram": "*=0.25",
            "moe.placement": "placement.promote@1", "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1",
            "moe.loan_margin": 1.0, "moe.loan_largest": 1.0}
    engine = _gpu_engine(tmp_path / "moe.gguf", host_memory=1 << 29, plugin_options=opts)
    store = engine.ctx.shared_objects[KEY]
    tokens, peak, buffers = _profiled_tokens(engine.ctx.options), 30 << 20, pass_rows << 11
    store.tail_bytes[:] = [lambda tokens=None: buffers]
    assert store.lend_activations(peak) and store.whole_tail
    big, cap, classes = store._largest(tokens), (lambda c: c.cap), list(store.classes.values())
    cls, n = store.tail
    for n_at in (n, min(n, store._split(buffers, store._step_loan_bytes(tokens), cls, cap, big))):    # the step's
        ranges = [n_at * classes[cls].unit_bytes - buffers,
                  *(m * classes[c].unit_bytes for c, m in store._down(n_at).items())]
        assert max(ranges) >= big > 0.99 * peak, (pass_rows, n_at, ranges, big)
    engine.close()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_a_staging_step_stages_in_rounds_through_fixed_slots(gpu_moe_wide):
    """D8s (docs/dev/v1-glm.md): under exclusive tiers and F10 a lending step stages its misses in rounds — RAM misses
    by a round's slots, SSD reads by the landing rows — through one fixed set of slots a class, each part computed
    before the next one's gather reuses them: it lends the fixed set, not a layer's most missing, and serves the
    resident store's outputs with every piece's bytes the file's. Each round plant is caught."""
    from holon_judge import planted
    from holon_std.tiered import store as tiered_store

    file, prompts, want = gpu_moe_wide
    opts = {"moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.ring_lend": "tail",
            "moe.lend_peak": "require", "moe.cache": "*=0.75", "moe.ram": "*=0.25",
            "moe.placement": "placement.promote@1", "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1",
            "moe.copy_beside": "off"}                # (D8s's own gathers; D8g's: the next test)
    for plant in (None, "round_slots_reused_early", "round_slot_shared", "part_one_role", "spare_lent_outside_tail"):
        # (the spare plant on steps that lend no tail: a tail step releases the spares before its other lending)
        tail = {"moe.ring_lend": "coldest", "moe.lend_peak": "off"} if plant == "spare_lent_outside_tail" else {}
        with planted.PLANTS[plant]() if plant else contextlib.nullcontext() as fired:
            engine = _gpu_engine(file, host_memory=1 << 29, plugin_options=opts | tail)
            store, lends, rounds = engine.ctx.shared_objects[KEY], [], []
            store.taps.append(lambda step, record, lends=lends: lends.append(record["lends"])
                              if record["lends"] is not None else None)
            begin = store.begin

            def watched(store=store, begin=begin, rounds=rounds):
                begin()
                rounds.append((store.lent is not None and store.lent.rounds, _spares_lent(store), _loan_free(store)))
            store.begin = watched
            try:                                     # (twice: the second's lending steps meet F10's spares, set up)
                got = [_served(engine, prompts) for _ in range(2)]
                ok = (got == [want, want], _bytes_exact(store) == 0)
            except RuntimeError as e:                    # (a plant the store refuses)
                assert plant is not None, e
                ok = (False,)
            ok = (*ok, not any(n for _, n, _ in rounds))        # no spare's slots lent outside the tail
            if plant is None:                         # (and D8w's tail over both down classes: no piece in the loan)
                assert any(r for r, _, _ in rounds) and all(ok) and all(f for _, _, f in rounds), (rounds, ok)
                with mock.patch.object(tiered_store, "ROUNDS_BELOW", 100):    # a step of more prompt tokens than
                    rounds.clear()                                            # the gate: no rounds (D8s)
                    engine.generate([[(t + 7) % 512 for t in prompts[-1]]], GREEDY)
                    assert rounds and not any(r for r, _, _ in rounds), rounds
                assert len(store.tail_down) == 2 and _slots_once(store), store.tail_down
                fixed = store.rounds_q + store.landing_rows
                assert max(int(x[1:].max()) for x in lends) <= fixed, lends   # each class lends the fixed set
                _held_once(store)
            else:
                assert fired and not all(ok), (plant, fired, ok)
            engine.close()


@pytest.fixture(scope="module")
def gpu_moe_wide_one(gpu_moe_wide):
    """`gpu_moe_wide`'s resident outputs at one sequence (a graphed step's staging, and so the landing rows, 8 wide)."""
    file, prompts, _ = gpu_moe_wide
    engine = _gpu_engine(file, max_seqs=1)
    want = _served(engine, prompts)
    engine.close()
    return file, prompts, want


@contextlib.contextmanager
def _beside_delayed(late: str, ns: int = 10_000_000):
    """D8g's delayed-producer harness: `late` "copies" (each copy's provider stream sleeps first), "producers" (and
    each SSD request and read-ahead set: the worker sees it late), "worker" (each eager signal of the I/O worker on a
    stream of its own after a sleep, a wait on its slot after it: the worker reads late, as a busy one does) or
    "readers" (main before each part's pass: its slots' readers late, for the fences that guard their reuse)."""
    from holon_std.tiered.resolver import _Beside
    from holon_std.tiered.store import TieredStore

    copy, rows, ahead, land = _Beside.copy, TieredStore.round_rows, TieredStore.read_ahead, _Beside.land
    launch = TieredStore.before_launch

    def late_copy(self, inst, *a):
        with torch.cuda.stream(inst.stream):
            torch.cuda._sleep(ns)
        return copy(self, inst, *a)

    def late_request(f):
        def run(self, *a, **kw):
            torch.cuda._sleep(ns)
            return f(self, *a, **kw)
        return run

    def late_land(self, i):
        land(self, i)
        torch.cuda._sleep(ns)
    side, signalled, patched = torch.cuda.Stream(), {}, []

    def late_worker(self, step):                 # (on the engine's I/O worker, once: graphs' signals as they are)
        io = self.io if self.exclusive else None
        if io is not None and io not in patched:
            signal, wait = io.signal, io.wait

            def deferred(slot):
                if torch.cuda.is_current_stream_capturing():
                    return signal(slot)
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    torch.cuda._sleep(ns)
                    signal(slot)
                    signalled[slot] = torch.cuda.Event()
                    signalled[slot].record()

            def waiting(slot):                   # (a wait follows its slot's deferred signal: the word's protocol)
                if slot in signalled and not torch.cuda.is_current_stream_capturing():
                    torch.cuda.current_stream().wait_event(signalled.pop(slot))
                wait(slot)
            io.signal, io.wait = deferred, waiting
            patched.append(io)
        return launch(self, step)
    with contextlib.ExitStack() as stack:
        stack.callback(lambda: [(delattr(io, "signal"), delattr(io, "wait")) for io in patched])
        patches = {"copies": [(_Beside, "copy", late_copy)], "readers": [(_Beside, "land", late_land)],
                   "worker": [(TieredStore, "before_launch", late_worker)],
                   "producers": [(_Beside, "copy", late_copy), (TieredStore, "round_rows", late_request(rows)),
                                 (TieredStore, "read_ahead", late_request(ahead))]}[late]
        for cls, name, f in patches:
            stack.enter_context(mock.patch.object(cls, name, f))
        yield


_BESIDE = [(v, None, late, 1) for v in ("rounds", "stage") for late in ("producers", "copies", "readers")]
_BESIDE += [("stage", None, "worker", 1), ("stage", None, "copies", 4)]       # (four: steps with decode rows too)
_BESIDE += [("rounds", p, late, 1) for p, late in (
    ("worker_late", "producers"), ("gather_part_unwaited", "copies"), ("round_set_reused_early", "readers"),
    ("ssd_set_reused_early", "readers"), ("fence_waits_unrecorded", "readers"), ("landing_half_reused_early", "copies"),
    ("round_rows_shared", "producers"), ("io_word_stale", "worker"), ("cuda_event_on_generic", "copies"),
    ("readback_unwaited", "producers"))]
_BESIDE += [("stage", p, late, 1) for p, late in (("readahead_wait_after_tiered", "worker"),
                                                 ("readahead_slot_reused", "copies"))]


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("variant, plant, late, seqs", _BESIDE)
def test_gathers_beside_the_compute_hold_their_fences(gpu_moe_wide, gpu_moe_wide_one, variant, plant, late, seqs):
    """D8g (docs/dev/v1-glm.md): an eager staging step's gathers by copy engines beside the compute — in rounds through
    the fixed set (steps below ROUNDS_BELOW), or each miss in its own staging slot (`stage`, here from any size, with
    read-ahead hits; at four sequences, steps with decode rows too) — front parts of 8, SSD rounds of 8 landing rows in
    two halves — serve the
    resident store's outputs with every piece's bytes the file's, under each of the delayed-producer harness's delays
    (`_beside_delayed`; and a late worker, `worker_late`). Each plant, one fence dropped under the delay that would
    expose it, is caught where its condition occurred."""
    from holon_judge import planted
    from holon_std.tiered import store as tiered_store
    from holon_std.tiered.resolver import _Beside

    file, prompts, want = gpu_moe_wide_one if seqs == 1 else gpu_moe_wide
    opts = {"moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.ring_lend": "tail",
            "moe.lend_peak": "require", "moe.cache": "*=0.75", "moe.ram": "*=0.1",
            "moe.placement": "placement.promote@1",
            "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1", "moe.copy_beside": variant,
            "cpu_task.timeout": 2.0}
    load, calls = _Beside.load, []

    def spy(self):
        fresh = self.parts is None
        parts = load(self)
        if fresh:
            calls.append((self.rounds, len(self.front), self.n_rounds, int(self.lists["n_hit"].max())))
        return parts
    below = tiered_store.ROUNDS_BELOW if variant == "rounds" else 0
    with (mock.patch.object(_Beside, "load", spy), mock.patch.object(tiered_store, "ROUNDS_BELOW", below),
          mock.patch.object(tiered_store, "ON_DEMAND", 4), mock.patch.object(tiered_store, "ROUND_SLOTS", 8),
          _beside_delayed(late), planted.PLANTS[plant]() if plant else contextlib.nullcontext() as fired):
        engine = _gpu_engine(file, host_memory=1 << 29, max_seqs=seqs, plugin_options=opts)
        store = engine.ctx.shared_objects[KEY]
        store.read_ahead_tokens = 8
        try:                                     # (each plant fails by its outputs or bytes — not by a wait's timeout
            same, wrong, failed = _served(engine, prompts) == want, _bytes_exact(store), None     # or a refusal —, but
        except (IndexError, RuntimeError) as e:  # readback_unwaited's lists, another call's, may index past a table,
            if not isinstance(e, IndexError) and "outside src or dst" not in str(e):     # or a copy refuse them)
                raise
            same, wrong, failed = False, -1, e
        engine.close()
    assert failed is None or plant == "readback_unwaited", failed
    if plant in (None, "worker_late"):
        assert calls and all(r == (variant == "rounds") for r, *_ in calls), calls
        fronts, rounds, hits = (max(c[i] for c in calls) for i in (1, 2, 3))
        assert same and not wrong, (same, wrong)
        assert fronts >= 2 and rounds >= (3 if seqs == 1 else 1), (fronts, rounds)
        assert plant is None or fired["deferred"], fired
        assert late != "worker" or hits, calls        # (the late worker delayed read-ahead hits)
    else:
        assert any(fired.values()) and not (same and not wrong), (plant, fired, same, wrong)


@pytest.fixture(scope="module")
def gpu_moe_long(gpu_moe_wide):
    """`gpu_moe_wide` with prompts of two passes and more (steps of 1,024 rows, passes of 512 on this model's
    types), the resident store's outputs at one and four sequences."""
    file, _, _ = gpu_moe_wide
    gen = torch.Generator().manual_seed(9)
    prompts = [torch.randint(0, 512, (n,), generator=gen).tolist() for n in (1300, 40, 1100, 7)]
    long = {"max_step_tokens": 1024, "max_model_len": 2048}
    want = {}
    for seqs in (1, 4):
        engine = _gpu_engine(file, max_seqs=seqs, **long)
        want[seqs] = _served(engine, prompts)
        engine.close()
    return file, prompts, want, long


_UNION = [(p, s, None) for p in ("placement.promote@1", "") for s in (1, 4)]
_UNION += [("placement.promote@1", 1, plant) for plant in ("union_one_pass", "gather_part_unwaited")]


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("policy, seqs, plant", _UNION)
def test_a_layer_gathers_its_misses_once_for_its_passes(gpu_moe_long, policy, seqs, plant):
    """D8h (docs/dev/v1-glm.md): a staging step of more than one pass, its gathers beside the compute (D8g's stage
    mode, here from any size), resolves each layer once over all its rows — one union of misses, one set of parts —,
    pass 1 computing part by part, the later passes at once; the resident store's outputs and every piece's bytes,
    under F10 and F11, at one sequence and at four (decode rows beside a prompt of two passes). Planted, the resolve
    over the first pass's rows only (`union_one_pass`) and the parts unwaited under late copies
    (`gather_part_unwaited`) each fail on bits."""
    from holon_judge import planted
    from holon_std.moe.offload import Offload
    from holon_std.tiered import store as tiered_store
    from holon_std.tiered.resolver import _Beside

    file, prompts, want, long = gpu_moe_long
    opts = {"moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.ring_lend": "tail",
            "moe.lend_peak": "require", "moe.cache": "*=0.75", "moe.ram": "*=0.1", "moe.placement": policy,
            "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"}
    union, load, calls = Offload.route_union, _Beside.load, collections.Counter()

    def spy_union(self, ids, x):
        calls["union"] += 1
        calls["rows"] = max(calls["rows"], len(ids))
        return union(self, ids, x)

    def spy_load(self):
        if self.parts is None:
            calls["moves"] += int(self.store.n_moves[self.resolver.group % 2].max())   # (a sync: a test's)
        return load(self)
    with (mock.patch.object(Offload, "route_union", spy_union), mock.patch.object(_Beside, "load", spy_load),
          mock.patch.object(tiered_store, "ROUNDS_BELOW", 0),
          _beside_delayed("copies") if plant else contextlib.nullcontext(),
          planted.PLANTS[plant]() if plant else contextlib.nullcontext() as fired):
        engine = _gpu_engine(file, host_memory=1 << 29, max_seqs=seqs, plugin_options=opts, **long)
        store = engine.ctx.shared_objects[KEY]
        ok = _served(engine, prompts) == want[seqs] and _bytes_exact(store) == 0
        engine.close()
    assert calls["union"] and calls["rows"] > 512, calls
    assert ok == (plant is None) and (plant is None or any(fired.values())), (plant, ok, fired)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("layout", ["ram", "full_step"])
def test_a_union_of_a_steps_rows_resolves_as_its_passes_do(gpu_moe_long, layout):
    """D8h's union outside its gate: `ram` (B's D8h-m2) forces it on a RAM-holds-all layout (a streaming ring and
    the union resolved at once: every entry part −1) — the resident store's bits; `full_step` (A's D8h-m4) resolves
    and places one union of a 32,768-row step's 32,768 × k entries under exclusive tiers — the resident store's
    bits and every piece's bytes."""
    from holon_std.moe.offload import Offload
    from holon_std.tiered import store as tiered_store

    file, prompts, want, long = gpu_moe_long
    rows, union = [], Offload.route_union
    spy = mock.patch.object(Offload, "route_union", lambda self, ids, x: rows.append(len(ids)) or union(self, ids, x))
    if layout == "ram":
        opts = {"moe.store": "std.moe.offload@1", "moe.cache": "*=0.25", "moe.ring_tokens": 512}
        with spy, mock.patch.object(Offload, "gathers_once", lambda self, tokens: tokens > self.pass_tokens(tokens)):
            engine = _gpu_engine(file, host_memory=1 << 29, max_seqs=1, plugin_options=opts, **long)
            assert _served(engine, prompts) == want[1]
            engine.close()
        assert rows and max(rows) > 512, rows
        return
    big = {"max_step_tokens": 32768, "max_model_len": 33792}
    prompt = [torch.randint(0, 512, (32768,), generator=torch.Generator().manual_seed(4)).tolist()]
    resident = _gpu_engine(file, max_seqs=1, **big)
    ref = _served(resident, prompt)
    resident.close()
    opts = {"moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.cache": "*=0.75", "moe.ram": "*=0.1",
            "moe.placement": "placement.promote@1", "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"}
    with spy, mock.patch.object(tiered_store, "ROUNDS_BELOW", 0):     # (the coldest lent: a 32K step's loan past
        #                                                                  this model's tail)
        engine = _gpu_engine(file, host_memory=1 << 29, max_seqs=1, plugin_options=opts, **big)
        assert _served(engine, prompt) == ref and _bytes_exact(engine.ctx.shared_objects[KEY]) == 0
        engine.close()
    assert max(rows) == 32768, rows


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("passes", ["one", "two"])
def test_exclusive_tier_steps_fit_the_profiled_reserve(request, passes):
    """Step 1's review C S1-m2 (test_reserve.py's rule under exclusive tiers, docs/dev/v1-glm.md D9): with the loan
    required, each step's peak beyond what was allocated before it is at most the profiled reserve plus the step's
    own activation loan — staged prompt steps of a full width and a tail, and a prompt's chunks beside decode rows (a
    step with a decode row never reads ahead: docs/dev/v1-glm.md D8f); and (`two`, B's D8h-m1) steps of two passes
    whose layers gather once (D8h: every pass's routes held across the first pass)."""
    from holon.core import platform
    from holon_std.moe.offload import Offload
    from holon_std.tiered import store as tiered_store

    two, unions, union = passes == "two", [], Offload.route_union
    file, _, _ = request.getfixturevalue("gpu_moe_wide" if two else "gpu_moe")     # (wide: a loan beside 2 passes)
    long = {"max_step_tokens": 1024, "max_model_len": 2048} if two else {}
    stack = contextlib.ExitStack()
    if two:                                          # (D8g's stage mode from any size: steps of two passes gather once)
        stack.enter_context(mock.patch.object(tiered_store, "ROUNDS_BELOW", 0))
        stack.enter_context(mock.patch.object(Offload, "route_union",
                                              lambda self, ids, x: unions.append(len(ids)) or union(self, ids, x)))
    peaks, measure = [], platform.peak_memory
    opts = {"moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.ring_lend": "tail",
            "moe.lend_peak": "require", "moe.cache": "*=0.75", "moe.ram": "*=0.5",
            "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"}
    with stack, mock.patch.object(platform, "peak_memory",
                                  lambda device, fn: peaks.append(measure(device, fn)) or peaks[-1]):
        engine = _gpu_engine(file, host_memory=1 << (29 if two else 28), plugin_options=opts, **long)
        store, reserve, gen = engine.ctx.shared_objects[KEY], peaks[-1], torch.Generator().manual_seed(3)
        _reserve_steps(engine, store, reserve, gen, 1300 if two else 300)
    assert two == bool(unions) and all(n > 512 for n in unions), unions
    engine.close()


def _reserve_steps(engine, store, reserve, gen, prompt) -> None:
    """`test_exclusive_tier_steps_fit_the_profiled_reserve`'s steps: two decoding rows, then a prompt of `prompt`
    tokens beside them; each step's peak within the reserve and its own loan."""
    store.read_ahead_tokens, hooks, ahead = 8, engine.executor._before, []

    def launched(step, launch=store.before_launch):  # each step: (a decode row, read ahead) — never both (C S1-m3)
        out = launch(step)
        ahead.append((bool(np.isin(step.row_kind, (RowKind.DECODE, RowKind.VERIFY)).any()), store._ahead))
        return out
    hooks[hooks.index(store.before_launch)] = launched
    params = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)
    for n in (40, 40, prompt):                       # two decoding rows, then a prompt beside them
        engine.add(torch.randint(0, 512, (n,), generator=gen).tolist(), params)
        if n == 40:
            engine.step()
    loans = []
    while engine.has_work():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        engine.step()
        torch.cuda.synchronize()
        loans.append(store._step_loan)
        took = torch.cuda.max_memory_allocated() - before
        assert took <= reserve + store._step_loan, (took, reserve, store._step_loan)
    assert store.exclusive and any(loans) and store.borrowed["stage"], (loans, store.borrowed)
    assert (False, True) in ahead and (True, True) not in ahead and (True, False) in ahead, ahead


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_a_roles_classes_share_its_landing_rows(gpu_moe, tmp_path):
    """Step 1's review C S1-m2 (docs/dev/v1-glm.md D8f): the down role of two classes (layer 3's Q8_0, the others' Q4_0,
    as GLM's two down types) reads its SSD misses into one landing pool, row j at the same bytes for both (a layer's
    on-demand rows and the next layers' read-ahead sets never overlap); the streams are the resident store's. Every
    piece's runs in a landing row are as the store derives them from its descriptors (D8g's copies by copy engines:
    the I/O worker's rule)."""
    file, prompts, _ = gpu_moe
    mixed = tmp_path / "mixed.gguf"
    write(file.parent / "hf", mixed, Q.Q8_0, types={"3.ffn_down_exps": Q.Q8_0, "ffn_down_exps": Q.Q4_0})
    resident = _gpu_engine(mixed)
    want = _served(resident, prompts)
    resident.close()
    engine = _gpu_engine(mixed, host_memory=1 << 27, plugin_options={
        "moe.store": "std.moe.offload@1", "moe.cache": "*=0.25", "moe.ram": "*=0.25",
        "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"})
    store = engine.ctx.shared_objects[KEY]
    store.read_ahead_tokens = 8                      # (prompt steps of 8 tokens or more read ahead)
    assert _served(engine, prompts) == want
    down = [store.landing_view(name, device=False)[0] for name, c in store.classes.items() if c.pieces[0][2] == 1]
    assert len(down) == 2 and down[0].data_ptr() == down[1].data_ptr() and down[0].shape == down[1].shape
    assert store.exclusive and engine.ctx.shared_objects["cpu_task.io"].native.io_stats()["bytes"] > 0
    torch.cuda.synchronize()                         # D8g: each piece's runs as the I/O worker lays it out (idle)
    a, table = store.landing_rows, store.io_pool.view(torch.int32, (store.halves, store.roles, 1 + store.landing_rows,
                                                                    3))[0]
    for reads in (x for x in store.reads if not x.ahead and not x.half):
        classes = store.layer_classes[reads.g]
        for at in range(0, store.items, a):
            pieces = [[store._piece_index(c, reads.g, i, r) for i in range(at, min(at + a, store.items))]
                      for r, c in enumerate(classes)]
            for r, p in enumerate(pieces):
                table[r, 0, 0] = len(p)
                table[r, 1:1 + len(p)] = torch.tensor([[x, j, -1] for j, x in enumerate(p)], dtype=torch.int32)
            reads()
            for r, (c, p) in enumerate(zip(classes, pieces, strict=True)):
                runs = store.landing_view(c.name, device=False)[1][:len(p)].numpy()
                assert np.array_equal(runs, store.piece_runs[c.name][p]), (reads.g, r, at)
    engine.close()



@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("heated", [True, False])
def test_a_read_ahead_step_reads_each_layers_hottest_ssd_only_experts(gpu_moe, heated):
    """D8u (b) (its narrow width tried and reverted; one pick kept): a read-ahead step reads ahead each layer's A_ra
    hottest SSD-only experts by the store's heat (`moe.ram_heat`), ties by id, into its set's rows in id order — here
    `moe.read_ahead` 2, so a set exceeds it: the hottest, not D8f's lowest ids —; without a heat the lowest ids
    (D8f's rule); the streams are the resident store's. A read-ahead signalled and never waited is refused at the next
    step's start (A's m2)."""
    from holon_std.tiered import store as tiered_store

    file, prompts, _ = gpu_moe
    resident = _gpu_engine(file)
    want = _served(resident, prompts)
    resident.close()
    width = 2                                        # (the tiny model's 8 experts a layer)
    engine = _gpu_engine(file, host_memory=1 << 27, plugin_options={
        "moe.store": "std.moe.offload@1", "moe.cache": "*=0.25", "moe.ram": "*=0.25", "moe.ram_heat": "0.999",
        "moe.read_ahead": str(width), "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"})
    store = engine.ctx.shared_objects[KEY]
    store.read_ahead_tokens = 8                      # (prompt steps of 8 tokens or more read ahead)
    heat = store.ram_heat                            # a heat against the ids' order, with ties (a pure prompt step's)
    heat.copy_((torch.arange(heat.numel(), device=heat.device).flip(0) % 3).view_as(heat).float())
    if not heated:
        store.heat = None
    seen, read = [], tiered_store.TieredStore.read_ahead

    def recorded(self, g):
        only = (self.residency.device[g] == -2).all(-1).cpu()
        h = self.heat[g].double().cpu() if self.heat is not None else torch.zeros(only.shape, dtype=torch.float64)
        read(self, g)
        seen.append((only, h, self.ahead_map[g].cpu().clone()))

    with mock.patch.object(tiered_store.TieredStore, "read_ahead", recorded):
        assert _served(engine, prompts) == want
        assert store.ahead_rows == width
        assert seen and any(int(only.sum()) > width for only, _, _ in seen)    # (the width bites,
        assert not heated or any(len(set(h[only].tolist())) > 1 for only, h, _ in seen)    # by heat)
        for only, h, rows in seen:
            ids = [int(e) for e in torch.nonzero(only).flatten()]
            hot = sorted(sorted(ids, key=lambda e: (-float(h[e]), e))[:width])     # the hottest, ties by id
            assert [int(e) for e in torch.nonzero(rows >= 0).flatten()] == hot
            assert rows[rows >= 0].tolist() == sorted(rows[rows >= 0].tolist())    # rows in id order, contiguous
        if heated:                                   # a read-ahead never waited: the next step refuses
            store.read_ahead(0)
            with pytest.raises(RuntimeError, match="never waited"):
                store.start_step()
            store.wait_ahead(0)
            torch.cuda.synchronize()
    engine.close()

def _ram_units_hold_their_pieces(store) -> int:
    """Every RAM-resident expert's units hold its pieces' bytes, as the file has them (exclusive tiers); how many."""
    from holon_std.tiered import files
    from holon_std.tiered.store import _offsets

    res, host = store.residency.device.cpu().numpy(), store.host_units.cpu().numpy()
    checked = 0
    for g, classes in enumerate(store.layer_classes):
        for i in np.flatnonzero((res[g] == -1).all(-1)):
            for r, c in enumerate(classes):
                piece = c.pieces[store._piece_index(c, g, int(i), r)][3]
                want = np.empty(c.unit_bytes, dtype=np.uint8)
                files.read([(e, want[o:o + e.nbytes]) for e, o in _offsets(piece)])
                got = c.host_view()[int(host[g, i, r])].numpy()[:c.unit_bytes]
                assert np.array_equal(got, want), (g, int(i), r)
            checked += 1
    return checked


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("terms", ["default", "mode2"])
def test_parked_experts_hold_their_pieces_beside_the_activation_loan(gpu_moe, mode2, terms):
    """Step 1's review S1-B1 (docs/dev/v1-glm.md D8b, D9): RAM with free units, the tail lent with the step's activation
    loan (`require`): lent experts are parked from intact slots — the step's lending runs before the loan is handed to
    the forward, an expert holding a loan slot goes to the SSD, the copies are waited for before a lent slot is reused —
    so every RAM-resident expert's units hold its pieces as the file does, and the streams are the resident store's."""
    from holon_judge.planted import parking_late, parking_unwaited

    file, prompts, want = gpu_moe
    opts = {"moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.ring_lend": "tail",
            "moe.lend_peak": "require", "moe.cache": "*=0.75", "moe.ram": "*=0.5",
            "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"} if terms == "default" else mode2()
    for plant in (contextlib.nullcontext, parking_late, parking_unwaited):     # the fence, then under a forced delay
        with plant() as fired:
            engine = _gpu_engine(file, host_memory=1 << 28, plugin_options=opts)
            store = engine.ctx.shared_objects[KEY]
            got = _served(engine, prompts)
            if terms == "mode2":                     # again, after decoding: F10's spares set up (released)
                assert int((store.state["spare"][..., 0] > 0).sum()) > 0
                got = got if got != want else _served(engine, prompts)
            assert store.exclusive and store.borrowed["stage"] and store.parked > 0, (store.borrowed, store.parked)
            assert store._parked is None             # each step's parking copies waited for (C S1-M2's fence)
            wrong = _bytes_exact(store)
            engine.close()
        assert plant is contextlib.nullcontext or fired["delayed"]
        assert (got == want and not wrong) == (plant is not parking_unwaited), wrong


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("plant, terms", [(p, "default") for p in (None, "loan_parking_late",
                                                                    "loan_write_before_parked", "parking_event_early",
                                                                    "loan_wait_unrecorded")]
                         + [(None, "mode2"), ("loan_write_before_parked", "mode2")])
def test_a_tail_steps_loan_experts_are_parked_before_the_forward_writes_the_loan(gpu_moe_wide, mode2, plant, terms):
    """D8l (docs/dev/v1-glm.md): a tail step's lent experts holding slots of its activation loan are parked in RAM as
    the others (once they went to the SSD): RAM with free units here, no lent expert goes to the SSD, and every RAM
    unit holds its pieces as the file does, the streams the resident store's — the forward waits for the parking
    copies before it writes the loan. Under the copies delayed on a stream of their own and the loan written whole as
    it is handed out (`loan_parking_late`: still exact): the copies issued after the loan's writes with no wait
    (`loan_write_before_parked`) or with the event recorded before them (`parking_event_early`) leave RAM units with
    overwritten bytes; no copy issued before the loan (`loan_wait_unrecorded`) is refused, a failed step's stale event
    notwithstanding (A's m2). The early start makes no synchronizing torch call (A's m1)."""
    from holon_judge import planted

    file, prompts, want = gpu_moe_wide
    opts = {"moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.ring_lend": "tail",
            "moe.lend_peak": "require", "moe.cache": "*=0.75", "moe.ram": "*=0.5",
            "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"} if terms == "default" else mode2()
    with getattr(planted, plant)() if plant else contextlib.nullcontext() as fired:
        engine = _gpu_engine(file, host_memory=1 << 29, plugin_options=opts)
        store, to_ssd = engine.ctx.shared_objects[KEY], []
        lend_tail = store.lend_tail

        def counted(cls, n):                     # the tail's experts the step sent to the SSD (read at the end)
            picked = lend_tail(cls, n)
            to_ssd.append((picked.sum(), ((store.residency.device == -2).all(-1) & picked).sum()))
            return picked
        store.lend_tail = counted
        start = store.start_step

        def strict(early=False):                 # an early start enqueues, never waits for the device (A's m1)
            if not early:
                return start(early)
            torch.cuda.set_sync_debug_mode("error")
            try:
                return start(early)
            finally:
                torch.cuda.set_sync_debug_mode("default")
        store.start_step = strict
        if plant == "loan_wait_unrecorded":      # (a stale event a failed step left is no parking either: A's m2)
            engine.add(max(prompts, key=len), SamplingParams(temperature=0.0, max_tokens=1))
            store._parked = torch.Event(device=torch.device("cuda"))
            store._parked.record()
            with pytest.raises(RuntimeError, match="parking copies were not issued"):
                engine.step()
            engine.close()
            return
        got = _served(engine, prompts)
        if terms == "mode2":                     # again, after decoding: F10's spares set up, the tail releases them
            assert int((store.state["spare"][..., 0] > 0).sum()) > 0
            got = got if got != want else _served(engine, prompts)
        wrong = _bytes_exact(store)
        to_ssd = [(int(n), int(m)) for n, m in to_ssd]
        assert store.exclusive and store.borrowed["stage"] and store._step_loan_bytes(64) and to_ssd
        engine.close()
    if plant in (None, "loan_parking_late"):
        assert got == want and not wrong and sum(n for n, _ in to_ssd) and not any(m for _, m in to_ssd), \
            (wrong, to_ssd)
        assert plant is None or fired["delayed"] and fired["filled"]
    else:
        assert fired["delayed"] and fired["filled"] and wrong, plant


@pytest.fixture(scope="module")
def gpu_moe_wide_fresh(gpu_moe_wide):
    """Two fresh prompts for `gpu_moe_wide`, each long enough to lend, and the resident store's greedy streams of each
    served alone (a prompt step of its own, then its decode)."""
    file, _, _ = gpu_moe_wide
    gen = torch.Generator().manual_seed(3)
    fresh = [torch.randint(0, 512, (n,), generator=gen).tolist() for n in (256, 250)]
    engine = _gpu_engine(file)
    want = [engine.generate([p], GREEDY) for p in fresh]
    engine.close()
    return fresh, want


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("plant", [None, "parked_row_superset_runs", "row_hit_before_parked",
                                   "row_move_before_unmapped", "row_mapped_before_copied", "row_reused_before_moved",
                                   "row_reused_before_mapped", "reserved_unit_named", "row_moved_after_promoted"])
def test_a_short_prompt_parks_its_lent_experts_in_the_read_ahead_rows(gpu_moe_wide, gpu_moe_wide_fresh, mode2, plant):
    """D8x (docs/dev/v1-glm.md), under the deployment's terms: a short prompt step (no decode row, below the read-ahead
    steps' size: every prompt step here) parks the lent experts RAM has no free unit for in the idle read-ahead rows —
    its later layers hit them there —; at its end each is paired with RAM's coldest of its classes, whose units it
    reserves; the thread copies the rows into them in chunks (4 here: several a prompt), chunk i mapped at launch
    (step + lag + i), the next short prompt mapping every chunk left first. Checked after every step: the bits are the
    resident store's, every piece the map names holds its bytes, each tier holds an expert once (a reservation
    accepted as such), no cache slot is lost; experts were parked, moved, and some routed meanwhile (reconciled). Each
    of D8x's plants breaks one rule and is caught (`holon_judge.planted._d8x`)."""
    from holon_judge.planted import d8x_plant

    file, prompts, want = gpu_moe_wide
    with d8x_plant(plant) if plant else contextlib.nullcontext(collections.Counter()) as fired:
        engine = _gpu_engine(file, host_memory=1 << 29, plugin_options=mode2(**{"moe.ram": "*=0.25"}))   # (RAM full)
        store, step, bad, cache, slots = engine.ctx.shared_objects[KEY], engine.step, collections.Counter(), {}, []
        store.move_chunk = 2
        heat = torch.rand(store.ram_heat.shape, generator=torch.Generator().manual_seed(0))   # RAM's heat spread (a
        store.ram_heat.copy_(heat * 8)                     # lent expert hotter than RAM's coldest: no profile here)

        def checked():
            out = step()
            bad["bytes"] += _bytes_exact(store, cache)
            getattr(fired, "release", lambda: None)()        # (a held thread goes on once the bytes were checked)
            try:
                _held_once(store)
            except AssertionError:
                bad["held"] += 1
            slots.append(_slots_held(store).sum(0))
            bad["slots"] += not torch.equal(slots[-1], slots[0])
            return out
        engine.step = checked
        got = _served(engine, prompts)
        assert int((store.state["spare"][..., 0] > 0).sum()) > 0            # again: F10's spares set up
        got = got if got != want else _served(engine, prompts)
        fresh, streams = gpu_moe_wide_fresh                 # a prompt step alone, its decode during its move
        same = got == want and [engine.generate([p], GREEDY) for p in fresh] == streams
        counts, stats = store.move_counts.tolist(), dict(store.move_stats)
        engine.close()
    if plant is None:
        assert same and not +bad, bad
        assert counts[0] > counts[1] > 0 and counts[2] > 0 and counts[3] > 0 and stats["moves"], (counts, stats)
    else:
        assert sum(fired.values()) and (not same or +bad), (plant, dict(fired), bad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("lend", ["tail", "coldest"])
def test_a_lending_prompt_step_stages_in_a_single_buffer(gpu_moe, lend):
    """Step 7, exclusive tiers: a lending prompt step stages in a single buffer of a layer's most missing pieces, not a
    pass of every expert (C's L3: with the coldest lent, 3 of 8 a class here, 8 before); the resident store's bits and
    every piece's bytes, with the tail lent and with the coldest."""
    file, prompts, want = gpu_moe
    opts = {"moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.ring_lend": lend,
            "moe.lend_peak": "require" if lend == "tail" else "off", "moe.cache": "*=0.75", "moe.ram": "*=0.25",
            "moe.ram_heat": 0.9, "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"}
    engine = _gpu_engine(file, host_memory=1 << 27, plugin_options=opts)
    store, lends = engine.ctx.shared_objects[KEY], []
    store.taps.append(lambda step, record: lends.append(record["lends"]) if record["lends"] is not None else None)
    assert _served(engine, prompts) == want
    assert store.exclusive and store.borrowed["stage"]
    if lend == "coldest":                              # the rounds' lending: a layer's most missing, not `prefill`
        most = [int(x[1:].sum(0).max()) for x in lends]
        assert any(most) and max(most) < store.prefill, most
    assert _bytes_exact(store) == 0
    engine.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_the_cpus_share_on_the_gpu_is_bounded_and_run_identical(gpu_moe):
    """F5b on the native path (kernels.expert_io's resolve and row patch, kernels.cpu_task's worker in decode graphs,
    kernels.ggml_cpu's experts): steps of up to 4 tokens give the CPU half the misses (rounded up) at a quarter cached;
    prompt logprobs stay close to the resident store's at the same step size (bounded), two engines serve the same
    bits (static placement; F11's residency is its replay's, the CPU's misses never admitted; with exclusive tiers too,
    the I/O worker reading SSD pieces beside the CPU's), and a dead worker fails
    open: the wait gives up after its time bound and the step's commit stops the engine."""
    from holon_judge.placement import parity, route_dump

    file, prompts, _ = gpu_moe
    small = {"max_step_tokens": 4}                            # every prompt pass small enough for a CPU share
    engine = _gpu_engine(file, **small)
    want = _served(engine, prompts)
    engine.close()
    split = {"moe.store": "std.moe.offload@1", "moe.cache": "*=0.25", "moe.cpu_plan": "1,1,2,2",
             "cpu_task.threads": 2, "cpu_task.timeout": 0.5}
    short = {"moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1", "moe.ram": "*=0.5"}   # + exclusive tiers
    runs = []
    for policy, ram in (("std.tiered.static@1", {}), ("std.tiered.static@1", {}), ("", short), ("", {})):
        engine = _gpu_engine(file, host_memory=1 << 28, plugin_options=split | ram | {"moe.placement": policy},
                             **small)
        with route_dump(engine) as trace:
            runs.append(_served(engine, prompts))
        worker = engine.ctx.shared_objects["cpu_task.compute"].native
        assert worker.served() > 0 and worker.flag() == 0 and engine.stats()["graphs"] > 0
        assert sum(s["cpu"].sum() for s in trace.steps) > 0
        if not policy:
            assert parity(trace)["first_differing_step"] is None and trace.admissions > 0
        if ram:                                   # the I/O worker beside the compute one, both clean
            io = engine.ctx.shared_objects["cpu_task.io"].native
            assert io.served() > 0 and io.flag() == 0
            close = [np.abs(np.array(a) - np.array(b)) for (_, _, a), (_, _, b) in zip(runs[-1][1], want[1],
                                                                                       strict=True) if a]
            assert np.median(np.concatenate([c.ravel() for c in close])) < 0.03
        if policy or ram:
            engine.close()
    assert runs[0] == runs[1]
    assert {(k.rpartition("/")[2], v["provider"]) for k, v in engine.ctx.kernels.items() if k.startswith("std.moe.cpu")
            } == {("cpu_task@1", "kernels.cpu_task"), ("cpu_experts@1", "kernels.ggml_cpu"),
                  ("moe_patch_rows@1", "kernels.expert_io")}
    diff = np.concatenate([np.abs(np.array(a) - np.array(b)).ravel() for (_, _, a), (_, _, b) in
                           zip(runs[0][1], want[1], strict=True) if a])
    same = np.mean([x[0] == y[0] for (_, ids, _), (_, ref, _) in zip(runs[0][1], want[1], strict=True)
                    for x, y in zip(ids, ref, strict=True)])
    assert np.median(diff) < 0.03 and diff.mean() < 0.05 and same > 0.85         # (measured 0.011, 0.016, 0.94; a
    #                                                                                routing flip moves a position)
    worker.kill()                                             # the coordinator stops answering
    engine.add(prompts[1], GREEDY)
    with pytest.raises(RuntimeError, match="kernel_errors"):
        while engine.has_work():
            engine.step()
    assert worker.flag() & 2
    engine.close()


_SIDE_STREAM = """
import sys, types, torch
from holon_std.ffn.units import Sum

class Double(torch.nn.Module):
    def forward(self, x):
        return x * 2

class Late(torch.nn.Module):                       # a first part that reads `acc` late, on the main stream
    joins = True

    def add_residual(self, x, acc, join):
        join()
        torch.cuda._sleep(50_000_000)
        return x + acc

units = {"first": Late(), "rest": Double()}
ctx = types.SimpleNamespace(device="cuda", options={}, unit=lambda p, **kw: units[p],
                            shared=lambda key, make: make(types.SimpleNamespace(device="cuda")))
s = Sum(ctx, hidden=1 << 22, parts=("first", "rest"))
xs = [torch.full((1 << 22,), float(i + 1), device="cuda") for i in range(4)]
torch.cuda.synchronize()
ys = []
if sys.argv[1] == "graph":                         # captured, the side stream's trailing wait inside the capture
    x = xs[0].clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = s(x)
    for xi in xs:
        x.copy_(xi)
        graph.replay()
        ys.append(y.clone())
for x in xs if sys.argv[1] != "graph" else ():
    ys.append(s(x))
    if sys.argv[1] == "release":
        torch.cuda.empty_cache()                   # what an out-of-memory retry does
torch.cuda.synchronize()
print([float(y.min()) for y in ys] == [float(y.max()) for y in ys] == [3.0 * (i + 1) for i in range(4)])
"""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("release", ["reuse", "release", "graph"])
def test_a_sums_side_stream_output_outlives_the_main_streams_read(release):
    """GLM step 6's illegal address (A's M2, the side-stream audit): `std.ffn.sum`'s later parts run on a side stream
    and their sum goes back to that stream's blocks with no event (`units.Sum.forward`) while the main stream's combine
    may not have read it yet. Neither the next sum's side-stream output, which takes that block at once (`reuse`), nor
    the allocator releasing free blocks (`release`: with expandable segments it unmaps one after waiting for its own
    stream only, as an out-of-memory retry does) may pass that read. A main stream that reads late (a sleep before the
    read) while the host runs ahead: each sum is still x + 2x. Captured in a CUDA graph (`graph`), the side stream's
    trailing wait inside the capture leaves no unjoined work and every replay is still x + 2x. In a process of its
    own: an illegal address ends it (`release` does without the wait: GLM step 6's repro)."""
    import subprocess
    import sys
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    out = subprocess.run([sys.executable, "-c", _SIDE_STREAM, release], env=env, capture_output=True, text=True,
                         timeout=300)
    assert out.stdout.strip() == "True", out.stderr[-2000:]
