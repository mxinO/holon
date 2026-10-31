"""Qwen3-MoE end to end on the CPU (test-only reference kernels) against an independent oracle: a tiny random HF
`Qwen3MoeForCausalLM`, written as a GGUF with quantized expert stacks, must serve what transformers computes on the
same (dequantized) weights (docs/dev/m02-moe.md, step 1); then the offload store on the CPU providers, bitwise the
resident store's. The GPU store's tests are `test_moe_engine_*.py`'s (their shared models and checks:
`moe_engine_common.py`)."""
import json
import pathlib
import types
from unittest import mock

import numpy as np
import pytest
import torch
from holon.api import OutOfUnits, RowKind, SamplingParams
from holon.core.batching import host_plan
from holon.core.engine import Engine, EngineConfig
from holon_std.moe.kernels import DECODE, PADDING
from holon_std.moe.offload import KEY
from holon_std.moe.units import CHUNK, MoE
from holon_testing.gguf_files import write
from holon_testing.ref_kernels.kernels import LinearExperts
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM

from .moe_engine_common import (
    GREEDY,
    Q,
    _bytes_exact,
    _engine,
    _fit_at,
    _gguf_and_dense,
    _held_once,
    _served,
    _tail_checked,
)
from .test_engine import _hf_greedy

PREBUILT = ("ggml_cuda", "expert_io", "cpu_task")  # CUDA extensions built if missing (conftest)


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


def test_the_offload_store_serves_what_the_resident_store_does(tiny_moe, tmp_path, capsys):
    """docs/dev/v1-offload.md D4, D5, F1, F5a: with `moe.store = std.moe.offload@1` the experts stay in the file's map
    and in pinned host pools (filled from its extents), classes by (role, type, shape) — here gate/up Q8_0 and down
    Q4_0 —, every miss fetched into staging slots: the outputs are bitwise the resident store's whatever the cache
    holds (every expert by default, as many as a short fill pool's bytes hold, half, a profile's pick per class, none).
    The static policy caches whole experts, F2's warm start: item 0, 1, … of every layer, or a profile's hottest; a
    layer's distinct misses never pass its staging slots, and a step that did would stop the engine. The adaptive
    policies (the default F11, F4, F2–F3's two-phase swaps through the copy stream) serve the same outputs at a
    quarter of the experts while they move them; F11's residency, step by step, is its replay oracle's (L1 parity).
    A profile that cannot seed the cache (a host path, another file's) is warned about and the cache starts cold
    (docs/dev/v1-auto-config.md D4)."""
    import json as json_

    from holon_judge.placement import parity, route_dump
    from holon_std.tiered import store as tiered
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
    capsys.readouterr()
    _engine(file, host_memory=1 << 24, plugin_options={"moe.store": "std.moe.offload@1",
                                                       "moe.placement.profile": str(profile)})
    assert "not used (a name relative to the checkpoint's directory)" in capsys.readouterr().err   # no host path
    shipped = json_.loads(profile.read_text())                 # counted with an MTP head's file and group too
    shipped["checkpoint"].append(["mtp-head.gguf", 1])
    shipped["layers"]["99"] = [1] * 8
    (tmp_path / "shipped").mkdir()
    (tmp_path / "shipped" / "p.json").write_text(json_.dumps(shipped))
    with mock.patch.object(tiered, "PROFILES", tmp_path / "shipped"):   # it seeds; `none` asks for a cold cache
        seeded, cold = (_engine(file, host_memory=1 << 24, plugin_options={"moe.store": "std.moe.offload@1"} | extra)
                        for extra in ({}, {"moe.placement.profile": "none"}))
        assert seeded.ctx.shared_objects[KEY]._profile_counts.any()
        assert not cold.ctx.shared_objects[KEY]._profile_counts.any()
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
    _engine(other, host_memory=1 << 24, plugin_options={"moe.store": "std.moe.offload@1",
                                                        "moe.placement.profile": "moe-a.profile.json"})
    err = capsys.readouterr().err
    assert "not used (counted on" in err and "the cache starts cold" in err
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


@pytest.mark.parametrize("stream, plant", [(0, False), (20, False),
                                           pytest.param(0, True, marks=pytest.mark.slow, id="0-planted")])
def test_a_tail_step_lends_its_passes_a_contiguous_range(wide_moe, capsys, stream, plant):
    """docs/dev/v1-moe-prefill.md D10, D11 on the CPU at FN's E = 512, top-10, under F11 (`moe.ring_lend = tail`): a
    streaming step first lends the tail — call 0, the last slots of the largest class arena (gate/up's), enough for
    the passes' `out` and `scratch` —, then X's rounds (calls 1…), every step's terms as `_tail_checked` asserts them;
    the first, on a cache with no spare empty slots, stages from the rounds' slots without falling short. The passes'
    buffers are the range's (the reference provider stages each chunk through `scratch`, then `out`), so an expert
    left mapped there would read them as weights: the outputs are the resident store's, mixed steps included, and F11's
    residency, empty slots and their stacks are its replay's step by step (the replay models `lend_tail` from the
    slots). With `moe.stream_tokens` 20, a step of 10 to 19 prompt tokens stages its routed misses in large passes
    instead, under the same terms (its tail lent, its rounds' slots its staging). The plant that leaves the tail's
    experts mapped fires and fails parity (`slow`). A configuration whose tail class cannot stage a pass outside the
    range, and every policy without `lend_tail` (F11 and F10 have it), are refused at bind."""
    from holon_judge.placement import parity, route_dump
    from holon_judge.planted import tail_unmap_undone
    from holon_plugins.placement.lfu.reference import replay

    file, prompts, want = wide_moe
    tail = {"moe.store": "std.moe.offload@1", "moe.cache": "*=0.75", "moe.ring_tokens": 10, "moe.ring_lend": "tail",
            "moe.stream_tokens": stream}
    if plant:
        with tail_unmap_undone():
            planted = _engine(file, state_memory=1 << 24, host_memory=1 << 26, chunk_with_decode=4,
                              plugin_options=tail)
            with route_dump(planted) as trace:
                _served(planted, prompts)
        fired = int(capsys.readouterr().err.split("tail-unmap-undone fired: {'lent': ")[1].split("}")[0])
        assert fired > 0 and parity(trace)["first_differing_step"] is not None
        return
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")   # (its byte check synchronizes the device)
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


def test_fit_solves_one_share_once_and_refuses_below_its_floor():
    """docs/dev/v1-glm-ram.md D1, D2, D4, engine-free: `moe.ram=fit`'s f = (left − reserve) / every piece's bytes,
    rounded down to 1/256 (less than 1/256 more RAM moves nothing), each class ⌊f × pieces⌋ units, below 1 — every
    piece fitting beside the reserve is the lock's store (`whole`) —; below the working floor (every class a call's
    keeps) the start is refused for memory with each term in bytes; a lower reading just before the RAM tier's pool
    refuses with both readings, never solving again; no host budget, or a state tier, refuses."""
    from holon_std.tiered import fit

    gib = 1 << 30
    classes = {"gate_up": (512, 3 << 20), "down.a": (256, 1 << 20), "down.b": (256, 1 << 20)}     # 2 GiB of experts
    half = fit.RESERVE + gib
    assert fit.share("moe", half, classes, 32, {}) == (0.5, {"gate_up": 256, "down.a": 128, "down.b": 128})
    assert fit.share("moe", half + (8 << 20) - 1, classes, 32, {})[0] == 0.5               # (1/256: 8 MiB)
    assert fit.share("moe", half + (8 << 20), classes, 32, {})[0] == 0.5 + 1 / 256
    assert fit.share("moe", fit.RESERVE + 2 * gib, classes, 32, {})[0] == 255 / 256       # (exclusive: below 1)
    assert not fit.whole(fit.RESERVE + 2 * gib - 1, 2 * gib) and fit.whole(fit.RESERVE + 2 * gib, 2 * gib)
    assert fit.floor(classes, 32) == 0.125 and fit.floor(classes, 1000) == 1.0       # (a call's keeps: its pieces)
    with pytest.raises(OutOfUnits, match=r"below the working floor: the store needs 3\.82 GiB .* own I/O pools 0\.01 "
                       r"GiB \(moe\.landing\.0 0\.008\), the reserve 3\.56 GiB .* a minimal RAM tier 0\.25 GiB "
                       r"\(f 0\.1250 of the experts' 2\.00 GiB: every class a call's keeps, 32 units\) — and 3\.77 "
                       r"GiB was left .* Short by 0\.05 GiB"):
        fit.share("moe", fit.RESERVE + gib // 5, classes, 32, {"moe.landing.0": 8 << 20})
    ctx = types.SimpleNamespace(options={}, host_left=lambda: 5 * gib)
    fit.recheck(ctx, "moe", 6 * gib, 5 * gib)
    with pytest.raises(OutOfUnits, match=r"fell from 6\.00 GiB, which .* to 5\.00 GiB before the RAM tier's pool"):
        fit.recheck(ctx, "moe", 6 * gib, 5 * gib + 1)
    assert fit.budget(ctx, "moe") == 5 * gib
    with pytest.raises(ValueError, match="needs a host budget"):
        fit.budget(types.SimpleNamespace(options={}, host_left=lambda: None), "moe")
    with pytest.raises(ValueError, match=r"the state tier \(std\.state_tier\) would take too"):
        fit.budget(types.SimpleNamespace(options={"slots": {"tier": "std.state_tier"}}, host_left=lambda: gib), "moe")


def test_fit_where_every_expert_fits_is_the_locks_store(tiny_moe, tmp_path):
    """docs/dev/v1-glm-ram.md D2, B-R10: `moe.ram=fit` (the spill's tiers too) on a host budget holding every expert
    beside the reserve runs as the lock: inclusive, the same arenas, seed and map, the same outputs bit for bit, and
    no landing, I/O or run pools — its exclusive kernels and I/O worker bound in the build, whatever f, and idle. At a
    share below 1 an auxiliary model's experts (a drafter's head) are refused for memory (m8: `holon serve` serves
    without speculation next), a share other than the whole value refused at once, and so is a CPU share beside it."""
    from holon_std.tiered.store import TieredStore

    checkpoint, prompts = tiny_moe
    file, _ = _gguf_and_dense(checkpoint, tmp_path, Q.Q8_0, types={"ffn_down_exps": Q.Q4_0})
    lock = {"moe.store": "std.moe.offload@1", "moe.cache": "*=0.25"}
    fitted = lock | {"moe.ram": "fit", "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"}
    engines = [_engine(file, host_memory=1 << 24, plugin_options=lock), _engine(file, host_memory=4 << 30,
                                                                               plugin_options=fitted)]
    stores = [e.ctx.shared_objects[KEY] for e in engines]
    layout = [([(c.name, c.held, c.at, c.size, c.staging) for c in s.classes.values()], s.residency.host.tolist())
              for s in stores]
    assert layout[0] == layout[1] and not stores[1].exclusive and stores[1].share == 1.0
    assert not [n for n in engines[1].ledger.pools if any(k in n for k in (".landing.", ".io", ".runs."))]
    assert "std.tiered[moe]/moe_tiers@2" in engines[1].ctx.kernels and stores[1].reads and all(
        r.task is not None for r in stores[1].reads)                   # (bound, each an idle task: never signalled)
    assert _served(engines[0], prompts) == _served(engines[1], prompts)
    before = TieredStore.before_materialize

    def headed(self, ctx):                       # the last layer taken as a drafter's head's (FN D12a)
        self.mark_auxiliary(max(self.groups))
        return before(self, ctx)
    with _fit_at(0.5), mock.patch.object(TieredStore, "before_materialize", headed), \
            pytest.raises(OutOfUnits, match="a drafter's head.*without speculation next"):
        _engine(file, host_memory=1 << 30, plugin_options=fitted)
    with pytest.raises(ValueError, match="fit is the whole value"):
        _engine(file, host_memory=1 << 30, plugin_options=fitted | {"moe.ram": "*=fit"})
    with pytest.raises(ValueError, match="moe.cpu_plan beside moe.ram=fit"):        # (its pools after fit's: A m3)
        _engine(file, host_memory=1 << 30, plugin_options=fitted | {"moe.cpu_plan": "1,1,2,2"})
