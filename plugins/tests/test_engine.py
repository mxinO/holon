"""Engine end to end: paged KV, chunked prefill and continuous batching reproduce HF greedy generation."""
import dataclasses
import pathlib

import numpy as np
import pytest
import torch
from holon.api import SamplingParams
from holon.core.engine import Engine, EngineConfig
from holon.core.memory import OutOfUnits
from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM

GREEDY = SamplingParams(temperature=0.0, max_tokens=12, ignore_eos=True)


def _hf_greedy(model, prompts, n):
    outs = []
    for p in prompts:
        ids = torch.tensor([p], device=model.device)
        gen = model.generate(ids, max_new_tokens=n, do_sample=False, attention_mask=torch.ones_like(ids))
        outs.append(gen[0, len(p):].tolist())
    return outs


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory) -> pathlib.Path:
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=False,
                      eos_token_id=None)
    path = tmp_path_factory.mktemp("tiny_qwen3")
    Qwen3ForCausalLM(cfg).save_pretrained(path)
    return path


def _tiny_engine(path, **kw):
    # Pages of 4 tokens; one page holds 2 (k/v) × 4 tokens × 2 heads × 16 dims × 4 B × 2 layers = 2 KiB.
    cfg = dict(model=str(path), device="cpu", dtype=torch.float32, page_size=4, max_seqs=3, max_step_tokens=10,
               max_model_len=40, state_memory=1 << 20, overlap=True)
    return Engine(EngineConfig(**cfg | kw))


@pytest.fixture(scope="module")
def tiny_case(tiny_checkpoint):
    torch.manual_seed(1)
    prompts = [torch.randint(0, 512, (n,)).tolist() for n in (3, 17, 8, 25)]
    hf = AutoModelForCausalLM.from_pretrained(tiny_checkpoint, dtype=torch.float32).eval()
    return prompts, _hf_greedy(hf, prompts, GREEDY.max_tokens)


def _released(engine):
    """No request holds anything; every unit is free or held only by the prefix cache."""
    index = engine.state.index
    ok = engine.state.held == {}
    for b in engine.state.bound:
        cached = index.cached(b.spec.name) if index else np.zeros(b.pool.capacity, dtype=bool)
        refs = b.pool.refcount(np.arange(b.pool.capacity))
        null = b.pool.null is not None                      # the graph padding unit, never handed out
        ok &= b.pool.num_free + int(cached.sum()) + null == b.pool.capacity and bool((refs[cached] == 1).all())
    return ok


def test_engine_matches_hf_greedy_on_cpu(tiny_checkpoint, tiny_case):
    # Few slots and a small step budget exercise paging, chunked prefill and continuous batching.
    prompts, expected = tiny_case
    engine = _tiny_engine(tiny_checkpoint)
    assert engine.generate(prompts, GREEDY) == expected
    assert engine.sched.preemptions == 0 and _released(engine)


def test_engine_preemption_matches_hf(tiny_checkpoint, tiny_case):
    # 12 pages (48 tokens) for requests needing 15–37 tokens each: requests are preempted and recomputed.
    prompts, expected = tiny_case
    engine = _tiny_engine(tiny_checkpoint, state_memory=12 * 2048)
    assert engine.ledger.pool("state.kv").capacity == 12
    assert engine.generate(prompts, GREEDY) == expected
    assert engine.sched.preemptions > 0 and _released(engine)


def test_engine_rejects_what_it_cannot_serve(tiny_checkpoint):
    engine = _tiny_engine(tiny_checkpoint)
    with pytest.raises(ValueError, match="max_model_len"):
        engine.add(list(range(40)))
    with pytest.raises(ValueError, match="empty"):
        engine.add([])
    engine.add([1, 2], SamplingParams(seed=2**64 - 1, max_tokens=1))   # any int seed is accepted (masked)
    engine.add([1, 2], SamplingParams(seed=-5, max_tokens=1))
    assert len(engine.generate([[3, 4]], SamplingParams(temperature=1.0, max_tokens=2))) == 1
    with pytest.raises(NotImplementedError, match="penalties"):
        engine.add([1, 2], SamplingParams(repetition_penalty=1.1))
    with pytest.raises(OutOfUnits, match="state.kv"):
        _tiny_engine(tiny_checkpoint, state_memory=9 * 2048)  # under one max-length request (10 pages)


def test_engine_stops_at_max_model_len(tiny_checkpoint):
    engine = _tiny_engine(tiny_checkpoint)
    out = engine.generate([[1] * 30], SamplingParams(temperature=0.0, ignore_eos=True))
    assert len(out[0]) == 10


def test_seeded_sampling_is_reproducible_under_preemption(tiny_checkpoint, tiny_case):
    prompts, _ = tiny_case
    params = SamplingParams(temperature=1.0, top_k=50, seed=7, max_tokens=12, ignore_eos=True)
    roomy = _tiny_engine(tiny_checkpoint).generate(prompts, params)
    tight = _tiny_engine(tiny_checkpoint, state_memory=12 * 2048)
    assert tight.generate(prompts, params) == roomy and tight.sched.preemptions > 0


SNAP = max(pathlib.Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*"), default=None)


@pytest.mark.skipif(SNAP is None or not torch.cuda.is_available(), reason="needs Qwen3-0.6B + GPU")
def test_engine_matches_hf_greedy_qwen3_0_6b():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SNAP)
    texts = ["The capital of France is", "def quicksort(xs):", "List three prime numbers:",
             "In 1969, astronauts first"]
    prompts = [tok(t)["input_ids"] for t in texts]
    hf = AutoModelForCausalLM.from_pretrained(SNAP, dtype=torch.float32).eval()
    expected = _hf_greedy(hf, prompts, GREEDY.max_tokens)
    del hf
    # fp32 on the CPU (the test-only reference kernels): exact against HF fp32.
    engine = Engine(EngineConfig(model=str(SNAP), device="cpu", dtype=torch.float32, page_size=16,
                                 max_step_tokens=24, max_model_len=256, state_memory=64 << 20))
    assert engine.generate(prompts, GREEDY) == expected


@pytest.mark.skipif(SNAP is None or not torch.cuda.is_available(), reason="needs Qwen3-0.6B + GPU")
def test_fast_kernels_agree_with_reference_and_are_run_identical_bf16():
    # bf16 rounding differs between kernels, so greedy paths may legitimately split late; exact checks are L0's.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SNAP)
    texts = ["The capital of France is", "def quicksort(xs):", "List three prime numbers:",
             "In 1969, astronauts first", "Water boils at", "The quick brown fox", "import numpy as np\n",
             "Photosynthesis converts"]
    prompts = [tok(t)["input_ids"] for t in texts]
    params = SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True)
    cfg = dict(model=str(SNAP), dtype=torch.bfloat16, page_size=16, max_step_tokens=64, max_model_len=2048)

    fast = Engine(EngineConfig(**cfg))
    # Pool sizes are measured: weights loaded, worst-case step profiled, the rest goes to KV pages.
    assert fast.ledger.pool("state.kv").capacity * 16 >= cfg["max_model_len"]
    assert {k.split("/")[-1]: v["provider"] for k, v in fast.ctx.kernels.items()} == {
        "attention.paged_prefill@1": "kernels.flashinfer", "attention.paged_decode@1": "kernels.triton_attn",
        "attention.paged_verify@1": "kernels.triton_attn", "rmsnorm@1": "kernels.flashinfer",
        "add_rmsnorm@1": "kernels.flashinfer", "rope@1": "kernels.flashinfer", "silu_mul@1": "kernels.flashinfer"}
    first, again = fast.generate(prompts, params), fast.generate(prompts, params)
    del fast
    # The oracles: the same engine on the CPU, where the test-only reference kernels serve every op.
    ref = Engine(EngineConfig(**cfg | {"device": "cpu", "state_memory": 512 << 20})).generate(prompts, params)

    assert first == again                                   # run-identical with the fast kernels
    agree = [next((i for i, (a, b) in enumerate(zip(x, y, strict=True)) if a != b), len(x))
             for x, y in zip(first, ref, strict=True)]
    assert all(n >= 1 for n in agree) and sum(agree) >= 0.75 * len(agree) * params.max_tokens, agree


def test_prompt_and_sampled_logprobs_match_hf(tiny_checkpoint, tiny_case):
    # Chunked prefill (budget 10) and preemption (12 pages) must not change or duplicate prompt logprobs.
    prompts, _ = tiny_case
    hf = AutoModelForCausalLM.from_pretrained(tiny_checkpoint, dtype=torch.float32).eval()
    for engine in (_tiny_engine(tiny_checkpoint), _tiny_engine(tiny_checkpoint, state_memory=12 * 2048)):
        rids = [engine.add(p, SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True, prompt_logprobs=3,
                                             logprobs=2)) for p in prompts]
        deltas = {r: [] for r in rids}
        while engine.has_work():
            for d in engine.step():
                deltas[d.rid].append(d)
        for p, r in zip(prompts, rids, strict=True):
            ds = deltas[r]
            full = p + [t for d in ds for t in d.token_ids]
            with torch.no_grad():
                lp = hf(torch.tensor([full])).logits[0].log_softmax(-1)
            plp = ds[0].prompt_logprobs
            assert all(d.prompt_logprobs is None for d in ds[1:]) and len(plp.token) == len(p) - 1
            want = lp[:len(p) - 1].gather(-1, torch.tensor(p[1:])[:, None])[:, 0]
            torch.testing.assert_close(torch.tensor(plp.token), want, atol=1e-4, rtol=1e-4)
            assert plp.top_ids == lp[:len(p) - 1].topk(3).indices.tolist()
            gen = torch.tensor([d.logprobs.token[0] for d in ds])
            want = lp[len(p) - 1: len(full) - 1].gather(-1, torch.tensor(full[len(p):])[:, None])[:, 0]
            torch.testing.assert_close(gen, want, atol=1e-4, rtol=1e-4)
            assert all(len(d.logprobs.top_ids[0]) == 2 for d in ds)


def test_n_sequences_per_request(tiny_checkpoint, tiny_case):
    prompts, expected = tiny_case
    engine = _tiny_engine(tiny_checkpoint, state_memory=12 * 2048)

    def run(params):
        rid = engine.add(prompts[1], params)
        seqs: dict[int, list[int]] = {}
        while engine.has_work():
            for d in engine.step():
                assert d.rid == rid
                seqs.setdefault(d.index, []).extend(d.token_ids)
        return [seqs[i] for i in sorted(seqs)]

    assert run(dataclasses.replace(GREEDY, n=3)) == [expected[1]] * 3
    sampled = SamplingParams(temperature=1.0, n=3, seed=5, max_tokens=12, ignore_eos=True)
    first = run(sampled)
    assert first == run(sampled) and len({tuple(s) for s in first}) > 1


def test_prefix_cache_hits_are_exact_across_turns(tiny_checkpoint, tiny_case):
    prompts, _ = tiny_case
    hf = AutoModelForCausalLM.from_pretrained(tiny_checkpoint, dtype=torch.float32).eval()
    engine = _tiny_engine(tiny_checkpoint, max_model_len=96)
    first = engine.generate([prompts[3]], GREEDY)[0]
    saved = engine.state.tokens_saved
    second_turn = prompts[3] + first + prompts[1]                 # a follow-up: history plus a new message
    assert engine.generate([second_turn], GREEDY) == _hf_greedy(hf, [second_turn], GREEDY.max_tokens)
    assert engine.state.tokens_saved - saved == (len(prompts[3] + first) - 1) // 4 * 4   # whole cached blocks
    assert engine.generate([prompts[3]], GREEDY) == [first] and _released(engine)


def test_prefix_cache_under_memory_pressure(tiny_checkpoint, tiny_case):
    # 12 pages: cached blocks must be evicted (and requests preempted) without changing any output.
    prompts, expected = tiny_case
    engine = _tiny_engine(tiny_checkpoint, state_memory=12 * 2048)
    for _ in range(2):
        assert engine.generate(prompts, GREEDY) == expected
    assert engine.state.tokens_saved > 0 and _released(engine)


def test_siblings_adopt_the_first_sequence_prompt(tiny_checkpoint, tiny_case):
    prompts, expected = tiny_case
    engine = _tiny_engine(tiny_checkpoint)
    rid = engine.add(prompts[3], dataclasses.replace(GREEDY, n=3))
    seqs: dict[int, list[int]] = {}
    while engine.has_work():
        for d in engine.step():
            seqs.setdefault(d.index, []).extend(d.token_ids)
    assert [seqs[i] for i in range(3)] == [expected[3]] * 3 and rid == "0"
    assert engine.state.tokens_saved == 2 * ((len(prompts[3]) - 1) // 4 * 4)


@pytest.mark.skipif(SNAP is None or not torch.cuda.is_available(), reason="needs Qwen3-0.6B + GPU")
def test_prefix_cache_on_vs_off_is_bounded_bf16():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SNAP)
    system = ("You are a careful assistant. Answer briefly and precisely, show units, and never invent facts. "
              "If a question is ambiguous, say what you assume. ") * 4
    questions = ["How far is the Moon?", "What is 17 times 23?", "Name a prime above 100.", "Who wrote Hamlet?"]
    prompts = [tok(system + q)["input_ids"] for q in questions]
    params = SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True)
    outs, saved = {}, {}
    for cache in (False, True):
        engine = Engine(EngineConfig(model=str(SNAP), dtype=torch.bfloat16, max_model_len=1024, prefix_cache=cache))
        outs[cache] = [engine.generate([p], params)[0] for p in prompts]       # one after another: later ones hit
        saved[cache] = engine.state.tokens_saved
        del engine
    assert saved[False] == 0 and saved[True] >= 3 * (len(tok(system)["input_ids"]) // 16 * 16)
    agree = [next((i for i, (a, b) in enumerate(zip(x, y, strict=True)) if a != b), len(x))
             for x, y in zip(outs[False], outs[True], strict=True)]
    assert all(n >= 1 for n in agree) and sum(agree) >= 0.75 * len(agree) * params.max_tokens, agree


def test_continuing_another_requests_cached_block_cannot_stall(tiny_checkpoint):
    # Regression: B recomputes the prefix A just cached, then publishes under it; A finishes. B must still fit.
    torch.manual_seed(2)
    shared = torch.randint(0, 512, (8,)).tolist()
    a, b = shared + [7], shared + torch.randint(0, 512, (31,)).tolist()
    engine = _tiny_engine(tiny_checkpoint, state_memory=11 * 2048, max_step_tokens=21)   # 10 pages + the null unit
    one = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    hf = AutoModelForCausalLM.from_pretrained(tiny_checkpoint, dtype=torch.float32).eval()
    ra, rb = engine.add(a, one), engine.add(b, one)
    out = {ra: [], rb: []}
    while engine.has_work():
        for d in engine.step():
            out[d.rid] += d.token_ids
    assert [out[ra], out[rb]] == _hf_greedy(hf, [a, b], 1) and _released(engine)


@pytest.mark.parametrize("state_memory", [1 << 20, 12 * 2048], ids=["roomy", "preempting"])
def test_overlap_matches_depth_0_with_stops_mid_batch(tiny_checkpoint, tiny_case, state_memory):
    # A stop token ends some sequences while their next decode row is already in flight: that row is discarded.
    prompts, expected = tiny_case
    stop = expected[1][3]                                    # a token sequence 1 emits early
    params = SamplingParams(temperature=0.0, max_tokens=12, ignore_eos=True, stop_token_ids=(stop,))
    outs = {}
    for overlap in (False, True):
        engine = _tiny_engine(tiny_checkpoint, state_memory=state_memory, overlap=overlap)
        outs[overlap] = engine.generate(prompts, params)
        assert _released(engine) and all(s.pending == 0 for s in engine.sched.waiting)
    assert outs[True] == outs[False]
    assert any(len(o) < 12 and o[-1] == stop for o in outs[True])


@pytest.mark.skipif(SNAP is None or not torch.cuda.is_available(), reason="needs Qwen3-0.6B + GPU")
def test_decode_launch_never_waits_for_the_device():
    from holon.core import platform

    engine = Engine(EngineConfig(model=str(SNAP), dtype=torch.bfloat16, max_model_len=4096, max_seqs=64))
    for i in range(64):
        engine.add([100 + i] * 2000, SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True))
    while engine.sched.waiting or engine._inflight is None or engine._inflight.rows[0][1] != 1:
        engine.step()                                        # through prefill, into steady decode
    for _ in range(3):
        prev, engine._inflight = engine._inflight, None
        scheduled = engine.sched.schedule()
        with platform.no_host_sync(engine.device):           # any implicit device->host sync raises here
            step = engine._launch(*scheduled)
        assert not step.out.done._event.query()              # a long step: the host got ahead of the device
        engine._commit(prev)
        engine._inflight = step
    while engine.has_work():
        engine.step()


def test_prompt_logprobs_are_complete_after_a_prefix_hit(tiny_checkpoint, tiny_case):
    # Regression: a request asking for prompt logprobs must not skip cached positions.
    prompts, _ = tiny_case
    engine = _tiny_engine(tiny_checkpoint)
    engine.generate([prompts[3]], GREEDY)                                   # caches the prompt's blocks
    rid = engine.add(prompts[3], SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1))
    deltas = [d for _ in iter(engine.has_work, False) for d in engine.step() if d.rid == rid]
    assert len(deltas[0].prompt_logprobs.token) == len(prompts[3]) - 1


@pytest.mark.skipif(SNAP is None or not torch.cuda.is_available(), reason="needs Qwen3-0.6B + GPU")
def test_cuda_graphs_are_bounded_and_run_identical_bf16():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SNAP)
    texts = ["The capital of France is", "def quicksort(xs):", "List three prime numbers:", "Water boils at",
             "In 1969, astronauts first"]                                    # 5 rows: padded to the 8-row bucket
    prompts = [tok(t)["input_ids"] for t in texts]
    params = SamplingParams(temperature=0.0, max_tokens=24, ignore_eos=True)
    sampled = SamplingParams(temperature=0.8, top_p=0.9, seed=3, max_tokens=24, ignore_eos=True)
    outs = {}
    for graphs in (False, True):
        engine = Engine(EngineConfig(model=str(SNAP), dtype=torch.bfloat16, max_model_len=1024, cuda_graphs=graphs))
        outs[graphs] = [engine.generate(prompts, params), engine.generate(prompts, sampled)]
        assert outs[graphs][0] == engine.generate(prompts, params)              # run-identical
        assert (engine.executor.graphs is not None and len(engine.executor.graphs) > 0) == graphs
        del engine
    for a, b in zip(outs[False], outs[True], strict=True):
        agree = [next((i for i, (x, y) in enumerate(zip(u, v, strict=True)) if x != y), len(u))
                 for u, v in zip(a, b, strict=True)]
        assert all(n >= 1 for n in agree) and sum(agree) >= 0.75 * len(agree) * 24, agree
