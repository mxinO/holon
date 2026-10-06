"""Engine end to end: paged KV, chunked prefill and continuous batching reproduce HF greedy generation."""
import pathlib

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
               max_model_len=40, state_memory=1 << 20)
    return Engine(EngineConfig(**cfg | kw))


@pytest.fixture(scope="module")
def tiny_case(tiny_checkpoint):
    torch.manual_seed(1)
    prompts = [torch.randint(0, 512, (n,)).tolist() for n in (3, 17, 8, 25)]
    hf = AutoModelForCausalLM.from_pretrained(tiny_checkpoint, dtype=torch.float32).eval()
    return prompts, _hf_greedy(hf, prompts, GREEDY.max_tokens)


def _released(engine):
    return engine.state.held == {} and all(p.num_free == p.capacity for p in engine.ledger.pools.values())


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
    with pytest.raises(NotImplementedError, match="n > 1"):
        engine.add([1, 2], SamplingParams(n=2))
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
    hf = AutoModelForCausalLM.from_pretrained(SNAP, dtype=torch.float32).to("cuda").eval()
    expected = _hf_greedy(hf, prompts, GREEDY.max_tokens)
    del hf
    # Pool sizes are measured: weights loaded, worst-case step profiled, the rest goes to KV pages.
    engine = Engine(EngineConfig(model=str(SNAP), dtype=torch.float32, page_size=16, max_step_tokens=24,
                                 max_model_len=4096))
    assert engine.ledger.pool("state.kv").capacity * 16 >= 4096
    assert engine.generate(prompts, GREEDY) == expected
