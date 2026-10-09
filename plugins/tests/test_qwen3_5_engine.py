"""The engine on a tiny Qwen3.5/3.8-architecture checkpoint, on the CPU with the reference kernels: recurrent state in
Slot pools across chunked prefill and decode, against transformers; and prefix reuse through checkpoints — a request
that adopts a checkpoint computes exactly what it computes without the cache (the reference recurrence is exact in
f32, so resuming is bitwise)."""
import pytest
import torch
from holon.api import SamplingParams
from holon.core.engine import Engine, EngineConfig
from holon_testing.qwen3_5_files import save, tiny_hf


@pytest.fixture(scope="module", params=[4, 6], ids=["ratio2", "ratio3"])
def tiny(request, tmp_path_factory):
    model = tiny_hf(request.param, seed=1)
    path = save(model, tmp_path_factory.mktemp(f"tiny_qwen3_5_{request.param}"))
    g = torch.Generator().manual_seed(2)
    prompts = [torch.randint(0, 320, (n,), generator=g).tolist() for n in (5, 70, 131, 30)]
    return model, path, prompts


def _engine(path, **kw) -> Engine:
    cfg = dict(model=str(path), device="cpu", dtype=torch.float32, page_size=4, max_seqs=3, max_step_tokens=48,
               max_model_len=256, state_memory=8 << 20)
    return Engine(EngineConfig(**cfg | kw))


def test_chunked_prefill_and_decode_match_transformers(tiny):
    model, path, prompts = tiny
    engine = _engine(path, prefix_cache=False)
    rids = [engine.add(p, SamplingParams(temperature=0.0, max_tokens=6, prompt_logprobs=1)) for p in prompts]
    lps, out = {}, {r: [] for r in rids}
    while engine.has_work():
        for d in engine.step():
            out[d.rid] += d.token_ids
            if d.prompt_logprobs is not None:
                lps[d.rid] = d.prompt_logprobs
    for p, r in zip(prompts, rids, strict=True):
        with torch.no_grad():
            lp = model(torch.tensor([p])).logits[0, :-1].log_softmax(-1)
            want = model.generate(torch.tensor([p]), max_new_tokens=6, do_sample=False)[0, len(p):].tolist()
        torch.testing.assert_close(torch.tensor(lps[r].token), lp.gather(-1, torch.tensor(p[1:])[:, None])[:, 0],
                                   atol=1e-4, rtol=1e-4)
        assert out[r] == want


def _run(engine, prompt, params):
    """Tokens and their sampled logprobs (sensitive to the recurrent state, unlike a tiny model's greedy tokens)."""
    rid, tokens, lps = engine.add(prompt, params), [], []
    while engine.has_work():
        for d in engine.step():
            if d.rid == rid:
                tokens += d.token_ids
                lps += d.logprobs.token if d.logprobs is not None else []
    return tokens, lps


@pytest.mark.parametrize("policy", ["prompt_end", "every:64"])
def test_a_checkpoint_hit_computes_what_recomputing_does(tiny, policy):
    _, path, prompts = tiny
    first = prompts[2]                                           # 131 tokens: a checkpoint at 128 (or 64, 128)
    turn = first + [7, 8, 9] + prompts[3]                        # the next turn resends it
    params = SamplingParams(temperature=0.0, max_tokens=5, logprobs=1)
    cold = _run(_engine(path, prefix_cache=False), turn, params)
    engine = _engine(path, plugin_options={"recurrent.policy": policy})
    _run(engine, first, params)
    assert engine.stats()["cached_blocks"] >= 2
    hot = _run(engine, turn, params)
    assert engine.stats()["prefix_tokens_saved"] == 128 and hot == cold
