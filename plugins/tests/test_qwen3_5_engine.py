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


def test_without_checkpoint_units_nothing_is_reused(tiny):
    """A checkpoint pool may get no units (its budget did not fit): publishes find no room, so a turn recomputes."""
    _, path, prompts = tiny
    first = prompts[2]
    turn = first + [7, 8, 9] + prompts[3]
    params = SamplingParams(temperature=0.0, max_tokens=5, logprobs=1)
    cold = _run(_engine(path, prefix_cache=False), turn, params)
    engine = _engine(path, plugin_options={"recurrent.checkpoints": "0"})
    assert engine.ledger.pool("plugin.ssm_state.checkpoints").capacity == 0
    _run(engine, first, params)
    assert _run(engine, turn, params) == cold and engine.stats()["prefix_tokens_saved"] == 0


def _prefill_steps(engine, prompt) -> int:
    rid, steps = engine.add(prompt, SamplingParams(temperature=0.0, max_tokens=1)), 0
    while engine.has_work():
        steps += 1
        if any(d.rid == rid and d.token_ids for d in engine.step()):
            break
    while engine.has_work():
        engine.step()
    return steps


EVERY = {"recurrent.policy": "every:64"}


@pytest.mark.parametrize("kw", [{"prefix_cache": False, "plugin_options": EVERY},
                                {"plugin_options": EVERY | {"recurrent.checkpoints": "0"}}],
                         ids=["no-index", "no-units"])
def test_nothing_to_publish_means_no_cut(tiny, kw):
    _, path, prompts = tiny
    cutting = _engine(path, max_step_tokens=256, overlap=False, plugin_options=EVERY)
    assert _prefill_steps(cutting, prompts[2]) == 2                # 131, every 64: 64 (in-step) | 128 (in-step) + 3
    assert _prefill_steps(_engine(path, max_step_tokens=256, overlap=False, **kw), prompts[2]) == 1


def test_a_turn_prefills_in_one_step_and_its_in_step_checkpoint_resumes_exactly(tiny):
    _, path, prompts = tiny
    first = prompts[2]                                             # 131 tokens: the checkpoint at 128, in-step
    turn = first + [7, 8, 9] + prompts[3]
    params = SamplingParams(temperature=0.0, max_tokens=5, logprobs=1)
    engine = _engine(path, max_step_tokens=256, overlap=False)
    assert _prefill_steps(engine, first) == 1
    assert _run(engine, turn, params) == _run(_engine(path, prefix_cache=False), turn, params)
    assert engine.stats()["prefix_tokens_saved"] == 128


def test_more_sessions_than_checkpoint_units(tiny):
    """Three concurrent conversations share one checkpoint unit: publishes evict each other's payloads (an adopted
    one stays pinned until copied), some turns still hit, and every turn computes what recomputing does."""
    _, path, prompts = tiny
    g = torch.Generator().manual_seed(5)
    firsts = [torch.randint(0, 320, (n,), generator=g).tolist() for n in (70, 131, 100)]
    params = SamplingParams(temperature=0.0, max_tokens=4, logprobs=1)
    cold = _engine(path, prefix_cache=False)
    want = [[_run(cold, p, params) for p in (f, f + [7, 8] + prompts[3])] for f in firsts]
    hot = _engine(path, plugin_options={"recurrent.checkpoints": "1"})
    rids = {hot.add(f, params): (i, 0) for i, f in enumerate(firsts)}
    got: dict[tuple[int, int], tuple[list, list]] = {}
    while hot.has_work() or len(got) < 2 * len(firsts):
        for d in hot.step():
            key = rids[d.rid]
            toks, lps = got.setdefault(key, ([], []))
            toks += d.token_ids
            lps += d.logprobs.token if d.logprobs is not None else []
            if d.finish_reason is not None and key[1] == 0:          # the next turn resends the history
                i = key[0]
                rids[hot.add(firsts[i] + [7, 8] + prompts[3], params)] = (i, 1)
    for i in range(len(firsts)):            # batching moves f32 sums by ulps; a wrong state moves them far more
        for t in (0, 1):
            (toks, lps), (want_toks, want_lps) = got[(i, t)], want[i][t]
            assert toks == want_toks and max(abs(a - b) for a, b in zip(lps, want_lps, strict=True)) < 1e-5
    assert hot.stats()["prefix_tokens_saved"] > 0
    pool = hot.ledger.pool("plugin.ssm_state.checkpoints")                      # nothing leaked: the index holds
    held = [p for e in hot.state.index.entries.values() for p in e.payloads.values() if p.pool == pool.name]
    assert pool.capacity - pool.num_free == len(held)                           # every used unit


class _Seq:
    """What `Recurrent.cut` reads of a request."""

    def __init__(self, tokens: int, computed: int, prompt: int | None = None) -> None:
        self.tokens, self.num_computed, self.prompt_len = [0] * tokens, computed, prompt or tokens


def _kind(policy: str):
    from holon.api import Slot, StateSpec
    from holon_std.state.recurrent.kind import Recurrent

    class _Ctx:
        options = {"plugins": {"recurrent.policy": policy}, "max_seqs": 2, "page_size": 4}

        def pool(self, name, **kw):
            return type("P", (), {"capacity": 2, "name": f"plugin.{name}", "num_free": 2})()
    spec = StateSpec("ssm_state", "recurrent@1", Slot(8), (0,), "t", alignment=64, in_step=64)
    return Recurrent(spec, type("P", (), {"name": "state.ssm_state"})(), _Ctx())


def test_the_kinds_cuts_follow_the_in_step_grid():
    prompt_end, every = _kind("prompt_end"), _kind("every:64")
    assert prompt_end.cut(_Seq(131, 0), 0, 131) == 131                 # 128 in-step: no cut
    assert prompt_end.cut(_Seq(131, 10), 10, 121) == 118               # off the grid from 10: cut at 128
    assert every.cut(_Seq(300, 0), 0, 300) == 64                       # 64 in-step; 128 is the next step's
    assert every.cut(_Seq(300, 64), 64, 236) == 64
    assert prompt_end.cut(_Seq(400, 0), 0, 100) == 64                  # budget-truncated: rounded to the grid
    assert prompt_end.cut(_Seq(400, 0), 0, 30) == 30                   # ... never to nothing
    # every 1024: the next position (2048) lies past the prompt, so its last step is not rounded
    assert _kind("every:1024").cut(_Seq(1500, 1024), 1024, 476) == 476


def test_every_n_keeps_both_positions_of_one_step(tiny):
    _, path, prompts = tiny
    engine = _engine(path, max_step_tokens=256, plugin_options=EVERY)
    _run(engine, prompts[2], SamplingParams(temperature=0.0, max_tokens=1))
    held = [e for e in engine.state.index.entries.values() if "ssm_state" in e.payloads]
    assert len(held) == 2                                              # the blocks ending at 64 and 128


def test_a_conv_window_one_token_early_changes_a_hit(tiny, monkeypatch):
    from holon_std.mixers.gdn import units

    _, path, prompts = tiny
    first, params = prompts[2], SamplingParams(temperature=0.0, max_tokens=5, logprobs=1)
    turn = first + [7, 8, 9] + prompts[3]
    cold = _run(_engine(path, prefix_cache=False), turn, params)
    plan_bounds = units.Planner.plan_bounds

    def early(self, batch, pages):
        meta = plan_bounds(self, batch, pages)
        if meta.conv is not None:
            meta.conv = (meta.conv[0], meta.conv[1] - 1, meta.conv[2])
        return meta
    monkeypatch.setattr(units.Planner, "plan_bounds", early)
    engine = _engine(path, max_step_tokens=256)
    _run(engine, first, params)
    assert _run(engine, turn, params) != cold and engine.stats()["prefix_tokens_saved"] == 128


def test_a_full_payload_pool_on_one_half_means_no_hit(tiny):
    _, path, prompts = tiny
    first, params = prompts[2], SamplingParams(temperature=0.0, max_tokens=5, logprobs=1)
    turn = first + [7, 8, 9] + prompts[3]
    engine = _engine(path, max_step_tokens=256, plugin_options={"recurrent.checkpoints": "1"})
    engine.ledger.pool("plugin.ssm_state.checkpoints").alloc(1)        # the delta-rule half cannot publish
    _run(engine, first, params)
    assert _run(engine, turn, params) == _run(_engine(path, prefix_cache=False), turn, params)
    assert engine.stats()["prefix_tokens_saved"] == 0
