"""The engine on a tiny Qwen3.5/3.8-architecture checkpoint, on the CPU with the reference kernels: recurrent state in
Slot pools across chunked prefill and decode, against transformers; and prefix reuse through checkpoints — a request
that adopts a checkpoint computes exactly what it computes without the cache (the reference recurrence is exact in
f32, so resuming is bitwise)."""
import dataclasses

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


def test_without_checkpoint_units_nothing_is_reused(tiny, capsys):
    """A checkpoint pool with no units recomputes a turn: `recurrent.checkpoints` 0, or left unset (best effort) with a
    state budget that cannot fit the default beside the paged floor (the ledger falls back to the minimum, 0), which
    says so once. Set explicitly, the units are required: the same budget refuses the start."""
    _, path, prompts = tiny
    first = prompts[2]
    turn = first + [7, 8, 9] + prompts[3]
    params = SamplingParams(temperature=0.0, max_tokens=5, logprobs=1)
    cold = _run(_engine(path, prefix_cache=False), turn, params)
    probe = _engine(path).ledger.pools.values()                        # the default: max_seqs units
    fits = sum((q.capacity if q.name.endswith(".checkpoints") else q.min_units) * q.unit_bytes for q in probe)
    with pytest.raises(RuntimeError, match="memory budget"):
        _engine(path, plugin_options={"recurrent.checkpoints": "3"}, state_memory=fits - 1)
    for options, warned in (({"recurrent.checkpoints": "0"}, False), ({}, True)):
        capsys.readouterr()
        engine = _engine(path, plugin_options=options, state_memory=fits - 1)
        assert engine.ledger.pool("plugin.ssm_state.checkpoints").capacity == 0
        _run(engine, first, params)
        assert _run(engine, turn, params) == cold and engine.stats()["prefix_tokens_saved"] == 0
        assert engine.stats()["dropped_publishes"] == 0                # nothing cut for, nothing published
        assert capsys.readouterr().err.count("got 0 units") == (2 if warned else 0)    # once per spec


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
    assert _prefill_steps(cutting, prompts[2]) == 2                # 131, every 64: 64 (step end) | 128 (in-step) + 3
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


MASK = "toy.even_after@1"


@pytest.mark.parametrize("memory", [8 << 20, 300 << 10], ids=["roomy", "preempting"])
def test_mixed_steps_compute_what_todays_steps_do(tiny, memory):
    """docs/m07-perf.md §3.1: with `chunk_with_decode` the same requests — chunked prompts beside decoding ones, two
    constrained by the toy processor, one sampled, a second pass hitting the prefix cache's checkpoints (roomy), or
    preempted — give today's tokens, and logprobs within batching's ulps. Mixed steps run GQA's decode rows
    through the decode kernel (the reference prefill does not declare `serves_decode_rows`) or, flipped, through the
    prefill kernel, and GDN's per row kind with their in-step checkpoints re-based."""
    _, path, prompts = tiny
    g = torch.Generator().manual_seed(7)
    prompts = prompts + [torch.randint(0, 320, (n,), generator=g).tolist() for n in (90, 12)]
    greedy = SamplingParams(temperature=0.0, max_tokens=12, logprobs=1, ignore_eos=True)
    requests = [(p, dataclasses.replace(greedy, ext={MASK: {"marker": p[-1]}}) if i in (1, 4) else
                 dataclasses.replace(greedy, temperature=1.0, seed=3) if i == 2 else greedy)
                for i, p in enumerate(prompts)]

    def served(engine):
        rids = [engine.add(p, params) for p, params in requests]
        out = {r: ([], []) for r in rids}
        while engine.has_work():
            for d in engine.step():
                out[d.rid][0].extend(d.token_ids)
                out[d.rid][1].extend(d.logprobs.token)
        return [out[r] for r in rids]

    kw = {"state_memory": memory, "max_model_len": 160, "slots": {"logits_processor": "testing.toy_mask"}}
    today = served(_engine(path, **kw))
    assert all(t % 2 == 0 for i in (1, 4) for t in today[i][0])                 # the constraint bites
    for cap, one_call in ((8, False), (72, True)):            # 72: a chunk may hold an in-step checkpoint
        engine = _engine(path, chunk_with_decode=cap, max_step_tokens=96, **kw)
        engine.executor.planners["kv"].prefill_mixes = one_call
        for _ in range(2):
            for (toks, lps), (want, want_lps) in zip(served(engine), today, strict=True):
                assert toks == want and max(abs(a - b) for a, b in zip(lps, want_lps, strict=True)) < 1e-5
        stats = engine.stats()
        assert stats["mixed_steps"] > 0
        assert stats["preemptions"] > 0 if memory < 8 << 20 else stats["prefix_tokens_saved"] > 0


class _Seq:
    """What `Recurrent.cut` reads of a request."""

    def __init__(self, tokens: int, computed: int, prompt: int | None = None) -> None:
        self.tokens, self.num_computed, self.prompt_len = [0] * tokens, computed, prompt or tokens


def _kind(policy: str, units: int = 2, unit=None):
    from holon.api import Slot, StateSpec
    from holon_std.state.recurrent.kind import Recurrent

    class _Ctx:
        options = {"plugins": {"recurrent.policy": policy}, "max_seqs": 2, "page_size": 4}

        def pool(self, name, **kw):
            return type("P", (), {"capacity": units, "name": f"plugin.{name}", "num_free": units})()

        def unit(self, ref, spec):
            return unit
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
    assert _kind("prompt_end", units=0).cut(_Seq(400, 0), 0, 100) == 100   # nothing to publish: no rounding
    # every 1024: the next position (2048) lies past the prompt, so its last step is not rounded
    assert _kind("every:1024").cut(_Seq(1500, 1024), 1024, 476) == 476


class _EveryBlock:
    """A policy unit: a tentative checkpoint at every 64-token position."""

    def next(self, view):
        return (view.num_computed // 64 + 1) * 64, True


@pytest.mark.parametrize("keep", [0, 3])
def test_a_policy_units_keep_is_the_kinds(keep):
    unit = _EveryBlock()
    if keep:
        unit.keep = keep
    kind = _kind("prompt_end,toy.every_block@1", unit=unit)
    assert kind.keep_tentative == keep                                # the engine refuses it above 0 without a tier
    assert kind._next(_Seq(300, 0)) == (64, keep > 0)                  # without `keep`, nothing keeps it: not tentative


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
