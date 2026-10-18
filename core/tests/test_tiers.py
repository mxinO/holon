"""Milestone K's core seams in an engine (docs/v1-kv-tiers.md §4: KS1, KS2, KS4), proven with the toy tier
(`testing.toy_tier`) on a tiny model on the CPU before std's tier uses them: a tight device pool demotes blocks to the
tier and promotes them back, a conversation recomputing what only the tier holds is promoted into place, and the tier's
copies run on a deferred stream (`holon_testing.deferred`) that lands them 0–2 launches late unless a launch waits.

- Tier invariance: sessions interleaved with a filler on the tight engine give the outputs and `cached_tokens` of the
  same sessions run back to back on a roomy engine without a tier.
- Run-identity: concurrent requests make the same steps and outputs under three deferral seeds and a load delay — no
  decision reads when a copy lands.
- Restart: a tier persisting at close restores the index, so a resumed conversation hits as it would without the
  restart; under another identity nothing is restored; an option declared `planned = false` leaves plans unchanged."""
import dataclasses
import random

import pytest
import torch
from holon.api import SamplingParams
from holon.core import plans, platform
from holon.core.engine import Engine, EngineConfig
from holon_testing.deferred import Deferred

GREEDY = SamplingParams(temperature=0.0, max_tokens=6, ignore_eos=True)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("tiny_qwen3")
    Qwen3ForCausalLM(Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                 num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                                 eos_token_id=None)).save_pretrained(path)
    return path


def _engine(tiny, pages: int, tier: dict | None = None, max_seqs: int = 1) -> Engine:
    """`pages` units of 16 tokens in the KV pool (2 layers × 2 heads × 16 dims × K, V × fp32 = 512 B per token)."""
    options = {f"toy_tier.{k}": v for k, v in (tier or {}).items()}
    return Engine(EngineConfig(str(tiny), device="cpu", dtype=torch.float32, page_size=16, max_seqs=max_seqs,
                               max_step_tokens=64, max_model_len=256, state_memory=pages * 16 * 512,
                               slots={"tier": "testing.toy_tier"} if tier is not None else {},
                               host_memory=64 << 20 if tier is not None else 0, plugin_options=options))


def _turns(engine: Engine, sessions: dict[int, list[int]], order: list[int], rng: random.Random) -> dict:
    """Each session in `order` takes a turn: its history plus new tokens, answered greedily; (session, turn) ->
    (output, cached_tokens)."""
    out, turns = {}, dict.fromkeys(sessions, 0)
    for s in order:
        prompt = sessions[s] + [rng.randrange(128) for _ in range(rng.randrange(5, 30))]
        rid = engine.add(prompt, GREEDY)
        tokens, cached = [], None
        while engine.has_work():
            for d in engine.step():
                if d.rid == rid:
                    tokens += d.token_ids
                    cached = d.cached_tokens if d.cached_tokens is not None else cached
        sessions[s] = prompt + tokens
        out[s, turns[s]] = (tokens, cached)
        turns[s] += 1
    return out


@pytest.mark.parametrize("seed", [0, 1])
def test_tier_invariance_with_demotions_promotions_and_late_copies(tiny, monkeypatch, seed):
    deferred = Deferred(seed, most=2)
    monkeypatch.setattr(platform, "Stream", lambda device: deferred.stream())
    starts = {s: [random.Random(seed * 10 + s).randrange(128) for _ in range(40)] for s in range(4)}
    reference = {}
    roomy = _engine(tiny, pages=256)
    for s in starts:                                      # back to back: every return hits on the device
        reference |= _turns(roomy, {s: list(starts[s])}, [s] * 4, random.Random(s))
    roomy.close()
    tight = _engine(tiny, pages=20, tier={"delay": 1, "seed": seed})
    rngs = {s: random.Random(s) for s in starts}
    sessions, tested = {s: list(starts[s]) for s in starts}, {}
    for turn in range(4):                                 # interleaved: each return comes through the tier
        for s in starts:
            out = _turns(tight, sessions, [s], rngs[s])
            tested[s, turn] = out[s, 0]
    tier = tight.tier
    assert tested == reference
    assert tier.counts["loaded_blocks"] >= 8 and any(c[0] == "load" for c in tier.calls)
    assert tight.stats()["testing.toy_tier.loads"] == tier.counts["loads"]
    tight.close()


def _closed_loop(engine: Engine, conversations: int, turns: int) -> tuple[list, dict]:
    """`conversations` clients, each sending its next turn once its previous answer is done, all at once: the steps
    (slots and tokens per row) and every answer."""
    steps, schedule = [], engine.sched.schedule

    def recorded():
        out = schedule()
        if out is not None:
            steps.append((out[0].slots.tolist(), out[0].n_query.tolist()))
        return out
    engine.sched.schedule = recorded
    rng, history, answers, live = random.Random(7), {}, {}, {}
    for c in range(conversations):
        history[c] = [rng.randrange(128) for _ in range(rng.randrange(20, 60))]
        live[engine.add(history[c], GREEDY)], answers[c] = c, []
    while engine.has_work():
        for d in engine.step():
            c = live[d.rid]
            answers[c].append(d.token_ids) if not answers[c] or answers[c][-1] is None else answers[c][-1].extend(
                d.token_ids)
            if d.finish_reason:
                history[c] += answers[c][-1]
                if len(answers[c]) < turns:
                    history[c] += [rng.randrange(128) for _ in range(rng.randrange(5, 20))]
                    live[engine.add(history[c], GREEDY)] = c
                    answers[c].append(None)
    return steps, answers


def test_steps_and_outputs_never_depend_on_when_copies_land(tiny, monkeypatch):
    runs = []
    for seed, delay in ((0, 2), (1, 2), (2, 2), (0, 0)):
        deferred = Deferred(seed, most=2)
        monkeypatch.setattr(platform, "Stream", lambda device, d=deferred: d.stream())
        engine = _engine(tiny, pages=20, tier={"delay": delay, "drops": 0.3}, max_seqs=2)   # and its own drops
        runs.append(_closed_loop(engine, conversations=6, turns=3))
        counts = engine.tier.counts
        assert counts["loaded_blocks"] >= 4 and counts["dropped"] >= 4 and not engine.tier.untold()
        engine.close()
    assert runs[0] == runs[1] == runs[2]
    assert runs[3][0] != runs[0][0]                     # the gate (a due time two launches out) moved steps


def test_a_restart_restores_the_index_under_the_same_identity_only(tiny, tmp_path):
    path = str(tmp_path / "tier.pt")
    rng = random.Random(3)
    history = [rng.randrange(128) for _ in range(50)]
    turns = [[rng.randrange(128) for _ in range(9)] for _ in range(3)]

    def run(engine, sent, last):
        out = engine.generate([sent + turns[last]], GREEDY)[0]
        return sent + turns[last] + out, out, engine.stats()["prefix_tokens_saved"]

    kept = _engine(tiny, pages=20, tier={"path": str(tmp_path / "kept.pt")})
    a = run(kept, history, 0)
    b = run(kept, a[0], 1)
    want = run(kept, b[0], 2)
    kept.close()
    first = _engine(tiny, pages=20, tier={"path": path})
    b2 = run(first, run(first, history, 0)[0], 1)
    first.close()
    restarted = _engine(tiny, pages=20, tier={"path": path})
    assert restarted.stats()["prefix_tokens_saved"] == 0 and restarted.state.index.lower
    got = run(restarted, b2[0], 2)
    assert got[1] == want[1] and got[2] == want[2] - b[2] > 0       # the same answer, hitting what `kept` hit
    restarted.cfg.plugin_options["toy_tier.path"] = "elsewhere"     # planned = false: the plan does not move
    assert plans.plan_of(restarted).plan_id == plans.plan_of(_with(restarted, path)).plan_id
    restarted.close()
    other = _engine(tiny, pages=20, tier={"path": path, "scale": 5})      # the tier's own option: same identity
    assert other.state.index.lower
    other.close()
    third = Engine(dataclasses.replace(other.cfg, plugin_options=other.cfg.plugin_options | {"tests.knob": "1"}))
    assert not third.state.index.lower                  # another identity: nothing restored
    third.close()


def _with(engine, path):
    engine.cfg.plugin_options["toy_tier.path"] = path
    return engine
