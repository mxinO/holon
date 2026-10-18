"""Milestone K's core seams in an engine (docs/v1-kv-tiers.md §4: KS1, KS2, KS4), proven with the toy tier
(`testing.toy_tier`) on a tiny model on the CPU before std's tier uses them: a tight device pool demotes blocks to the
tier and promotes them back, a conversation recomputing what only the tier holds is promoted into place, and the tier's
copies run on a deferred stream (`holon_testing.deferred`) that lands them 0–2 launches late unless a launch waits.

- Tier invariance: sessions interleaved with a filler on the tight engine give the outputs and `cached_tokens` of the
  same sessions run back to back on a roomy engine without a tier.
- Run-identity: concurrent requests make the same steps and outputs under three deferral seeds and a load delay — no
  decision reads when a copy lands —, through trimmed promotions, blocks promoted into place, aborts of load-gated
  requests (their loads abandoned or reported unread) and the tier's own drops.
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
BLOCK = 16                                              # the tiny model's prefix-index block (one page)


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


def _ask(engine: Engine, prompt: list[int]) -> tuple[list[int], int | None]:
    """One request answered greedily on its own: (output, cached_tokens)."""
    rid, tokens, cached = engine.add(prompt, GREEDY), [], None
    while engine.has_work():
        for d in engine.step():
            if d.rid == rid:
                tokens += d.token_ids
                cached = d.cached_tokens if d.cached_tokens is not None else cached
    return tokens, cached


def _turns(engine: Engine, sessions: dict, prompts: dict, s: int, rng: random.Random) -> dict:
    """Session s takes a turn: first its previous prompt again, cut to whole blocks (a regenerate: its hit leaves its
    last block to recompute, a block the tier may hold alone by then), then its history plus new tokens; (kind,
    session, turn) -> (output, cached_tokens)."""
    out, turn = {}, sum(1 for k in prompts if k[0] == s)
    if turn:
        last = prompts[s, turn - 1]
        out["regenerate", s, turn] = _ask(engine, last[:len(last) // BLOCK * BLOCK])
    prompts[s, turn] = sessions[s] + [rng.randrange(128) for _ in range(rng.randrange(5, 30))]
    tokens, cached = out["turn", s, turn] = _ask(engine, prompts[s, turn])
    sessions[s] = prompts[s, turn] + tokens
    return out


@pytest.mark.parametrize("seed", [0, 1])
def test_tier_invariance_with_demotions_promotions_and_late_copies(tiny, monkeypatch, seed):
    deferred = Deferred(seed, most=2)
    monkeypatch.setattr(platform, "Stream", lambda device: deferred.stream())
    starts = {s: [random.Random(seed * 10 + s).randrange(128) for _ in range(70)] for s in range(4)}
    reference = {}
    roomy = _engine(tiny, pages=256)
    for s in starts:                                      # back to back: every return hits on the device
        sessions, prompts, rng = {s: list(starts[s])}, {}, random.Random(s)
        for _ in range(4):
            reference |= _turns(roomy, sessions, prompts, s, rng)
    roomy.close()
    tight = _engine(tiny, pages=20, tier={"delay": 1, "seed": seed})
    promoted = _count_promote_into_place(tight)
    rngs, sessions, prompts, tested = {s: random.Random(s) for s in starts}, {s: list(starts[s]) for s in starts}, {}, {}
    for _ in range(4):                                    # interleaved: each return comes through the tier
        for s in starts:
            tested |= _turns(tight, sessions, prompts, s, rngs[s])
    tier = tight.tier
    assert tested == reference
    assert tier.counts["loaded_blocks"] >= 8 and promoted() > 0     # promotions, and recomputed blocks into place
    assert tight.stats()["testing.toy_tier.loads"] == tier.counts["loads"]
    tight.close()


def _count_promote_into_place(engine: Engine):
    """Count the blocks `publish` promotes into place: the count, as a function."""
    publish, promote, n = engine.state.publish, engine.state.index.promote, [0]

    def publishing(seq):
        engine.state.index.promote = lambda entries, *a: (n.__setitem__(0, n[0] + len(entries)), promote(entries,
                                                                                                         *a))[1]
        publish(seq)
        engine.state.index.promote = promote
    engine.state.publish = publishing
    return lambda: n[0]


def _closed_loop(engine: Engine, conversations: int, turns: int, aborts: int = 0, start: int = 20
                 ) -> tuple[list, dict]:
    """`conversations` clients, each sending its next turn once its previous answer is done, all at once: the steps
    (slots and tokens per row) and every answer. The first `aborts` requests found load-gated are aborted and sent
    again. Counts the blocks promoted into place in `engine.promoted()`."""
    steps, schedule = [], engine.sched.schedule
    engine.promoted = _count_promote_into_place(engine)

    def recorded():
        out = schedule()
        if out is not None:
            steps.append((out[0].slots.tolist(), out[0].n_query.tolist()))
        return out
    engine.sched.schedule = recorded
    rng, history, answers, live = random.Random(7), {}, {}, {}
    for c in range(conversations):
        history[c] = [rng.randrange(128) for _ in range(rng.randrange(start, start + 40))]
        live[engine.add(history[c], GREEDY)], answers[c] = c, []
    while engine.has_work():
        gated = sorted(rid for rid, seqs in engine._live.items() if any(s.slot in engine.state.due for s in seqs))
        if aborts and gated:                          # abort a request waiting for its loads, and send it again
            aborts -= 1
            c = live[gated[0]]
            engine.abort(gated[0])
            live[engine.add(history[c], GREEDY)] = c
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
        engine = _engine(tiny, pages=20, tier={"delay": delay, "drops": 0.3}, max_seqs=4)   # and its own drops
        runs.append(_closed_loop(engine, conversations=8, turns=3, aborts=3, start=70))
        counts = engine.tier.counts
        assert counts["dropped"] and not engine.tier.untold()
        if delay:            # promotions trimmed short of units, recomputed blocks promoted into place; loads of
            assert min(engine.promoted(), counts["abandoned"], counts["waiterless"]) > 0    # aborted requests
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
