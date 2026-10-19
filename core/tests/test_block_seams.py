"""FN's core seams (docs/v1-flash-next.md §4), proven with toys (`holon_testing.toy_streams`) on a tiny Qwen3 before a
model uses them:

- **C1, the residual width** (`ModelSpec.residual`): the trunk carries two float32 streams ([T, 2H]); the final norm
  collapses them. A step's trunk, `forward`, prompt logprobs, a tap and the drafter see [T, 2H] float32, eager on the
  CPU and from CUDA graphs on the GPU, where the startup profile checks the declared width (a wrong one fails there).
- **C2, state declared by the block**, through **C3** (`BuildCtx.unit_class`): the block forwards a nested Ref's
  state hooks — a Slot running sum of the token ids at layer 1 (the `recurrent` kind, alignment 64, `in_step` 64). A
  200-token prompt in steps of at most 100 tokens is cut at 64 and 128 (the budget rounded to the grid) and checkpoints
  at 192 in-step; a second request with that prefix hits at 192 and adopts the payload; speculative verify steps
  (prompt lookup) rewind it to their accepted tokens. Greedy output equals a model-only build's (the sum moves the
  logits, so a wrong state shows); the GPU run captures its decode and verify graphs.
- The planner rule: on the block's path a Slot group without a planner is refused; a mixer's group without one keeps
  today's default planner and says once that its steps run without CUDA graphs.
"""
import dataclasses
import re
import types

import pytest
import torch
from holon.api import Ref, Residual, SamplingParams, Slot, StateSpec
from holon.core import assembly
from holon.core.build import BuildContext, StepContext, build_model, load_weights
from holon.core.engine import Engine, EngineConfig
from holon.core.plugins import PluginError
from holon_testing.toy_streams import plugin as toy

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
GREEDY = SamplingParams(temperature=0.0, max_tokens=12, ignore_eos=True)
SPEC = dict(max_draft_tokens=3, plugin_options={"spec.k": "3", "spec.fallback": 2},
            slots={"drafter": "testing.toy_streams", "draft_policy": "std.spec.fixed_k", "verifier": "std.spec.greedy"})


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("tiny_qwen3_streams")
    Qwen3ForCausalLM(Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                 num_attention_heads=2, num_key_value_heads=1, head_dim=64, eos_token_id=None,
                                 tie_word_embeddings=False)).save_pretrained(path)
    return path


@pytest.fixture
def streams(monkeypatch):
    """The Qwen3 recipe on two float32 streams with the running sum in its block; `streams.block` / `streams.width`
    pick the block (planned or not) and the declared width (in hidden sizes)."""
    from holon_plugins.models.qwen3 import plugin as qwen3

    recipe, knobs = qwen3.Plugin.recipe, types.SimpleNamespace(block="toy.planned_block@1", width=2)

    def move(target: str) -> str:                   # the toys wrap the units the rules fill
        return re.sub(r"^(embed|norm|layers\.\{i\})\.", r"\1.inner.", target)

    def wide(c, source):
        spec = recipe(c, source)
        rules = tuple(dataclasses.replace(r, target=move(r.target)) for r in spec.weights.rules)
        return dataclasses.replace(
            spec, embed=Ref("toy.wide_embed@1", inner=spec.embed), norm=Ref("toy.wide_norm@1", inner=spec.norm),
            block=Ref(knobs.block, inner=spec.block, inner_norm=spec.norm, pre=Ref("toy.running_sum@1"), pre_at=(1,)),
            residual=Residual(knobs.width * spec.hidden, "float32"),
            weights=dataclasses.replace(spec.weights, rules=rules))

    monkeypatch.setattr(qwen3.Plugin, "recipe", staticmethod(wide))
    return knobs


def _prompts() -> tuple[list[int], list[int]]:
    g = torch.Generator().manual_seed(1)
    a = torch.randint(1, 128, (50,), generator=g).tolist() * 4                # repeats: prompt lookup drafts
    return a, a + torch.randint(1, 128, (20,), generator=g).tolist()          # b: a's 200 tokens, then its own


def _reference(engine: Engine, prompts: list[list[int]], device: str) -> list[list[int]]:
    """Greedy continuations from a model-only build of the same spec (no state: whole sequences from zero)."""
    ctx = BuildContext(engine.plugins, torch.device(device), engine.cfg.dtype)
    model = build_model(engine.spec, ctx)
    load_weights(model, ctx, engine.source.tensors())
    out = []
    with torch.no_grad():
        for p in prompts:
            ids = list(p)
            for _ in range(GREEDY.max_tokens):
                ids.append(int(model(StepContext.whole_sequences([ids], device))[-1].argmax()))
            out.append(ids[len(p):])
    return out


def _run(tiny, device: str, monkeypatch) -> tuple[Engine, list, list[list[int]], dict]:
    """One engine with prefix caching and speculation: a, then b; the toy planner's in-step publishes and row
    lengths recorded."""
    plans, publishes = [], []
    plan_bounds = toy.SumPlanner.plan_bounds

    def record(self, batch, pages):
        plans.append(batch.plan.n_query.tolist())
        publishes.extend(p.offset + int(batch.plan.start_upper[p.row]) for p in batch.plan.publishes)
        return plan_bounds(self, batch, pages)

    monkeypatch.setattr(toy.SumPlanner, "plan_bounds", record)
    gpu = device == "cuda"
    engine = Engine(EngineConfig(str(tiny), device=device, dtype=torch.bfloat16 if gpu else torch.float32,
                                 page_size=16, max_seqs=3, max_step_tokens=100, max_model_len=320,
                                 state_memory=None if gpu else 1 << 22, **SPEC))
    accepted = []
    observe = engine.policy.observe
    engine.policy.observe = lambda seq, k, n, a: (accepted.append(a), observe(seq, k, n, a))
    a, b = _prompts()
    out = [engine.generate([a], GREEDY)[0]]
    saved = engine.stats()["prefix_tokens_saved"]
    out.append(engine.generate([b], GREEDY)[0])
    return engine, out, plans, {"publishes": publishes, "accepted": accepted,
                                "hit": engine.stats()["prefix_tokens_saved"] - saved}


def test_streams_and_block_state_on_the_cpu(tiny, streams, monkeypatch):
    engine, out, plans, seen = _run(tiny, "cpu", monkeypatch)
    a, b = _prompts()
    assert out == _reference(engine, [a, b], "cpu")
    assert [64, 64, 72] == [p[0] for p in plans[:3]]          # a's prefill: cut at 64 and 128 (the step budget)
    assert 192 in seen["publishes"] and seen["hit"] == 192    # in-step at 192, then b's hit adopts it
    assert any(seen["accepted"])                              # verify steps accepted drafts: the sum rewound
    width = (2 * engine.spec.hidden,)
    assert engine.executor.drafter.seen == {("run", width, torch.float32), ("tap", width, torch.float32)}
    rid = engine.add(a, SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1))
    deltas = []
    while not any(d.finish_reason for d in deltas if d.rid == rid):
        deltas += engine.step()
    lp = next(d.prompt_logprobs for d in deltas if d.prompt_logprobs is not None)
    assert len(lp.token) == len(a) - 1 and all(x <= 0 for x in lp.token)


def test_block_state_needs_a_planner_and_a_mixers_default_says_so(tiny, streams, capsys):
    streams.block = "toy.wide_block@1"                        # the block forwards its specs, not a planner
    with pytest.raises(PluginError, match="declares Slot state without a planner"):
        Engine(EngineConfig(str(tiny), device="cpu", dtype=torch.float32, max_model_len=128, state_memory=1 << 22))

    class Mixer:                                              # paged state, no planner: today's default
        @classmethod
        def state_specs(cls, params, layers, ctx):
            return [StateSpec("kv", "paged@1", Slot(8), layers, "test")]

    ctx = types.SimpleNamespace(unit_class=lambda ref: Mixer, device=torch.device("cuda"))
    layers = (types.SimpleNamespace(mixer=Ref("toy.mixer@1")),)
    _, planners = assembly._bind_layers(layers, ctx, None, None, types.SimpleNamespace(cuda_graphs=True),
                                        plan_only=True)
    assert isinstance(planners["kv"], assembly._PagesPlanner)
    assert "state group 'kv' has no planner: steps run without CUDA graphs" in capsys.readouterr().err


@CUDA
def test_streams_and_block_state_from_cuda_graphs(tiny, streams, monkeypatch):
    engine, out, plans, seen = _run(tiny, "cuda", monkeypatch)
    stats = engine.stats()
    assert stats["graphs"] > 0 and 0 < stats["spec_verify_steps_graphed"] == stats["spec_verify_steps"]
    assert seen["hit"] == 192 and any(seen["accepted"]) and all(len(o) == GREEDY.max_tokens for o in out)
    width = (2 * engine.spec.hidden,)
    assert engine.executor.drafter.seen == {("run", width, torch.float32), ("tap", width, torch.float32)}
    engine.close()
    streams.width = 3                                         # declared 3H, the trunk returns 2H: the profile says so
    with pytest.raises(ValueError, match=r"not the \[T, 192\] torch.float32 the model declares"):
        Engine(EngineConfig(str(tiny), device="cuda", max_model_len=128))
