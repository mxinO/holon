"""Taps and sampled drafts (docs/m04-spec.md §3.4, §3.5; Gate B): a toy drafter (`tests.tap.drafter`) taps the target's
residual stream after its first and last layers and drafts from sparse laws it publishes in the SeqTable.

- What it taps — a copy, and an fp32 accumulation of both layers — equals the residual stream of a model-only
  reference run over the committed tokens at every computed position: across chunked prefill, verify rows (rejected
  drafts' positions recomputed), preemption, at overlap depth 0 and 1.
- Its drafts verify losslessly through `std.spec.rejection` (greedy: plain greedy's tokens).
Taps and sparse laws inside captured verify graphs: test_dflash (a quantized DFlash2 on Qwen3-0.6B).
"""
import dataclasses
import pathlib

import pytest
import torch
from holon.api import Manifest
from holon.core.build import BuildContext, StepContext, build_model, load_weights
from holon.core.engine import Engine, EngineConfig
from holon.core.plugins import Installed, PluginError, PluginSet, discover

from .fixtures.oracle import ORACLE
from .test_engine import GREEDY, _released, tiny_case, tiny_checkpoint  # noqa: F401 — fixtures

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
SLOTS = {"drafter": "tests.tap.drafter", "draft_policy": "tests.oracle.policy", "verifier": "std.spec.rejection"}


def _plugins() -> PluginSet:
    found = discover()
    for name in ("oracle_policy", "tap_drafter"):
        manifest = Manifest.load(FIXTURES / name / "plugin.toml")
        found[manifest.name] = Installed(manifest, f"tests.fixtures.{name}", FIXTURES / name)
    return PluginSet(found)


def _oracle(prompts, continuations, vocab):
    ORACLE.sequences = {tuple(p): p + c for p, c in zip(prompts, continuations, strict=True)}
    ORACLE.by_slot.clear()
    ORACLE.hidden.clear()
    ORACLE.hidden_slots.clear()
    ORACLE.steps, ORACLE.k, ORACLE.mode, ORACLE.vocab = 0, 3, "right", vocab


def _engine(path, drafting=True, **kw) -> Engine:
    cfg = dict(model=str(path), device="cpu", dtype=torch.float32, page_size=4, max_seqs=3, max_step_tokens=10,
               max_model_len=40, state_memory=1 << 20, prefix_cache=False,      # a prefix hit's tokens are not computed
               max_draft_tokens=3 if drafting else 0, slots=SLOTS) | kw
    return Engine(EngineConfig(**cfg), plugins=_plugins())


def _reference(engine) -> tuple:
    """A model-only build of the engine's checkpoint, and the residual stream after each layer for given tokens."""
    ctx = BuildContext(engine.plugins, torch.device("cpu"), torch.float32)
    model = build_model(engine.spec, ctx)
    load_weights(model, ctx, engine.source.tensors())

    @torch.no_grad()
    def stream(tokens: list[int]) -> list[torch.Tensor]:
        step = StepContext.whole_sequences([tokens], "cpu")
        h, out = model.embed(step.batch.dev.input_ids), []
        for layer in model.layers:
            h = layer(h, step)
            out.append(h.clone())
        return out
    return stream


@pytest.mark.parametrize(("overlap", "memory"), [(False, 1 << 20), (True, 12 * 2048)],
                         ids=["depth0-roomy", "depth1-preempting"])
def test_taps_see_the_residual_stream_of_a_reference_run(tiny_checkpoint, tiny_case, memory, overlap):  # noqa: F811
    prompts, expected = tiny_case
    _oracle(prompts, expected, 512)
    engine = _engine(tiny_checkpoint, state_memory=memory, overlap=overlap)
    drafter, stream = engine.executor.drafter, _reference(engine)
    rids = [engine.add(p, dataclasses.replace(GREEDY, seed=1)) for p in prompts]     # seeded: the drafter's coins
    out, checked = {r: [] for r in rids}, 0
    while engine.has_work():
        for d in engine.step():
            out[d.rid] += d.token_ids
        for seq in engine.sched.running:
            # Computed with committed tokens: at depth 1 all but the last, which the step in flight takes as input.
            n = seq.num_computed if not overlap else len(seq.tokens) - 1
            if n:
                first, *_, last = stream(seq.tokens[:n])
                torch.testing.assert_close(drafter.copy[seq.slot, :n], first, rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(drafter.accum[seq.slot, :n], first + last, rtol=1e-5, atol=1e-5)
                checked += 1
    assert [out[r] for r in rids] == expected and _released(engine) and checked
    stats = engine.stats()
    accepted = sum(n for key, n in stats.items() if key.startswith("spec_rows_k3_accepting_") and key[-1] != "0")
    assert accepted and stats["spec_rows_k3_accepting_0"]            # drafts accepted and rejected
    assert memory == 1 << 20 or engine.sched.preemptions > 0


def test_taps_are_checked_against_the_target(tiny_checkpoint, tiny_case):  # noqa: F811
    _oracle(*tiny_case, 512)
    with pytest.raises(PluginError, match=r"taps layers \[2\]; the target has 2"):
        _engine(tiny_checkpoint, plugin_options={"tap.layers": "0,2"})
