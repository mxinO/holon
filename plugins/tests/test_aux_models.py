"""Auxiliary models (M0.4 §3.1): a drafter builds one beside the target through `BuildCtx.aux_model` — units from a
child context that maps state names under its namespace, weights by its own map over the target's checkpoint,
state in its own pools and page tables that the prefix cache shares and preemption releases — and runs it on the
step's tokens through `StepCtx.view`. The toy aux model is the target's layer 0 loaded again, so its KV must equal
the target's layer-0 KV at every computed position."""
import pathlib

import pytest
import torch
from holon.api import Manifest, SamplingParams
from holon.core.engine import Engine, EngineConfig
from holon.core.plugins import Installed, PluginSet, discover

from .fixtures.oracle import ORACLE
from .test_engine import tiny_case, tiny_checkpoint  # noqa: F401 — fixtures

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
GREEDY = SamplingParams(temperature=0.0, max_tokens=12, ignore_eos=True)


def _engine(path, drafting=True, **kw) -> Engine:
    found = discover()
    if drafting:
        for name in ("oracle_policy", "aux_drafter"):
            manifest = Manifest.load(FIXTURES / name / "plugin.toml")
            found[manifest.name] = Installed(manifest, f"tests.fixtures.{name}", FIXTURES / name)
    cfg = dict(model=str(path), device="cpu", dtype=torch.float32, page_size=4, max_seqs=3, max_step_tokens=10,
               max_model_len=40, state_memory=1 << 20, max_draft_tokens=3 if drafting else 0, overlap=False) | kw
    return Engine(EngineConfig(**cfg), plugins=PluginSet(found))


def _kv(engine, pool: str, segment: int, units: list[int], n: int) -> torch.Tensor:
    """The first `n` positions' K and V of a request whose pages are `units`, from layer `segment` of `pool` (GQA's
    layout: [layers, units, 2, kv_heads, page, head_dim])."""
    attn = dict(engine.spec.layers[0].mixer.params)
    heads, dim = attn["kv_heads"], attn["head_dim"]
    kv = engine.ledger.pool(pool).units()[segment, units].view(torch.float32).view(len(units), 2, heads, 4, dim)
    return kv.permute(1, 2, 0, 3, 4).reshape(2, heads, -1, dim)[:, :, :n]


@pytest.mark.parametrize("memory", [1 << 20, 18 * 2048], ids=["roomy", "preempting"])
def test_an_aux_model_writes_its_own_state_like_the_layer_it_copies(tiny_checkpoint, tiny_case, memory):  # noqa: F811
    prompts, expected = tiny_case
    ORACLE.sequences = {tuple(p): p + e for p, e in zip(prompts, expected, strict=True)}
    ORACLE.by_slot.clear()
    ORACLE.steps, ORACLE.k, ORACLE.mode = 0, 3, "right"
    engine = _engine(tiny_checkpoint, state_memory=memory)
    assert "aux.kv" in engine.executor.planners and engine.ledger.pool("state.aux.kv").segments == 1
    for _ in range(2):                                             # the second pass hits the prefix cache
        rids = [engine.add(p, GREEDY) for p in prompts]
        out = {r: [] for r in rids}
        while engine.has_work():
            for d in engine.step():
                out[d.rid] += d.token_ids
            for seq in engine.sched.running:
                n, held = seq.num_computed, engine.state.held[seq.slot]
                assert torch.equal(_kv(engine, "state.aux.kv", 0, held["aux.kv"], n),
                                   _kv(engine, "state.kv", 0, held["kv"], n))
        assert [out[r] for r in rids] == expected
    assert engine.state.tokens_saved > 0 and (memory == 1 << 20 or engine.sched.preemptions > 0)


def test_no_aux_model_without_a_drafter(tiny_checkpoint):  # noqa: F811
    engine = _engine(tiny_checkpoint, drafting=False)
    assert "aux.kv" not in engine.executor.planners and "state.aux.kv" not in engine.ledger.pools
