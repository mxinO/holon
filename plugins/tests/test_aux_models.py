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
from holon_testing.qwen3_5_files import save, tiny_hf

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


def _oracle(prompts, continuations, mode="mixed", ahead=False, vocab=512):
    ORACLE.sequences = {tuple(p): p + e for p, e in zip(prompts, continuations, strict=True)}
    ORACLE.by_slot.clear()
    ORACLE.hidden.clear()
    ORACLE.hidden_slots.clear()
    ORACLE.steps, ORACLE.k, ORACLE.mode, ORACLE.ahead, ORACLE.vocab = 0, 3, mode, ahead, vocab
    ORACLE.real, ORACLE.rewinds, ORACLE.max_position = None, 0, -1


def _kv(engine, pool: str, segment: int, units: list[int], n: int) -> torch.Tensor:
    """The first `n` positions' K and V of a request whose pages are `units`, from layer `segment` of `pool` (GQA's
    layout: [layers, units, 2, kv_heads, page, head_dim])."""
    attn = dict(engine.spec.layers[0].mixer.params)
    heads, dim = attn["kv_heads"], attn["head_dim"]
    kv = engine.ledger.pool(pool).units()[segment, units].view(torch.float32).view(len(units), 2, heads, 4, dim)
    return kv.permute(1, 2, 0, 3, 4).reshape(2, heads, -1, dim)[:, :, :n]


@pytest.mark.parametrize("overlap", [False, True], ids=["depth0", "depth1"])
@pytest.mark.parametrize("ahead", [False, True], ids=["absorb", "ahead"])
@pytest.mark.parametrize("memory", [1 << 20, 18 * 2048], ids=["roomy", "preempting"])
def test_an_aux_model_writes_its_own_state_like_the_layer_it_copies(tiny_checkpoint, tiny_case, memory, ahead,  # noqa: F811
                                                                     overlap):
    # "ahead": sub-steps write KV past each row (reserved by the drafter's lookahead) — never into another request's
    # pages, and equal to what the absorb pass writes there once those positions are committed (M0.4 §3.2). At depth
    # 1 the host knows the committed tokens, so the check covers the positions before the last one.
    prompts, expected = tiny_case
    _oracle(prompts, expected, ahead=ahead)
    engine = _engine(tiny_checkpoint, state_memory=memory, overlap=overlap)
    assert "aux.kv" in engine.executor.planners and engine.ledger.pool("state.aux.kv").segments == 1
    for _ in range(2):                                             # the second pass hits the prefix cache
        rids = [engine.add(p, GREEDY) for p in prompts]
        out = {r: [] for r in rids}
        while engine.has_work():
            for d in engine.step():
                out[d.rid] += d.token_ids
            for seq in engine.sched.running:
                n, held = seq.num_computed if not overlap else len(seq.tokens) - 1, engine.state.held[seq.slot]
                assert torch.equal(_kv(engine, "state.aux.kv", 0, held["aux.kv"], n),
                                   _kv(engine, "state.kv", 0, held["kv"], n))
        assert [out[r] for r in rids] == expected
    assert engine.state.tokens_saved > 0 and (memory == 1 << 20 or engine.sched.preemptions > 0)


@pytest.fixture(scope="module")
def tiny_gdn(tmp_path_factory):
    path = save(tiny_hf(4, seed=1), tmp_path_factory.mktemp("tiny_qwen3_5_aux"))
    g = torch.Generator().manual_seed(2)
    prompts = [torch.randint(0, 320, (n,), generator=g).tolist() for n in (5, 70, 131, 30)]
    plain = _engine(path, drafting=False, max_model_len=256, max_step_tokens=48, state_memory=8 << 20).generate(
        prompts, GREEDY)
    return path, prompts, plain


def test_an_aux_recurrent_model_keeps_its_own_state_and_checkpoints(tiny_gdn):
    # The aux model is the target's first layer, a GDN layer: its recurrent state (rolled back after each verify in
    # its own namespace) must equal the target's layer-0 state, and its in-step checkpoints — published beside the
    # target's under its own names — must be filled and adopted for itself, leaving the target's intact: the second
    # pass hits them and still decodes plain greedy (M0.4 §3.1).
    path, prompts, plain = tiny_gdn
    _oracle(prompts, plain, vocab=320)
    # Whole prompts in one step, so their checkpoints (at 64, 128) are filled in-step by the forward.
    engine = _engine(path, max_model_len=256, max_step_tokens=256, state_memory=8 << 20)
    assert "aux.recurrent" in engine.executor.planners
    states = [(pool, engine.ledger.pool(f"state.{pool}")) for pool in ("conv_state", "ssm_state")]
    for _ in range(2):
        rids = [engine.add(p, GREEDY) for p in prompts]
        out = {r: [] for r in rids}
        while engine.has_work():
            for d in engine.step():
                out[d.rid] += d.token_ids
            for seq in engine.sched.running:
                for name, pool in states:
                    aux = engine.ledger.pool(f"state.aux.{name}")
                    assert torch.equal(aux.units()[0, seq.slot], pool.units()[0, seq.slot]), name
        assert [out[r] for r in rids] == plain
    assert engine.state.tokens_saved > 0 and ORACLE.rewinds > 0         # verify steps were rolled back


def test_sub_steps_at_the_length_limit(tiny_checkpoint, tiny_case):  # noqa: F811
    # Two requests decoding up to max_model_len with drafts right to the end: verify padding and the drafter's writes
    # ahead go past the limit — into the block tables' null overflow columns, at positions clamped below the limit
    # for rope — and change nothing: both decode exactly what a run without speculation does (M0.4 §3.2).
    prompts, _ = tiny_case
    two = [prompts[3], prompts[0]]
    params = SamplingParams(temperature=0.0, ignore_eos=True)
    plain = _engine(tiny_checkpoint, drafting=False).generate(two, params)
    _oracle(two, plain, mode="right", ahead=True)
    out = _engine(tiny_checkpoint).generate(two, params)
    assert out == plain and all(len(p) + len(o) == 40 for p, o in zip(two, out, strict=True))
    assert ORACLE.max_position == 39


def test_no_aux_model_without_a_drafter(tiny_checkpoint):  # noqa: F811
    engine = _engine(tiny_checkpoint, drafting=False)
    assert "aux.kv" not in engine.executor.planners and "state.aux.kv" not in engine.ledger.pools
