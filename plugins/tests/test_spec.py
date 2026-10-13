"""The SeqTable seam (plan §3.4-5): a test drafter proposing fixed tokens runs at overlap depth 1 with correct
positions and lengths. Greedy-exact verification must reproduce plain greedy decoding token for token, whatever the
drafts — right, wrong, mixed, or drafting every other step — with preemption, prefix hits and stops."""
import pathlib

import pytest
import torch
from holon.api import Manifest, SamplingParams
from holon.core.engine import Engine, EngineConfig
from holon.core.plugins import Installed, PluginSet, discover

from .fixtures.oracle import ORACLE
from .test_engine import GREEDY, _released, tiny_case, tiny_checkpoint  # noqa: F401 — fixtures

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _plugins() -> PluginSet:
    found = discover()
    for name in ("oracle_policy", "oracle_drafter"):
        manifest = Manifest.load(FIXTURES / name / "plugin.toml")
        found[manifest.name] = Installed(manifest, f"tests.fixtures.{name}", FIXTURES / name)
    return PluginSet(found)


def _engine(path, **kw) -> Engine:
    cfg = dict(model=str(path), device="cpu", dtype=torch.float32, page_size=4, max_seqs=3, max_step_tokens=10,
               max_model_len=40, state_memory=1 << 20, max_draft_tokens=3) | kw
    return Engine(EngineConfig(**cfg), plugins=_plugins())


@pytest.fixture
def oracle(tiny_case):  # noqa: F811
    prompts, expected = tiny_case
    ORACLE.sequences = {tuple(p): p + e for p, e in zip(prompts, expected, strict=True)}
    ORACLE.by_slot.clear()
    ORACLE.steps, ORACLE.k, ORACLE.vocab, ORACLE.mode = 0, 3, 512, "right"
    ORACLE.observed.clear()
    ORACLE.hidden.clear()
    ORACLE.hidden_slots.clear()
    return ORACLE


def _run(engine, prompts, params):
    """generate(), counting steps and checking after each one that the device SeqTable agrees with the host
    wherever the host knows the length exactly (depth 0: every committed step)."""
    rids = [engine.add(p, params) for p in prompts]
    out, steps = {r: [] for r in rids}, 0
    while engine.has_work():
        for d in engine.step():
            out[d.rid] += d.token_ids
        steps += 1
        if not engine.cfg.overlap:
            for seq in engine.sched.running:
                assert seq.inexact == 0 and int(engine.executor.num_computed[seq.slot]) == seq.num_computed
    return [out[r] for r in rids], steps


@pytest.mark.parametrize("memory", [1 << 20, 12 * 2048], ids=["roomy", "preempting"])
@pytest.mark.parametrize("overlap", [False, True], ids=["depth0", "depth1"])
@pytest.mark.parametrize("mode", ["right", "wrong", "mixed", "alternate"])
def test_greedy_verification_reproduces_greedy(tiny_checkpoint, tiny_case, oracle, mode, overlap, memory):  # noqa: F811
    prompts, expected = tiny_case
    oracle.mode = mode
    engine = _engine(tiny_checkpoint, overlap=overlap, state_memory=memory)
    outs, steps = _run(engine, prompts, GREEDY)
    assert outs == expected and _released(engine)
    if mode == "right" and memory == 1 << 20:
        _, plain_steps = _run(_engine(tiny_checkpoint, overlap=overlap, max_draft_tokens=0), prompts, GREEDY)
        assert steps < plain_steps                       # accepted drafts save steps


@pytest.mark.parametrize("overlap", [False, True], ids=["depth0", "depth1"])
def test_drafts_past_n_draft_are_never_accepted(tiny_checkpoint, tiny_case, oracle, overlap):  # noqa: F811
    # Every row of a verify step has the step's width; a row whose drafter wrote no real drafts (n_draft 0) still
    # verifies, and its drafts — here the right tokens — must not be accepted: it decodes one token per step.
    prompts, expected = tiny_case
    oracle.hidden = {tuple(prompts[1])}
    outs, _ = _run(_engine(tiny_checkpoint, overlap=overlap), prompts, GREEDY)
    assert outs == expected
    hidden = [o for o in oracle.observed if o[0] == tuple(prompts[1])]
    others = [o for o in oracle.observed if o[0] != tuple(prompts[1])]
    assert hidden and all(n_draft == 0 and n_accept == 0 for _, _, n_draft, n_accept in hidden)
    assert any(n_accept > 0 for *_, n_accept in others) and all(k == 3 for _, k, _, _ in oracle.observed)


@pytest.mark.parametrize("memory", [1 << 20, 12 * 2048], ids=["roomy", "preempting"])
@pytest.mark.parametrize("mode", ["right", "mixed"])
def test_the_drafter_sees_every_committed_token(tiny_checkpoint, tiny_case, oracle, mode, memory):  # noqa: F811
    # What a drafter admits (the whole history, prefix hits included) and absorbs (every step's inputs) must equal
    # the committed tokens after every step — through chunked prefill, rejected drafts, preemption and a second
    # pass over the same prompts that hits the prefix cache (M0.4 §3.3).
    prompts, expected = tiny_case
    oracle.mode = mode
    engine = _engine(tiny_checkpoint, state_memory=memory, max_step_tokens=6, overlap=False)   # exact host lengths
    for _ in range(2):
        rids = [engine.add(p, GREEDY) for p in prompts]
        out = {r: [] for r in rids}
        while engine.has_work():
            for d in engine.step():
                out[d.rid] += d.token_ids
            for seq in engine.sched.running:
                n = seq.num_computed
                assert oracle.history[seq.slot, :n].tolist() == seq.tokens[:n]
                absorbed = oracle.absorbed[seq.slot, :n].tolist()            # from the first computed position on:
                first = next(i for i, t in enumerate(absorbed) if t >= 0)    # no gaps (every chunk was absorbed)
                assert absorbed[first:] == seq.tokens[first:n]
        assert [out[r] for r in rids] == expected
    assert engine.state.tokens_saved > 0


def test_a_stop_inside_accepted_drafts(tiny_checkpoint, tiny_case, oracle):  # noqa: F811
    prompts, expected = tiny_case
    stop = expected[1][4]
    params = SamplingParams(temperature=0.0, max_tokens=12, ignore_eos=True, stop_token_ids=(stop,))
    plain, _ = _run(_engine(tiny_checkpoint, max_draft_tokens=0), prompts, params)
    spec, _ = _run(_engine(tiny_checkpoint), prompts, params)
    assert spec == plain and any(o and o[-1] == stop for o in spec)


def test_prefix_hits_and_drafts(tiny_checkpoint, tiny_case, oracle):  # noqa: F811
    prompts, expected = tiny_case
    engine = _engine(tiny_checkpoint, max_model_len=40)
    assert _run(engine, prompts, GREEDY)[0] == expected
    assert _run(engine, prompts, GREEDY)[0] == expected and engine.state.tokens_saved > 0


def test_drafts_never_cross_max_model_len(tiny_checkpoint, tiny_case, oracle):  # noqa: F811
    prompts, expected = tiny_case
    engine = _engine(tiny_checkpoint)
    outs, _ = _run(engine, [prompts[3]], SamplingParams(temperature=0.0, ignore_eos=True))   # stops at 40 tokens
    assert len(prompts[3]) + len(outs[0]) == 40 and outs[0][:12] == expected[3]


SNAP = max(pathlib.Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*"), default=None)


@pytest.mark.skipif(SNAP is None or not torch.cuda.is_available(), reason="needs Qwen3-0.6B + GPU")
def test_host_length_decode_kernels_wait_for_verify_commits():
    # FlashInfer decode plans from host lengths; with drafts every other step, a request's next decode row must wait
    # until its verify row commits (overlap on). Verification is greedy-exact, so outputs stay plain greedy's
    # (bf16: up to kernel rounding, hence an agreement bound).
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SNAP)
    prompts = [tok(t)["input_ids"] for t in ("The capital of France is", "def quicksort(xs):", "Water boils at")]
    params = SamplingParams(temperature=0.0, max_tokens=24, ignore_eos=True)
    cfg = dict(model=str(SNAP), dtype=torch.bfloat16, max_model_len=512, max_seqs=8,
               kernels={"attention.paged_decode": "kernels.flashinfer"})
    plain = _generate(EngineConfig(**cfg), prompts, params)
    expected = _generate(EngineConfig(**cfg, cuda_graphs=False), prompts, params)
    ORACLE.sequences = {tuple(p): p + e for p, e in zip(prompts, expected, strict=True)}
    ORACLE.by_slot.clear()
    ORACLE.mode, ORACLE.k, ORACLE.vocab, ORACLE.steps = "alternate", 3, 151936, 0
    spec = _generate(EngineConfig(**cfg, max_draft_tokens=3), prompts, params, plugins=_plugins(),
                     check=lambda e: e.sched.host_lens_only)
    agree = [next((i for i, (a, b) in enumerate(zip(x, y, strict=True)) if a != b), len(x))
             for x, y in zip(plain, spec, strict=True)]
    assert sum(agree) >= 0.75 * len(agree) * params.max_tokens, agree


def _generate(cfg, prompts, params, plugins=None, check=lambda engine: True):
    """One engine's outputs, freeing its GPU memory before the next engine sizes its pools."""
    import gc

    engine = Engine(cfg, plugins=plugins)
    assert check(engine)
    out = engine.generate(prompts, params)
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return out
