"""The logits-processor seams (docs/dev/m05-grammar.md §3), proven with the toy processor `testing.toy_mask` (after a
marker token, only even tokens) on a tiny Qwen3 on the CPU: masks on every path that samples, constrained rows at
depth 0 beside overlapping free rows, the drafts on the host for verify rows, per-request errors, requests blocked
until their processor is ready, and resumed requests."""
import dataclasses

import pytest
import torch
from holon.api import SamplingParams, Status
from holon.core.engine import BLOCKED_POLL_S, Engine, EngineConfig
from transformers import AutoModelForCausalLM

from .fixtures.oracle import ORACLE
from .test_engine import GREEDY, _released, tiny_case, tiny_checkpoint  # noqa: F401 — fixtures
from .test_spec import ORACLE_SLOTS, _plugins, oracle  # noqa: F401 — fixtures

KEY = "toy.even_after@1"
TOY = {"logits_processor": "testing.toy_mask"}
SAMPLED = SamplingParams(temperature=1.0, top_p=0.9, seed=7, max_tokens=12, ignore_eos=True, n=2)


def _engine(path, spec=False, **kw) -> Engine:
    cfg = dict(model=str(path), device="cpu", dtype=torch.float32, page_size=4, max_seqs=3, max_step_tokens=10,
               max_model_len=40, state_memory=1 << 20, max_draft_tokens=3 if spec else 0,
               slots=ORACLE_SLOTS | TOY if spec else TOY) | kw
    return Engine(EngineConfig(**cfg), plugins=_plugins())


def _constrained(params, marker, **opts):
    return dataclasses.replace(params, ext={KEY: {"marker": marker, **opts}})


@pytest.fixture(scope="module")
def masked(tiny_checkpoint, tiny_case):  # noqa: F811
    """The workload: prompts 0 and 2 constrained after their first odd greedy token from the third on (the marker: a
    mask shifted by one column refuses it), 1 and 3 free, and an n = 2 sampled request constrained from its prompt
    on; with the greedy references — HF's logits masked by the rule — and the sampled outputs of a depth-0 engine
    without speculation."""
    prompts, expected = tiny_case
    hf = AutoModelForCausalLM.from_pretrained(tiny_checkpoint, dtype=torch.float32).eval()
    requests, refs = [], []
    for i, (p, e) in enumerate(zip(prompts, expected, strict=True)):
        if i % 2:
            requests.append((p, GREEDY))
            refs.append(e)
            continue
        ids, seen, marker = list(p), False, next(t for t in e[2:] if t % 2)
        for _ in range(GREEDY.max_tokens):
            logits = hf(torch.tensor([ids])).logits[0, -1]
            if seen:
                logits[1::2] = float("-inf")
            ids.append(int(logits.argmax()))
            seen |= ids[-1] == marker
        requests.append((p, _constrained(GREEDY, marker)))
        refs.append(ids[len(p):])
    sampled = prompts[1][:5] + [prompts[0][0]]
    requests.append((sampled, _constrained(SAMPLED, sampled[-1])))
    engine = _engine(tiny_checkpoint, overlap=False)
    refs.append(_run(engine, requests[-1:])[0])
    return requests, refs


def _run(engine, requests):
    """Every request's output per sequence (one list for n = 1); no error finishes."""
    rids = [engine.add(p, params) for p, params in requests]
    out = {(r, i): [] for r, (_, params) in zip(rids, requests, strict=True) for i in range(params.n)}
    while engine.has_work():
        for d in engine.step():
            out[d.rid, d.index] += d.token_ids
            assert d.finish_reason in (None, "length"), d
    return [out[r, 0] if params.n == 1 else [out[r, i] for i in range(params.n)]
            for r, (_, params) in zip(rids, requests, strict=True)]


@pytest.mark.parametrize("memory", [1 << 20, 12 * 2048], ids=["roomy", "preempting"])
@pytest.mark.parametrize("depth", ["depth0", "depth1"])
@pytest.mark.parametrize("mode", ["plain", "right", "wrong", "mixed"])
def test_constrained_outputs_are_the_masked_reference(tiny_checkpoint, masked, oracle, mode, depth, memory):  # noqa: F811
    # Constrained and free requests together through chunked prefill, preemption, prefix hits (a second pass),
    # depth 0 and 1 (D2's gate: a constrained row is committed before the next step), n = 2, verify rows with right,
    # wrong and mixed drafts, greedy and sampled: every output equals its reference (so every sampled row was masked
    # with its request's current state), the processor saw exactly the tokens committed and the drafts the drafter
    # wrote, and at depth 1 every decode step holds every running constrained sequence (none sits a step out).
    requests, refs = masked
    oracle.mode = mode
    oracle.sequences = {tuple(p): p + (ref[0] if params.n > 1 else ref)
                        for (p, params), ref in zip(requests, refs, strict=True)}
    engine = _engine(tiny_checkpoint, spec=mode != "plain", overlap=depth != "depth0", state_memory=memory)
    toy, left_out, launch, fill = engine.processor, [], engine._launch, engine.processor.fill

    def launch_(plan, seqs):
        if all(seq.status is Status.RUNNING for seq in seqs):          # a decode step
            left_out.append(any(seq.constrained and seq not in seqs for seq in engine.sched.running))
        return launch(plan, seqs)

    def fill_(rows, buf):
        for row in rows:                         # §3.3: the drafts the drafter wrote for this row
            assert row.drafts == ORACLE.written.get(row.seq.slot, [])[: row.n - 1]
        fill(rows, buf)

    engine._launch, toy.fill = launch_, fill_
    for _ in range(2):
        assert _run(engine, requests) == refs
    constrained = [ref if params.n > 1 else [ref] for (_, params), ref in zip(requests, refs, strict=True)
                   if KEY in params.ext]
    assert [[toy.out[rid, i] for i in range(len(outs))] for rid, outs in zip(toy.opened, constrained * 2,
                                                                              strict=True)] == constrained * 2
    assert sorted(toy.opened) == sorted(toy.closed) and _released(engine)
    assert not any(left_out)
    stats = engine.stats()
    assert (stats["preemptions"] > 0) == (memory < 1 << 20)
    assert stats["testing.toy_mask.rows_filled"] == len(toy.rows) > 0     # a slot object's own counters
    if mode != "plain":
        assert any(drafts for *_, drafts in toy.rows)
    if mode == "right":
        assert stats["spec_drafted"] > 0 and any(n > 1 for *_, n, _ in oracle.observed)


def test_a_failing_constraint_ends_only_its_request(tiny_checkpoint, masked):  # noqa: F811
    # §3.1, §3.4, §3.5 on one engine at depth 1: a request blocked on an idle engine until its processor is ready,
    # one whose compile times out, one rejected at `open`, an `ext` key nobody serves, an n = 2 request whose second
    # sequence fails mid-generation beside requests that finish unchanged, and a resumed request.
    requests, refs = masked
    engine = _engine(tiny_checkpoint)
    toy = engine.processor
    (p0, c0), (p1, free), (p2, c2) = requests[:3]

    engine.add(p0, dataclasses.replace(c0, ext={KEY: {**c0.ext[KEY], "polls": 3}}))
    for _ in range(3):                                   # compiling: blocked, nothing else to do, nothing raised
        assert engine.idle_for() == BLOCKED_POLL_S and engine.step() == []
    out = []
    while engine.has_work():
        out += [t for d in engine.step() for t in d.token_ids]
    assert out == refs[0] and engine.idle_for() is None

    busy = engine.add(p0, dataclasses.replace(c0, ext={KEY: {"marker": 0, "busy": True}}))
    assert [(d.rid, d.finish_reason, d.kind) for d in engine.step()] == [(busy, "error", "busy")]
    with pytest.raises(ValueError, match="toy rejects"):
        engine.add(p0, dataclasses.replace(c0, ext={KEY: {"marker": 0, "reject": True}}))
    with pytest.raises(ValueError, match="serves"):
        engine.add(p0, dataclasses.replace(GREEDY, ext={"nobody.serves@1": {}}))

    failing = engine.add(p2, dataclasses.replace(c2, n=2, ext={KEY: {**c2.ext[KEY], "fail_at": (1, 5)}}))
    others = [engine.add(p1, free), engine.add(p2, c2)]
    finals, outs = {}, {r: [] for r in others}
    while engine.has_work():
        for d in engine.step():
            if d.rid == failing:
                finals[d.index] = d.finish_reason, d.kind
            else:
                outs[d.rid] += d.token_ids
    assert finals == {0: ("error", "engine"), 1: ("error", "engine")} and [outs[r] for r in others] == refs[1:3]
    assert toy.out[failing, 1] == refs[2][:4]            # the failing step's tokens were dropped

    assert c2.ext[KEY]["marker"] in refs[2][:5]
    resumed = engine.add(p2 + refs[2][:5], c2, resume_output=5)    # its marker is in the resumed output
    out = []
    while engine.has_work():
        out += [t for d in engine.step() for t in d.token_ids]
    assert out == refs[2][5:] and toy.out[resumed, 0] == refs[2]
    assert sorted(toy.opened) == sorted(toy.closed) and _released(engine)

    with pytest.raises(ValueError, match="serves"):      # no processor selected: a constraint is never ignored
        _engine(tiny_checkpoint, slots={}).add(p0, c0)
