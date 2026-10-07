"""Snapshots: a trace through the real engine (CPU) is run-identical; any change is located."""
import dataclasses

import pytest
import torch
from holon.api import SamplingParams
from holon.core.engine import Engine, EngineConfig
from holon_judge.__main__ import main
from holon_judge.snapshot import Snapshot, diff, record

GREEDY = SamplingParams(temperature=0.0, max_tokens=6, ignore_eos=True)
# Seeded sampling with n > 1 and logprobs, as the snapshot trace of design §7.10 requires.
SAMPLED = SamplingParams(temperature=1.0, top_k=50, seed=7, n=2, max_tokens=6, ignore_eos=True, logprobs=3,
                         prompt_logprobs=3)


def _engine(path) -> Engine:
    # A small step budget makes prefill chunked and batches mixed, so the schedule itself is part of the snapshot.
    return Engine(EngineConfig(model=str(path), device="cpu", dtype=torch.float32, page_size=4, max_seqs=3,
                               max_model_len=48, max_step_tokens=10, state_memory=1 << 20))


@pytest.fixture(scope="module")
def trace(tiny_prompts):
    return [(p.ids, SAMPLED if i == 1 else GREEDY) for i, p in enumerate(tiny_prompts.prompts)]


@pytest.fixture(scope="module")
def snap(tiny_checkpoint, trace) -> Snapshot:
    return record(_engine(tiny_checkpoint), trace, meta={"trace": "tiny"})


def test_trace_is_run_identical(tiny_checkpoint, trace, snap):
    again = record(_engine(tiny_checkpoint), trace)
    assert diff(snap, again) is None
    assert [[len(seq) for seq in s] for s in snap.streams] == [[6], [6, 6], [6], [6]]
    assert any(step == [] for step in snap.steps)          # prefill-only steps are kept


def test_unseeded_sampling_is_refused(tiny_checkpoint, tiny_prompts):
    with pytest.raises(ValueError, match="without a seed"):
        record(_engine(tiny_checkpoint), [(tiny_prompts.prompts[0].ids, SamplingParams(temperature=1.0))])


def test_changes_are_located(snap):
    # the first delta of a second sequence (n = 2) changes
    s, k = next((s, k) for s, step in enumerate(snap.steps) for k, e in enumerate(step) if e[1] == 1)
    steps = [[list(e) for e in step] for step in snap.steps]
    steps[s][k][2] = "0" * 64
    i = steps[s][k][0]
    changed = dataclasses.replace(snap, steps=steps)
    assert diff(snap, changed) == f"step {s}: deltas differ for (request, sequence) [({i}, 1)]"
    assert diff(snap, dataclasses.replace(snap, steps=snap.steps[:-1])).startswith("step count")


def test_save_load_and_cli(snap, tmp_path, capsys):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    snap.save(a)
    assert Snapshot.load(a) == snap
    dataclasses.replace(snap, steps=snap.steps[:-1]).save(b)
    assert main(["snapshot-diff", str(a), str(a)]) == 0
    assert main(["snapshot-diff", str(a), str(b)]) == 1
    n = len(snap.steps)
    assert capsys.readouterr().out.splitlines() == ["identical", f"step count {n} != {n - 1}"]
