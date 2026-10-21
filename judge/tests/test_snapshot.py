"""Snapshots: a trace through the real engine (CPU) is run-identical; any change is located."""
import dataclasses
import pathlib

import pytest
import torch
from holon.api import SamplingParams
from holon.core.engine import Engine, EngineConfig
from holon_judge.__main__ import main
from holon_judge.recording import PromptSet
from holon_judge.snapshot import Request, Snapshot, diff, record, repeat, trace

GREEDY = SamplingParams(temperature=0.0, max_tokens=6, ignore_eos=True)
# Seeded sampling with n > 1 and logprobs, as the snapshot trace of design §7.10 requires.
SAMPLED = SamplingParams(temperature=1.0, top_k=50, seed=7, n=2, max_tokens=6, ignore_eos=True, logprobs=3,
                         prompt_logprobs=3)


def _engine(path, **kw) -> Engine:
    # A small step budget makes prefill chunked and batches mixed, so the schedule itself is part of the snapshot.
    return Engine(EngineConfig(**dict(model=str(path), device="cpu", dtype=torch.float32, page_size=4, max_seqs=3,
                                      max_model_len=48, max_step_tokens=10, state_memory=1 << 20) | kw))


@pytest.fixture(scope="module")
def requests(tiny_prompts):
    return [Request(p.ids, SAMPLED if i == 1 else GREEDY) for i, p in enumerate(tiny_prompts.prompts)]


@pytest.fixture(scope="module")
def snap(tiny_checkpoint, requests) -> Snapshot:
    return record(_engine(tiny_checkpoint), requests, meta={"trace": "tiny"})


def test_trace_is_run_identical(tiny_checkpoint, requests, snap):
    again = record(_engine(tiny_checkpoint), requests)
    assert diff(snap, again) is None
    assert [[len(seq) for seq in s] for s in snap.streams] == [[6], [6, 6], [6], [6]]
    assert any(step == [] for step in snap.steps)          # prefill-only steps are kept


def test_unseeded_sampling_is_refused(tiny_checkpoint, tiny_prompts):
    with pytest.raises(ValueError, match="without a seed"):
        record(_engine(tiny_checkpoint), [Request(tiny_prompts.prompts[0].ids, SamplingParams(temperature=1.0))])
    with pytest.raises(ValueError, match="does not come before it"):
        record(_engine(tiny_checkpoint), [Request(tiny_prompts.prompts[0].ids, GREEDY, after=0)])


def _covers_buckets(widths, buckets):
    """Every bucket b is used by some step width in (previous bucket, b]."""
    return all(any(lo < w <= b for w in widths) for lo, b in zip((0, *buckets), buckets, strict=False))


def test_snapshot_trace_on_cpu(tiny_checkpoint, tiny_prompts):
    # Eight sequences at once; the late request reuses the long one's cached prefix.
    requests = trace(tiny_prompts, width=8, short_len=6, long_len=40)
    assert requests[1].after == 0 and requests[1].ids[:len(requests[0].ids)] == requests[0].ids
    runs = [record(_engine(tiny_checkpoint, max_seqs=8, max_model_len=64, max_step_tokens=16), requests)
            for _ in range(3)]
    assert diff(runs[0], runs[1]) is None and diff(runs[0], runs[2]) is None
    assert runs[0].meta["tokens_saved"] > 0
    assert _covers_buckets(runs[0].meta["step_widths"], (1, 2, 4, 8))


QWEN = max(pathlib.Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*"), default=None)


@pytest.mark.model
@pytest.mark.skipif(QWEN is None or not torch.cuda.is_available(), reason="needs Qwen3-0.6B + GPU")
def test_snapshot_trace_is_run_identical_on_qwen3():
    """The acceptance check of design §7.10 on the production engine (graphs, overlap, prefix cache)."""
    prompts = PromptSet.load(pathlib.Path(__file__).resolve().parents[1] / "prompts" / "v1.jsonl")
    requests = trace(prompts)
    runs, pools = repeat(lambda memory: Engine(EngineConfig(model=str(QWEN), state_memory=memory)), requests, 3)
    assert pools[0] == pools[1] == pools[2]
    assert diff(runs[0], runs[1]) is None and diff(runs[0], runs[2]) is None
    assert runs[0].meta["tokens_saved"] > 0 and len(requests[0].ids) == 2048
    assert _covers_buckets(runs[0].meta["step_widths"], (1, 2, 4, 8, 16, 24, 32, 48, 64))


def test_repeated_runs_get_the_first_runs_pools(tiny_checkpoint, requests):
    # Pool sizes set admission and preemption, so the schedule: engines built one after another measure different free
    # memory (a predecessor's cached blocks; the process's own growth) — here each would size its own differently —
    # so later runs get the bytes of the first run's pools.
    measured = iter([3 << 19, 1 << 20, 1 << 20])
    runs, pools = repeat(lambda memory: _engine(tiny_checkpoint, state_memory=memory or next(measured)), requests, 3)
    assert pools[0] == pools[1] == pools[2] and diff(runs[0], runs[2]) is None


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
