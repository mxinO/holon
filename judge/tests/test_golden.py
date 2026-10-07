"""Recorder: HF teacher forcing on a tiny random Qwen3 (CPU), storage round trip, provenance, CLI."""
import os

import pytest
import torch
from holon_judge import golden
from holon_judge.__main__ import main
from holon_judge.compare import compare
from holon_judge.recording import PromptSet, Recording
from transformers import AutoModelForCausalLM


@pytest.fixture(scope="module")
def recording(tiny_checkpoint, tiny_prompts) -> Recording:
    return golden.record(str(tiny_checkpoint), tiny_prompts, k=20)


def test_recording_matches_hf_logprobs(tiny_checkpoint, tiny_prompts, recording):
    lens = [len(p.ids) - 1 for p in tiny_prompts.prompts]
    assert recording.offsets.tolist() == [0, *torch.tensor(lens).cumsum(0).tolist()]
    assert recording.topk_ids.shape == (sum(lens), 20) and recording.topk_ids.dtype == torch.int32
    hf = AutoModelForCausalLM.from_pretrained(tiny_checkpoint, dtype=torch.float32).eval()
    ids = tiny_prompts.prompts[1].ids
    with torch.no_grad():
        lp = hf(torch.tensor([ids])).logits[0, :-1].double().log_softmax(-1)
    rows = slice(*recording.offsets[1:3].tolist())
    assert torch.equal(recording.topk_ids[rows], lp.topk(20, -1).indices.int())
    assert torch.allclose(recording.target_lp[rows].double(), lp.gather(1, torch.tensor(ids[1:])[:, None])[:, 0])
    # top-k and tail together hold all of the probability mass
    mass = torch.cat([recording.topk_lp, recording.tail_lp[:, None]], 1).double().logsumexp(-1)
    assert mass.abs().max() < 1e-6


def test_provenance(recording, tiny_prompts):
    m = recording.meta
    assert m["prompts_sha256"] == tiny_prompts.sha256 and m["categories"] == ["chat", "code", "chat", "agent"]
    assert m["dtype"] == "float32" and m["device"] == "cpu" and m["threads"] == torch.get_num_threads()
    assert list(m["weight_sha256"]) == ["model.safetensors"] and len(m["config_sha256"]) == 64
    assert m["revision"] is None and "/" not in m["model"]       # a plain directory: its name, never its path


def test_hub_cache_snapshot_provenance(tmp_path):
    blob = "ab" * 32
    snap = tmp_path / "models--Org--Some-Model" / "snapshots" / "rev123"
    snap.mkdir(parents=True)
    (tmp_path / "models--Org--Some-Model" / "blobs").mkdir()
    (tmp_path / "models--Org--Some-Model" / "blobs" / blob).write_bytes(b"weights")
    os.symlink(f"../../blobs/{blob}", snap / "model.safetensors")
    meta = golden.model_meta(snap)
    assert meta["model"] == "Org/Some-Model" and meta["revision"] == "rev123"
    assert meta["weight_sha256"] == {"model.safetensors": blob}


def test_save_load_round_trip_and_self_compare(recording, tmp_path):
    recording.save(tmp_path / "g.safetensors")
    back = Recording.load(tmp_path / "g.safetensors")
    assert back.same_tensors(recording) and back.meta == recording.meta
    assert compare(recording, back, floor=back).passed


def test_prompt_set_is_validated(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text('{"id": "a", "category": "chat", "ids": [1, 2, 3]}\n')
    (tmp_path / "p.jsonl.sha256").write_text("0" * 64 + "  p.jsonl\n")
    with pytest.raises(ValueError, match="does not match"):
        PromptSet.load(path)
    (tmp_path / "p.jsonl.sha256").unlink()
    path.write_text('{"id": "a", "category": "chat", "ids": [1]}\n')
    with pytest.raises(ValueError, match="nothing to score"):
        PromptSet.load(path)


def test_cli_golden_is_reproducible(tiny_checkpoint, tiny_prompts, recording, tmp_path, capsys):
    prompts = tmp_path / "p.jsonl"
    prompts.write_text("".join(f'{{"id": "{p.id}", "category": "{p.category}", "ids": {p.ids}}}\n'
                               for p in tiny_prompts.prompts))
    out = tmp_path / "g.safetensors"
    threads = torch.get_num_threads()
    try:
        assert main(["golden", "--model", str(tiny_checkpoint), "--prompts", str(prompts), "--out", str(out),
                     "--threads", str(threads)]) == 0
    finally:
        torch.set_num_threads(threads)
    assert Recording.load(out).same_tensors(recording)
    assert '"n_positions"' in capsys.readouterr().out
