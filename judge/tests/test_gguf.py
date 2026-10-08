"""GGUF on the judge's side (M0.1 design §4.6), on a tiny Qwen3 written as a Q8_0 GGUF: a file's golden is the golden
of its dequantized weights, the per-tensor gate, and Tier A — the engine on the dequantized weights — with each
planted bug wired."""
import gguf
import numpy as np
import pytest
import torch
from holon.core.engine import Engine, EngineConfig
from holon_judge import gguf as judge_gguf
from holon_judge import golden
from holon_judge.compare import compare
from holon_judge.engine import score
from holon_judge.planted import PLANTS
from holon_testing.gguf_files import write
from safetensors.torch import save_file


@pytest.fixture(scope="module")
def files(tiny_checkpoint, tmp_path_factory):
    """The GGUF file and an HF checkpoint of its dequantized weights."""
    d = tmp_path_factory.mktemp("gguf")
    file, dense = d / "tiny-Q8_0.gguf", d / "dense"
    dequantized = write(tiny_checkpoint, file, gguf.GGMLQuantizationType.Q8_0)
    dense.mkdir()
    (dense / "config.json").write_text((tiny_checkpoint / "config.json").read_text())
    save_file({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in dequantized.items()},
              dense / "model.safetensors")
    return file, dense


@pytest.fixture(scope="module")
def gold(files, tiny_checkpoint, tiny_prompts):
    return golden.record(str(files[0]), tiny_prompts, config=str(tiny_checkpoint))


def test_a_files_golden_is_the_golden_of_its_dequantized_weights(files, tiny_prompts, gold):
    assert gold.same_tensors(golden.record(str(files[1]), tiny_prompts))
    assert gold.meta["model"] == "tiny-Q8_0" and gold.meta["name_map"] == judge_gguf.NAME_MAP_HASH
    assert len(gold.meta["gguf_sha256"]) == 64 and gold.meta["config_sha256"]


def test_the_per_tensor_gate(files, tiny_checkpoint):
    nmse = judge_gguf.gate(files[0], tiny_checkpoint)
    assert len(nmse) == 2 * 11 + 3 and max(nmse.values()) < 1e-3            # Q8_0: far inside the gate
    assert judge_gguf.gate(files[0], files[1]) == dict.fromkeys(nmse, 0.0)


def _tier_a(file, prompts):
    engine = Engine(EngineConfig(model=str(file), source="gguf_dense", device="cpu", dtype=torch.float32,
                                 page_size=4, max_seqs=4, max_model_len=64, max_step_tokens=16, state_memory=1 << 20))
    assert engine.source.kind == "gguf"          # the GGUF configuration and weight map, dense weights
    return score(engine, prompts)


@pytest.fixture(scope="module")
def clean(files, tiny_prompts, gold):
    return compare(gold, _tier_a(files[0], tiny_prompts)).overall


def test_tier_a_scores_the_dequantized_model(clean):
    assert clean.top1 == 1 and clean.kl_p99 < 1e-8


@pytest.mark.parametrize("plant", sorted(p for p in PLANTS if p.startswith("gguf-")))
def test_a_planted_tier_a_bug_moves_the_scores(files, tiny_prompts, gold, clean, plant):
    # Wiring only: a random tiny model's flat distributions hide how much (calibrated on the 8B files, §4.6).
    with PLANTS[plant]():
        r = compare(gold, _tier_a(files[0], tiny_prompts))
    assert r.overall.kl_p99 > 100 * clean.kl_p99, r.table()
