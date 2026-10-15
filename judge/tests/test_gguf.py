"""GGUF on the judge's side (M0.1 design §4.6), on a tiny Qwen3 written as a Q8_0 GGUF: a file's golden is the golden
of its dequantized weights, the per-tensor gate, and Tier A — the engine on the dequantized weights — with each
planted bug wired."""
import json
import pathlib

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
from transformers import AutoModelForCausalLM


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
    assert max(judge_gguf.gate(files[0], files[1]).values()) == 0           # the reference in its stored dtype


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


@pytest.mark.parametrize("kernel_path", ["mmq", "mmvq"])
def test_tier_b_floor_rounds_every_quantized_linear_input_to_q8_1(files, tiny_checkpoint, tiny_prompts, gold,
                                                                  kernel_path):
    from holon_judge import q8_1

    model = AutoModelForCausalLM.from_pretrained(files[1], dtype=torch.float32)        # the file's weights
    types = judge_gguf.emulate_q8_1(model, files[0], kernel_path)
    assert set(types) == {n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    proj = model.model.layers[0].mlp.down_proj
    x = torch.randn(3, proj.in_features)
    want = q8_1.quantize(x, q8_1.blocking("q8_0", kernel_path)).float() @ proj.weight.T
    torch.testing.assert_close(proj(x), want, rtol=0, atol=1e-6)
    floor = golden.record(str(files[0]), tiny_prompts, config=str(tiny_checkpoint), q8_1=kernel_path)
    r = compare(gold, floor).overall
    assert floor.meta["q8_1"] == kernel_path and 0 < r.kl_p99 < 1e-3
    if torch.cuda.is_available():               # D2's per-shape floor (asks the ggml plugin's mmvq_rows): Q8_0 takes
        rows = 8 if kernel_path == "mmvq" else 9    # mmvq up to 8 rows on Blackwell, so it is this path's floor
        per_shape = golden.record(str(files[0]), tiny_prompts, config=str(tiny_checkpoint), q8_1="rows", rows=rows,
                                  crossover="rtx-5090:120")
        assert set(per_shape.meta["routes"].values()) == {kernel_path} and per_shape.meta["step_rows"] == rows
        assert torch.equal(per_shape.target_lp, floor.target_lp) and torch.equal(per_shape.topk_lp, floor.topk_lp)


GOLDENS = pathlib.Path(__file__).resolve().parents[1] / "goldens"
CLASS_SM = {"rtx-5080": 120, "rtx-5090": 120, "rtx-4090": 89, "rtx-3090": 86}      # the lock classes' SMs


@pytest.mark.parametrize("floor", sorted(GOLDENS.glob("*/*-rows*-floor.safetensors")), ids=lambda p: p.stem)
def test_a_per_shape_floor_routes_as_the_table_does(floor):
    """D2 (docs/m07-perf.md §6): a committed per-shape floor emulates each shape on the path the ggml plugin's
    `mmvq_rows` picks today at its step rows on its class (`compare` refuses a candidate whose paths differ), so a
    change to the crossover table that moves a shape must re-record the floor. Needs the GGUF file and a prebuilt
    extension (llama.cpp's rule is host code), no GPU."""
    from holon.api import Fingerprint
    from holon_plugins.kernels.ggml_cuda import extension, plugin
    from huggingface_hub.constants import HF_HUB_CACHE
    from safetensors import safe_open

    meta = json.loads(safe_open(floor, "pt").metadata()["meta"])
    org, repo, name = meta["model"].split("/")
    path = next(pathlib.Path(HF_HUB_CACHE).glob(f"models--{org}--{repo}/snapshots/*/{name}.gguf"), None)
    if path is None:
        pytest.skip(f"{meta['model']} not cached")
    try:
        plugin.rule_module(CLASS_SM[meta["crossover"]])
    except extension.NotBuilt:
        pytest.skip("no prebuilt ggml extension")
    paths = judge_gguf.row_paths(path, meta["step_rows"], Fingerprint(gpu=meta["crossover"],
                                 sm=CLASS_SM[meta["crossover"]], vram_gib=0, toolchain=""))
    routes = {f"{t}/{k}x{o}": paths[n] for n, (t, k, o) in judge_gguf.engine_shapes(path).items()}
    assert meta["routes"] == routes
