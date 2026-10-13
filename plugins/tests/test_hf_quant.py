"""Quantized HF checkpoints on the CPU (docs/m06-lowp.md step A2): tiny checkpoints in every dialect
(`holon_testing.hf_quant_files`) load through std.weights.hf's dialects and serve, on the test-only reference kernels,
what transformers computes on the weights as each file means them; the reference dequantization is each format
owner's; the dialects refuse what the engine cannot honour; the Tier A and structural plants fail."""
import functools
import json
import re
import shutil
import sys
from unittest import mock

import pytest
import torch
from holon.api import QTensor, SamplingParams
from holon.core.engine import Engine, EngineConfig
from holon_judge.hf_quant import owner_check
from holon_judge.planted import hf_moe_plants, hf_quant_plants
from holon_judge.structure import mismatches
from holon_std.weights.hf.source import HFSafetensors
from holon_testing import hf_quant_files as hq
from holon_testing.qwen3_5_files import LLAMA
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM

from .test_engine import _hf_greedy

CASES = [(d, moe) for moe in (False, True) for d in hq.DIALECTS]
ELEM = {"fp8": "fp8_e4m3", "ct-fp8": "fp8_e4m3", "ct-nvfp4": "nvfp4", "modelopt": "nvfp4", "ct-mxfp4": "mxfp4",
        "quark": "mxfp4"}
PLANTS = {**hf_quant_plants(layer=1), **hf_moe_plants(layer=1, experts=4)}
GREEDY = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)
SUFFIXES = re.compile(r".*\.(weight_scale(_inv|_2)?|weight_packed|weight_global_scale|input_scale|input_global_scale"
                      r"|[kv]_scale)$")


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    """(dialect, moe) -> (the quantized checkpoint, the oracle: an fp32 checkpoint of its weights as the file means
    them), written once."""
    root = tmp_path_factory.mktemp("hf_quant")
    tiny = {moe: hq.tiny(root / f"tiny-{moe}", moe) for moe in (False, True)}

    @functools.cache
    def get(dialect: str, moe: bool):
        quantized, dense = root / f"{dialect}-{moe}", root / f"{dialect}-{moe}-dense"
        meant = hq.write(tiny[moe], quantized, dialect)
        dense.mkdir()
        shutil.copy(tiny[moe] / "config.json", dense / "config.json")
        save_file({k: v.contiguous() for k, v in meant.items()}, dense / "model.safetensors")
        return quantized, dense
    return get


@functools.cache
def _oracle(dense) -> torch.nn.Module:
    return AutoModelForCausalLM.from_pretrained(dense, dtype=torch.float32).eval()


def _prompts() -> list[list[int]]:
    g = torch.Generator().manual_seed(1)
    return [torch.randint(0, 512, (n,), generator=g).tolist() for n in (3, 17, 8, 25)]


def _engine(model, **kw) -> Engine:
    cfg = dict(model=str(model), device="cpu", dtype=torch.float32, page_size=4, max_seqs=3, max_step_tokens=10,
               max_model_len=40, state_memory=1 << 20)
    return Engine(EngineConfig(**cfg | kw))


def _error(engine: Engine, oracle: torch.nn.Module) -> float:
    """The largest difference between the engine's prompt logprobs and the oracle's."""
    prompts = _prompts()
    rids = [engine.add(p, SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)) for p in prompts]
    got = {}
    while engine.has_work():
        got |= {d.rid: d.prompt_logprobs for d in engine.step() if d.prompt_logprobs is not None}
    worst = 0.0
    for p, r in zip(prompts, rids, strict=True):
        with torch.no_grad():
            want = oracle(torch.tensor([p])).logits[0, :-1].log_softmax(-1).gather(-1, torch.tensor(p[1:])[:, None])
        worst = max(worst, (torch.tensor(got[r].token) - want[:, 0]).abs().max().item())
    return worst


@pytest.mark.parametrize("dialect,moe", CASES)
def test_a_quantized_checkpoint_serves_what_transformers_computes(files, dialect, moe):
    quantized, dense = files(dialect, moe)
    engine = _engine(quantized)
    assert _error(engine, _oracle(dense)) <= 1e-4
    prompts = _prompts()
    assert _engine(quantized).generate(prompts, GREEDY) == _hf_greedy(_oracle(dense), prompts, GREEDY.max_tokens)
    # q/k/v and gate/up merge into one quantized matmul each (D5: FP8 block-scale rows, FP4 tied global scales), so
    # the only (in, out) are o's, down's and the merged q/k/v's and gate/up's
    shapes = {tuple(int(re.search(rf"\b{f}=(\d+)", k)[1]) for f in ("in", "out"))
              for k in engine.ctx.kernels if k.endswith("/qlinear@2")}
    attention = {(256, 256), (256, 512)}
    assert shapes == (set() if dialect == "quark" else attention) if moe else attention | {(384, 256), (256, 768)}


@pytest.mark.parametrize("dialect,moe", CASES)
def test_each_quantized_module_is_one_qtensor_its_primary_part_first(files, dialect, moe):
    quantized, dense = files(dialect, moe)
    shapes = {k: tuple(v.shape) for k, v in load_file(dense / "model.safetensors").items()}
    yielded = list(HFSafetensors(quantized).tensors())
    assert not [n for n, _ in yielded if SUFFIXES.fullmatch(n)]                     # parts never yielded alone
    assert sorted(n for n, _ in yielded) == sorted(shapes)
    for name, t in yielded:
        if not isinstance(t, QTensor):
            continue
        assert (t.layout.scheme.elem, t.layout.name, next(iter(t.parts))) == (ELEM[dialect], "hf", "weight")
        assert tuple(t.logical_shape) == shapes[name]
        assert t.parts["weight"].dtype == torch.uint8
        assert t.parts["scale"].dtype == (torch.float32 if ELEM[dialect] == "fp8_e4m3" else torch.uint8)
        if "scale2" in t.parts:
            assert t.parts["scale2"].dtype == torch.float32 and t.parts["scale2"].shape == ()


def test_compressed_tensors_global_scales_are_inverted_in_f32(files):
    quantized, _ = files("ct-nvfp4", False)
    raw = {}
    for f in quantized.glob("*.safetensors"):
        raw |= load_file(f)
    for name, t in HFSafetensors(quantized).tensors():
        if isinstance(t, QTensor):
            want = 1.0 / raw[name.replace(".weight", ".weight_global_scale")]
            assert t.parts["scale2"].view(torch.int32) == want.reshape(()).view(torch.int32)


def test_each_part_is_read_once(files):
    from holon_std.weights.hf import source

    quantized, _ = files("modelopt", False)
    reads, opened = {}, source.safe_open

    class Counted:
        def __init__(self, *a, **kw):
            self.f = opened(*a, **kw)

        def __enter__(self):
            self.f.__enter__()
            return self

        def __exit__(self, *exc):
            return self.f.__exit__(*exc)

        def __getattr__(self, name):
            return getattr(self.f, name)

        def get_tensor(self, name):
            reads[name] = reads.get(name, 0) + 1
            return self.f.get_tensor(name)

    with mock.patch.object(source, "safe_open", Counted):
        list(HFSafetensors(quantized).tensors())
    everything = set().union(*(load_file(f).keys() for f in quantized.glob("*.safetensors")))
    assert set(reads.values()) == {1}
    assert everything - set(reads) == {n for n in everything if n.endswith(".input_scale")}    # dropped unread


@pytest.mark.parametrize("dialect,moe", [(d, True) for d in hq.DIALECTS])
def test_the_reference_is_the_format_owners_decoding(files, dialect, moe):
    """D10 on the fixtures (`python -m holon_judge hf-quant-check` runs it on the real files): transformers,
    compressed-tensors, ModelOpt and amd-quark where importable, and gguf-py on llama.cpp's repacks."""
    quantized, _ = files(dialect, moe)
    llama = LLAMA if (LLAMA / "conversion" / "base.py").is_file() else None
    path = list(sys.path)
    report = owner_check(quantized, llama)
    assert sys.path == path             # llama.cpp's converter adds its gguf-py, whose `tests` shadows ours
    expected = {"fp8": "transformers.Fp8Dequantize", "ct-fp8": "compressed-tensors", "ct-nvfp4": "compressed-tensors",
                "ct-mxfp4": "compressed-tensors", "modelopt": "modelopt.NVFP4QTensor", "quark": "quark.Pack_fp4"}
    owners = {expected[dialect]} if dialect.startswith(("fp8", "ct-")) else set()      # optional packages aside
    owners |= {"gguf-py on llama.cpp's repack"} if llama and ELEM[dialect] != "fp8_e4m3" else set()
    assert owners <= set(report), report
    modules = 24 if dialect == "quark" else 32
    assert all(r["modules"] == modules and not r["over"] for r in report.values()), report


@pytest.mark.parametrize("plant,dialect,moe", [
    ("ct-global-not-inverted", "ct-nvfp4", False),               # N1
    ("dialect-k-scale-on-v", "modelopt", False),                 # T1
    ("hf-stack-off-by-one", "quark", True),                      # M2
    ("mxfp4-scale-as-e4m3", "quark", True),                      # M3
    ("mxfp4-scale-halved", "ct-mxfp4", False),                   # M5
    ("gate-up-merge-gate-scales", "fp8", False),                 # P5
    ("gate-up-merge-gate-scales", "modelopt", False),
    ("hf-scale-stack-off-by-one", "fp8", True)])                 # P6
def test_the_tier_a_plants_fail(files, plant, dialect, moe):
    quantized, dense = files(dialect, moe)
    with PLANTS[plant]():                                # loaded and run planted (the reference decodes lazily)
        assert _error(_engine(quantized), _oracle(dense)) > 1e-3            # 10× the passing bound


@pytest.mark.parametrize("dialect", hq.DIALECTS)
def test_every_stacked_expert_slice_dequantizes_to_the_checkpoints(files, dialect):
    quantized, _ = files(dialect, True)
    assert mismatches(_engine(quantized), quantized) == []


@pytest.mark.parametrize("plant,dialect", [("hf-stack-off-by-one", "quark"), ("hf-two-experts-swapped", "ct-nvfp4"),
                                           ("hf-gate-up-swapped", "ct-mxfp4"), ("hf-scale-stack-off-by-one", "fp8"),
                                           ("hf-scale-stack-off-by-one", "modelopt")])
def test_the_structural_test_catches_a_misplaced_quantized_slice(files, plant, dialect):
    quantized, _ = files(dialect, True)
    with PLANTS[plant]():
        engine = _engine(quantized)
    assert mismatches(engine, quantized)


def _edited(src, dst, tensors=None, config=None):
    """A copy of checkpoint `src`: `tensors(name, tensor)` -> tensor or None (dropped), `config(quantization
    config)` edited in place."""
    shutil.copytree(src, dst)
    for f in dst.glob("*.safetensors"):
        kept = {n: e for n, t in load_file(f).items() if (e := tensors(n, t) if tensors else t) is not None}
        save_file(kept, f)
    if config:
        cfg = json.loads((dst / "config.json").read_text())
        config(cfg["quantization_config"])
        (dst / "config.json").write_text(json.dumps(cfg))
    return dst


def _poke(suffix: str, byte: int):
    """The first byte of every `suffix` tensor of layer 0's q_proj set to `byte`."""
    def edit(name, t):
        if name != f"model.layers.0.self_attn.q_proj.{suffix}":
            return t
        bits = t.view(torch.uint8).clone()
        bits.view(-1)[0] = byte
        return bits.view(t.dtype)
    return edit


@pytest.mark.parametrize("dialect,tensors,config,error", [
    ("fp8", None, lambda qc: qc.update(quant_method="gptq"), r"quant_method 'gptq': not supported"),
    ("modelopt", None, lambda qc: qc.update(quant_algo="FP8"), r"only NVFP4"),
    ("quark", None, lambda qc: qc["global_quant_config"]["weight"].update(group_size=16), r"only MXFP4"),
    ("ct-nvfp4", None, lambda qc: qc.update(format="mixed-precision"), r"compressed-tensors format"),
    ("fp8", None, lambda qc: qc["modules_to_not_convert"].append("model.layers.0.self_attn.q_proj"),
     r"self_attn\.q_proj: quantized, but the quantization config leaves it dense"),
    ("ct-mxfp4", None, lambda qc: qc["ignore"].append("re:.*layers\\.0\\.mlp\\.down_proj$"),
     r"mlp\.down_proj: quantized, but"),
    ("ct-nvfp4", lambda n, t: None if n.endswith("q_proj.weight_global_scale") else t, None,
     r"q_proj: a quantized module without \['weight_global_scale'\]"),
    ("modelopt", lambda n, t: torch.tensor(2.0) if n.endswith("k_scale") else t, None, r"k_scale = 2\.0: must be 1\.0"),
    ("fp8", _poke("weight", 0x7F), None, r"q_proj: an FP8 weight is NaN"),
    ("modelopt", _poke("weight_scale", 0x88), None, r"q_proj: an NVFP4 block scale is negative or NaN"),
    ("ct-nvfp4", _poke("weight_scale", 0x7F), None, r"q_proj: an NVFP4 block scale is negative or NaN"),
    ("quark", _poke("weight_scale", 0xFF), None, r"q_proj: an MXFP4 scale is NaN"),
    ("fp8", lambda n, t: torch.cat([t, t[:1]]) if n.endswith("q_proj.weight_scale_inv") else t, None,
     r"the blocks tiling it")])
def test_what_the_engine_cannot_honour_is_a_load_error(files, tmp_path, dialect, tensors, config, error):
    quantized, _ = files(dialect, False)
    with pytest.raises(ValueError, match=error):
        list(HFSafetensors(_edited(quantized, tmp_path / "edited", tensors, config)).tensors())


def test_hf_quant_config_json_is_read_and_hashed_into_the_identity(files, tmp_path):
    quantized, _ = files("modelopt", True)                       # its config only in hf_quant_config.json
    assert "quantization_config" not in json.loads((quantized / "config.json").read_text())
    copy = shutil.copytree(quantized, tmp_path / "copy")
    hq_config = json.loads((copy / "hf_quant_config.json").read_text())
    before = HFSafetensors(copy).identity()
    hq_config["quantization"]["exclude_modules"].append("model.layers.9.mlp.gate")
    (copy / "hf_quant_config.json").write_text(json.dumps(hq_config))
    assert HFSafetensors(copy).identity() != before
