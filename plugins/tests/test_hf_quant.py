"""Quantized HF checkpoints on the CPU (docs/m06-lowp.md step A2): tiny checkpoints in every dialect
(`holon_testing.hf_quant_files`) load through std.weights.hf's dialects and serve, on the test-only reference kernels,
what transformers computes on the weights as each file means them; the reference dequantization is each format
owner's; the dialects refuse what the engine cannot honour; the Tier A and structural plants fail. On a GPU the FP4
MoE fixtures load through kernels.ggml_cuda, whose `prepare` repacks their stacks: the structural test reads those
ggml blocks (gguf-py × the scale2 rows) and still sees every misplaced slice."""
import contextlib
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
from holon_judge import golden
from holon_judge.compare import compare
from holon_judge.hf_quant import GATES, gate, owner_check
from holon_judge.planted import fp4_plants, hf_moe_plants, hf_quant_plants
from holon_judge.recording import Prompt, PromptSet
from holon_judge.structure import mismatches
from holon_std.moe.units import ExpertLinear
from holon_std.weights.hf.source import HFSafetensors
from holon_testing import hf_quant_files as hq
from holon_testing.qwen3_5_files import LLAMA
from holon_testing.ref_kernels import kernels as ref_kernels
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM

CASES = [(d, moe) for moe in (False, True) for d in hq.DIALECTS]
ELEM = {"fp8": "fp8_e4m3", "ct-fp8": "fp8_e4m3", "ct-nvfp4": "nvfp4", "modelopt": "nvfp4", "ct-mxfp4": "mxfp4",
        "quark": "mxfp4"}
PLANTS = {**hf_quant_plants(layer=1), **hf_moe_plants(layer=1, experts=4), **fp4_plants(layer=1)}


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


def _prompt_set() -> PromptSet:
    return PromptSet("tiny", "-", [Prompt(str(i), "chat", p) for i, p in enumerate(_prompts())])


def _engine(model, device: str = "cpu", **kw) -> Engine:
    cfg = dict(model=str(model), device=device, dtype=torch.float32, page_size=4, max_seqs=3, max_step_tokens=10,
               max_model_len=40, state_memory=1 << 20)
    if device == "cuda":                     # kernels.ggml_cuda: bf16 activations, no graphs for a load check
        cfg |= dict(dtype=torch.bfloat16, cuda_graphs=False)
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
    """Each quantized module reaches the engine as one QTensor (primary part first, parts never alone, each read
    once, compressed-tensors' global inverted exactly), and the engine serves the oracle's logprobs."""
    quantized, dense = files(dialect, moe)
    raw = {n: t for f in quantized.glob("*.safetensors") for n, t in load_file(f).items()}
    shapes = {k: tuple(v.shape) for k, v in load_file(dense / "model.safetensors").items()}
    reads = _counted_reads(lambda: list(HFSafetensors(quantized).tensors()))
    yielded = reads.pop(None)
    dropped = {n for n in raw if n.endswith((".input_scale", ".input_global_scale"))}
    assert set(reads.values()) == {1} and set(raw) - set(reads) == dropped               # dropped tensors unread
    assert sorted(n for n, _ in yielded) == sorted(shapes)                          # parts never yielded alone
    for name, t in yielded:
        if isinstance(t, QTensor):
            assert (t.layout.scheme.elem, t.layout.name, next(iter(t.parts))) == (ELEM[dialect], "hf", "weight")
            assert tuple(t.logical_shape) == shapes[name] and t.parts["weight"].dtype == torch.uint8
            assert t.parts["scale"].dtype == (torch.float32 if ELEM[dialect] == "fp8_e4m3" else torch.uint8)
            if dialect == "ct-nvfp4":
                want = 1.0 / raw[name.replace(".weight", ".weight_global_scale")].reshape(())
                assert t.parts["scale2"].shape == () and t.parts["scale2"].view(torch.int32) == want.view(torch.int32)
    engine = _engine(quantized)
    assert _error(engine, _oracle(dense)) <= 1e-4
    if moe:                                  # a 30B's stacks: decoded per call, only the routed experts
        with mock.patch.object(ref_kernels, "CACHE_BYTES", 0):
            assert _error(_engine(quantized), _oracle(dense)) <= 1e-4
    # The judge's golden: the file through its own map and the reference, one layer at a time (docs/m06-lowp.md §4.8);
    # compressed-tensors' NVFP4 decoding is within 2 ulp of the reference's (D10)
    r = compare(golden.record(str(dense), _prompt_set()), golden.record(str(quantized), _prompt_set())).overall
    assert r.top1 == 1 and r.kl_max < 1e-6
    # q/k/v and gate/up merge into one quantized matmul each (D5: FP8 block-scale rows, FP4 tied global scales), so
    # the only (in, out) are o's, down's and the merged q/k/v's and gate/up's
    sites = {tuple(int(re.search(rf"\b{f}=(\d+)", k)[1]) for f in ("in", "out"))
             for k in engine.ctx.kernels if k.endswith("/qlinear@2")}
    attention = {(256, 256), (256, 512)}
    assert sites == (set() if dialect == "quark" else attention) if moe else attention | {(384, 256), (256, 768)}


def _counted_reads(run) -> dict:
    """Tensor name -> how often std.weights.hf read it while `run()` ran; under None, what `run()` returned."""
    from holon_std.weights.hf import source

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
        reads[None] = run()
    return reads


@pytest.mark.parametrize("dialect", hq.DIALECTS)
def test_the_reference_is_the_format_owners_decoding(files, dialect):
    """D10 on the MoE fixtures (`python -m holon_judge hf-quant-check` runs it on the real files): transformers,
    compressed-tensors, ModelOpt and amd-quark where importable, and gguf-py on llama.cpp's repacks."""
    quantized, _ = files(dialect, True)
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


@pytest.mark.parametrize("plant,dialect", [
    ("ct-global-not-inverted", "ct-nvfp4"),                      # N1
    ("dialect-k-scale-on-v", "modelopt"),                        # T1
    ("mxfp4-scale-halved", "ct-mxfp4"),                          # M5
    ("gate-up-merge-gate-scales", "fp8"),                        # P5
    ("mxfp4-scale-as-e4m3", "quark")])                           # M3 (the structural test shares the reference)
def test_the_tier_a_plants_fail(files, plant, dialect):
    """Plants in what the dense fixtures mean; the stack plants (M2, P6) are the structural test's."""
    quantized, dense = files(dialect, False)
    with PLANTS[plant]():                                # loaded and run planted (the reference decodes lazily)
        assert _error(_engine(quantized), _oracle(dense)) > 1e-3            # 10× the passing bound


@pytest.mark.parametrize("dialect,plant", [(d, None) for d in hq.DIALECTS] + [
    ("modelopt", "smoothed"),                                    # the original smoothed: a SmoothQuant file, inverted
    ("ct-mxfp4", "mxfp4-scale-halved"),                          # M5
    ("quark", "mxfp4-scale-as-e4m3")])                           # M3
def test_the_gate_holds_what_the_source_yields_to_the_original(files, tmp_path, dialect, plant):
    """§4.8's gates: every tensor the engine's source yields (per expert; a dense one exactly; a smoothed norm folded
    into its consumers) within its format's NMSE of the original; a dialect's or the reference's misreading of MXFP4
    scales fails them."""
    quantized, _ = files(dialect, True)
    original = quantized.parent / "tiny-True"
    if plant == "smoothed":                  # scale moved from each norm into its consumers' columns, as SmoothQuant
        shutil.copytree(original, tmp_path / "smoothed")
        original, tensors = tmp_path / "smoothed", load_file(original / "model.safetensors")
        for name, t in list(tensors.items()):
            if m := re.fullmatch(r"(.+)\.(input|post_attention)_layernorm\.weight", name):
                s = torch.rand(t.shape, generator=torch.Generator().manual_seed(len(name))) + 0.5
                tensors[name] = t * s
                for c, w in tensors.items():
                    if re.fullmatch(rf"{re.escape(m[1])}\.{'self_attn' if m[2] == 'input' else 'mlp'}\."
                                    r"(.+\.)?(q_proj|k_proj|v_proj|gate_proj|up_proj|gate)\.weight", c):
                        tensors[c] = w / s
        save_file(tensors, original / "model.safetensors")
    with PLANTS[plant]() if plant in PLANTS else contextlib.nullcontext():
        nmse, folded = gate(quantized, original)
    over = [n for n, (elem, v) in nmse.items() if not v <= GATES[elem]]
    assert bool(over) == (plant in PLANTS), over
    assert len(folded) == (4 if plant == "smoothed" else 0)                  # 2 layers × 2 norms, folded if moved
    assert len(nmse) == 2 * (4 * 3 + 9) + 3 - len(folded)
    assert {elem for elem, _ in nmse.values()} == {ELEM[dialect], None} | ({"folded"} if folded else set())


@pytest.mark.parametrize("dialect,moe,modes", [("ct-mxfp4", True, ["q8_1", "fp4"]),
                                               ("modelopt", False, ["q8_1", "fp4", "fp4-static"])])
def test_the_floors_emulate_each_activation_mode(files, dialect, moe, modes):
    """A floor's mode reaches its quantized matmuls (dense, and experts through the MoE block): each moves the
    floor from the plain bf16 one, coarser activations further from the golden (W4A8 < W4A4)."""
    quantized, _ = files(dialect, moe)
    prompts = _prompt_set()
    gold = golden.record(str(quantized), prompts)
    plain = golden.record(str(quantized), prompts, torch.bfloat16)
    kl = {"a16": compare(gold, plain).overall.kl_mean}
    for act in modes:
        floor = golden.record(str(quantized), prompts, torch.bfloat16, q8_1="mmq", act=act)
        assert floor.meta["act"] == act and floor.meta["hf_reference"] and floor.meta["emulation"]
        kl[act] = compare(gold, floor).overall.kl_mean
    assert kl["a16"] < kl["q8_1"] < min(kl[a] for a in modes if a != "q8_1"), kl
    with pytest.raises(ValueError, match="Gate B"):           # FP8's floors are W8A8 / W8A16
        golden.record(str(files("fp8", False)[0]), prompts, torch.bfloat16, q8_1="mmq")


GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("dialect,plants,device", [
    ("fp8", ["hf-scale-stack-off-by-one"], "cpu"),                                            # P6
    ("ct-fp8", ["hf-gate-up-swapped"], "cpu"),
    ("ct-nvfp4", ["hf-two-experts-swapped", "hf-scale2-stack-off-by-one", "hf-gate-up-scale2-swapped"], "cpu"),
    ("ct-mxfp4", ["hf-gate-up-swapped"], "cpu"),
    ("modelopt", ["hf-scale-stack-off-by-one", "hf-scale2-stack-off-by-one", "hf-gate-up-scale2-swapped"], "cpu"),
    ("quark", ["hf-stack-off-by-one"], "cpu"),                                                # M2
    # prepared by kernels.ggml_cuda (docs/m06-lowp.md D8)
    pytest.param("modelopt", ["hf-scale2-stack-off-by-one", "hf-gate-up-scale2-swapped"], "cuda", marks=GPU),
    pytest.param("ct-mxfp4", ["mxfp4-repack-scale-off-by-one"], "cuda", marks=GPU),            # M1
    pytest.param("quark", ["hf-stack-off-by-one"], "cuda", marks=GPU)])
def test_the_structural_test_checks_every_stacked_slice(files, dialect, plants, device):
    """Every slice of the MoE fixture's stacks dequantizes to the checkpoint's; each plant misplaces some."""
    quantized, _ = files(dialect, True)
    engine = _engine(quantized, device)
    if device == "cuda":
        stacks = [m for m in engine.executor.model.modules() if isinstance(m, ExpertLinear) and m.weight is not None]
        assert {m.layout.name for m in stacks} == {"gguf.native"} and len(stacks) == 4        # 2 layers × 2
    assert mismatches(engine, quantized) == []
    del engine
    for plant in plants:
        with PLANTS[plant]():
            engine = _engine(quantized, device)
        assert mismatches(engine, quantized), plant
        del engine


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
    ("fp8", lambda n, t: t[:200] if n == "model.layers.0.self_attn.q_proj.weight" else t, None,
     r"q_proj: FP8 weight float8_e4m3fn \[200, 256\] and scale bfloat16 \[2, 2\]: expected .* whole 128 × 128 blocks")])
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
