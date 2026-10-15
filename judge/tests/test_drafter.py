"""L0-drafter (`holon_judge drafter`) on a tiny Qwen3.5, on CPU: the engine's draft heads in fp32 agree with their
references to fp32 rounding, so they pass against the fp32 golden with the bf16 reference as the floor; planted bugs
fail. The MTP block: against transformers' modules; its concat halves swapped (`mtp_concat_order`) fails. A tiny
random DFlash2: against `reference.py` on the target streams the judge records itself; taps a layer late
(`dflash-tap-offset`) fails. The MTP block from a GGUF, chained: against the judge's own forward of the file's head."""
import pytest
import torch
from holon.core.engine import Engine, EngineConfig
from holon_judge import drafter
from holon_judge.compare import compare_drafter
from holon_judge.recording import Prompt, PromptSet
from holon_testing.qwen3_5_files import save, tiny_hf


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    path = save(tiny_hf(4, seed=1), tmp_path_factory.mktemp("tiny_qwen3_5_l0_drafter"), mtp=True)
    g = torch.Generator().manual_seed(9)
    prompts = PromptSet("tiny", "0", [Prompt(str(i), "x", torch.randint(0, 320, (n,), generator=g).tolist())
                                      for i, n in enumerate((40, 70, 90))])
    return path, prompts, drafter.reference(str(path), prompts), drafter.reference(str(path), prompts, torch.bfloat16)


def _engine(path):
    return Engine(EngineConfig(model=str(path), device="cpu", dtype=torch.float32, page_size=4, max_seqs=2,
                               max_step_tokens=128, max_model_len=128, state_memory=8 << 20, prefix_cache=False,
                               max_draft_tokens=3, plugin_options={"spec.k": "3"},
                               slots={"drafter": "drafters.mtp", "draft_policy": "std.spec.fixed_k",
                                      "verifier": "std.spec.greedy"}))


def test_the_engine_head_passes(case):
    path, prompts, golden, floor = case
    result = compare_drafter(golden, drafter.candidate(_engine(path), prompts), floor)
    assert result.passed and result.overall.top1 > 0.99


def test_a_swapped_concat_fails(case, monkeypatch):
    path, prompts, golden, floor = case
    from holon_std.spec.mtp_head import units

    init = units.MTPHead.__init__
    monkeypatch.setattr(units.MTPHead, "__init__", lambda self, ctx, **kw: init(self, ctx, **{**kw, "order":
                                                                                               "hidden_first"}))
    assert compare_drafter(golden, drafter.candidate(_engine(path), prompts), floor).passed is False


def test_dflash_passes_and_its_tap_offset_fails(case, tmp_path):
    from holon_judge.planted import dflash_tap_offset
    from holon_testing import dflash_files

    path, prompts, _, _ = case
    draft = dflash_files.save(tmp_path / "draft", 64, 320, (0, 2), mask=7)
    cuts = drafter.cut_prompts(prompts)

    def judged():
        engine = Engine(EngineConfig(
            model=str(path), device="cpu", dtype=torch.float32, page_size=4, max_seqs=2, max_step_tokens=128,
            max_model_len=128, state_memory=8 << 20, prefix_cache=False, max_draft_tokens=7,
            plugin_options={"spec.k": "7", "dflash.path": str(draft)},
            slots={"drafter": "drafters.dflash", "draft_policy": "std.spec.fixed_k", "verifier": "std.spec.rejection"}))
        cand, inputs = drafter.dflash_candidate(engine, cuts)
        golden, floor = (drafter.dflash_reference(str(draft), str(path), cuts, inputs, dtype)
                         for dtype in (torch.float32, torch.bfloat16))
        return compare_drafter(golden, cand, floor)

    result = judged()
    assert result.passed and result.overall.top1 > 0.99 and result.overall.n == 7 * len(cuts.prompts)
    with dflash_tap_offset():
        assert judged().passed is False


def test_a_gguf_mtp_chain_passes_and_its_swapped_concat_fails(tiny_qwen3_5_gguf, monkeypatch):
    # The head stored in a GGUF (llama.cpp's converter, f32), judged per cut with its chain of 3 drafts: steps 2 and
    # 3 teacher-forced with the candidate's drafts. The concat halves swapped fail.
    import gguf
    from holon_std.spec.mtp_head import units

    _, path = tiny_qwen3_5_gguf
    g = torch.Generator().manual_seed(9)
    cuts = drafter.cut_prompts(PromptSet("tiny", "0", [Prompt(str(i), "x", torch.randint(0, 320, (n,), generator=g)
                                                              .tolist()) for i, n in enumerate((40, 70))]), 3, 4)

    reader = gguf.GGUFReader(path)
    references = [drafter.MTPReference(str(path), dtype, reader=reader) for dtype in (torch.float32, torch.bfloat16)]

    def judged():
        engine = Engine(EngineConfig(model=str(path), device="cpu", dtype=torch.float32, page_size=4, max_seqs=2,
                                     max_step_tokens=128, max_model_len=128, state_memory=64 << 20,
                                     prefix_cache=False, max_draft_tokens=3, plugin_options={"spec.k": "3"},
                                     slots={"drafter": "drafters.mtp", "draft_policy": "std.spec.fixed_k",
                                            "verifier": "std.spec.greedy"}))
        cand, inputs = drafter.mtp_candidate(engine, cuts, 3)
        golden, floor = (r.record(cuts, inputs) for r in references)
        return compare_drafter(golden, cand, floor)

    result = judged()
    assert result.passed and result.overall.top1 > 0.99 and result.overall.n == 3 * len(cuts.prompts)
    init = units.MTPHead.__init__
    monkeypatch.setattr(units.MTPHead, "__init__", lambda self, ctx, **kw: init(self, ctx, **{**kw, "order":
                                                                                               "hidden_first"}))
    assert judged().passed is False


def test_a_draft_floor_routes_by_the_engine_s_merged_shapes():
    """The DFlash and MTP floors' GGUF matmuls take the path of the matmul the engine runs (docs/m07-perf.md §17): a
    draft's gate and up merged, as the target's floors do (`gguf.merged`) — on the RTX 5090 IQ4_XS's 5120x34816 entry
    is 7 rows, the gate's own 5120x17408 6. Needs a prebuilt extension (llama.cpp's rule is host code), no GPU."""
    import types

    import gguf
    from holon.api import Fingerprint
    from holon_plugins.kernels.ggml_cuda import extension, plugin

    try:
        lib = plugin.rule_module(120)
    except extension.NotBuilt:
        pytest.skip("no prebuilt ggml extension")

    def tensor(name: str, k: int, out: int, kind: str = "IQ4_XS"):
        return types.SimpleNamespace(name=name, tensor_type=gguf.GGMLQuantizationType[kind], shape=(k, out))
    named = [(f"layers.0.mlp.{p}_proj.weight", tensor(f"blk.0.ffn_{p}.weight", *shape))
             for p, shape in (("gate", (5120, 17408)), ("up", (5120, 17408)), ("down", (17408, 5120)))]
    named.append(("layers.0.input_layernorm.weight", tensor("blk.0.attn_norm.weight", 5120, 1, "F32")))
    limits = drafter.row_limits(named, Fingerprint(gpu="rtx-5090", sm=120, vram_gib=0, toolchain=""), lib)
    assert limits == {"layers.0.mlp.gate_proj.weight": ("iq4_xs", 7), "layers.0.mlp.up_proj.weight": ("iq4_xs", 7),
                      "layers.0.mlp.down_proj.weight": ("iq4_xs", 6)}
