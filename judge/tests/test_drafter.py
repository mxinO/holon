"""L0-drafter (`holon_judge drafter`) on a tiny Qwen3.5 with an MTP block, on CPU: the engine's head in fp32 agrees
with the transformers reference to fp32 rounding, so it passes against the fp32 golden with the bf16 reference as
the floor; the head with its concat halves swapped (the `mtp_concat_order` plant) fails."""
import pytest
import torch
from holon.core.engine import Engine, EngineConfig
from holon_judge import drafter
from holon_judge.compare import compare
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
    result = compare(golden, drafter.candidate(_engine(path), prompts), floor)
    assert result.passed and result.overall.top1 > 0.99


def test_a_swapped_concat_fails(case, monkeypatch):
    path, prompts, golden, floor = case
    from holon_std.spec.mtp_head import units

    init = units.MTPHead.__init__
    monkeypatch.setattr(units.MTPHead, "__init__", lambda self, ctx, **kw: init(self, ctx, **{**kw, "order":
                                                                                               "hidden_first"}))
    assert compare(golden, drafter.candidate(_engine(path), prompts), floor).passed is False
