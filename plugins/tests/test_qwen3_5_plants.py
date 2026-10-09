"""The judge's Qwen3.5/3.8 plants bite on the tiny model (CPU): each misreading of the converted GGUF moves the
logprobs well beyond the fp32 agreement the clean run keeps, and a payload filed one block early makes a hit
compute something else than recomputing. (The GPU-only plants are calibrated in the 27B runs.)"""
import pytest
import torch
from holon.api import SamplingParams
from holon.core.engine import Engine, EngineConfig
from holon_judge.planted import PLANTS
from holon_testing.qwen3_5_files import can_convert, convert, save, tiny_hf

from .test_qwen3_5_engine import _run

pytestmark = pytest.mark.skipif(not can_convert(), reason="needs llama.cpp's converter and the Qwen3.8 tokenizer")


@pytest.fixture(scope="module")
def tiny_gguf(tmp_path_factory):
    model = tiny_hf(6, vocab=248320).to(torch.bfloat16)
    return model.float(), convert(save(model, tmp_path_factory.mktemp("tiny_plants"), mtp=True))


def _logprobs(path, prompt):
    engine = Engine(EngineConfig(model=str(path), device="cpu", dtype=torch.float32, page_size=4, max_seqs=2,
                                 max_step_tokens=40, max_model_len=128, state_memory=64 << 20))
    rid, got = engine.add(prompt, SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)), None
    while engine.has_work():
        got = next((d.prompt_logprobs for d in engine.step() if d.rid == rid and d.prompt_logprobs), got)
    return torch.tensor(got.token)


@pytest.mark.parametrize("plant", ["qwen35-tiled-as-grouped", "qwen35-norm-offset-twice", "qwen35-a-log-sign",
                                   "qwen35-gate-one-block"])
def test_a_misread_gguf_moves_the_logprobs(tiny_gguf, plant):
    model, path = tiny_gguf
    prompt = torch.randint(0, 248320, (60,), generator=torch.Generator().manual_seed(4)).tolist()
    with torch.no_grad():
        want = model(torch.tensor([prompt])).logits[0, :-1].log_softmax(-1).gather(
            -1, torch.tensor(prompt[1:])[:, None])[:, 0]
    assert (_logprobs(path, prompt) - want).abs().max() < 1e-3                  # clean (fp32, A_log's ulp)
    with PLANTS[plant]():
        assert (_logprobs(path, prompt) - want).abs().max() > 1e-2


def test_a_payload_filed_one_block_early_changes_a_hit(tmp_path):
    path = save(tiny_hf(6, seed=1), tmp_path)
    prompt = torch.randint(0, 320, (140,), generator=torch.Generator().manual_seed(2)).tolist()
    turn = prompt + [7, 8, 9]
    params = SamplingParams(temperature=0.0, max_tokens=5, logprobs=1)

    def engine(**kw):
        return Engine(EngineConfig(model=str(path), device="cpu", dtype=torch.float32, page_size=4, max_seqs=3,
                                   max_step_tokens=48, max_model_len=256, state_memory=8 << 20, **kw))
    cold = _run(engine(prefix_cache=False), turn, params)
    with PLANTS["payload-one-block-early"]():
        hot = engine(plugin_options={"recurrent.policy": "every:64"})
        _run(hot, prompt, params)
        assert _run(hot, turn, params) != cold and hot.stats()["prefix_tokens_saved"] > 0
