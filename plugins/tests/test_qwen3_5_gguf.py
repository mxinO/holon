"""The GGUF path of models/qwen3_5 against an oracle that does not share its beliefs: a tiny random checkpoint (bf16
weights, Qwen3.8's head ratio 3, an MTP block) converted by llama.cpp's own converter (`convert_hf_to_gguf.py`, f32),
then served by the engine on the CPU — it must compute what transformers computes on the checkpoint. The converter
writes norms as w + 1, `ssm_a = −exp(A_log)` and value heads tiled, and appends the MTP block as an extra layer."""
import pytest
import torch
from holon.api import SamplingParams
from holon.core.engine import Engine, EngineConfig
from holon_testing.qwen3_5_files import can_convert, convert, save, tiny_hf

pytestmark = pytest.mark.skipif(not can_convert(), reason="needs llama.cpp's converter and the Qwen3.8 tokenizer")


@pytest.fixture(scope="module")
def tiny_gguf(tmp_path_factory):
    model = tiny_hf(6, vocab=248320).to(torch.bfloat16)      # bf16 values: the converter's f32 w + 1 undoes exactly
    path = save(model, tmp_path_factory.mktemp("tiny_qwen3_5_gguf"), mtp=True)
    return model.float(), convert(path)


def _logprobs(engine, prompt):
    rid, got = engine.add(prompt, SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)), None
    while engine.has_work():
        got = next((d.prompt_logprobs for d in engine.step() if d.rid == rid and d.prompt_logprobs), got)
    return torch.tensor(got.token)


def _engine(path):
    return Engine(EngineConfig(model=str(path), device="cpu", dtype=torch.float32, page_size=4, max_seqs=2,
                               max_step_tokens=40, max_model_len=128, state_memory=64 << 20))


def test_the_gguf_serves_what_transformers_computes(tiny_gguf):
    model, path = tiny_gguf
    engine = _engine(path)
    assert engine.source.kind == "gguf" and len(engine.spec.layers) == 4          # block_count 5 = 4 + MTP
    assert [d.kind for d in engine.spec.layers] == ["linear_attention", "full_attention"] * 2
    assert dict(engine.spec.layers[0].mixer.params)["v_head_order"] == "tiled"
    prompt = torch.randint(0, 248320, (60,), generator=torch.Generator().manual_seed(4)).tolist()
    with torch.no_grad():
        lp = model(torch.tensor([prompt])).logits[0, :-1].log_softmax(-1)
    want = lp.gather(-1, torch.tensor(prompt[1:])[:, None])[:, 0]
    torch.testing.assert_close(_logprobs(engine, prompt), want, atol=1e-4, rtol=1e-4)


def test_reading_tiled_heads_as_grouped_fails(tiny_gguf, monkeypatch):
    model, path = tiny_gguf
    from holon_plugins.models.qwen3_5 import recipe

    facts = recipe._facts
    monkeypatch.setattr(recipe, "_facts", lambda c, s: facts(c, s) | {"order": "grouped"})     # the plant
    prompt = torch.randint(0, 248320, (60,), generator=torch.Generator().manual_seed(4)).tolist()
    with torch.no_grad():
        lp = model(torch.tensor([prompt])).logits[0, :-1].log_softmax(-1)
    want = lp.gather(-1, torch.tensor(prompt[1:])[:, None])[:, 0]
    assert (_logprobs(_engine(path), prompt) - want).abs().max() > 1e-2
