"""The GGUF path of models/qwen3_5 against an oracle that does not share its beliefs: a tiny random checkpoint (bf16
weights, Qwen3.8's head ratio 3, an MTP block) converted by llama.cpp's own converter (`convert_hf_to_gguf.py`, f32),
then served by the engine on the CPU — it must compute what transformers computes on the checkpoint. The converter
writes norms as w + 1, `ssm_a = −exp(A_log)` and value heads tiled, and appends the MTP block as an extra layer."""
import json
import os
import pathlib
import subprocess
import sys

import pytest
import torch
from holon.api import SamplingParams
from holon.core.engine import Engine, EngineConfig
from safetensors.torch import save_file

from .test_qwen3_5_cpu import checkpoint, hf, tiny_config

ROOT = pathlib.Path(__file__).resolve().parents[2]
LLAMA = ROOT / "spikes" / "s2_gguf" / "third_party" / "llama.cpp"
TOKENIZER = sorted(pathlib.Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*"))

pytestmark = pytest.mark.skipif(not (LLAMA / "convert_hf_to_gguf.py").is_file() or not TOKENIZER,
                                reason="needs llama.cpp's converter and the Qwen3.8 tokenizer")


@pytest.fixture(scope="module")
def tiny_gguf(tmp_path_factory):
    torch.manual_seed(0)
    cfg = tiny_config(6)
    cfg.vocab_size = 248320                       # the converter takes the real tokenizer
    model = hf.Qwen3_5ForCausalLM(cfg).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(("norm.weight", "A_log", "dt_bias")):
                p.copy_(torch.randn_like(p) * 0.5)
    model = model.to(torch.bfloat16)              # bf16 values, so the converter's f32 w + 1 undoes exactly
    path = tmp_path_factory.mktemp("tiny_qwen3_5_gguf")
    tensors = dict(checkpoint(model))
    g = torch.Generator().manual_seed(3)
    rand = lambda *s: (torch.randn(*s, generator=g) * 0.05).bfloat16()          # noqa: E731
    h = cfg.hidden_size
    tensors |= {"mtp.fc.weight": rand(h, 2 * h), "mtp.norm.weight": rand(h),
                "mtp.pre_fc_norm_embedding.weight": rand(h), "mtp.pre_fc_norm_hidden.weight": rand(h)}
    tensors |= {"mtp.layers.0." + k.removeprefix("model.layers.1."): rand(*v.shape)       # a full-attention layer
                for k, v in model.state_dict().items() if k.startswith("model.layers.1.")}
    save_file({k: v.contiguous() for k, v in tensors.items()}, path / "model.safetensors")
    (path / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5", "tie_word_embeddings": False,
        "text_config": cfg.to_dict() | {"mtp_num_hidden_layers": 1}}))
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (path / name).write_bytes((TOKENIZER[0] / name).read_bytes())
    out = path / "tiny.gguf"
    subprocess.run([sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), str(path), "--outtype", "f32",
                    "--outfile", str(out)], env=os.environ | {"PYTHONPATH": str(LLAMA / "gguf-py")}, check=True,
                   capture_output=True)
    return model.float(), out


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
