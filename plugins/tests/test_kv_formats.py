"""KV formats (docs/m03-qwen38.md D6): the q8_0 reference against gguf-py's own Q8_0 (an independent oracle), the
fp8 cast's saturation, and the engine serving the tiny Qwen3.5 model with each format on the CPU — the quantized
caches close to the model-dtype one, which is exact."""
import gguf
import numpy as np
import pytest
import torch
from holon.api import SamplingParams
from holon.core.engine import Engine, EngineConfig
from holon_std.mixers.gqa import reference as gqa
from holon_testing.qwen3_5_files import save, tiny_hf


def test_q8_0_is_ggufs():
    x = torch.randn(64, 256, generator=torch.Generator().manual_seed(0)) * 3
    x[1] = 0                                                                  # an all-zero block
    values, scales = gqa.quantize_q8_0(x)
    blocks = gguf.quants.quantize(x.numpy(), gguf.GGMLQuantizationType.Q8_0).reshape(64, 8, 34)
    want_d = torch.from_numpy(blocks[..., :2].copy().view(np.float16)[..., 0])
    want_q = torch.from_numpy(blocks[..., 2:].copy().view(np.int8)).reshape(64, 256)
    assert torch.equal(scales, want_d) and torch.equal(values, want_q)
    torch.testing.assert_close(gqa.dequantize_q8_0(values, scales),
                               torch.from_numpy(gguf.quants.dequantize(blocks.reshape(64, -1),
                                                                       gguf.GGMLQuantizationType.Q8_0)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_q8_0_is_the_same_on_cuda():
    x = torch.randn(4096, 256, generator=torch.Generator().manual_seed(1)) * 3
    want, got = gqa.quantize_q8_0(x), gqa.quantize_q8_0(x.cuda())
    assert all(torch.equal(a, b.cpu()) for a, b in zip(want, got, strict=True))


def test_fp8_saturates():
    x = torch.tensor([1.0, -1000.0, 600.0, 0.0])
    assert gqa.to_fp8(x).float().tolist() == [1.0, -448.0, 448.0, 0.0]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    path = save(tiny_hf(6, seed=1).to(torch.bfloat16), tmp_path_factory.mktemp("tiny_kv"))
    prompt = torch.randint(0, 320, (90,), generator=torch.Generator().manual_seed(2)).tolist()
    return path, prompt


def _logprobs(path, prompt, fmt):
    engine = Engine(EngineConfig(model=str(path), device="cpu", dtype=torch.bfloat16, page_size=4, max_seqs=2,
                                 max_step_tokens=32, max_model_len=128, state_memory=8 << 20,
                                 plugin_options={} if fmt is None else {"kv.format": fmt}))
    assert engine.spec is not None and engine.ledger.pool("state.kv") is not None
    rid, got = engine.add(prompt, SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)), None
    while engine.has_work():
        got = next((d.prompt_logprobs for d in engine.step() if d.rid == rid and d.prompt_logprobs), got)
    return torch.tensor(got.token)


@pytest.mark.parametrize("fmt", ["q8_0", "fp8"])
def test_a_quantized_cache_serves_close_to_the_model_dtype(tiny, fmt):
    path, prompt = tiny
    ref, got = _logprobs(path, prompt, None), _logprobs(path, prompt, fmt)
    err = (got - ref).abs()
    assert 0 < err.max() < (0.05 if fmt == "q8_0" else 0.3)                  # changed, and only a little
