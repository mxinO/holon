"""A floor with a quantized KV cache (`kv.emulate`, M0.3 D7) computes what an engine with that cache computes: on a
tiny Qwen3.5 checkpoint in f32 on the CPU, the engine's q8_0 and fp8 caches agree with floors emulating them far more
closely than with the exact floor; and `compare` refuses a floor of another KV format."""
import pytest
import torch
from holon_judge import golden
from holon_judge import kv as kv_format
from holon_judge.compare import compare
from holon_judge.engine import score_model
from holon_std.mixers.gqa import reference as gqa
from holon_testing.qwen3_5_files import save, tiny_hf

VOCAB = 512                                                   # the tiny prompts' vocabulary (conftest)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    return str(save(tiny_hf(6, seed=1, vocab=VOCAB).to(torch.bfloat16), tmp_path_factory.mktemp("tiny_kv_floor")))


def test_the_floors_q8_0_is_the_engines():
    x = torch.randn(3, 4, 7, 256, generator=torch.Generator().manual_seed(0)) * 3
    torch.testing.assert_close(kv_format.stored(x, "q8_0"), gqa.dequantize_q8_0(*gqa.quantize_q8_0(x)), rtol=0, atol=0)


@pytest.mark.parametrize("fmt", ["q8_0", "fp8"])
def test_an_emulating_floor_is_the_quantized_engine(checkpoint, tiny_prompts, fmt):
    exact = golden.record(checkpoint, tiny_prompts)
    floor = golden.record(checkpoint, tiny_prompts, kv=fmt)
    cand = score_model(checkpoint, tiny_prompts, torch.float32, "cpu", state_memory=8 << 20, max_seqs=4, kv=fmt)
    assert floor.meta["kv"] == cand.meta["kv"] == fmt
    near, far = ((rec.target_lp - cand.target_lp).abs().mean() for rec in (floor, exact))
    assert near < 1e-6 and far > 1e-4                       # f32 noise vs the cache's rounding
    assert not any("KV format" in f for f in compare(exact, cand, floor).failures)
    assert any("KV format" in f for f in compare(exact, cand, exact).failures)
