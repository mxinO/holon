"""An fp32 golden of a Qwen3.5/3.8 checkpoint runs in fp32: transformers loads a multimodal checkpoint's text model in
its stored dtype whatever `dtype` asks for, so without the judge's cast a bf16 checkpoint's "fp32" golden ran in bf16.
On a tiny bf16 checkpoint in the real layout, the golden equals the engine in f32 (and differs from bf16)."""
import torch
from holon_judge import golden
from holon_judge.engine import score_model
from holon_testing.qwen3_5_files import save, tiny_hf

VOCAB = 512                                                   # the tiny prompts' vocabulary (conftest)


def test_an_fp32_golden_of_a_bf16_checkpoint_is_fp32(tmp_path, tiny_prompts):
    checkpoint = str(save(tiny_hf(6, seed=1, vocab=VOCAB).to(torch.bfloat16), tmp_path))
    gold = golden.record(checkpoint, tiny_prompts)
    f32 = score_model(checkpoint, tiny_prompts, torch.float32, "cpu", state_memory=8 << 20, max_seqs=4)
    bf16 = score_model(checkpoint, tiny_prompts, torch.bfloat16, "cpu", state_memory=8 << 20, max_seqs=4)
    assert (gold.target_lp - f32.target_lp).abs().max() < 1e-5 < (gold.target_lp - bf16.target_lp).abs().max()
