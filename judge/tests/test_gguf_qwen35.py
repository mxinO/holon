"""The judge's own reading of a Qwen3.5/3.8 GGUF (`gguf.hf_state`): llama.cpp's converter writes zero-centred norms as
w + 1, `ssm_a = −exp(A_log)`, a squeezed conv1d and value heads tiled, and appends the MTP block; the judge must undo
all of it — the tiny converted file's HF state equals the checkpoint bit for bit, its gate is exact, and its golden
is the checkpoint's golden."""
import pytest
import torch
import transformers
from holon_judge import golden
from holon_judge import gguf as judge_gguf
from holon_judge.compare import compare
from holon_testing.qwen3_5_files import can_convert, convert, save, tiny_hf

pytestmark = pytest.mark.skipif(not can_convert(), reason="needs llama.cpp's converter and the Qwen3.8 tokenizer")


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    model = tiny_hf(6, vocab=248320).to(torch.bfloat16)
    path = save(model, tmp_path_factory.mktemp("tiny_qwen3_5"), mtp=True)
    return path, convert(path)


def test_the_judge_undoes_the_conversion_exactly(converted):
    checkpoint, file = converted
    ref = transformers.AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float32).state_dict()
    got = dict(judge_gguf.hf_state(file))
    assert set(got) == set(ref)                                        # the MTP block skipped, nothing missing
    # A_log comes back through the converter's f32 exp and our log: the checkpoint's bf16 value, after a bf16 cast.
    assert all(torch.equal(got[k], ref[k]) for k in ref if not k.endswith("A_log"))
    assert all(torch.equal(got[k].bfloat16(), ref[k].bfloat16()) for k in ref if k.endswith("A_log"))
    assert max(judge_gguf.gate(file, checkpoint).values()) < 1e-12


def test_the_gguf_golden_is_the_checkpoints(converted, tiny_prompts):
    checkpoint, file = converted
    a = golden.record(str(file), tiny_prompts, config=str(checkpoint))
    b = golden.record(str(checkpoint), tiny_prompts)
    assert compare(b, a).overall.kl_max < 1e-8           # only A_log's f32 ulp differs (near-uniform tiny model)
