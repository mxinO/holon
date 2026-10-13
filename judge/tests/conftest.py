import json
import pathlib

import pytest
import torch
from holon_judge.recording import PromptSet
from transformers import Qwen3Config, Qwen3ForCausalLM

VOCAB = 512


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory) -> pathlib.Path:
    """A random two-layer Qwen3 with untied embeddings, small enough for CPU tests."""
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=False,
                      eos_token_id=None)
    path = tmp_path_factory.mktemp("tiny_qwen3")
    Qwen3ForCausalLM(cfg).save_pretrained(path)
    return path


@pytest.fixture(scope="session")
def tiny_prompts(tmp_path_factory) -> PromptSet:
    gen = torch.Generator().manual_seed(1)
    rows = [{"id": f"{cat}-{i}", "category": cat, "ids": torch.randint(0, VOCAB, (n,), generator=gen).tolist()}
            for i, (cat, n) in enumerate([("chat", 9), ("code", 23), ("chat", 5), ("agent", 30)])]
    path = tmp_path_factory.mktemp("prompts") / "tiny.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return PromptSet.load(path)


@pytest.fixture(scope="session")
def tiny_qwen3_5_gguf(tmp_path_factory) -> tuple[pathlib.Path, pathlib.Path]:
    """A tiny Qwen3.5 checkpoint with an MTP block (bf16 values: the converter's f32 w + 1 undoes exactly) and its GGUF
    by llama.cpp's own converter (f32): (checkpoint directory, .gguf). Converted once per session."""
    from holon_testing.qwen3_5_files import can_convert, convert, save, tiny_hf

    if not can_convert():
        pytest.skip("needs llama.cpp's converter and the Qwen3.8 tokenizer")
    path = save(tiny_hf(6, vocab=248320).to(torch.bfloat16), tmp_path_factory.mktemp("tiny_qwen3_5"), mtp=True)
    return path, convert(path)
