import json
import pathlib

import pytest
import torch
from holon.core.engine import Engine, EngineConfig
from holon_judge.recording import PromptSet
from transformers import Qwen3Config, Qwen3ForCausalLM

VOCAB = 512


@pytest.fixture(scope="session")
def cpu_flags() -> dict:
    """EngineConfig fields of a CPU engine small enough for unit tests; the step budget chunks prefill."""
    return {"device": "cpu", "dtype": torch.float32, "page_size": 4, "max_seqs": 4, "max_model_len": 64,
            "max_step_tokens": 16, "state_memory": 1 << 20}


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
def tiny_prompts_file(tmp_path_factory) -> pathlib.Path:
    gen = torch.Generator().manual_seed(1)
    rows = [{"id": f"{cat}-{i}", "category": cat, "ids": torch.randint(0, VOCAB, (n,), generator=gen).tolist()}
            for i, (cat, n) in enumerate([("chat", 9), ("code", 23), ("chat", 5), ("agent", 30)])]
    path = tmp_path_factory.mktemp("prompts") / "tiny.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


@pytest.fixture(scope="session")
def tiny_prompts(tiny_prompts_file) -> PromptSet:
    return PromptSet.load(tiny_prompts_file)


@pytest.fixture
def tiny_engine(tiny_checkpoint, cpu_flags):
    def build(plugins=None, **kw) -> Engine:
        return Engine(EngineConfig(**{"model": str(tiny_checkpoint)} | cpu_flags | kw), plugins)
    return build
