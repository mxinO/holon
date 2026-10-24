import gc
import os

# Expandable allocator segments from the session's first CUDA allocation, as launchers set them (the engine requires
# them, and its in-process fallback cannot convert segments earlier tests cached: an engine whose small pools landed
# there failed its check). Read when CUDA's allocator starts, so setting it at collection is early enough.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pathlib  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

QWEN3_0_6B = max(pathlib.Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*"), default=None)


@pytest.fixture(autouse=True)
def _free_gpu_memory():
    """Engines hold reference cycles (captured graphs, callbacks), so their GPU pools only return when the garbage
    collector runs; collect after every test that left GPU memory reserved, so the next engine can size its pools
    (collecting after every test cost ≈ 0.24 s each)."""
    yield
    if torch.cuda.is_available() and torch.cuda.memory_reserved():
        gc.collect()
        torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def qwen3_0_6b_engine():
    """One Qwen3-0.6B bf16 engine at the default settings (graphs, overlap, prefix cache), shared by the tests that only
    generate on it. Its pools are small, so the engines other tests build beside it still fit."""
    if QWEN3_0_6B is None or not torch.cuda.is_available():
        pytest.skip("needs Qwen3-0.6B + GPU")
    from holon.core.engine import Engine, EngineConfig

    engine = Engine(EngineConfig(model=str(QWEN3_0_6B), dtype=torch.bfloat16, max_model_len=1024,
                                 state_memory=256 << 20))
    yield engine
    engine.close()
