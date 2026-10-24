import gc
import importlib
import os

# Expandable allocator segments from the session's first CUDA allocation, as launchers set them (the engine requires
# them, and its in-process fallback cannot convert segments earlier tests cached: an engine whose small pools landed
# there failed its check). Read when CUDA's allocator starts, so setting it at collection is early enough.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pathlib  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

QWEN3_0_6B = max(pathlib.Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*"), default=None)


@pytest.fixture(scope="module", autouse=True)
def _prebuilt(request):
    """The engine never compiles (a missing prebuilt module raises NotBuilt naming `holon prebuild`), so a test module
    lists the CUDA extensions it needs in `PREBUILT` (e.g. `("expert_io",)`) and any missing one is built here for
    the visible GPU's SM, into the user cache, before its tests run: a fresh cache does not fail them."""
    names = getattr(request.module, "PREBUILT", ())
    if not names or not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability()
    sm = 10 * major + minor
    for name in names:
        ext = importlib.import_module(f"holon_plugins.kernels.{name}.extension")
        try:
            ext.load(sm)
        except ext.NotBuilt:
            ext.build(sm, os.cpu_count()) if name == "ggml_cuda" else ext.build(sm)


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
