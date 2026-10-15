import gc
import os

# Expandable allocator segments from the session's first CUDA allocation, as launchers set them (the engine requires
# them, and its in-process fallback cannot convert segments earlier tests cached: an engine whose small pools landed
# there failed its check). Read when CUDA's allocator starts, so setting it at collection is early enough.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pytest  # noqa: E402
import torch  # noqa: E402


@pytest.fixture(autouse=True)
def _free_gpu_memory():
    """Engines hold reference cycles (captured graphs, callbacks), so their GPU pools only return when the garbage
    collector runs; collect after every test that left GPU memory reserved, so the next engine can size its pools
    (collecting after every test cost ≈ 0.24 s each)."""
    yield
    if torch.cuda.is_available() and torch.cuda.memory_reserved():
        gc.collect()
        torch.cuda.empty_cache()
