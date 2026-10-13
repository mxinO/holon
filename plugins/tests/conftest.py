import gc

import pytest
import torch


@pytest.fixture(autouse=True)
def _free_gpu_memory():
    """Engines hold reference cycles (captured graphs, callbacks), so their GPU pools only return when the garbage
    collector runs; collect after every test that left GPU memory reserved, so the next engine can size its pools
    (collecting after every test cost ≈ 0.24 s each)."""
    yield
    if torch.cuda.is_available() and torch.cuda.memory_reserved():
        gc.collect()
        torch.cuda.empty_cache()
