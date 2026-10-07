import gc

import pytest
import torch


@pytest.fixture(autouse=True)
def _free_gpu_memory():
    """Engines hold reference cycles (captured graphs, callbacks), so their GPU pools only return when the garbage
    collector runs; collect after every test so the next engine can size its pools."""
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
