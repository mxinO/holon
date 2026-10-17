"""tiers.pagecache (docs/v1-offload.md acceptance 1: the held-out low-RAM tier): the store's file tier read through the
OS page cache. With RAM holding a quarter of the pieces (exclusive tiers, D10) every other piece is read by
`cpu_task@1`'s copy task from `pread` extents with O_DIRECT off, and the outputs are the resident store's, bit for bit
— under F11 (its victims demoted into RAM), on the CPU providers and on the GPU (the native copy task inside
decode graphs)."""
import pytest
import torch

from .test_moe_engine import KEY, Q, _engine, _gguf_and_dense, _gpu_engine, _served, gpu_moe, tiny_moe  # noqa: F401

TIERS = {"moe.store": "std.moe.offload@1", "moe.cache": "*=0.25", "moe.ram": "*=0.25",
         "moe.tiers": "std.tiered.pinned@1,tiers.pagecache@1"}


def _check(engine) -> None:
    """Every SSD piece read through the page cache: the tier's descriptors are buffered (`direct` 0) extents."""
    store = engine.ctx.shared_objects[KEY]
    assert store.exclusive and type(store.tiers[1]).__name__ == "PageCache"
    assert all((d[..., 1] == 0).all() and (d[..., 3] > 0).any() for d in store.descriptors.values())


def test_the_page_cache_tier_serves_what_the_resident_store_does(tiny_moe, tmp_path):  # noqa: F811
    checkpoint, prompts = tiny_moe
    file, _ = _gguf_and_dense(checkpoint, tmp_path, Q.Q8_0, types={"ffn_down_exps": Q.Q4_0})
    want = _served(_engine(file), prompts)
    engine = _engine(file, host_memory=1 << 24, plugin_options=TIERS)
    _check(engine)
    assert _served(engine, prompts) == want
    engine.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_the_page_cache_tier_on_the_gpu(gpu_moe):  # noqa: F811
    file, prompts, want = gpu_moe
    engine = _gpu_engine(file, host_memory=1 << 27, plugin_options=TIERS)
    _check(engine)
    assert _served(engine, prompts) == want and engine.stats()["graphs"] > 0
    io = engine.ctx.shared_objects["cpu_task.io"].native
    assert io.served() > 0 and io.flag() == 0
    engine.close()
