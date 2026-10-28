"""GLM's deployments on the real file's first blocks (docs/dev/v1-glm.md D9, §13 steps 5-6): the store's layout and the
device's free memory at a deployment's --memory-fraction. In a module of their own: an engine another module keeps
alive (test_reserve's module-scoped fixtures) takes device memory these budgets are measured against. Each skips when
its file is not given (`holon_testing.gguf_files.slice_blocks` of Maya-S; docs/dev/release.md §22)."""
import os
from unittest import mock

import pytest
import torch
import triton
import triton.language as tl
from holon.api import SamplingParams
from holon.core import platform
from holon.core.engine import Engine, EngineConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
GLM_SLICE8 = os.environ.get("GLM_SLICE8_GGUF")        # the first 8 blocks (five MoE layers)
GLM_SLICE12 = os.environ.get("GLM_SLICE12_GGUF")      # the first 12 blocks (nine MoE layers: a tail can lend 32K)


def test_glm_a_loan_over_the_minimum_layout_is_refused():
    """GLM step 5 (an RTX 5080 served every greedy token as 0): a store at its minimum layout (staging only, no cache)
    keeps the step-peak loan in its own tail (`moe.lend_peak = on`); where the arena cannot hold the classes' staging
    below that tail the store refuses at build, naming it, instead of lending the classes' own slots. The first 8
    blocks of the real file on ≈ 6.3 GB of the card's memory (the A5000's case); a store that fits must serve the
    unlent run's tokens."""
    if GLM_SLICE8 is None:
        pytest.skip("GLM_SLICE8_GGUF not set")
    frac = 6.3e9 / torch.cuda.get_device_properties(0).total_memory
    ids = torch.randint(0, 150000, (3000,), generator=torch.Generator().manual_seed(0)).tolist()
    params = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)

    def served(lend: str) -> list[int]:
        opts = {"moe.store": "std.moe.offload@1", "moe.ring_lend": "tail", "moe.lend_peak": lend}
        e = Engine(EngineConfig(model=GLM_SLICE8, device="cuda", dtype=torch.bfloat16, page_size=16, max_seqs=1,
                                max_step_tokens=4096, max_model_len=8192, memory_fraction=frac,
                                host_memory=24 << 30, plugin_options=opts))
        try:
            return e.generate([ids], params)[0]
        finally:
            e.close()
    want = served("off")
    try:
        got = served("on")
    except RuntimeError as e:
        assert "own tail" in str(e) and "moe.lend_peak" in str(e)
        return
    assert got == want


@triton.jit
def _bump(x):                                      # a kernel no engine launches: its module loads at its first launch
    tl.store(x, tl.load(x) + 1)


@pytest.mark.parametrize("fraction, lend", [(0.99, "require"), (0.985, "require"), (0.99, "off")],
                         ids=["0.99", "0.985", "0.99-unlent"])
def test_glm_a_fragmenting_sequence_fits_at_the_locks_fraction(fraction, lend):
    """The lead's step-5 call on `--memory-fraction`: at the lock's fraction, with the deployment's lending
    (`moe.lend_peak`, `moe.loan_fixed_mib`), a sequence of steps that fragments the caching allocator — a long prompt,
    short prompts, mid-size prompts with prompt logprobs (the head's full-vocabulary chunks), 4 sequences decoding
    beside a prompt — runs without running out of memory, and the device keeps some memory free after every step.
    The first 12 blocks of the real file (a store whose tail can lend a 32K step's loan), at 32K steps. Unlent (GLM
    step 6: an RTX 5090 at 0.99 kept 3 MiB free after 32K steps, the caching allocator's segments past the allocated
    peak the profile measured), the rest of the fraction's free memory is left after every step — the profile counts
    the allocator's reserved growth — and a kernel whose module first loads then launches."""
    if GLM_SLICE12 is None:
        pytest.skip("GLM_SLICE12_GGUF not set")
    opts = {"moe.store": "std.moe.offload@1", "moe.ring_lend": "tail", "moe.lend_peak": lend,
            "moe.loan_fixed_mib": 1536}
    seen, free_memory = [], platform.free_memory
    with mock.patch.object(platform, "free_memory", lambda device: seen.append(free_memory(device)) or seen[-1]):
        e = Engine(EngineConfig(model=GLM_SLICE12, device="cuda", dtype=torch.bfloat16, page_size=16, max_seqs=4,
                                max_step_tokens=32768, max_model_len=33792, memory_fraction=fraction,
                                host_memory=48 << 30, plugin_options=opts))
    gen, least, headroom = torch.Generator().manual_seed(9), [], seen[-1] - int(seen[-1] * fraction)
    store = e.ctx.shared_objects["std.moe.offload"]                  # (the engine lets go of it at close)
    ids = lambda n: torch.randint(0, 150000, (n,), generator=gen).tolist()        # noqa: E731
    step = e.step

    def recorded():
        out = step()
        torch.cuda.synchronize()
        least.append(torch.cuda.mem_get_info()[0])
        return out
    e.step = recorded
    try:
        decode = SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True)
        plp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=20)
        e.generate([ids(24000)], decode)
        e.generate([ids(n) for n in (40, 300, 1500, 900)], decode)
        for n in (2048, 3000, 5451, 8000, 12000, 20000):
            e.generate([ids(n)], plp)
        for n in (40, 40, 40):
            e.add(ids(n), SamplingParams(temperature=0.0, max_tokens=200, ignore_eos=True))
        e.step()
        e.add(ids(6000), plp)
        while e.has_work():
            e.step()
        if lend == "off":
            _bump[(1,)](x := torch.zeros(1, dtype=torch.int32, device="cuda"))
            assert int(x) == 1
    finally:
        e.close()
    print(f"fraction {fraction}, lend {lend}: least free {min(least) / 2**20:.0f} MiB over {len(least)} steps")
    assert store.loan_overflows == 0 and min(least) >= 128 << 20    # no step overflowed its loan; a margin left
    if lend == "off":
        assert min(least) >= headroom - (64 << 20)       # (what modules loaded since took)
