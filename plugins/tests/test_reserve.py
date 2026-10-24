"""The startup profile's activation reserve covers every step the engine then runs (FN §22.9: a prompt step of two
prompts' chunks once needed 2.7× the GDN temporaries of the one-row step the profile measures, and on an RTX 5090 at
--memory-fraction 0.985 the overrun showed up as an illegal address in another kernel, not as an out-of-memory error).

A tiny Qwen3.5 (GDN and attention layers, GDN wide enough that its temporaries dominate a step) on the GPU kernels,
eager steps, with fla's sub-chunks at one FLA chunk so a step splits: after each step, the peak it allocated beyond
what was allocated before it must be at most the reserve the profile measured (and the margin below), across step
shapes — one long prompt (full chunks and a tail), several prompts' rows in one step, prefill beside decode rows, a
1-token tail. Before d2c759c8 the several-rows step took 54.0 MB against a 32.0 MB reserve.

One term grows with a step's rows and is bounded, so it is the stated margin: GDN's in-step conv checkpoint takes each
checkpointed row's last `kernel − 1` inputs (`(kernel − 1) × channels` elements) — one row's in the profiled step,
at most `max_seqs` rows' in any step, one layer's at a time. Test-only: serving pays nothing for it."""
import pytest
import torch
from holon.api import SamplingParams
from holon.core import platform
from holon.core.engine import Engine, EngineConfig
from holon_testing.qwen3_5_files import save, tiny_hf

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
STEP, HEADS, DIM = 1024, 16, 128


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    from holon_plugins.kernels.fla import plugin as fla
    path = save(tiny_hf(HEADS, seed=1, dim=DIM), tmp_path_factory.mktemp("tiny_qwen3_5_reserve"))
    peaks, measure, sub = [], platform.peak_memory, fla.SUB_TOKENS

    def recorded(device, fn):
        peaks.append(measure(device, fn))
        return peaks[-1]
    platform.peak_memory, fla.SUB_TOKENS = recorded, 64
    try:
        e = Engine(EngineConfig(model=str(path), device="cuda", dtype=torch.bfloat16, cuda_graphs=False, page_size=16,
                                max_seqs=4, max_step_tokens=STEP, max_model_len=4096, chunk_with_decode=384,
                                memory_fraction=0.05))
        yield e, peaks[-1]
        e.close()
    finally:
        platform.peak_memory, fla.SUB_TOKENS = measure, sub


def _margin(engine) -> int:
    """GDN's conv checkpoint windows of `max_seqs − 1` rows more than the profiled step's (the module docstring)."""
    from holon_std.mixers.gdn.units import GDN
    gdns = [m for m in engine.executor.model.modules() if isinstance(m, GDN)]
    window = max((m.kernel - 1) * m.channels * m.conv_weight.element_size() for m in gdns)
    return (engine.cfg.max_seqs - 1) * window


def _steps(engine, prompts: list[int], reserve: int) -> list[int]:
    """Serve `prompts` (lengths) to the end; every step's peak against the reserve and the stated margin."""
    reserve += _margin(engine)
    gen = torch.Generator().manual_seed(len(prompts))
    params = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)
    for n in prompts:
        engine.add(torch.randint(0, 320, (n,), generator=gen).tolist(), params)
    peaks = []
    while engine.has_work():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        engine.step()
        torch.cuda.synchronize()
        peaks.append(torch.cuda.max_memory_allocated() - before)
        assert peaks[-1] <= reserve, f"a step of {prompts} took {peaks[-1]} bytes, past the reserve {reserve}"
    return peaks


@pytest.mark.parametrize("prompts", [
    [2400],                   # one row: full steps, then a tail
    [400, 400, 400, 400],     # several prompts' rows in one step
    [STEP + 1],               # a 1-token tail
], ids=["one-row", "several-rows", "tail"])
def test_every_step_fits_the_profiled_reserve(engine, prompts):
    e, reserve = engine
    _steps(e, prompts, reserve)


def test_prefill_beside_decode_rows_fits_the_profiled_reserve(engine):
    e, reserve = engine
    gen = torch.Generator().manual_seed(7)
    params = SamplingParams(temperature=0.0, max_tokens=24, ignore_eos=True)
    for n in (40, 40, 40):                 # three decoding rows …
        e.add(torch.randint(0, 320, (n,), generator=gen).tolist(), params)
    e.step()
    e.add(torch.randint(0, 320, (2000,), generator=gen).tolist(), params)  # … then a prompt's chunks beside them
    assert max(_steps(e, [], reserve)) > 0
