"""The startup profile's activation reserve covers every step the engine then runs (FN §22.9: a prompt step of two
prompts' chunks once needed 2.7× the GDN temporaries of the one-row step the profile measures, and on an RTX 5090 at
--memory-fraction 0.985 the overrun showed up as an illegal address in another kernel, not as an out-of-memory error).

A tiny Qwen3.5 (GDN and attention layers, GDN wide enough that its temporaries dominate a step) on the GPU kernels,
eager steps, with fla's sub-chunks at one FLA chunk so a step splits: after each step, the peak it allocated beyond
what was allocated before it must be at most the reserve the profile measured (and the margin below), across step
shapes — one long prompt (full chunks and a tail), several prompts' rows in one step, prefill beside decode rows, a
1-token tail. Before d2c759c8 the several-rows step took 54.0 MB against a 32.0 MB reserve.

One term grows with a step's rows and is bounded, so it is the stated margin: GDN's (and KDA's) in-step conv
checkpoint takes each checkpointed row's last `kernel − 1` inputs (`(kernel − 1) × channels` elements) — one row's in
the profiled step, at most `max_seqs` rows' in any step, one layer's at a time. Test-only: serving pays nothing for it.

GLM (docs/dev/v1-glm.md acceptance 3, D9) from its first kernel: the tiny Q8_0 fixture (`GLM_TINY_Q8_GGUF`) over the
same step shapes with KDA past `SUB_TOKENS` and the mixers' sub-batching on (`rows` patched to 256); and the real
model's first 4 blocks (`GLM_SLICE_GGUF`: KDA ×3 with the dense FFN, DSA with the routed experts at their real types,
`holon_testing.gguf_files.slice_blocks` of Maya-S) at a 16K and a 32K prompt step with the recipe's `rows`: at most
150 KiB of live activations a token, R included, outside the routed experts' lent passes. Each skips with a warning
when its file is not given."""
import os
import warnings

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
    """GDN's and KDA's conv checkpoint windows of `max_seqs − 1` rows more than the profiled step's (the module
    docstring); the windows are held in the model dtype."""
    from holon_std.mixers.gdn.units import GDN, KDA
    size = torch.empty(0, dtype=engine.cfg.dtype).element_size()
    window = max((m.kernel - 1) * m.channels * size for m in engine.executor.model.modules()
                 if isinstance(m, (GDN, KDA)))
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


GLM_TINY, GLM_SLICE = os.environ.get("GLM_TINY_Q8_GGUF"), os.environ.get("GLM_SLICE_GGUF")

for name, value in (("GLM_TINY_Q8_GGUF", GLM_TINY), ("GLM_SLICE_GGUF", GLM_SLICE)):
    if value is None:
        warnings.warn(f"test_reserve's GLM case did not run: set {name} (the module docstring)", stacklevel=1)


@pytest.fixture(scope="module")
def glm(tmp_path_factory):
    from holon_plugins.kernels.fla import plugin as fla
    from holon_plugins.models.glm5_next import recipe

    if GLM_TINY is None:
        pytest.skip("GLM_TINY_Q8_GGUF not set")
    peaks, measure, sub, rows = [], platform.peak_memory, fla.SUB_TOKENS, (recipe.ROWS, recipe.DSA_ROWS)

    def recorded(device, fn):
        peaks.append(measure(device, fn))
        return peaks[-1]
    platform.peak_memory, fla.SUB_TOKENS, (recipe.ROWS, recipe.DSA_ROWS) = recorded, 64, (256, 256)
    try:
        e = Engine(EngineConfig(model=GLM_TINY, device="cuda", dtype=torch.bfloat16, cuda_graphs=False, page_size=16,
                                max_seqs=4, max_step_tokens=STEP, max_model_len=4096, chunk_with_decode=384,
                                memory_fraction=0.1))
        assert e.executor.model.layers[0].mixer.rows == e.executor.model.layers[3].mixer.rows == 256
        yield e, peaks[-1]
        e.close()
    finally:
        platform.peak_memory, fla.SUB_TOKENS, (recipe.ROWS, recipe.DSA_ROWS) = measure, sub, rows


@pytest.mark.parametrize("prompts", [[2400], [400, 400, 400, 400], [STEP + 1]],
                         ids=["one-row", "several-rows", "tail"])
def test_glm_steps_fit_the_profiled_reserve(glm, prompts):
    e, reserve = glm
    _steps(e, prompts, reserve)


def test_glm_prefill_beside_decode_rows_fits_the_profiled_reserve(glm):
    e, reserve = glm
    gen = torch.Generator().manual_seed(7)
    params = SamplingParams(temperature=0.0, max_tokens=24, ignore_eos=True)
    for n in (40, 40, 40):
        e.add(torch.randint(0, 320, (n,), generator=gen).tolist(), params)
    e.step()
    e.add(torch.randint(0, 320, (2000,), generator=gen).tolist(), params)
    assert max(_steps(e, [], reserve)) > 0


@pytest.mark.parametrize("tokens", [16384, 32768], ids=["16K", "32K"])
def test_glm_live_activations_a_token(tokens):
    """Acceptance 3: a sub-batched prompt step of the real model's shapes (the first 4 blocks: three KDA blocks with the
    dense FFN, a DSA block with the routed and shared experts) holds at most 150 KiB of live activations a token, R
    included (N4: ≈ 119 / 113 KiB on the A5000), and fits the profiled reserve. The routed experts' passes run on
    buffers the deployment's store lends (D9: `moe.lend_peak`), not on the activations' reserve, so the step's peak is
    taken outside the routed experts' calls (a 4-block store is too small to lend: here they allocate)."""
    from holon_std.moe.units import MoE

    if GLM_SLICE is None:
        pytest.skip("GLM_SLICE_GGUF not set")
    peaks, measure = [], platform.peak_memory

    def recorded(device, fn):
        peaks.append(measure(device, fn))
        return peaks[-1]
    platform.peak_memory = recorded
    try:
        e = Engine(EngineConfig(model=GLM_SLICE, device="cuda", dtype=torch.bfloat16, cuda_graphs=False, page_size=16,
                                max_seqs=1, max_step_tokens=tokens, max_model_len=tokens + 64, memory_fraction=0.95))
    finally:
        platform.peak_memory = measure
    outside = [0]
    for moe in (m for m in e.executor.model.modules() if isinstance(m, MoE)):
        for name in ("forward", "add_residual"):             # (std.ffn.sum calls the routed part's `add_residual`)
            call = getattr(moe, name)

            def routed(*args, call=call, **kw):              # the peak so far, then the routed experts' own
                torch.cuda.synchronize()
                outside[0] = max(outside[0], torch.cuda.max_memory_allocated())
                out = call(*args, **kw)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                return out
            setattr(moe, name, routed)
    try:
        e.add(torch.randint(0, 150000, (tokens,), generator=torch.Generator().manual_seed(3)).tolist(),
              SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        e.step()
        torch.cuda.synchronize()
        total = torch.cuda.max_memory_allocated() - before
        live = max(outside[0], torch.cuda.max_memory_allocated()) - before
        print(f"{tokens} tokens: {live / tokens / 1024:.1f} KiB a token outside the routed experts "
              f"(reserve {peaks[-1] / 2**30:.2f} GiB)")
        assert live <= 150 * 1024 * tokens
        assert total <= peaks[-1]
    finally:
        e.close()


def test_glm_prompt_passes_are_16384_rows():
    """D9: under GLM's `ggml_cuda.pass_rows` = 16,384 (sm_86's and sm_120's proven bound for its expert types) a 32K
    prompt step runs every routed stack in passes of 16,384 rows: two mmq calls a stack, the router GEMM's pass too."""
    from holon_plugins.kernels.ggml_cuda.plugin import LinearExperts, MoeRouter

    if GLM_SLICE is None:
        pytest.skip("GLM_SLICE_GGUF not set")
    e = Engine(EngineConfig(model=GLM_SLICE, device="cuda", dtype=torch.bfloat16, cuda_graphs=False, page_size=16,
                            max_seqs=1, max_step_tokens=32768, max_model_len=32768 + 64, memory_fraction=0.95,
                            plugin_options={"ggml_cuda.pass_rows": "16384"}))
    try:
        stacks, routers = _instances(e, LinearExperts), _instances(e, MoeRouter)
        assert stacks and routers and all(k.pass_rows == 16384 for k in stacks + routers)
        before = [sum(n for key, n in k.calls.items() if key.startswith("mmq/")) for k in stacks]
        e.add(torch.randint(0, 150000, (32768,), generator=torch.Generator().manual_seed(4)).tolist(),
              SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True))
        e.step()
        after = [sum(n for key, n in k.calls.items() if key.startswith("mmq/")) for k in stacks]
        assert [a - b for a, b in zip(after, before, strict=True)] == [2] * len(stacks)
    finally:
        e.close()


def _instances(engine, cls) -> list:
    """The engine's kernel instances of `cls`: those its modules hold, each once."""
    found = {id(v): v for m in engine.executor.model.modules() for v in vars(m).values() if isinstance(v, cls)}
    return list(found.values())
