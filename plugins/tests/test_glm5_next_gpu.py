"""GLM-5.3-Flash's quantized tiny model on the GPU (docs/v1-glm.md acceptance 1(a′), §6.1's GPU row): the Q8_0 fixture
(H = I = 512, `llama-quantize`'s Q8_0, the head scaled to a logit std ≥ 2) through the GPU providers in bfloat16 —
mmvq / mmq, the fused router, the clamp, KDA, mHC, DSA — against the CPU engine on the same file in float32 (the
testing CPU providers), judged by KL over the top-20 (`holon_judge.compare.kl_topk`).

Each of the 3 prompts of 300 tokens runs on the GPU with prompt logprobs, chunked at 100 and at 250 tokens, then 32
greedy decode steps with their logprobs, eager and with CUDA graphs. The CPU engine scores the prompt and the GPU's own
32 tokens in one prompt-logprob pass, so every decode step is compared on the same context. KL mean and p99, over the
prompt positions and over the decode steps, stay within `BOUNDS`: the clean run's measured maxima × 3, written before
any plant ran (§6.1). §6.3's (a′) plants — kernel paths the CPU providers do not take — are pre-registered against
these bounds only where their measured effect is ≥ 10× a bound (§6.3); the others are measured and reported.

The tiny model is chaotic: its router (8 experts, top-2) and DSA selection flip on bf16 and q8_1 noise, so R's error
grows to ≈ 0.2 relative by the last block even on the CPU in bf16; the GPU's tracks the CPU-bf16 run's at ≈ 1.3–1.5×
per block (q8_1 activations), with no block standing out (step 4's diagnosis).

Needs the GPU fixture: `GLM_TINY_Q8_GGUF=<…/tiny-glm5next-gpu/tiny-glm5next-Q8_0.gguf>` (built by `python -m
holon_testing.glm5next_files --kind gpu` in the golden venv; its sha256 is pinned), else skipped.
"""
import hashlib
import os
import pathlib
import warnings

import pytest
import torch
from holon.api import SamplingParams
from holon.core.engine import Engine, EngineConfig
from holon_judge.compare import kl_topk
from holon_judge.recording import Recording, tail_from_topk
from holon_testing import glm5next_files as g

GGUF = os.environ.get("GLM_TINY_Q8_GGUF")
SKIP = ("GLM's GPU tiny-model check (§6.1, acceptance 1(a′)) did not run: set GLM_TINY_Q8_GGUF to the Q8_0 fixture "
        "that `python -m holon_testing.glm5next_files --kind gpu` builds in the golden venv (docs/release.md §22)")
if GGUF is None:
    warnings.warn(SKIP, stacklevel=1)
pytestmark = [pytest.mark.skipif(GGUF is None, reason=SKIP),
              pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")]
K, DECODE = 20, 32
# KL bounds (mean, p99) per part: the clean run's maxima over every configuration × 3 (§6.1), set before any plant ran.
BOUNDS = {"prompt": (0.040, 0.44), "decode": (0.035, 0.54)}    # measured (A5000): prompt ≤ 0.0132 / 0.146, decode
#                                                              ≤ 0.0116 / 0.180; graphs bitwise eager
# (a′)'s plants pre-registered against BOUNDS (measured effect ≥ 10× a bound, §6.3); the others are reported only.
# Measured at step 4 (A5000, KL mean / p99 against the bounds): router_prompt_softmax prompt 0.084 / 0.38 (2.1× the
# mean's bound); kda_gate_softplus decode 0.068 / 0.22 (1.9×); glu_clamp_off decode 0.017 / 0.21 (inside);
# dsa_decode_select_shift decode 0.011 / 0.089 (no effect: the tiny model's DSA output is small beside R). None
# reaches 10×: their L0s carry them (test_ggml_cuda_moe, test_kda_kernels), dsa_decode_select_shift the cover (step 5).
REGISTERED: tuple[str, ...] = ()


@pytest.fixture(scope="module")
def cpu():
    """The CPU engine (float32 on the same file), scoring a prompt and its continuation: top-20 rows per position."""
    path = pathlib.Path(GGUF)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == g.GGUF_SHA256["gpu"]
    engine = Engine(EngineConfig(str(path), device="cpu", dtype=torch.float32, page_size=16, max_seqs=1,
                                 max_step_tokens=512, max_model_len=512, state_memory=256 << 20))

    def score(ids: list[int]) -> tuple:
        lp = _run(engine, ids, SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True, prompt_logprobs=K))[1]
        return _rows(lp.top_ids, lp.top)
    yield score
    engine.close()


def _run(engine: Engine, ids: list[int], params: SamplingParams):
    """One request: its tokens, its prompt logprobs (merged over chunks), its decode logprobs (per token)."""
    rid, tokens, prompt, decode = engine.add(ids, params), [], [], []
    while engine.has_work():
        for d in engine.step():
            if d.rid != rid:
                continue
            tokens += d.token_ids
            if d.prompt_logprobs is not None:
                prompt.append(d.prompt_logprobs)
            if d.logprobs is not None:
                decode.append(d.logprobs)

    class Merged:
        top_ids = [x for p in prompt for x in p.top_ids]
        top = [x for p in prompt for x in p.top]
    return tokens, Merged, decode


def _rows(top_ids, top) -> tuple:
    ids = torch.tensor([x[:K] for x in top_ids], dtype=torch.int32)
    lp = torch.tensor([x[:K] for x in top], dtype=torch.float32)
    return ids, lp, lp[:, 0], tail_from_topk(lp)


def _kl(want: tuple, got: tuple) -> torch.Tensor:
    return kl_topk(Recording.from_rows([want], {}), Recording.from_rows([got], {}))[0]


def _gpu(chunk: int, graphs: bool) -> Engine:
    return Engine(EngineConfig(GGUF, device="cuda", dtype=torch.bfloat16, page_size=16, max_seqs=1,
                               max_step_tokens=chunk, max_model_len=512, cuda_graphs=graphs, memory_fraction=0.3))


def _measure(engine: Engine, cpu) -> dict:
    """KL of the GPU run against the CPU's scoring of the same tokens: (mean, p99) over prompt positions and over
    decode steps."""
    prompt, decode = [], []
    for ids in g.prompts("gpu"):
        got, lp, steps = _run(engine, ids, SamplingParams(temperature=0.0, max_tokens=DECODE, ignore_eos=True,
                                                          prompt_logprobs=K, logprobs=K))
        want, n = cpu(ids + got), len(ids) - 1
        prompt.append(_kl(tuple(r[:n] for r in want), _rows(lp.top_ids, lp.top)))
        decode.append(_kl(tuple(r[n:] for r in want), _rows([x for s in steps for x in s.top_ids],
                                                             [x for s in steps for x in s.top])))
    stat = lambda kl: (float(kl.mean()), float(kl.quantile(0.99)))       # noqa: E731
    return {"prompt": stat(torch.cat(prompt)), "decode": stat(torch.cat(decode))}


def _past(stats: dict) -> bool:
    return any(stats[p][0] > BOUNDS[p][0] or stats[p][1] > BOUNDS[p][1] for p in BOUNDS)


@pytest.mark.parametrize("graphs", [False, True], ids=["eager", "graphs"])
@pytest.mark.parametrize("chunk", [100, 250])
def test_the_gpu_matches_the_cpu_engine(cpu, chunk, graphs):
    engine = _gpu(chunk, graphs)
    stats = _measure(engine, cpu)
    engine.close()
    print(f"chunk {chunk} graphs {graphs}: {stats}")
    if BOUNDS["prompt"][0] is not None:
        assert not _past(stats), stats


@pytest.mark.parametrize("plant", ["dsa_decode_select_shift", "kda_gate_softplus", "router_prompt_softmax",
                                   "glu_clamp_off"])
def test_plants_move_the_kl(cpu, plant):
    """§6.3's (a′) plants on the GPU at chunk 100, eager: a pre-registered one moves the prompt or decode KL past its
    bound; the others are measured (printed) only."""
    from holon_judge.planted import PLANTS

    if BOUNDS["prompt"][0] is None:
        pytest.skip("BOUNDS not yet set from the clean run")
    with PLANTS[plant]():
        engine = _gpu(100, False)
        stats = _measure(engine, cpu)
    engine.close()
    print(f"plant {plant}: {stats}")
    if plant in REGISTERED:
        assert _past(stats), (plant, stats)
