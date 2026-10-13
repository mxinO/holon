"""L2 candidate side: scoring through the engine interface (a fake backed by HF, and the real engine once it serves
prompt logprobs)."""
import collections
import types

import pytest
import torch
from holon.api import Logprobs, SamplingParams, TokenDelta
from holon.core.engine import Engine, EngineConfig
from holon_judge import golden
from holon_judge.compare import compare
from holon_judge.engine import score
from transformers import AutoModelForCausalLM


class FakeEngine:
    """`add`/`step`/`has_work` like `holon.core.Engine`, answering prompt logprobs from an HF model; a prompt's
    logprobs arrive split over two steps, as with chunked prefill. Its model counts calls as kernels.ggml_cuda's
    instances do, where std's units keep them: a linear's quantized weight's kernel (`_q._kernel`) and an expert
    stack's (`_kernel`), each step one call per path and mode."""

    def __init__(self, path, drop_last: bool = False) -> None:
        self.hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
        self.queue: list[tuple[str, list[int], SamplingParams]] = []
        self.drop_last = drop_last
        kernel = lambda: types.SimpleNamespace(calls=collections.Counter())          # noqa: E731
        linear, experts = torch.nn.Module(), torch.nn.Module()
        linear._q, experts._kernel = types.SimpleNamespace(_kernel=kernel()), kernel()
        self.executor = types.SimpleNamespace(model=torch.nn.ModuleList([linear, experts]))
        self.counters = {"mmq/fp4": linear._q._kernel.calls, "mmvq/q8_1": experts._kernel.calls}

    def add(self, token_ids: list[int], params: SamplingParams) -> str:
        rid = f"r{len(self.queue)}"
        self.queue.append((rid, token_ids, params))
        return rid

    def has_work(self) -> bool:
        return bool(self.queue)

    @torch.no_grad()
    def step(self) -> list[TokenDelta]:
        for mode, counter in self.counters.items():
            counter[mode] += 1
        rid, ids, params = self.queue.pop(0)
        lp = self.hf(torch.tensor([ids])).logits[0, :-1].float().log_softmax(-1)
        top, top_ids = lp.topk(params.prompt_logprobs, -1)
        token = lp.gather(1, torch.tensor(ids[1:])[:, None])[:, 0]
        n = len(token) - self.drop_last
        cut = n // 2

        def part(a: int, b: int) -> Logprobs:
            return Logprobs(token[a:b].tolist(), top_ids[a:b].tolist(), top[a:b].tolist())

        return [TokenDelta(rid, [], prompt_logprobs=part(0, cut)),
                TokenDelta(rid, [int(lp[-1].argmax())], prompt_logprobs=part(cut, n), finish_reason="length")]


@pytest.fixture(scope="module")
def gold(tiny_checkpoint, tiny_prompts):
    return golden.record(str(tiny_checkpoint), tiny_prompts)


def test_fake_engine_scores_like_the_golden(tiny_checkpoint, tiny_prompts, gold):
    engine = FakeEngine(tiny_checkpoint)
    engine.counters["mmq/fp4"]["mmq/fp4"] = 5                              # start-up calls: not the scoring's
    cand = score(engine, tiny_prompts, meta={"dtype": "float32"})
    assert torch.equal(cand.offsets, gold.offsets) and cand.meta["source"] == "holon"
    assert cand.meta["calls"] == {"mmq/fp4": 4, "mmvq/q8_1": 4}            # dense and expert kernels, by mode
    r = compare(gold, cand)
    assert r.overall.top1 == 1 and r.overall.kl_p99 < 1e-6 and r.overall.missing == 0
    assert abs(r.overall.dppl) < 1e-6


def test_wrong_number_of_positions_is_an_error(tiny_checkpoint, tiny_prompts):
    with pytest.raises(ValueError, match="prompt logprobs at"):
        score(FakeEngine(tiny_checkpoint, drop_last=True), tiny_prompts)


def test_real_engine_scores_like_the_golden(tiny_checkpoint, tiny_prompts, gold):
    engine = Engine(EngineConfig(model=str(tiny_checkpoint), device="cpu", dtype=torch.float32, page_size=4,
                                 max_seqs=4, max_model_len=64, max_step_tokens=16, state_memory=1 << 20))
    try:
        cand = score(engine, tiny_prompts)
    except NotImplementedError as e:
        pytest.skip(f"engine does not serve prompt logprobs yet (E3): {e}")
    r = compare(gold, cand)
    assert r.overall.top1 > 0.99 and r.overall.kl_p99 < 1e-5
