"""The engine behind a process boundary: concurrent streams, n > 1, rejections, aborts, stats."""
import asyncio

import torch
from holon.api import SamplingParams
from holon.core.engine import EngineConfig
from holon.core.ipc import engine_process

from .test_engine import GREEDY, tiny_case, tiny_checkpoint  # noqa: F401 — fixtures


def _config(path):
    return EngineConfig(model=str(path), device="cpu", dtype=torch.float32, page_size=4, max_seqs=3,
                        max_step_tokens=10, max_model_len=40, state_memory=1 << 20)


def test_engine_process_serves_concurrent_streams(tiny_checkpoint, tiny_case):  # noqa: F811
    prompts, expected = tiny_case

    async def run():
        async with engine_process(_config(tiny_checkpoint)) as client:
            async def collect(prompt, params):
                out: dict[int, list[int]] = {}
                async for d in client.generate(prompt, params):
                    out.setdefault(d.index, []).extend(d.token_ids)
                    last = d
                return out, last

            results = await asyncio.gather(*(collect(p, GREEDY) for p in prompts),
                                           collect(prompts[1], SamplingParams(temperature=0.0, max_tokens=12,
                                                                             ignore_eos=True, n=2)),
                                           collect(list(range(40)), GREEDY))                     # too long
            stream = client.generate(prompts[3], SamplingParams(temperature=0.0, max_tokens=30, ignore_eos=True))
            first = await anext(stream)
            await stream.aclose()                                                               # aborts it
            stats = await client.stats()
            return results, first, stats

    results, first, stats = asyncio.run(run())
    assert [r[0][0] for r in results[:4]] == expected
    assert results[4][0] == {0: expected[1], 1: expected[1]}
    assert results[5][1].finish_reason == "error" and "max_model_len" in results[5][1].error
    assert first.token_ids and stats["running"] + stats["waiting"] == 0
