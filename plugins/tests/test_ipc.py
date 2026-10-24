"""The engine behind a process boundary: concurrent streams, n > 1, rejections, aborts, stats, failures."""
import asyncio
import contextlib
import dataclasses
import multiprocessing
import pathlib
import time

import pytest
import torch
from holon.api import SamplingParams
from holon.core.engine import BLOCKED_POLL_S, EngineConfig
from holon.core.ipc import engine_process

from .test_engine import GREEDY, tiny_case, tiny_checkpoint  # noqa: F401 — fixtures


def _config(path):
    return EngineConfig(model=str(path), device="cpu", dtype=torch.float32, page_size=4, max_seqs=3,
                        max_step_tokens=10, max_model_len=40, state_memory=1 << 20)


def _closing_engine(config, endpoints, lock=None):
    """Engine-process target that records its engine's `close` (runs in the spawned child)."""
    from holon.core.engine import Engine
    from holon.core.ipc import serve

    def factory():
        engine = Engine(config)
        close = engine.close
        engine.close = lambda: (close(), pathlib.Path(config.plugin_options["tests.closed"]).touch())
        return engine
    serve(factory, endpoints)


@pytest.mark.slow
def test_engine_process_serves_concurrent_streams(tiny_checkpoint, tiny_case, tmp_path):  # noqa: F811
    """Concurrent streams, n > 1, a rejection, an abort, stats and a processor ready late; at shutdown the engine is
    closed, so the logits processor and shared objects stop their helpers (docs/v1-offload.md S0)."""
    prompts, expected = tiny_case
    marker = tmp_path / "closed"
    config = dataclasses.replace(_config(tiny_checkpoint), slots={"logits_processor": "testing.toy_mask"},
                                 plugin_options={"tests.closed": str(marker)})
    waits = SamplingParams(temperature=0.0, max_tokens=12, ignore_eos=True,      # its processor is ready at the 20th
                           ext={"toy.even_after@1": {"marker": -1, "polls": 19}})   # poll; it constrains nothing

    async def run():
        async with engine_process(config, target=_closing_engine) as client:
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
            start = time.monotonic()
            blocked = await collect(prompts[0], waits)           # on an idle engine: it polls, without spinning
            return results, first, await client.stats(), blocked, time.monotonic() - start

    results, first, stats, blocked, elapsed = asyncio.run(run())
    assert [r[0][0] for r in results[:4]] == expected
    assert results[4][0] == {0: expected[1], 1: expected[1]}
    assert results[5][1].finish_reason == "error" and "max_model_len" in results[5][1].error
    assert results[5][1].kind == "client"
    assert first.token_ids and stats["running"] + stats["waiting"] == 0
    assert blocked[0][0] == expected[0] and elapsed > 19 * BLOCKED_POLL_S / 2
    assert marker.exists()


def _failing_engine(config, endpoints, lock=None):
    """Engine-process target whose third step raises (runs in the spawned child)."""
    from holon.core.engine import Engine
    from holon.core.ipc import serve

    def factory():
        engine, calls = Engine(config), [0]
        step = engine.step

        def failing_step():
            calls[0] += 1
            if calls[0] == 3:
                raise RuntimeError("boom")
            return step()
        engine.step = failing_step
        return engine
    serve(factory, endpoints)


@pytest.mark.slow
def test_a_failing_step_ends_every_request_with_an_error(tiny_checkpoint, tiny_case):  # noqa: F811
    prompts, _ = tiny_case

    async def run():
        async with engine_process(_config(tiny_checkpoint), target=_failing_engine) as client:
            async def last(prompt):
                async for d in client.generate(prompt, SamplingParams(temperature=0.0, max_tokens=30,
                                                                      ignore_eos=True, n=2)):
                    final = d
                return final
            return await asyncio.gather(*(last(p) for p in prompts[:3]))

    finals = asyncio.run(run())
    assert all(d.finish_reason == "error" and "boom" in d.error for d in finals)


def test_a_killed_engine_ends_open_streams(tiny_checkpoint, tiny_case):  # noqa: F811
    prompts, _ = tiny_case

    async def run():
        async with engine_process(_config(tiny_checkpoint)) as client:
            stream = client.generate(prompts[3], SamplingParams(temperature=0.0, max_tokens=30, ignore_eos=True))
            await anext(stream)
            stats = await asyncio.gather(client.stats(), client.stats())          # concurrent stats both answer
            with contextlib.suppress(asyncio.TimeoutError):                        # a caller that gives up early
                await asyncio.wait_for(client.stats(), timeout=0)
            stats.append(await client.stats())                                     # the reader survived it
            first = client.generate([1, 2], SamplingParams(temperature=0.0, max_tokens=30, ignore_eos=True), rid="same")
            await anext(first)
            with pytest.raises(ValueError, match="in flight"):
                await anext(client.generate([1, 2], GREEDY, rid="same"))
            await first.aclose()
            for proc in multiprocessing.active_children():
                proc.kill()
            rest = [d async for d in stream]
            with pytest.raises(RuntimeError, match="exited"):
                await anext(client.generate([1, 2], GREEDY))
            return stats, rest

    stats, rest = asyncio.run(asyncio.wait_for(run(), timeout=60))
    assert len(stats) == 3 and rest[-1].finish_reason == "error" and "exited" in rest[-1].error
