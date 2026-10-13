"""The supervisor: restart and resubmission, hang detection, poison-request quarantine, rollback, restart time
(design §7.7)."""
import asyncio
import dataclasses
import multiprocessing
import pathlib
import time
import types

import pytest
from holon.api import SamplingParams, TokenDelta, sequence_seed
from holon.core.supervisor import Policy, Supervisor, _Request, supervised

from .test_engine import GREEDY, tiny_case, tiny_checkpoint  # noqa: F401 — fixtures
from .test_ipc import _config

MARKER = 511           # a request whose prompt starts with this token crashes the engine (`_poisoned_engine`)
FAST = Policy(ping_every=0.2, ping_timeout=30)


async def _collect(client, prompt, params, rid=None):
    out: dict[int, list[int]] = {}
    async for d in client.generate(prompt, params, rid):
        out.setdefault(d.index, []).extend(d.token_ids)
        last = d
    return out, last


def _patched(config, endpoints, step):
    """Run an engine whose step is `step(engine, original_step)`, counting steps in `engine.steps` (in the spawned
    child)."""
    from holon.core.engine import Engine
    from holon.core.ipc import serve

    def factory():
        engine = Engine(config)
        engine.steps, original = 0, engine.step

        def patched():
            engine.steps += 1
            return step(engine, original)
        engine.step = patched
        return engine
    serve(factory, endpoints)


def _slow_step(engine, step):
    time.sleep(0.02)                                  # so that a kill lands mid-stream
    return step()


def _slow_engine(config, endpoints):
    _patched(config, endpoints, _slow_step)


def _poisoned_step(engine, step):
    if any(seq.tokens[0] == MARKER for seqs in engine._live.values() for seq in seqs):
        raise RuntimeError("poisoned")
    return step()


def _poisoned_engine(config, endpoints):
    _patched(config, endpoints, _poisoned_step)


def _hanging_step(engine, step):
    flag = pathlib.Path(engine.cfg.model) / "hung"
    if engine.steps == 3 and not flag.exists():       # the first engine hangs; the restarted one serves
        flag.touch()
        time.sleep(3600)
    return step()


def _hanging_engine(config, endpoints):
    _patched(config, endpoints, _hanging_step)


def _broken_step(engine, step):
    if engine.cfg.max_seqs == 2 and engine.steps == 3:     # the newest config in the rollback test crashes
        raise RuntimeError("broken config")
    return step()


def _broken_engine(config, endpoints):
    _patched(config, endpoints, _broken_step)


def test_resubmission_continues_each_unfinished_sequence():
    params = SamplingParams(n=3, max_tokens=10, seed=5, prompt_logprobs=2)
    req = _Request("r", [1, 2], params)
    assert req.submissions() == [("r#1", [1, 2], params, None, 0)]       # nothing streamed: the request as it was
    req.out, req.done = {1: [9], 2: [7, 8]}, {1}
    subs = req.submissions()
    assert [(rid, tokens, i, resumed) for rid, tokens, _, i, resumed in subs] == [
        ("r#2.0", [1, 2], 0, 0), ("r#2.2", [1, 2, 7, 8], 2, 2)]           # its output, marked as such
    (_, _, p0, _, _), (_, _, p2, _, _) = subs
    assert (p0.n, p0.seed, p0.max_tokens, p0.prompt_logprobs) == (1, sequence_seed(5, 0), 10, 2)   # not sent yet
    assert (p2.n, p2.seed, p2.max_tokens, p2.prompt_logprobs) == (1, sequence_seed(5, 2), 10, None)


def test_a_failed_sequence_ends_its_siblings_engine_requests():
    # After a restart a request runs as one engine request per sequence: an error on one (a request error from the
    # logits processor) aborts the others at once, whether or not anyone reads the request's stream.
    aborted = []

    class Client:
        failure = None

        async def generate(self, tokens, params, rid, resume_output):
            try:
                if rid.endswith(".1"):
                    yield TokenDelta(rid, [], finish_reason="error", error="refused", kind="engine")
                    return
                await asyncio.sleep(60)
            finally:
                aborted.append(rid)

    async def run():
        sup = Supervisor([None])
        req = _Request("r", [1], SamplingParams(n=2, seed=1, max_tokens=10))
        req.out = {0: [5], 1: [6]}
        engine = types.SimpleNamespace(client=Client())
        for rid, *sub in req.submissions():
            req.pumps[rid] = sup._spawn(sup._pump(req, engine, rid, *sub))
        await asyncio.sleep(0.1)
        return req, sorted(aborted)                   # before the loop's end cancels what is left

    req, aborted_by_then = asyncio.run(run())
    assert aborted_by_then == ["r#1.0", "r#1.1"] and req.finished and req.queue.qsize() == 1


def test_a_killed_engine_restarts_and_streams_continue_exactly(tiny_checkpoint, tiny_case):  # noqa: F811
    # The request the kill interrupts is constrained by the toy logits processor from its first token on: after the
    # restart the processor must resume from the output (M0.5 §3.5), or its continuation differs.
    prompts, expected = tiny_case
    sampled = SamplingParams(temperature=1.0, top_p=0.9, max_tokens=12, ignore_eos=True, seed=7)
    requests = [(p, GREEDY) for p in prompts] + [(prompts[1], sampled),
                                                 (prompts[2], dataclasses.replace(sampled, n=2, seed=11))]
    config = dataclasses.replace(_config(tiny_checkpoint), slots={"logits_processor": "testing.toy_mask"})
    constrained = dataclasses.replace(GREEDY, max_tokens=30, ext={"toy.even_after@1": {"marker": expected[0][0]}})

    async def run(kill):
        async with supervised([config], FAST, target=_slow_engine) as sup:
            async def killer():
                stream = sup.generate(prompts[0], constrained)
                first = await anext(stream)
                if kill:
                    for proc in multiprocessing.active_children():
                        proc.kill()
                return first.token_ids + [t async for d in stream for t in d.token_ids]

            results = await asyncio.gather(*(_collect(sup, p, params) for p, params in requests), killer(),
                                           _collect(sup, list(range(40)), GREEDY))           # rejected: too long
            return results, await sup.stats()

    (*reference, long_reference, _), _ = asyncio.run(run(kill=False))
    (*results, long, rejected), stats = asyncio.run(asyncio.wait_for(run(kill=True), timeout=120))
    assert [r[0][0] for r in results[:4]] == expected
    assert [r[0] for r in results] == [r[0] for r in reference]        # greedy and seeded sampling, n = 2
    assert len(long) == 30 and long == long_reference and all(t % 2 == 0 for t in long[1:])
    assert rejected[1].finish_reason == "error" and "max_model_len" in rejected[1].error
    assert stats["crashes"] == 1 and stats["resubmitted"] >= 1 and stats["restarts"] == 1
    assert 0 < stats["restart_s"] < 60 and stats["start_s"] > 0 and stats["quarantined"] == stats["rollbacks"] == 0


def test_a_hung_engine_is_killed_and_its_requests_resubmitted(tiny_checkpoint, tiny_case):  # noqa: F811
    prompts, expected = tiny_case

    async def run():
        policy = dataclasses.replace(FAST, ping_timeout=1)
        async with supervised([_config(tiny_checkpoint)], policy, target=_hanging_engine) as sup:
            results = await asyncio.gather(*(_collect(sup, p, GREEDY) for p in prompts[:2]))
            return results, await sup.stats()

    flag = tiny_checkpoint / "hung"                   # written by the engine that hangs
    flag.unlink(missing_ok=True)
    try:
        results, stats = asyncio.run(asyncio.wait_for(run(), timeout=120))
    finally:
        flag.unlink(missing_ok=True)
    assert [r[0][0] for r in results] == expected[:2]
    assert stats["crashes"] == stats["restarts"] == 1 and stats["resubmitted"] == 2


def test_a_poison_request_is_quarantined_without_rollback(tiny_checkpoint, tiny_case):  # noqa: F811
    prompts, expected = tiny_case

    config = _config(tiny_checkpoint)

    async def run():
        async with supervised([config, dataclasses.replace(config)], FAST, target=_poisoned_engine) as sup:
            results = await asyncio.gather(_collect(sup, [MARKER] + prompts[0], GREEDY),
                                           *(_collect(sup, p, GREEDY) for p in prompts[1:]))
            after = await _collect(sup, prompts[0], GREEDY)                     # the engine still serves
            return results, after, await sup.stats()

    (poison, *others), after, stats = asyncio.run(asyncio.wait_for(run(), timeout=240))
    assert poison[1].finish_reason == "error" and "quarantined" in poison[1].error and "poisoned" in poison[1].error
    assert [r[0][0] for r in others] == expected[1:] and after[0][0] == expected[0]
    assert stats["quarantined"] == 1 and stats["rollbacks"] == 0 and stats["config"] == 1    # still the newest


def test_unexplained_crashes_roll_back_to_the_previous_config(tiny_checkpoint, tiny_case):  # noqa: F811
    prompts, expected = tiny_case
    good = _config(tiny_checkpoint)
    history = [good, dataclasses.replace(good, max_seqs=2)]                  # oldest first; the newest crashes
    policy = dataclasses.replace(FAST, rollback_crashes=2)

    async def run():
        async with supervised(history, policy, target=_broken_engine) as sup:
            results = await asyncio.gather(*(_collect(sup, p, GREEDY) for p in prompts[:2]))
            return results, await sup.stats()

    results, stats = asyncio.run(asyncio.wait_for(run(), timeout=240))
    assert [r[0][0] for r in results] == expected[:2]
    assert stats["rollbacks"] == 1 and stats["config"] == 0 and stats["quarantined"] == 0


def _unstartable_engine(config, endpoints):
    """The newest config in the rollback tests fails to start."""
    from holon.core.engine import Engine
    from holon.core.ipc import serve

    def factory():
        if config.max_seqs == 2:
            raise RuntimeError("unstartable config")
        return Engine(config)
    serve(factory, endpoints)


def test_failed_starts_roll_back_and_an_exhausted_history_gives_up(tiny_checkpoint, tiny_case):  # noqa: F811
    prompts, expected = tiny_case
    good = _config(tiny_checkpoint)
    bad = dataclasses.replace(good, max_seqs=2)

    async def run():
        async with supervised([good, bad], FAST, target=_unstartable_engine) as sup:
            result = await _collect(sup, prompts[0], GREEDY)
            stats = await sup.stats()
        with pytest.raises(RuntimeError, match="every configuration.*unstartable"):
            async with supervised([bad], FAST, target=_unstartable_engine):
                pass
        return result, stats

    result, stats = asyncio.run(asyncio.wait_for(run(), timeout=240))
    assert result[0][0] == expected[0] and stats["rollbacks"] == 1 and stats["config"] == 0


def test_a_lone_poison_request_on_a_quiet_server_is_quarantined(tiny_checkpoint, tiny_case):  # noqa: F811
    prompts, expected = tiny_case
    config = _config(tiny_checkpoint)

    async def run():
        async with supervised([config, dataclasses.replace(config)], FAST, target=_poisoned_engine) as sup:
            poison = await _collect(sup, [MARKER] + prompts[0], GREEDY)       # nothing else ever completed
            after = await _collect(sup, prompts[0], GREEDY)
            return poison, after, await sup.stats()

    poison, after, stats = asyncio.run(asyncio.wait_for(run(), timeout=240))
    assert poison[1].finish_reason == "error" and "quarantined" in poison[1].error
    assert after[0][0] == expected[0]
    assert stats["quarantined"] == 1 and stats["rollbacks"] == 0 and stats["config"] == 1 and stats["crashes"] == 3
