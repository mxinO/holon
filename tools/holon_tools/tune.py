"""`tune`: the fastest installed kernel provider per op on this GPU class, measured by a short decode benchmark
(design §4.4: tuned defaults live in the lock). The winners become kernel pins; the lock they produce goes through
`propagate`'s checks like any other change.

Ops are tuned one at a time (coordinate descent), the others held at their current pins. Only the decode path's ops
are tuned: a decode benchmark says nothing about prefill kernels.
"""
from __future__ import annotations

import gc
import random
import statistics
import time
from collections.abc import Callable, Mapping

from holon.api import SamplingParams, parse_versioned
from holon.core.engine import Engine, EngineConfig

OPS = ("attention.paged_decode", "rmsnorm", "add_rmsnorm", "rope", "silu_mul")


def candidates(engine: Engine, ops: tuple[str, ...] = OPS) -> dict[str, list[str]]:
    """For each of `ops` the engine binds: the providers that run here and support every site of the op."""
    found: dict[str, set[str]] = {}
    for key, bound in engine.ctx.kernels.items():
        op = key.rpartition("/")[2]
        name = parse_versioned(op)[0]
        if name in ops:
            ok = {s.provider for s in engine.ctx.candidates(op, bound["facts"])}
            found[name] = found.get(name, ok) & ok
    return {name: sorted(providers) for name, providers in found.items()}


def tune(pins: Mapping[str, str], ops: Mapping[str, list[str]], rate: Callable[[dict[str, str]], float],
         margin: float = 0.03) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    """The pins after giving each op its fastest provider by `rate(pins)` (higher is better), and every rate. The
    pinned provider stays unless beaten by more than `margin`, so re-tuning does not flip pins on noise."""
    best: dict[str, str] = dict(pins)
    rates: dict[str, dict[str, float]] = {}
    measured: dict[tuple, float] = {}           # one op's winner is the next op's incumbent: measure it once
    for op, providers in ops.items():
        if len(providers) < 2:
            continue
        for provider in providers:
            trial = best | {op: provider}
            key = tuple(sorted(trial.items()))
            if key not in measured:
                measured[key] = rate(trial)
            rates.setdefault(op, {})[provider] = measured[key]
        top = max(rates[op], key=rates[op].__getitem__)
        if rates[op][top] > rates[op].get(best.get(op, ""), 0.0) * (1 + margin):
            best[op] = top
    return best, rates


def decode_rate(config: EngineConfig, batch: int = 8, prompt_len: int = 32, tokens: int = 64,
                repeats: int = 3) -> float:
    """Generated tokens/s for `batch` random prompts, the median of `repeats` runs after a warm-up (repeats hit the
    prefix cache, so the time is decode)."""
    gc.collect()                                  # free the previous candidate's engine first
    engine = Engine(config)
    gen = random.Random(0)
    prompts = [[gen.randrange(engine.spec.vocab) for _ in range(prompt_len)] for _ in range(batch)]
    params = SamplingParams(temperature=0.0, max_tokens=tokens, ignore_eos=True)
    engine.generate(prompts, params)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        engine.generate(prompts, params)
        times.append(time.perf_counter() - t0)
    return batch * tokens / statistics.median(times)
