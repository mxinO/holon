"""`tune`: the fastest installed kernel provider per op on this GPU class, measured by a short decode benchmark
(design §4.4: tuned defaults live in the lock). The winners become kernel pins; the lock they produce goes through
`propagate`'s checks like any other change.

Ops are tuned one at a time (coordinate descent), the others held at their current pins. Only the decode path's ops
are tuned: a decode benchmark says nothing about prefill kernels.

`split` decides F5b, the CPU's share of an offloaded MoE's misses (docs/v1-offload.md D4, §8), by the same decode
benchmark: the CPU worker's thread count at half the misses, then the share at that count (M2 found plan[n] ≈ n / 2 and
flat within ±1, so a share of the misses per miss count is the table); the lock takes the best only if it gains at
least `GAIN` over F5a, with the CPU tier it was measured on.

`clock` fits a tier's `state_tier.step_cost` (docs/v1-kv-tiers.md D4: a + b·tokens + c·rows seconds per step) to timed
steps of this lock's engine: prefill steps of one row at several chunk sizes and decode steps at every row count.
"""
from __future__ import annotations

import gc
import random
import statistics
import time
from collections.abc import Callable, Mapping

import numpy as np
from holon.api import SamplingParams, parse_versioned
from holon.core.engine import Engine, EngineConfig

OPS = ("attention.paged_decode", "rmsnorm", "add_rmsnorm", "rope", "silu_mul")
GAIN = 0.05                                    # D4: F5b enters a lock only at a ≥ 5% decode gain
THREADS = (2, 4, 6)                            # the compute worker's threads (S6, M3: DRAM saturates at 2–4)
SHARES = (0.25, 0.4, 0.5, 0.6, 0.75, 1.0)      # the CPU's share of a pass's misses


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


def plan(share: float, n: int) -> str:
    """`moe.cpu_plan` for a share of the misses: plan[m] = m · share rounded half up, m = 1 … n."""
    return ",".join(str(min(m, int(m * share + 0.5))) for m in range(1, n + 1))


def split(rate: Callable[[dict[str, str]], float], n: int, isa: str, threads: tuple[int, ...] = THREADS,
          shares: tuple[float, ...] = SHARES) -> tuple[dict[str, str], dict]:
    """F5b by measurement: `rate(options)` (higher is better) for F5a (no options), for each thread count at half the
    misses, then for each share at the best count; the plugin options of the best when it gains at least `GAIN` over
    F5a (else none: the lock stays F5a), and every rate. `n`: the most misses of a pass the table spells out (past it
    the last entry's share); `isa`: this host's CPU tier."""
    def options(share: float, t: int) -> dict[str, str]:
        return {"moe.cpu_plan": plan(share, n), "moe.cpu_isa": isa, "cpu_task.threads": str(t)}
    f5a = rate({"moe.cpu_plan": ""})                 # F5a whatever the engine flags say
    by_threads = {t: rate(options(0.5, t)) for t in threads}
    t = max(by_threads, key=by_threads.__getitem__)
    by_share = {s: by_threads[t] if s == 0.5 else rate(options(s, t)) for s in shares}
    best = max(by_share, key=by_share.__getitem__)
    gain = by_share[best] / f5a - 1
    record = {"f5a": f5a, "threads_at_half": by_threads, "shares": by_share, "best": {"share": best, "threads": t},
              "gain": gain, "taken": gain >= GAIN}
    return (options(best, t) if gain >= GAIN else {}), record


def decode_rate(config: EngineConfig, batch: int = 8, prompt_len: int = 32, tokens: int = 64,
                repeats: int = 3, prompts: list[list[int]] | None = None) -> float:
    """Generated tokens/s for `batch` prompts (`prompts`' first, else random ones of `prompt_len` tokens), the median of
    `repeats` runs after a warm-up (repeats hit the prefix cache, so the time is decode)."""
    gc.collect()                                  # free the previous candidate's engine first
    engine = Engine(config)
    gen = random.Random(0)
    prompts = prompts[:batch] if prompts else [[gen.randrange(engine.spec.vocab) for _ in range(prompt_len)]
                                               for _ in range(batch)]
    params = SamplingParams(temperature=0.0, max_tokens=tokens, ignore_eos=True)
    engine.generate(prompts, params)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        engine.generate(prompts, params)
        times.append(time.perf_counter() - t0)
    return batch * tokens / statistics.median(times)


def step_times(config: EngineConfig, chunks: tuple[int, ...] = (256, 512, 1024, 2048), decode: int = 32,
               repeats: int = 3) -> list[tuple[int, int, float]]:
    """(tokens, rows, seconds) per step: a prefill of each of `chunks` (capped by the step budget) as one row, and a
    decode step at each row count up to `max_seqs`, timed as `decode` more tokens; medians of `repeats` after a
    warm-up. Prompts are fresh, so nothing hits the prefix cache."""
    gc.collect()
    engine = Engine(config)
    gen = random.Random(0)

    def prompt(n: int) -> list[int]:
        return [gen.randrange(engine.spec.vocab) for _ in range(n)]

    def timed(prompts: list[list[int]], tokens: int) -> float:
        t0 = time.perf_counter()
        engine.generate(prompts, SamplingParams(temperature=0.0, max_tokens=tokens, ignore_eos=True))
        return time.perf_counter() - t0
    timed([prompt(8)], 4)
    out = []
    for n in sorted({min(c, config.max_step_tokens) for c in chunks}):
        out.append((n, 1, statistics.median(timed([prompt(n)], 1) for _ in range(repeats))))
    for rows in range(1, config.max_seqs + 1):
        step = statistics.median((timed([prompt(32) for _ in range(rows)], decode + 1)
                                  - timed([prompt(32) for _ in range(rows)], 1)) / decode for _ in range(repeats))
        out.append((rows, rows, step))
    engine.close()
    return out


def clock(samples: list[tuple[int, int, float]]) -> tuple[float, float, float]:
    """`step_cost`'s a, b, c: the fit of seconds to a + b·tokens + c·rows least in relative error (a plain fit lets
    the long prefill steps decide and misses decode steps by half), each at least 0."""
    x = np.array([[1.0, t, r] for t, r, _ in samples])
    y = np.array([s for _, _, s in samples])
    return tuple(max(0.0, float(v)) for v in np.linalg.lstsq(x / y[:, None], np.ones_like(y), rcond=None)[0])
