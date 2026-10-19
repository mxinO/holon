"""Bounded comparison of engine configurations on a multi-turn trace (design §7.10; L1.5 state invariants, §10.3).

Configurations that change batch shape, chunking or path (prefix cache on vs off, overlap depth 1 vs 0) are not
compared token for token: greedy decoding may take either side of a near-tie. Each configuration instead generates
greedily on its own production path, without logprobs (logprob steps run eagerly, which would hide graph replay),
and one reference engine (prefix cache off, depth 0, eager) re-scores the generated tokens teacher-forced: how often
its argmax is the generated token, overall and at confident positions. Agreement is coarse, though: a bug that
perturbs the state a little (one prefix token's state never computed) moves it by less than the noise. So each
configuration also runs the trace with logprobs (eager, but at its own depth and with its own prefix cache) and is
judged as L2 judges an engine: KL(reference ‖ configuration) at the same positions.

As in L2, the floor is the same comparison without the thing under test: the baseline configuration (prefix cache
off, depth 0, graphs on) re-scored the same way. Two bf16 paths (decode vs prefill kernels) disagree at near-ties no
matter what, and generated text has more near-ties than the L2 prompt set, so the L2 floor does not transfer. A
configuration passes if its agreement is within `margin` of the floor's, where `margin` is the L2 slack
(`compare.TOP1_DELTA`) or 2.5 standard errors of the difference, whichever is larger, and its KL p50 and p99 are at
most `compare.KL_FACTOR` times the floor's, as in L2, or above it by 2.5 standard errors of the difference, whichever
is larger. The standard errors are the floor's (sessions as clusters; a session bootstrap for the KL quantiles), so
the noise is that of two runs without the change. Planted bugs in prefix adoption and in the
depth-1 token feed fail it (`planted`; tested in judge/tests/test_bounded.py and calibrated on Qwen3-8B in the
evidence).

The trace: sessions share a system prompt and every turn resends the whole history, generated tokens included, plus
a new user chunk, so later turns hit the prefix cache (`tokens_saved`). Each turn is one wave of requests. For the
checkpoint policies (`--policies`, docs/v1-kv-tiers.md §6), late sessions start one or two waves after the first:
a session's own wave never shows it a branch, so only they meet the system prompt's end as a branch point, and those
of the third wave adopt the `branch` checkpoints the second published.
"""
from __future__ import annotations

import dataclasses
import gc
import math

import torch
from holon.api import Logprobs, SamplingParams

from .compare import CONFIDENT_GAP, KL_FACTOR, TOP1_DELTA, kl_topk
from .engine import score
from .recording import Prompt, PromptSet, Recording, tail_from_topk

Turn = tuple[list[int], list[int], list[Logprobs]]   # prompt ids, generated ids, their logprobs (if asked for)
BASELINE = {"prefix_cache": False, "overlap": False}
SE_FACTOR = 2.5
KL_EPS = 1e-9             # nats: the KL margin when the baseline's and the noise are ~0 (fp32 tests)
BOOTSTRAP = 200           # session resamples for the standard error of a KL quantile


@dataclasses.dataclass(frozen=True)
class Trace:
    system: list[int]
    chunks: list[list[list[int]]]           # per session, per turn: the new user tokens
    max_tokens: int
    starts: tuple[int, ...] = ()            # per session, the wave of its first turn (none: all the first)

    @classmethod
    def from_prompts(cls, prompts: PromptSet, sessions: int = 48, turns: int = 4, system_len: int = 512,
                     chunk_len: int = 96, max_tokens: int = 128, late: int = 0) -> Trace:
        """A deterministic trace cut from the prompt set: the system prompt from its longest prompt, user chunks
        from the others in order; `late` more sessions start in the second and third waves, alternately."""
        longest = max(prompts.prompts, key=lambda p: len(p.ids))
        if len(longest.ids) < system_len:
            raise ValueError(f"no prompt has {system_len} tokens for the system prompt")
        rest = [p.ids[:chunk_len] for p in prompts.prompts if p is not longest]
        n = sessions + late
        chunks = [[rest[(s * turns + t) % len(rest)] for t in range(turns)] for s in range(n)]
        return cls(longest.ids[:system_len], chunks, max_tokens, (0,) * sessions + tuple(1 + i % 2 for i in range(late))
                   if late else ())

    @property
    def max_len(self) -> int:
        """Tokens per request at most, plus the one token the re-scoring samples."""
        return len(self.system) + sum(max(len(c[t]) for c in self.chunks) + self.max_tokens
                                      for t in range(len(self.chunks[0]))) + 1

    def run(self, engine, logprobs: int | None = None) -> list[Turn]:
        """Every session's turns, generated greedily; one wave per turn of the sessions started by then. In turn
        order, then session order (turn i belongs to session i % sessions)."""
        params = SamplingParams(temperature=0.0, max_tokens=self.max_tokens, ignore_eos=True, logprobs=logprobs)
        starts, turns = self.starts or (0,) * len(self.chunks), len(self.chunks[0])
        history = [list(self.system) for _ in self.chunks]
        done: dict[tuple[int, int], Turn] = {}
        for wave in range(max(starts) + turns):
            now = [(s, wave - starts[s]) for s in range(len(self.chunks)) if 0 <= wave - starts[s] < turns]
            prompts = [history[s] + self.chunks[s][t] for s, t in now]
            for (s, t), prompt, (out, lps) in zip(now, prompts, _generate(engine, prompts, params), strict=True):
                done[s, t] = (prompt, out, lps)
                history[s] = prompt + out
        return [done[s, t] for t in range(turns) for s in range(len(self.chunks))]


def _generate(engine, prompts: list[list[int]], params: SamplingParams) -> list[tuple[list[int], list[Logprobs]]]:
    index = {engine.add(p, params): i for i, p in enumerate(prompts)}
    out: list[tuple[list[int], list[Logprobs]]] = [([], []) for _ in prompts]
    while engine.has_work():
        for d in engine.step():
            tokens, lps = out[index[d.rid]]
            tokens += d.token_ids
            if d.logprobs is not None:
                lps.append(d.logprobs)
    return out


@dataclasses.dataclass(frozen=True)
class Mean:
    value: float
    se: float                 # cluster-robust standard error, sessions as clusters (turns of one session correlate)
    n: int

    @classmethod
    def of(cls, values: torch.Tensor, clusters: torch.Tensor) -> Mean:
        n = len(values)
        if n == 0:
            return cls(math.nan, math.nan, 0)          # fails every check
        _, clusters = clusters.unique(return_inverse=True)
        s = int(clusters.max()) + 1
        v_s = torch.zeros(s, dtype=torch.float64).index_add_(0, clusters, values.double())
        n_s = torch.zeros(s, dtype=torch.float64).index_add_(0, clusters, torch.ones(n, dtype=torch.float64))
        mean = v_s.sum().item() / n
        se = math.sqrt(s / (s - 1) * ((v_s - mean * n_s) ** 2).sum().item()) / n if s > 1 else math.nan
        return cls(mean, se, n)

    def noise(self) -> float:
        """As a floor: SE_FACTOR standard errors of the difference between two runs like it (0 when unknown)."""
        return _noise(self.se)


@dataclasses.dataclass(frozen=True)
class Agreement:
    top1: Mean                # the reference's argmax is the generated token
    confident: Mean           # the same where the reference's top-2 gap >= CONFIDENT_GAP
    regret: Mean              # nats: reference logprob of its argmax minus that of the generated token (reported:
                              # heavy-tailed, its noise is too large to judge by)

    def checks(self, floor: Agreement) -> dict[str, tuple[bool, float]]:
        """name -> (passed, margin): agreement may fall below the floor's by the L2 slack or the noise."""
        out = {}
        for name in ("top1", "confident"):
            mine, theirs = getattr(self, name), getattr(floor, name)
            margin = max(TOP1_DELTA, theirs.noise())
            out[name] = (mine.value >= theirs.value - margin, margin)
        return out


@dataclasses.dataclass(frozen=True)
class Divergence:
    kl: Mean                  # KL(reference ‖ configuration) at the generated positions, top-k bucketed as in L2
    p50: float
    p99: float
    p50_se: float             # session-bootstrap standard errors of the quantiles
    p99_se: float
    by_wave: dict = dataclasses.field(default_factory=dict)   # reported: "first" / "late" sessions' p50, p99, n

    def checks(self, floor: Divergence) -> dict[str, tuple[bool, float]]:
        out = {}
        for name in ("p50", "p99"):
            mine, theirs = getattr(self, name), getattr(floor, name)
            margin = max((KL_FACTOR - 1) * theirs, _noise(getattr(floor, f"{name}_se")), KL_EPS)
            out[f"kl_{name}"] = (mine <= theirs + margin, margin)
        return out


def _noise(floor_se: float) -> float:
    """SE_FACTOR standard errors of the difference between two independent runs that behave like the floor: the
    null hypothesis. (A configuration's own SE would let a bug that hits some sessions widen its own margin.)"""
    x = SE_FACTOR * math.sqrt(2) * floor_se
    return 0.0 if math.isnan(x) else x


def _quantiles(values: torch.Tensor, clusters: torch.Tensor, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantiles `q` of `values` and their standard errors, by resampling whole clusters (a fixed seed)."""
    order = torch.argsort(clusters, stable=True)
    groups = torch.split(values[order], torch.unique_consecutive(clusters[order], return_counts=True)[1].tolist())
    gen = torch.Generator().manual_seed(0)
    draws = torch.stack([torch.quantile(torch.cat([groups[i] for i in torch.randint(len(groups), (len(groups),),
                                                                                       generator=gen).tolist()]), q)
                         for _ in range(BOOTSTRAP)])
    return torch.quantile(values, q), draws.std(0)


def _reference(reference, turns: list[Turn], sessions: int, k: int) -> tuple[Recording, torch.Tensor]:
    """The reference's teacher-forced rows at every turn's generated positions (`turns` in `Trace.run` order: turn i
    belongs to session i % sessions), and each row's session."""
    prompts = PromptSet("bounded", "", [Prompt(str(i), "turn", p + o) for i, (p, o, _) in enumerate(turns)])
    rec = score(reference, prompts, k)
    # Row j of a sequence predicts its token j + 1: the generated tokens are predicted by its last len(out) rows.
    rows = torch.cat([torch.arange(int(rec.offsets[i + 1]) - len(o), int(rec.offsets[i + 1]))
                      for i, (_, o, _) in enumerate(turns)])
    session = torch.repeat_interleave(torch.arange(len(turns)) % sessions, torch.tensor([len(o) for _, o, _ in turns]))
    sliced = Recording(rec.topk_ids[rows], rec.topk_lp[rows], rec.target_lp[rows], rec.tail_lp[rows], rec.offsets, {})
    return sliced, session


def agreement(reference, turns: list[Turn], sessions: int, k: int) -> Agreement:
    """How the reference, teacher-forced, judges the generated tokens."""
    ref, session = _reference(reference, turns, sessions, k)
    agree = ref.topk_ids[:, 0] == torch.tensor([t for _, o, _ in turns for t in o])
    sure = ref.topk_lp[:, 0] - ref.topk_lp[:, 1] >= CONFIDENT_GAP
    regret = ref.topk_lp[:, 0] - ref.target_lp
    return Agreement(Mean.of(agree, session), Mean.of(agree[sure], session[sure]), Mean.of(regret, session))


def divergence(reference, turns: list[Turn], sessions: int, k: int, starts: tuple[int, ...] = ()) -> Divergence:
    """KL(reference ‖ configuration) at the generated positions, from the logprobs the configuration emitted; with
    `starts` (each session's first wave), also the first wave's sessions apart from the late ones (reported)."""
    ref, session = _reference(reference, turns, sessions, k)
    ids = torch.tensor([x[:k] for _, _, lps in turns for lp in lps for x in lp.top_ids])
    top = torch.tensor([x[:k] for _, _, lps in turns for lp in lps for x in lp.top], dtype=torch.float32)
    cand = Recording(ids, top, top[:, 0], tail_from_topk(top), ref.offsets, {})
    kl, _ = kl_topk(ref, cand)
    (p50, p99), (p50_se, p99_se) = (t.tolist() for t in _quantiles(kl, session, torch.tensor([0.5, 0.99],
                                                                                              dtype=torch.float64)))
    late = torch.tensor([bool(starts and starts[i] > 0) for i in range(sessions)])[session]
    by_wave = {name: {"n": int(mask.sum()), **dict(zip(("p50", "p99"), torch.quantile(
        kl[mask].double(), torch.tensor([0.5, 0.99], dtype=torch.float64)).tolist(), strict=True))}
        for name, mask in (("first", ~late), ("late", late)) if mask.any()} if starts else {}
    return Divergence(Mean.of(kl, session), p50, p99, p50_se, p99_se, by_wave)


def compare_configs(build, configs: dict[str, dict], trace: Trace, k: int = 20, baseline: dict = BASELINE,
                    reference: dict | None = None) -> dict:
    """Run `trace` under the baseline and every configuration (EngineConfig overrides by name), without and with
    logprobs, re-score each run with the reference engine and judge it against the baseline. `build(**overrides)`
    returns an engine. Graphs vs eager: `baseline` with `cuda_graphs=False`, the configuration with it on.
    Speculation vs plain decoding: `baseline` and `reference` (more overrides of the reference engine) without
    drafts."""
    runs: dict[str, tuple[list[Turn], list[Turn], dict]] = {}
    for name, overrides in {"baseline": baseline, **configs}.items():
        engine = build(**overrides)
        plain = trace.run(engine)
        stats = engine.stats()
        runs[name] = plain, trace.run(engine, logprobs=k), stats
        _free(engine)
    reference = build(**(BASELINE | {"cuda_graphs": False, "max_step_tokens": max(trace.max_len, 4096)}
                         | (reference or {})))
    sessions = len(trace.chunks)
    scored = {name: (agreement(reference, plain, sessions, k), divergence(reference, lp, sessions, k, trace.starts),
                     stats)
              for name, (plain, lp, stats) in runs.items()}
    _free(reference)
    floor_a, floor_d, _ = scored["baseline"]
    results = {}
    for name, (agree, div, stats) in scored.items():
        checks = agree.checks(floor_a) | div.checks(floor_d)
        values = {"top1": agree.top1.value, "confident": agree.confident.value, "kl_p50": div.p50, "kl_p99": div.p99}
        failures = [f"{what} {values[what]:.4g} beyond {margin:.3g} of the baseline"
                    for what, (ok, margin) in checks.items() if not ok]
        if name != "baseline" and configs[name].get("chunk_with_decode") and not stats["mixed_steps"]:
            failures.append("no step mixed decode rows with prefill chunks")
        results[name] = {"config": {"cuda_graphs": True} | (baseline if name == "baseline" else configs[name]),
                         **dataclasses.asdict(agree), "kl": dataclasses.asdict(div),
                         "tokens_saved": stats["prefix_tokens_saved"], "mixed_steps": stats["mixed_steps"],
                         "margins": {what: margin for what, (_, margin) in checks.items()},
                         "failures": failures, "passed": not failures}
    return {"sessions": sessions, "turns": len(trace.chunks[0]), "system_len": len(trace.system),
            "max_tokens": trace.max_tokens, "results": results,
            "passed": all(r["passed"] for r in results.values())}


def _free(engine) -> None:
    engine.close()
    gc.collect()


def table(report: dict) -> str:
    rows = [f"{'config':22s} {'n':>6s} {'top-1 ± SE':>17s} {'confident ± SE':>17s} {'regret':>8s} {'KL p50':>9s} "
            f"{'KL p99':>9s} {'saved':>7s}  verdict"]
    for name, r in report["results"].items():
        verdict = "PASS" if r["passed"] else "FAIL: " + "; ".join(r["failures"])
        t, c = r["top1"], r["confident"]
        rows.append(f"{name:22s} {t['n']:6d} {t['value']:9.3%} ± {t['se']:.3%} {c['value']:9.3%} ± {c['se']:.3%} "
                    f"{r['regret']['value']:8.4f} {r['kl']['p50']:9.2e} {r['kl']['p99']:9.2e} {r['tokens_saved']:7d}  "
                    f"{verdict}")
    return "\n".join(rows)
