"""Bounded comparison of engine configurations on a multi-turn trace (design §7.10; L1.5 state invariants, §10.3).

Configurations that change batch shape, chunking or path (prefix cache on vs off, overlap depth 1 vs 0) are not
compared token for token: greedy decoding may take either side of a near-tie. Each configuration instead generates
greedily on its own production path, without logprobs (logprob steps commit synchronously and run eagerly, which
would hide depth 1 and graph replay), and one reference engine (prefix cache off, depth 0, eager) re-scores the
generated tokens teacher-forced: how often its argmax is the generated token, overall and at confident positions.

As in L2, the floor is the same comparison without the thing under test: the baseline configuration (prefix cache
off, depth 0, graphs on) re-scored the same way. Two bf16 paths (decode vs prefill kernels) disagree at near-ties no
matter what, and generated text has more near-ties than the L2 prompt set, so the L2 floor does not transfer. A
configuration passes if its agreement is within `margin` of the floor's, where `margin` is the L2 slack
(`compare.TOP1_DELTA`) or 2.5 standard errors of the difference, whichever is larger (sequences are the clusters).

The trace: sessions share a system prompt and every turn resends the whole history, generated tokens included, plus
a new user chunk, so later turns hit the prefix cache (`tokens_saved`).
"""
from __future__ import annotations

import dataclasses
import gc
import math

import torch
from holon.api import SamplingParams

from .compare import CONFIDENT_GAP, TOP1_DELTA
from .engine import score
from .recording import Prompt, PromptSet

Turn = tuple[list[int], list[int]]          # prompt ids, generated ids
BASELINE = {"prefix_cache": False, "overlap": False}
SE_FACTOR = 2.5


@dataclasses.dataclass(frozen=True)
class Trace:
    system: list[int]
    chunks: list[list[list[int]]]           # per session, per turn: the new user tokens
    max_tokens: int

    @classmethod
    def from_prompts(cls, prompts: PromptSet, sessions: int = 48, turns: int = 4, system_len: int = 512,
                     chunk_len: int = 96, max_tokens: int = 128) -> Trace:
        """A deterministic trace cut from the prompt set: the system prompt from its longest prompt, user chunks
        from the others in order."""
        longest = max(prompts.prompts, key=lambda p: len(p.ids))
        if len(longest.ids) < system_len:
            raise ValueError(f"no prompt has {system_len} tokens for the system prompt")
        rest = [p.ids[:chunk_len] for p in prompts.prompts if p is not longest]
        chunks = [[rest[(s * turns + t) % len(rest)] for t in range(turns)] for s in range(sessions)]
        return cls(longest.ids[:system_len], chunks, max_tokens)

    @property
    def max_len(self) -> int:
        """Tokens per request at most, plus the one token the re-scoring samples."""
        return len(self.system) + sum(max(len(c[t]) for c in self.chunks) + self.max_tokens
                                      for t in range(len(self.chunks[0]))) + 1

    def run(self, engine) -> list[Turn]:
        """Every session's turns, generated greedily; one wave per turn."""
        params = SamplingParams(temperature=0.0, max_tokens=self.max_tokens, ignore_eos=True)
        history = [list(self.system) for _ in self.chunks]
        turns: list[Turn] = []
        for t in range(len(self.chunks[0])):
            prompts = [h + c[t] for h, c in zip(history, self.chunks, strict=True)]
            for s, (prompt, out) in enumerate(zip(prompts, engine.generate(prompts, params), strict=True)):
                turns.append((prompt, out))
                history[s] = prompt + out
        return turns


@dataclasses.dataclass(frozen=True)
class Rate:
    value: float
    se: float                 # cluster-robust standard error, sequences as clusters
    n: int

    @classmethod
    def of(cls, hits: torch.Tensor, seq: torch.Tensor) -> Rate:
        n = len(hits)
        if n == 0:
            return cls(math.nan, math.nan, 0)          # fails every check
        _, seq = seq.unique(return_inverse=True)
        s = int(seq.max()) + 1
        h_s = torch.zeros(s, dtype=torch.float64).index_add_(0, seq, hits.double())
        n_s = torch.zeros(s, dtype=torch.float64).index_add_(0, seq, torch.ones(n, dtype=torch.float64))
        p = h_s.sum().item() / n
        se = math.sqrt(s / (s - 1) * ((h_s - p * n_s) ** 2).sum().item()) / n if s > 1 else math.nan
        return cls(p, se, n)

    def within(self, floor: Rate) -> tuple[bool, float]:
        """Whether this rate is at most `margin` below the floor's; the margin."""
        noise = SE_FACTOR * math.hypot(self.se, floor.se)
        margin = TOP1_DELTA if math.isnan(noise) else max(TOP1_DELTA, noise)
        return self.value >= floor.value - margin, margin


@dataclasses.dataclass(frozen=True)
class Agreement:
    top1: Rate                # the reference's argmax is the generated token
    confident: Rate           # the same where the reference's top-2 gap >= CONFIDENT_GAP
    regret: float             # mean nats: reference logprob of its argmax minus that of the generated token


def rescore(reference, turns: list[Turn], k: int) -> Agreement:
    """Teacher-force every turn's prompt + output through `reference` and judge the generated tokens."""
    prompts = PromptSet("bounded", "", [Prompt(str(i), "turn", p + o) for i, (p, o) in enumerate(turns)])
    rec = score(reference, prompts, k)
    # Row j of a sequence predicts its token j + 1: the generated tokens are predicted by its last len(out) rows.
    rows = torch.cat([torch.arange(int(rec.offsets[i + 1]) - len(o), int(rec.offsets[i + 1]))
                      for i, (_, o) in enumerate(turns)])
    seq = torch.repeat_interleave(torch.arange(len(turns)), torch.tensor([len(o) for _, o in turns]))
    agree = rec.topk_ids[rows, 0] == torch.tensor([t for _, o in turns for t in o])
    sure = rec.topk_lp[rows, 0] - rec.topk_lp[rows, 1] >= CONFIDENT_GAP
    return Agreement(Rate.of(agree, seq), Rate.of(agree[sure], seq[sure]),
                     (rec.topk_lp[rows, 0] - rec.target_lp[rows]).double().mean().item())


def compare_configs(build, configs: dict[str, dict], trace: Trace, k: int = 20) -> dict:
    """Run `trace` under the baseline and every configuration (EngineConfig overrides by name), re-score each with
    the reference engine and judge it against the baseline. `build(**overrides)` returns an engine."""
    outputs: dict[str, tuple[list[Turn], dict]] = {}
    for name, overrides in {"baseline": BASELINE, **configs}.items():
        engine = build(**overrides)
        outputs[name] = trace.run(engine), engine.stats()
        _free(engine)
    reference = build(**BASELINE, cuda_graphs=False, max_step_tokens=max(trace.max_len, 4096))
    scored = {name: (rescore(reference, turns, k), stats) for name, (turns, stats) in outputs.items()}
    _free(reference)
    floor = scored["baseline"][0]
    results = {}
    for name, (agreement, stats) in scored.items():
        checks = {"top-1": agreement.top1.within(floor.top1),
                  "confident top-1": agreement.confident.within(floor.confident)}
        failures = [f"{what} {getattr(agreement, attr).value:.3%} more than {margin:.3%} below the baseline"
                    for (what, (ok, margin)), attr in zip(checks.items(), ("top1", "confident"), strict=True)
                    if not ok]
        results[name] = {"config": {"cuda_graphs": True} | (BASELINE if name == "baseline" else configs[name]),
                         **dataclasses.asdict(agreement), "tokens_saved": stats["prefix_tokens_saved"],
                         "margins": {what: margin for what, (_, margin) in checks.items()},
                         "failures": failures, "passed": not failures}
    return {"sessions": len(trace.chunks), "turns": len(trace.chunks[0]), "system_len": len(trace.system),
            "max_tokens": trace.max_tokens, "results": results,
            "passed": all(r["passed"] for r in results.values())}


def _free(engine) -> None:
    engine.close()
    gc.collect()


def table(report: dict) -> str:
    rows = [f"{'config':22s} {'n':>6s} {'top-1 ± SE':>17s} {'confident ± SE':>17s} {'regret':>8s} {'saved':>7s}  "
            f"verdict"]
    for name, r in report["results"].items():
        verdict = "PASS" if r["passed"] else "FAIL: " + "; ".join(r["failures"])
        t, c = r["top1"], r["confident"]
        rows.append(f"{name:22s} {t['n']:6d} {t['value']:9.3%} ± {t['se']:.3%} {c['value']:9.3%} ± {c['se']:.3%} "
                    f"{r['regret']:8.4f} {r['tokens_saved']:7d}  {verdict}")
    return "\n".join(rows)
