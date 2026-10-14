"""The constrained bounded check (docs/m05-grammar.md §5, acceptance 3): grammar with speculation emits what grammar
without speculation emits, up to near-ties, on a real model.

JSON-schema prompts (chat-rendered, thinking off; strict schemas from JSONSchemaBench's Glaiveai2K and Github_easy
test splits that D9's reject list accepts) run greedily without `ignore_eos` on two engines with the grammar
processor: plain and speculating (the engine args' drafter). Each runs them free and constrained, on its production
path (no logprobs) and again with logprobs (eager), so there are two pairs of runs to compare per path.

- **Identity up to near-ties.** Per prompt, the two engines' tokens are identical (same length, same finish reason),
  or they first differ at a position where the reference — the plain engine's weights, eager, depth 0, no prefix
  cache, teacher-forced on the common prefix — is near a tie: its top-2 gap there, among the tokens a fresh matcher
  replaying the prefix allows (unmasked for free runs), is below the threshold. The threshold is `CONFIDENT_GAP`
  raised to the largest first-divergence gap of the free pair, which speculation without a grammar (lossless since
  M0.4) shows to be kernel noise: calibrated by the free runs, then applied to the constrained ones.
- **No errors.** No constrained request ends with `error` (a token its matcher refuses).
- **Agreement.** How often the generated token is the reference's masked argmax — overall and where its masked
  top-2 gap is at least the threshold — for the speculating engine, at least the plain engine's minus the L2 slack or
  2.5 standard errors of the difference (prompts as clusters), as `bounded` judges.
- **KL.** KL(plain ‖ spec) between the two engines' emitted logprobs (of the masked distribution) before their first
  divergence: p50 and p99 at most `KL_FACTOR` times the free pair's (−∞ logprobs of masked tokens carry no mass).

The masked argmax comes from the reference's top-k: the first allowed id among them (a gap is a lower bound when
only one allowed id is among them; a position with none is unresolved, and an unresolved first divergence fails).
"""
from __future__ import annotations

import dataclasses
import gc
import json
import pathlib
import time

import numpy as np
import torch
from holon.api import RequestError, SamplingParams

from .bounded import Mean, _noise
from .compare import CONFIDENT_GAP, KL_FACTOR, TOP1_DELTA, kl_topk
from .engine import score
from .recording import Prompt, PromptSet, Recording, tail_from_topk

KEY = "grammar.constraint@1"
SUBSETS = ("Glaiveai2K", "Github_easy")
SYSTEM = "You need to generate a JSON object that matches the schema below."
MAX_CHARS = 4000                # small schemas: compiled in milliseconds anywhere
KL_EPS = 1e-9


@dataclasses.dataclass
class Run:
    tokens: list[list[int]]
    finish: list[str]
    logprobs: list[tuple[list[list[int]], list[list[float]]]] | None     # per prompt: top ids, top logprobs


def schemas(jsb: pathlib.Path, n: int) -> list[tuple[str, dict]]:
    """`n` strict-accepted schemas, alternating the subsets, each subset's in id order."""
    import pandas as pd
    from holon_plugins.grammar.xgrammar.constraint import strict_violation
    per = []
    for subset in SUBSETS:
        frame = pd.read_parquet(jsb / subset / "test.parquet").sort_values("unique_id")
        per.append([(r.unique_id, json.loads(r.json_schema)) for r in frame.itertuples()
                    if len(r.json_schema) < MAX_CHARS and strict_violation(json.loads(r.json_schema)) is None])
    out = [s for pair in zip(*per, strict=False) for s in pair][:n]
    if len(out) < n:
        raise ValueError(f"only {len(out)} schemas qualify")
    return out


def generate(engine, prompts: list[list[int]], constraints: list[dict | None], max_tokens: int,
             k: int | None) -> Run:
    """Every prompt, greedy; a constraint the processor's vetting queue cannot take yet waits for the engine."""
    rids = {}
    tokens, finish = [[] for _ in prompts], [""] * len(prompts)
    lps = [([], []) for _ in prompts]

    def step() -> None:
        if wait := engine.idle_for():
            time.sleep(wait)
        for d in engine.step():
            i = rids[d.rid]
            tokens[i] += d.token_ids
            if d.logprobs is not None:
                lps[i][0].extend(x[:k] for x in d.logprobs.top_ids)
                lps[i][1].extend(x[:k] for x in d.logprobs.top)
            if d.finish_reason:
                finish[i] = d.finish_reason

    for i, (p, c) in enumerate(zip(prompts, constraints, strict=True)):
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=k, ext={KEY: c} if c else {})
        while True:
            try:
                rids[engine.add(p, params)] = i
                break
            except RequestError as e:
                if e.kind != "busy" or not engine.has_work():
                    raise
                step()
    while engine.has_work():
        step()
    return Run(tokens, finish, lps if k else None)


def _first_diff(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


class Masks:
    """A fresh matcher's masks along a token sequence (D3's replay), as allowed-id tests."""

    def __init__(self, tokenizer, vocab: int, stops: list[int]) -> None:
        import xgrammar as xgr
        self.xgr, self.vocab, self.stops, self.compiled = xgr, vocab, stops, {}
        self.compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab,
                                                                               stop_token_ids=stops))

    def along(self, constraint: dict, tokens: list[int]) -> list[np.ndarray | None]:
        """Per position, the allowed-token bitmask words before that token (None after a refused token)."""
        from holon_plugins.grammar.xgrammar.constraint import grammar
        key = json.dumps(constraint, sort_keys=True)
        if key not in self.compiled:
            self.compiled[key] = self.compiler.compile_grammar(grammar(constraint))
        matcher = self.xgr.GrammarMatcher(self.compiled[key], override_stop_tokens=self.stops)
        mask, out = self.xgr.allocate_token_bitmask(1, self.vocab), []
        for t in tokens:
            matcher.fill_next_token_bitmask(mask)
            out.append(mask[0].numpy().view(np.uint32).copy())
            if not matcher.accept_token(t):
                return out + [None] * (len(tokens) - len(out))
        return out


def _top(ids: list[int], lps: list[float], words: np.ndarray | None) -> tuple[int | None, float | None]:
    """The masked argmax among the top-k and its masked top-2 gap (a lower bound with one allowed id; None: none)."""
    allowed = [(i, lp) for i, lp in zip(ids, lps, strict=True)
               if words is None or (int(words[i >> 5]) >> (i & 31)) & 1]
    if not allowed:
        return None, None
    second = allowed[1][1] if len(allowed) > 1 else lps[-1]
    return allowed[0][0], allowed[0][1] - second


def judge(build, tokenizer, render, jsb: pathlib.Path, n: int = 64, max_tokens: int = 512, k: int = 20) -> dict:
    """`build(name)` returns the "plain" or "spec" engine (the grammar processor selected) or the "reference" (none);
    `render(messages)` the chat prompt's ids (thinking off); `tokenizer` the model's, for the fresh matchers."""
    items = schemas(jsb, n)
    prompts = [render([{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(s)}])
               for _, s in items]
    constraints = [{"type": "json_schema", "schema": s, "strict": True} for _, s in items]
    runs: dict[str, Run] = {}
    counters = {}
    for name in ("plain", "spec"):
        engine = build(name)
        vocab, stops, before = engine.spec.vocab, sorted(engine.eos), engine.stats()
        for mode, cs in (("free", [None] * n), ("grammar", constraints)):
            runs[f"{name}/{mode}"] = generate(engine, prompts, cs, max_tokens, None)
            runs[f"{name}/{mode}/lp"] = generate(engine, prompts, cs, max_tokens, k)
        counters[name] = {key: v - before.get(key, 0) for key, v in engine.stats().items()
                          if key.startswith("spec_") or key.endswith("drafts_refused")}
        engine.close()
        gc.collect()

    # Teacher-forced reference rows for every (prompt, output) of the runs.
    seqs = sorted({(i, tuple(r.tokens[i])) for r in runs.values() for i in range(n)})
    reference = build("reference")
    rec = score(reference, PromptSet("constrained", "", [Prompt(str(j), "json", prompts[i] + list(out))
                                                         for j, (i, out) in enumerate(seqs)]), k)
    reference.close()
    gc.collect()
    rows = {}
    for j, (i, out) in enumerate(seqs):
        end = int(rec.offsets[j + 1])
        rows[i, out] = (rec.topk_ids[end - len(out): end].tolist(), rec.topk_lp[end - len(out): end].tolist())
    masks = Masks(tokenizer, vocab, stops)

    def masked(mode: str, i: int, out: tuple[int, ...]) -> list[tuple[int | None, float | None]]:
        words = masks.along(constraints[i], list(out)) if mode == "grammar" else [None] * len(out)
        ids, lps = rows[i, out]
        return [_top(a, b, w) if (w is not None or mode == "free") else (None, None)
                for a, b, w in zip(ids, lps, words, strict=True)]

    report: dict = {"prompts": n, "max_tokens": max_tokens, "k": k, "counters": counters,
                    "schemas": [sid for sid, _ in items]}
    gaps = {}
    for mode in ("free", "grammar"):
        for path in ("", "/lp"):
            a, b = runs[f"plain/{mode}{path}"], runs[f"spec/{mode}{path}"]
            diverged = []
            for i in range(n):
                d = _first_diff(a.tokens[i], b.tokens[i])
                if d is not None:
                    top, gap = masked(mode, i, tuple(a.tokens[i]))[d] if d < len(a.tokens[i]) else \
                        masked(mode, i, tuple(b.tokens[i]))[d]
                    diverged.append({"prompt": i, "at": d, "gap": gap, "reference_top": top,
                                     "plain": a.tokens[i][d] if d < len(a.tokens[i]) else None,
                                     "spec": b.tokens[i][d] if d < len(b.tokens[i]) else None})
            gaps[mode + path] = diverged
            report[f"{mode}{path}"] = {
                "identical": n - len(diverged), "diverged": diverged,
                "finish_mismatch_identical": sum(a.finish[i] != b.finish[i] for i in range(n)
                                                 if _first_diff(a.tokens[i], b.tokens[i]) is None),
                "errors": {"plain": a.finish.count("error"), "spec": b.finish.count("error")},
                "finish": {"plain": {f: a.finish.count(f) for f in set(a.finish)},
                           "spec": {f: b.finish.count(f) for f in set(b.finish)}},
                "tokens": {"plain": sum(map(len, a.tokens)), "spec": sum(map(len, b.tokens))}}
    free_gaps = [g["gap"] for path in ("free", "free/lp") for g in gaps[path] if g["gap"] is not None]
    threshold = max([CONFIDENT_GAP, *free_gaps])
    report["threshold"] = {"confident_gap": CONFIDENT_GAP, "free_first_divergence_gaps": sorted(free_gaps),
                           "used": threshold}

    # Agreement with the reference's masked argmax, per engine, on the constrained production runs.
    agreement = {}
    for name in ("plain", "spec"):
        r = runs[f"{name}/grammar"]
        hit, sure, cluster, unresolved = [], [], [], 0
        for i in range(n):
            for (top, gap), t in zip(masked("grammar", i, tuple(r.tokens[i])), r.tokens[i], strict=True):
                if top is None:
                    unresolved += 1
                    continue
                hit.append(float(top == t))
                sure.append(gap >= threshold)
                cluster.append(i)
        hit_t, sure_t, cl = torch.tensor(hit), torch.tensor(sure), torch.tensor(cluster)
        agreement[name] = {"top1": Mean.of(hit_t, cl), "confident": Mean.of(hit_t[sure_t], cl[sure_t]),
                           "unresolved": unresolved}
    report["agreement"] = {name: {"top1": dataclasses.asdict(v["top1"]),
                                  "confident": dataclasses.asdict(v["confident"]), "unresolved": v["unresolved"]}
                           for name, v in agreement.items()}

    # KL(plain ‖ spec) before the first divergence, from the logprob runs, per mode.
    kl = {}
    for mode in ("free", "grammar"):
        a, b = runs[f"plain/{mode}/lp"], runs[f"spec/{mode}/lp"]
        ids_a, lp_a, ids_b, lp_b = [], [], [], []
        for i in range(n):
            d = _first_diff(a.tokens[i], b.tokens[i])
            m = len(a.tokens[i]) if d is None else d + 1          # the divergent position's distribution too
            ids_a += a.logprobs[i][0][:m]
            lp_a += a.logprobs[i][1][:m]
            ids_b += b.logprobs[i][0][:m]
            lp_b += b.logprobs[i][1][:m]
        top_a, top_b = torch.tensor(lp_a, dtype=torch.float32), torch.tensor(lp_b, dtype=torch.float32)
        div, _ = kl_topk(Recording(torch.tensor(ids_a), top_a, top_a[:, 0], tail_from_topk(top_a), None, {}),
                         Recording(torch.tensor(ids_b), top_b, top_b[:, 0], tail_from_topk(top_b), None, {}))
        p50, p99 = torch.quantile(div, torch.tensor([0.5, 0.99], dtype=torch.float64)).tolist()
        kl[mode] = {"positions": len(div), "p50": p50, "p99": p99, "max": div.max().item(),
                    "nan": int(div.isnan().sum())}
    report["kl"] = kl

    failures = []
    for path in ("grammar", "grammar/lp"):
        r = report[path]
        if r["errors"]["plain"] or r["errors"]["spec"]:
            failures.append(f"{path}: error finishes {r['errors']}")
        if r["finish_mismatch_identical"]:
            failures.append(f"{path}: identical tokens with different finish reasons")
        for g in r["diverged"]:
            if g["gap"] is None or g["gap"] >= threshold:
                failures.append(f"{path}: prompt {g['prompt']} diverges at {g['at']} with masked gap {g['gap']}")
    for what in ("top1", "confident"):
        mine, floor = agreement["spec"][what], agreement["plain"][what]
        margin = max(TOP1_DELTA, _noise(floor.se))
        report["agreement"][f"{what}_margin"] = margin
        if not mine.value >= floor.value - margin:
            failures.append(f"agreement {what} {mine.value:.4f} below plain {floor.value:.4f} - {margin:.4f}")
    for q in ("p50", "p99"):
        budget = KL_FACTOR * kl["free"][q] + KL_EPS
        if not kl["grammar"][q] <= budget:
            failures.append(f"KL {q} {kl['grammar'][q]:.3g} above {budget:.3g} (2x the free pair's)")
    report |= {"failures": failures, "passed": not failures}
    return report


def table(report: dict) -> str:
    lines = [f"threshold {report['threshold']['used']:.3g} nats (free first-divergence gaps "
             f"{[round(g, 3) for g in report['threshold']['free_first_divergence_gaps']]})"]
    for path in ("free", "free/lp", "grammar", "grammar/lp"):
        r = report[path]
        lines.append(f"{path:11s} identical {r['identical']}/{report['prompts']}  errors {r['errors']}  first "
                     f"divergence gaps {[None if g['gap'] is None else round(g['gap'], 3) for g in r['diverged']]}")
    for name, a in report["agreement"].items():
        if isinstance(a, dict):
            lines.append(f"agreement {name}: top-1 {a['top1']['value']:.4%} ± {a['top1']['se']:.3%}, confident "
                         f"{a['confident']['value']:.4%} ({a['confident']['n']}), unresolved {a['unresolved']}")
    for mode, v in report["kl"].items():
        lines.append(f"KL plain‖spec {mode}: p50 {v['p50']:.3g} p99 {v['p99']:.3g} max {v['max']:.3g} "
                     f"({v['positions']} positions)")
    lines.append("PASS" if report["passed"] else "FAIL: " + "; ".join(report["failures"]))
    return "\n".join(lines)
