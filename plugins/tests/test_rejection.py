"""L0-dist for `std.spec.rejection` (docs/m04-spec.md §5): speculative sampling with deterministic drafts emits tokens
with exactly the target's distribution. 10⁵ independent sequences (one seed each) at fixed seeds, χ² at p = 1e-3 with
cells of expected count < 5 pooled:

- the first token, for a draft inside, at the edge of, and outside the filtered support (top-p), exact zero counts
  outside the support, and p(d) ≈ 1 (the residual's guard);
- the joint law of the first three tokens of a toy Markov target (vocabulary 16, k = 3), the drafter proposing
  (prev + 1) mod 16 — accepted drafts condition the next column, rejections end the step and the next step resumes;
- plants that must fail: the residual keeping the draft (marginal), one coin for every column (joint).
"""
import dataclasses

import numpy as np
import pytest
import torch
from holon.api import Batch, DeviceBatch, SamplingParams, SeqTable
from holon.core.batching import host_plan
from holon.core.build import StepContext
from holon_std.sampling.sampler import Sampler
from holon_std.spec.rejection import verifier as rejection
from scipy import stats

N, V, K = 100_000, 16, 3


def _chi2(counts: np.ndarray, probs: np.ndarray) -> float:
    """p-value of `counts` against `probs`, cells of expected count < 5 pooled."""
    expected = probs * counts.sum()
    small = expected < 5
    obs = np.append(counts[~small], counts[small].sum())
    exp = np.append(expected[~small], expected[small].sum())
    keep = exp > 0
    assert counts[probs == 0].sum() == 0, "a token outside the support was drawn"
    return stats.chisquare(obs[keep], exp[keep] * obs[keep].sum() / exp[keep].sum()).pvalue


def _step(sampler, logits, drafts, positions, verify=rejection.verify):
    """One verify step of N rows of width K + 1: (tokens [N, K + 1], n_accept [N])."""
    n, width = drafts.shape[0], drafts.shape[1] + 1
    slots = torch.arange(n)
    plan = host_plan(0, slots.numpy().astype(np.int32), positions.numpy().astype(np.int32),
                     np.full(n, width, np.int32), np.full(n, width, np.int32), np.zeros(n * width, np.int64))
    flat_pos = (positions[:, None] + torch.arange(width)).reshape(-1)
    dev = DeviceBatch(input_ids=torch.zeros(n * width, dtype=torch.long), positions=flat_pos,
                      query_start=torch.arange(n + 1) * width, slots=slots, logits_index=torch.arange(n * width),
                      logits_rows=torch.arange(n).repeat_interleave(width))
    table = SeqTable(torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long),
                     n_accept=torch.zeros(n, dtype=torch.long), draft_tokens=drafts.clone(),
                     n_draft=torch.full((n,), drafts.shape[1], dtype=torch.long))
    ctx = StepContext(Batch(plan, dev), {}, seq_table=table, sampler=sampler)
    out, n_accept = verify(logits.reshape(n * width, -1), ctx)
    return out.tokens.view(n, width), n_accept


def _sampler(params: SamplingParams) -> Sampler:
    sampler = Sampler(N, torch.device("cpu"))
    for slot in range(N):
        sampler.admit(slot, dataclasses.replace(params, seed=1000 + slot))
    return sampler


@pytest.fixture(scope="module")
def target():
    g = torch.Generator().manual_seed(0)
    return torch.randn(V, generator=g) * 1.5, torch.randn(V, V, generator=g) * 1.5      # logits; Markov rows


@pytest.mark.parametrize("case", ["inside", "edge", "outside", "certain"])
def test_first_token_has_the_target_law(target, case):
    logits, _ = target
    top_p = 0.9
    p = torch.softmax(logits, -1)
    order = p.argsort(descending=True)
    keep = (p[order].cumsum(0) - p[order]) <= top_p
    support = order[keep]
    probs = torch.zeros(V)
    probs[support] = p[support] / p[support].sum()
    if case == "certain":                                    # p(d) ≈ 1: the residual's mass is ~0
        logits = torch.full((V,), -20.0)
        logits[3] = 20.0
        probs = torch.softmax(logits, -1)
        d, top_p = 3, 1.0
    else:
        d = {"inside": int(support[0]), "edge": int(support[-1]), "outside": int(order[~keep][0])}[case]
    sampler = _sampler(SamplingParams(temperature=1.0, top_p=top_p))
    rows = logits.expand(N, K + 1, V)
    drafts = torch.full((N, K), d)
    tokens, n_accept = _step(sampler, rows, drafts, torch.full((N,), 10))
    first = tokens[:, 0].numpy()
    assert _chi2(np.bincount(first, minlength=V), probs.numpy()) > 1e-3
    assert (n_accept <= K).all()


def _markov_sequences(target, verify) -> np.ndarray:
    """The first three tokens of N sequences, speculating k = 3 with drafts (prev + 1) mod V."""
    _, trans = target
    sampler = _sampler(SamplingParams(temperature=1.0))
    prev = torch.zeros(N, dtype=torch.long)                  # context token 0
    out = torch.full((N, 3), -1)
    filled = torch.zeros(N, dtype=torch.long)
    while (filled < 3).any():
        drafts = (prev[:, None] + 1 + torch.arange(K)) % V   # d1 = prev + 1, d2 = d1 + 1, …
        cols = torch.cat([prev[:, None], drafts], 1)         # column j's logits condition on its predecessor
        tokens, n_accept = _step(sampler, trans[cols], drafts, 10 + filled, verify)
        for j in range(K + 1):
            take = (j <= n_accept) & (filled < 3)
            out[take, filled[take]] = tokens[take, j]
            filled = filled + take.long()
        prev = tokens.gather(1, n_accept[:, None])[:, 0]
    return out.numpy()


def _markov_law(target) -> np.ndarray:
    _, trans = target
    p = torch.softmax(trans, -1).numpy()
    law = p[0][:, None, None] * p[:, :, None] * p[None, :, :]   # P(x1) P(x2|x1) P(x3|x2)
    return law.reshape(-1)


def test_three_tokens_have_the_markov_law(target):
    seqs = _markov_sequences(target, rejection.verify)
    counts = np.bincount((seqs[:, 0] * V + seqs[:, 1]) * V + seqs[:, 2], minlength=V ** 3)
    assert _chi2(counts, _markov_law(target)) > 1e-3


def test_plants_fail(target, monkeypatch):
    # The residual keeping the draft: the first token's law moves.
    logits, _ = target
    sampler = _sampler(SamplingParams(temperature=1.0))
    original_scatter = torch.Tensor.scatter_

    def no_removal(self, dim, index, value):
        return self if value == 0.0 and self.shape[-1] == V else original_scatter(self, dim, index, value)

    monkeypatch.setattr(torch.Tensor, "scatter_", no_removal)
    d = int(torch.softmax(logits, -1).argmax())
    tokens, _ = _step(sampler, logits.expand(N, K + 1, V), torch.full((N, K), d), torch.full((N,), 10))
    monkeypatch.undo()
    assert _chi2(np.bincount(tokens[:, 0].numpy(), minlength=V), torch.softmax(logits, -1).numpy()) < 1e-3

    # One coin for every column: the joint law moves (the first token's alone does not).
    class OneCoin(Sampler):
        def uniform(self, ctx, stream=0):
            u = super().uniform(ctx, stream)
            return u.view(-1, K + 1)[:, :1].expand(-1, K + 1).reshape(-1) if stream == 1 else u

    monkeypatch.setattr("tests.test_rejection.Sampler", OneCoin)
    seqs = _markov_sequences(target, rejection.verify)
    counts = np.bincount((seqs[:, 0] * V + seqs[:, 1]) * V + seqs[:, 2], minlength=V ** 3)
    assert _chi2(counts, _markov_law(target)) < 1e-3
