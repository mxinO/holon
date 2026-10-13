"""L0-dist for `std.spec.rejection` (docs/m04-spec.md §5): speculative sampling with deterministic drafts emits tokens
with exactly the target's distribution. 10⁵ independent sequences (one seed each) at fixed seeds, χ² at p = 1e-3 with
cells of expected count < 5 pooled:

- the first token, for a draft inside, at the edge of, and outside the filtered support (top-p), exact zero counts
  outside the support, and p(d) ≈ 1 (the residual's guard);
- the joint law of the first three tokens of a toy Markov target (vocabulary 16, k = 3), the drafter proposing
  (prev + 1) mod 16 — accepted drafts condition the next column, rejections end the step and the next step resumes;
- plants that must fail: the residual keeping the draft (marginal), one coin for every column (joint);
- the token after the last real draft (all accepted, n_draft = 1 and K) is the plain draw from its column's law;
- a coin is below 1: a sure draft is accepted at the one-in-2²⁴ draw that rounds to 1.0 unclamped.

Sampled drafts (Gate B, D5: a drafter publishing the sparse laws it drew from, `SeqTable.draft_q_ids` / `draft_q`):
- the first token for laws of Kq = 4 inside, straddling and outside the filtered support, and equal to p;
- the joint law of three tokens on the Markov target with a drafter sampling each draft from a Kq = 4 law given the
  previous one, on a uniform stream of its own (on the verifier's, the law fails);
- n_draft < k with stale rows past it (tokens the target likes, laws that would accept them): never read — the
  `unmasked-padding` plant (judge/holon_judge/planted.py) must fail it;
- plant: the residual keeping q's mass (drawn from p instead of max(p − q, 0)) moves the first token's law. (The
  deterministic rule applied to sampled drafts is no plant: it is lossless for any proposal independent of the coins,
  only slower — accepting d with p(d) ≤ min(1, p(d)/q(d)).)
- one-hot laws (Kq = 1) verify bitwise as deterministic drafts do; ids past n_draft are never read (any value).
"""
import dataclasses

import numpy as np
import pytest
import torch
from holon.api import Batch, DeviceBatch, SamplingParams, SeqTable
from holon.core.batching import host_plan
from holon.core.build import StepContext
from holon_std.sampling.sampler import Sampler, uniform
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


def _step(sampler, logits, drafts, positions, verify=rejection.verify, n_draft=K, laws=None):
    """One verify step of N rows of width K + 1: (tokens [N, K + 1], n_accept [N]); `laws`: the drafts' sparse laws
    (ids, probabilities [N, K, Kq])."""
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
                     n_draft=torch.as_tensor(n_draft, dtype=torch.long).expand(n).clone(),
                     **({} if laws is None else {"draft_q_ids": laws[0].clone(), "draft_q": laws[1].clone()}))
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


def _markov_sequences(target, verify, drafter=None) -> np.ndarray:
    """The first three tokens of N sequences, speculating k = 3 with drafts (prev + 1) mod V, or what
    `drafter(prev, positions)` samples (drafts, laws)."""
    _, trans = target
    sampler = _sampler(SamplingParams(temperature=1.0))
    prev = torch.zeros(N, dtype=torch.long)                  # context token 0
    out = torch.full((N, 3), -1)
    filled = torch.zeros(N, dtype=torch.long)
    while (filled < 3).any():
        # d1 = prev + 1, d2 = d1 + 1, … unless a drafter samples them
        drafts, laws = drafter(prev, 10 + filled) if drafter else ((prev[:, None] + 1 + torch.arange(K)) % V, None)
        cols = torch.cat([prev[:, None], drafts], 1)         # column j's logits condition on its predecessor
        tokens, n_accept = _step(sampler, trans[cols], drafts, 10 + filled, verify, laws=laws)
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

    # One coin for every column: the joint law moves (the first token's alone does not). The plant intercepts
    # stream 1, so it names where the coins are drawn; which stream that is does not matter for the law.
    class OneCoin(Sampler):
        def uniform(self, ctx, stream=0):
            u = super().uniform(ctx, stream)
            return u.view(-1, K + 1)[:, :1].expand(-1, K + 1).reshape(-1) if stream == 1 else u

    monkeypatch.setattr("tests.test_rejection.Sampler", OneCoin)
    seqs = _markov_sequences(target, rejection.verify)
    counts = np.bincount((seqs[:, 0] * V + seqs[:, 1]) * V + seqs[:, 2], minlength=V ** 3)
    assert _chi2(counts, _markov_law(target)) < 1e-3


@pytest.mark.parametrize("n_draft", [1, K])
def test_the_token_after_the_last_real_draft_has_its_columns_law(target, n_draft):
    # Drafts sure to be accepted up to column n_draft, whose law the next token must have — token 0 (the padding
    # draft past the real ones) the likeliest, so drawing it from a residual would show.
    logits = target[0].clone()
    logits[0] = logits.max() + 1
    rows = torch.full((K + 1, V), -30.0)
    rows[:, 3] = 30.0
    rows[n_draft] = logits
    sampler = _sampler(SamplingParams(temperature=1.0))
    tokens, n_accept = _step(sampler, rows.expand(N, K + 1, V), torch.full((N, K), 3), torch.full((N,), 10),
                             n_draft=n_draft)
    assert (n_accept == n_draft).all()
    last = tokens[:, n_draft].numpy()
    assert _chi2(np.bincount(last, minlength=V), torch.softmax(logits, -1).numpy()) > 1e-3


def test_a_sure_draft_is_accepted_at_the_largest_coin():
    # (2²⁴ − 0.5) / 2²⁴ rounds to 1.0 in fp32: seed 9111193 draws it at position 7 on stream 1, which would reject
    # the greedy row's argmax draft and emit the last vocabulary id from an empty residual.
    assert uniform(torch.tensor([9111193]), torch.tensor([7]), 1).item() < 1
    sampler = Sampler(1, torch.device("cpu"))
    sampler.admit(0, SamplingParams(temperature=0.0, seed=9111193))
    logits = torch.zeros(1, K + 1, V)
    logits[..., 5] = 10.0
    tokens, n_accept = _step(sampler, logits, torch.full((1, K), 5), torch.tensor([6]))     # column 0 at position 7
    assert n_accept.item() == K and (tokens == 5).all()


def test_a_sure_draft_is_accepted_at_a_coin_of_one():
    # The guard itself, whatever the uniform's range: a draft whose residual is empty (p(d) = 1 in fp32) is accepted.
    class CoinOfOne(Sampler):
        def uniform(self, ctx, stream=0):
            u = super().uniform(ctx, stream)
            return torch.ones_like(u) if stream == 1 else u

    sampler = CoinOfOne(1, torch.device("cpu"))
    sampler.admit(0, SamplingParams(temperature=1.0, seed=0))
    logits = torch.full((1, K + 1, V), -200.0)
    logits[..., 5] = 0.0                                  # p(5) = 1: the rest underflows to 0 in fp32
    tokens, n_accept = _step(sampler, logits, torch.full((1, K), 5), torch.tensor([6]))
    assert n_accept.item() == K and (tokens == 5).all()


# Sampled drafts: sparse laws ------------------------------------------------------------------------------------------

def _sample(ids: torch.Tensor, q: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """One draw per row of the laws (ids, q) [N, Kq], as a sampling drafter makes it."""
    return ids.gather(-1, torch.multinomial(q, 1, generator=g))[:, 0]


def _filtered(logits: torch.Tensor, top_p: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(the law after top-p, its support, the tokens outside it, likeliest first)."""
    p = torch.softmax(logits, -1)
    order = p.argsort(descending=True)
    keep = (p[order].cumsum(0) - p[order]) <= top_p
    probs = torch.zeros(V)
    probs[order[keep]] = p[order[keep]] / p[order[keep]].sum()
    return probs, order[keep], order[~keep]


@pytest.mark.parametrize("case", ["inside", "straddling", "outside", "padded", "equal"])
def test_first_token_has_the_target_law_with_sampled_drafts(target, case):
    # Every draft is drawn from its own Kq = 4 law, which may cover tokens outside the filtered support (never emitted),
    # pad its support by repeating an id at zero mass, or equal p exactly (an empty residual: every draft accepted).
    logits, _ = target
    top_p = 0.9
    probs, support, outside = _filtered(logits, top_p)
    g = torch.Generator().manual_seed(5)
    if case == "equal":
        logits = torch.full((V,), -1e4)
        logits[support[:4]] = target[0][support[:4]]
        probs, top_p = torch.softmax(logits, -1), 1.0
        ids = support[:4]
        q = probs[ids].clone()
    else:
        ids = {"inside": support[:4], "straddling": torch.cat([support[-2:], outside[:2]]),
               "outside": outside[:4], "padded": support[:4]}[case]
        q = torch.rand(4, generator=g) + 0.2
        q /= q.sum()
        if case == "padded":                                 # two real entries, then the first id again at mass 0
            ids, q = support[[0, 1, 0, 0]], torch.cat([q[:2] / q[:2].sum(), torch.zeros(2)])
    ids, q = ids.expand(N, K, 4), q.expand(N, K, 4)
    drafts = torch.stack([_sample(ids[:, j], q[:, j], g) for j in range(K)], 1)
    sampler = _sampler(SamplingParams(temperature=1.0, top_p=top_p))
    tokens, n_accept = _step(sampler, logits.expand(N, K + 1, V), drafts, torch.full((N,), 10), laws=(ids, q))
    assert _chi2(np.bincount(tokens[:, 0].numpy(), minlength=V), probs.numpy()) > 1e-3
    assert (n_accept <= K).all() and (case != "equal" or (n_accept == K).all())


def _markov_drafter(target, stream: int):
    """Samples each draft from a Kq = 4 law given the previous token — the target's four likeliest successors, flatter
    than the target (its logits at temperature 2), so drafts are often accepted and the residual still has mass — with
    the request's counter-based uniform on `stream` at the draft's position, as a drafter drawing through
    `Sampler.uniform` does."""
    _, trans = target
    top = trans.topk(4, -1)
    ids_of, q_of = top.indices, torch.softmax(top.values / 2, -1)            # [V, 4] per previous token
    seeds = 1000 + torch.arange(N)                                         # `_sampler`'s

    def draft(prev, start):
        drafts, ids, qs = [], [], []
        for j in range(K):
            ids.append(ids_of[prev]), qs.append(q_of[prev])
            prev = ids[-1].gather(1, rejection.draw(qs[-1], uniform(seeds, start + 1 + j, stream))[:, None])[:, 0]
            drafts.append(prev)
        return torch.stack(drafts, 1), (torch.stack(ids, 1), torch.stack(qs, 1))
    return draft


@pytest.mark.parametrize("stream", [8, 1])
def test_three_tokens_have_the_markov_law_with_sampled_drafts(target, stream):
    # A drafter sampling each draft from its law on a uniform stream of its own (8 and up, api.decoding.Sampler): the
    # joint law holds. On the verifier's coins (stream 1) a draft is correlated with its own acceptance: it moves.
    seqs = _markov_sequences(target, rejection.verify, _markov_drafter(target, stream=stream))
    counts = np.bincount((seqs[:, 0] * V + seqs[:, 1]) * V + seqs[:, 2], minlength=V ** 3)
    p = _chi2(counts, _markov_law(target))
    assert p > 1e-3 if stream == 8 else p < 1e-3


def _stale(target, n_draft: int, verify):
    """N rows of width K + 1 on one law p, whose first `n_draft` drafts are sampled from Kq = 4 laws; past them, stale
    rows: the token p likes best, under a law that would accept it at any coin (q(t) = 0.05 ≤ p(t)). Returns the
    first emitted token, the emitted token after the real drafts (rows that accepted them all), n_accept, and p."""
    logits = target[0]
    p = torch.softmax(logits, -1)
    top = logits.topk(4)
    g = torch.Generator().manual_seed(9)
    ids, q = top.indices.repeat(N, K, 1), torch.softmax(top.values / 2, -1).repeat(N, K, 1)
    drafts = torch.stack([_sample(ids[:, j], q[:, j], g) for j in range(K)], 1)
    t = int(p.argmax())
    drafts[:, n_draft:] = t
    ids[:, n_draft:] = (t + torch.arange(4)) % V
    q[:, n_draft:] = torch.tensor([0.05, 0.05, 0.05, 0.85])
    sampler = _sampler(SamplingParams(temperature=1.0))
    tokens, n_accept = _step(sampler, logits.expand(N, K + 1, V), drafts, torch.full((N,), 10), verify, n_draft,
                             (ids, q))
    rows = n_accept >= n_draft
    return tokens[:, 0], tokens[rows, n_draft], n_accept, p


@pytest.mark.parametrize("n_draft", [1, 2])
def test_rows_past_n_draft_are_never_read(target, n_draft):
    first, after, n_accept, p = _stale(target, n_draft, rejection.verify)
    assert (n_accept <= n_draft).all() and len(after) > 0
    assert _chi2(np.bincount(first.numpy(), minlength=V), p.numpy()) > 1e-3
    assert _chi2(np.bincount(after.numpy(), minlength=V), p.numpy()) > 1e-3


def test_sampled_draft_plants_fail(target, monkeypatch):
    from holon_judge.planted import unmasked_padding

    # Stale rows read as real drafts: accepted past n_draft, and the token after the real drafts loses its law.
    with unmasked_padding():
        _, after, n_accept, p = _stale(target, 1, rejection.verify)
    assert (n_accept > 1).any()
    assert _chi2(np.bincount(after.numpy(), minlength=V), p.numpy()) < 1e-3

    # The residual keeping q's mass (drawn from p, not max(p − q, 0)): the first token's law moves.
    sampled = rejection._sampled
    monkeypatch.setattr(rejection, "_sampled", lambda p, *a: (*sampled(p, *a)[:2], p.clone()))
    first, _, _, p = _stale(target, K, rejection.verify)
    assert _chi2(np.bincount(first.numpy(), minlength=V), p.numpy()) < 1e-3


def test_one_hot_laws_verify_as_deterministic_drafts(target):
    # Kq = 1 at the draft with probability 1 is the deterministic proposal: the same tokens and acceptances, bitwise,
    # on filtered rows of different laws per column with every n_draft — drafts drawn from their columns' laws, so
    # many coins fall near the acceptance bound.
    _, trans = target
    g = torch.Generator().manual_seed(3)
    rows = trans[torch.randint(0, V, (N, K + 1), generator=g)]
    drafts = torch.multinomial(torch.softmax(rows[:, :K], -1).view(-1, V), 1, generator=g).view(N, K)
    n_draft = torch.randint(0, K + 1, (N,), generator=g)
    sampler = _sampler(SamplingParams(temperature=1.0, top_p=0.9))
    positions = torch.randint(0, 100, (N,), generator=g)
    plain = _step(sampler, rows, drafts, positions, n_draft=n_draft)
    one_hot = _step(sampler, rows, drafts, positions, n_draft=n_draft,
                    laws=(drafts[..., None], torch.ones(N, K, 1)))
    assert all(torch.equal(a, b) for a, b in zip(plain, one_hot, strict=True)) and plain[1].sum() > 0


@pytest.mark.parametrize("sampled", [False, True])
def test_ids_past_n_draft_may_be_anything(sampled):
    # A drafter leaves whatever it likes past its real drafts (the test oracle writes ids no vocabulary has): the
    # verifier never reads them, so they cannot index out of range.
    sampler = Sampler(1, torch.device("cpu"))
    sampler.admit(0, SamplingParams(temperature=1.0, seed=0))
    logits = torch.zeros(1, K + 1, V)
    logits[..., 5] = 30.0
    drafts = torch.tensor([[5, V + 7, V + 7]])
    laws = (torch.tensor([[[5], [V + 7], [V + 7]]]), torch.tensor([[[1.0], [float("nan")], [2.0]]]))
    tokens, n_accept = _step(sampler, logits, drafts, torch.tensor([6]), n_draft=1, laws=laws if sampled else None)
    assert n_accept.item() == 1 and tokens[0, :2].tolist() == [5, 5]


def test_an_empty_residual_accepts_a_sampled_draft_at_a_coin_of_one():
    # q ≥ p everywhere (q a hair above p, as rounding may leave it): the residual is empty, so a rejection would have
    # nothing to draw; the draft is accepted whatever the coin.
    class CoinOfOne(Sampler):
        def uniform(self, ctx, stream=0):
            u = super().uniform(ctx, stream)
            return torch.ones_like(u) if stream == 1 else u

    sampler = CoinOfOne(1, torch.device("cpu"))
    sampler.admit(0, SamplingParams(temperature=1.0, seed=0))
    logits = torch.full((1, K + 1, V), -1e4)
    logits[..., :2] = 0.0                                 # p = (½, ½, 0, …)
    laws = (torch.tensor([0, 1]).expand(1, K, 2), torch.full((1, K, 2), 0.5 + 2**-20))
    tokens, n_accept = _step(sampler, logits, torch.tensor([[1, 0, 1]]), torch.tensor([6]), laws=laws)
    assert n_accept.item() == K and tokens[0, :K].tolist() == [1, 0, 1]


def test_a_draft_outside_its_own_law_is_never_accepted(target):
    # A drafter publishing a law that cannot have drawn its draft (q(d) = 0) breaks the contract; the verifier rejects
    # the draft and draws that column from p as past the last real draft, so the emitted token keeps p's law.
    logits = target[0]
    d = int(logits.argmax())                                     # the target's likeliest token, outside the law
    others = torch.tensor([t for t in range(V) if t != d][:2])
    laws = (others.expand(N, K, 2), torch.full((N, K, 2), 0.5))
    sampler = _sampler(SamplingParams(temperature=1.0))
    tokens, n_accept = _step(sampler, logits.expand(N, K + 1, V), torch.full((N, K), d), torch.full((N,), 10),
                             laws=laws)
    assert (n_accept == 0).all()
    assert _chi2(np.bincount(tokens[:, 0].numpy(), minlength=V), torch.softmax(logits, -1).numpy()) > 1e-3
