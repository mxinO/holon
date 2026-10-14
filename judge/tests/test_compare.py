"""L2 comparator math on synthetic recordings."""
import pytest
import torch
from holon_judge.compare import Budgets, compare, compare_drafter, kl_topk
from holon_judge.recording import Recording

V, K = 64, 20
CATS = ["chat", "code", "recall", "agent"]


def _recording(logits: torch.Tensor, targets: torch.Tensor, lens: list[int], k: int = K) -> Recording:
    """What the recorder stores, from full logits [N, V]."""
    lp = logits.double().log_softmax(-1)
    top_lp, top_ids = lp.topk(k, -1)
    tail = lp.scatter(1, top_ids, float("-inf")).logsumexp(-1)
    meta = {"prompts_sha256": "x", "categories": [CATS[i % 4] for i in range(len(lens))], "dtype": "test"}
    target = lp.gather(1, targets[:, None])[:, 0]
    rows = zip(*(t.split(lens) for t in (top_ids, top_lp, target, tail)), strict=True)
    return Recording.from_rows(list(rows), meta)


@pytest.fixture(scope="module")
def case():
    gen = torch.Generator().manual_seed(0)
    lens = [400, 600, 500, 500, 300, 700, 400, 600]
    n = sum(lens)
    logits = torch.randn(n, V, generator=gen) * 3
    targets = torch.multinomial(logits.softmax(-1), 1, generator=gen)[:, 0]

    def noise(scale: float) -> torch.Tensor:
        return logits + scale * torch.randn(n, V, generator=gen)

    return lens, logits, targets, noise


def test_identical_recordings_agree_exactly(case):
    lens, logits, targets, _ = case
    g = _recording(logits, targets, lens)
    r = compare(g, g, floor=g)
    assert r.overall.top1 == 1 and r.overall.kl_max < 1e-12 and r.overall.dppl == 0 and r.overall.missing == 0
    assert r.passed and list(r.categories) == CATS and sum(m.n for m in r.categories.values()) == r.overall.n


def test_floor_level_noise_passes_and_a_shift_fails(case):
    lens, logits, targets, noise = case
    g = _recording(logits, targets, lens)
    # ΔNLL noise is ~ scale / √n per run: at 0.02 the 0.25% ΔPPL margin is > 5 SE, while the shift raises KL 100×.
    floor = _recording(noise(0.02), targets, lens)
    assert compare(g, _recording(noise(0.02), targets, lens), floor).passed
    shifted = compare(g, _recording(noise(0.2), targets, lens), floor)
    assert shifted.passed is False and any("KL p99" in f for f in shifted.failures)
    assert "FAIL" in shifted.table() and shifted.to_json()["passed"] is False


def test_bucketed_kl_is_a_tight_lower_bound(case):
    lens, logits, targets, noise = case
    cand_logits = noise(0.1)
    exact = (logits.double().log_softmax(-1).exp()
             * (logits.double().log_softmax(-1) - cand_logits.double().log_softmax(-1))).sum(-1)
    full, _ = kl_topk(_recording(logits, targets, lens, k=V), _recording(cand_logits, targets, lens, k=V))
    assert torch.allclose(full, exact, atol=1e-6)    # no tail: the bucketed KL is the KL (up to fp32 storage)
    approx, missing = kl_topk(_recording(logits, targets, lens), _recording(cand_logits, targets, lens))
    assert (approx <= exact + 1e-6).all() and approx.sum() > 0.9 * exact.sum() and missing.sum() > 0
    # Masked distributions (a grammar's): a refused token's logprob is -inf in both and carries no mass.
    refused = torch.arange(V) % 3 == 0
    g_lp, c_lp = (x.masked_fill(refused, float("-inf")).double().log_softmax(-1) for x in (logits, cand_logits))
    exact = torch.where(g_lp > float("-inf"), g_lp.exp() * (g_lp - c_lp), 0.0).sum(-1)
    full, _ = kl_topk(*(_recording(x, targets, lens, k=V) for x in (logits.masked_fill(refused, float("-inf")),
                                                                     cand_logits.masked_fill(refused, float("-inf")))))
    assert torch.allclose(full, exact, atol=1e-6)


def test_budgets_follow_design_rule():
    from holon_judge.compare import Metrics

    floor = Metrics(n=1, top1=0.98, top1_confident=0.99, kl_mean=0, kl_p50=1e-4, kl_p99=5e-3, kl_max=1, dppl=-0.001,
                    dppl_se=0.01, missing=0)
    b = Budgets.from_floor(floor, paired_se=0.004)                  # ΔPPL vs floor within 0.25% + 2.5 SE
    assert Budgets.from_floor(floor, paired_se=0.5).dppl_max == pytest.approx(0.0025 + 2.5 * 2 ** 0.5 * 0.01)
    assert b == Budgets(top1_min=pytest.approx(0.9775), kl_p50_max=2e-4, kl_p99_max=1e-2,
                        dppl_max=pytest.approx(0.0125))
    assert b.failures(floor, dppl_vs_floor=-0.012) == []
    worse = Metrics(**{**floor.__dict__, "top1": 0.97, "kl_p99": 2e-2})
    assert len(b.failures(worse, dppl_vs_floor=0.013)) == 3


def test_refuses_different_prompt_sets(case):
    lens, logits, targets, _ = case
    g = _recording(logits, targets, lens)
    other = _recording(logits, targets, lens[::-1])
    with pytest.raises(ValueError, match="different prompt sets"):
        compare(g, other)


def test_expert_calls_off_the_floors_path_fail(case):
    lens, logits, targets, _ = case
    g, floor = _recording(logits, targets, lens), _recording(logits, targets, lens)
    floor.meta |= {"q8_1": "mmvq", "moe_block": "d3-reference@x"}
    on_path, off_path, uncounted = (_recording(logits, targets, lens) for _ in range(3))
    on_path.meta["expert_calls"] = {"mmvq": 40, "mmq": 0}
    off_path.meta["expert_calls"] = {"mmvq": 40, "mmq": 3}
    assert compare(g, on_path, floor).passed
    r = compare(g, off_path, floor)
    assert r.passed is False and r.failures == ("3 expert calls on mmq, the floor emulates mmvq",)
    assert compare(g, uncounted, floor).failures == ("no expert calls counted on mmvq, the floor's path",)
    uncounted.meta["expert_calls"] = {"mmvq": 0, "mmq": 0}                 # counters that never ran
    assert compare(g, uncounted, floor).failures == ("no expert calls counted on mmvq, the floor's path",)
    del floor.meta["moe_block"]                                            # a dense Tier B floor: nothing to count
    assert compare(g, uncounted, floor).passed


def test_the_ppl_guard_is_paired_with_the_floor(case):
    """A consistent NLL shift of 1% fails though top-1 and KL stay within budget; per-sequence noise as large as the
    floor's own (two correct paths), which averages out, passes (an unpaired margin of 0.25% would fail it)."""
    lens, logits, targets, noise = case
    base = noise(0.02)
    seq = torch.repeat_interleave(torch.arange(len(lens)), torch.tensor(lens))

    def nudged(delta: torch.Tensor) -> Recording:    # the base, the target logit moved: NLL by ~ −δ(1 − p)
        out = base.clone()
        out[torch.arange(len(targets)), targets] += delta
        return _recording(out, targets, lens)
    g = _recording(logits, targets, lens)
    floor = nudged(torch.where(seq % 4 < 2, 0.03, -0.03))           # a correct path: per-sequence noise
    shifted = compare(g, nudged(torch.where(seq % 4 < 2, 0.03, -0.03) - 0.04), floor)
    assert any("ΔPPL vs floor" in f for f in shifted.failures) and not any("KL" in f for f in shifted.failures)
    noisy = compare(g, nudged(torch.where(seq % 2 == 0, -0.03, 0.03)), floor)   # another correct path
    assert noisy.passed and abs(noisy.overall.dppl - noisy.floor.dppl) > 0.0025


def test_an_error_in_one_sequence_cannot_widen_its_own_budget(case):
    """The paired SE (sequences as clusters) grows with an error confined to one sequence; capped at √2 × the floor's
    own, it cannot make room for that error (review of the paired guard)."""
    lens, logits, targets, noise = case
    base = noise(0.02)
    g, floor = _recording(logits, targets, lens), _recording(base, targets, lens)
    out = base.clone()
    one = torch.arange(lens[0])                                  # the first sequence's positions only
    out[one, targets[one]] -= 2.0
    r = compare(g, _recording(out, targets, lens), floor)
    d, se = r.dppl_vs_floor
    assert d < 0.0025 + 2.5 * se                                 # within the uncapped budget: it would have passed
    assert any("ΔPPL vs floor" in f for f in r.failures)


def test_the_drafter_rule_tolerates_near_tie_noise(case):
    # Draft heads are judged on few positions (docs/m04-spec.md §5, §12): 288 rows of 24 prompts. A candidate flipping
    # two near-ties beyond the floor's own — a correct path's noise — passes, though L2's fixed 0.25-point bar fails it
    # (one row is 0.35 points); one that flips confident positions fails.
    _, logits, targets, _ = case
    n, lens = 288, [12] * 24
    gen = torch.Generator().manual_seed(1)
    logits, targets = logits[:n], targets[:n]
    g, floor = _recording(logits, targets, lens), _recording(logits + 0.02 * torch.randn(n, V, generator=gen), targets,
                                                             lens)
    top2 = logits.topk(2, -1)
    margin = top2.values[:, 0] - top2.values[:, 1]

    def flipped(rows: torch.Tensor) -> Recording:                 # a correct path's noise, the top-2 swapped at `rows`
        out = logits + 0.02 * torch.randn(n, V, generator=gen)
        out[rows, top2.indices[rows, 0]], out[rows, top2.indices[rows, 1]] = top2.values[rows, 1], top2.values[rows, 0]
        return _recording(out, targets, lens)

    near = compare_drafter(g, flipped(margin.argsort()[:2]), floor)
    assert near.passed and near.drafter["l2_failures"]
    far = compare_drafter(g, flipped(margin.argsort()[-15:]), floor)
    assert far.passed is False and any("confident" in f for f in far.failures)
