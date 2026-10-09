"""L2 comparator math on synthetic recordings."""
import pytest
import torch
from holon_judge.compare import Budgets, compare, kl_topk
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


def test_budgets_follow_design_rule():
    from holon_judge.compare import Metrics

    floor = Metrics(n=1, top1=0.98, top1_confident=0.99, kl_mean=0, kl_p50=1e-4, kl_p99=5e-3, kl_max=1, dppl=-0.001,
                    dppl_se=0, missing=0)
    b = Budgets.from_floor(floor)
    assert b == Budgets(top1_min=pytest.approx(0.9775), kl_p50_max=2e-4, kl_p99_max=1e-2,
                        dppl_max=pytest.approx(0.0035))
    assert b.failures(floor) == []
    worse = Metrics(**{**floor.__dict__, "top1": 0.97, "kl_p99": 2e-2})
    assert len(b.failures(worse)) == 2


def test_refuses_different_prompt_sets(case):
    lens, logits, targets, _ = case
    g = _recording(logits, targets, lens)
    other = _recording(logits, targets, lens[::-1])
    with pytest.raises(ValueError, match="different prompt sets"):
        compare(g, other)


def test_expert_calls_off_the_floors_path_fail(case):
    lens, logits, targets, _ = case
    g, floor = _recording(logits, targets, lens), _recording(logits, targets, lens)
    floor.meta["q8_1"] = "mmvq"
    on_path, off_path = _recording(logits, targets, lens), _recording(logits, targets, lens)
    on_path.meta["expert_calls"] = {"mmvq": 40, "mmq": 0}
    off_path.meta["expert_calls"] = {"mmvq": 40, "mmq": 3}
    assert compare(g, on_path, floor).passed
    r = compare(g, off_path, floor)
    assert r.passed is False and r.failures == ("3 expert calls on mmq, the floor emulates mmvq",)
