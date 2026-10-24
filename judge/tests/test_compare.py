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


def test_a_run_on_two_paths_passes_only_against_both_paths_floors(case):
    """An offload deployment's calls take mmq and mmvq (its short passes): judged by a floor per path, each call on a
    path one of them emulates and the run within every floor's budgets — a loose floor alone does not pass it."""
    lens, logits, targets, noise = case
    g = _recording(logits, targets, lens)
    loose, tight = _recording(noise(0.2), targets, lens), _recording(noise(0.02), targets, lens)
    loose.meta |= {"q8_1": "mmq"}
    tight.meta |= {"q8_1": "mmvq"}
    close, far = _recording(noise(0.02), targets, lens), _recording(noise(0.1), targets, lens)
    for run in (close, far):
        run.meta["calls"] = {"mmq/q8_1": 40, "mmvq/q8_1": 3}
    assert compare(g, close, [loose]).failures == ("3 calls on mmvq/q8_1, the floor emulates mmq/q8_1",)
    both = compare(g, close, [loose, tight])
    assert both.passed and [path for path, *_ in both.others] == ["mmvq"]
    assert "calls: mmq/q8_1 40, mmvq/q8_1 3" in both.table()
    r = compare(g, far, [loose, tight])
    assert r.passed is False and r.failures and all(f.startswith("against the mmvq floor: ") for f in r.failures)
    far.meta["calls"] = {"mmq/q8_1": 40, "mmvq/q8_1": 3, "mmq/fp4": 2}           # a path no floor emulates
    assert "2 calls on mmq/fp4, no floor emulates it (mmq/q8_1, mmvq/q8_1)" in compare(g, far, [loose, tight]).failures
    tight.meta["act"] = "fp4"                                                     # floors of two modes
    assert compare(g, close, [loose, tight]).failures[0] == "several floors in different modes: ['fp4', 'q8_1']"


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


def test_calls_off_the_floors_path_or_mode_fail(case):
    """Every quantized call while scoring must have run the floor's path and activation mode (docs/m06-lowp.md D3);
    plant N7: the `fp4` spec's prefill scored against the W4A8 floor fails on the mode alone."""
    lens, logits, targets, _ = case
    g, floor = _recording(logits, targets, lens), _recording(logits, targets, lens)
    floor.meta |= {"q8_1": "mmvq"}                                         # a floor without `act`: q8_1
    on_path, off_path, uncounted = (_recording(logits, targets, lens) for _ in range(3))
    on_path.meta["calls"] = {"mmvq/q8_1": 40, "mmq/q8_1": 0}
    off_path.meta["calls"] = {"mmvq/q8_1": 40, "mmq/q8_1": 3}
    assert compare(g, on_path, floor).passed
    on_path.meta["head_calls"] = {"mmq/q8_1": 9, "mmvq/q8_1": 1}          # an LM head: either path, the floor's mode
    assert compare(g, on_path, floor).passed
    on_path.meta["head_calls"] = {"mmq/fp4": 9, "mmvq/q8_1": 1}           # ... not another mode
    assert compare(g, on_path, floor).failures == ("9 LM-head calls on mmq/fp4, the floor's mode is q8_1",)
    legacy = _recording(logits, targets, lens)
    legacy.meta["expert_calls"] = {"mmvq": 40, "mmq": 3}                   # a pre-M0.6 recording: by path, q8_1
    assert compare(g, legacy, floor).failures == ("3 calls on mmq/q8_1, the floor emulates mmvq/q8_1",)
    r = compare(g, off_path, floor)
    assert r.passed is False and r.failures == ("3 calls on mmq/q8_1, the floor emulates mmvq/q8_1",)
    assert compare(g, uncounted, floor).failures == ("no calls counted on mmvq/q8_1, the floor's path and mode",)
    uncounted.meta["calls"] = {"mmvq/q8_1": 0, "mmq/q8_1": 0}             # counters that never ran
    assert compare(g, uncounted, floor).failures == ("no calls counted on mmvq/q8_1, the floor's path and mode",)
    floor.meta |= {"q8_1": "mmq", "act": "q8_1"}                           # N7: the W4A8 floor ...
    w4a4 = _recording(logits, targets, lens)
    w4a4.meta["calls"] = {"mmq/fp4": 12, "mmvq/q8_1": 0}                   # ... of a native W4A4 prefill
    assert compare(g, w4a4, floor).failures == ("no calls counted on mmq/q8_1, the floor's path and mode",
                                                "12 calls on mmq/fp4, the floor emulates mmq/q8_1")
    floor.meta["act"] = "fp4"
    assert compare(g, w4a4, floor).passed
    del floor.meta["q8_1"]                                                 # A16: no activation quantization ...
    assert compare(g, uncounted, floor).passed
    decode = _recording(logits, targets, lens)
    decode.meta["calls"] = {"decode/bf16": 30, "prefill/fp8": 0}           # ... so only bf16 calls (FP8's W8A16)
    assert compare(g, decode, floor).passed
    decode.meta["calls"]["prefill/fp8"] = 2
    assert compare(g, decode, floor).failures == ("2 calls on prefill/fp8, the floor quantizes no activations "
                                                  "(bf16 only)",)
    # D2's per-shape floor (docs/m07-perf.md §6): each shape on the floor's path at its rows, in its mode, on its
    # class at its step size
    floor.meta |= {"q8_1": "rows", "act": "q8_1", "step_rows": 8, "crossover": "rtx-5080",
                   "routes": {"iq3_s/5120x17408": "mmq", "q4_k/5120x1024": "mmvq"}}
    mixed = _recording(logits, targets, lens)
    mixed.meta |= {"calls": {"mmvq/q8_1": 40, "mmq/q8_1": 9}, "gpu_class": "rtx-5080", "max_step_tokens": 8,
                   "mmvq_rows": {"iq3_s/5120x17408": 7, "q4_k/5120x1024": 8}}
    assert compare(g, mixed, floor).passed
    mixed.meta |= {"calls": {"mmvq/q8_1": 40, "mmq/fp4": 9}, "gpu_class": "rtx-5090", "max_step_tokens": 4,
                   "mmvq_rows": {"iq3_s/5120x17408": 8}}
    assert compare(g, mixed, floor).failures == (
        "the candidate ran 'rtx-5090' for gpu_class, the floor's 'rtx-5080'",
        "the candidate ran 4 for max_step_tokens, the floor's 8",
        "iq3_s/5120x17408: the candidate runs mmvq at 8 rows, the floor emulates mmq",
        "q4_k/5120x1024: the candidate runs no kernel at 8 rows, the floor emulates mmvq",
        "9 calls on mmq/fp4, the floor's mode is q8_1")


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
