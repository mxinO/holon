"""L2's reference form (docs/v1-glm.md D15a) on synthetic recordings: budgets from pairs that should agree, a
candidate at their noise passing, a shift and a one-sided perplexity loss failing, the reference's preconditions."""
import dataclasses

import pytest
import torch
from holon_judge import reference
from holon_judge.recording import Recording


def _recording(logits: torch.Tensor, targets: torch.Tensor, lens: list[int], k: int = 20) -> Recording:
    """What the recorder stores, from full logits [N, V] (`test_compare`'s)."""
    lp = logits.double().log_softmax(-1)
    top_lp, top_ids = lp.topk(k, -1)
    tail = lp.scatter(1, top_ids, float("-inf")).logsumexp(-1)
    meta = {"prompts_sha256": "x", "categories": ["chat"] * len(lens), "dtype": "test"}
    rows = zip(*(t.split(lens) for t in (top_ids, top_lp, lp.gather(1, targets[:, None])[:, 0], tail)), strict=True)
    return Recording.from_rows(list(rows), meta)

ORACLE = "judge/tests/test_reference.py"          # a tracked file standing for the model's tiny-model HF match


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    gen = torch.Generator().manual_seed(1)
    lens = [300 + 50 * i for i in range(12)]
    n = sum(lens)
    logits = torch.randn(n, 64, generator=gen) * 3
    targets = torch.multinomial(logits.softmax(-1), 1, generator=gen)[:, 0]

    def rec(scale: float, shift: torch.Tensor | None = None, **meta) -> Recording:
        r = _recording(logits + scale * torch.randn(n, 64, generator=gen) + (0 if shift is None else shift),
                       targets, lens)
        return dataclasses.replace(r, meta=r.meta | {"prompts": "p.jsonl"} | meta)

    path = tmp_path_factory.mktemp("ref") / "v1-reference.safetensors"
    ref = rec(0.0, kind="reference", oracle=ORACLE)
    ref.save(path)
    return path, ref, rec, targets


def test_budgets_pass_a_candidate_at_the_pairs_noise_and_fail_a_shift(case):
    path, ref, rec, targets = case
    budgets = reference.calibrate(path, [("cross-card", ref, rec(0.05))], [("chunked", ref, ref)])
    assert budgets.pairs[1]["bitwise"] and not budgets.pairs[0]["bitwise"] and 0 <= budgets.false_fail < 0.5
    budgets.save(reference.budgets_path(path))
    loaded = reference.Budgets.load(reference.budgets_path(path))
    assert loaded == budgets and loaded.reference == reference.sha256(path)
    clean = reference.judge(ref, rec(0.05), loaded, reference.sha256(path))
    assert clean.passed, clean.table()
    shifted = reference.judge(ref, rec(0.3), loaded)
    assert not shifted.passed and any("KL p50" in f for f in shifted.failures)
    # one-sided ΔPPL: the target's logit lowered everywhere loses perplexity at KL the pairs allow …
    worse = torch.zeros(len(targets), 64).scatter_(1, targets[:, None], -0.05)
    loss = reference.judge(ref, rec(0.0, worse), loaded)
    assert any("one-sided" in f for f in loss.failures), loss.table()
    # … and the same gain is no failure
    assert not any("one-sided" in f for f in reference.judge(ref, rec(0.0, -worse), loaded).failures)
    assert reference.judge(ref, rec(0.05), loaded, "0" * 64).failures[0].startswith("the budgets were calibrated")


def test_the_reference_needs_its_oracle_and_a_noisy_source_pair(case, tmp_path):
    path, ref, rec, _ = case
    with pytest.raises(ValueError, match="no source pair differs in bits"):
        reference.calibrate(path, [("same", ref, ref)])
    bare = tmp_path / "bare.safetensors"
    dataclasses.replace(ref, meta=ref.meta | {"oracle": "judge/tests/no_such_test.py"}).save(bare)
    with pytest.raises(ValueError, match="no tiny-model HF match"):
        reference.calibrate(bare, [("cross-card", ref, rec(0.05))])
