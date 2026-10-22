"""Bounded comparison of engine configurations: cluster standard errors, the checks, a clean CPU run end to end, and
planted bugs it must catch."""
import contextlib
import math

import pytest
import torch
from holon.core.engine import Engine, EngineConfig
from holon_judge import bounded
from holon_judge.bounded import Agreement, Divergence, Mean, Trace
from holon_judge.planted import PLANTS

CONFIGS = {"prefix on, depth 1": {"prefix_cache": True, "overlap": True},
           "prefix on, depth 0": {"prefix_cache": True, "overlap": False},
           "prefix off, depth 1": {"prefix_cache": False, "overlap": True}}
MIXED = {"mixed 8": {"prefix_cache": False, "overlap": False, "chunk_with_decode": 8},       # docs/m07-perf.md §3.1
         "mixed 8, one sequence": {"prefix_cache": False, "overlap": False, "chunk_with_decode": 8, "max_seqs": 1}}


def test_mean_has_a_cluster_robust_standard_error():
    hits = torch.tensor([1, 1, 1, 1, 0, 0, 1, 1], dtype=torch.bool)
    clusters = torch.tensor([0, 0, 0, 0, 5, 5, 2, 2])
    m = Mean.of(hits, clusters)
    # per cluster: sums (4, 0, 2), sizes (4, 2, 2); mean 6/8; residuals (1, -1.5, 0.5)
    assert m.value == 0.75 and m.n == 8 and math.isclose(m.se, math.sqrt(1.5 * 3.5) / 8)
    assert math.isnan(Mean.of(hits[:0], clusters[:0]).value)


def _agreement(top1, confident, se=0.0):
    return Agreement(Mean(top1, se, 1000), Mean(confident, se, 1000), Mean(0.0, se, 1000))


def test_agreement_may_fall_by_the_l2_slack_or_the_noise():
    floor = _agreement(0.98, 0.995)
    assert all(ok for ok, _ in _agreement(0.978, 0.993).checks(floor).values())
    assert not any(ok for ok, _ in _agreement(0.97, 0.99).checks(floor).values())
    ok, margin = _agreement(0.97, 0.995, se=0.0).checks(_agreement(0.98, 0.995, se=0.003))["top1"]
    assert ok and math.isclose(margin, 2.5 * math.sqrt(2) * 0.003)          # the floor's noise only
    assert not _agreement(0.97, 0.995, se=0.05).checks(floor)["top1"][0]     # a noisy candidate earns no margin


def test_kl_quantiles_may_double_or_exceed_by_the_noise():
    floor = Divergence(Mean(0.0, 0.0, 1), 1e-5, 1e-2, 1e-7, 1e-4)
    assert Divergence(Mean(0.0, 0.0, 1), 1.9e-5, 1.9e-2, 1e-7, 1e-4).checks(floor) == {
        "kl_p50": (True, 1e-5), "kl_p99": (True, 1e-2)}
    noisy_floor = Divergence(Mean(0.0, 0.0, 1), 1e-5, 1e-2, 1e-5, 1e-4)
    ok, margin = Divergence(Mean(0.0, 0.0, 1), 3e-5, 1e-2, 0.0, 1e-4).checks(noisy_floor)["kl_p50"]
    assert ok and math.isclose(margin, 2.5 * math.sqrt(2) * 1e-5)
    assert not Divergence(Mean(0.0, 0.0, 1), 3e-5, 1e-2, 1e-4, 1e-4).checks(floor)["kl_p50"][0]


def test_kl_quantile_standard_errors_resample_whole_clusters():
    values = torch.tensor([1.0, 1.0, 1.0, 5.0, 5.0, 5.0], dtype=torch.float64)
    q = torch.tensor([0.5], dtype=torch.float64)
    _, same = bounded._quantiles(values, torch.tensor([0, 1, 2, 0, 1, 2]), q)    # every cluster holds both values
    _, apart = bounded._quantiles(values, torch.tensor([0, 0, 0, 1, 1, 1]), q)   # clusters differ
    assert same.item() == 0 and apart.item() > 1


@pytest.fixture(scope="module")
def run(tiny_checkpoint, tiny_prompts):
    trace = Trace.from_prompts(tiny_prompts, sessions=8, turns=3, system_len=24, chunk_len=5, max_tokens=12)

    def build(**overrides):
        return Engine(EngineConfig(model=str(tiny_checkpoint), device="cpu", dtype=torch.float32, page_size=4,
                                   max_model_len=trace.max_len, state_memory=1 << 21,
                                   **{"max_step_tokens": 32, "max_seqs": 4} | overrides))

    def compare(plant=None, configs=CONFIGS):
        with PLANTS[plant]() if plant else contextlib.nullcontext():
            return bounded.compare_configs(build, configs, trace, k=5)
    return compare


@pytest.mark.slow
def test_configurations_agree_with_the_reference_and_the_prefix_cache_saves(run):
    report = run(configs=CONFIGS | MIXED)
    results = report["results"]
    assert results.pop("mixed 8, one sequence")["failures"] == ["no step mixed decode rows with prefill chunks"]
    assert all(r["passed"] for r in results.values()), bounded.table(report)
    assert results["baseline"]["top1"]["n"] == results["prefix on, depth 1"]["top1"]["n"] == 8 * 3 * 12
    assert results["prefix on, depth 1"]["top1"]["value"] > 0.99
    assert results["baseline"]["tokens_saved"] == 0 and results["prefix on, depth 1"]["tokens_saved"] > 0
    assert results["baseline"]["mixed_steps"] == 0 and results["mixed 8"]["mixed_steps"] > 0


@pytest.mark.slow
@pytest.mark.parametrize("plant, broken", [
    ("adoption-off-by-one", {"prefix on, depth 1", "prefix on, depth 0"}),
    ("wrong-block", {"prefix on, depth 1", "prefix on, depth 0"}),
    ("stale-device-token", {"prefix on, depth 1", "prefix off, depth 1"})])     # the rare one needs a long trace
def test_a_planted_bug_fails_exactly_the_configurations_using_its_path(run, plant, broken):
    report = run(plant)
    assert {name for name, r in report["results"].items() if not r["passed"]} == broken, bounded.table(report)
