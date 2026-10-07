"""Bounded comparison of engine configurations: cluster standard errors, the margin, and a CPU run end to end."""
import math

import torch
from holon.core.engine import Engine, EngineConfig
from holon_judge import bounded
from holon_judge.bounded import Rate, Trace


def test_rate_has_a_cluster_robust_standard_error():
    hits = torch.tensor([1, 1, 1, 1, 0, 0, 1, 1], dtype=torch.bool)
    seq = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2])
    r = Rate.of(hits, seq)
    # per sequence: hits (4, 0, 2), sizes (4, 2, 2); p = 6/8; residuals (1, -1.5, 0.5)
    assert r.value == 0.75 and r.n == 8 and math.isclose(r.se, math.sqrt(1.5 * 3.5) / 8)


def test_the_margin_is_the_l2_slack_or_the_noise_whichever_is_larger():
    floor = Rate(0.98, 0.0, 1000)
    assert Rate(0.978, 0.0, 1000).within(floor) == (True, bounded.TOP1_DELTA)
    assert not Rate(0.97, 0.0, 1000).within(floor)[0]
    ok, margin = Rate(0.97, 0.004, 1000).within(Rate(0.98, 0.003, 1000))
    assert ok and math.isclose(margin, 2.5 * 0.005)


def test_configurations_agree_with_the_reference_and_the_prefix_cache_saves(tiny_checkpoint, tiny_prompts):
    trace = Trace.from_prompts(tiny_prompts, sessions=3, turns=2, system_len=12, chunk_len=4, max_tokens=6)

    def build(**overrides):
        return Engine(EngineConfig(model=str(tiny_checkpoint), device="cpu", dtype=torch.float32, page_size=4,
                                   max_seqs=4, max_model_len=trace.max_len, state_memory=1 << 20,
                                   **{"max_step_tokens": 16} | overrides))

    configs = {"prefix on, depth 1": {"prefix_cache": True, "overlap": True}}
    report = bounded.compare_configs(build, configs, trace, k=5)
    assert report["passed"], bounded.table(report)
    base, cand = report["results"]["baseline"], report["results"]["prefix on, depth 1"]
    assert base["top1"]["n"] == cand["top1"]["n"] == 3 * 2 * 6
    assert cand["top1"]["value"] > 0.99 and base["tokens_saved"] == 0 and cand["tokens_saved"] > 0
