"""`tune`: coordinate descent over providers with a scripted rate, and the decode benchmark on a CPU engine."""
from holon.core.engine import EngineConfig
from holon_tools import tune


def test_each_op_takes_its_fastest_provider():
    speed = {"fast": 3.0, "slow": 1.0, "ref": 0.5}
    calls = []

    def rate(pins):
        calls.append(pins)
        return sum(speed[p] for p in pins.values())

    ops = {"attn": ["ref", "fast", "slow"], "norm": ["ref", "slow"], "rope": ["ref"]}
    pins, rates = tune.tune({"attn": "ref", "norm": "ref", "rope": "ref"}, ops, rate)
    assert pins == {"attn": "fast", "norm": "slow", "rope": "ref"}
    assert rates == {"attn": {"ref": 1.5, "fast": 4.0, "slow": 2.0}, "norm": {"ref": 4.0, "slow": 4.5}}
    assert len(calls) == 4                    # the attention winner is not re-measured as norm's incumbent


def test_the_pinned_provider_stays_within_the_noise_margin():
    speed = {"pinned": 100.0, "other": 102.0}
    pins, _ = tune.tune({"attn": "pinned"}, {"attn": ["other", "pinned"]}, lambda p: speed[p["attn"]])
    assert pins == {"attn": "pinned"}
    pins, _ = tune.tune({"attn": "pinned"}, {"attn": ["other", "pinned"]}, lambda p: speed[p["attn"]], margin=0.01)
    assert pins == {"attn": "other"}


def test_candidates_and_decode_rate_on_cpu(tiny_engine, tiny_checkpoint, cpu_flags):
    assert tune.candidates(tiny_engine()) == {
        "attention.paged_decode": ["std.mixers.gqa"], "rmsnorm": ["std.ops"], "add_rmsnorm": ["std.ops"],
        "rope": ["std.ops"], "silu_mul": ["std.ffn"]}
    config = EngineConfig(**{"model": str(tiny_checkpoint)} | cpu_flags)
    assert tune.decode_rate(config, batch=2, prompt_len=8, tokens=8, repeats=1) > 0
