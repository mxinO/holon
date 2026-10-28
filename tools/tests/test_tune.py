"""`tune`: coordinate descent over providers with a scripted rate, and the decode benchmark on a CPU engine."""
import pytest
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
    ops = ("attention.paged_decode", "rmsnorm", "add_rmsnorm", "rope", "silu_mul")
    assert tune.candidates(tiny_engine()) == dict.fromkeys(ops, ["testing.ref_kernels"])
    config = EngineConfig(**{"model": str(tiny_checkpoint)} | cpu_flags)
    assert tune.decode_rate(config, batch=2, prompt_len=8, tokens=8, repeats=1) > 0
    samples = tune.step_times(config, chunks=(8, 16, 64), decode=4, repeats=1)
    assert [(t, r) for t, r, _ in samples] == [(8, 1), (16, 1)] + [(r, r) for r in range(1, config.max_seqs + 1)]
    assert all(s > 0 for *_, s in samples)


def test_the_clock_is_the_least_squares_step_cost():
    samples = [(t, r, 0.01 + 2e-4 * t + 3e-3 * r) for t, r in ((256, 1), (1024, 1), (1, 1), (4, 4))]
    assert tune.clock(samples) == pytest.approx((0.01, 2e-4, 3e-3))
    assert tune.clock([(t, r, 0.01 - 1e-3 * r) for t, r in ((256, 1), (1024, 1), (1, 1), (4, 4))])[2] == 0.0
    # the 27B's steps on an RTX 5080 (ms): fit in relative error, the decode steps are not lost to the prefill ones
    steps = [(256, 1, 107.5), (512, 1, 197.8), (1024, 1, 390.2), (2048, 1, 793.3), (1, 1, 15.5), (2, 2, 16.7),
             (3, 3, 19.9), (4, 4, 20.0)]
    a, b, c = tune.clock([(t, r, ms / 1e3) for t, r, ms in steps])
    assert all(abs((a + b * t + c * r) * 1e3 / ms - 1) < 0.06 for t, r, ms in steps)


def test_f5b_enters_the_lock_only_at_a_five_percent_gain():
    """The split (docs/dev/v1-offload.md D4): threads at half the misses, then the share at the best count; the options
    (the table spelled out to n misses, the tier, the threads) only when the best gains ≥ 5% over F5a."""
    assert tune.plan(0.5, 6) == "1,1,2,2,3,3" and tune.plan(0.25, 4) == "0,1,1,1"

    def rate(peak: float):
        def measure(options):
            if not options.get("moe.cpu_plan"):
                return 100.0
            share = int(options["moe.cpu_plan"].split(",")[-1]) / 10
            return peak - 20 * abs(share - 0.4) - (0 if options["cpu_task.threads"] == "4" else 1)
        return measure
    options, record = tune.split(rate(110.0), 10, "x86-avx2")
    assert options == {"moe.cpu_plan": tune.plan(0.4, 10), "moe.cpu_isa": "x86-avx2", "cpu_task.threads": "4"}
    assert record["best"] == {"share": 0.4, "threads": 4} and record["taken"]
    options, record = tune.split(rate(104.0), 10, "x86-avx2")
    assert options == {} and not record["taken"] and 0 < record["gain"] < tune.GAIN


def test_ssd_rates_go_through_the_level_and_leave_no_files(tmp_path):
    got = tune.ssd_rates(str(tmp_path), 1 << 16, threads=2, seconds=0.2, write_seconds=0.2, gib=1)
    assert all(got[k] > 0 for k in ("read", "write", "mixed", "read_beside")) and 0 < got["read_share"] < 1
    assert not list(tmp_path.iterdir())
