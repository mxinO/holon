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
    ops = ("attention.paged_decode", "rmsnorm", "add_rmsnorm", "rope", "silu_mul")
    assert tune.candidates(tiny_engine()) == dict.fromkeys(ops, ["testing.ref_kernels"])
    config = EngineConfig(**{"model": str(tiny_checkpoint)} | cpu_flags)
    assert tune.decode_rate(config, batch=2, prompt_len=8, tokens=8, repeats=1) > 0


def test_f5b_enters_the_lock_only_at_a_five_percent_gain():
    """The split (docs/v1-offload.md D4): threads at half the misses, then the share at the best count; the options
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
