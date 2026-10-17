"""The core seams of v1 expert offload (docs/v1-offload.md §4), proven with toys before a plugin uses them: S0 shared
objects and their lifecycle, S1 step hooks, S2 host pools and asynchronous MemOps, S3 fill pools, S5 CPU tiers.

The engine's toy is a kernel provider for `rmsnorm` (the reference norm) whose instances share one object
(`BuildCtx.shared`). It requests a host pool and a fill pool before the pools are materialized and seeds them after
every materialize. Its forward copies two one-value "maps" into telemetry: one the object writes before a launch, one
a MemOp brings from the host pool. At the commit of step c it decides both maps' value for launch c + 2 (c itself), so
every step's telemetry shows the decision of two commits before, whatever the overlap depth and graphs."""
import dataclasses
import os

import numpy as np
import pytest
import torch
from holon.api import Constraint, KernelSpec, MemOp, SamplingParams
from holon.core import plans, platform
from holon.core.build import BuildContext
from holon.core.engine import Engine, EngineConfig
from holon.core.memory import Copies, Ledger, OutOfUnits
from holon.core.plugins import PluginSet, discover
from holon_testing.ref_kernels.kernels import RMSNorm

UNIT = 64                              # bytes per unit of the toy's pools
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


class Toy:
    """The shared object of every toy kernel instance."""

    def __init__(self, ctx) -> None:
        self.ctx, self.events = ctx, ["built"]
        self.map = ctx.static("toy.map", (1,), torch.int64)
        self.telemetry = {name: ctx.static(f"toy.{name}", (1,), torch.int64) for name in ("seen", "landed", "calls")}
        self.decided: dict[int, int] = {}          # launch -> the value decided for it
        self.observed: dict[int, tuple[int, int, int]] = {}    # committed step -> (seen, landed, calls)

    def before_materialize(self, ctx) -> None:
        self.events.append("before_materialize")
        self.host = ctx.pool("toy.host", unit_bytes=UNIT, min_units=2, pref_units=2, device="host", mapped=True)
        self.fill = ctx.pool("toy.fill", unit_bytes=UNIT, min_units=1, pref_units=1 << 20, fill=True)

    def after_materialize(self, ctx) -> None:
        self.events.append("after_materialize")
        self.map.fill_(-1)
        self.fill.view(torch.int64, (self.fill.capacity, UNIT // 8))[0, 0] = -1

    def before_launch(self, step):
        writes = [(self.telemetry["calls"], np.zeros(1, np.int64))]           # a counter zeroed every step
        if (value := self.decided.pop(step.step, None)) is None:
            return writes, []
        landing = MemOp("plugin.toy.host", np.array([value % 2]), "plugin.toy.fill", np.array([0]), lands_by=step.step)
        return writes + [(self.map, np.array([value]))], [landing]

    def after_commit(self, step, telemetry) -> None:
        self.observed[step.step] = tuple(int(telemetry[name][0]) for name in ("seen", "landed", "calls"))
        c = step.step
        self.host.view(torch.int64, (2, UNIT // 8))[c % 2, 0] = c     # read by launch c + 2's copy only
        self.decided[c + 2] = c

    def shutdown(self) -> None:
        self.events.append("shutdown")


class ToyNorm(RMSNorm):
    def __init__(self, ctx, **facts) -> None:
        super().__init__(ctx, **facts)
        self.toy = ctx.shared("tests.toy", Toy)

    def __call__(self, x, weight):
        toy = self.toy
        toy.telemetry["seen"].copy_(toy.map)
        toy.telemetry["landed"].copy_(toy.fill.view(torch.int64, (toy.fill.capacity, UNIT // 8))[0, :1])
        toy.telemetry["calls"].add_(1)
        return super().__call__(x, weight)


def _plugins() -> PluginSet:
    plugins = PluginSet(discover())
    plugins.regs.kernels.add("rmsnorm", KernelSpec("rmsnorm", "tests.toy", ToyNorm, Constraint(
        devices=frozenset({"cpu", "cuda"})), props=frozenset({"graph_safe"})))
    return plugins


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("tiny_qwen3")
    Qwen3ForCausalLM(Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                 num_attention_heads=2, num_key_value_heads=1, head_dim=64,     # GPU kernels' size
                                 eos_token_id=None)).save_pretrained(path)
    return path


@pytest.mark.parametrize("device, overlap", [("cpu", False), ("cpu", True), pytest.param("cuda", True, marks=CUDA)])
def test_shared_objects_live_with_the_engine_and_edit_at_launch_c_plus_2(tiny, device, overlap):
    """S0, S1, S2 and S3 in one engine: one object for every kernel instance, built with the engine's own context
    (an auxiliary model's context gets the same one); its pools requested before the first materialize, reseeded after
    each (the GPU's startup profile materializes twice); a step's device writes and its MemOp landing ahead of its
    forward, the telemetry each step leaves read at its commit — at depth 0, at depth 1 and from CUDA graphs; shut down
    after the drain; the loop pinned to `loop_core`."""
    core = max(os.sched_getaffinity(0))
    affinity = os.sched_getaffinity(0)
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    cfg = EngineConfig(str(tiny), device=device, dtype=dtype, page_size=16, max_seqs=2, max_step_tokens=8,
                       max_model_len=64, overlap=overlap, kernels={"rmsnorm": "tests.toy"}, host_memory=3 * UNIT,
                       loop_core=core, state_memory=1 << 20 if device == "cpu" else None)
    try:
        engine = Engine(cfg, _plugins())
        assert os.sched_getaffinity(0) == {core}
    finally:
        os.sched_setaffinity(0, affinity)
    toy = engine.ctx.shared_objects["tests.toy"]
    assert toy.ctx is engine.ctx and engine.ctx.child("aux").shared("tests.toy", None) is toy
    materialized = 1 if device == "cpu" else 2
    assert toy.events == ["built", "before_materialize"] + ["after_materialize"] * materialized
    assert toy.host.capacity == 2 and toy.host.buffer.numel() == 2 * UNIT and not toy.host.buffer.is_cuda
    kv = engine.ledger.pool("state.kv")
    assert toy.fill.capacity > 1 and (kv.capacity - 1) * kv.tokens_per_unit >= cfg.max_model_len   # past the floor

    if device == "cuda":                       # every copy late: the launches it lands by must wait for it
        run = engine.executor.copies.stream.run
        engine.executor.copies.stream.run = lambda fn, after=None: run(lambda: (torch.cuda._sleep(20_000_000), fn()),
                                                                       after)
    out = engine.generate([list(range(5)), list(range(9, 20))], SamplingParams(temperature=0.0, max_tokens=8))
    assert [len(o) for o in out] == [8, 8]
    steps = sorted(toy.observed)
    assert steps == list(range(1, len(steps) + 1)) and len(steps) > 8
    want = {c: (c - 2, c - 2) if c > 2 else (-1, -1) for c in steps}
    assert {c: toy.observed[c][:2] for c in steps} == want
    decode = {toy.observed[c][2] for c in steps[-4:]}
    assert len(decode) == 1 and decode.pop() > 0                  # zeroed every step: no step counts another's
    if device == "cuda":
        assert engine.stats()["graphs"] > 0
    engine.close()
    assert toy.events[-1] == "shutdown" and toy.host.buffer is None


def test_host_pools_are_exact_once_and_within_their_budget():
    """S2: host pools are allocated as requested — `pref_units` when the host budget allows, else `min_units`, else a
    per-pool report —, never re-materialized, and requested before the first materialize only."""
    ledger = Ledger(torch.device("cpu"), host_memory=10 * UNIT)
    a = ledger.plugin_pool("a", unit_bytes=UNIT, min_units=2, pref_units=6, device="host")
    b = ledger.plugin_pool("b", unit_bytes=UNIT, min_units=3, pref_units=5, device="host")
    assert (a.capacity, b.capacity, a.buffer.numel()) == (6, 3, 6 * UNIT)
    with pytest.raises(OutOfUnits, match=r"plugin.c needs 2 units .* plugin.a holds 6 units .* plugin.b holds 3"):
        ledger.plugin_pool("c", unit_bytes=UNIT, min_units=2, device="host")
    with pytest.raises(ValueError, match="a host pool"):
        ledger.plugin_pool("d", unit_bytes=UNIT, device="host", fill=True)
    buffer = a.buffer
    ledger.materialize(None)
    ledger.free()
    assert a.buffer is buffer
    with pytest.raises(ValueError, match="after the pools were materialized"):
        ledger.plugin_pool("e", unit_bytes=UNIT, min_units=1, device="host")


@pytest.mark.parametrize("budget, sizes", [
    (10_000, (6, 4, 88)),         # fill pools at their preference, paged state takes the rest
    (2_600, (6, 2, 16)),          # the higher priority grows first; paged state keeps its floor (15 units + null)
    (2_000, (1, 1, 16)),          # no room past the floor: fill pools at their minimum
    (1_500, (1, 1, 11)),          # paged state below its floor, down to its own minimum (9 units + null)
    (1_300, None),
])
def test_fill_pools_grow_by_priority_past_the_paged_floor(budget, sizes):
    """S3: after fixed pools, fill pools take their minimum, then grow by priority up to `pref_units` once paged state
    has `paged_min_tokens` (the floor); paged state takes what is left."""
    ledger = Ledger(torch.device("cpu"), paged_min_tokens=60)
    ledger.request("fixed", unit_bytes=100, min_units=2, pref_units=2)
    low = ledger.plugin_pool("low", unit_bytes=100, min_units=1, pref_units=4, fill=True)
    high = ledger.plugin_pool("high", unit_bytes=100, min_units=1, pref_units=6, fill=True, priority=1)
    kv = ledger.request("kv", unit_bytes=100, min_units=9, tokens_per_unit=4, null_unit=True)
    if sizes is None:
        with pytest.raises(OutOfUnits, match="kv"):
            ledger.materialize(budget)
        return
    ledger.materialize(budget)
    assert (high.capacity, low.capacity, kv.capacity) == sizes


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
def test_units_demoted_to_host_and_back_are_bitwise_and_a_late_copy_stalls_its_step(device):
    """S2: MemOps between device and host pools on the copy stream — after a step's work, landing by a later step's
    launch, which waits for a late copy rather than reading what was there; the mapped host pool's device alias
    reads the demoted bytes in place."""
    device = torch.device(device)
    ledger = Ledger(device, host_memory=1 << 20)
    host = ledger.plugin_pool("host", unit_bytes=4096, min_units=2, device="host", mapped=True)
    dev = ledger.plugin_pool("dev", unit_bytes=4096, min_units=4)
    ledger.materialize(None)
    units = dev.view(torch.uint8, (4, 4096))
    units[:2] = torch.randint(0, 256, (2, 4096), dtype=torch.uint8, device=device)
    units[2:] = 0
    want = units[:2].cpu()
    copies = Copies(device, ledger.pools)
    copies.add([MemOp("plugin.dev", np.array([0, 1]), "plugin.host", np.array([0, 1]), after=1)], 1)   # held
    step1 = platform.Event(device)
    step1.record()
    copies.launched(1, step1)                                  # demoted once step 1's work is done
    if device.type == "cuda":                                  # the copy stream is late
        with torch.cuda.stream(copies.stream._stream):
            torch.cuda._sleep(200_000_000)
    copies.add([MemOp("plugin.host", np.array([0, 1]), "plugin.dev", np.array([2, 3]), after=1, lands_by=3)], 2)
    if device.type == "cuda":
        assert not copies.landing[3][0]._event.query()         # not landed when step 3 launches
    copies.launching(3)
    assert torch.equal(units[2:].cpu(), want)                  # read in stream order after the wait
    assert torch.equal(host.view(torch.uint8, (2, 4096)), want)
    assert torch.equal(host.device_view(torch.uint8, (2, 4096)).sum(1).cpu(), want.sum(1, dtype=torch.int64))
    with pytest.raises(ValueError, match="lands_by 2 must be after step 2"):
        copies.add([MemOp("plugin.dev", np.array([0]), "plugin.host", np.array([0]), after=2, lands_by=2)], 2)
    platform.synchronize(device)
    ledger.free(host=True)


def test_kernels_are_selected_by_the_cpu_tier_and_name_it_in_the_lock_class():
    """S5: a kernel constrained to CPU tiers runs only on those, and a plan binding one has the tier in its class's
    first segment; the host's tier is the best flag set its CPU lists."""
    assert platform.cpu_tier("x86_64", {"sse4_2", "avx2", "fma", "f16c"}) == "x86-avx2"
    assert platform.cpu_tier("x86_64", {"avx512f", "avx512bw", "avx512dq", "avx512vl", "avx2", "fma", "f16c"}) == \
        "x86-avx512"
    assert (platform.cpu_tier("x86_64", {"avx"}), platform.cpu_tier("riscv64", set())) == ("x86", "")
    assert platform.fingerprint(torch.device("cpu")).cpu_isa == platform.cpu_tier()
    plugins = PluginSet({})
    for provider, tiers, priority in (("plain", (), 0), ("avx2", ("x86-avx2", "x86-avx512"), 1),
                                      ("avx512", ("x86-avx512",), 2)):
        plugins.regs.kernels.add("toy", KernelSpec("toy", provider, None, Constraint(
            devices=frozenset({"cpu"}), cpu_isa=frozenset(tiers)), priority=priority))
    for tier, chosen in (("x86-avx2", "avx2"), ("x86-avx512", "avx512"), ("arm64", "plain")):
        ctx = BuildContext(plugins, torch.device("cpu"))
        ctx.fingerprint = dataclasses.replace(ctx.fingerprint, gpu="rtx-5080", cpu_isa=tier)
        assert ctx.select_kernel("toy@1", site="s").provider == chosen
        assert plans.lock_class(ctx.fingerprint, ctx.cpu_kernels) == (
            f"rtx-5080+{tier}/{ctx.fingerprint.toolchain}" if chosen != "plain" else ctx.fingerprint.cls)
