"""holon_judge tiers on the CPU (docs/v1-kv-tiers.md acceptance 2, §6): the clean checks pass on a tiny dense and a
tiny hybrid model with std.state_tier, each tier plant gets its calibrated verdict, the checkpoint policies' plants
fail `bounded --policies`, and a `branch` checkpoint resumes bitwise as the state recomputed with equal chunks."""
import json
import random
import types

import pytest
import torch
from holon.api import SamplingParams
from holon.core.engine import Engine, EngineConfig
from holon_judge import bounded, tiers
from holon_judge.__main__ import main
from holon_judge.planted import PLANTS, TIER_PLANTS

# Loads due a step or more out (gated), write-back admission: the pressure configuration of run-identity.
PRESSURE = {"state_tier.step_cost": "1,0,0", "state_tier.h2d_rate": "20000", "state_tier.stall_bytes": "0",
            "state_tier.host.admit": "evict"}
TIGHT_HOST = {"state_tier.host.bytes": "520KiB"}     # 65 page units, the floor 64: the host level evicts


def _trace(vocab: int, sessions: int, system: int, chunks: tuple[int, int], max_tokens: int,
           starts: tuple[int, ...] = (), turns: int = 4) -> bounded.Trace:
    rng = random.Random(0)
    return bounded.Trace([rng.randrange(vocab) for _ in range(system)],
                         [[[rng.randrange(vocab) for _ in range(rng.randrange(*chunks))] for _ in range(turns)]
                          for _ in range(sessions)], max_tokens, starts)


def _builder(path, **base):
    def build(tier: bool = True, options: dict | None = None, **overrides):
        cfg = dict(model=str(path), device="cpu", dtype=torch.float32, page_size=16, max_seqs=4, max_step_tokens=64,
                   host_memory=64 << 20, slots={"tier": "std.state_tier"} if tier else {}) | base
        cfg["plugin_options"] = cfg.get("plugin_options", {}) | (options or {})
        return Engine(EngineConfig(**cfg | overrides))
    return build


def _setup(build, trace, pressure, tight=None, loops=("identity",)):
    deferred = lambda i=0: tiers.Deferred(i)                      # noqa: E731
    clean = {"invariance": tiers.invariance(build, trace, deferred)} | {
        name: tiers.identity(build, trace, deferred, loop=tiers.LOOPS[name], options=pressure) for name in loops}
    run = {"invariance": lambda: tiers.invariance(build, trace, deferred)} | {
        name: lambda name=name: tiers.identity(build, trace, deferred, 3, clean[name], loop=tiers.LOOPS[name],
                                               options=pressure) for name in loops}
    if tight is not None:                     # a host level at its floor (interior-drop's), publish admission
        clean["tight"] = tiers.identity(build, trace, deferred, 1, options=tight)
        run["tight"] = lambda: tiers.identity(build, trace, deferred, 1, clean["tight"], options=tight)
    return types.SimpleNamespace(build=build, trace=trace, clean=clean, run=run)


@pytest.fixture(scope="module")
def dense(tiny_checkpoint):
    build = _builder(tiny_checkpoint, max_model_len=256, state_memory=17 * 16 * 512,   # 17 pages of 16 tokens
                     plugin_options={"state_tier.host.bytes": "4MiB", "state_tier.host.pages": "1"})
    return _setup(build, _trace(512, 12, 64, (10, 40), 8), PRESSURE, TIGHT_HOST | {"state_tier.step_cost": "1,0,0"})


@pytest.fixture(scope="module")
def dense_ssd(tiny_checkpoint, tmp_path_factory):
    """The dense configuration with the SSD level under a host level at its floor: returns come from the level."""
    root = tmp_path_factory.mktemp("ssd")
    build = _builder(tiny_checkpoint, max_model_len=256, state_memory=17 * 16 * 512, plugin_options={
        "state_tier.host.bytes": "520KiB", "state_tier.host.pages": "1", "state_tier.levels": "ssd",
        "state_tier.ssd.dir": str(root / "level"), "state_tier.ssd.bytes": "8MiB"})
    m = _setup(build, _trace(512, 12, 64, (10, 40), 8), PRESSURE)
    deferred = lambda i=0: tiers.Deferred(i)                      # noqa: E731
    m.clean["restart"] = tiers.restart(build, m.trace, deferred, root / "restart")
    m.run["restart"] = lambda: tiers.restart(build, m.trace, deferred, root / "restart")
    return m


@pytest.fixture(scope="module")
def tiny_hybrid(tmp_path_factory):
    """The tiny Qwen3.5 with its recurrent state decaying slowly (A_log -6: ≈ 0.85 over a 64-token block), so a
    checkpoint resumed at the wrong position shows in what follows (with the random A_log, the state of a block ago is
    forgotten within a block)."""
    from holon_testing.qwen3_5_files import save, tiny_hf

    model = tiny_hf(4, seed=1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("A_log"):
                p.fill_(-6.0)
    return save(model, tmp_path_factory.mktemp("tiny_qwen3_5"))


@pytest.fixture(scope="module")
def hybrid(tiny_hybrid):
    build = _builder(tiny_hybrid, max_model_len=512, state_memory=920 << 10, plugin_options={
        "state_tier.host.bytes": "24MiB", "state_tier.host.pages": "0.5", "recurrent.checkpoints": "6",
        "recurrent.policy": "prompt_end,branch,answer:1"})
    return _setup(build, _trace(320, 8, 140, (20, 60), 8), PRESSURE | {"state_tier.h2d_rate": "2e6"})


@pytest.mark.parametrize("model", ["dense", "hybrid"])
def test_the_clean_checks_pass(model, request):
    clean = request.getfixturevalue(model).clean
    inv, ident = clean["invariance"], clean["identity"]
    assert inv["verdict"] == ident["verdict"] == "pass", (inv["failures"], ident["failures"])
    assert inv["hybrid"] == (model == "hybrid") and inv["tested"]["promoted_into_place"] > 0
    assert inv["tested"]["page_loads"] >= tiers.MIN_LOADS and not inv["reference"].get("page_loads")
    assert min(ident["loads"]) >= tiers.MIN_CONCURRENT_LOADS and ident["counts"][0]["aborted_gated"] > 0
    if model == "hybrid":
        assert inv["tested"]["payload_loads"] > 0
        null = tiers.null_test(request.getfixturevalue(model).build, request.getfixturevalue(model).trace)
        assert null["verdict"] == "pass" and null["tokens_saved"] > 0
    else:                                     # the host level at its floor evicts, leaf first
        assert clean["tight"]["verdict"] == "pass" and clean["tight"]["counts"][0].get("tier.evicted_pages", 0) > 0


# Tier plants caught on the CPU. Not here (the evidence records each; the GPU judges the stream-timing ones):
# - load-before-store: a redundant guard for the host level (D4's free rule and a load's `after` order the read after
#   the store); plugins/tests/test_state_tier.py holds the tier's own ordering on deferred streams;
# - load-reuse-unwaited: with copies at most 2 launches late, the late copy lands before the units' next writer (its
#   own test below defers them up to 8);
# - swap-onto-lower: recomputed and cached bytes give the same greedy outputs on the tiny fp32 model;
# - adopt-source-reused and its join variant: a CPU step's adopt has run by the end of its launch (the GPU judges them
#   in the `fork` loop).
@pytest.mark.parametrize("plant", ["promote-unwaited", "waiter-unref", "store-unwaited", "promote-wrong-unit",
                                   "gate-on-landing", "promote-preempts", "kids-any-level", "interior-drop"])
def test_a_tier_plant_is_caught(plant, dense):
    m = dense
    factory, checks = TIER_PLANTS[plant]
    result = tiers.judge_plant(plant, factory, ("tight",) if plant == "interior-drop" else checks, m.run, m.clean)
    assert result["verdict"] == "caught", result


def test_a_hybrid_resumes_through_the_level_bitwise(tiny_hybrid, tmp_path):
    """Tier invariance with pages through the SSD level, and checkpoints written back to it: the host level at its
    floor for pages and one unit over it for each payload pool (a checkpoint read back from the level is the bare
    tier's test, plugins/tests/test_state_tier_level.py). `branch` only: a branch point's checkpoint is
    never superseded, while a dropped `prompt_end` (superseded by the next turn's) or tentative one changes which
    requests hit in one order and not the other (the hit sets, judged by `bounded`, not the bytes)."""
    build = _builder(tiny_hybrid, max_model_len=512, state_memory=920 << 10, plugin_options={
        "state_tier.host.bytes": "2177928", "state_tier.host.pages": str(2105344 / 2177928),
        "recurrent.checkpoints": "6", "recurrent.policy": "branch", "state_tier.levels": "ssd",
        "state_tier.ssd.dir": str(tmp_path / "level"), "state_tier.ssd.bytes": "32MiB"})
    got = tiers.invariance(build, _trace(320, 8, 140, (20, 60), 8), lambda i=0: tiers.Deferred(i))
    assert got["verdict"] == "pass", got["failures"]
    assert got["tested"]["tier.level_reads"] >= tiers.MIN_LOADS and got["tested"]["payload_loads"] > 0
    assert got["tested"]["tier.level_units.plugin.ssm_state.checkpoints"] > 0      # checkpoints written back too


def test_the_level_below_host_passes_the_checks(dense_ssd):
    clean = dense_ssd.clean
    assert {c: clean[c]["verdict"] for c in ("invariance", "identity", "restart")} == dict.fromkeys(
        ("invariance", "identity", "restart"), "pass"), {c: clean[c]["failures"] for c in clean}
    assert clean["invariance"]["tested"]["tier.level_reads"] >= tiers.MIN_LOADS
    assert min(c["tier.level_reads"] for c in clean["identity"]["counts"]) >= tiers.MIN_CONCURRENT_LOADS
    assert clean["restart"]["restored_blocks"] > 0 and clean["restart"]["level_reads_after_restart"] > 0


def test_the_level_without_a_watermark_passes_run_identity(tiny_checkpoint, tmp_path):
    """No `host.reserve`: a load's read-backs make room in the host level themselves, and never by evicting what the
    same load already read back."""
    build = _builder(tiny_checkpoint, max_model_len=256, state_memory=17 * 16 * 512, plugin_options={
        "state_tier.host.bytes": "520KiB", "state_tier.host.pages": "1", "state_tier.levels": "ssd",
        "state_tier.ssd.dir": str(tmp_path / "level"), "state_tier.ssd.bytes": "8MiB", "state_tier.host.reserve": "0"})
    got = tiers.identity(build, _trace(512, 12, 64, (10, 40), 8), lambda i=0: tiers.Deferred(i), 2, options=PRESSURE)
    assert got["verdict"] == "pass", got["failures"]


# free-on-ticket is not here: it fires and changes nothing, on the CPU (with or without the watermark) and on the
# A5000 (docs/v1-kv-tiers.md §13, "Step 3 as built").
@pytest.mark.parametrize("plant", ["read-unwaited", "torn-manifest", "torn-manifest-unchecked", "stale-slot"])
def test_a_level_plant_is_caught(plant, dense_ssd):
    factory, checks = TIER_PLANTS[plant]
    result = tiers.judge_plant(plant, factory, checks, dense_ssd.run, dense_ssd.clean)
    assert result["verdict"] == "caught", result


def test_a_late_copy_onto_reused_units_is_caught(tiny_checkpoint):
    """load-reuse-unwaited in the `reuse` loop (an aborted gated request's loading blocks evictable at once, a heavy
    newcomer) with copies landing up to 8 launches late and answers long enough to read the reused pages back."""
    build = _builder(tiny_checkpoint, max_model_len=512, state_memory=34 * 16 * 512, max_step_tokens=256,
                     plugin_options={"state_tier.host.bytes": "4MiB", "state_tier.host.pages": "1"})
    trace, loop = _trace(512, 12, 64, (10, 40), 32), tiers.LOOPS["reuse"]
    deferred = lambda i=0: tiers.Deferred(i, most=8)              # noqa: E731
    clean = tiers.identity(build, trace, deferred, 1, loop=loop, options=PRESSURE)
    factory, _ = TIER_PLANTS["load-reuse-unwaited"]
    result = tiers.judge_plant("load-reuse-unwaited", factory, ("reuse",), {"reuse": lambda: tiers.identity(
        build, trace, deferred, 2, clean, loop=loop, options=PRESSURE)}, {"reuse": clean})
    assert clean["verdict"] == "pass" and clean["counts"][0]["aborted_gated"] > 0
    assert result["verdict"] == "caught", result


def test_the_audit_catches_a_tentative_filed_a_block_early(tiny_hybrid):
    """`answer:1` alone, answers crossing aligned positions: the block before each tentative's holds no checkpoint, so
    tentative-shift files there, and the audit (every payload at (depth + 1) × block) fails run-identity. With
    `prompt_end` or an earlier answer there, the shifted one finds the block taken and is only lost."""
    build = _builder(tiny_hybrid, max_model_len=512, state_memory=920 << 10, plugin_options={
        "state_tier.host.bytes": "24MiB", "state_tier.host.pages": "0.5", "recurrent.checkpoints": "6",
        "recurrent.policy": "answer:1"})
    trace, options = _trace(320, 8, 140, (20, 60), 80, turns=2), PRESSURE | {"state_tier.h2d_rate": "2e6"}
    deferred = lambda i=0: tiers.Deferred(i)                      # noqa: E731
    clean = tiers.identity(build, trace, deferred, 1, options=options)
    factory, checks = TIER_PLANTS["tentative-shift"]
    result = tiers.judge_plant("tentative-shift", factory, checks, {"identity": lambda: tiers.identity(
        build, trace, deferred, 2, clean, options=options)}, {"identity": clean})
    assert clean["verdict"] == "pass" and result["verdict"] == "caught", result
    assert result["checks"]["identity"]["findings"]["a payload filed away from its position"] > 0


@pytest.mark.parametrize("plant, broken", [("branch-off-by-one", {"branch, tier"}),
                                           ("tentative-shift", {"answer:1, tier"})])
def test_the_policies_plants_fail_bounded(tiny_hybrid, plant, broken):
    """Each plant fails exactly its policy's configuration (the clean policies pass: each plant leaves the other's
    configuration passing). Late sessions (second and third waves) meet the system prompt's end as a branch point;
    the first turn's prompt_end lies a block past it, and answers cross two aligned positions, so a checkpoint filed one
    block early lands where no other one is."""
    trace = _trace(320, 6, 140, (60, 90), 140, (0, 0, 1, 2, 1, 2), turns=2)
    opts = {"state_tier.host.bytes": "32MiB", "state_tier.host.pages": "0.5"}
    build = _builder(tiny_hybrid, max_model_len=trace.max_len, state_memory=16 << 20, plugin_options=opts, slots={})
    tier = {"prefix_cache": True, "overlap": True, "slots": {"tier": "std.state_tier"}}
    configs = {f"{p}, tier": tier | {"plugin_options": opts | {"recurrent.policy": "prompt_end," + p}}
               for p in ("branch", "answer:1")}
    with PLANTS[plant]() as fired:
        report = bounded.compare_configs(build, configs, trace, k=5)
    assert {name for name, r in report["results"].items() if not r["passed"]} == broken, bounded.table(report)
    assert fired["fired"] > 0


def test_a_branch_checkpoint_resumes_bitwise_as_recomputed_with_equal_chunks(tiny_hybrid):
    """Steps of 128 tokens: the branch checkpoint at the system prompt's end (128) ends a step, as the recompute's
    first step does; the request adopting it then computes the same chunks as the recompute after 128."""
    rng = random.Random(5)
    system, *rest = ([rng.randrange(320) for _ in range(n)] for n in (128, 70, 70, 70))
    params = SamplingParams(temperature=1.0, seed=3, max_tokens=8, ignore_eos=True, logprobs=0)

    def ask(engine, prompt):
        rid, out, lps, cached = engine.add(prompt, params), [], [], None
        while engine.has_work():
            for d in engine.step():
                if d.rid == rid:
                    out, lps = out + d.token_ids, lps + d.logprobs.token
                    cached = d.cached_tokens if d.cached_tokens is not None else cached
        return out, lps, cached

    build = _builder(tiny_hybrid, max_step_tokens=128, max_model_len=512, state_memory=8 << 20, max_seqs=2)
    tiered = build(options={"recurrent.policy": "prompt_end,branch", "state_tier.host.bytes": "32MiB",
                            "state_tier.host.pages": "0.5"})
    ask(tiered, system + rest[0])                    # prompt_end at 192, past the system prompt
    ask(tiered, system + rest[1])                    # the system prompt's end is a branch point: a checkpoint at 128
    got = ask(tiered, system + rest[2])
    tiered.close()
    recompute = build(tier=False, prefix_cache=False)
    want = ask(recompute, system + rest[2])
    recompute.close()
    assert got[2] == 128 and got[:2] == want[:2]


def test_the_cli_judges_a_tiny_model(tiny_checkpoint, tmp_path, capsys):
    rng = random.Random(1)
    prompts = tmp_path / "prompts.jsonl"
    rows = [{"id": f"p{i}", "category": "chat", "ids": [rng.randrange(512) for _ in range(n)]}
            for i, n in enumerate([80] + [20] * 32)]
    prompts.write_text("".join(json.dumps(r) + "\n" for r in rows))
    engine = ("--device cpu --page-size 16 --max-seqs 4 --max-step-tokens 64 --max-model-len 256 --state-memory 139264 "
              "--host-memory 16777216 --plugin-options state_tier.host.bytes=4MiB state_tier.host.pages=1")
    pressure = "--plugin-options state_tier.host.admit=evict state_tier.stall_bytes=0 state_tier.step_cost=1,0,0"
    args = ["tiers", "--model", str(tiny_checkpoint), "--prompts", str(prompts), "--dtype", "float32", "--sessions",
            "8", "--system-len", "64", "--chunk-len", "20", "--max-tokens", "6", "--runs", "2", "--json",
            str(tmp_path / "out.json"), "--engine-args", engine, "--pressure-args", pressure, "--plant",
            "promote-wrong-unit", "--plant-runs", "2"]
    assert main(args) == 0, capsys.readouterr().out
    assert (tmp_path / "out.json").exists()
    policies = {"recurrent.policy": "prompt_end,branch,answer:1"}            # tiers off: no tentative checkpoints
    assert tiers.tiers_off(policies) == {"recurrent.policy": "prompt_end,branch"}
