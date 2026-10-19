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


def _setup(build, trace, pressure, tight=None):
    deferred = lambda i=0: tiers.Deferred(i)                      # noqa: E731
    clean = {"invariance": tiers.invariance(build, trace, deferred),
             "identity": tiers.identity(build, trace, deferred, options=pressure)}
    run = {"invariance": lambda: tiers.invariance(build, trace, deferred),
           "identity": lambda: tiers.identity(build, trace, deferred, 3, clean["identity"], options=pressure)}
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
# - load-before-store: the next launch's wait on freed units runs a deferred store before any later load reads its
#   host units, so the window is one schedule wide;
# - load-reuse-unwaited: the late copy lands before the units' next writer, since a launch waiting on any later load
#   runs the copy stream in order;
# - swap-onto-lower: recomputed and cached bytes give the same greedy outputs on the tiny fp32 model;
# - adopt-source-reused (`loaded` ignoring the reading step's end): a CPU step's adopt has run by the end of its launch;
# - adopt-source-reused-join: the earlier load's host units outlive the joining adopt here (no payload is evicted and
#   restored in between).
@pytest.mark.parametrize("plant", ["promote-unwaited", "waiter-unref", "store-unwaited", "promote-wrong-unit",
                                   "gate-on-landing", "promote-preempts", "kids-any-level", "interior-drop"])
def test_a_tier_plant_is_caught(plant, dense):
    m = dense
    factory, checks = TIER_PLANTS[plant]
    result = tiers.judge_plant(plant, factory, ("tight",) if plant == "interior-drop" else checks, m.run, m.clean)
    assert result["verdict"] == "caught", result


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
