"""F10 (`placement.promote@1`) under exclusive tiers on the GPU (docs/dev/v1-glm.md D8t, step 7): its spares outside
what a step lends, its admitted moves and demotions on the store's moves stream beside the next calls, fenced, and
the demotions' order against the next call's gathers and table — the resident store's streams and every piece's
bytes; each plant (`slow`) breaks one rule and is caught."""
import contextlib
from unittest import mock

import pytest
import torch
from holon.api import SamplingParams
from holon_std.moe.offload import KEY

from .moe_engine_common import _bytes_exact, _gpu_engine, _held_once, _served

PREBUILT = ("ggml_cuda", "expert_io", "cpu_task")  # CUDA extensions built if missing (conftest)
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("plant", [False, pytest.param(True, marks=pytest.mark.slow)], ids=["late", "moves_unfenced"])
def test_a_spare_policys_moves_run_beside_and_fence_the_staging_slots(gpu_moe, plant):
    """GLM step 7: under a policy with spares (F10) and exclusive tiers a call's admitted moves and demotions run on
    the store's moves stream — the admissions first, the next call's staging writes after them (`admitted`), the
    demotions beside the next calls, a later call reusing their table or their victims' slots after them (`moved`,
    the step's join) — and the store still serves the resident store's bits, one copy an expert, its demotions late
    (a sleep before each); under `moves_unfenced` (no fence, each call's demotions enqueued after the call two later
    rewrote their table: a deterministic delay, not a race) it does not."""
    from holon_judge.planted import _late_demotions, moves_unfenced

    file, prompts, _ = gpu_moe
    params = SamplingParams(temperature=0.0, max_tokens=48, ignore_eos=True)     # (long enough to read demoted
    resident = _gpu_engine(file)                                                  # experts back from RAM)
    want = resident.generate(prompts, params)
    resident.close()
    opts = {"moe.store": "std.moe.offload@1", "moe.cache": "*=0.25", "moe.ram": "*=0.25", "moe.ring_tokens": 512,
            "moe.placement": "placement.promote@1", "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"}
    with moves_unfenced() if plant else _late_demotions(20_000_000) as fired:
        engine = _gpu_engine(file, host_memory=1 << 27, plugin_options=opts)
        store = engine.ctx.shared_objects[KEY]
        assert store.moves_async
        got = engine.generate(prompts, params)
        wrong = _bytes_exact(store)
        if not plant:
            _held_once(store)
        engine.close()
    assert (got == want and not wrong) != plant and (not plant or fired["deferred"]), (wrong, fired)


@pytest.mark.parametrize("cache", ["*=0.5", "*=0.75"])
def test_a_layout_lends_only_past_the_policys_reserved_slots(gpu_moe, cache):
    """F10 under exclusive tiers: its spares (`reserved` a layer in each role's class) are cache slots no step lends.
    A cache whose seeded experts less the spares cannot lend a prefill pass's staging stages it in slots of its own
    (it stopped at the first prompt step with the store's SHORT error when the layout counted the spares as lendable);
    one that can still lends. Either way the resident store's bits."""
    file, prompts, _ = gpu_moe
    resident = _gpu_engine(file, max_seqs=1)               # (one sequence a step: a pass of 8 misses, staging of 2)
    want = _served(resident, prompts)
    resident.close()
    opts = {"moe.store": "std.moe.offload@1", "moe.cache": cache, "moe.ram": "*=0.25",
            "moe.placement": "placement.promote@1", "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"}
    engine = _gpu_engine(file, host_memory=1 << 27, plugin_options=opts, max_seqs=1)
    store = engine.ctx.shared_objects[KEY]
    assert _served(engine, prompts) == want
    held = store._held_out()
    assert all(n == 3 * len(store.layers) for n in held.values())          # 3 spares a layer, one class a role
    lendable = min(c.size - c.null - c.staging - held[n] for n, c in store.classes.items())
    assert store.lending == (lendable >= store.prefill) == (cache == "*=0.75")
    engine.close()


@pytest.mark.parametrize("plant", ["pending_victims",
                                   pytest.param("keep_into_pending_demotion", marks=pytest.mark.slow)])
def test_a_keep_never_takes_a_unit_whose_demotion_is_pending(gpu_moe, plant):
    """GLM step 7 (A's D8p-M3): a call's demotions run beside the next call, whose keeps the I/O worker writes from the
    host, so `moe_tiers@2`'s stamps keep the units the previous call demoted into out of its displacements. With
    RAM's coldest forced onto those experts at every call and the demotions late, the store serves the resident
    store's bits; with the stamps dropped (`keep_into_pending_demotion`) keeps land on such units and it does not."""
    from holon_judge import planted
    from holon_plugins.placement.promote import plugin as promote

    file, prompts, _ = gpu_moe
    params = SamplingParams(temperature=0.0, max_tokens=48, ignore_eos=True)
    resident = _gpu_engine(file, cuda_graphs=False)
    want = resident.generate(prompts, params)
    resident.close()
    opts = {"moe.store": "std.moe.offload@1", "moe.cache": "*=0.25", "moe.ram": "*=0.25", "moe.ring_tokens": 512,
            "moe.placement": "placement.promote@1", "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"}
    with getattr(planted, plant)() as fired, mock.patch.object(promote, "BUDGET", None):    # (admitting every miss:
        engine = _gpu_engine(file, host_memory=1 << 27, plugin_options=opts, cuda_graphs=False)    # the most demotions)
        got = engine.generate(prompts, params)
        wrong = _bytes_exact(engine.ctx.shared_objects[KEY])
        engine.close()
    stamped = plant == "pending_victims"
    assert fired["forced"] and (fired["keeps_on_pending"] == 0) == stamped, dict(fired)
    assert (wrong == 0) == stamped and (got == want or not stamped), wrong


@pytest.mark.parametrize("plant", [None, pytest.param("demotion_before_gather", marks=pytest.mark.slow)])
def test_a_calls_demotions_follow_the_next_calls_gathers(gpu_moe, plant):
    """D8t (docs/dev/v1-glm.md): under F10 a call's demotions are enqueued after the next call's gathers (no longer
    under them), its table's `moved` after them; a call of the same layer next (mixed steps of 1,024 rows in passes
    of 512) enqueues them first, and the step's last call at its join. The store serves the resident store's streams
    and every piece's bytes. Demotions before the call's own gathers (`demotion_before_gather`: a RAM-sourced
    admission staged from its victim's bytes) leave wrong bytes. (A same-layer call after one with demotions cannot
    occur — decode rows never span a policy's passes, D4, and a drafter's sub-steps join at their last layer — so
    the guard is defensive and has no plant.)"""
    from holon_judge import planted

    file, prompts, _ = gpu_moe
    prompts = prompts + [torch.randint(0, 400, (900,), generator=torch.Generator().manual_seed(5)).tolist()]
    params = SamplingParams(temperature=0.0, max_tokens=24, ignore_eos=True)
    base = {"max_step_tokens": 1024, "max_model_len": 2048, "chunk_with_decode": 1, "cuda_graphs": False}
    resident = _gpu_engine(file, **base)
    want = resident.generate(prompts, params)
    resident.close()
    opts = {"moe.store": "std.moe.offload@1", "moe.cache": "*=0.25", "moe.ram": "*=0.25", "ggml_cuda.pass_rows": 512,
            "moe.placement": "placement.promote@1", "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"}
    with getattr(planted, plant)() if plant else contextlib.nullcontext() as fired:
        engine = _gpu_engine(file, host_memory=1 << 27, plugin_options=opts, **base)
        store = engine.ctx.shared_objects[KEY]
        same, tier_op = [], store.resolvers[0].__class__._tier_op

        def counted(self, ids, kinds, fetch, n):  # a same-layer call (its pending table its own)
            pending = self.store.pending_demotions
            same.append(pending is not None and pending[0] == self.group)
            return tier_op(self, ids, kinds, fetch, n)
        rids = {engine.add(p, params): i for i, p in enumerate(prompts)}
        got, wrong, seen = [[] for _ in rids], 0, 0
        with mock.patch.object(store.resolvers[0].__class__, "_tier_op", counted) if plant is None else \
                contextlib.nullcontext():
            while engine.has_work():
                for d in engine.step():
                    got[rids[d.rid]] += d.token_ids
                if plant and sum(fired.values()) > seen:     # the bytes right after a planted step (before a later
                    seen = sum(fired.values())                # eviction hides them)
                    wrong += _bytes_exact(store)
        wrong += _bytes_exact(store)
        engine.close()
    if plant is None:
        assert got == want and wrong == 0 and any(same), (wrong, sum(same))
    else:
        assert seen and wrong, (plant, dict(fired))


@pytest.mark.parametrize("plant", [None, pytest.param("demotion_table_rewritten", marks=pytest.mark.slow)])
def test_a_layers_next_call_enqueues_its_last_calls_demotions_first(gpu_moe, plant):
    """D8t (A's M1, m1): a layer called twice in a row with decode rows (a drafter's sub-steps on an auxiliary group;
    here layer 1's routed experts run twice every decode step, the first result dropped): the first call's
    promotions refill its spares, their demotions deferred; the second call, of the same move table, enqueues them
    before its tier op rewrites the table. The streams are the resident store's and every piece's bytes the file's,
    read right after each step; with that flush dropped (`demotion_table_rewritten`) the first call's demotions copy
    the second call's list and its victims' RAM units are left stale (RAM with free units: the victims demoted)."""
    from holon_judge import planted

    file, prompts, _ = gpu_moe
    params = SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True)
    resident = _gpu_engine(file, cuda_graphs=False)
    want = resident.generate(prompts, params)
    resident.close()
    opts = {"moe.store": "std.moe.offload@1", "moe.cache": "*=0.25", "moe.ram": "*=0.75",
            "moe.placement": "placement.promote@1", "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"}
    with getattr(planted, plant)() if plant else contextlib.nullcontext() as fired:
        engine = _gpu_engine(file, host_memory=1 << 28, plugin_options=opts, cuda_graphs=False)
        store, moe = engine.ctx.shared_objects[KEY], engine.executor.model.layers[1].ffn
        forward, flushed = moe._forward, []

        def twice(x, residual=None, join=None):  # (decode steps: every row a decode row)
            if x.shape[0] <= len(prompts):
                forward(x, None, None)
            return forward(x, residual, join)
        moe._forward = twice
        flush = store.flush_demotions

        def counted():
            flushed.append(store.pending_demotions is not None and store.pending_demotions[0] == 1)
            return flush()
        store.flush_demotions = counted
        rids = {engine.add(p, params): i for i, p in enumerate(prompts)}
        got, wrong = [[] for _ in rids], 0
        while engine.has_work():
            for d in engine.step():
                got[rids[d.rid]] += d.token_ids
            wrong += _bytes_exact(store)                     # right after each step (a later eviction hides them)
        engine.close()
    if plant is None:
        assert got == want and wrong == 0 and sum(flushed) > 1, (wrong, sum(flushed))
    else:
        assert fired["rewritten"] and wrong, (dict(fired), wrong)
