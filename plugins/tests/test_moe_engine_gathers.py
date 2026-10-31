"""The exclusive-tier store's staging gathers (docs/dev/v1-glm.md D8g, D8h, D8s): a staging step's misses staged in
rounds through a fixed set of slots, or gathered by copy engines beside the compute, each layer's once for all its
passes — the resident store's outputs and every piece's bytes, under delayed producers; each plant breaks one fence
and is caught. All `slow` (an engine a case; the SSD reads' landing rows and read-ahead: `test_moe_engine_gpu.py`)."""
import collections
import contextlib
from unittest import mock

import pytest
import torch
from holon_std.moe import offload as offload_unit
from holon_std.moe.offload import KEY

from .moe_engine_common import (
    GREEDY,
    _bytes_exact,
    _gpu_engine,
    _held_once,
    _loan_free,
    _served,
    _slots_once,
)

PREBUILT = ("ggml_cuda", "expert_io", "cpu_task")  # CUDA extensions built if missing (conftest)
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _spares_lent(store) -> int:
    """D8s (C): the policy's spare slots among the slots this step's non-tail lending took (calls ≥ 1)."""
    spare = store.state.get("spare")
    if spare is None or store.lent is None:
        return 0
    lent = torch.where((store.lent_call >= 1)[..., None], store.lent_slots, -1)        # [G, E, R]
    owned = torch.where((spare[..., 0] > 0)[..., None], spare[..., 1:], -2)          # [G, S, R]
    return int((lent[:, :, None] == owned[:, None]).all(-1).sum())


@pytest.mark.slow
def test_a_staging_step_stages_in_rounds_through_fixed_slots(gpu_moe_wide):
    """D8s (docs/dev/v1-glm.md): under exclusive tiers and F10 a lending step stages its misses in rounds — RAM misses
    by a round's slots, SSD reads by the landing rows — through one fixed set of slots a class, each part computed
    before the next one's gather reuses them: it lends the fixed set, not a layer's most missing, and serves the
    resident store's outputs with every piece's bytes the file's. Each round plant is caught."""
    from holon_judge import planted
    from holon_std.tiered import store as tiered_store

    file, prompts, want = gpu_moe_wide
    opts = {"moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.ring_lend": "tail",
            "moe.lend_peak": "require", "moe.cache": "*=0.75", "moe.ram": "*=0.25",
            "moe.placement": "placement.promote@1", "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"}
    for plant in (None, "round_slots_reused_early", "round_slot_shared", "part_one_role", "spare_lent_outside_tail"):
        # (the spare plant on steps that lend no tail: a tail step releases the spares before its other lending)
        tail = {"moe.ring_lend": "coldest", "moe.lend_peak": "off"} if plant == "spare_lent_outside_tail" else {}
        with planted.PLANTS[plant]() if plant else contextlib.nullcontext() as fired, \
                mock.patch.object(offload_unit, "BESIDE", "off"):      # (D8s's own gathers; D8g's: the next test)
            engine = _gpu_engine(file, host_memory=1 << 29, plugin_options=opts | tail)
            store, lends, rounds = engine.ctx.shared_objects[KEY], [], []
            store.taps.append(lambda step, record, lends=lends: lends.append(record["lends"])
                              if record["lends"] is not None else None)
            begin = store.begin

            def watched(store=store, begin=begin, rounds=rounds):
                begin()
                rounds.append((store.lent is not None and store.lent.rounds, _spares_lent(store), _loan_free(store)))
            store.begin = watched
            try:                                     # (twice: the second's lending steps meet F10's spares, set up)
                got = [_served(engine, prompts) for _ in range(2)]
                ok = (got == [want, want], _bytes_exact(store) == 0)
            except RuntimeError as e:                    # (a plant the store refuses)
                assert plant is not None, e
                ok = (False,)
            ok = (*ok, not any(n for _, n, _ in rounds))        # no spare's slots lent outside the tail
            if plant is None:                         # (and D8w's tail over both down classes: no piece in the loan)
                assert any(r for r, _, _ in rounds) and all(ok) and all(f for _, _, f in rounds), (rounds, ok)
                with mock.patch.object(tiered_store, "ROUNDS_BELOW", 100):    # a step of more prompt tokens than
                    rounds.clear()                                            # the gate: no rounds (D8s)
                    engine.generate([[(t + 7) % 512 for t in prompts[-1]]], GREEDY)
                    assert rounds and not any(r for r, _, _ in rounds), rounds
                assert len(store.tail_down) == 2 and _slots_once(store), store.tail_down
                fixed = store.rounds_q + store.landing_rows
                assert max(int(x[1:].max()) for x in lends) <= fixed, lends   # each class lends the fixed set
                _held_once(store)
            else:
                assert fired and not all(ok), (plant, fired, ok)
            engine.close()


@pytest.fixture(scope="module")
def gpu_moe_wide_one(gpu_moe_wide):
    """`gpu_moe_wide`'s resident outputs at one sequence (a graphed step's staging, and so the landing rows, 8 wide)."""
    file, prompts, _ = gpu_moe_wide
    engine = _gpu_engine(file, max_seqs=1)
    want = _served(engine, prompts)
    engine.close()
    return file, prompts, want


@contextlib.contextmanager
def _beside_delayed(late: str, ns: int = 10_000_000):
    """D8g's delayed-producer harness: `late` "copies" (each copy's provider stream sleeps first), "producers" (and
    each SSD request and read-ahead set: the worker sees it late), "worker" (each eager signal of the I/O worker on a
    stream of its own after a sleep, a wait on its slot after it: the worker reads late, as a busy one does) or
    "readers" (main before each part's pass: its slots' readers late, for the fences that guard their reuse)."""
    from holon_std.tiered.resolver import _Beside
    from holon_std.tiered.store import TieredStore

    copy, rows, ahead, land = _Beside.copy, TieredStore.round_rows, TieredStore.read_ahead, _Beside.land
    launch = TieredStore.before_launch

    def late_copy(self, inst, *a):
        with torch.cuda.stream(inst.stream):
            torch.cuda._sleep(ns)
        return copy(self, inst, *a)

    def late_request(f):
        def run(self, *a, **kw):
            torch.cuda._sleep(ns)
            return f(self, *a, **kw)
        return run

    def late_land(self, i):
        land(self, i)
        torch.cuda._sleep(ns)
    side, signalled, patched = torch.cuda.Stream(), {}, []

    def late_worker(self, step):                 # (on the engine's I/O worker, once: graphs' signals as they are)
        io = self.io if self.exclusive else None
        if io is not None and io not in patched:
            signal, wait = io.signal, io.wait

            def deferred(slot):
                if torch.cuda.is_current_stream_capturing():
                    return signal(slot)
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    torch.cuda._sleep(ns)
                    signal(slot)
                    signalled[slot] = torch.cuda.Event()
                    signalled[slot].record()

            def waiting(slot):                   # (a wait follows its slot's deferred signal: the word's protocol)
                if slot in signalled and not torch.cuda.is_current_stream_capturing():
                    torch.cuda.current_stream().wait_event(signalled.pop(slot))
                wait(slot)
            io.signal, io.wait = deferred, waiting
            patched.append(io)
        return launch(self, step)
    with contextlib.ExitStack() as stack:
        stack.callback(lambda: [(delattr(io, "signal"), delattr(io, "wait")) for io in patched])
        patches = {"copies": [(_Beside, "copy", late_copy)], "readers": [(_Beside, "land", late_land)],
                   "worker": [(TieredStore, "before_launch", late_worker)],
                   "producers": [(_Beside, "copy", late_copy), (TieredStore, "round_rows", late_request(rows)),
                                 (TieredStore, "read_ahead", late_request(ahead))]}[late]
        for cls, name, f in patches:
            stack.enter_context(mock.patch.object(cls, name, f))
        yield


_BESIDE = [(v, None, late, 1) for v in ("rounds", "stage") for late in ("producers", "copies", "readers")]
_BESIDE += [("stage", None, "worker", 1), ("stage", None, "copies", 4)]       # (four: steps with decode rows too)
_BESIDE += [("rounds", p, late, 1) for p, late in (
    ("worker_late", "producers"), ("gather_part_unwaited", "copies"), ("round_set_reused_early", "readers"),
    ("ssd_set_reused_early", "readers"), ("fence_waits_unrecorded", "readers"), ("landing_half_reused_early", "copies"),
    ("round_rows_shared", "producers"), ("io_word_stale", "worker"), ("cuda_event_on_generic", "copies"),
    ("readback_unwaited", "producers"))]
_BESIDE += [("stage", p, late, 1) for p, late in (("readahead_wait_after_tiered", "worker"),
                                                 ("readahead_slot_reused", "copies"))]


@pytest.mark.slow
@pytest.mark.parametrize("variant, plant, late, seqs", _BESIDE)
def test_gathers_beside_the_compute_hold_their_fences(gpu_moe_wide, gpu_moe_wide_one, variant, plant, late, seqs):
    """D8g (docs/dev/v1-glm.md): an eager staging step's gathers by copy engines beside the compute — in rounds through
    the fixed set (steps below ROUNDS_BELOW), or each miss in its own staging slot (`stage`, here from any size, with
    read-ahead hits; at four sequences, steps with decode rows too) — front parts of 8, SSD rounds of 8 landing rows in
    two halves — serve the
    resident store's outputs with every piece's bytes the file's, under each of the delayed-producer harness's delays
    (`_beside_delayed`; and a late worker, `worker_late`). Each plant, one fence dropped under the delay that would
    expose it, is caught where its condition occurred."""
    from holon_judge import planted
    from holon_std.tiered import store as tiered_store
    from holon_std.tiered.resolver import _Beside

    file, prompts, want = gpu_moe_wide_one if seqs == 1 else gpu_moe_wide
    opts = {"moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.ring_lend": "tail",
            "moe.lend_peak": "require", "moe.cache": "*=0.75", "moe.ram": "*=0.1",
            "moe.placement": "placement.promote@1",
            "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1",
            "cpu_task.timeout": 30.0}               # (a bound no busy machine meets by accident: A's m2)
    load, calls = _Beside.load, []

    def spy(self):
        fresh = self.parts is None
        parts = load(self)
        if fresh:
            calls.append((self.rounds, len(self.front), self.n_rounds, int(self.lists["n_hit"].max())))
        return parts
    below = tiered_store.ROUNDS_BELOW if variant == "rounds" else 0
    with (mock.patch.object(_Beside, "load", spy), mock.patch.object(tiered_store, "ROUNDS_BELOW", below),
          mock.patch.object(tiered_store, "ON_DEMAND", 4), mock.patch.object(tiered_store, "ROUND_SLOTS", 8),
          mock.patch.object(offload_unit, "BESIDE", variant),
          _beside_delayed(late), planted.PLANTS[plant]() if plant else contextlib.nullcontext() as fired):
        engine = _gpu_engine(file, host_memory=1 << 29, max_seqs=seqs, plugin_options=opts)
        store = engine.ctx.shared_objects[KEY]
        store.read_ahead_tokens = 8
        try:                                     # (each plant fails by its outputs or bytes — not by a wait's timeout
            same, wrong, failed = _served(engine, prompts) == want, _bytes_exact(store), None     # or a refusal —, but
        except (IndexError, RuntimeError) as e:  # readback_unwaited's lists, another call's, may index past a table,
            if not isinstance(e, IndexError) and "outside src or dst" not in str(e):     # or a copy refuse them)
                raise
            same, wrong, failed = False, -1, e
        engine.close()
    assert failed is None or plant == "readback_unwaited", failed
    if plant in (None, "worker_late"):
        assert calls and all(r == (variant == "rounds") for r, *_ in calls), calls
        fronts, rounds, hits = (max(c[i] for c in calls) for i in (1, 2, 3))
        assert same and not wrong, (same, wrong)
        assert fronts >= 2 and rounds >= (3 if seqs == 1 else 1), (fronts, rounds)
        assert plant is None or fired["deferred"], fired
        assert late != "worker" or hits, calls        # (the late worker delayed read-ahead hits)
    else:
        assert any(fired.values()) and not (same and not wrong), (plant, fired, same, wrong)


@pytest.fixture(scope="module")
def gpu_moe_long(gpu_moe_wide):
    """`gpu_moe_wide` with prompts of two passes and more (steps of 1,024 rows, passes of 512 on this model's
    types), the resident store's outputs at one and four sequences."""
    file, _, _ = gpu_moe_wide
    gen = torch.Generator().manual_seed(9)
    prompts = [torch.randint(0, 512, (n,), generator=gen).tolist() for n in (1300, 40, 1100, 7)]
    long = {"max_step_tokens": 1024, "max_model_len": 2048}
    want = {}
    for seqs in (1, 4):
        engine = _gpu_engine(file, max_seqs=seqs, **long)
        want[seqs] = _served(engine, prompts)
        engine.close()
    return file, prompts, want, long


_UNION = [(p, s, None) for p in ("placement.promote@1", "") for s in (1, 4)]
_UNION += [("placement.promote@1", 1, plant) for plant in ("union_one_pass", "gather_part_unwaited")]


@pytest.mark.slow
@pytest.mark.parametrize("policy, seqs, plant", _UNION)
def test_a_layer_gathers_its_misses_once_for_its_passes(gpu_moe_long, policy, seqs, plant):
    """D8h (docs/dev/v1-glm.md): a staging step of more than one pass, its gathers beside the compute (D8g's stage
    mode, here from any size), resolves each layer once over all its rows — one union of misses, one set of parts —,
    pass 1 computing part by part, the later passes at once; the resident store's outputs and every piece's bytes,
    under F10 and F11, at one sequence and at four (decode rows beside a prompt of two passes). Planted, the resolve
    over the first pass's rows only (`union_one_pass`) and the parts unwaited under late copies
    (`gather_part_unwaited`) each fail on bits."""
    from holon_judge import planted
    from holon_std.moe.offload import Offload
    from holon_std.tiered import store as tiered_store
    from holon_std.tiered.resolver import _Beside

    file, prompts, want, long = gpu_moe_long
    opts = {"moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.ring_lend": "tail",
            "moe.lend_peak": "require", "moe.cache": "*=0.75", "moe.ram": "*=0.1", "moe.placement": policy,
            "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"}
    union, load, calls = Offload.route_union, _Beside.load, collections.Counter()

    def spy_union(self, ids, x):
        calls["union"] += 1
        calls["rows"] = max(calls["rows"], len(ids))
        return union(self, ids, x)

    def spy_load(self):
        if self.parts is None:
            calls["moves"] += int(self.store.n_moves[self.resolver.group % 2].max())   # (a sync: a test's)
        return load(self)
    with (mock.patch.object(Offload, "route_union", spy_union), mock.patch.object(_Beside, "load", spy_load),
          mock.patch.object(tiered_store, "ROUNDS_BELOW", 0),
          _beside_delayed("copies") if plant else contextlib.nullcontext(),
          planted.PLANTS[plant]() if plant else contextlib.nullcontext() as fired):
        engine = _gpu_engine(file, host_memory=1 << 29, max_seqs=seqs, plugin_options=opts, **long)
        store = engine.ctx.shared_objects[KEY]
        ok = _served(engine, prompts) == want[seqs] and _bytes_exact(store) == 0
        engine.close()
    assert calls["union"] and calls["rows"] > 512, calls
    assert ok == (plant is None) and (plant is None or any(fired.values())), (plant, ok, fired)


@pytest.mark.slow
@pytest.mark.parametrize("layout", ["ram", "full_step"])
def test_a_union_of_a_steps_rows_resolves_as_its_passes_do(gpu_moe_long, layout):
    """D8h's union outside its gate: `ram` (B's D8h-m2) forces it on a RAM-holds-all layout (a streaming ring and
    the union resolved at once: every entry part −1) — the resident store's bits; `full_step` (A's D8h-m4) resolves
    and places one union of a 32,768-row step's 32,768 × k entries under exclusive tiers — the resident store's
    bits and every piece's bytes."""
    from holon_std.moe.offload import Offload
    from holon_std.tiered import store as tiered_store

    file, prompts, want, long = gpu_moe_long
    rows, union = [], Offload.route_union
    spy = mock.patch.object(Offload, "route_union", lambda self, ids, x: rows.append(len(ids)) or union(self, ids, x))
    if layout == "ram":
        opts = {"moe.store": "std.moe.offload@1", "moe.cache": "*=0.25", "moe.ring_tokens": 512}
        with spy, mock.patch.object(Offload, "gathers_once", lambda self, tokens: tokens > self.pass_tokens(tokens)):
            engine = _gpu_engine(file, host_memory=1 << 29, max_seqs=1, plugin_options=opts, **long)
            assert _served(engine, prompts) == want[1]
            engine.close()
        assert rows and max(rows) > 512, rows
        return
    big = {"max_step_tokens": 32768, "max_model_len": 33792}
    prompt = [torch.randint(0, 512, (32768,), generator=torch.Generator().manual_seed(4)).tolist()]
    resident = _gpu_engine(file, max_seqs=1, **big)
    ref = _served(resident, prompt)
    resident.close()
    opts = {"moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.cache": "*=0.75", "moe.ram": "*=0.1",
            "moe.placement": "placement.promote@1", "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"}
    with spy, mock.patch.object(tiered_store, "ROUNDS_BELOW", 0):     # (the coldest lent: a 32K step's loan past
        #                                                                  this model's tail)
        engine = _gpu_engine(file, host_memory=1 << 29, max_seqs=1, plugin_options=opts, **big)
        assert _served(engine, prompt) == ref and _bytes_exact(engine.ctx.shared_objects[KEY]) == 0
        engine.close()
    assert max(rows) == 32768, rows
