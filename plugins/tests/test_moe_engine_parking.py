"""Parking under exclusive tiers (docs/dev/v1-glm.md D8b, D8l, D8x): a lending step's lent experts are parked in RAM
from intact slots before the forward writes the activation loan, the copies waited for before a lent slot is reused;
a short prompt step parks those RAM has no unit for in the idle read-ahead rows and moves them into RAM's coldest
later. Every piece's bytes are the file's and the streams the resident store's; each plant (`slow`) breaks one rule
and is caught."""
import collections
import contextlib

import pytest
import torch
from holon.api import SamplingParams
from holon_std.moe.offload import KEY

from .moe_engine_common import GREEDY, _bytes_exact, _gpu_engine, _held_once, _served, _slots_held

PREBUILT = ("ggml_cuda", "expert_io", "cpu_task")  # CUDA extensions built if missing (conftest)
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.slow
@pytest.mark.parametrize("terms", ["default", "mode2"])
def test_parked_experts_hold_their_pieces_beside_the_activation_loan(gpu_moe, mode2, terms):
    """Step 1's review S1-B1 (docs/dev/v1-glm.md D8b, D9): RAM with free units, the tail lent with the step's activation
    loan (`require`): lent experts are parked from intact slots — the step's lending runs before the loan is handed to
    the forward, an expert holding a loan slot goes to the SSD, the copies are waited for before a lent slot is reused —
    so every RAM-resident expert's units hold its pieces as the file does, and the streams are the resident store's.
    (`slow`: the plants and their control; the dev run's control under the same terms is D8l's, next.)"""
    from holon_judge.planted import parking_late, parking_unwaited

    file, prompts, want = gpu_moe
    opts = {"moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.ring_lend": "tail",
            "moe.lend_peak": "require", "moe.cache": "*=0.75", "moe.ram": "*=0.5",
            "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"} if terms == "default" else mode2()
    for plant in (contextlib.nullcontext, parking_late, parking_unwaited):     # the fence, then under a forced delay
        with plant() as fired:
            engine = _gpu_engine(file, host_memory=1 << 28, plugin_options=opts)
            store = engine.ctx.shared_objects[KEY]
            got = _served(engine, prompts)
            if terms == "mode2":                     # again, after decoding: F10's spares set up (released)
                assert int((store.state["spare"][..., 0] > 0).sum()) > 0
                got = got if got != want else _served(engine, prompts)
            assert store.exclusive and store.borrowed["stage"] and store.parked > 0, (store.borrowed, store.parked)
            assert store._parked is None             # each step's parking copies waited for (C S1-M2's fence)
            wrong = _bytes_exact(store)
            engine.close()
        assert plant is contextlib.nullcontext or fired["delayed"]
        assert (got == want and not wrong) == (plant is not parking_unwaited), wrong


@pytest.mark.parametrize("plant, terms", [(None, "default"), (None, "mode2")] + [
    pytest.param(p, terms, marks=pytest.mark.slow) for p, terms in (
        ("loan_parking_late", "default"), ("loan_write_before_parked", "default"), ("parking_event_early", "default"),
        ("loan_wait_unrecorded", "default"), ("loan_write_before_parked", "mode2"))])
def test_a_tail_steps_loan_experts_are_parked_before_the_forward_writes_the_loan(gpu_moe_wide, mode2, plant, terms):
    """D8l (docs/dev/v1-glm.md): a tail step's lent experts holding slots of its activation loan are parked in RAM as
    the others (once they went to the SSD): RAM with free units here, no lent expert goes to the SSD, and every RAM
    unit holds its pieces as the file does, the streams the resident store's — the forward waits for the parking
    copies before it writes the loan. Under the copies delayed on a stream of their own and the loan written whole as
    it is handed out (`loan_parking_late`: still exact): the copies issued after the loan's writes with no wait
    (`loan_write_before_parked`) or with the event recorded before them (`parking_event_early`) leave RAM units with
    overwritten bytes; no copy issued before the loan (`loan_wait_unrecorded`) is refused, a failed step's stale event
    notwithstanding (A's m2). The early start makes no synchronizing torch call (A's m1). Unplanted, D8b's terms
    too: the lent experts were parked, each step's parking copies waited for (C S1-M2's fence)."""
    from holon_judge import planted

    file, prompts, want = gpu_moe_wide
    opts = {"moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.ring_lend": "tail",
            "moe.lend_peak": "require", "moe.cache": "*=0.75", "moe.ram": "*=0.5",
            "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"} if terms == "default" else mode2()
    with getattr(planted, plant)() if plant else contextlib.nullcontext() as fired:
        engine = _gpu_engine(file, host_memory=1 << 29, plugin_options=opts)
        store, to_ssd = engine.ctx.shared_objects[KEY], []
        lend_tail = store.lend_tail

        def counted(cls, n):                     # the tail's experts the step sent to the SSD (read at the end)
            picked = lend_tail(cls, n)
            to_ssd.append((picked.sum(), ((store.residency.device == -2).all(-1) & picked).sum()))
            return picked
        store.lend_tail = counted
        start = store.start_step

        def strict(early=False):                 # an early start enqueues, never waits for the device (A's m1)
            if not early:
                return start(early)
            torch.cuda.set_sync_debug_mode("error")
            try:
                return start(early)
            finally:
                torch.cuda.set_sync_debug_mode("default")
        store.start_step = strict
        if plant == "loan_wait_unrecorded":      # (a stale event a failed step left is no parking either: A's m2)
            engine.add(max(prompts, key=len), SamplingParams(temperature=0.0, max_tokens=1))
            store._parked = torch.Event(device=torch.device("cuda"))
            store._parked.record()
            with pytest.raises(RuntimeError, match="parking copies were not issued"):
                engine.step()
            engine.close()
            return
        got = _served(engine, prompts)
        if terms == "mode2":                     # again, after decoding: F10's spares set up, the tail releases them
            assert int((store.state["spare"][..., 0] > 0).sum()) > 0
            got = got if got != want else _served(engine, prompts)
        wrong = _bytes_exact(store)
        to_ssd = [(int(n), int(m)) for n, m in to_ssd]
        assert store.exclusive and store.borrowed["stage"] and store._step_loan_bytes(64) and to_ssd
        parked = (int(store.parked), store._parked is None)
        engine.close()
    if plant in (None, "loan_parking_late"):
        assert got == want and not wrong and sum(n for n, _ in to_ssd) and not any(m for _, m in to_ssd), \
            (wrong, to_ssd)
        assert parked[0] > 0 and parked[1], parked
        assert plant is None or fired["delayed"] and fired["filled"]
    else:
        assert fired["delayed"] and fired["filled"] and wrong, plant


@pytest.fixture(scope="module")
def gpu_moe_wide_fresh(gpu_moe_wide):
    """Two fresh prompts for `gpu_moe_wide`, each long enough to lend, and the resident store's greedy streams of each
    served alone (a prompt step of its own, then its decode)."""
    file, _, _ = gpu_moe_wide
    gen = torch.Generator().manual_seed(3)
    fresh = [torch.randint(0, 512, (n,), generator=gen).tolist() for n in (256, 250)]
    engine = _gpu_engine(file)
    want = [engine.generate([p], GREEDY) for p in fresh]
    engine.close()
    return fresh, want


@pytest.mark.parametrize("plant", [None] + [pytest.param(p, marks=pytest.mark.slow) for p in (
    "parked_row_superset_runs", "row_hit_before_parked", "row_move_before_unmapped", "row_mapped_before_copied",
    "row_reused_before_moved", "row_reused_before_mapped", "reserved_unit_named", "row_moved_after_promoted")])
def test_a_short_prompt_parks_its_lent_experts_in_the_read_ahead_rows(gpu_moe_wide, gpu_moe_wide_fresh, mode2, plant):
    """D8x (docs/dev/v1-glm.md), under the deployment's terms: a short prompt step (no decode row, below the read-ahead
    steps' size: every prompt step here) parks the lent experts RAM has no free unit for in the idle read-ahead rows —
    its later layers hit them there —; at its end each is paired with RAM's coldest of its classes, whose units it
    reserves; the thread copies the rows into them in chunks (4 here: several a prompt), chunk i mapped at launch
    (step + lag + i), the next short prompt mapping every chunk left first. Checked after every step: the bits are the
    resident store's, every piece the map names holds its bytes, each tier holds an expert once (a reservation
    accepted as such), no cache slot is lost; experts were parked, moved, and some routed meanwhile (reconciled). Each
    of D8x's plants breaks one rule and is caught (`holon_judge.planted._d8x`; `slow`, an engine each)."""
    from holon_judge.planted import d8x_plant

    file, prompts, want = gpu_moe_wide
    with d8x_plant(plant) if plant else contextlib.nullcontext(collections.Counter()) as fired:
        engine = _gpu_engine(file, host_memory=1 << 29, plugin_options=mode2(**{"moe.ram": "*=0.25"}))   # (RAM full)
        store, step, bad, cache, slots = engine.ctx.shared_objects[KEY], engine.step, collections.Counter(), {}, []
        store.move_first, store.move_chunk = 1, 2
        heat = torch.rand(store.ram_heat.shape, generator=torch.Generator().manual_seed(0))   # RAM's heat spread (a
        store.ram_heat.copy_(heat * 8)                     # lent expert hotter than RAM's coldest: no profile here)

        def checked():
            out = step()
            bad["bytes"] += _bytes_exact(store, cache)
            getattr(fired, "release", lambda: None)()        # (a held thread goes on once the bytes were checked)
            try:
                _held_once(store)
            except AssertionError:
                bad["held"] += 1
            slots.append(_slots_held(store).sum(0))
            bad["slots"] += not torch.equal(slots[-1], slots[0])
            return out
        engine.step = checked
        got = _served(engine, prompts)
        assert int((store.state["spare"][..., 0] > 0).sum()) > 0            # again: F10's spares set up
        got = got if got != want else _served(engine, prompts)
        fresh, streams = gpu_moe_wide_fresh                 # a prompt step alone, its decode during its move
        same = got == want and [engine.generate([p], GREEDY) for p in fresh] == streams
        counts, stats = store.move_counts.tolist(), dict(store.move_stats)
        engine.close()
    if plant is None:
        assert same and not +bad, bad
        assert counts[0] > counts[1] > 0 and stats["entries"] > counts[2] > 0 and stats["moves"], (counts, stats)
    else:
        assert sum(fired.values()) and (not same or +bad), (plant, dict(fired), bad)
