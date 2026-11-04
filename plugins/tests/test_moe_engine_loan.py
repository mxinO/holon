"""The tail and the activation loan (docs/dev/v1-moe-prefill.md D11, FN step 7, docs/dev/v1-glm.md D8k, D8w, D9): a
tail store lends its streaming steps their activations from the tail's last bytes, a range in proportion to the step's
prompt tokens plus its fixed bytes; a whole-expert tail under exclusive tiers lends its experts whole; the loan's
sizing — margin, taper, a range for the step's largest allocation — and where a layout cannot lend it; each step's
peak within the profiled reserve and its own loan. Then the lent experts' parking (D8b, D8l, D8x): parked in RAM from
intact slots before the forward writes the loan, the copies waited for before a lent slot is reused; a short prompt
step parks those RAM has no unit for in the idle read-ahead rows and moves them into RAM's coldest later. Each plant
(`slow`) breaks one rule and is caught. The wide tests share one model, `glm_shaped`."""
import collections
import contextlib
import os
import subprocess
import sys
import types
from unittest import mock

import numpy as np
import pytest
import torch
from holon.api import RowKind, SamplingParams
from holon_std.moe.offload import KEY

from .moe_engine_common import (
    GREEDY,
    Q,
    _bytes_exact,
    _gpu_engine,
    _held_once,
    _loan_free,
    _served,
    _slots_held,
    _slots_once,
    _wide,
)

PREBUILT = ("ggml_cuda", "expert_io", "cpu_task", "promote")  # CUDA extensions built if missing (conftest)
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.slow
@GPU
def test_a_tail_store_lends_its_streaming_steps_their_activations(gpu_moe):
    """The core's ActivationLender under `moe.lend_peak` (FN step 7, the toy proof): the startup profile measures the
    worst step, then reserves only the worst one below `ring_tokens` (smaller); a streaming step's forward allocates
    from the tail's last bytes (`activation_loan@1`: kernels.expert_io's MemPool), a range in proportion to its prompt
    tokens, past the passes' buffers; the outputs are the resident store's bit for bit (CUDA graphs on, overlap on).
    The loan options' defaults lend the bytes the loan had before them (FN's RTX 5080 lock's layout). A buffer kept
    from inside a lending step stops the engine at the next launch."""
    from holon.core import platform
    from holon_std.tiered.store import LOAN_MARGIN, LOAN_SLACK, _profiled_tokens

    file, prompts, want = gpu_moe
    peaks, measure = [], platform.peak_memory

    def peak_memory(device, fn):
        peaks.append(measure(device, fn))
        return peaks[-1]
    opts = {"moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.ring_lend": "tail", "moe.lend_peak": "on"}
    with mock.patch.object(platform, "peak_memory", peak_memory):
        engine = _gpu_engine(file, host_memory=1 << 27, plugin_options=opts)
    store = engine.ctx.shared_objects[KEY]
    assert len(peaks) == 2 and 0 < peaks[1] < peaks[0] and store._loan_bytes >= peaks[0]
    ranges, use = [], store._loan.use
    store._loan.use = lambda r: (ranges.append((r.data_ptr(), r.numel(), store.step_tokens, store._step_loan,
                                                store._step_tail, store._step_gu_loan)), use(r))[1]
    assert _served(engine, prompts) == want
    cls, n = store.tail
    c = list(store.classes.values())[cls]
    end = c.slots()[c.size - 1].data_ptr() + c.unit_bytes
    # the loan options' defaults (moe.loan_margin, moe.loan_taper, moe.loan_largest: provisional, docs/dev/v1-glm.md
    # D8g) lend the bytes the loan had before them, as FN's RTX 5080 lock's layout (RAM holds every expert) takes it
    align, tokens = store._align, _profiled_tokens(engine.ctx.options)
    before = lambda share: -(-(int(share) + LOAN_SLACK) // align) * align                   # noqa: E731
    assert store._loan_bytes == before(store._peak_per_token * tokens * LOAN_MARGIN)
    buffers = lambda at: max(f(at) for f in store.tail_bytes)                               # noqa: E731
    assert store.borrowed["ring"] and ranges                 # each the tail's last bytes (512-byte aligned), its
    for at, nbytes, tokens, loan, tail, gu in ranges:        # step's share of the worst step's loan
        share = store._loan_per_token * tokens
        assert at + nbytes == end and at % 512 == 0 and share < nbytes + 512 and nbytes <= store._loan_bytes
        assert loan == gu == min(store._loan_bytes, before(share))
        assert tail == (cls, min(n, -(-(buffers(tokens) + loan) // c.unit_bytes)))
    assert store._loan.held() == 0 and store._loan.lib.loan_overflow() == 0     # the range held every segment
    kept, real = [], store.activations

    def leaky(step):                                 # the plant: a buffer made in the loan and kept
        loan = real(step)
        if loan is None:
            return None

        @contextlib.contextmanager
        def keeping():
            with loan:
                kept.append(torch.empty(1 << 16, device="cuda"))
                yield
        return keeping()
    store.activations = leaky
    engine.add([(t + 7) % 512 for t in prompts[-1]], GREEDY)              # no prefix hit: a streaming step
    with pytest.raises(RuntimeError, match="outlive"):
        while engine.has_work():
            engine.step()
    assert kept
    kept.clear()
    engine.close()                                  # (the loan's range free for the next engine)


_LENT_STEPS = """
import sys
import torch
from holon.api import SamplingParams
from holon.core.engine import Engine, EngineConfig
engine = Engine(EngineConfig(model=sys.argv[1], device="cuda", dtype=torch.bfloat16, page_size=16, max_seqs=4,
                             max_step_tokens=256, max_model_len=512, host_memory=1 << 27, plugin_options={
                                 "moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.ring_lend": "tail",
                                 "moe.lend_peak": "on"}))
loan, uses = engine.ctx.shared_objects["std.moe.offload"]._loan, []
use = loan.use
loan.use = lambda r: (uses.append(r.numel()), use(r))[1]
g = torch.Generator().manual_seed(0)
for _ in range(int(sys.argv[2])):
    n = int(torch.randint(40, 250, (1,), generator=g))
    engine.generate([torch.randint(0, 512, (n,), generator=g).tolist()], SamplingParams(temperature=0.0, max_tokens=1))
assert len(uses) >= int(sys.argv[2]), len(uses)
"""


@pytest.mark.slow
@GPU
def test_lent_steps_leave_the_host_heap_whole(gpu_moe, tmp_path):
    """A process serving 600 lent steps under glibc's heap checks (every Python object on glibc's heap, freed memory
    poisoned) exits cleanly. The loan's `held` was torch's `MemPool.snapshot()`, which on torch 2.11 corrupts the host
    heap: a glibc abort in a later, unrelated allocation within a few hundred steps (GLM step 7's engine restarts,
    about one a thousand requests); it is now the native loan's bytes still out after `release`."""
    file, _, _ = gpu_moe
    script = tmp_path / "lent.py"
    script.write_text(_LENT_STEPS)
    env = os.environ | {"PYTHONMALLOC": "malloc", "MALLOC_CHECK_": "3", "MALLOC_PERTURB_": "165",
                        "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    run = subprocess.run([sys.executable, str(script), str(file), "600"], env=env, capture_output=True, text=True,
                         timeout=1200)
    assert run.returncode == 0, run.stderr[-3000:]


@GPU
@pytest.mark.parametrize("context", [{}, {"max_seqs": 1, "max_model_len": 128}], ids=["step", "short-context"])
def test_a_lent_steps_loan_holds_its_fixed_bytes(gpu_moe, context):
    """`moe.loan_fixed_mib` (GLM step 5): a lent step's loan is its share of the worst step's by its prompt tokens plus
    the fixed bytes (a step's fixed costs do not shrink with its tokens: mid-size steps once overflowed their loans
    into the device's own memory and ran out of it), at most the worst step's loan — which holds them too (step 6: FN's
    4K-token steps with prompt logprobs overflowed a worst-step loan without them). The resident store's bits. The
    worst step's tokens are those the startup profile ran: at a context shorter than a step (`short-context`:
    `holon_tools profile --max-model-len 4096` on GLM's 32K-step deployment) the profile's worst step is `max_seqs`
    rows of `max_model_len` tokens, and a share counted per `max_step_tokens` left a 1,293-token step 32 MiB short."""
    file, prompts, want = gpu_moe
    opts = {"moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.ring_lend": "tail", "moe.lend_peak": "on",
            "moe.loan_fixed_mib": 32}
    engine = _gpu_engine(file, host_memory=1 << 27, plugin_options=opts, **context)
    store, cfg = engine.ctx.shared_objects[KEY], engine.cfg
    worst = min(cfg.max_step_tokens, cfg.max_seqs * cfg.max_model_len)
    assert abs(store._loan_per_token * worst + (32 << 20) - store._loan_bytes) <= store._align
    if not context:                                  # (the short context's prompts do not fit it: its sizing only)
        loans, use = [], store._loan.use
        store._loan.use = lambda r: (loans.append((store.step_tokens, r.numel())), use(r))[1]
        assert _served(engine, prompts) == want
        assert loans and all(n >= min(store._loan_per_token * t + (32 << 20), store._loan_bytes) - 1024
                              for t, n in loans)
    engine.close()


@GPU
def test_a_loan_overflow_is_counted_and_said_once(gpu_moe, capsys):
    """Step 5's review B m1: a lent step whose activations outgrow its loan (here its per-token share forced to 0 and
    its fixed bytes the 20 MiB floor) takes the device's own memory; the store counts each such step and its most bytes,
    and says so once, naming `moe.loan_fixed_mib`. The resident store's bits."""
    file, prompts, want = gpu_moe
    opts = {"moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.ring_lend": "tail", "moe.lend_peak": "on"}
    engine = _gpu_engine(file, host_memory=1 << 27, plugin_options=opts)
    store = engine.ctx.shared_objects[KEY]
    store._loan_per_token, store._loan_fixed = 0.0, 1 << 20        # (a loan of 1 MiB: every lent step overflows)
    assert _served(engine, prompts) == want
    engine.add([3] * 4, GREEDY)                                     # (a lent step is counted at the next)
    while engine.has_work():
        engine.step()
    said = capsys.readouterr().out
    assert store.loan_overflows >= 1 and store.loan_overflow_most > 0 and said.count("moe.loan_fixed_mib") == 1
    engine.close()


@GPU
@pytest.mark.parametrize("cache, kept", [("*=0.72", "own tail"), ("*=0.5", "reserved")])
def test_a_loan_the_layout_cannot_lend_stays_reserved(gpu_moe, capsys, cache, kept):
    # (a cache of 23 a class keeps its own tail at tails of 14-15 slots, this loan's 14: one slot of room for a peak
    # the profile measures larger in a long-running process; 22 a class, at 13-14, ran out at 15 in the full suite)
    """`moe.lend_peak` where the layout cannot lend (decided at build, never refused at a launch): a cache too small to
    lend a prompt step's staging keeps the loan in its own tail (the pool's last bytes, the cache that much smaller),
    each large step's activations allocated at its end; a tail class too small to hold the loan beside a pass's
    staging drops the loan (the engine reserves the worst step's peak). Both say so, and serve the resident store's
    bits."""
    file, prompts, want = gpu_moe
    opts = {"moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.ring_lend": "tail", "moe.lend_peak": "on",
            "moe.cache": cache, "moe.loan_largest": 0.9}
    engine = _gpu_engine(file, host_memory=1 << 27, plugin_options=opts)
    store = engine.ctx.shared_objects[KEY]
    ranges = []
    if store._loan is not None:
        use = store._loan.use
        store._loan.use = lambda r: (ranges.append((r.data_ptr(), r.numel())), use(r))[1]
    assert _served(engine, prompts) == want
    said = capsys.readouterr().out
    if kept == "own tail":
        at, n = store._own_tail
        end = store.arena.view(torch.uint8, (store.arena.capacity * store.granule,))[at:at + n][-1:].data_ptr() + 1
        assert not store.lending and n >= store._loan_bytes and ranges and "own tail" in said
        assert all(a + nbytes == end for a, nbytes in ranges)
    else:
        assert store._loan is None and store._loan_bytes == 0 and "stay reserved" in said
        assert store._largest(256) == 0             # no loan, so no step asks its tail for a range (A's D8k m1)
    engine.close()


@pytest.fixture(scope="module")
def glm_shaped(tmp_path_factory):
    """`gpu_moe_wide`'s model in GLM's shape — gate/up Q4_0, down Q8_0 (layers 0-1) and Q5_0 (2-3): down pieces near a
    gate/up piece's size, as GLM's IQ2_XXS, IQ3_XXS and IQ2_S —, its prompts and the resident store's outputs."""
    return _wide(tmp_path_factory, Q.Q4_0, {"0.ffn_down_exps": Q.Q8_0, "1.ffn_down_exps": Q.Q8_0,
                                            "ffn_down_exps": Q.Q5_0})


@GPU
def test_a_layout_whose_down_tails_leave_no_staging_does_not_lend(glm_shaped):
    """D8w's whole-expert tail lends each down class's last slots too: a layout lends only where every class with a
    tail still stages a prompt pass's misses outside it — decided at build, not flagged by a step's kernels (errors 4:
    a 24 GB card at mode (ii)'s flags, its IQ3_XXS down class 377 slots, a tail of 227, a pass's 288 misses). Here a
    200 MiB budget caches ≈ 160 of each down class's 256 pieces, a tail of 77 in each and a pass of 128 misses: the
    layout stages (its own tail), and serves the resident store's bits."""
    file, prompts, _ = glm_shaped
    one = {"max_seqs": 1, "prefix_cache": False}    # (its KV holds one sequence: the reference schedules alike)
    reference = _gpu_engine(file, **one)
    want = _served(reference, prompts)
    reference.close()
    opts = {"moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.ring_lend": "tail", "moe.lend_peak": "on",
            "moe.ram": "*=0.25", "moe.loan_fixed_mib": 64, "moe.placement": "placement.promote@1",
            "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"}
    engine = _gpu_engine(file, host_memory=1 << 29, plugin_options=opts, **one,
                         memory_fraction=(200 << 20) / torch.cuda.mem_get_info()[0])
    store = engine.ctx.shared_objects[KEY]
    names = list(store.classes)
    short = [c for c, m in store.tail_down.items() if store._lend_room[names[c]] - m < store.prefill]
    assert store.whole_tail and store.tail_down and short, (store._lend_room, store.tail_down, store.prefill)
    assert not store.lending
    assert _served(engine, prompts) == want and _bytes_exact(store) == 0
    engine.close()


def test_a_whole_expert_tail_fits_glms_mode_ii_geometry():
    """D8w at GLM's RTX 5090 mode (ii) geometry (no GPU: the store's sizing alone): gate/up IQ2_XXS 4.125 MiB a slot,
    3,265 cached; down IQ3_XXS 3.0625 MiB on 8 of 42 layers, 611 cached; down IQ2_S 2.5625 MiB on 34, 2,654 cached;
    a pass's staging 288; the 16,384-row passes' buffers ≈ 2.05 GiB and the 32K loan 7,948 MiB. The tail holds the
    passes' buffers in gate/up's slots (their views are one range: the startup profile once found it short, the
    buffers allocated beside it, the peak and the loan doubled, and the layout refused to start), the loan in the
    rest and the down classes' tails, each down class keeping a pass's staging; and ≈ 1.6× fewer experts than a
    gate/up-only tail of the same bytes."""
    from holon_std.tiered.store import SlotClass, TieredStore

    mib = 1 << 20
    classes = {"gate_up": SlotClass("gate_up", 4224 * 1024, 1, 32, cache=3265),
               "down.iq3": SlotClass("down.iq3", int(3.0625 * mib), 1, 32, cache=611),
               "down.iq2": SlotClass("down.iq2", int(2.5625 * mib), 1, 32, cache=2654)}
    store = object.__new__(TieredStore)
    store.exclusive, store.roles, store.prefill, store.classes, store._auxiliary = True, 2, 288, classes, []
    store._order = list(range(42))
    store.groups = {g: {0: classes["gate_up"], 1: classes["down.iq3" if g < 8 else "down.iq2"]} for g in range(42)}
    buffers, loan = int(2.05 * 1024 * mib), 7948 * mib
    n = store._split(buffers, loan, 0, lambda c: c.cap)
    m = store._down(n)
    unit = classes["gate_up"].unit_bytes
    assert n * unit >= buffers + 2 * 1024 * mib                         # the buffers, and the 32K R (2 GiB) beside
    assert n * unit + sum(k * list(classes.values())[c].unit_bytes for c, k in m.items()) >= buffers + loan
    assert all(list(classes.values())[c].cap - k >= store.prefill for c, k in m.items()), m
    for small in (0, 256 * mib):                                        # a step's own (the profile's: no loan)
        assert store._split(buffers, small, 0, lambda c: c.cap) * unit >= buffers
    gate_up_only = -(-(buffers + loan) // unit)
    assert 1.3 < gate_up_only / n < 1.7 and sorted(m) == [1, 2], (n, gate_up_only, m)


_WHOLE = [("", None), ("placement.promote@1", None)] + [
    pytest.param("placement.promote@1", plant, marks=pytest.mark.slow) for plant in (
        "tail_down_not_compacted", "compaction_map_stale", "compaction_return_wrong_layer", "compaction_beside_moves",
        "loan_range_overlap")]


@GPU
@pytest.mark.parametrize("policy, plant", _WHOLE)
def test_a_whole_expert_tail_lends_its_experts_whole(gpu_moe, policy, plant):
    """D8w (docs/dev/v1-glm.md): under exclusive tiers the tail's experts are lent whole — the tail's class's last n
    slots and the down class's last m, its residents relocated into the slots the lent experts free (copied after their
    pieces are parked, before the forward writes the loan), so the loan's ranges hold no mapped piece; a layer whose
    expert was lent gets the tail slot back at the step's end; each slot is held once; the host mirror (map and empty
    stacks) is the device's; every piece's bytes are the file's and the streams the resident store's, with the
    tail's class holding only part of the loan. Each of the compaction's plants is caught (D8w's list; `slow`).
    `moe.lend_peak = require` (D9): this mode (ii)-shaped layout — RAM holding a quarter of the experts, three quarters
    cached, tail lending — starts, its large steps stage (no ring under exclusive tiers) with their activations in the
    lent tail, a range a class with a tail, under F11 and under F10 (its spares in the tail released, their slots
    returned: step 1's review S1-M1, D8a); a layout that would lend none, or whose startup profile did not run (an
    explicit state budget: it once served on, lending nothing, until a large step ran out of memory), refuses to start,
    naming why."""
    from holon_judge import planted

    file, prompts, want = gpu_moe
    opts = {"moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.ring_lend": "tail",
            "moe.lend_peak": "require", "moe.cache": "*=0.75", "moe.ram": "*=0.25", "moe.placement": policy,
            "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"}
    with planted.PLANTS[plant]() if plant else contextlib.nullcontext():
        engine = _gpu_engine(file, host_memory=1 << 27, plugin_options=opts)
        store = engine.ctx.shared_objects[KEY]
        checks, lend_tail, ranges, use = [], store.lend_tail, [], store._loan.use_ranges
        store._loan.use_ranges = lambda rs: (ranges.append([r.numel() for r in rs]), use(rs))[1]

        def watched(cls, n):
            before = store.residency.device.clone()
            picked = lend_tail(cls, n)
            kept = (before >= 0) & (store.residency.device >= 0)
            checks.append((_loan_free(store), int((kept & (before != store.residency.device)).sum())))
            return picked
        store.lend_tail = watched
        # whole-expert placement keeps an expert's pieces at aligned slots here (one down class, every layer's);
        # GLM's down classes cover disjoint layers, so its slots do not stay aligned: the down slots shuffled
        torch.cuda.synchronize()
        res, down = store.residency.device, store._class_index[:, 1].long()
        gen = torch.Generator().manual_seed(0)
        for c in store.tail_down:
            mine = (down == c)[:, None] & (res[..., 1] >= 0)
            slots = res[..., 1][mine]
            moved = slots[torch.randperm(len(slots), generator=gen).to(slots.device)]
            arena = list(store.classes.values())[c].slots()
            arena[moved.long()] = arena[slots.long()].clone()
            res[..., 1][mine] = moved
        store.residency.host[...] = res.cpu().numpy()
        try:
            got = _served(engine, prompts)
            held = _slots_held(store)
            for shift in (7, 11):                            # more steps: a returned slot's next admissions, and
                engine.generate([[(t + shift) % 512 for t in p] for p in prompts], GREEDY)    # F10's spares (refilled
            torch.cuda.synchronize()                                                       # by the decodes) released
            ok = (got == want, all(free for free, _ in checks), _slots_once(store), _bytes_exact(store) == 0)
        except RuntimeError as e:                            # (a plant the store or the provider refuses)
            assert plant is not None, e
            ok = ("overlap" in str(e),) if plant == "loan_range_overlap" else (False,)
        if plant is None:
            assert store.whole_tail and store.tail_down and any(moved for _, moved in checks), checks
            assert all(ok), ok
            assert store.exclusive and store.borrowed["stage"] and not store.borrowed["ring"], store.borrowed
            assert ranges and max(map(len, ranges)) == 1 + len(store.tail_down), ranges
            assert torch.equal(_slots_held(store), held)
            _held_once(store)
            mirror = store.residency
            assert (mirror.host == store.residency.device.cpu().numpy()).all()
            n_empty, empty = store.n_empty.cpu().numpy(), store.empty.cpu().numpy()
            assert all(len(mirror.empty[g]) == n_empty[g] and all((x == empty[g, :, j]).all() for j, x in
                                                                   enumerate(mirror.empty[g]))
                       for g in range(len(n_empty)))
            unit = list(store.classes.values())[store.tail[0]].unit_bytes
            assert store.tail[1] * unit < store._loan_bytes     # the tail's class holds only part of the loan
        elif plant == "loan_range_overlap":
            assert ok == (True,), ok
        else:
            assert not all(ok), (plant, ok)
        engine.close()
    if plant is None:
        # (the refusals without exclusive tiers: an engine refused at its start is never closed, and an exclusive
        # store's D8x move thread kept it, ~190 MiB of VRAM and its pinned pools, for the rest of the worker)
        bare = {k: opts[k] for k in ("moe.store", "moe.ring_tokens", "moe.ring_lend", "moe.lend_peak", "moe.placement")}
        with pytest.raises(RuntimeError, match="require.*moe.ring_lend is not tail"):
            _gpu_engine(file, host_memory=1 << 27, plugin_options=bare | {"moe.ring_lend": "coldest"})
        engine = _gpu_engine(file, host_memory=1 << 27, state_memory=64 << 20, plugin_options=bare)
        with pytest.raises(RuntimeError, match="startup profile did not run"):
            _served(engine, prompts)
        engine.close()


@GPU
def test_a_loan_takes_its_margin_its_tapered_fixed_bytes_and_a_range_its_largest_allocation_fits(glm_shaped):
    """The 32K borrow (docs/dev/v1-glm.md §13, step 7): `moe.loan_margin` scales the lent peak, `moe.loan_taper` takes
    the fixed bytes whole up to its first token count and down to LOAN_SLACK at its second, and under a whole-expert
    tail `moe.loan_largest` sizes the tail so one loan range holds that share of a step's peak — the gate/up range is
    all the tail's gate/up slots past the passes' buffers. Lent steps serve the resident store's outputs."""
    from holon_std.tiered.store import LOAN_SLACK, _profiled_tokens

    file, prompts, want = glm_shaped
    opts = {"moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.ring_lend": "tail",
            "moe.lend_peak": "require", "moe.cache": "*=0.75", "moe.ram": "*=0.25",
            "moe.placement": "placement.promote@1", "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1",
            "moe.loan_fixed_mib": 64, "moe.loan_margin": 1.2, "moe.loan_taper": "64,192", "moe.loan_largest": 0.9}
    engine = _gpu_engine(file, host_memory=1 << 29, plugin_options=opts)
    store = engine.ctx.shared_objects[KEY]
    tokens, align = _profiled_tokens(engine.ctx.options), store._align
    peak = store._peak_per_token * tokens
    assert store._loan_bytes == -(-(int(peak * 1.2) + LOAN_SLACK) // align) * align       # (256 tokens: past 192)
    share = [store._loan_per_token * at for at in (64, 128, 192)]
    assert [store._step_loan_bytes(at) for at in (64, 128, 192)] == [
        -(-(int(s) + f) // align) * align for s, f in zip(share, (64 << 20, 32 << 20, LOAN_SLACK), strict=True)]
    cls, _ = store.tail                          # a largest allocation past every range the loan's sizing gives
    classes, cap = list(store.classes.values()), (lambda c: c.cap)
    buffers, loan = max(f() for f in store.tail_bytes), store._loan_bytes
    plain, n = store._split(buffers, loan, cls, cap), store._split(buffers, loan, cls, cap, 2 * loan)
    ranges = [n * classes[cls].unit_bytes - buffers, *(m * classes[c].unit_bytes for c, m in store._down(n).items())]
    assert n > plain and max(ranges) >= 2 * loan, (plain, n, ranges, loan)
    steps, use = [], store._loan.use_ranges

    def recorded(ranges):
        cls, n = store._step_tail
        unit = list(store.classes.values())[cls].unit_bytes
        buffers = max(f(store.step_tokens) for f in store.tail_bytes)
        steps.append((max(len(r) for r in ranges), store._largest(store.step_tokens),
                      store._step_gu_loan == n * unit - buffers))
        return use(ranges)
    store._loan.use_ranges = recorded
    assert _served(engine, prompts) == want
    assert steps and all(big >= need > 0 and whole for big, need, whole in steps), steps
    assert store.whole_tail and store.loan_overflows == 0
    engine.close()


def _tail_sized(buffers: int, peak: int):
    """The startup profile's `lend_activations(peak)` on a store of `gpu_moe_wide`'s model requantized to GLM's shape,
    down pieces near a gate/up piece's size (no engine: the tail's sizing alone; GLM: IQ2_XXS, IQ3_XXS and IQ2_S):
    gate/up Q4_0 288 KiB a slot, 384 cached; down Q5_0 176 KiB (layers 2-3) and Q8_0 272 KiB (0-1), 192 cached each;
    a pass's staging 128; the loan at margin 1, its largest range the whole peak; the profile's worst step of 256
    tokens; the passes' buffers `buffers` bytes."""
    from holon_std.tiered.store import LOAN_SLACK, SlotClass, TieredStore

    classes = {name: SlotClass(name, unit, 8, 32, cache=cache) for name, unit, cache in (
        ("gate_up", 294912, 384), ("down.q5_0", 180224, 192), ("down.q8_0", 278528, 192))}
    store = object.__new__(TieredStore)
    store.name, store.exclusive, store.roles, store.prefill, store.classes = "moe", True, 2, 128, classes
    store._order, store._auxiliary = list(range(4)), []
    store.groups = {g: {0: classes["gate_up"], 1: classes["down.q8_0" if g < 2 else "down.q5_0"]} for g in range(4)}
    store.lend_peak, store._require_loan, store.ring_lend, store.ring_tokens = True, True, "tail", 32
    store._loan_margin, store._loan_taper, store._loan_largest, store._loan_fixed = 1.0, None, 1.0, LOAN_SLACK
    store._align, store.tail_bytes = 16384, [lambda tokens=None: buffers]
    store.policy = types.SimpleNamespace(device_owned=True)
    loan = types.SimpleNamespace(impl=lambda ctx: None)                      # (the loan kernel: not used here)
    store.ctx = types.SimpleNamespace(options={"max_step_tokens": 256, "max_seqs": 4, "max_model_len": 512},
                                      candidates=lambda *a: [loan], select_kernel=lambda *a, **kw: loan)
    assert store.lend_activations(peak) == store.ring_tokens and store.whole_tail
    return store


@pytest.mark.parametrize("pass_rows", [512, 2048, 4096, 8192, 16384])
def test_a_tail_holds_a_loan_range_for_the_largest_allocation_at_every_pass_size(pass_rows):
    """D8k (docs/dev/v1-glm.md): the tail the startup profile sizes holds one loan range of a step's largest
    allocation (`moe.loan_largest` of the profiled peak) whatever its passes' buffers, at every `ggml_cuda.pass_rows`
    choice (2 KiB a row here: 1-32 MiB beside a 30 MiB peak), and so does a lent step's share of it — on GLM's shape,
    down pieces near a gate/up piece's size (`_tail_sized`). The tail was sized before the profile's peak per token was
    set, its largest range there by chance: GLM's 32K step at 8,192-row passes found none for its 2 GiB residual
    streams."""
    from holon_std.tiered.store import _profiled_tokens

    peak, buffers = 30 << 20, pass_rows << 11
    store = _tail_sized(buffers, peak)
    tokens = _profiled_tokens(store.ctx.options)
    big, cap, classes = store._largest(tokens), (lambda c: c.cap), list(store.classes.values())
    cls, n = store.tail
    for n_at in (n, min(n, store._split(buffers, store._step_loan_bytes(tokens), cls, cap, big))):    # the step's
        ranges = [n_at * classes[cls].unit_bytes - buffers,
                  *(m * classes[c].unit_bytes for c, m in store._down(n_at).items())]
        assert max(ranges) >= big > 0.99 * peak, (pass_rows, n_at, ranges, big)


@GPU
@pytest.mark.parametrize("passes", ["one", "two"])
def test_exclusive_tier_steps_fit_the_profiled_reserve(request, passes):
    """Step 1's review C S1-m2 (test_reserve.py's rule under exclusive tiers, docs/dev/v1-glm.md D9): with the loan
    required, each step's peak beyond what was allocated before it is at most the profiled reserve plus the step's
    own activation loan — staged prompt steps of a full width and a tail, and a prompt's chunks beside decode rows (a
    step with a decode row never reads ahead: docs/dev/v1-glm.md D8f); and (`two`, B's D8h-m1) steps of two passes
    whose layers gather once (D8h: every pass's routes held across the first pass)."""
    from holon.core import platform
    from holon_std.moe.offload import Offload
    from holon_std.tiered import store as tiered_store

    two, unions, union = passes == "two", [], Offload.route_union
    file, _, _ = request.getfixturevalue("glm_shaped" if two else "gpu_moe")     # (wide: a loan beside 2 passes)
    long = {"max_step_tokens": 1024, "max_model_len": 2048} if two else {}
    stack = contextlib.ExitStack()
    if two:                                          # (D8g's stage mode from any size: steps of two passes gather once)
        stack.enter_context(mock.patch.object(tiered_store, "ROUNDS_BELOW", 0))
        stack.enter_context(mock.patch.object(Offload, "route_union",
                                              lambda self, ids, x: unions.append(len(ids)) or union(self, ids, x)))
    peaks, measure = [], platform.peak_memory
    opts = {"moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.ring_lend": "tail",
            "moe.lend_peak": "require", "moe.cache": "*=0.75", "moe.ram": "*=0.5",
            "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"}
    with stack, mock.patch.object(platform, "peak_memory",
                                  lambda device, fn: peaks.append(measure(device, fn)) or peaks[-1]):
        engine = _gpu_engine(file, host_memory=1 << (29 if two else 28), plugin_options=opts, **long)
        store, reserve, gen = engine.ctx.shared_objects[KEY], peaks[-1], torch.Generator().manual_seed(3)
        _reserve_steps(engine, store, reserve, gen, 1300 if two else 300)
    assert two == bool(unions) and all(n > 512 for n in unions), unions
    engine.close()


def _reserve_steps(engine, store, reserve, gen, prompt) -> None:
    """`test_exclusive_tier_steps_fit_the_profiled_reserve`'s steps: two decoding rows, then a prompt of `prompt`
    tokens beside them; each step's peak within the reserve and its own loan."""
    store.read_ahead_tokens, hooks, ahead = 8, engine.executor._before, []

    def launched(step, launch=store.before_launch):  # each step: (a decode row, read ahead) — never both (C S1-m3)
        out = launch(step)
        ahead.append((bool(np.isin(step.row_kind, (RowKind.DECODE, RowKind.VERIFY)).any()), store._ahead))
        return out
    hooks[hooks.index(store.before_launch)] = launched
    params = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)
    for n in (40, 40, prompt):                       # two decoding rows, then a prompt beside them
        engine.add(torch.randint(0, 512, (n,), generator=gen).tolist(), params)
        if n == 40:
            engine.step()
    loans = []
    while engine.has_work():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        engine.step()
        torch.cuda.synchronize()
        loans.append(store._step_loan)
        took = torch.cuda.max_memory_allocated() - before
        assert took <= reserve + store._step_loan, (took, reserve, store._step_loan)
    assert store.exclusive and any(loans) and store.borrowed["stage"], (loans, store.borrowed)
    assert (False, True) in ahead and (True, True) not in ahead and (True, False) in ahead, ahead


@GPU
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


@GPU
@pytest.mark.parametrize("plant, terms", [(None, "default"), (None, "mode2")] + [
    pytest.param(p, terms, marks=pytest.mark.slow) for p, terms in (
        ("loan_parking_late", "default"), ("loan_write_before_parked", "default"), ("parking_event_early", "default"),
        ("loan_wait_unrecorded", "default"), ("loan_write_before_parked", "mode2"))])
def test_a_tail_steps_loan_experts_are_parked_before_the_forward_writes_the_loan(glm_shaped, mode2, plant, terms):
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

    file, prompts, want = glm_shaped
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
def glm_fresh(glm_shaped):
    """Two fresh prompts for `glm_shaped`, each long enough to lend, and the resident store's greedy streams of each
    served alone (a prompt step of its own, then its decode)."""
    file, _, _ = glm_shaped
    gen = torch.Generator().manual_seed(3)
    fresh = [torch.randint(0, 512, (n,), generator=gen).tolist() for n in (256, 250)]
    engine = _gpu_engine(file)
    want = [engine.generate([p], GREEDY) for p in fresh]
    engine.close()
    return fresh, want


@GPU
@pytest.mark.parametrize("plant", [None] + [pytest.param(p, marks=pytest.mark.slow) for p in (
    "parked_row_superset_runs", "row_hit_before_parked", "row_move_before_unmapped", "row_mapped_before_copied",
    "row_reused_before_moved", "row_reused_before_mapped", "reserved_unit_named", "row_moved_after_promoted")])
def test_a_short_prompt_parks_its_lent_experts_in_the_read_ahead_rows(glm_shaped, glm_fresh, mode2, plant):
    """D8x (docs/dev/v1-glm.md), under the deployment's terms: a short prompt step (no decode row, below the read-ahead
    steps' size: every prompt step here) parks the lent experts RAM has no free unit for in the idle read-ahead rows —
    its later layers hit them there —; at its end each is paired with RAM's coldest of its classes, whose units it
    reserves; the thread copies the rows into them in chunks (4 here: several a prompt), chunk i mapped at launch
    (step + lag + i), the next short prompt mapping every chunk left first. Checked after every step: the bits are the
    resident store's, every piece the map names holds its bytes, each tier holds an expert once (a reservation
    accepted as such), no cache slot is lost; experts were parked, moved, and some routed meanwhile (reconciled). Each
    of D8x's plants breaks one rule and is caught (`holon_judge.planted._d8x`; `slow`, an engine each)."""
    from holon_judge.planted import d8x_plant

    file, prompts, want = glm_shaped
    with d8x_plant(plant) if plant else contextlib.nullcontext(collections.Counter()) as fired:
        engine = _gpu_engine(file, host_memory=1 << 29, plugin_options=mode2(0.25))   # (RAM full)
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
        fresh, streams = glm_fresh                 # a prompt step alone, its decode during its move
        same = got == want and [engine.generate([p], GREEDY) for p in fresh] == streams
        counts, stats = store.move_counts.tolist(), dict(store.move_stats)
        engine.close()
    if plant is None:
        assert same and not +bad, bad
        assert counts[0] > counts[1] > 0 and stats["entries"] > counts[2] > 0 and stats["moves"], (counts, stats)
    else:
        assert sum(fired.values()) and (not same or +bad), (plant, dict(fired), bad)
