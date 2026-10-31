"""The offload store on the GPU (docs/dev/v1-offload.md §6's exact class): kernels.ggml_cuda over class arenas,
kernels.expert_io's resolve and gathers, decode steps from CUDA graphs — bitwise the resident store's outputs under
every placement policy, with exclusive tiers (RAM and the SSD, the I/O worker's reads), the CPU's share of the misses
(bounded), the SSD reads' landing rows and read-ahead sets (docs/dev/v1-glm.md D8f, D8u), and the fences the store's
streams use."""
import contextlib
import os
import warnings
from unittest import mock

import numpy as np
import pytest
import torch
from holon.api import RowKind, SamplingParams
from holon_std.moe.offload import KEY
from holon_std.tiered.ring import Lent
from holon_testing.gguf_files import write

from .moe_engine_common import GREEDY, Q, _bytes_exact, _fit_at, _gpu_engine, _no_host_sync, _served, _tail_checked

PREBUILT = ("ggml_cuda", "expert_io", "cpu_task")  # CUDA extensions built if missing (conftest)
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("policy, cache", [("", ""), ("", "*=0.25"), ("", "*=0"), ("placement.lru@1", "*=0.25"),
                                           ("placement.profile_decay@1", "*=0.25"), ("std.tiered.static@1", "*=0.25"),
                                           ("", "*=0.5 ring"), ("", "*=0.5 dma"), ("", "*=0.75 tail"),
                                           ("", "*=0.25 split"), ("", "*=0.75 pipe"),
                                           ("", "*=0.25 ssd"), ("", "*=0.25 ssdfit"), ("", "*=0.25 fit"),
                                           ("", "*=0.25 ssd1"), ("", "*=0.25 ssdsplit"),
                                           ("std.tiered.static@1", "*=0.25 ssd"),
                                           ("placement.promote@1", "*=0.25 ssd")])
def test_the_offload_store_on_the_gpu_is_bitwise_the_resident_store(gpu_moe, policy, cache):
    """docs/dev/v1-offload.md §6's exact class on the GPU kernels (kernels.ggml_cuda over class arenas,
    kernels.expert_io's
    resolve, its F11 and F4 variants and zero-copy gather from mapped host pools), decode steps replayed from CUDA
    graphs and prefill chunked past the mmvq limit: the offload store's streams and prompt logprobs are the resident
    store's, bit for bit, under every placement policy (arenas of 42, 18, 10 slots: stream-k unpatched), and with RAM
    holding a quarter of the experts (exclusive tiers, D10: SSD reads on the I/O worker inside decode graphs; "ssd" its
    share by `fit` at the working floor's f, a quarter here, "ssdfit" at half, docs/dev/v1-glm-ram.md D2 — "fit" on a
    budget holding every expert is the inclusive store, its I/O worker started idle and D8g's gathers off (B-R10, the
    review's A-M1) —; at one
    sequence, "ssd1", its prompt steps borrow their staging from the cache, the lent experts parked in RAM over its
    coldest or sent to the SSD, while its own staging slots stay a graphed step's, and its SSD reads land in landing
    rows of two, a call's in rounds, its prompt steps of 8 tokens or more reading the next layers' sets ahead:
    docs/dev/v1-glm.md D8b+e, D8f). F11's
    residency, step by step, is its replay oracle's (L1 parity), with admissions in graphed and eager steps, the ring's
    copies on SMs or on copy engines; a borrowing step's lending and ring never wait for the device (with copy
    engines, but for one explicit sync to read the copy list). A
    corrupt map entry is flagged by the resolve and stops the engine at the commit; the kernels' id-error word is
    watched too. The arena pool is released when the ledger frees its pools (the startup profile reads free memory
    then)."""
    from holon_judge.placement import parity, route_dump

    file, prompts, want = gpu_moe
    cache, ring = cache.partition(" ")[::2]          # "ring" / "dma": steps of 32 prompt tokens or more stream (D8),
    one = ring == "ssd1"                             # (one sequence: a graphed step's staging below a pass's)
    if one:                                          # the resident store at one sequence (GPU bits follow the rows)
        resident = _gpu_engine(file, max_seqs=1)
        want = _served(resident, prompts)
        resident.close()
    ssd, split = ring in ("ssd", "ssdfit", "ssd1", "ssdsplit"), ring in ("split", "ssdsplit")   # their copies on SMs /
    #                                                copy engines, "tail" a tail (D11), "ssdsplit" both (D8c)
    share = {"ssd": None, "ssdfit": 0.5, "fit": 1.0}.get(ring, False)    # `fit` solving f: the floor's, half, all
    ring = "" if ssd or split or ring == "fit" else ring      # first; "ssd": RAM a quarter; "split": the decode passes'
    pipe = ring == "pipe"                            # resident entries first, the misses' fetch on a side stream;
                                                     # "pipe": prompt steps stage, their misses in 4 fetched parts
    from holon_std.tiered import store as tiered_store
    from holon_std.tiered.resolver import DeviceResolver
    tiers, splits = DeviceResolver._tiers, []                # exclusive tiers' calls, split or not (graphs captured
    spy_tiers = mock.patch.object(DeviceResolver, "_tiers",  # at the build included), and what each enqueues first
                                  lambda self, *a: splits.append(a[-1]) or order.append("call") or tiers(self, *a))
    order, rows, gather = [], tiered_store.TieredStore.round_rows, DeviceResolver._gather
    spy_rows = mock.patch.object(tiered_store.TieredStore, "round_rows",
                                 lambda self, r, **kw: order.append(f"rows {r}") or rows(self, r, **kw))
    spy_gather = mock.patch.object(DeviceResolver, "_gather", lambda self, f, n, landed=False, part=0: order.append(
        "ssd" if landed else "ram") or gather(self, f, n, landed=landed, part=part))
    with (spy_tiers, spy_rows, spy_gather, _fit_at(share) if share is not False else contextlib.nullcontext(),
          mock.patch.object(tiered_store, "ON_DEMAND", 2 if one else tiered_store.ON_DEMAND)):   # ssd1: rounds of 2
        engine = _gpu_engine(
            file, host_memory=1 << 28, **({"max_seqs": 1} if one else {}), plugin_options={
            "moe.store": "std.moe.offload@1", "moe.cache": cache, "moe.placement": policy,
            "moe.ring_tokens": 32 if ring else 512, "moe.ring_copy": "dma" if ring == "dma" else "sm",
            "moe.ring_lend": "tail" if ring in ("tail", "pipe") else "coldest", "moe.split": "on" if split else "off"}
            | ({"moe.stream_tokens": 1 << 20, "moe.pipeline": 4} if pipe else {})
            | ({"moe.ring_tokens": 32, "moe.stream_tokens": 1 << 20, "moe.split": "on"} if share == 1.0 else {})
            | ({"moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1",
                "moe.ram": "*=0.25" if share is False else "fit"} if ssd or share else {}))
    store = engine.ctx.shared_objects[KEY]
    if share is not False:                           # fit: f the floor's (each class a graphed step's 8 misses of its
        held = {c.held for c in store.classes.values()}          # 32 pieces: a quarter) or half, held ⌊f × pieces⌋
        assert (store.share, held) == {None: (0.25, {8}), 0.5: (0.5, {16}), 1.0: (1.0, {32})}[share], held
        assert store.exclusive == (share != 1.0)
    if ssd:                                          # (prompt steps of 8 tokens or more read ahead; never one with
        store.read_ahead_tokens = 8                  # a decode row: at 4 sequences, mixed steps keep: D8f)
    calls = []                                       # eager calls' SSD misses (round 0's count) and read-ahead hits
    spy_rows = mock.patch.object(tiered_store.TieredStore, "round_rows", lambda self, r, **kw: (
        r or calls.append(("ssd", self.n_ssd[:1].clone())), rows(self, r, **kw))[1])
    spy_hits = mock.patch.object(DeviceResolver, "_gather", lambda self, f, n, landed=False, part=0: (
        landed and part is None and calls.append(("hit", n[:1].clone())), gather(self, f, n, landed=landed,
                                                                                part=part))[1])
    commit, routed = store.after_commit, []          # per step: its decode rows' routes, and the counts the step left
    store.after_commit = lambda step, t: routed.append((int(step.n_query[np.isin(step.row_kind, (    # its decode and
        RowKind.DECODE, RowKind.VERIFY))].sum()) * 2 * 4, int(t["counts"].sum()))) or commit(step, t)   # verify rows
    mode = "warn" if ring == "dma" else "error"                   # "dma": one sync per streaming step, counted
    from holon_std.tiered.resolver import DeviceResolver
    piped, pipelined, rings = [], DeviceResolver._pipelined, []

    def piping(self, *a):                            # (a pipelined pass's misses against its staging ring's width)
        lent = self.store.lent
        if lent is not None and lent.width is not None and self.group < self.store.borrowing_groups:
            rings.append((a[3].max(), lent.width[torch.as_tensor(self.store.class_table[self.group])].min()))
        piped.append(a[-1])
        return pipelined(self, *a)
    spy = mock.patch.object(DeviceResolver, "_pipelined", piping)
    gathers = mock.patch.object(tiered_store, "ROUNDS_BELOW", 8 if share == 1.0 else tiered_store.ROUNDS_BELOW)
    with (spy, route_dump(engine) as trace, warnings.catch_warnings(record=True) as syncs,
          _no_host_sync(Lent, "begin", "sources", "done", mode=mode), _tail_checked(store) as tails, spy_rows,
          spy_hits, gathers):                        # ("fit": D8g's gathers' steps from 8 tokens, which it never runs)
        warnings.simplefilter("always")
        assert _served(engine, prompts, mirrored=store if one else None) == want
    assert bool(tails) == (ring in ("tail", "pipe"))
    syncs = [w for w in syncs if "called a synchronizing" in str(w.message)]
    assert len(syncs) == (store.borrowed["ring"] if ring == "dma" else 0), (len(syncs), store.borrowed)
    assert engine.stats()["graphs"] > 0 and (store.peak_misses > 0) == (cache != "")
    assert all(a == b for a, b in routed) and max(routed)[0] > (0 if one else 8)   # top-2 x 4 layers; no padding
    variant = {"": ("moe_resolve.lfu@1", "kernels.expert_io"),
               "placement.lru@1": ("moe_resolve.lru@1", "kernels.expert_io"),
               "placement.promote@1": ("moe_resolve.promote@1", "kernels.promote")}.get(
        policy, ("moe_resolve@1", "kernels.expert_io"))
    assert {(k.rpartition("/")[2], v["provider"]) for k, v in engine.ctx.kernels.items() if k.startswith("std.tiered")
            } == {variant, ("expert_fetch@1", "kernels.expert_io"),
                  ("expert_prefetch@1", "kernels.expert_io")} | ({("moe_tiers@2", "kernels.expert_io"),
                                                                 ("cpu_task@1", "kernels.cpu_task")} if ssd or share
                                                               else set())
    if share == 1.0:                                 # bound in the build, started, never signalled; no step runs
        io = engine.ctx.shared_objects["cpu_task.io"].native         # D8g's gathers in parts (the lock never does:
        assert io.served() == 0 and io.flag() == 0 and not piped     # A-M1) though its copy engines are bound
        assert store.copies and store.borrowed["stage"]
    if ssd:                                          # the SSD reads ran on the I/O worker, fail-open untouched
        io = engine.ctx.shared_objects["cpu_task.io"].native
        assert store.exclusive and io.served() > 0 and io.flag() == 0
        assert any(splits) == split              # "ssdsplit": RAM misses beside the hits' passes (D8c)
        starts = [i for i, e in enumerate(order) if e == "call"]  # round 0's reads requested before the RAM gather
        assert starts and all(order[i + 1:i + 3] == ["rows 0", "ram"] for i in starts), order[:8]   # (C S1-M1)
        # ssd1: a call's SSD misses past its landing rows, read in rounds of 2 (docs/dev/v1-glm.md D8f)
        peak = max((int(t) for k, t in calls if k == "ssd"), default=0)
        assert not one or (store.landing_rows == 2 and peak > 2), peak
        assert not one or sum(int(t) for k, t in calls if k == "hit") > 0     # ssd1: prompt steps of 8 tokens or
        #                                                                       more read their sets ahead
    if one:                                          # exclusive tiers lend their prompt steps' staging (GLM D8b+e)
        assert store.staging < store.prefill and store.lending and store.borrowed["stage"] > 0, store.borrowed
        assert int(store.parked) > 0                 # lent experts parked in RAM (D8b), read back by later misses
    if not policy:
        report = parity(trace)
        assert report["first_differing_step"] is None and (report["admissions"] > 0) == (cache != "" and cache != "*=0")
        assert report["device_is_mirror"] and bool(store.borrowed["ring"]) == bool(ring and not pipe), (
            report, store.borrowed)
        assert not pipe or (store.borrowed["stage"] and piped)
        # "pipe": its prompt steps' staging a ring of two parts (FN §21.16), less than a pass's worth: a pass's misses
        # past its ring's slots (reused)
        rings = [(int(n), int(w)) for n, w in rings]                 # (read after serving: no sync between passes)
        assert not pipe or (max(w for _, w in rings) < store.prefill and any(n > w for n, w in rings)), rings
    elif policy == "placement.profile_decay@1":
        assert store.policy.rounds > 0
    experts = {k.rpartition("/")[2]: v["provider"] for k, v in engine.ctx.kernels.items() if "linear_experts" in k}
    assert experts == {"linear_experts_arena@1": "kernels.ggml_cuda"}            # arenas never bind @2
    words = [t for name, t in store.telemetry.items() if name.startswith("kernel_errors")]
    assert len(words) == 1 and all(            # the ggml id-error word, one per engine (the I/O worker checks its own)
        m._kernel.errors is words[0] for m in engine.executor.model.modules() if hasattr(m, "cls"))
    if not cache:                                # a corrupt map entry: null slots read, the commit stops the engine
        store.residency.device[0, :, 0] = 1 << 20
        engine.add(prompts[0], GREEDY)
        with pytest.raises(RuntimeError, match=r"flagged errors \{'errors': 1\}"):
            while engine.has_work():
                engine.step()
    held, nbytes = torch.cuda.memory_allocated(), store.arena.capacity * store.arena.unit_bytes
    store.arena._set(0)                         # as the ledger frees it: no view outlives the arena pool's buffer
    assert held - torch.cuda.memory_allocated() >= nbytes > 0
    engine.close()


def test_an_ssd_read_past_its_time_bound_stops_the_engine(gpu_moe):
    """Exclusive tiers: the I/O worker fails open on the device (`cpu_task@1`: a wait past its time bound, or a failed
    task, answers at once, and every later wait too), and is fail-stop by construction: its own error word, zeroed at
    each launch, stops the engine at the commit — the store, its consumer, watches nothing (unchecked, the reads it
    gave up on left wrong bytes in the slots and the units, silently). With a bound no read meets the step's commit
    refuses to go on."""
    file, prompts, _ = gpu_moe
    engine = _gpu_engine(file, host_memory=1 << 27, plugin_options={
        "moe.store": "std.moe.offload@1", "moe.cache": "*=0.25", "moe.ram": "*=0.25", "cpu_task.timeout": 1e-6,
        "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"})
    store = engine.ctx.shared_objects[KEY]
    assert not any(t is store.io.errors for t in store.telemetry.values())        # no watch of its own
    with pytest.raises(RuntimeError, match=r"cpu_task\.io: step \d+'s CPU tasks failed open"):
        _served(engine, prompts)
    engine.close()


def test_a_start_refused_after_its_pools_releases_them(gpu_moe, monkeypatch):
    """An engine refused at its start is never closed: it releases what it built as `close` would. An exclusive-tier
    layout refused after its first materialize (`moe.lend_peak = require` with no tail to lend) once kept its store —
    the move thread held it —, ≈ 170 MiB of VRAM and its pinned pools, for the process's life."""
    import gc
    import threading
    import weakref

    from holon.core import platform

    regions, real = [], platform._Pinned

    class Counted(real):
        def __init__(self, *a, **k):
            super().__init__()
            regions.append(weakref.ref(self))
    monkeypatch.setattr(platform, "_Pinned", Counted)
    file, _, _ = gpu_moe
    torch.cuda.synchronize()
    held = torch.cuda.memory_allocated()
    with pytest.raises(RuntimeError, match="moe.ring_lend is not tail"):
        _gpu_engine(file, host_memory=1 << 27, plugin_options={
            "moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.ring_lend": "coldest",
            "moe.lend_peak": "require", "moe.cache": "*=0.75", "moe.ram": "*=0.25",
            "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"})
    gc.collect()
    assert len(regions) > 2 and not [r for r in regions if r() is not None]
    assert torch.cuda.memory_allocated() <= held                 # (an earlier test's engine may go too)
    assert not [t for t in threading.enumerate() if t.name.endswith(".move")]


def test_rams_long_heat_is_its_oracles(gpu_moe):
    """`moe.ram_heat` (docs/dev/v1-glm.md §7.4's lever 1, step 6): an exclusive RAM tier's victims by a long-memory heat
    of its own — each step's decode tokens decay it by the factor a token, then add the step's routed counts — which the
    tiers' placement and the parking rank by in place of the policy's. Against its oracle over the committed steps'
    counts (the last step's are folded at the next step's start), graphed decode steps included; the resident store's
    bits."""
    file, prompts, want = gpu_moe
    opts = {"moe.store": "std.moe.offload@1", "moe.cache": "*=0.5", "moe.ram": "*=0.25", "moe.ram_heat": 0.9,
            "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"}
    engine = _gpu_engine(file, host_memory=1 << 27, plugin_options=opts)
    store, counts = engine.ctx.shared_objects[KEY], []
    assert store.heat is store.ram_heat
    store.taps.append(lambda step, record: counts.append(record["counts"].astype(np.float64)))
    assert _served(engine, prompts) == want
    assert engine.generate([[5, 6, 7]], SamplingParams(temperature=0.0, max_tokens=12, ignore_eos=True))
    heat = np.zeros_like(counts[0])                   # (no profile: seeded at 0)
    for c in counts[:-1]:
        heat = heat * 0.9 ** (c[0].sum() / store.top_k) + c
    assert any(c.sum() for c in counts[:-1]) and np.allclose(store.ram_heat.cpu().numpy(), heat, rtol=1e-5, atol=1e-5)
    engine.close()
    with pytest.raises(ValueError, match="moe.ram_heat ranks an exclusive RAM tier"):
        _gpu_engine(file, host_memory=1 << 27,
                    plugin_options=opts | {"moe.tiers": "std.tiered.pinned@1", "moe.ram": ""})


def test_a_roles_classes_share_its_landing_rows(gpu_moe, tmp_path):
    """Step 1's review C S1-m2 (docs/dev/v1-glm.md D8f): the down role of two classes (layer 3's Q8_0, the others' Q4_0,
    as GLM's two down types) reads its SSD misses into one landing pool, row j at the same bytes for both (a layer's
    on-demand rows and the next layers' read-ahead sets never overlap); the streams are the resident store's. Every
    piece's runs in a landing row are as the store derives them from its descriptors (D8g's copies by copy engines:
    the I/O worker's rule)."""
    file, prompts, _ = gpu_moe
    mixed = tmp_path / "mixed.gguf"
    write(file.parent / "hf", mixed, Q.Q8_0, types={"3.ffn_down_exps": Q.Q8_0, "ffn_down_exps": Q.Q4_0})
    resident = _gpu_engine(mixed)
    want = _served(resident, prompts)
    resident.close()
    engine = _gpu_engine(mixed, host_memory=1 << 27, plugin_options={
        "moe.store": "std.moe.offload@1", "moe.cache": "*=0.25", "moe.ram": "*=0.25",
        "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"})
    store = engine.ctx.shared_objects[KEY]
    store.read_ahead_tokens = 8                      # (prompt steps of 8 tokens or more read ahead)
    assert _served(engine, prompts) == want
    down = [store.landing_view(name, device=False)[0] for name, c in store.classes.items() if c.pieces[0][2] == 1]
    assert len(down) == 2 and down[0].data_ptr() == down[1].data_ptr() and down[0].shape == down[1].shape
    assert store.exclusive and engine.ctx.shared_objects["cpu_task.io"].native.io_stats()["bytes"] > 0
    torch.cuda.synchronize()                         # D8g: each piece's runs as the I/O worker lays it out (idle)
    a, table = store.landing_rows, store.io_pool.view(torch.int32, (store.halves, store.roles, 1 + store.landing_rows,
                                                                    3))[0]
    for reads in (x for x in store.reads if not x.ahead and not x.half):
        g = reads.resolver.group
        classes = store.layer_classes[g]
        for at in range(0, store.items, a):
            pieces = [[store._piece_index(c, g, i, r) for i in range(at, min(at + a, store.items))]
                      for r, c in enumerate(classes)]
            for r, p in enumerate(pieces):
                table[r, 0, 0] = len(p)
                table[r, 1:1 + len(p)] = torch.tensor([[x, j, -1] for j, x in enumerate(p)], dtype=torch.int32)
            reads()
            for r, (c, p) in enumerate(zip(classes, pieces, strict=True)):
                runs = store.landing_view(c.name, device=False)[1][:len(p)].numpy()
                assert np.array_equal(runs, store.piece_runs[c.name][p]), (g, r, at)
    engine.close()



@pytest.mark.parametrize("heated", [True, False])
def test_a_read_ahead_step_reads_each_layers_hottest_ssd_only_experts(gpu_moe, heated):
    """D8u (b) (its narrow width tried and reverted; one pick kept): a read-ahead step reads ahead each layer's A_ra
    hottest SSD-only experts by the store's heat (`moe.ram_heat`), ties by id, into its set's rows in id order — here
    `moe.read_ahead` 2, so a set exceeds it: the hottest, not D8f's lowest ids —; without a heat the lowest ids
    (D8f's rule); the streams are the resident store's. A read-ahead signalled and never waited is refused at the next
    step's start (A's m2)."""
    from holon_std.tiered import store as tiered_store

    file, prompts, _ = gpu_moe
    resident = _gpu_engine(file)
    want = _served(resident, prompts)
    resident.close()
    width = 2                                        # (the tiny model's 8 experts a layer)
    engine = _gpu_engine(file, host_memory=1 << 27, plugin_options={
        "moe.store": "std.moe.offload@1", "moe.cache": "*=0.25", "moe.ram": "*=0.25", "moe.ram_heat": "0.999",
        "moe.read_ahead": str(width), "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"})
    store = engine.ctx.shared_objects[KEY]
    store.read_ahead_tokens = 8                      # (prompt steps of 8 tokens or more read ahead)
    heat = store.ram_heat                            # a heat against the ids' order, with ties (a pure prompt step's)
    heat.copy_((torch.arange(heat.numel(), device=heat.device).flip(0) % 3).view_as(heat).float())
    if not heated:
        store.heat = None
    seen, read = [], tiered_store.TieredStore.read_ahead

    def recorded(self, g):
        only = (self.residency.device[g] == -2).all(-1).cpu()
        h = self.heat[g].double().cpu() if self.heat is not None else torch.zeros(only.shape, dtype=torch.float64)
        read(self, g)
        seen.append((only, h, self.ahead_map[g].cpu().clone()))

    with mock.patch.object(tiered_store.TieredStore, "read_ahead", recorded):
        assert _served(engine, prompts) == want
        assert store.ahead_rows == width
        assert seen and any(int(only.sum()) > width for only, _, _ in seen)    # (the width bites,
        assert not heated or any(len(set(h[only].tolist())) > 1 for only, h, _ in seen)    # by heat)
        for only, h, rows in seen:
            ids = [int(e) for e in torch.nonzero(only).flatten()]
            hot = sorted(sorted(ids, key=lambda e: (-float(h[e]), e))[:width])     # the hottest, ties by id
            assert [int(e) for e in torch.nonzero(rows >= 0).flatten()] == hot
            assert rows[rows >= 0].tolist() == sorted(rows[rows >= 0].tolist())    # rows in id order, contiguous
        if heated:                                   # a read-ahead never waited: the next step refuses
            store.read_ahead(0)
            with pytest.raises(RuntimeError, match="never waited"):
                store.start_step()
            store.wait_ahead(0)
            torch.cuda.synchronize()
    engine.close()


def test_an_auxiliary_groups_call_in_a_read_ahead_step_finds_no_hit(gpu_moe):
    """A read-ahead step (D8f) reads ahead the target's layers only; an auxiliary model's group (an MTP head's: here the
    last layer taken as one, `borrowing_groups`) has no set, so its tier op's row map must name no row — the map
    starts at −1 at every materialize (zeroed, it named row 0 for every expert: an SSD miss of that layer gathered
    from whatever row 0 held) — and its call waits no read-ahead (one never signalled: the wait ran out its time bound,
    then the worker failed open). The streams are the resident store's."""
    from holon_std.tiered import store as tiered_store

    file, prompts, want = gpu_moe
    with mock.patch.object(tiered_store.TieredStore, "borrowing_groups", property(lambda self: len(self.groups) - 1)):
        engine = _gpu_engine(file, host_memory=1 << 27, plugin_options={
            "moe.store": "std.moe.offload@1", "moe.cache": "*=0.25", "moe.ram": "*=0.25",
            "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"})
        store = engine.ctx.shared_objects[KEY]
        store.read_ahead_tokens = 8                  # (prompt steps of 8 tokens or more read ahead)
        assert bool((store.ahead_map == -1).all())
        assert _served(engine, prompts) == want
        engine.close()


def test_a_stores_fence_orders_a_delayed_producer():
    """The fence std's stores use (`torch.Event` on `torch.accelerator`'s streams; scripts/ci/check_events.py): a
    consumer stream waiting on the event of a producer stream that writes late reads the producer's bytes — the
    pattern item 2's moves and the parking use (a generic stream ignores a CUDA-typed event's wait)."""
    device = torch.device("cuda")
    main = torch.accelerator.current_stream(device)
    producer, consumer = torch.Stream(device=device), torch.Stream(device=device)
    x = torch.zeros(1 << 20, dtype=torch.int32, device=device)
    y = torch.empty_like(x)
    producer.wait_stream(main)
    with producer:
        torch.cuda._sleep(50_000_000)
        x.fill_(7)
        done = torch.Event(device=device)
        done.record()
    consumer.wait_event(done)
    with consumer:
        y.copy_(x)
    main.wait_stream(consumer)
    assert bool((y == 7).all())


def test_a_lending_prompt_step_stages_in_a_single_buffer(gpu_moe):
    """Step 7, exclusive tiers: a lending prompt step stages in a single buffer of a layer's most missing pieces, not a
    pass of every expert (C's L3: with the coldest lent, 3 of 8 a class here, 8 before); the resident store's bits and
    every piece's bytes. (With the tail lent: `test_moe_engine_loan.py`'s whole-expert tail.)"""
    file, prompts, want = gpu_moe
    opts = {"moe.store": "std.moe.offload@1", "moe.ring_tokens": 32, "moe.ring_lend": "coldest", "moe.cache": "*=0.75",
            "moe.ram": "*=0.25", "moe.ram_heat": 0.9, "moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1"}
    engine = _gpu_engine(file, host_memory=1 << 27, plugin_options=opts)
    store, lends = engine.ctx.shared_objects[KEY], []
    store.taps.append(lambda step, record: lends.append(record["lends"]) if record["lends"] is not None else None)
    assert _served(engine, prompts) == want
    assert store.exclusive and store.borrowed["stage"]
    most = [int(x[1:].sum(0).max()) for x in lends]         # the rounds' lending: a layer's most missing, not `prefill`
    assert any(most) and max(most) < store.prefill, most
    assert _bytes_exact(store) == 0
    engine.close()


def test_the_cpus_share_on_the_gpu_is_bounded_and_run_identical(gpu_moe):
    """F5b on the native path (kernels.expert_io's resolve and row patch, kernels.cpu_task's worker in decode graphs,
    kernels.ggml_cpu's experts): steps of up to 4 tokens give the CPU half the misses (rounded up) at a quarter cached;
    prompt logprobs stay close to the resident store's at the same step size (bounded), two engines serve the same
    bits (static placement; F11's residency is its replay's, the CPU's misses never admitted; with exclusive tiers too,
    the I/O worker reading SSD pieces beside the CPU's), and a dead worker fails
    open: the wait gives up after its time bound and the step's commit stops the engine."""
    from holon_judge.placement import parity, route_dump

    file, prompts, _ = gpu_moe
    small = {"max_step_tokens": 4}                            # every prompt pass small enough for a CPU share
    engine = _gpu_engine(file, **small)
    want = _served(engine, prompts)
    engine.close()
    split = {"moe.store": "std.moe.offload@1", "moe.cache": "*=0.25", "moe.cpu_plan": "1,1,2,2",
             "cpu_task.threads": 2, "cpu_task.timeout": 0.5}
    short = {"moe.tiers": "std.tiered.pinned@1,std.tiered.gguf_direct@1", "moe.ram": "*=0.5"}   # + exclusive tiers
    runs = []
    for policy, ram in (("std.tiered.static@1", {}), ("std.tiered.static@1", {}), ("", short), ("", {})):
        engine = _gpu_engine(file, host_memory=1 << 28, plugin_options=split | ram | {"moe.placement": policy},
                             **small)
        with route_dump(engine) as trace:
            runs.append(_served(engine, prompts))
        worker = engine.ctx.shared_objects["cpu_task.compute"].native
        assert worker.served() > 0 and worker.flag() == 0 and engine.stats()["graphs"] > 0
        assert sum(s["cpu"].sum() for s in trace.steps) > 0
        if not policy:
            assert parity(trace)["first_differing_step"] is None and trace.admissions > 0
        if ram:                                   # the I/O worker beside the compute one, both clean
            io = engine.ctx.shared_objects["cpu_task.io"].native
            assert io.served() > 0 and io.flag() == 0
            close = [np.abs(np.array(a) - np.array(b)) for (_, _, a), (_, _, b) in zip(runs[-1][1], want[1],
                                                                                       strict=True) if a]
            assert np.median(np.concatenate([c.ravel() for c in close])) < 0.03
        if policy or ram:
            engine.close()
    assert runs[0] == runs[1]
    assert {(k.rpartition("/")[2], v["provider"]) for k, v in engine.ctx.kernels.items() if k.startswith("std.moe.cpu")
            } == {("cpu_task@1", "kernels.cpu_task"), ("cpu_experts@1", "kernels.ggml_cpu"),
                  ("moe_patch_rows@1", "kernels.expert_io")}
    diff = np.concatenate([np.abs(np.array(a) - np.array(b)).ravel() for (_, _, a), (_, _, b) in
                           zip(runs[0][1], want[1], strict=True) if a])
    same = np.mean([x[0] == y[0] for (_, ids, _), (_, ref, _) in zip(runs[0][1], want[1], strict=True)
                    for x, y in zip(ids, ref, strict=True)])
    assert np.median(diff) < 0.03 and diff.mean() < 0.05 and same > 0.85         # (measured 0.011, 0.016, 0.94; a
    #                                                                                routing flip moves a position)
    worker.kill()                                             # the coordinator stops answering
    engine.add(prompts[1], GREEDY)
    with pytest.raises(RuntimeError, match=r"cpu_task\.compute: .* errors 2"):        # the worker's own word
        while engine.has_work():
            engine.step()
    assert worker.flag() & 2
    engine.close()


_SIDE_STREAM = """
import sys, types, torch
from holon_std.ffn.units import Sum

class Double(torch.nn.Module):
    def forward(self, x):
        return x * 2

class Late(torch.nn.Module):                       # a first part that reads `acc` late, on the main stream
    joins = True

    def add_residual(self, x, acc, join):
        join()
        torch.cuda._sleep(50_000_000)
        return x + acc

units = {"first": Late(), "rest": Double()}
ctx = types.SimpleNamespace(device="cuda", options={}, unit=lambda p, **kw: units[p],
                            shared=lambda key, make: make(types.SimpleNamespace(device="cuda")))

def run(mode):
    s = Sum(ctx, hidden=1 << 22, parts=("first", "rest"))
    xs = [torch.full((1 << 22,), float(i + 1), device="cuda") for i in range(4)]
    torch.cuda.synchronize()
    ys = []
    if mode == "graph":                            # captured, the side stream's trailing wait inside the capture
        x = xs[0].clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y = s(x)
        for xi in xs:
            x.copy_(xi)
            graph.replay()
            ys.append(y.clone())
    for x in xs if mode != "graph" else ():
        ys.append(s(x))
        if mode == "release":
            torch.cuda.empty_cache()               # what an out-of-memory retry does
    torch.cuda.synchronize()
    return [float(y.min()) for y in ys] == [float(y.max()) for y in ys] == [3.0 * (i + 1) for i in range(4)]

print({mode: run(mode) for mode in sys.argv[1:]})
"""


def test_a_sums_side_stream_output_outlives_the_main_streams_read():
    """GLM step 6's illegal address (A's M2, the side-stream audit): `std.ffn.sum`'s later parts run on a side stream
    and their sum goes back to that stream's blocks with no event (`units.Sum.forward`) while the main stream's combine
    may not have read it yet. Neither the next sum's side-stream output, which takes that block at once (`reuse`), nor
    the allocator releasing free blocks (`release`: with expandable segments it unmaps one after waiting for its own
    stream only, as an out-of-memory retry does) may pass that read. A main stream that reads late (a sleep before the
    read) while the host runs ahead: each sum is still x + 2x. Captured in a CUDA graph (`graph`), the side stream's
    trailing wait inside the capture leaves no unjoined work and every replay is still x + 2x. In a process of its
    own, the three in turn: an illegal address ends it (`release` does without the wait: GLM step 6's repro)."""
    import subprocess
    import sys
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    modes = ["reuse", "release", "graph"]
    out = subprocess.run([sys.executable, "-c", _SIDE_STREAM, *modes], env=env, capture_output=True, text=True,
                         timeout=300)
    assert out.stdout.strip() == str(dict.fromkeys(modes, True)), out.stderr[-2000:]
