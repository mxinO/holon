"""std.state_tier's level below host (docs/v1-kv-tiers.md D11): the SSD level alone (slot files, flock, bytes, the
manifest's snapshot, restore with crc, geometry and identity, and the plants on it: a torn manifest, a stale slot),
then the tier over it on CPU pools (write-back at host eviction and the watermark, read-back bitwise, what is never
written back, the level's own eviction order, read priority and dependent writes, abandonment before the enqueue, one
read per copy, the lanes, and a restart)."""
import threading
import time
import types

import numpy as np
import pytest
import torch
from holon.api import BlockRecord, Payload
from holon.core import platform
from holon.core.memory import Ledger
from holon_std.state_tier.ssd import MANIFEST, SsdLevel
from holon_std.state_tier.tier import HostTier

SHAPES = {"kv": (8192, 1, 0.5), "ckpt": (3 * 1000, 3, 0.5)}      # an aligned page unit, a payload unaligned


def _level(path, persist="close", nbytes=1 << 20, identity="id"):
    level = SsdLevel({"dir": str(path), "bytes": nbytes, "persist": persist}, threads=2)
    return level, level.bind(None, SHAPES, identity)


def _unit(seed: int, shape: str) -> list[memoryview]:
    unit_bytes, segments, _ = SHAPES[shape]
    data = np.random.default_rng(seed).integers(0, 256, unit_bytes, dtype=np.uint8)
    return [memoryview(s) for s in np.split(data, segments)]


def _blank(shape: str) -> list[memoryview]:
    unit_bytes, segments, _ = SHAPES[shape]
    return [memoryview(s) for s in np.split(np.zeros(unit_bytes, np.uint8), segments)]


def _record(key, parent, payload=False):
    return BlockRecord(key, parent, (key,), {} if payload else {"kv": ()},
                       {"c": Payload((0,), "ckpt")} if payload else {})


def test_the_ssd_level_moves_units_and_owns_its_directory(tmp_path):
    level, cap = _level(tmp_path, persist="off")
    assert cap == {"kv": 64, "ckpt": 128}                             # 512 KiB per shape, 8 KiB / 4 KiB slots
    with pytest.raises(RuntimeError, match="another engine holds it"):
        _level(tmp_path)
    with pytest.raises(RuntimeError, match="more bytes"):
        _level(tmp_path / "big", nbytes=1 << 62)
    src = {"kv": [_unit(1, "kv"), _unit(2, "kv")], "ckpt": [_unit(3, "ckpt")]}
    level.write(7, None, src, {"kv": [5, 63], "ckpt": [127]})            # the tier's slots
    dst = {"kv": [_blank("kv"), _blank("kv")], "ckpt": [_blank("ckpt")]}
    level.read(7, None, dst)
    assert all(bytes(a) == bytes(b) for name in src for u, v in zip(src[name], dst[name], strict=True)
               for a, b in zip(u, v, strict=True))
    level.drop(7, None)
    assert not level.held
    level.close([])


def test_a_write_into_a_full_level_returns_at_once(tmp_path):
    """Every slot held, and a copy dropped whose last job is still queued: the tier gives the next write that copy's
    slot (and orders it after that job). The level writes it at once; it never waits for room (an urgent write that
    did, on a thread, behind a queued write no thread would run, hung the tier)."""
    level, cap = _level(tmp_path, persist="off")
    for key in range(cap["kv"]):
        level.write(key, None, {"kv": [_unit(key, "kv")]}, {"kv": [key]})
    writer = threading.Thread(target=lambda: level.write(99, None, {"kv": [_unit(99, "kv")]}, {"kv": [0]}), daemon=True)
    writer.start()
    writer.join(10)
    assert not writer.is_alive()
    level.drop(0, None)
    dst = {"kv": [_blank("kv")]}
    level.read(99, None, dst)
    assert bytes(dst["kv"][0][0]) == bytes(_unit(99, "kv")[0])
    level.close([])


def test_a_restart_restores_what_close_wrote_and_nothing_it_cannot_trust(tmp_path):
    level, _ = _level(tmp_path)
    for key in (1, 2, 3, 4):                                         # key k in page slot k - 1
        level.write(key, None, {"kv": [_unit(key, "kv")]}, {"kv": [key - 1]})
    level.write(2, "c", {"ckpt": [_unit(9, "ckpt")]}, {"ckpt": [0]})
    slot3 = 2
    level.close([_record(1, None), BlockRecord(2, 1, (2,), {"kv": ()}, {"c": Payload((0,), "ckpt")}), _record(3, 2),
                 _record(4, 1)])
    assert (tmp_path / MANIFEST).exists()
    with open(tmp_path / "kv.slots", "r+b") as f:                    # a stale slot: 3's bytes rewritten
        f.seek(slot3 * 8192)
        f.write(b"\xff" * 8192)
    again, _ = _level(tmp_path)
    assert not (tmp_path / MANIFEST).exists()                        # unlinked before any slot can be rewritten
    got = again.restore()
    assert [(r.key, sorted(r.payloads)) for r in got] == [(1, []), (2, ["c"]), (4, [])]   # 3 fails its crc
    assert got[1].units == {"kv": (1,)} and got[1].payloads["c"] == Payload((0,), "ckpt")  # their slots
    dst = {"kv": [_blank("kv")]}
    again.read(4, None, dst)
    assert bytes(dst["kv"][0][0]) == bytes(_unit(4, "kv")[0])
    again.close([])
    for identity, persist in (("other", "close"), ("id", "off")):    # another engine's state, or no persistence
        first, _ = _level(tmp_path)
        first.write(1, None, {"kv": [_unit(1, "kv")]}, {"kv": [0]})
        first.close([_record(1, None)])
        third, _ = _level(tmp_path, persist=persist, identity=identity)
        assert third.restore() == [] and not (tmp_path / MANIFEST).exists()
        third.close([])


def test_restore_discards_slots_past_the_geometry_and_orphans(tmp_path):
    level, _ = _level(tmp_path)
    for key in range(1, 40):                                          # key k in slot k - 1: 39 in slot 38
        level.write(key, None, {"kv": [_unit(key, "kv")]}, {"kv": [key - 1]})
    level.close([_record(1, None), _record(39, 1), _record(5, 39)])
    small, cap = _level(tmp_path, nbytes=1 << 19)                     # 32 page slots: 39's (slot 38) is past them
    assert cap["kv"] == 32
    assert [r.key for r in small.restore()] == [1]                   # 5's parent is gone with 39
    small.close([])


def test_a_torn_manifest_restores_nothing_wrong(tmp_path):
    """The `torn_manifest` plant: the manifest is in place before the data it names was written (the write killed)."""
    level, _ = _level(tmp_path)
    level.write(1, None, {"kv": [_unit(1, "kv")]}, {"kv": [0]})
    real = level._io
    level._io = lambda *a, **k: None                                  # the data write never lands
    level.write(2, None, {"kv": [_unit(2, "kv")]}, {"kv": [1]})
    level._io = real
    level.close([_record(1, None), _record(2, 1)])
    again, _ = _level(tmp_path)
    assert [r.key for r in again.restore()] == [1]
    again.close([])


# -- the tier over a level ------------------------------------------------------------------------------------------
class _Done:
    def wait(self) -> None:
        pass

    synchronize = wait


class _Recording:
    """A level in memory recording its calls; a read of `hold` waits for `go`."""

    persist = "off"

    def __init__(self, hold=None) -> None:
        self.data, self.calls, self.hold, self.go = {}, [], hold, threading.Event()

    def bind(self, ctx, shapes, identity):
        return {name: 64 for name in shapes}

    def write(self, key, spec, src, slots):
        self.calls.append(("write", key, spec))
        self.data[key, spec] = {name: [[bytes(mv) for mv in unit] for unit in units] for name, units in src.items()}

    def read(self, key, spec, dst):
        if (key, spec) == self.hold:
            self.go.wait()
        self.calls.append(("read", key, spec))
        for name, units in dst.items():
            for unit, saved in zip(units, self.data[key, spec][name], strict=True):
                for mv, b in zip(unit, saved, strict=True):
                    mv[:] = b

    def drop(self, key, spec):
        self.data.pop((key, spec), None)

    def rates(self):
        return {"read": 1e6, "write": 1e6, "mixed": 0.5, "lane": None}

    def restore(self):
        return []

    def close(self, records):
        pass


def _tier(tmp_path, level=None, pages=4, payloads=3, extra=None):
    """A tier over one 2-segment page pool and one payload pool (room for `pages` blocks and `payloads` units on
    host), with the SSD level in `tmp_path` or `level`."""
    ledger = Ledger(torch.device("cpu"), host_memory=1 << 20)
    kv = ledger.request("state.kv", unit_bytes=64, segments=2, min_units=16, tokens_per_unit=16)
    ckpt = ledger.plugin_pool("ckpt", unit_bytes=32, min_units=4, pref_units=4)
    ledger.materialize(None)
    for p in (kv, ckpt):
        p.buffer.copy_(torch.arange(p.buffer.numel(), dtype=torch.uint8))
    total = pages * 64 + payloads * 32
    opts = {"state_tier.host.bytes": str(total), "state_tier.host.pages": str(pages * 64 / total),
            "state_tier.step_cost": "1,0,0", "state_tier.stall_bytes": "0", "state_tier.host.reserve": "0",
            "state_tier.levels": "ssd", "state_tier.ssd.dir": str(tmp_path), "state_tier.ssd.bytes": "64KiB",
            "state_tier.ssd.persist": "close", "state_tier.io_threads": "1"} | (extra or {})
    ctx = types.SimpleNamespace(options={"page_size": 16, "plugins": opts}, candidates=lambda op, facts: [])
    tier = HostTier(ctx)
    if level is not None:
        tier.level = level
    tier.attach_records = tier.attach(types.SimpleNamespace(
        pools={"kv": kv, "plugin.ckpt": ckpt}, block=16, max_seqs=1, max_model_len=64, identity="id",
        stream=lambda: platform.Stream(torch.device("cpu")),
        host_pool=lambda name, *, unit_bytes, segments, units, mapped=False: ledger.plugin_pool(
            name, unit_bytes=unit_bytes, segments=segments, min_units=units, pref_units=units, device="host",
            mapped=mapped, late=True)))
    return tier, kv, ckpt


def _pages(tier, keys, parents=None):
    for i, key in enumerate(keys):
        parent = parents[i] if parents else (keys[i - 1] if i else None)
        tier.store([BlockRecord(key, parent, (key,), {"kv": (i % 16,)}, {})], [], False)


def _payload(tier, key, unit, tentative=False):
    tier.store([BlockRecord(key, None, (key,), {}, {"c": Payload((unit,), "plugin.ckpt", tentative)})], [], False)


def test_host_evictions_go_to_the_level_and_come_back_bitwise(tmp_path):
    tier, kv, ckpt = _tier(tmp_path)
    _pages(tier, [1, 2, 3, 4, 5, 6])                                  # a chain: its host leaves 4, then 3, go down
    assert tier.dropped() == [] and tier.counts["write_backs"] == 2 and set(tier.low) == {(3, None), (4, None)}
    for key, unit in ((1, 0), (20, 1), (21, 2), (22, 3)):             # 1's payload goes to the level too
        _payload(tier, key, unit)
    wlane = tier.wlane
    assert wlane > tier.t                                               # background writes on their lane
    load = tier.load([3], {"kv": [10]}, [(1, "c")], [])
    cost = tier.counts["level_read_bytes"] / (2.17e9 * 0.80)          # D4: the read-side factor beside them
    assert tier.wlane == pytest.approx(wlane + cost)                  # and they wait for the reads
    load.synchronize()
    tier.loaded(load, _Done())
    assert torch.equal(kv.units()[:, 10], kv.units()[:, 2])          # 3's pages, from the level, bitwise
    (unit,) = load.payloads[1, "c"].units
    assert torch.equal(tier.host["plugin.ckpt"].units()[:, unit], ckpt.units()[:, 0])
    reads = tier.counts["level_reads"]
    joiner = tier.load([], {}, [(1, "c")], [])                        # held by host since: no second read
    assert tier.counts["level_reads"] == reads and joiner.due == tier.t
    assert load.due == pytest.approx(tier.lane) and tier.lane > tier.nvme > 0     # the level's lane, then H2D


def test_what_the_level_never_takes_and_what_it_gives_up_first(tmp_path):
    tier, _, _ = _tier(tmp_path, pages=4, payloads=3)
    _pages(tier, [1, 2, 3, 4])                                        # 1 -> 2 -> 3 -> 4
    _payload(tier, 1, 0)                                              # superseded by 3's (single-child links)
    _payload(tier, 3, 1)
    _payload(tier, 4, 2, tentative=True)                              # tentative: never written back
    _payload(tier, 2, 3)                                              # evicts 1's (superseded): dropped
    _payload(tier, 5, 0)                                              # evicts 3's: written back
    _payload(tier, 6, 1)                                              # evicts 4's (tentative): dropped
    assert tier.dropped() == [(1, "c"), (4, "c")] and (3, "c") in tier.low
    tier.store([BlockRecord(7, 1, (7,), {"kv": (9,)}, {})], [], False)   # 1 now branches
    assert tier._keeps((1, "c")) and not tier._keeps((2, "c"))        # a branch point keeps its checkpoint


def test_reads_go_before_queued_writes_and_after_the_writes_they_need(tmp_path):
    """One I/O thread, held in a read. The watermark then queues two background write-backs; a read decided after them
    reuses the host unit the first frees: that write goes ahead of it, the other one after it."""
    level = _Recording()
    tier, kv, _ = _tier(tmp_path, level=level, pages=4, extra={"state_tier.host.reserve": "2"})
    _pages(tier, [1, 2, 3, 4, 5, 6])                                  # 3 and 4 go down
    tier.io.drain()
    level.hold = (3, None)
    held = tier.load([3], {"kv": [10]}, [], [])                       # its units: 2 written back first
    tier.before_launch(types.SimpleNamespace(n_query=np.ones(1, np.int32), slots=[0]))   # 1, then 6, queued
    late = tier.load([4], {"kv": [11]}, [], [])                       # reuses 1's host unit
    level.go.set()
    for load in (held, late):
        load.synchronize()
    tier.io.drain()
    order = level.calls.index
    assert order(("write", 2, None)) < order(("read", 3, None))       # the write freeing its unit, first
    assert order(("read", 3, None)) < order(("write", 1, None)) < order(("read", 4, None)) < order(("write", 6, None))
    assert torch.equal(kv.units()[:, 10], kv.units()[:, 2]) and torch.equal(kv.units()[:, 11], kv.units()[:, 3])


def test_a_read_reusing_units_a_queued_write_frees_takes_that_write_ahead(tmp_path):
    """One I/O thread, held in a read; the watermark queues a background write-back, and a read decided next reuses
    the host unit it frees: the write must go ahead of that read, or the read waits forever behind it."""
    level = _Recording()
    tier, kv, _ = _tier(tmp_path, level=level, pages=4, extra={"state_tier.host.reserve": "1"})
    _pages(tier, [1, 2, 3, 4, 5, 6])
    tier.io.drain()
    first, second = sorted(k for k, _ in tier.low)
    level.hold = (first, None)
    held = tier.load([first], {"kv": [10]}, [], [])
    tier.before_launch(types.SimpleNamespace(n_query=np.ones(1, np.int32), slots=[0]))   # a background write-back
    late = tier.load([second], {"kv": [11]}, [], [])
    level.go.set()
    waiter = threading.Thread(target=lambda: [x.synchronize() for x in (held, late)], daemon=True)
    waiter.start()
    waiter.join(10)
    assert not waiter.is_alive(), "the read waits for a write queued behind it"
    assert torch.equal(kv.units()[:, 11], kv.units()[:, second - 1])


def test_a_second_load_of_a_copy_being_read_back_waits_on_an_io_thread(tmp_path):
    """A load of a copy whose read-back (an earlier load's) is still running takes the level path: its H2D is enqueued
    after that read, due no earlier than it; the engine thread never waits for the read."""
    level = _Recording()
    tier, kv, _ = _tier(tmp_path, level=level)
    _pages(tier, [1, 2, 3, 4, 5])
    tier.io.drain()
    (key, _), = tier.low
    level.hold = (key, None)
    first = tier.load([key], {"kv": [10]}, [], [])
    got = []
    thread = threading.Thread(target=lambda: got.append(tier.load([key], {"kv": [11]}, [], [])), daemon=True)
    thread.start()
    thread.join(5)
    assert got and got[0].due >= first.due and got[0].level           # decided without waiting for the read
    level.go.set()
    for load in (first, got[0]):
        load.synchronize()
    assert torch.equal(kv.units()[:, 11], kv.units()[:, key - 1])


def test_an_abandoned_load_is_never_enqueued_and_never_waited_for(tmp_path):
    level = _Recording()
    tier, _, _ = _tier(tmp_path, level=level)
    _pages(tier, [1, 2, 3, 4, 5])
    tier.io.drain()
    (key, _), = tier.low
    level.hold = (key, None)
    load = tier.load([key], {"kv": [10]}, [], [])
    tier.loaded(load, None)                                           # abandoned while its read is held
    start = time.perf_counter()
    load.wait()                                                       # a launch waiting on it returns at once
    assert time.perf_counter() - start < 1 and load.copy is None
    level.go.set()
    tier.io.drain()
    assert load.copy is None and ("read", key, None) in level.calls    # the read ran, its H2D never did


def test_a_restarted_tier_restores_the_level_and_loads_it_bitwise(tmp_path):
    tier, kv, _ = _tier(tmp_path)
    _pages(tier, [1, 2, 3])
    _payload(tier, 2, 0)
    tier.close()                                                      # persist = close: everything goes down
    again, kv2, _ = _tier(tmp_path)
    assert {(r.key, tuple(r.payloads)) for r in again.attach_records} == {(1, ()), (2, ("c",)), (3, ())}
    load = again.load([2], {"kv": [10]}, [], [])
    load.synchronize()
    assert torch.equal(kv2.units()[:, 10], kv.units()[:, 1])
    again.close()


def test_the_watermark_keeps_host_units_free(tmp_path):
    tier, _, _ = _tier(tmp_path, extra={"state_tier.host.reserve": "2"})
    _pages(tier, [1, 2, 3, 4])
    tier.before_launch(types.SimpleNamespace(n_query=np.ones(1, np.int32), slots=[0]))
    assert len(tier.free["kv"]) == 2 and tier.counts["write_backs"] == 2
    tier.close()


def test_branch_checkpoints_come_back_from_the_level_bitwise(tmp_path):
    """The tiny hybrid with `branch` checkpoints and the payload host level at its floor (6 units, 5 required): eight
    prefixes each branched by a second request, then each resumed in turn. Most branch checkpoints are on the level by
    then, so the resumes adopt checkpoints read back from it, and give a roomy host level's outputs and hits."""
    from holon.api import SamplingParams
    from holon.core.engine import Engine, EngineConfig
    from holon_testing.qwen3_5_files import save, tiny_hf

    path = save(tiny_hf(4, seed=1), tmp_path)
    rng = np.random.default_rng(0)
    prefixes = [rng.integers(0, 320, 128).tolist() for _ in range(8)]
    tails = [[rng.integers(0, 320, 80).tolist() for _ in range(3)] for _ in prefixes]   # past a block: a branch
    params = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)

    def run(level: bool):
        opts = {"recurrent.policy": "branch", "recurrent.checkpoints": "6", "state_tier.step_cost": "1,0,0"} | (
            {"state_tier.host.bytes": "2177928", "state_tier.host.pages": str(2105344 / 2177928),
             "state_tier.levels": "ssd", "state_tier.ssd.dir": str(tmp_path / "level"),
             "state_tier.ssd.bytes": "64MiB"} if level else {"state_tier.host.bytes": "64MiB"})
        engine = Engine(EngineConfig(model=str(path), device="cpu", dtype=torch.float32, page_size=16, max_seqs=4,
                                     max_step_tokens=64, max_model_len=512, state_memory=920 << 10,
                                     host_memory=96 << 20, slots={"tier": "std.state_tier"}, plugin_options=opts))
        out = []
        for order in ((0, 1), (2,)):                   # each prefix branched, then every prefix resumed
            for p, tail in zip(prefixes, tails, strict=True):
                for i in order:
                    rid, got = engine.add(p + tail[i], params), [None, None]
                    while engine.has_work():
                        for d in engine.step():
                            if d.rid == rid:
                                got[0] = (got[0] or []) + d.token_ids
                                got[1] = d.cached_tokens if d.cached_tokens is not None else got[1]
                    out.append(tuple(map(lambda x: tuple(x) if isinstance(x, list) else x, got)))
        stats = engine.stats()
        engine.close()
        return out, stats
    want, _ = run(level=False)
    got, stats = run(level=True)
    assert got == want and sum(c for _, c in got[16:]) >= 8 * 128      # every resume adopts its branch checkpoint
    assert stats["std.state_tier.level_payload_reads"] > 0
