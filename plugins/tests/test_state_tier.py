"""std.state_tier's host level (docs/v1-kv-tiers.md §5) on a bare tier over CPU pools: leaf-first page eviction, the
admission options, payload stores that always succeed, nothing being loaded evicted before `loaded`, the H2D lane and
the clock, the payload share, the host units' fences and per-key records, and a load ordered after the store that wrote
its host units (on deferred copy streams). Tier invariance on a hybrid model through this host level is the judge's
(`judge/tests/test_judge_tiers.py`, `holon_judge tiers`)."""
import types

import pytest
import torch
from holon.api import BlockRecord, Payload
from holon.core import platform
from holon.core.memory import Ledger
from holon_judge.deferred import Deferred
from holon_std.state_tier.tier import HostTier


class _Done:
    def wait(self) -> None:
        pass

    synchronize = wait

    def done(self) -> bool:
        return True


def _bare(admit: str = "publish", pages: int = 6, payloads: int = 3, stall: int = 0, wide: bool = False,
          stream=None):
    """A host level over one 2-segment page pool and one payload pool (`wide`: and one of twice its unit), room for
    `pages` blocks and `payloads` units of the first payload pool's size; its copy streams from `stream`."""
    ledger = Ledger(torch.device("cpu"), host_memory=1 << 20)
    kv = ledger.request("state.kv", unit_bytes=64, segments=2, min_units=16, tokens_per_unit=16)
    ckpt = ledger.plugin_pool("ckpt", unit_bytes=32, min_units=4, pref_units=4)
    pools = {"kv": kv, "plugin.ckpt": ckpt} | ({"plugin.ckpt2": ledger.plugin_pool(
        "ckpt2", unit_bytes=64, min_units=2, pref_units=2)} if wide else {})
    ledger.materialize(None)
    for p in pools.values():
        p.buffer.copy_(torch.arange(p.buffer.numel(), dtype=torch.uint8))
    total = pages * 64 + payloads * 32
    tier = HostTier(types.SimpleNamespace(options={"page_size": 16, "plugins": {
        "state_tier.host.bytes": str(total), "state_tier.host.pages": str(pages * 64 / total),
        "state_tier.host.admit": admit, "state_tier.step_cost": "1,0,0", "state_tier.stall_bytes": str(stall)}},
        candidates=lambda op, facts: []))
    tier.attach(types.SimpleNamespace(
        pools=pools, block=16, max_seqs=1, max_model_len=64,
        stream=stream or (lambda: platform.Stream(torch.device("cpu"))),
        host_pool=lambda name, *, unit_bytes, segments, units, mapped=False: ledger.plugin_pool(
            name, unit_bytes=unit_bytes, segments=segments, min_units=units, pref_units=units, device="host",
            mapped=mapped, late=True)))
    return tier, kv, ckpt


def _chain(tier, keys, evicting=False):
    accepted = set()
    for i, key in enumerate(keys):
        accepted |= tier.store([BlockRecord(key, keys[i - 1] if i else None, (key,), {"kv": (i % 16,)}, {})], [],
                               evicting)[0]
    return accepted


def test_the_host_level_evicts_leaves_first_and_never_what_it_is_loading():
    tier, kv, _ = _bare(pages=4)
    _chain(tier, [1, 2, 3])                                   # a chain 1 -> 2 -> 3
    _chain(tier, [10, 11])                                    # and 10 -> 11: 5 blocks, room for 4
    assert tier.dropped() == [(3, None)]                      # 3 is the least recent leaf; 1 and 2 are interior
    load = tier.load([10], {"kv": [5]}, [], [])
    _chain(tier, [20, 21])                                    # 10 is being loaded: the leaves 2, then 1, go
    assert tier.dropped() == [(2, None), (1, None)] and (10, None) in tier.copies
    assert torch.equal(kv.units()[:, 5], kv.units()[:, 0])   # 10's pages: what unit 0 held when it was stored
    tier.loaded(load, _Done())
    _chain(tier, [30])                                        # loaded: 10 may go once a leaf, after its child 11
    assert tier.dropped() == [(11, None)]
    assert load.due == tier.t == pytest.approx(64 / 43e9)     # charged on the H2D lane; `loaded` brought the clock


def test_a_parent_stored_after_its_child_is_not_a_leaf():
    tier, _, _ = _bare("evict", pages=4)                     # write-back stores a chain's leaves first
    tier.store([BlockRecord(2, 1, (2,), {"kv": (1,)}, {})], [], True)
    tier.store([BlockRecord(1, None, (1,), {"kv": (0,)}, {})], [], True)
    tier.hit([1, 2], [])                                      # 1 is the least recent copy, but not a leaf
    _chain(tier, [5, 6, 7], evicting=True)
    assert tier.dropped() == [(2, None)]


@pytest.mark.parametrize("admit, stored", [("publish", {1, 2}), ("hit:2", {2}), ("evict", set())])
def test_page_admission_options_and_mandatory_payloads(admit, stored):
    tier, _, ckpt = _bare(admit)
    tier.hit([2], [])
    tier.hit([2], [])
    got = {k for k, s in _chain(tier, [1, 2]) if s is None}
    assert got == stored
    assert _chain(tier, [7], evicting=True) == ({(7, None)} if admit != "hit:2" else set())
    for key in range(4):                                       # 3 payload units: the fourth evicts the oldest
        tier.store([BlockRecord(key, None, (key,), {}, {"c": Payload((key % 2,), "plugin.ckpt", key == 3)})], [],
                   False)
    assert ((0, "c") not in tier.copies and (0, "c") in tier.dropped() and tier.tentative == {(3, "c")})
    tier.settled([(3, "c")])
    assert not tier.tentative


def test_attach_refuses_a_host_level_below_the_loads_in_flight():
    with pytest.raises(RuntimeError, match="too small for the loads in flight"):
        _bare(pages=3)                                        # one max_model_len load is 4 blocks


def test_page_loads_take_the_lane_and_payload_loads_do_not():
    tier, _, _ = _bare(stall=64, payloads=4)          # 4 payload units: the share is exact
    _chain(tier, [1, 2])
    tier.store([BlockRecord(1, None, (1,), {}, {"c": Payload((0, 1, 2), "plugin.ckpt")})], [], False)
    r = 43e9
    assert tier.load([1], {"kv": [5]}, [], []).due == 0 and tier.lane == pytest.approx(64 / r)   # small: due now,
    assert tier.load([], {}, [(1, "c")], []).due == 0 and tier.lane == pytest.approx(64 / r)     # on the lane
    assert tier.load([1, 2], {"kv": [6, 7]}, [], []).due == pytest.approx(192 / r)             # queued behind it


def test_the_payload_share_gives_each_payload_pool_as_many_units():
    tier, _, _ = _bare(payloads=6, wide=True)
    assert tier.host["plugin.ckpt"].capacity == tier.host["plugin.ckpt2"].capacity == 2


def test_host_units_keep_one_fence_per_source_and_keys_leave_no_records():
    tier, _, _ = _bare("hit:1")
    tier.hit([1, 2, 3], [])
    _chain(tier, [1, 2])
    tier.store([BlockRecord(2, 1, (2,), {}, {"c": Payload((0,), "plugin.ckpt")})], [], False)
    for _ in range(5):
        tier.loaded(tier.load([1, 2], {"kv": [5, 6]}, [(2, "c")], []), _Done())
        tier.loaded(load := tier.load([2], {"kv": [7]}, [], []), load)      # no step waited for it
    assert max(len(f) for f in tier.fences.values()) == 3 and not tier.loading and not tier.reading
    tier.drop([1, 2, 3])
    assert not (tier.copies or tier.specs or tier.kids or tier.parent or tier.hits)
    tier.hit([4], [])
    _chain(tier, [4, 5])
    tier.clear()                                                        # flush_prefix
    assert not (tier.copies or tier.specs or tier.kids or tier.parent or tier.hits)


def test_a_load_follows_the_store_of_its_host_units_and_counts_late():
    """On deferred copy streams (work runs only when waited for): a load issued right after the store that wrote its
    host units, with nothing to wait for (`after` empty), still reads the stored bytes; a step that reads it before
    its copy ran counts a late load."""
    deferred = Deferred(0, most=2)
    tier, kv, _ = _bare(stream=deferred.stream)
    _chain(tier, [1])
    load = tier.load([1], {"kv": [5]}, [], [])
    tier.loaded(load, _Done())
    load.synchronize()
    assert torch.equal(kv.units()[:, 5], kv.units()[:, 0]) and tier.stats()["late_loads"] == 1
