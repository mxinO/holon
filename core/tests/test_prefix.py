"""PrefixIndex: chain keys, verified matches, leaf-first LRU eviction that never touches units still in use, and an
eviction policy choosing among the eligible entries (docs/v1-kv-tiers.md KS5)."""
import numpy as np
import pytest
import torch
from holon.core.memory import Ledger
from holon.core.prefix import PrefixIndex


def _index(units: int = 16, block: int = 4):
    ledger = Ledger(torch.device("cpu"))
    pool = ledger.request("kv", unit_bytes=8, min_units=units, tokens_per_unit=block)
    ledger.materialize(None)
    return PrefixIndex(7, block, {"kv": pool}), pool


def _publish(index, pool, tokens):
    """Publish every full block of `tokens` the way a request would (its own units, then released)."""
    keys = index.keys(tokens, 0, len(tokens) // index.block, None)
    units = pool.alloc(len(keys))
    for i, key in enumerate(keys):
        index.insert(key, keys[i - 1] if i else None, tuple(tokens[i * 4:(i + 1) * 4]), {"kv": (int(units[i]),)})
    pool.decref(units)                     # the request finishes; only the index holds the blocks
    return keys


def test_keys_identify_the_whole_prefix():
    index, _ = _index()
    a = index.keys(list(range(12)), 0, 3, None)
    assert index.keys(list(range(12)), 1, 3, a[0]) == a[1:]                  # resumable chain
    assert index.keys([9] + list(range(1, 12)), 0, 3, None)[2] != a[2]       # an early change changes later keys


def test_match_returns_the_longest_cached_prefix_and_verifies_tokens():
    index, pool = _index()
    keys = _publish(index, pool, list(range(12)))
    assert len(index.match(keys, list(range(12)))) == 3
    assert len(index.match(index.keys(list(range(8)) + [0, 0, 0, 0], 0, 3, None), list(range(8)) + [0] * 4)) == 2
    index.entries[keys[1]].tokens = (99, 99, 99, 99)                         # a forged collision is a miss
    assert len(index.match(keys, list(range(12)))) == 1


def test_eviction_takes_unused_leaves_least_recently_used_first():
    index, pool = _index(units=6)
    a = _publish(index, pool, list(range(8)))                                  # chain a0 -> a1
    b = _publish(index, pool, [50, 51, 52, 53, 60, 61, 62, 63])               # chain b0 -> b1
    index.match(a, list(range(8)))                                            # a is now more recent than b
    in_use = np.asarray(index.entries[b[1]].units["kv"], dtype=np.int32)
    pool.incref(in_use)                                                       # a running request adopted b1
    index.evict("kv", pool.num_free + 2)
    assert set(index.entries) == {b[0], b[1]}                                 # a went (leaf a1, then a0); b in use
    index.evict("kv", pool.capacity)
    assert set(index.entries) == {b[0], b[1]}                                 # nothing in use is ever dropped
    pool.decref(in_use)
    index.clear()
    assert pool.num_free == pool.capacity and not index.entries


def test_a_collision_with_another_prefix_is_a_miss():
    index, pool = _index()
    p_keys = _publish(index, pool, list(range(8)))                 # P: blocks (0,1,2,3) -> (4,5,6,7)
    other = [9, 9, 9, 9, 4, 5, 6, 7]                                 # Q: same second block, different first
    q_keys = _publish(index, pool, other[:4]) + [p_keys[1]]          # Q's block 0 cached; its block-1 key collides
    assert len(index.match(q_keys, other)) == 1                      # P's block 1 has the right tokens, wrong parent


def test_blocks_of_several_pages_publish_and_evict_cleanly():
    ledger = Ledger(torch.device("cpu"))
    pool = ledger.request("kv", unit_bytes=8, min_units=8, tokens_per_unit=4)
    ledger.materialize(None)
    index = PrefixIndex(7, 8, {"kv": pool})                               # a block spans two pages
    tokens = list(range(16))
    keys = index.keys(tokens, 0, 2, None)
    units = pool.alloc(4)
    for i, key in enumerate(keys):
        index.insert(key, keys[i - 1] if i else None, tuple(tokens[i * 8:(i + 1) * 8]),
                     {"kv": tuple(int(u) for u in units[2 * i:2 * i + 2])})
    assert index.cached("kv")[units].all() and (pool.refcount(units) == 2).all()
    pool.decref(units)                                                    # the request finishes
    assert len(index.match(keys, tokens)) == 2
    index.evict("kv", pool.capacity)
    assert not index.entries and not index.cached("kv").any() and pool.num_free == pool.capacity


def test_eviction_stops_climbing_a_chain_once_enough_is_free():
    index, pool = _index(units=6)
    a = _publish(index, pool, list(range(12)))                             # chain a0 -> a1 -> a2, 3 free units
    index.evict("kv", pool.num_free + 1)
    assert set(index.entries) == {a[0], a[1]}                              # only the leaf went


class _KeepRoots:
    """A toy eviction policy: entries of depth ≤ 1 stay while deeper ones can go."""

    def victim(self, candidates):
        shallow = None
        for e in candidates:
            if e.depth > 1:
                return e
            shallow = shallow or e
        return shallow


def test_an_eviction_policy_keeps_the_roots_under_pressure():
    index, pool = _index(units=12)
    a = _publish(index, pool, list(range(16)))                              # depths 0 … 3
    b = _publish(index, pool, [50 + t for t in range(12)])                  # depths 0 … 2
    index.match(a, list(range(16)))                                         # b is the least recently used chain
    index.policy = _KeepRoots()
    index.evict("kv", pool.num_free + 3)
    assert set(index.entries) == {a[0], a[1], b[0], b[1]}                   # the deep ones went
    index.evict("kv", pool.capacity)
    assert not index.entries                                                # then the roots, when nothing else is left
    a = _publish(index, pool, list(range(8)))
    index.policy = type("Lazy", (), {"victim": lambda self, candidates: None})()
    index.evict("kv", pool.num_free + 1)                                    # none: the least recently used instead
    assert set(index.entries) == {a[0]}
    asked = []                                                              # a second ask would loop forever

    def wrong(candidates):
        assert not asked, "asked again: the core took a non-candidate"
        asked.append(1)
        return index.entries[a[0]]
    index.policy = type("Wrong", (), {"victim": lambda self, candidates: wrong(candidates)})()
    pool.incref(np.asarray(index.entries[a[0]].units["kv"]))                # held: not eligible
    with pytest.raises(RuntimeError, match="not one of the eligible"):
        index.evict("kv", pool.capacity)
