"""The core's generic seams for state that is not paged (docs/m03-qwen38.md §4.3), proven with a toy kind before any
model uses them: Slot units addressed by the request slot, prefill cuts, payloads published where a step ends and
registered at its commit only when committed tokens cover them, the fixed-point hit with adopt MemOps pinned until
launched, payload eviction that never costs paged units, and unit copies between pools.

The toy kind keeps one Slot unit per request (think: a running sum of the tokens) and publishes a copy of it every
EVERY tokens into its own plugin pool."""
import numpy as np
import pytest
import torch
from holon.api import HitLen, MemOp, Paged, Payload, SamplingParams, Seq, Slot, StateSpec
from holon.core.batching import host_plan
from holon.core.memory import Ledger, copy_units, run_mem_ops
from holon.core.prefix import PrefixIndex
from holon.core.sched import Scheduler
from holon.core.state import BoundSpec, StateManager

PAGE, EVERY, SLOTS, MAX_LEN = 4, 8, 3, 64


class _PagedKind:
    def __init__(self, spec):
        self.spec = spec

    def units_needed(self, seq, new_tokens):
        return max(0, -(-(seq.num_computed + new_tokens) // PAGE) - len(seq.units(self.spec.name)))


class _ToyKind:
    """Slot state with payloads every EVERY tokens in `plugin.toy`."""

    payload_pool = "plugin.toy"

    def __init__(self, spec, payloads, every=EVERY):
        self.spec, self.payloads, self.every = spec, payloads, every

    def units_needed(self, seq, new_tokens):
        return 0

    def cut(self, seq, start, n):
        return min(n, (start // self.every + 1) * self.every - start)

    def next_publish(self, seq):
        return (seq.num_computed // self.every + 1) * self.every

    def publish(self, seq, at):
        if self.payloads.num_free == 0:
            return None
        unit = self.payloads.alloc(1)
        payload = Payload((int(unit[0]),), self.payload_pool)
        return payload, [MemOp(f"state.{self.spec.name}", np.array(seq.units(self.spec.name)), self.payload_pool,
                               unit, "post_sample")]

    def hit_length(self, idx, upto):
        for i in reversed(range(idx.blocks(upto))):
            if idx.payload(i) is not None:
                return HitLen((i + 1) * idx.block)
        return HitLen(0)

    def adopt(self, seq, payload):
        return [MemOp(self.payload_pool, np.array(payload.units), f"state.{self.spec.name}",
                      np.array(seq.units(self.spec.name)), "pre_forward")]


def _state(payload_units: int = 4):
    ledger = Ledger(torch.device("cpu"))
    kv = ledger.request("state.kv", unit_bytes=8, min_units=64, tokens_per_unit=PAGE)
    toy = ledger.request("state.toy", unit_bytes=8, min_units=SLOTS + 1, pref_units=SLOTS + 1)
    payloads = ledger.plugin_pool("toy", unit_bytes=8, min_units=payload_units, pref_units=payload_units)
    ledger.materialize(None)
    kv_spec = StateSpec("kv", "paged@1", Paged(8, PAGE), (0,), "test")
    toy_spec = StateSpec("toy", "toy@1", Slot(8), (1,), "test", alignment=EVERY)
    index = PrefixIndex(1, EVERY, {"kv": kv}, ledger.pools)
    toy_kind = _ToyKind(toy_spec, payloads)
    state = StateManager([BoundSpec(kv_spec, kv, _PagedKind(kv_spec)), BoundSpec(toy_spec, toy, toy_kind)],
                         SLOTS, MAX_LEN, torch.device("cpu"), index=index)
    return state, ledger, payloads


def _seq(tokens):
    return Seq(rid="r", tokens=list(tokens), params=SamplingParams())


def _step(state, seqs_n, sample=True):
    """Plan, launch and commit one step for [(seq, n)] the way the engine does (no device)."""
    seqs = [s for s, _ in seqs_n]
    n = np.array([k for _, k in seqs_n], dtype=np.int32)
    plan = host_plan(1, np.array([s.slot for s in seqs], np.int32), np.array([s.num_computed for s in seqs], np.int32),
                     n, np.ones_like(n), np.zeros(int(n.sum()), np.int64))
    ops = state.step_ops(plan, seqs)
    state.unpin(ops.pins)
    for s, k in seqs_n:
        s.num_computed += k
        if sample and s.num_computed >= len(s.tokens):
            s.tokens.append(7)
        state.publish(s)
    for s, spec, at, payload in ops.publishes:
        state.register(s, spec, at, payload)
    return ops


def test_slot_units_are_the_request_slot_and_cost_nothing():
    state, ledger, _ = _state()
    seq = _seq(range(5))
    assert state.admit(seq) and state.reserve(seq, 5)
    from holon.core.state import _View
    view = _View(seq, state.held[seq.slot], state._slot_units)
    assert view.units("toy") == [seq.slot] and view.prompt_len == 5
    assert ledger.pool("state.toy").capacity == SLOTS + 1                    # the padding slot owns a unit too


def test_prefill_is_cut_where_the_kind_publishes():
    state, _, _ = _state()
    sched = Scheduler(state, max_step_tokens=100, max_model_len=MAX_LEN)
    seq = _seq(range(20))
    sched.add(seq)
    plan, rows = sched.schedule()
    assert int(plan.n_query[0]) == EVERY                                       # 20 tokens: stops at 8
    seq.num_computed = 13
    assert state.cut(seq, 7) == 3 and state.cut(seq, 1) == 1                   # to 16; always ≥ 1


def test_a_payload_published_at_a_step_end_is_registered_at_its_commit():
    state, _, payloads = _state()
    seq = _seq(range(17))
    assert state.admit(seq) and state.reserve(seq, 16)
    ops = _step(state, [(seq, EVERY)], sample=False)
    assert [(op.phase, op.src_pool, op.dst_pool) for op in ops.mem_ops] == [("post_sample", "state.toy", "plugin.toy")]
    entry = next(iter(state.index.entries.values()))
    assert set(entry.payloads) == {"toy"} and payloads.refcount(np.array(entry.payloads["toy"].units))[0] == 1
    ops = _step(state, [(seq, 5)], sample=False)                               # ends at 13: nothing published
    assert not ops.mem_ops and not ops.publishes


def test_a_payload_whose_tokens_are_not_committed_is_freed():
    # The plant of §4.7 at the core: registration with a commit that does not cover the position must not attach.
    state, _, payloads = _state()
    seq = _seq(range(17))
    assert state.admit(seq) and state.reserve(seq, 16)
    unit = payloads.alloc(1)
    payload = Payload((int(unit[0]),), "plugin.toy")
    state.register(seq, "toy", 24, payload)                                    # 24 > the 17 committed tokens
    assert payloads.num_free == payloads.capacity and not any(e.payloads for e in state.index.entries.values())


def test_a_hit_is_the_fixed_point_and_adopts_the_payload():
    state, _, payloads = _state()
    first = _seq(range(17))
    assert state.admit(first) and state.reserve(first, 16)
    _step(state, [(first, EVERY)], sample=False)                               # payload at 8
    _step(state, [(first, 6)], sample=False)                                   # to 14: no new block or payload
    state.release(first)
    second = _seq(range(17))
    assert state.admit(second)
    assert second.num_computed == EVERY                                        # one cached block, its payload at 8
    adopt = state._adopting[second.slot]
    unit = adopt[0][1].units
    assert payloads.refcount(np.array(unit))[0] == 2                           # pinned by the adopter
    state.index.evict_payloads("plugin.toy", payloads.capacity)               # a full payload pool
    assert payloads.refcount(np.array(unit))[0] >= 1                           # the pin keeps it
    ops = _step(state, [(second, 8)], sample=False)
    pre = [op for op in ops.mem_ops if op.phase == "pre_forward"]
    assert [(op.src_pool, op.dst_pool) for op in pre] == [("plugin.toy", "state.toy")]
    assert not state._adopting


def test_without_the_pin_a_publish_can_reuse_the_adopted_unit():
    # The pinning plant: drop the pin and the adopted unit can be evicted and handed to another publish.
    state, _, payloads = _state(payload_units=1)
    first = _seq(range(17))
    assert state.admit(first) and state.reserve(first, 16)
    _step(state, [(first, EVERY)], sample=False)
    state.release(first)
    second = _seq(range(17))
    assert state.admit(second)
    (_, payload), = state._adopting[second.slot]
    payloads.decref(np.array(payload.units))                                   # the plant: no pin
    state.index.evict_payloads("plugin.toy", 1)
    assert payloads.num_free == 1                                              # the unit to be adopted is free


def test_payload_eviction_takes_interior_payloads_and_never_paged_units():
    state, ledger, payloads = _state(payload_units=2)
    seq = _seq(range(33))
    assert state.admit(seq) and state.reserve(seq, 32)
    for _ in range(4):
        _step(state, [(seq, EVERY)], sample=False)                             # payloads at 8, 16 (pool of 2)
    kv = ledger.pool("state.kv")
    held = kv.num_free
    attached = [e for e in state.index.entries.values() if e.payloads]
    assert len(attached) == 2 and attached[0].children == 1                    # the older one is interior
    state.release(seq)
    state.index.evict_payloads("plugin.toy", 2)
    assert payloads.num_free == 2 and kv.num_free >= held                      # payloads gone, pages untouched
    assert len(state.index.entries) == 4                                       # the chain itself stays cached


def test_unit_copies_between_pools():
    ledger = Ledger(torch.device("cpu"))
    a = ledger.request("a", unit_bytes=8, segments=2, min_units=3, pref_units=3)
    b = ledger.request("b", unit_bytes=8, segments=2, min_units=2, pref_units=2)
    c = ledger.request("c", unit_bytes=4, min_units=1, pref_units=1)
    ledger.materialize(None)
    a.units().copy_(torch.arange(a.buffer.numel(), dtype=torch.uint8).view(2, 3, 4))
    run_mem_ops([MemOp("a", np.array([2, 0]), "b", np.array([0, 1]))], ledger.pools, "pre_forward")
    assert torch.equal(b.units()[:, 0], a.units()[:, 2]) and torch.equal(b.units()[:, 1], a.units()[:, 0])
    run_mem_ops([MemOp("a", np.array([1]), "b", np.array([0]), "post_sample")], ledger.pools, "pre_forward")
    assert torch.equal(b.units()[:, 0], a.units()[:, 2])                       # other phase: not run
    with pytest.raises(ValueError, match="unit shapes differ"):
        copy_units(a, [0], c, [0])
