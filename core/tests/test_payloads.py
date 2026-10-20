"""The core's generic seams for state that is not paged (docs/m03-qwen38.md §4.3), proven with a toy kind before any
model uses them: Slot units addressed by the request slot, prefill cuts, payloads published where a step ends and
registered at its commit only when committed tokens cover them, the fixed-point hit with adopt MemOps pinned until
launched, payload eviction that never costs paged units, and unit copies between pools.

The toy kind keeps one Slot unit per request (think: a running sum of the tokens) and publishes a copy of it every
EVERY tokens into its own plugin pool — where a step ends (a MemOp copy), or, when its spec declares an `in_step` grid,
inside a step: then the forward fills it (`BatchPlan.publishes`; the tests play the toy mixer).

With the toy tier (docs/v1-kv-tiers.md §4: KS1–KS3, KS5) the tests also play the forward (`_launch`: each token's id
into its KV slot, the toy state adding it up) and the tier's copies wait on a deferred stream until something waits for
them, so a missing wait shows as wrong bytes: a unit freed under a store, a staging unit under its payload's store, the
units of an abandoned load; blocks only the tier holds promoted into a hit or into place; payloads adopted from the
tier's host copy; tentative payloads; a running request preempting the gated request below it, never itself."""
import numpy as np
import pytest
import torch
from holon.api import HitLen, MemOp, Paged, Payload, SamplingParams, Seq, Slot, StateSpec, Status
from holon.core.batching import host_plan
from holon.core.memory import Ledger, copy_units, run_mem_ops
from holon.core.prefix import PrefixIndex
from holon.core.sched import Scheduler
from holon.core.state import BoundSpec, StateManager
from holon_judge.deferred import Deferred
from holon_testing.toy_tier.plugin import bind

PAGE, EVERY, SLOTS, MAX_LEN = 4, 8, 3, 64


class _PagedKind:
    def __init__(self, spec):
        self.spec = spec

    def units_needed(self, seq, new_tokens):
        return max(0, -(-(seq.num_computed + new_tokens) // PAGE) - len(seq.units(self.spec.name)))


class _ToyKind:
    """Slot state with payloads every EVERY tokens in `plugin.toy`."""

    payload_pool = "plugin.toy"

    def __init__(self, spec, payloads, every=EVERY, keep=0):
        self.spec, self.payloads, self.every, self.keep_tentative = spec, payloads, every, keep

    def units_needed(self, seq, new_tokens):
        return 0

    def cut(self, seq, start, n):
        if self.spec.in_step:                     # the forward fills payloads inside the step: no cut
            return n
        return min(n, (start // self.every + 1) * self.every - start)

    def next_publish(self, seq):
        return (seq.num_computed // self.every + 1) * self.every

    def publish(self, seq, at, in_step=False):
        if self.payloads.num_free == 0:
            return None
        unit = self.payloads.alloc(1)
        payload = Payload((int(unit[0]),), self.payload_pool, bool(self.keep_tentative) and at > seq.prompt_len)
        if in_step:
            return payload, []
        return payload, [MemOp(f"state.{self.spec.name}", np.array(seq.units(self.spec.name)), self.payload_pool,
                               unit, "post_sample")]

    def hit_length(self, idx, upto):
        for i in reversed(range(idx.blocks(upto))):
            if idx.payload(i) is not None:
                return HitLen((i + 1) * idx.block)
        return HitLen(0)

    def adopt(self, seq, payload):
        return [MemOp(payload.pool, np.array(payload.units), f"state.{self.spec.name}",
                      np.array(seq.units(self.spec.name)), "pre_forward")]


def _state(payload_units: int = 4, in_step: int = 0, payload_bytes: int = 8, kv_units: int = 64, keep: int = 0):
    ledger = Ledger(torch.device("cpu"), host_memory=1 << 20)
    kv = ledger.request("state.kv", unit_bytes=8, min_units=kv_units, tokens_per_unit=PAGE)
    toy = ledger.request("state.toy", unit_bytes=8, min_units=SLOTS + 1, pref_units=SLOTS + 1)
    payloads = ledger.plugin_pool("toy", unit_bytes=payload_bytes, min_units=payload_units, pref_units=payload_units)
    ledger.materialize(None)
    kv_spec = StateSpec("kv", "paged@1", Paged(8, PAGE), (0,), "test")
    toy_spec = StateSpec("toy", "toy@1", Slot(8), (1,), "test", alignment=EVERY, in_step=in_step)
    index = PrefixIndex(1, EVERY, {"kv": kv}, ledger.pools)
    toy_kind = _ToyKind(toy_spec, payloads, keep=keep)
    state = StateManager([BoundSpec(kv_spec, kv, _PagedKind(kv_spec)), BoundSpec(toy_spec, toy, toy_kind)],
                         SLOTS, MAX_LEN, torch.device("cpu"), index=index)
    return state, ledger, payloads


def _seq(tokens):
    return Seq(rid="r", tokens=list(tokens), params=SamplingParams())


def _step(state, seqs_n, sample=True, n_logits=None):
    """Plan, launch and commit one step for [(seq, n)] the way the engine does (no device)."""
    seqs = [s for s, _ in seqs_n]
    n = np.array([k for _, k in seqs_n], dtype=np.int32)
    plan = host_plan(1, np.array([s.slot for s in seqs], np.int32), np.array([s.num_computed for s in seqs], np.int32),
                     n, np.ones_like(n) if n_logits is None else np.array(n_logits, np.int32),
                     np.zeros(int(n.sum()), np.int64))
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
    # Two things refuse it — `register`'s bound and the key of a block past the committed tokens, which cannot
    # match a cached one — so this proves the property, not the bound alone.
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
    assert len(attached) == 2 and len(attached[0].children) == 1                   # the older one is interior
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


def test_an_in_step_payload_is_listed_for_the_forward_and_registered_at_the_commit():
    state, _, payloads = _state(in_step=EVERY)
    seq = _seq(range(17))
    assert state.admit(seq) and state.reserve(seq, 16)
    assert state.cut(seq, 12) == 12                                            # no cut: 8 lies inside the step
    ops = _step(state, [(seq, 12)], sample=False)
    assert not ops.mem_ops and [(p.row, p.spec, p.offset) for p in ops.in_step] == [(0, "toy", EVERY)]
    unit = ops.in_step[0].payload.units
    entry = next(e for e in state.index.entries.values() if e.payloads)        # filed on the block ending at 8
    assert entry.payloads["toy"].units == unit and payloads.refcount(np.array(unit))[0] == 1


def test_in_step_publishes_need_the_grid_a_non_verify_row_and_no_mem_ops():
    state, _, payloads = _state(in_step=EVERY)
    seq = _seq(range(40))
    assert state.admit(seq) and state.reserve(seq, 30)
    seq.num_computed = 3                                                       # 8 − 3 is off the grid: skipped
    assert not _step(state, [(seq, 9)], sample=False).in_step
    seq.num_computed = 16                                                      # a verify row (1 + k logits): skipped
    assert not _step(state, [(seq, 12)], sample=False, n_logits=[3]).in_step
    kind = next(b.kind for b in state.bound if b.spec.name == "toy")
    kind.publish = lambda seq, at, in_step=False: (Payload((0,), "plugin.toy"), [MemOp("a", np.array([0]), "b",
                                                                                        np.array([0]), "post_sample")])
    seq.num_computed = 24                                                      # 32 inside, on the grid
    with pytest.raises(RuntimeError, match="MemOps"):
        _step(state, [(seq, 12)], sample=False)


def test_an_in_step_payload_pool_must_have_the_specs_layout():
    state, _, _ = _state(in_step=EVERY, payload_bytes=16)
    seq = _seq(range(17))
    assert state.admit(seq) and state.reserve(seq, 16)
    with pytest.raises(RuntimeError, match="differ from the spec"):
        _step(state, [(seq, 12)], sample=False)


def test_no_payload_where_a_verify_row_or_an_inexact_start_ends():
    # A verify row's end, and the start of a row after an uncommitted verify, are known only on the device: the host
    # value is an upper bound (drafts may be rejected), so a payload published there could land at the wrong
    # position and a later request adopting it would skip tokens (M0.4 §3.7).
    state, _, _ = _state()
    seq = _seq(range(40))
    assert state.admit(seq) and state.reserve(seq, 30)
    seq.num_computed = 4                                                       # 4 + 4 = 8, where the toy publishes
    assert not _step(state, [(seq, 4)], sample=False, n_logits=[4]).publishes
    seq.num_computed, seq.inexact = 7, 1                                       # 7 + 1 = 8, after a verify in flight
    assert not _step(state, [(seq, 1)], sample=False).publishes
    seq.num_computed, seq.inexact = 7, 0
    assert _step(state, [(seq, 1)], sample=False).publishes


# -- with the toy tier ------------------------------------------------------------------------------------------------
class _Done:
    def wait(self) -> None:
        pass

    synchronize = wait


def _tiered(delay: int = 0, **kw):
    """`_state` with the toy tier, whose copies land only when waited for."""
    state, ledger, payloads = _state(**kw)
    deferred = Deferred(0, most=1 << 30)
    tier = bind(state.index, ledger, {"toy_tier.delay": delay}, deferred.stream, payloads=("plugin.toy",))
    return state, ledger, deferred, tier


def _launch(state, ledger, seqs_n, mark=None, sched=None):
    """One step as the engine runs it, with a toy forward — each row reading the KV byte before its first token (what
    it attends to), each computed token's id into its KV slot (`mark` instead: a recompute that came out different),
    the toy state adding them up —, then its commit; what each row read."""
    seqs = [s for s, _ in seqs_n]
    n = np.array([k for _, k in seqs_n], dtype=np.int32)
    plan = host_plan(1, np.array([s.slot for s in seqs], np.int32), np.array([s.num_computed for s in seqs], np.int32),
                     n, np.ones_like(n), np.zeros(int(n.sum()), np.int64))
    ops, loads = state.step_ops(plan, seqs), state.launching(seqs)
    run_mem_ops(ops.mem_ops, ledger.pools, "pre_forward")
    kv, toy = ledger.pool("state.kv").units(), ledger.pool("state.toy").units()
    read = [int(kv[0, state.held[s.slot]["kv"][(s.num_computed - 1) // PAGE], (s.num_computed - 1) % PAGE])
            if s.num_computed else None for s, _ in seqs_n]
    for s, k in seqs_n:
        for pos in range(s.num_computed, s.num_computed + k):
            kv[0, state.held[s.slot]["kv"][pos // PAGE], pos % PAGE] = s.tokens[pos] if mark is None else mark
            toy[0, s.slot, 0] = s.tokens[pos] + (toy[0, s.slot, 0] if pos else 0)
    run_mem_ops(ops.mem_ops, ledger.pools, "post_sample")
    state.launched(loads, done := _Done())
    state.unpin(ops.pins)
    for s, k in seqs_n:
        s.num_computed += k
        if s.num_computed >= len(s.tokens):
            s.tokens.append(s.num_computed % 7 + 1)
            if sched is not None and s.status is not Status.RUNNING:
                sched.finished_prefill(s)
        state.publish(s)
    for s, spec, at, payload in ops.publishes:
        state.register(s, spec, at, payload, done)
    return read


def _kv(state, ledger, seq, n):
    """The bytes `seq`'s units hold for its first n tokens."""
    kv = ledger.pool("state.kv").units()
    return [int(kv[0, state.held[seq.slot]["kv"][p // PAGE], p % PAGE]) for p in range(n)]


def _admit(state, tokens, n=None):
    seq = _seq(tokens)
    assert state.admit(seq) and state.reserve(seq, (n or len(tokens)) - seq.num_computed)
    return seq


def _run_through(state, ledger, seq, upto, **kw):
    while seq.num_computed < upto:
        _launch(state, ledger, [(seq, min(EVERY - seq.num_computed % EVERY, upto - seq.num_computed))], **kw)


def test_a_unit_freed_under_a_store_is_reused_only_after_it_and_a_hit_reads_the_tier():
    state, ledger, deferred, tier = _tiered(kv_units=8)
    a = _admit(state, range(1, 18))
    _run_through(state, ledger, a, 16)                       # 2 blocks and payloads at 8, 16: stores still queued
    state.release(a)
    b = _admit(state, range(50, 81))                         # all 8 units: a's blocks demoted under their stores
    assert len(state.index.lower) == 2 and state.index.freed
    _run_through(state, ledger, b, 30)
    state.release(b)
    c = _admit(state, range(1, 18))                          # a's prefix again: promoted from the tier, gated
    assert c.num_computed == 16 and state.due[c.slot] == tier.t and c.branch_at == 0      # a's chain ends there
    assert tier.calls[-1] == ("hit", state.chain[c.slot], [(state.chain[c.slot][-1], "toy")])
    assert _launch(state, ledger, [(c, 1)]) == [16]           # it waited for the load
    deferred.drain()
    assert _kv(state, ledger, c, 16) == list(range(1, 17))
    assert int(ledger.pool("state.toy").units()[0, c.slot, 0]) == (sum(range(1, 18)) % 256)   # adopted, then 17


def test_a_staging_unit_is_reused_only_after_its_payload_is_stored():
    state, ledger, deferred, tier = _tiered(payload_units=1)
    a = _admit(state, range(1, 25))
    _run_through(state, ledger, a, 16)                       # the payload at 16 reuses the one staging unit
    state.release(a)
    state.index.evict("kv", ledger.pool("state.kv").num_free + 1)            # block 1 demoted: block 0 has no
    assert len(state.index.lower) == 1                                        # resident child, a cached one still
    b = _admit(state, list(range(1, 9)) + [99] * 9)          # shares block 0 only: adopts the payload at 8
    assert b.num_computed == EVERY and b.branch_at == EVERY  # it leaves a's cached chain there
    _launch(state, ledger, [(b, 1)])
    deferred.drain()
    assert int(ledger.pool("state.toy").units()[0, b.slot, 0]) == (sum(range(1, 9)) + 99) % 256


def test_a_recomputed_block_only_the_tier_holds_is_promoted_into_place():
    state, ledger, deferred, tier = _tiered()
    a = _admit(state, range(1, 18))
    _run_through(state, ledger, a, 16)
    state.release(a)
    state.index.evict("kv", 64)                              # both blocks demoted
    second = state.index.keys(list(range(1, 17)), 0, 2, None)[1]
    tier._evict((second, "toy"))                             # the tier drops the payload at 16 …
    state.index.poll()
    b = _admit(state, range(1, 18))                          # … so the hit stops at 8 and block 1 is recomputed
    assert b.num_computed == EVERY
    _run_through(state, ledger, b, 16, mark=200)             # its own bytes differ; publishing takes the tier's
    assert _launch(state, ledger, [(b, 1)]) == [16]
    deferred.drain()
    assert second in state.index.entries and _kv(state, ledger, b, 16) == list(range(1, 17))
    assert all(c[3] for c in tier.calls if c[0] == "load")  # every load after the last launched step at least


def test_an_evicted_loading_entry_is_abandoned_and_its_units_wait_for_the_load():
    state, ledger, deferred, tier = _tiered(kv_units=8)
    a = _admit(state, range(1, 18))
    _run_through(state, ledger, a, 16)
    state.release(a)
    state.index.evict("kv", 64)
    b = _admit(state, range(1, 18))                          # promoted: its load queued, never read …
    state.release(b)                                         # … b aborted
    c = _admit(state, range(50, 81))                         # evicts the loading entries: the load is abandoned
    assert tier.counts["abandoned"] == 1 and state.index.freed
    _run_through(state, ledger, c, 30)
    deferred.drain()                                         # a late copy would land on c's units now
    assert _kv(state, ledger, c, 30) == list(range(50, 80))


def test_a_running_request_preempts_the_gated_one_below_it_never_itself():
    state, ledger, deferred, tier = _tiered(delay=100, kv_units=12)
    a = _admit(state, range(1, 26))
    _run_through(state, ledger, a, 24)
    state.release(a)
    state.index.evict("kv", 64)
    sched = Scheduler(state, max_step_tokens=8, max_model_len=MAX_LEN, clock=tier.clock)
    r, g = _seq(range(40, 56)), _seq(range(1, 26))
    sched.add(r)
    for _ in range(3):                                       # r prefills, then decodes
        plan, rows = sched.schedule()
        _launch(state, ledger, list(zip(rows, plan.n_query.tolist(), strict=True)), sched=sched)
    sched.add(g)
    plan, rows = sched.schedule()                            # g's hit is promoted, due 100 launches out
    assert rows == [r] and g.num_computed == 24 and state.due[g.slot] > tier.t
    while r.num_computed < 40:                               # r's growth needs g's units: g goes, never r
        _launch(state, ledger, list(zip(rows, plan.n_query.tolist(), strict=True)), sched=sched)
        plan, rows = sched.schedule()
        assert r.status is Status.RUNNING and rows[0] is r
    assert sched.preemptions >= 1 and g.status is Status.WAITING                  # re-admitted after each, gated
    assert tier.counts["abandoned"] >= 1                      # a load whose entries all went unread


def test_a_request_keeps_its_last_tentative_payloads():
    state, ledger, deferred, tier = _tiered(keep=2)
    a = _admit(state, range(1, 6), n=MAX_LEN)
    while a.num_computed < 40:                               # decodes past 8, 16, …, 40: tentatives
        _launch(state, ledger, [(a, 1)])
    keys = state.index.keys(a.tokens, 0, 5, None)
    held = [i for i, k in enumerate(keys) if "toy" in state.index.get(k).payloads]
    assert held == [3, 4]                                    # the payloads at 32 and 40
    assert tier.tentative == {(keys[3], "toy"), (keys[4], "toy")}
    state.release(a)                                         # they stay; one registered from now on is ordinary
    assert not tier.tentative and not state.index.get(keys[4]).payloads["toy"].tentative
    state.register(a, "toy", 24, Payload((int(state.index.payload_pools["plugin.toy"].alloc(1)[0]),), "plugin.toy",
                                         True), _Done())
    assert state.index.get(keys[2]).payloads["toy"].tentative is False
    b = _admit(state, a.tokens[:41])
    assert b.num_computed == 40


def _demoted(state, ledger, tokens, upto):
    """A request computes `tokens` up to `upto`, leaves, and its blocks are demoted (copies landed)."""
    a = _admit(state, tokens)
    _run_through(state, ledger, a, upto)
    state.release(a)
    state.index.evict("kv", 64)
    return a


def _overwrite(state, ledger, n):
    """Another request computes n tokens into the free units and leaves: what they held is gone."""
    x = _admit(state, range(100, 101 + n))
    _run_through(state, ledger, x, n)
    state.release(x)
    state.index.evict("kv", 64)


def test_clear_leaves_requests_their_units_and_loads_with_any_lru_order():
    state, ledger, deferred, tier = _tiered(kv_units=8)
    _demoted(state, ledger, range(1, 18), 16)
    _overwrite(state, ledger, 30)
    b = _admit(state, range(1, 18))                          # promoted, its load queued
    state.index.clear()                                      # under b: the load is not abandoned and still lands
    assert tier.counts["abandoned"] == 0 and _launch(state, ledger, [(b, 1)]) == [16]
    deferred.drain()
    assert _kv(state, ledger, b, 16) == list(range(1, 17)) and not state.index.lower
    state, ledger, _ = _state()                              # no tier: a match moves block 0 behind block 1
    a = _admit(state, range(1, 18))
    _run_through(state, ledger, a, 16)
    state.release(a)
    b = _admit(state, list(range(1, 9)) + [99] * 9)
    assert b.num_computed == EVERY
    state.index.clear()
    assert not state.index.entries and ledger.pool("state.kv").num_free == 64 - len(state.held[b.slot]["kv"])


def test_a_hit_is_trimmed_to_its_units_and_to_payloads_the_tier_still_holds():
    state, ledger, deferred, tier = _tiered(kv_units=12)
    _demoted(state, ledger, range(1, 26), 24)                # payloads at 8, 16, 24
    keys = state.index.keys(list(range(1, 25)), 0, 3, None)
    tier._evict((keys[0], "toy"))
    state.index.poll()                                       # the payload at 8 is gone
    x = _admit(state, range(60, 68))                         # a resident block whose copy the tier then drops:
    _launch(state, ledger, [(x, 8)])                         # promotion may not evict it (no pure free)
    kept = state.chain[x.slot][0]
    tier._evict((kept, None))
    state.index.poll()
    state.release(x)
    _admit(state, range(70, 98), n=28)                       # 7 + 2 of 12 units held: room for 1.5 blocks
    loads, b = len([c for c in tier.calls if c[0] == "load"]), _seq(range(1, 26))
    assert state.admit(b) and b.num_computed == 0            # block 0 alone has no payload: no hit, no load
    assert len([c for c in tier.calls if c[0] == "load"]) == loads and kept in state.index.entries
    state.release(b)
    load = tier.load                                         # the tier drops the payload at 16 while it loads
    tier.load = lambda *a: (tier._evict((keys[1], "toy")), load(*a))[1] if a[0] else load(*a)
    state.index.evict("kv", 64)
    c = _seq(range(1, 18))
    assert state.admit(c) and c.num_computed == 0            # the hit falls back past the dropped payload


def test_a_promotion_into_units_freed_under_a_store_waits_for_it():
    state, ledger, deferred, tier = _tiered(kv_units=8)
    _demoted(state, ledger, range(50, 67), 16)               # b's blocks, landed in the tier
    deferred.drain()
    a = _admit(state, range(1, 18))
    _run_through(state, ledger, a, 16)                       # a's stores still queued …
    state.release(a)
    state.index.evict("kv", 64)                              # … when its units are freed
    c = _admit(state, range(50, 67))                         # and b's blocks are promoted into them
    _launch(state, ledger, [(c, 1)])
    state.release(c)
    deferred.drain()
    state.index.evict("kv", 64)
    d = _admit(state, range(1, 18))                          # a's blocks, as stored
    _launch(state, ledger, [(d, 1)])
    deferred.drain()
    assert _kv(state, ledger, d, 16) == list(range(1, 17))


def test_a_joining_request_waits_for_the_load_and_an_eviction_store_after_it():
    state, ledger, deferred, tier = _tiered(kv_units=8)
    _demoted(state, ledger, range(1, 18), 16)
    _overwrite(state, ledger, 30)
    b = _admit(state, range(1, 18))                          # promoted: its load queued
    c = _admit(state, range(1, 18))                          # hits the loading blocks: joins b's load
    assert _launch(state, ledger, [(c, 1)]) == [16]          # it waited for it
    state.release(b)
    state.release(c)
    e = _demoted(state, ledger, range(1, 18), 16)            # again: promoted, then left unread
    deferred.drain()
    _overwrite(state, ledger, 14)
    b = _admit(state, range(1, 18))
    state.release(b)
    _launch(state, ledger, [(_admit(state, [7] * 5), 1)])    # a launch: the tier hears b's load has no reader
    unread = [keys for _, keys, _ in state.index.loads.values() if keys]
    assert len(unread) == 1, "b's page load is read by a step that never waited for it"
    for key in unread[0]:
        tier._evict((key, None))                             # and drops their copies: they stay resident, unheld
    state.index.poll()
    state.index.evict("kv", 64)                              # eviction stores, after the load into those units
    deferred.drain()
    f = _admit(state, e.tokens[:17])
    _launch(state, ledger, [(f, 1)])
    deferred.drain()
    assert _kv(state, ledger, f, 16) == list(range(1, 17))


def test_when_only_gated_requests_are_left_the_earliest_due_runs():
    state, ledger, deferred, tier = _tiered(delay=10, kv_units=24)
    _demoted(state, ledger, range(1, 18), 16)
    _demoted(state, ledger, range(31, 48), 16)
    sched = Scheduler(state, max_step_tokens=64, max_model_len=MAX_LEN, clock=tier.clock)
    r = Seq("r", list(range(60, 70)), SamplingParams(max_tokens=3))
    first, second = _seq(range(1, 18)), _seq(range(31, 48))
    for seq in (r, first):
        sched.add(seq)
    while r.status is not Status.FINISHED:
        plan, rows = sched.schedule()
        assert first not in rows                             # gated while r runs
        tier.before_launch(plan)                             # the clock: one per launch
        _launch(state, ledger, list(zip(rows, plan.n_query.tolist(), strict=True)), sched=sched)
        if second.slot < 0 and second not in sched.waiting:
            sched.add(second)                                # admitted a launch later: due later
        if len(r.tokens) - r.prompt_len >= 3:
            sched.finish(r, "length")
    assert state.due[first.slot] < state.due[second.slot] and tier.t < state.due[first.slot]
    assert sched.schedule()[1] == [first]
