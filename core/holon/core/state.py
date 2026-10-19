"""Per-request state bookkeeping (design §6.2).

The state manager gives each admitted request a persistent slot — its row in every per-request table — owns which
pool units it holds, asking each bound StateKind how many more a step needs, and keeps one persistent block table
per page group, mirrored on the device. Everything here is generic (unit ids, page sizes, lengths); the units that
own the specs decide what a unit holds.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from holon.api import BatchPlan, InStepPublish, MemOp, Paged, Pages, Payload, Seq, Slot, StateSpec

from .inputs import Uploader
from .memory import OutOfUnits, Pool
from .prefix import PrefixIndex


@dataclasses.dataclass
class BoundSpec:
    spec: StateSpec
    pool: Pool
    kind: Any


class BlockTable:
    """Unit ids per slot for one page group: `[max_seqs, max_pages]` on the host, mirrored to the device. Columns from
    `real_pages` on only ever hold the null unit (positions past the length limit)."""

    def __init__(self, page: int, max_seqs: int, max_pages: int, device: torch.device,
                 real_pages: int | None = None) -> None:
        self.page, self.max_seqs, self.max_pages = page, max_seqs, max_pages
        self.real_pages = max_pages if real_pages is None else real_pages
        self.host = np.zeros((max_seqs, max_pages), dtype=np.int32)
        self.dev = torch.zeros((max_seqs, max_pages), dtype=torch.int32, device=device)
        self._dirty: set[int] = set()

    def write(self, slot: int, start: int, units: np.ndarray) -> None:
        self.host[slot, start:start + len(units)] = units
        self._dirty.add(slot)

    def fields(self, name: str) -> dict[str, tuple[int, torch.dtype]]:
        return {f"{name}.rows": (self.max_seqs, torch.int64),
                f"{name}.data": (self.max_seqs * self.max_pages, torch.int32)}

    def sync(self, up: Uploader, name: str) -> None:
        """Copy rows changed since the last step to the device, asynchronously and in stream order."""
        if self._dirty:
            rows = np.fromiter(sorted(self._dirty), dtype=np.int64)
            data = up.put(f"{name}.data", self.host[rows].ravel()).view(len(rows), self.max_pages)
            index = up.put(f"{name}.rows", rows)
            up.after(lambda: self.dev.index_copy_(0, index, data))          # once the staged rows are on the device
            self._dirty.clear()


class _View:
    """The SeqView a StateKind sees: a read-only window on the request (no copies; this runs per request per step)."""

    __slots__ = ("_seq", "_units", "_slot_units")

    def __init__(self, seq: Seq, units: dict[str, list[int]], slot_units: dict[str, int] | None = None) -> None:
        self._seq, self._units, self._slot_units = seq, units, slot_units or {}

    @property
    def slot(self) -> int:
        return self._seq.slot

    @property
    def num_computed(self) -> int:
        return self._seq.num_computed

    @property
    def tokens(self) -> list[int]:
        return self._seq.tokens

    @property
    def prompt_len(self) -> int:
        return self._seq.prompt_len

    @property
    def branch_at(self) -> int:
        return self._seq.branch_at

    @property
    def matched(self) -> int:
        return self._seq.matched

    def units(self, spec: str) -> list[int]:
        per = self._slot_units.get(spec)
        if per is not None:                       # Slot state: the request's slot owns its units (api.Slot)
            return list(range(self._seq.slot * per, (self._seq.slot + 1) * per))
        return self._units.get(spec, [])


class _Hits:
    """The PrefixView a kind's `hit_length` sees: the payloads of one spec on a request's matched chain."""

    def __init__(self, entries: list, spec: str, block: int) -> None:
        self.entries, self.spec, self.block = entries, spec, block

    def payload(self, block: int) -> Payload | None:
        return self.entries[block].payloads.get(self.spec) if block < len(self.entries) else None

    def blocks(self, upto: int) -> int:
        return min(upto // self.block, len(self.entries))


@dataclasses.dataclass
class StepOps:
    """The state work of one step, decided when it is launched: MemOps to run around it, payloads it publishes (to
    register at its commit) and payload units pinned until the step is enqueued."""

    mem_ops: list[MemOp]
    publishes: list[tuple[Seq, str, int, Payload]]
    pins: list[Payload]
    in_step: list[InStepPublish] = dataclasses.field(default_factory=list)   # payloads the forward fills


class StateManager:
    """Slots, units and block tables of admitted requests, plus the prefix cache (when given an index).

    Paged state is shared by reference: a hit adopts whole cached blocks and publishing a completed block hands its
    pages to the index. State that cannot be shared that way (e.g. recurrent state) joins through its kind: the hit
    is the fixed point over every kind's `hit_length` (design §6.3), a hit copies the kind's payload in (`adopt`
    MemOps before the request's first forward) and the kind publishes payloads at positions it chooses (`publish`
    MemOps after a step that ends there, or filled by the forward of a step that contains it on the spec's `in_step`
    grid), registered at the step's commit once committed tokens cover them.

    With a tier (docs/v1-kv-tiers.md D7) a hit may reach blocks only the tier holds: they get device units (free ones,
    or ones that evicting blocks the tier holds frees), the tier loads them, and the request waits for its loads — on
    the device at its first step's launch, and before that until the tier's clock reaches their due time (`due`, the
    scheduler's gate). A payload in the tier is loaded to a host unit and adopted from there. A block a request
    publishes that only the tier holds takes the request's own units and is loaded into them (promote into place).
    """

    def __init__(self, bound: list[BoundSpec], max_seqs: int, max_tokens: int, device: torch.device,
                 on_admit: Callable[[Seq], None] = lambda seq: None, index: PrefixIndex | None = None,
                 overflow: int = 0) -> None:
        """`overflow`: positions past `max_tokens` a step may touch (a verify row's padding, a drafter's lookahead
        near the limit); every block table gets columns for them that always point at the null unit."""
        self.bound, self.device, self.on_admit, self.index = bound, device, on_admit, index
        self.padding_slot = max_seqs                           # graph padding rows use this table row
        self.held: dict[int, dict[str, list[int]]] = {}       # slot -> spec name -> unit ids
        self.chain: dict[int, list[int] | None] = {}          # slot -> keys of its published blocks (None: broken)
        self.tokens_saved = 0                                  # prefix-hit tokens not recomputed
        self.capacity: dict[int, int] = {}                     # slot -> tokens its units hold (fast-path reserve)
        self._sized = all(hasattr(b.kind, "capacity") for b in bound)
        self._free_slots = list(range(max_seqs - 1, -1, -1))
        self._slot_units = {b.spec.name: 1 + b.spec.shape.draft_slots for b in bound if isinstance(b.spec.shape, Slot)}
        self._cutting = [b.kind for b in bound if hasattr(b.kind, "cut")]
        self._publishing = [b for b in bound if hasattr(b.kind, "next_publish")]
        self._adopting: dict[int, list[tuple[list[MemOp], Payload | None]]] = {}   # slot -> adopts not yet launched
        self.waits: dict[int, list] = {}                       # slot -> the loads its next step waits for
        self.due: dict[int, float] = {}                        # slot -> when it may run (load-gated, not yet run)
        self._tentative: dict[int, dict[str, list]] = {}       # slot -> spec -> its kept tentatives (key, payload)
        self.dropped_publishes = 0                             # publishes that found no payload unit
        self.tables: dict[str, BlockTable] = {}
        self._table_of: dict[str, BlockTable] = {}            # paged spec name -> its group's table
        for b in bound:
            if not isinstance(b.spec.shape, Paged):
                continue
            if b.spec.group in self.tables:
                raise NotImplementedError(f"page group {b.spec.group!r} shared by several specs")
            page = b.spec.shape.tokens_per_unit
            table = BlockTable(page, max_seqs + 1, -(-(max_tokens + overflow) // page), device,
                               real_pages=-(-max_tokens // page))
            self.tables[b.spec.group] = self._table_of[b.spec.name] = table

    def limit(self, tokens: int) -> None:
        """Hold at most `tokens` per request from now on (the engine's clamp): later columns hold only the null unit."""
        for table in self.tables.values():
            table.real_pages = -(-tokens // table.page)

    def admit(self, seq: Seq) -> bool:
        """Give `seq` a slot and adopt its longest cached prefix (always leaving one token to compute); sets
        `seq.num_computed` to the hit length. False if no slot is free."""
        if not self._free_slots:
            return False
        seq.slot = self._free_slots.pop()
        self.held[seq.slot], self.chain[seq.slot] = {}, []
        entries, seq.branch_at, seq.matched, idx = [], 0, 0, self.index
        if idx is not None and seq.params.prompt_logprobs is None:   # those need every position computed
            keys = idx.keys(seq.tokens, 0, (len(seq.tokens) - 1) // idx.block, None)
            entries = idx.match(keys, seq.tokens)
            seq.matched = len(entries) * idx.block
            seq.branch_at = seq.matched if entries and entries[-1].children else 0
            entries = entries[:self._hit_blocks(seq, entries)]
            resident = [e for e in entries if e.key in idx.entries]          # a prefix: taken first (D7)
            for name in resident[0].units if resident else ():          # one incref per spec, not per block
                self._take(seq.slot, name, np.asarray([u for e in resident for u in e.units[name]], np.int32), True)
            if len(resident) < len(entries):
                entries = self._promote(seq, entries, len(resident))
            if idx.tier is not None:          # the tier may have dropped a payload while it loaded: the hit shrinks
                entries = entries[:self._hit_blocks(seq, entries)]
            for name, mine in self.held[seq.slot].items():           # what a trimmed hit took past its end goes back
                self._pool(name).decref(np.asarray(mine[self._span(name, len(entries)).start:], dtype=np.int32))
                del mine[self._span(name, len(entries)).start:]
            if loads := list({id(e.loading): e.loading for e in entries if e.loading is not None}.values()):
                self.waits[seq.slot] = loads      # the loads writing its blocks (another request's, or its own)
            self.chain[seq.slot] = [e.key for e in entries]
            if entries:
                self._adopt(seq, entries[-1])
            if idx.tier is not None:
                idx.tier.hit(list(self.chain[seq.slot]), [(e.key, name) for e in entries[-1:] for name, p in
                                                          e.payloads.items() if not p.units])
            if self.waits.get(seq.slot):
                self.due[seq.slot] = max(load.due for load in self.waits[seq.slot])
        seq.num_computed = len(entries) * (idx.block if idx else 0)
        self._resize(seq)
        try:
            self.on_admit(seq)
        except BaseException:
            self.release(seq)
            seq.num_computed = 0
            raise
        self.tokens_saved += seq.num_computed
        return True

    def _hit_blocks(self, seq: Seq, entries: list) -> int:
        """The cached blocks every kind can resume from: the fixed point of the non-paged kinds' `hit_length` over
        the matched chain (paged state resumes from any block; a kind without `hit_length` from none)."""
        n, block = len(entries), self.index.block
        others = [b for b in self.bound if not isinstance(b.spec.shape, Paged)]
        while n:
            hit = min((b.kind.hit_length(_Hits(entries, b.spec.name, block), n * block).length // block
                       if hasattr(b.kind, "hit_length") else 0 for b in others), default=n)
            if hit >= n:
                return n
            n = hit
        return 0

    def _adopt(self, seq: Seq, entry) -> None:
        """Queue the non-paged kinds' adopt MemOps for a hit ending at `entry`; their payloads stay pinned until the
        ops are launched, so a publish cannot evict and reuse those units in between."""
        view, todo, idx = _View(seq, self.held[seq.slot], self._slot_units), [], self.index
        adopting = [(b, p) for b in self.bound if (p := entry.payloads.get(b.spec.name)) and not isinstance(
            b.spec.shape, Paged)]
        if tier := [(entry.key, b.spec.name) for b, p in adopting if not p.units]:   # one load of the tier's copies
            load = idx.tier.load([], {}, tier, idx.after())
            idx.loads[id(load)] = [load, set(), False]
            self.waits.setdefault(seq.slot, []).append(load)
            idx.poll()                       # what it dropped to make room: never what this load reads
        for b, payload in adopting:
            if payload.units:
                idx.payload_pools[payload.pool].incref(payload.units)
                todo.append((b.kind.adopt(view, payload), payload))
            else:                    # in the tier: from its host copy straight into the request's units (D7)
                todo.append((b.kind.adopt(view, load.payloads[entry.key, b.spec.name]), None))
        if todo:
            self._adopting[seq.slot] = todo

    def _promote(self, seq: Seq, entries: list, resident: int) -> list:
        """Device units for the hit's blocks only the tier holds, from free units and pure evictions — never more: a
        short tail is trimmed and the fixed point runs again —, then the tier loads them (D7); the caller gives back
        what the hit no longer reaches."""
        idx, paged = self.index, [b for b in self.bound if isinstance(b.spec.shape, Paged)]
        per = {b.spec.name: idx.block // self._page(b.spec.name) for b in paged}
        n = len(entries)
        for b in paged:
            idx.evict(b.spec.name, (n - resident) * per[b.spec.name], pure=True)
            n = min(n, resident + b.pool.num_free // per[b.spec.name])
        n = self._hit_blocks(seq, entries[:n]) if n < len(entries) else n
        if n > resident:
            units = {b.spec.name: b.pool.alloc((n - resident) * per[b.spec.name]) for b in paged}
            for name, us in units.items():
                self._take(seq.slot, name, us, shared=False)
            idx.promote(entries[resident:n], {name: us.tolist() for name, us in units.items()}, idx.after())
        return entries[:n]

    def launching(self, seqs: list[Seq]) -> list:
        """At a step's launch, before its device work: wait (on the device) for its requests' loads and for copies on
        units freed since the last launch (D4); the loads it reads, for `launched`."""
        if self.index is None or not (self.waits or self.index.freed):
            return []
        loads = list({id(load): load for seq in seqs for load in self.waits.pop(seq.slot, ())}.values())
        for fence in [*loads, *self.index.freed.values()]:
            fence.wait()
        self.index.freed.clear()
        for seq in seqs:
            self.due.pop(seq.slot, None)
        return loads

    def launched(self, loads: list, done) -> None:
        """After a launch ending at `done`: tell the tier, once per load, of its first reader, and of loads no request
        waits for and no step read (their entries still mark them, so a later hitter waits)."""
        if (idx := self.index) is None:
            return
        idx.last = done
        for load in loads:
            if (track := idx.loads.pop(id(load), None)) is not None:     # else read by an earlier step
                _, keys, told = track
                for key in keys:
                    idx.get(key).loading = None
                if not told:
                    idx.tier.loaded(load, done)
        waited = {id(load) for ws in self.waits.values() for load in ws}
        for i, track in list(idx.loads.items()):
            load, keys, told = track
            if i not in waited and not told:
                idx.tier.loaded(load, load)
                track[2] = True                                      # told: a later reader does not tell again
                if not keys:
                    del idx.loads[i]

    def cut(self, seq: Seq, n: int) -> int:
        """The tokens (1 … n) a prefill row of `seq` may take, so the step ends where a kind publishes (without a
        prefix index nothing is published: no cut)."""
        if self._cutting and self.index is not None:
            view = _View(seq, self.held[seq.slot], self._slot_units)
            for kind in self._cutting:
                n = min(n, max(1, kind.cut(view, seq.num_computed, n)))
        return n

    def step_ops(self, plan: BatchPlan, seqs: list[Seq]) -> StepOps:
        """A step's state work at launch: pending adopts of its requests (before the forward) and the payloads
        kinds publish where a row's step ends (after the sample) or, for a spec with an `in_step` grid, inside a
        non-verify row's step on that grid (the forward fills them: `BatchPlan.publishes`). Host lengths are exact
        for prefill and decode rows here (before the launch advances them); a verify row, or a request with an
        uncommitted one, knows where it ends only on the device, so it publishes nothing."""
        ops, publishes, pins, in_step = [], [], [], []
        if self._adopting:
            for seq in seqs:
                for adopt, payload in self._adopting.pop(seq.slot, ()):
                    ops += adopt
                    pins += [payload] if payload is not None else []
        if self._publishing and self.index is not None:
            rows = zip(seqs, plan.n_query.tolist(), plan.n_logits.tolist(), strict=True)
            for row, (seq, n, n_logits) in enumerate(rows):
                if n_logits > 1 or seq.inexact:     # a verify row, or one whose host start is only an upper bound
                    continue
                view = _View(seq, self.held[seq.slot], self._slot_units)
                start = seq.num_computed
                for b in self._publishing:
                    at = b.kind.next_publish(view)
                    inside = (at is not None and start < at < start + n and b.spec.in_step > 0 and n_logits <= 1
                              and (at - start) % b.spec.in_step == 0)
                    if at is None or (at != start + n and not inside):
                        continue
                    if (pool := getattr(b.kind, "payload_pool", None)) is not None and self.index.tier is None:
                        self.index.evict_payloads(pool, 1)       # with a tier the pool only stages (D7)
                    if (made := b.kind.publish(view, at, in_step=True) if inside else b.kind.publish(view, at)) is None:
                        self.dropped_publishes += 1
                        continue
                    payload, mem_ops = made
                    if inside:
                        self._check_in_step(b, payload, mem_ops)
                        in_step.append(InStepPublish(row, b.spec.name, at - start, payload))
                    ops += mem_ops
                    publishes.append((seq, b.spec.name, at, payload))
        return StepOps(ops, publishes, pins, in_step)

    def _check_in_step(self, b: BoundSpec, payload: Payload, mem_ops: list[MemOp]) -> None:
        """An in-step payload is filled by the spec's forward: no MemOps, and units laid out like the spec's own."""
        if mem_ops:
            raise RuntimeError(f"{b.spec.name}: an in-step publish returned MemOps (the forward fills it)")
        pool = self.index.payload_pools[payload.pool]
        if (pool.unit_bytes, pool.segments) != (b.pool.unit_bytes, b.pool.segments):
            raise RuntimeError(f"{b.spec.name}: payload pool {payload.pool} units ({pool.unit_bytes} B, "
                               f"{pool.segments} segments) differ from the spec's ({b.pool.unit_bytes} B, "
                               f"{b.pool.segments})")

    def unpin(self, pins: list[Payload]) -> None:
        """Drop the adopt pins once the step that copies them is enqueued (stream order protects the copy)."""
        for payload in pins:
            self.index.payload_pools[payload.pool].decref(payload.units)

    def register(self, seq: Seq, spec: str, at: int, payload: Payload, done=None) -> None:
        """At a step's commit: add a payload published at `at` to the block ending there, if committed tokens cover
        `at` and the block is cached; otherwise free it. `done`: the end of the step that wrote it (a tier stores it
        after that). Of a request's tentative payloads only its last `keep_tentative` stay; one registered after its
        request was released is an ordinary payload."""
        block, held = self.index.block, False
        if payload.tentative and seq.slot < 0:
            payload = dataclasses.replace(payload, tentative=False)
        if 0 < at <= len(seq.tokens) and at % block == 0:
            chain, i = self.chain.get(seq.slot), at // block
            if chain is not None and len(chain) >= i:        # the request's own published chain: no re-hash
                entry = self.index.get(chain[i - 1])
            else:
                keys = self.index.keys(seq.tokens, 0, i, None)
                found = self.index.match(keys, seq.tokens)
                entry = found[-1] if len(found) == len(keys) else None
            held = entry is not None and self.index.attach(entry, spec, payload, [done] if done is not None else [])
        if not held:
            self.index.payload_pools[payload.pool].decref(payload.units)
        elif payload.tentative:
            kept = self._tentative.setdefault(seq.slot, {}).setdefault(spec, [])
            kept.append((entry.key, entry.payloads[spec]))
            if len(kept) > next(b.kind.keep_tentative for b in self.bound if b.spec.name == spec):
                self.index.settle(*kept.pop(0), spec, drop=True)

    def reserve(self, seq: Seq, new_tokens: int) -> bool:
        """The units admitted `seq` needs for `new_tokens` more tokens, evicting unused cached blocks if a pool is
        short; all-or-nothing."""
        if seq.num_computed + new_tokens <= self.capacity.get(seq.slot, -1):
            return True                          # its units already hold these tokens
        view = _View(seq, self.held[seq.slot], self._slot_units)
        taken: list[tuple[BoundSpec, np.ndarray]] = []
        try:
            for b in self.bound:
                if n := b.kind.units_needed(view, new_tokens):
                    if b.pool.num_free < n and self.index is not None:
                        self.index.evict(b.spec.name, n)
                    taken.append((b, b.pool.alloc(n)))
        except OutOfUnits:
            for b, units in taken:
                b.pool.decref(units)
            return False
        for b, units in taken:
            self._take(seq.slot, b.spec.name, units, shared=False)
        self._resize(seq, view)
        return True

    def _resize(self, seq: Seq, view: _View | None = None) -> None:
        if self._sized:
            view = view or _View(seq, self.held[seq.slot], self._slot_units)
            self.capacity[seq.slot] = min([b.kind.capacity(view) for b in self.bound], default=1 << 62)

    def _take(self, slot: int, spec: str, units: np.ndarray, shared: bool) -> None:
        if shared:
            self._pool(spec).incref(units)
        mine = self.held[slot].setdefault(spec, [])
        if table := self._table_of.get(spec):
            table.write(slot, len(mine), units)
        mine.extend(units.tolist())

    def publish(self, seq: Seq) -> None:
        """Hand `seq`'s newly completed blocks to the prefix index (after the step that computed them). A block
        another request already cached replaces `seq`'s own copy, so `seq` holds exactly the units of the chain it
        follows (the index's eviction invariant) and duplicate state is freed."""
        chain = self.chain.get(seq.slot)
        if self.index is None or chain is None:
            return
        idx, stored = self.index, []
        for i in range(len(chain), min(seq.num_computed, len(seq.tokens)) // idx.block):
            prev = chain[-1] if chain else None
            tokens = tuple(seq.tokens[i * idx.block:(i + 1) * idx.block])
            key = idx.key(prev, tokens)
            e = idx.get(key)
            mine = {name: tuple(us[self._span(name, i)]) for name, us in self.held[seq.slot].items()}
            if e is None:
                stored.append(idx.insert(key, prev, tokens, mine))
            elif e.parent == prev and e.tokens == tokens:
                if key in idx.lower:           # only the tier holds it: into this request's own units (D7)
                    idx.promote([e], {name: list(us) for name, us in mine.items()}, idx.after())
                for name, units in e.units.items():
                    self._swap(seq.slot, name, self._span(name, i), units)
                if e.loading is not None:
                    self.waits.setdefault(seq.slot, []).append(e.loading)
            else:                                            # a hash collision: stop publishing this chain
                self.chain[seq.slot] = None
                break
            chain.append(key)
        if stored and idx.tier is not None:   # write-through, after the last launched step (it may still write them)
            idx.store(stored)

    def _span(self, spec: str, block_index: int) -> slice:
        per = self.index.block // self._page(spec)
        return slice(block_index * per, (block_index + 1) * per)

    def _swap(self, slot: int, spec: str, span: slice, units: tuple[int, ...]) -> None:
        mine = self.held[slot][spec]
        if tuple(mine[span]) == units:
            return
        new, old = np.asarray(units, dtype=np.int32), np.asarray(mine[span], dtype=np.int32)
        self._pool(spec).incref(new)
        mine[span] = units
        if table := self._table_of.get(spec):
            table.write(slot, span.start, new)
        self._pool(spec).decref(old)

    def could_fit(self, seq: Seq, new_tokens: int, releasing: list[Seq]) -> bool:
        """Whether `seq` could be admitted (if it has no slot) and get units for `new_tokens` more tokens once every
        request in `releasing` is released and unused cached blocks are evicted."""
        if seq.slot < 0 and not self._free_slots and not any(s.slot >= 0 for s in releasing):
            return False
        view = _View(seq, self.held.get(seq.slot, {}), self._slot_units)
        for b in self.bound:
            need = b.kind.units_needed(view, new_tokens)
            if need <= b.pool.num_free:
                continue
            cap, name = b.pool.capacity, b.spec.name
            dropped = np.bincount(np.asarray([u for s in releasing if s.slot >= 0
                                              for u in self.held[s.slot].get(name, ())], dtype=np.int64),
                                  minlength=cap)
            left = b.pool.refcount(np.arange(cap)) - dropped           # references once `releasing` is gone
            cached = self.index.cached(name) if self.index is not None else np.zeros(cap, dtype=bool)
            freed = np.count_nonzero(cached & (left == 1)) + np.count_nonzero(~cached & (dropped > 0) & (left == 0))
            if need > b.pool.num_free + freed:
                return False
        return True

    def release(self, seq: Seq) -> None:
        """Drop everything `seq` holds, including its slot (published blocks stay cached); no-op if it holds nothing."""
        if seq.slot < 0:
            return
        for name, units in self.held.pop(seq.slot).items():
            self._pool(name).decref(np.asarray(units, dtype=np.int32))
        for _, payload in self._adopting.pop(seq.slot, ()):
            if payload is not None:
                self.index.payload_pools[payload.pool].decref(payload.units)
        for d in (self.waits, self.due):
            d.pop(seq.slot, None)
        settled = [(key, spec) for spec, kept in self._tentative.pop(seq.slot, {}).items() for key, p in kept
                   if self.index.settle(key, p, spec, drop=False)]   # its kept tentatives are ordinary from now on
        if settled and self.index.tier is not None:
            self.index.tier.settled(settled)
        del self.chain[seq.slot]
        self.capacity.pop(seq.slot, None)
        self._free_slots.append(seq.slot)
        seq.slot = -1

    def _pool(self, spec: str) -> Pool:
        return next(b.pool for b in self.bound if b.spec.name == spec)

    def _page(self, spec: str) -> int:
        return self._table_of[spec].page

    def reset_padding(self) -> None:
        """Point the padding slot, and every slot's overflow columns, at every pool's null unit (after the pools are
        allocated)."""
        for b in self.bound:
            if table := self._table_of.get(b.spec.name):
                for slot in range(table.max_seqs):
                    start = 0 if slot == self.padding_slot else table.real_pages
                    table.write(slot, start, np.full(table.max_pages - start, b.pool.null, dtype=np.int32))

    def upload_fields(self, max_seqs: int, max_tokens: int) -> dict[str, tuple[int, torch.dtype]]:
        """The per-step device inputs `pages` stages through the engine's uploader."""
        fields = {"pages.lens": (max_seqs, torch.int32)}
        for group, t in self.tables.items():
            fields |= t.fields(f"{group}.table") | {f"{group}.write": (max_tokens, torch.int64)}
        return fields

    def pages(self, plan: BatchPlan, up: Uploader, positions: np.ndarray | None = None) -> dict[str, Pages]:
        """This step's view of every page group. The block tables are synced now; lengths and write indices are
        static buffers the engine's device prep fills from the device SeqTable (see `Executor._prepare`), or, given
        the step's `positions` (exact on the host), write indices are staged from the host tables now."""
        rows, tokens = len(plan.slots), int(plan.n_query.sum())
        slot_of_token = None if positions is None else np.repeat(plan.slots, plan.n_query)
        out = {}
        for group, t in self.tables.items():
            t.sync(up, f"{group}.table")
            if positions is not None:
                unit = t.host[slot_of_token, positions // t.page].astype(np.int64)
                up.put(f"{group}.write", unit * t.page + positions % t.page)
            out[group] = Pages(t.page, t.dev, t.host, up.dev["pages.lens"][:rows], up.dev[f"{group}.write"][:tokens])
        return out
