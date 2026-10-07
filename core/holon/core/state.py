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

from holon.api import BatchPlan, Paged, Pages, Seq, StateSpec

from .memory import OutOfUnits, Pool
from .prefix import PrefixIndex


@dataclasses.dataclass
class BoundSpec:
    spec: StateSpec
    pool: Pool
    kind: Any


class BlockTable:
    """Unit ids per slot for one page group: `[max_seqs, max_pages]` on the host, mirrored to the device."""

    def __init__(self, page: int, max_seqs: int, max_pages: int, device: torch.device) -> None:
        self.page = page
        self.host = np.zeros((max_seqs, max_pages), dtype=np.int32)
        self.dev = torch.zeros((max_seqs, max_pages), dtype=torch.int32, device=device)
        self._dirty: set[int] = set()

    def write(self, slot: int, start: int, units: np.ndarray) -> None:
        self.host[slot, start:start + len(units)] = units
        self._dirty.add(slot)

    def sync(self) -> None:
        """Copy rows changed since the last step to the device."""
        if self._dirty:
            rows = np.fromiter(sorted(self._dirty), dtype=np.int64)
            self.dev[torch.as_tensor(rows, device=self.dev.device)] = torch.as_tensor(self.host[rows],
                                                                                      device=self.dev.device)
            self._dirty.clear()


class _View:
    """The SeqView a StateKind sees."""

    def __init__(self, seq: Seq, units: dict[str, list[int]]) -> None:
        self.slot, self.num_computed, self.tokens = seq.slot, seq.num_computed, tuple(seq.tokens)
        self._units = units

    def units(self, spec: str) -> tuple[int, ...]:
        return tuple(self._units.get(spec, ()))


class StateManager:
    """Slots, units and block tables of admitted requests, plus the prefix cache (when given an index).

    v0 prefix caching covers paged state only: pages are shared by reference, so a hit adopts whole cached blocks
    and publishing a completed block hands its pages to the index. Checkpointed state (recurrent mixers) joins
    through StateKind publish/adopt MemOps in M0.3.
    """

    def __init__(self, bound: list[BoundSpec], max_seqs: int, max_tokens: int, device: torch.device,
                 on_admit: Callable[[Seq], None] = lambda seq: None, index: PrefixIndex | None = None) -> None:
        self.bound, self.device, self.on_admit, self.index = bound, device, on_admit, index
        self.held: dict[int, dict[str, list[int]]] = {}       # slot -> spec name -> unit ids
        self.chain: dict[int, list[int] | None] = {}          # slot -> keys of its published blocks (None: broken)
        self.tokens_saved = 0                                  # prefix-hit tokens not recomputed
        self._free_slots = list(range(max_seqs - 1, -1, -1))
        self.tables: dict[str, BlockTable] = {}
        self._table_of: dict[str, BlockTable] = {}            # paged spec name -> its group's table
        for b in bound:
            if not isinstance(b.spec.shape, Paged):
                continue
            if b.spec.group in self.tables:
                raise NotImplementedError(f"page group {b.spec.group!r} shared by several specs")
            page = b.spec.shape.tokens_per_unit
            table = BlockTable(page, max_seqs, -(-max_tokens // page), device)
            self.tables[b.spec.group] = self._table_of[b.spec.name] = table

    def admit(self, seq: Seq) -> bool:
        """Give `seq` a slot and adopt its longest cached prefix (always leaving one token to compute); sets
        `seq.num_computed` to the hit length. False if no slot is free."""
        if not self._free_slots:
            return False
        seq.slot = self._free_slots.pop()
        self.held[seq.slot], self.chain[seq.slot] = {}, []
        entries = []
        if self.index is not None:
            keys = self.index.keys(seq.tokens, 0, (len(seq.tokens) - 1) // self.index.block, None)
            entries = self.index.match(keys, seq.tokens)
            for e in entries:
                for name, units in e.units.items():
                    self._take(seq.slot, name, np.asarray(units, dtype=np.int32), shared=True)
            self.chain[seq.slot] = [e.key for e in entries]
        seq.num_computed = len(entries) * (self.index.block if self.index else 0)
        try:
            self.on_admit(seq)
        except BaseException:
            self.release(seq)
            seq.num_computed = 0
            raise
        self.tokens_saved += seq.num_computed
        return True

    def reserve(self, seq: Seq, new_tokens: int) -> bool:
        """The units admitted `seq` needs for `new_tokens` more tokens, evicting unused cached blocks if a pool is
        short; all-or-nothing."""
        view = _View(seq, self.held[seq.slot])
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
        return True

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
        block = self.index.block
        for i in range(len(chain), min(seq.num_computed, len(seq.tokens)) // block):
            prev = chain[-1] if chain else None
            key = self.index.keys(seq.tokens, i, i + 1, prev)[0]
            tokens = tuple(seq.tokens[i * block:(i + 1) * block])
            e = self.index.entries.get(key)
            if e is None:
                self.index.insert(key, prev, tokens, {name: tuple(us[self._span(name, i)])
                                                      for name, us in self.held[seq.slot].items()})
            elif e.parent == prev and e.tokens == tokens:
                for name, units in e.units.items():
                    self._swap(seq.slot, name, self._span(name, i), units)
            else:                                            # a hash collision: stop publishing this chain
                self.chain[seq.slot] = None
                return
            chain.append(key)

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
        view = _View(seq, self.held.get(seq.slot, {}))
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
        del self.chain[seq.slot]
        self._free_slots.append(seq.slot)
        seq.slot = -1

    def _pool(self, spec: str) -> Pool:
        return next(b.pool for b in self.bound if b.spec.name == spec)

    def _page(self, spec: str) -> int:
        return self._table_of[spec].page

    def pages(self, plan: BatchPlan, positions: np.ndarray) -> dict[str, Pages]:
        """This step's view of every page group; `positions` are the step's token positions, flat."""
        rows = np.repeat(plan.slots, plan.n_query)
        lens = torch.as_tensor(plan.start_upper + plan.n_query, device=self.device)
        out = {}
        for group, t in self.tables.items():
            t.sync()
            write = t.host[rows, positions // t.page].astype(np.int64) * t.page + positions % t.page
            out[group] = Pages(t.page, t.dev, t.host, lens, torch.as_tensor(write, device=self.device))
        return out
