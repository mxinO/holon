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
    def __init__(self, bound: list[BoundSpec], max_seqs: int, max_tokens: int, device: torch.device,
                 on_admit: Callable[[Seq], None] = lambda seq: None) -> None:
        self.bound, self.device, self.on_admit = bound, device, on_admit
        self.held: dict[int, dict[str, list[int]]] = {}       # slot -> spec name -> unit ids (exclusive)
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

    def reserve(self, seq: Seq, new_tokens: int) -> bool:
        """Give `seq` a slot (if it has none) and the units it needs for `new_tokens` more tokens; all-or-nothing."""
        fresh = seq.slot < 0
        if fresh:
            if not self._free_slots:
                return False
            seq.slot = self._free_slots.pop()
            self.held[seq.slot] = {}
        held = self.held[seq.slot]
        view = _View(seq, held)
        taken: list[tuple[BoundSpec, np.ndarray]] = []
        try:
            for b in self.bound:
                if n := b.kind.units_needed(view, new_tokens):
                    taken.append((b, b.pool.alloc(n)))
            if fresh:
                self.on_admit(seq)
        except BaseException as e:
            for b, units in taken:
                b.pool.decref(units)
            if fresh:
                self._free_slot(seq)
            if isinstance(e, OutOfUnits):
                return False
            raise
        for b, units in taken:
            mine = held.setdefault(b.spec.name, [])
            if table := self._table_of.get(b.spec.name):
                table.write(seq.slot, len(mine), units)
            mine.extend(units.tolist())
        return True

    def could_fit(self, seq: Seq, new_tokens: int, releasing: list[Seq]) -> bool:
        """Whether `reserve(seq, new_tokens)` would succeed once every request in `releasing` is released.
        Assumes requests own their units exclusively (true until prefix sharing, E4)."""
        if seq.slot < 0 and not self._free_slots and not any(s.slot >= 0 for s in releasing):
            return False
        view = _View(seq, self.held.get(seq.slot, {}))
        for b in self.bound:
            freed = sum(len(self.held[s.slot].get(b.spec.name, ())) for s in releasing if s.slot >= 0)
            if b.kind.units_needed(view, new_tokens) > b.pool.num_free + freed:
                return False
        return True

    def release(self, seq: Seq) -> None:
        """Free everything `seq` holds, including its slot; no-op if it holds nothing."""
        if seq.slot < 0:
            return
        pools = {b.spec.name: b.pool for b in self.bound}
        for name, units in self.held[seq.slot].items():
            pools[name].decref(np.asarray(units, dtype=np.int32))
        self._free_slot(seq)

    def _free_slot(self, seq: Seq) -> None:
        del self.held[seq.slot]
        self._free_slots.append(seq.slot)
        seq.slot = -1

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
