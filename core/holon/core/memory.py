"""Memory ledger and pools (design §6.1).

Pools are requested while the model is built and allocated by `materialize` once weights are loaded and the
activation peak has been profiled. Fixed pools (`pref_units`, e.g. per-request slots) get their preferred size, or
their minimum when memory is short. Fill pools (`tokens_per_unit`, e.g. paged KV) share the remainder so that every
fill pool holds the same number of tokens.

A pool's buffer is `[segments, capacity, unit_bytes / segments]`: a unit is one chunk in each segment (e.g. one
KV page in every layer, laid out layer-major). The core can therefore copy units generically without knowing what
they hold.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import torch


class OutOfUnits(RuntimeError):
    pass


class Pool:
    def __init__(self, name: str, *, unit_bytes: int, segments: int, min_units: int, pref_units: int | None,
                 tokens_per_unit: int | None, priority: int, device: torch.device) -> None:
        if unit_bytes % segments:
            raise ValueError(f"pool {name!r}: unit_bytes {unit_bytes} not divisible by {segments} segments")
        if (pref_units is None) == (tokens_per_unit is None):
            raise ValueError(f"pool {name!r}: give exactly one of pref_units (fixed) or tokens_per_unit (fill)")
        self.name, self.unit_bytes, self.segments, self.device = name, unit_bytes, segments, device
        self.min_units, self.pref_units, self.tokens_per_unit = min_units, pref_units, tokens_per_unit
        self.priority = priority
        self._set(0)

    def _set(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer = (torch.empty(capacity * self.unit_bytes, dtype=torch.uint8, device=self.device)
                       if capacity else None)
        self.capacity_dev = torch.tensor(capacity, dtype=torch.int32, device=self.device)
        self._free = list(range(capacity - 1, -1, -1))  # pop() hands out low ids first
        self._refs = np.zeros(capacity, dtype=np.int32)

    @property
    def base(self) -> int:
        if self.buffer is None:
            raise RuntimeError(f"pool {self.name!r} is not materialized")
        return self.buffer.data_ptr()

    @property
    def num_free(self) -> int:
        return len(self._free)

    def view(self, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
        """The whole buffer as `dtype` with `shape`, which starts with (segments, capacity) for segmented pools."""
        return self.buffer.view(dtype).view(shape)

    def alloc(self, n: int) -> np.ndarray:
        if n > len(self._free):
            raise OutOfUnits(f"pool {self.name!r}: need {n} units, {len(self._free)} free")
        units = np.array([self._free.pop() for _ in range(n)], dtype=np.int32)
        self._refs[units] = 1
        return units

    def incref(self, units: np.ndarray) -> None:
        np.add.at(self._refs, units, 1)

    def decref(self, units: np.ndarray) -> None:
        np.subtract.at(self._refs, units, 1)
        if (self._refs[units] < 0).any():
            raise RuntimeError(f"pool {self.name!r}: unit released twice")
        self._free.extend(int(u) for u in np.unique(units) if self._refs[u] == 0)

    def copy(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        """Copy units `src` → `dst` (device index tensors) in every segment, in stream order."""
        units = self.buffer.view(self.segments, self.capacity, self.unit_bytes // self.segments)
        units[:, dst] = units[:, src]


@dataclasses.dataclass
class Ledger:
    device: torch.device
    pools: dict[str, Pool] = dataclasses.field(default_factory=dict)

    def request(self, name: str, *, unit_bytes: int, min_units: int, pref_units: int | None = None,
                tokens_per_unit: int | None = None, segments: int = 1, priority: int = 0, resize: str = "fixed",
                device: torch.device | None = None) -> Pool:
        if name in self.pools:
            raise ValueError(f"pool {name!r} requested twice")
        if resize != "fixed":
            raise NotImplementedError(f"resize mode {resize!r} (v1)")
        pool = Pool(name, unit_bytes=unit_bytes, segments=segments, min_units=min_units, pref_units=pref_units,
                    tokens_per_unit=tokens_per_unit, priority=priority, device=device or self.device)
        self.pools[name] = pool
        return pool

    def pool(self, name: str) -> Pool:
        return self.pools[name]

    def materialize(self, budget_bytes: int | None) -> None:
        """(Re)allocate every pool within `budget_bytes`, or at minimum sizes if None; frees previous buffers."""
        self.free()
        if budget_bytes is None:
            sizes = {p.name: p.min_units for p in self.pools.values()}
        else:
            sizes = self._sizes(budget_bytes)
        short = [f"{p.name} ({sizes[p.name]} < {p.min_units} units)" for p in self.pools.values()
                 if sizes[p.name] < p.min_units]
        if short:
            raise OutOfUnits(f"memory budget {budget_bytes / 2**30:.3f} GiB too small: {', '.join(short)}")
        for p in self.pools.values():
            p._set(sizes[p.name])

    def _sizes(self, budget: int) -> dict[str, int]:
        fixed = [p for p in self.pools.values() if p.pref_units is not None]
        fill = [p for p in self.pools.values() if p.tokens_per_unit is not None]
        pref_bytes = sum(p.pref_units * p.unit_bytes for p in fixed)
        sizes = {p.name: p.pref_units if pref_bytes <= budget else p.min_units for p in fixed}
        left = budget - sum(sizes[p.name] * p.unit_bytes for p in fixed)
        if fill:  # equal tokens in every fill pool, a whole number of units each
            lcm = math.lcm(*(p.tokens_per_unit for p in fill))
            tokens = left * lcm // sum(p.unit_bytes * (lcm // p.tokens_per_unit) for p in fill)
            sizes |= {p.name: tokens // p.tokens_per_unit for p in fill}
        return sizes

    def free(self) -> None:
        for p in self.pools.values():
            p._set(0)
