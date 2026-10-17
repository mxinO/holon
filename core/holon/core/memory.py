"""Memory ledger and pools (design §6.1).

Pools are requested while the model is built and allocated by `materialize` once weights are loaded and the
activation peak has been profiled. Fixed pools (`pref_units`, e.g. per-request slots) get their preferred size, or
their minimum when memory is short. Plugin fill pools (`fill`) grow by priority towards their preferred size, past a
floor of paged state. Paged pools (`tokens_per_unit`, e.g. paged KV) share the remainder so that every paged pool
holds the same number of tokens. Host pools are allocated once, as they are requested, within their own budget.

A pool's buffer is `[segments, capacity, unit_bytes / segments]`: a unit is one chunk in each segment (e.g. one
KV page in every layer, laid out layer-major). The core can therefore copy units generically without knowing what
they hold.
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable

import numpy as np
import torch

from . import platform


class OutOfUnits(RuntimeError):
    pass


class Pool:
    def __init__(self, name: str, *, unit_bytes: int, segments: int, min_units: int, pref_units: int | None,
                 tokens_per_unit: int | None, priority: int, device: torch.device, null_unit: bool = False,
                 fill: bool = False, host: bool = False, mapped: bool = False) -> None:
        if unit_bytes % segments:
            raise ValueError(f"pool {name!r}: unit_bytes {unit_bytes} not divisible by {segments} segments")
        if (pref_units is None) == (tokens_per_unit is None):
            raise ValueError(f"pool {name!r}: give exactly one of pref_units (fixed, fill) or tokens_per_unit (paged)")
        self.name, self.unit_bytes, self.segments, self.device = name, unit_bytes, segments, device
        self.min_units, self.pref_units, self.tokens_per_unit = min_units, pref_units, tokens_per_unit
        self.priority, self.has_null, self.fill, self.host, self.mapped = priority, null_unit, fill, host, mapped
        self.buffer = None
        self._set(0)

    def _set(self, capacity: int) -> None:
        if self.host and self.buffer is not None:
            platform.free_host(self.buffer, self.device)
        self.capacity = capacity
        nbytes = capacity * self.unit_bytes
        self.buffer = (None if not nbytes else platform.host_memory(nbytes, self.device, self.mapped) if self.host
                       else torch.empty(nbytes, dtype=torch.uint8, device=self.device))
        self._alias = platform.device_alias(self.buffer, self.device) if self.mapped and nbytes else None
        self.capacity_dev = torch.tensor(capacity, dtype=torch.int32, device=self.device)
        # The null unit (the last one, never handed out) is where padding rows of a captured graph write.
        self.null = capacity - 1 if self.has_null and capacity else None
        self._free = list(range(capacity - (self.null is not None) - 1, -1, -1))  # pop() hands out low ids first
        self._refs = np.zeros(capacity, dtype=np.int32)
        if self.null is not None:   # zeroed, so what padding rows read is finite
            self.buffer.view(self.segments, capacity, self.unit_bytes // self.segments)[:, self.null] = 0

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

    def device_view(self, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
        """`view` on the device: a mapped host pool's alias of its buffer, a device pool's buffer."""
        if self.host and not self.mapped:
            raise ValueError(f"pool {self.name!r} is a host pool the device cannot address (mapped=False)")
        return (self._alias if self.host else self.buffer).view(dtype).view(shape)

    def alloc(self, n: int) -> np.ndarray:
        if n > len(self._free):
            raise OutOfUnits(f"pool {self.name!r}: need {n} units, {len(self._free)} free")
        if n == 1:                                # the common case: one page more
            unit = self._free.pop()
            self._refs[unit] = 1
            return np.array((unit,), dtype=np.int32)
        units = np.array(self._free[len(self._free) - n:][::-1], dtype=np.int32)    # as n pops would
        del self._free[len(self._free) - n:]
        self._refs[units] = 1
        return units

    def refcount(self, units: np.ndarray) -> np.ndarray:
        return self._refs[units]

    def held_once(self, units: tuple[int, ...]) -> bool:
        """Whether each of `units` has exactly one reference."""
        return self._refs[units[0]] == 1 if len(units) == 1 else bool((self._refs[list(units)] == 1).all())

    def incref(self, units: np.ndarray | tuple[int, ...]) -> None:
        if len(units) == 1:                       # a block of one page: skip the slow ufunc.at
            self._refs[units[0]] += 1
        else:
            np.add.at(self._refs, np.asarray(units, dtype=np.int64), 1)

    def decref(self, units: np.ndarray | tuple[int, ...]) -> None:
        if len(units) == 1:                       # a block of one page: skip the slow ufunc.at and unique
            unit = int(units[0])
            if self._refs[unit] <= 0:
                raise RuntimeError(f"pool {self.name!r}: unit released twice")
            self._refs[unit] -= 1
            if not self._refs[unit]:
                self._free.append(unit)
            return
        units = np.asarray(units, dtype=np.int64)
        np.subtract.at(self._refs, units, 1)
        if (self._refs[units] < 0).any():
            raise RuntimeError(f"pool {self.name!r}: unit released twice")
        self._free.extend(int(u) for u in np.unique(units) if self._refs[u] == 0)

    def units(self) -> torch.Tensor:
        """The buffer as `[segments, capacity, bytes per segment]`: unit u is `[:, u]`."""
        return self.buffer.view(self.segments, self.capacity, self.unit_bytes // self.segments)

    def copy(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        """Copy units `src` → `dst` (device index tensors) in every segment, in stream order."""
        units = self.units()
        units[:, dst] = units[:, src]


def copy_units(src: Pool, src_units, dst: Pool, dst_units) -> None:
    """Copy units between pools of the same unit shape, one unit at a time in stream order, with no temporary (a
    unit of recurrent state can be hundreds of MiB, beyond the profiled peak)."""
    if (src.unit_bytes, src.segments) != (dst.unit_bytes, dst.segments):
        raise ValueError(f"copy {src.name!r} → {dst.name!r}: unit shapes differ")
    s, d = src.units(), dst.units()
    for a, b in zip(np.asarray(src_units).tolist(), np.asarray(dst_units).tolist(), strict=True):
        d[:, b].copy_(s[:, a], non_blocking=True)


def run_mem_ops(ops, pools: dict[str, Pool], phase: str) -> None:
    """Run the step's MemOps of `phase` (design §7.3), in order, on the current stream."""
    for op in ops:
        if op.phase == phase:
            copy_units(pools[op.src_pool], op.src_units, pools[op.dst_pool], op.dst_units)


class Copies:
    """The MemOps with `after` or `lands_by`: run on the core's copy stream as soon as step `after`'s device work is
    done (at once without it), and waited for on the device by the launch of step `lands_by`, which therefore stalls
    for a late copy instead of reading what was there before."""

    def __init__(self, device: torch.device, pools: dict[str, Pool]) -> None:
        self.stream, self.pools = platform.Stream(device), pools
        self.held: list = []                              # ops after a step not launched yet
        self.landing: dict[int, list[platform.Event]] = {}  # step -> the copies its launch waits for
        self.last: platform.Event | None = None           # the end of the last launched step

    def add(self, ops: list, step: int) -> None:
        """The asynchronous ops handed in while step `step` is being launched (before its device work)."""
        for op in ops:
            if op.lands_by is not None and op.lands_by < max(step, (op.after or -1) + 1):
                raise ValueError(f"MemOp {op.src_pool} → {op.dst_pool}: lands_by {op.lands_by} must be after step "
                                 f"{op.after} and not before the step being launched ({step})")
            if op.after is not None and op.after >= step:
                self.held.append(op)
            else:
                self._run([op], self.last if op.after is not None else None)

    def launching(self, step: int) -> None:
        """Make the launch of `step` wait for the copies that land by it."""
        for event in self.landing.pop(step, ()):
            event.wait()

    def launched(self, step: int, done: platform.Event) -> None:
        """Step `step`'s device work is enqueued up to `done`: start the copies after it."""
        self.last = done
        ready = [op for op in self.held if op.after <= step]
        if ready:
            self.held = [op for op in self.held if op.after > step]
            self._run(ready, done)

    def _run(self, ops: list, after: platform.Event | None) -> None:
        done = self.stream.run(lambda: [copy_units(self.pools[op.src_pool], op.src_units, self.pools[op.dst_pool],
                                                   op.dst_units) for op in ops], after)
        for op in ops:
            if op.lands_by is not None:
                self.landing.setdefault(op.lands_by, []).append(done)


@dataclasses.dataclass
class Ledger:
    device: torch.device
    pools: dict[str, Pool] = dataclasses.field(default_factory=dict)
    host_memory: int = 0                  # bytes host pools may take
    paged_min_tokens: int = 0             # tokens every paged pool keeps before fill pools grow
    after_materialize: list[Callable[[], None]] = dataclasses.field(default_factory=list)
    materialized: bool = False            # host pools can no longer be requested

    def plugin_pool(self, name: str, *, unit_bytes: int, segments: int = 1, min_units: int = 0, pref_units: int = 0,
                    device: str = "device", mapped: bool = False, fill: bool = False, priority: int = 0) -> Pool:
        """A plugin's pool (`BuildCtx.pool`), namespaced so it cannot collide with the core's."""
        name, pref_units = f"plugin.{name}", max(pref_units, min_units)
        if device not in ("device", "host") or (mapped and device != "host") or (fill and device == "host"):
            raise ValueError(f"pool {name!r}: a device pool (fill or not) or a host pool (mapped or not), not "
                             f"device={device!r}, mapped={mapped}, fill={fill}")
        if device == "device":
            return self.request(name, unit_bytes=unit_bytes, segments=segments, min_units=min_units,
                                pref_units=pref_units, priority=priority, fill=fill)
        if self.materialized:
            raise ValueError(f"host pool {name!r} requested after the pools were materialized (request host pools in "
                             "`before_materialize` at the latest)")
        pool = self.request(name, unit_bytes=unit_bytes, segments=segments, min_units=min_units, pref_units=pref_units,
                            host=True, mapped=mapped)
        held = [p for p in self.pools.values() if p.host and p.capacity]
        left = self.host_memory - sum(p.capacity * p.unit_bytes for p in held)
        units = pref_units if pref_units * unit_bytes <= left else min_units
        if units * unit_bytes > left:
            del self.pools[name]
            raise OutOfUnits(f"host memory {self.host_memory / 2**30:.3f} GiB too small: {name} needs {min_units} "
                             f"units ({min_units * unit_bytes / 2**30:.3f} GiB), {left / 2**30:.3f} GiB left"
                             + "".join(f"; {p.name} holds {p.capacity} units "
                                       f"({p.capacity * p.unit_bytes / 2**30:.3f} GiB)" for p in held))
        pool._set(units)                  # allocated once, now, for the engine's lifetime
        return pool

    def request(self, name: str, *, unit_bytes: int, min_units: int, pref_units: int | None = None,
                tokens_per_unit: int | None = None, segments: int = 1, priority: int = 0, resize: str = "fixed",
                device: torch.device | None = None, null_unit: bool = False, fill: bool = False, host: bool = False,
                mapped: bool = False) -> Pool:
        """`null_unit`: reserve one unit that is never allocated (counted in `min_units`)."""
        if name in self.pools:
            raise ValueError(f"pool {name!r} requested twice")
        if resize != "fixed":
            raise NotImplementedError(f"resize mode {resize!r} (v1)")
        pool = Pool(name, unit_bytes=unit_bytes, segments=segments, min_units=min_units + null_unit,
                    pref_units=pref_units, tokens_per_unit=tokens_per_unit, priority=priority,
                    device=device or self.device, null_unit=null_unit, fill=fill, host=host, mapped=mapped)
        self.pools[name] = pool
        return pool

    def pool(self, name: str) -> Pool:
        return self.pools[name]

    def materialize(self, budget_bytes: int | None) -> None:
        """(Re)allocate every device pool within `budget_bytes`, or at minimum sizes if None; frees previous buffers.
        Then the `after_materialize` callbacks run."""
        self.free()
        self.materialized = True
        pools = [p for p in self.pools.values() if not p.host]
        if budget_bytes is None:
            sizes = {p.name: p.min_units for p in pools}
        else:
            sizes = self._sizes(pools, budget_bytes)
        short = [f"{p.name} ({sizes[p.name]} < {p.min_units} units)" for p in pools if sizes[p.name] < p.min_units]
        if short:
            raise OutOfUnits(f"memory budget {budget_bytes / 2**30:.3f} GiB too small: {', '.join(short)}")
        for p in pools:
            p._set(sizes[p.name])
        for fn in self.after_materialize:
            fn()

    def _sizes(self, pools: list[Pool], budget: int) -> dict[str, int]:
        fixed = [p for p in pools if p.pref_units is not None and not p.fill]
        fills = sorted((p for p in pools if p.fill), key=lambda p: -p.priority)     # stable: in request order
        paged = [p for p in pools if p.tokens_per_unit is not None]
        pref_bytes = sum(p.pref_units * p.unit_bytes for p in fixed)
        sizes = {p.name: p.pref_units if pref_bytes <= budget else p.min_units for p in fixed}
        sizes |= {p.name: p.min_units for p in fills}
        left = budget - sum(sizes[p.name] * p.unit_bytes for p in fixed + fills)
        # Fill pools grow by priority up to `pref_units` once paged state has its floor; paged state takes the rest.
        spare = left - sum(max(p.min_units, -(-self.paged_min_tokens // p.tokens_per_unit) + p.has_null)
                           * p.unit_bytes for p in paged)
        for p in fills:
            grow = max(0, min(p.pref_units - p.min_units, spare // p.unit_bytes))
            sizes[p.name] += grow
            spare -= grow * p.unit_bytes
            left -= grow * p.unit_bytes
        if paged:  # equal tokens in every paged pool, a whole number of units each
            lcm = math.lcm(*(p.tokens_per_unit for p in paged))
            tokens = left * lcm // sum(p.unit_bytes * (lcm // p.tokens_per_unit) for p in paged)
            sizes |= {p.name: tokens // p.tokens_per_unit for p in paged}
        return sizes

    def free(self, host: bool = False) -> None:
        """Release the device pools' buffers; with `host`, the host pools' too (the engine is closing)."""
        for p in self.pools.values():
            if host or not p.host:
                p._set(0)
