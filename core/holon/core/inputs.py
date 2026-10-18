"""Per-step device inputs without host synchronization (design §7.3, §7.4).

Every per-step input lives at a fixed offset of one static device arena, so its address is stable for graph capture.
A step writes its inputs into a pinned host arena with plain memory copies and `end()` moves exactly the written byte
ranges to the device (ranges a small gap apart are merged, so a step costs a copy or two). Fields the device computes
itself are never written, so never copied (a merged gap may still cover them): each step must rewrite them in full
before reading them. Two host arenas alternate, and an event per arena guards its reuse, so preparing step N+1 never
waits for step N's copy unless it is still pending. Device work that reads staged inputs is queued with `after()` and
runs once the copy is enqueued.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from . import platform

_ALIGN = 64
_MERGE_GAP = 4096        # bytes: copying a small unwritten gap is cheaper than another copy launch


class Uploader:
    def __init__(self, device: torch.device, fields: dict[str, tuple[int, torch.dtype]]) -> None:
        """`fields`: name -> (maximum length, dtype)."""
        self._layout: dict[str, tuple[int, int, torch.dtype]] = {}       # name -> (byte offset, length, dtype)
        size = 0
        # Smallest first, so a small step's inputs lie together: its written ranges merge into one copy.
        for name, (n, dtype) in sorted(fields.items(), key=lambda f: f[1][0] * f[1][1].itemsize):
            self._layout[name] = (size, n, dtype)
            size += -(-n * dtype.itemsize // _ALIGN) * _ALIGN
        self._device_arena = torch.zeros(size, dtype=torch.uint8, device=device)
        self._host_arenas = [platform.pinned_empty(size, torch.uint8, device) for _ in range(2)]
        self.dev = {name: self._view(self._device_arena, name) for name in fields}
        self._host = [{name: self._view(arena, name).numpy() for name in fields} for arena in self._host_arenas]
        self._done = [platform.Event(device), platform.Event(device)]
        self._after: list[Callable[[], None]] = []
        self._written: list[tuple[int, int]] = []
        self._turn = 0

    def _view(self, arena: torch.Tensor, name: str) -> torch.Tensor:
        offset, n, dtype = self._layout[name]
        return arena[offset: offset + n * dtype.itemsize].view(dtype)

    def begin(self) -> None:
        """Start a step: switch to the other host arena once its previous copy has landed."""
        self._turn ^= 1
        self._done[self._turn].synchronize()
        self._written.clear()

    def put(self, name: str, values: np.ndarray) -> torch.Tensor:
        """Stage `values`; returns the device view that holds them once `end()` has run."""
        host, dev = self.stage(name, len(values))
        host[:] = values
        return dev

    def stage(self, name: str, n: int) -> tuple[np.ndarray, torch.Tensor]:
        """The host view of `name`'s first `n` values, for the caller to write in full before `end()`, and the
        device view that holds them afterwards."""
        offset, _, dtype = self._layout[name]
        if n:
            self._written.append((offset, offset + n * dtype.itemsize))
        return self._host[self._turn][name][:n], self.dev[name][:n]

    def after(self, fn: Callable[[], None]) -> None:
        """Run `fn` (device work reading staged inputs) right after this step's copy is enqueued."""
        self._after.append(fn)

    def end(self) -> None:
        host = self._host_arenas[self._turn]
        for lo, hi in _merged(self._written):
            self._device_arena[lo:hi].copy_(host[lo:hi], non_blocking=True)
        self._done[self._turn].record()
        for fn in self._after:
            fn()
        self._after.clear()


def _merged(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for lo, hi in sorted(ranges):
        if out and lo - out[-1][1] <= _MERGE_GAP:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [(lo, hi) for lo, hi in out]


class Writes:
    """Device writes handed in at launch (`before_launch`: `(dst, values)` pairs, `dst` of a dtype numpy holds), staged
    through two alternating pinned host buffers that grow to the largest launch's writes, so the launch never waits for
    the device."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._host: list[torch.Tensor | None] = [None, None]
        self._done = [platform.Event(device), platform.Event(device)]
        self._turn = 0

    def run(self, writes: list[tuple[torch.Tensor, np.ndarray]]) -> None:
        if not writes:
            return
        self._turn ^= 1
        self._done[self._turn].synchronize()
        sizes = [-(-dst.numel() * dst.element_size() // _ALIGN) * _ALIGN for dst, _ in writes]
        host = self._host[self._turn]
        if host is None or host.numel() < sum(sizes):
            host = self._host[self._turn] = platform.pinned_empty(sum(sizes), torch.uint8, self.device)
        at = 0
        for (dst, values), size in zip(writes, sizes, strict=True):
            staged = host[at: at + dst.numel() * dst.element_size()].view(dst.dtype).view(dst.shape)
            staged.numpy()[...] = values
            dst.copy_(staged, non_blocking=True)
            at += size
        self._done[self._turn].record()
