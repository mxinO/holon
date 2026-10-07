"""Per-step device inputs without host synchronization (design §7.3, §7.4).

Every per-step input lives at a fixed offset of one static device arena, so its address is stable for graph capture.
A step writes its inputs into a pinned host arena with plain memory copies and `end()` moves the whole arena to the
device with one asynchronous copy. Two host arenas alternate, and an event per arena guards its reuse, so preparing
step N+1 never waits for step N's copy unless it is still pending. Device work that reads staged inputs is queued with
`after()` and runs once the copy is enqueued.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from . import platform

_ALIGN = 64


class Uploader:
    def __init__(self, device: torch.device, fields: dict[str, tuple[int, torch.dtype]]) -> None:
        """`fields`: name -> (maximum length, dtype)."""
        self._layout: dict[str, tuple[int, int, torch.dtype]] = {}       # name -> (byte offset, length, dtype)
        size = 0
        for name, (n, dtype) in fields.items():
            self._layout[name] = (size, n, dtype)
            size += -(-n * dtype.itemsize // _ALIGN) * _ALIGN
        self._device_arena = torch.zeros(size, dtype=torch.uint8, device=device)
        self._host_arenas = [platform.pinned_empty(size, torch.uint8, device) for _ in range(2)]
        self.dev = {name: self._view(self._device_arena, name) for name in fields}
        self._host = [{name: self._view(arena, name).numpy() for name in fields} for arena in self._host_arenas]
        self._done = [platform.Event(device), platform.Event(device)]
        self._after: list[Callable[[], None]] = []
        self._used = 0
        self._turn = 0

    def _view(self, arena: torch.Tensor, name: str) -> torch.Tensor:
        offset, n, dtype = self._layout[name]
        return arena[offset: offset + n * dtype.itemsize].view(dtype)

    def begin(self) -> None:
        """Start a step: switch to the other host arena once its previous copy has landed."""
        self._turn ^= 1
        self._done[self._turn].synchronize()
        self._used = 0

    def put(self, name: str, values: np.ndarray) -> torch.Tensor:
        """Stage `values`; returns the device view that holds them once `end()` has run."""
        n = len(values)
        self._host[self._turn][name][:n] = values
        offset, _, dtype = self._layout[name]
        self._used = max(self._used, offset + n * dtype.itemsize)
        return self.dev[name][:n]

    def after(self, fn: Callable[[], None]) -> None:
        """Run `fn` (device work reading staged inputs) right after this step's copy is enqueued."""
        self._after.append(fn)

    def end(self) -> None:
        if self._used:
            self._device_arena[: self._used].copy_(self._host_arenas[self._turn][: self._used], non_blocking=True)
        self._done[self._turn].record()
        for fn in self._after:
            fn()
        self._after.clear()
