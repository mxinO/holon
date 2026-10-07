"""Per-step device inputs without host synchronization (design §7.3, §7.4).

Every per-step input lives in a static device buffer of maximum size, so its address is stable for graph capture.
It is filled from pinned host staging with asynchronous copies; two staging sets alternate, and an event per set
guards its reuse, so preparing step N+1 never waits for step N's copies to land unless they are still pending.
"""
from __future__ import annotations

import numpy as np
import torch

from . import platform


class Uploader:
    def __init__(self, device: torch.device, fields: dict[str, tuple[int, torch.dtype]]) -> None:
        """`fields`: name -> (maximum length, dtype)."""
        self.device = device
        self.dev = {k: torch.zeros(n, dtype=dt, device=device) for k, (n, dt) in fields.items()}
        self._host = [{k: platform.pinned_empty(n, dt, device) for k, (n, dt) in fields.items()} for _ in range(2)]
        self._done = [platform.Event(device), platform.Event(device)]
        self._turn = 0

    def begin(self) -> None:
        """Start a step: switch to the other staging set once its previous copies have landed."""
        self._turn ^= 1
        self._done[self._turn].synchronize()

    def put(self, name: str, values: np.ndarray) -> torch.Tensor:
        """Stage `values` and copy them asynchronously; returns the device view holding them."""
        n = len(values)
        host = self._host[self._turn][name][:n]
        host.copy_(torch.from_numpy(np.ascontiguousarray(values)))
        out = self.dev[name][:n]
        out.copy_(host, non_blocking=True)
        return out

    def end(self) -> None:
        self._done[self._turn].record()
