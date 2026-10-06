"""Memory ledger and pools (design §6.1).

Plugins never allocate device memory directly: they request named pools from the ledger. A pool's base address is
stable for its lifetime, so captured graphs stay valid; `capacity_dev` lets in-graph code read the current size.
Units are reference-counted: `alloc` returns units with one reference, `decref` to zero frees them.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    import numpy as np
    import torch

ResizeMode = Literal["fixed", "recapture", "vmm"]


class Pool(Protocol):
    name: str
    device: torch.device
    unit_bytes: int
    capacity: int
    base: int                        # stable device address
    capacity_dev: torch.Tensor       # device scalar mirror of `capacity`

    def view(self, dtype: torch.dtype, unit_shape: tuple[int, ...]) -> torch.Tensor: ...

    def alloc(self, n: int) -> np.ndarray: ...

    def incref(self, units: np.ndarray) -> None: ...

    def decref(self, units: np.ndarray) -> None: ...


class Ledger(Protocol):
    def request(self, name: str, *, unit_bytes: int, min_units: int, pref_units: int, priority: int = 0,
                resize: ResizeMode = "fixed", page_group: str | None = None,
                device: torch.device | None = None) -> Pool: ...

    def pool(self, name: str) -> Pool: ...
