"""Memory ledger and pools (design §6.1).

Plugins never allocate device memory directly: they request named pools from the ledger while the model is built;
the ledger sizes and allocates every pool once weights are loaded (`materialize`). A pool's base address is stable
for its lifetime, so captured graphs stay valid; `capacity_dev` lets in-graph code read the current size. Units are
reference-counted: `alloc` returns units with one reference, `decref` to zero frees them.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    import numpy as np
    import torch

ResizeMode = Literal["fixed", "recapture", "vmm"]


class Pool(Protocol):
    """A pool's buffer is `[segments, capacity, unit_bytes / segments]`: one unit is a chunk in every segment (e.g.
    one KV page in every layer, layer-major), so the core can copy units without knowing what they hold."""

    name: str
    device: torch.device
    unit_bytes: int
    segments: int
    capacity: int
    base: int                        # stable device address
    capacity_dev: torch.Tensor       # device scalar mirror of `capacity`

    def view(self, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
        """The whole pool buffer as a tensor; the owning unit decides the shape, which starts with
        (segments, capacity) when the pool is segmented."""

    def alloc(self, n: int) -> np.ndarray: ...

    def incref(self, units: np.ndarray) -> None: ...

    def decref(self, units: np.ndarray) -> None: ...

    def copy(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        """Copy units `src` → `dst` in every segment, in stream order (the core's MemOp)."""


class Ledger(Protocol):
    def request(self, name: str, *, unit_bytes: int, min_units: int, pref_units: int | None = None,
                tokens_per_unit: int | None = None, segments: int = 1, priority: int = 0,
                resize: ResizeMode = "fixed", device: torch.device | None = None) -> Pool:
        """A fixed pool (`pref_units`, e.g. per-request slots) or a fill pool (`tokens_per_unit`, e.g. paged KV)
        that shares the memory left after fixed pools, every fill pool holding the same number of tokens."""

    def pool(self, name: str) -> Pool: ...

    def materialize(self, budget_bytes: int | None) -> None:
        """Size and allocate every requested pool within `budget_bytes`; `None` allocates minimum sizes
        (design §6.1)."""
