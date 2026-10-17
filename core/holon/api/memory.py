"""Memory pools (design §6.1).

Plugins never allocate device memory directly: they request named pools while the model is built
(`BuildCtx.pool`); the core's ledger sizes and allocates every pool once weights are loaded. A pool's buffer is
stable for its lifetime, so captured graphs stay valid. Units are reference-counted by the core: `alloc` returns
units holding one reference, which the core releases.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np
    import torch

class Pool(Protocol):
    """A pool's buffer is `[segments, capacity, unit_bytes / segments]`: one unit is a chunk in every segment (e.g.
    one KV page in every layer, layer-major), so the core can copy units without knowing what they hold."""

    name: str
    unit_bytes: int
    capacity: int
    num_free: int                    # units not allocated

    def view(self, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
        """The whole pool buffer as a tensor; the owning unit decides the shape, which starts with
        (segments, capacity) when the pool is segmented."""

    def alloc(self, n: int) -> np.ndarray: ...

    def device_view(self, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
        """A mapped host pool's buffer (`BuildCtx.pool(device="host", mapped=True)`) as a device tensor over the same
        memory, which kernels read and write in place across the bus (`view` is its host side)."""

