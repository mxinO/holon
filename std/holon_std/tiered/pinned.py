"""std.tiered.pinned@1: the pinned RAM tier — every piece of a class in a mapped host pool, filled from the
checkpoint's extents at `before_materialize` and read by kernels in place (docs/v1-offload.md D10: inclusive, as the
RAM tier is wherever RAM holds every expert)."""
from __future__ import annotations


class Pinned:
    device_readable = True

    def __init__(self, ctx, **site) -> None:
        self.units: dict[tuple[str, int], int] = {}            # (class, piece) -> host unit

    def capacity(self, cls: str, pieces: int) -> int:
        return pieces

    def admit(self, cls: str, piece: int, unit: int) -> None:
        self.units[cls, piece] = unit

    def evict(self, cls: str, piece: int) -> None:
        del self.units[cls, piece]

    def read(self, cls: str, piece: int) -> tuple[str, int]:
        return "host_slot", self.units[cls, piece]
