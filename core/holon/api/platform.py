"""Thin device abstraction (design §8): everything device-specific the core needs, nothing more."""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch


@dataclasses.dataclass(frozen=True, slots=True)
class Fingerprint:
    """Static facts that select tuned locks (design §4.4). UUIDs and measured numbers go to evidence only."""

    gpu: str                 # SKU class from the product name, e.g. "rtx-5080", "rtx-a5000"; "cpu" without a GPU
    sm: int                  # 10*major + minor (0 on CPU)
    vram_gib: int
    toolchain: str           # e.g. "cu130-torch2.11"
    num_sms: int = 0         # implied by the SKU; kernels size their grids with it

    @property
    def cls(self) -> str:
        """Fingerprint class: the GPU SKU (which fixes SM and VRAM) plus toolchain majors."""
        return f"{self.gpu}/{self.toolchain}"


class Platform(Protocol):
    def fingerprint(self) -> Fingerprint: ...

    def device(self, index: int = 0) -> torch.device: ...

    def stream(self) -> Any: ...

    def event(self) -> Any:
        """A device event for ordering and host waits (commit at depth 1, phase timers)."""

    def mem_info(self) -> tuple[int, int]:
        """(free, total) bytes on the current device."""

    def pinned_empty(self, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor: ...

    def capture(self, fn: Callable[[], None], pool: Any = None) -> Callable[[], None]:
        """Capture `fn` into a graph and return its replay function."""

    # Stream memory operations for the CPU handshake inside graphs (design §7.4); signatures settled by spike S6.
    def write_value(self, flag: torch.Tensor, value: int) -> None: ...

    def wait_value(self, flag: torch.Tensor, value: int) -> None: ...
