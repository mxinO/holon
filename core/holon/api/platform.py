"""The device fingerprint that selects tuned locks (design §4.4). The device layer itself is the core's
(`holon.core.platform`, design §8), not a plugin."""
from __future__ import annotations

import dataclasses


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

