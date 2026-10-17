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
    cpu_isa: str = ""        # the CPU's tier: the best prebuilt compiler flag set it runs, e.g. "x86-avx2",
                             # "x86-avx512" (never a CPU model); in a lock's class only when CPU kernels run

    @property
    def cls(self) -> str:
        """Fingerprint class: the GPU SKU (which fixes SM and VRAM) plus toolchain majors."""
        return f"{self.gpu}/{self.toolchain}"

