"""models.qwen4_exp: Qwen3.8-Flash-Next. For now its PLE op contracts (`kernels.py`) and their oracles
(`reference.py`); the kernels are `kernels.ple_triton`'s. The model's recipe and weight rules land with FN's core seams
(step 3)."""
from __future__ import annotations

from . import kernels
from . import reference as ref


class Plugin:
    def register(self, regs) -> None:
        regs.ops.add(kernels.ROWS.split("@")[0], ref.ple_rows)
        regs.ops.add(kernels.BLOCK.split("@")[0], ref.ple_block)
