"""models.qwen4_exp: Qwen3.8-Flash-Next (docs/v1-flash-next.md): its recipe over llama.cpp's `qwen4exp` GGUF and its
MTP draft head (`recipe.py`, D9), its PLE unit (`units.py`, D4), and PLE's op contracts (`kernels.py`) with their
oracles (`reference.py`); the kernels are `kernels.ple_triton`'s."""
from __future__ import annotations

import torch

from . import kernels
from . import reference as ref
from .recipe import draft_head, recipe
from .units import PLE


class Plugin:
    recipe = staticmethod(recipe)
    draft_head = staticmethod(draft_head)

    @staticmethod
    def trained_context(cfg, source) -> int | None:
        return cfg.get("max_position_embeddings")

    def register(self, regs) -> None:
        regs.ops.add(kernels.ROWS.split("@")[0], ref.ple_rows)
        regs.ops.add(kernels.BLOCK.split("@")[0], ref.ple_block)
        regs.units.add("qwen4_exp.ple", PLE)
        # Value transforms of the weight map (in f32: exact for these values, as models.qwen3_5's)
        regs.transforms.add("qwen4_exp.minus_one", lambda t: t.float() - 1)          # GGUF norms store w + 1
        regs.transforms.add("qwen4_exp.a_log", lambda t: torch.log(-t.float()))      # GGUF stores -exp(A_log)
