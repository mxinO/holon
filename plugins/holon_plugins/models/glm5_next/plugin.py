"""models.glm5_next: GLM-5.3-Flash (docs/dev/v1-glm.md): its recipe over llama.cpp's `glm5-next` GGUF (`recipe.py`) and
the frontend bundle (`bundle.py`: the GGUF's tokenizer and chat template, `glm_xml@1` tool calls, `think@1`)."""
from __future__ import annotations

import torch

from .recipe import recipe


class Plugin:
    recipe = staticmethod(recipe)

    @staticmethod
    def trained_context(cfg, source) -> int | None:
        return cfg.get("max_position_embeddings")

    def register(self, regs) -> None:
        # Value transforms of the weight map (in f32: exact for these values)
        regs.transforms.add("glm5_next.a_log", lambda t: torch.log(-t.float()))     # GGUF stores −exp(A_log)
        regs.transforms.add("glm5_next.conv", lambda t: t.float().reshape(-1, t.shape[-1]))   # [1, C, 1, K] → [C, K]

    def bundle(self, source):
        from .bundle import Bundle

        return Bundle(source)
