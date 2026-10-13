import torch

from .recipe import draft_head, recipe


class Plugin:
    recipe = staticmethod(recipe)
    draft_head = staticmethod(draft_head)

    def register(self, regs) -> None:
        # Value transforms of the weight maps (applied in f32: exact for these values, see recipe.py).
        regs.transforms.add("qwen3_5.squeeze", lambda t: t.squeeze(1))                  # conv1d [C, 1, K] -> [C, K]
        regs.transforms.add("qwen3_5.minus_one", lambda t: t.float() - 1)             # GGUF norms store w + 1
        regs.transforms.add("qwen3_5.a_log", lambda t: torch.log(-t.float()))         # GGUF stores -exp(A_log)

    def bundle(self, source):
        from .bundle import Bundle

        return Bundle(source)
