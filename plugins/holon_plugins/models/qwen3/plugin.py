from .recipe import recipe


class Plugin:
    archs = ("Qwen3ForCausalLM",)
    recipe = staticmethod(recipe)

    def register(self, regs) -> None:
        pass
