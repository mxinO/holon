from .recipe import recipe


class Plugin:
    recipe = staticmethod(recipe)

    def register(self, regs) -> None:
        pass
