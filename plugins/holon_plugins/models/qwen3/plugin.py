from .recipe import recipe


class Plugin:
    recipe = staticmethod(recipe)

    def register(self, regs) -> None:
        pass

    def bundle(self, checkpoint: str):
        from .bundle import Bundle  # transformers is imported only by frontends that need it

        return Bundle(checkpoint)
