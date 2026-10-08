from .recipe import recipe


class Plugin:
    recipe = staticmethod(recipe)

    def register(self, regs) -> None:
        pass

    def bundle(self, source):
        from .bundle import Bundle

        return Bundle(source)
