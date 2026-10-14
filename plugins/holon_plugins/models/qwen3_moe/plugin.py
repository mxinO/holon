from .recipe import recipe


class Plugin:
    recipe = staticmethod(recipe)

    @staticmethod
    def trained_context(cfg, source) -> int | None:
        return cfg.get("max_position_embeddings")     # HF config; GGUF's context_length, mapped by the source

    def register(self, regs) -> None:
        pass

    def bundle(self, source):
        from .bundle import Bundle

        return Bundle(source)
