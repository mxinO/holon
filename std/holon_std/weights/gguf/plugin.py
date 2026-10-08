from .source import GGUFSource


class Plugin:
    def register(self, regs) -> None:
        regs.weight_sources.add(GGUFSource.kind, GGUFSource)
