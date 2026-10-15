from . import hub
from .source import GGUFSource


class Plugin:
    def register(self, regs) -> None:
        regs.weight_sources.add(GGUFSource.kind, GGUFSource)

    def resolve(self, ref: str):
        """A GGUF file of a hub repository (`hooks.ResolvesRefs`; `hub.resolve`)."""
        return hub.resolve(ref)
