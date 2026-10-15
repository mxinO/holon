from . import hub
from .source import HFSafetensors


class Plugin:
    def register(self, regs) -> None:
        regs.weight_sources.add("hf_safetensors", HFSafetensors)

    def resolve(self, ref: str):
        """A hub repository id (`hooks.ResolvesRefs`; `hub.resolve`)."""
        return hub.resolve(ref)
