from .source import HFSafetensors


class Plugin:
    def register(self, regs) -> None:
        regs.weight_sources.add("hf_safetensors", HFSafetensors)
