from . import units


class Plugin:
    def register(self, regs) -> None:
        regs.units.add("std.ffn.swiglu", units.SwiGLU)
