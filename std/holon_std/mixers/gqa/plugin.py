from . import units


class Plugin:
    def register(self, regs) -> None:
        regs.units.add("std.mixers.gqa", units.GQA)
