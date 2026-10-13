from .units import MTPHead


class Plugin:
    def register(self, regs) -> None:
        regs.units.add("std.spec.mtp_head", MTPHead)
