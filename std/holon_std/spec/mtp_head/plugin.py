from .units import MTPHCHead, MTPHead


class Plugin:
    def register(self, regs) -> None:
        regs.units.add("std.spec.mtp_head", MTPHead)
        regs.units.add("std.spec.mtp_hc_head", MTPHCHead)
