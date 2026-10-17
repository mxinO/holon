from . import pinned, static, store


class Plugin:
    def register(self, regs) -> None:
        regs.units.add("std.tiered.store", store.TieredStore)
        regs.units.add("std.tiered.pinned", pinned.Pinned)
        regs.units.add("std.tiered.static", static.Static)
