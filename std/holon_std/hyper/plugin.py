from . import kernels
from . import reference as ref
from .units import Expand, GRBlock, GRMix


class Plugin:
    def register(self, regs) -> None:
        regs.ops.add(kernels.MIX.split("@")[0], ref.hc_gr_mix)
        regs.ops.add(kernels.COMBINE.split("@")[0], ref.hc_gr_combine)
        regs.units.add("std.hyper.expand", Expand)
        regs.units.add("std.hyper.gr_mix", GRMix)
        regs.units.add("std.hyper.gr_block", GRBlock)
