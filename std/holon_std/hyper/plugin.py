from . import kernels
from . import reference as ref


class Plugin:
    def register(self, regs) -> None:
        regs.ops.add(kernels.MIX.split("@")[0], ref.hc_gr_mix)
        regs.ops.add(kernels.COMBINE.split("@")[0], ref.hc_gr_combine)
