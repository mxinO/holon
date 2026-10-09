from . import kernels, units
from . import reference as ref


class Plugin:
    def register(self, regs) -> None:
        regs.units.add("std.mixers.gdn", units.GDN)
        for op, oracle in ((kernels.CAUSAL_CONV1D, ref.causal_conv1d), (kernels.GATED_DELTA_RULE, ref.gated_delta_rule),
                           (kernels.GATED_RMSNORM, ref.gated_rmsnorm)):
            regs.ops.add(op.split("@")[0], oracle)
