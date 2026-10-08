from . import kernels, units
from . import reference as ref


class Plugin:
    def register(self, regs) -> None:
        regs.units.add("std.mixers.gqa", units.GQA)
        for target in (kernels.PREFILL, kernels.DECODE, kernels.VERIFY):
            regs.ops.add(target.split("@")[0], ref.paged_attention)
        regs.ops.add(kernels.CAUSAL.split("@")[0], ref.causal_attention)
