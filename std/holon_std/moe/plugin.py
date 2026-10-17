from . import kernels, units
from . import reference as ref


class Plugin:
    def register(self, regs) -> None:
        regs.units.add("std.moe", units.MoE)
        for op, oracle in ((kernels.MOE_TOPK, ref.moe_topk), (kernels.LINEAR_EXPERTS, ref.linear_experts),
                           (kernels.MOE_COMBINE, ref.moe_combine), (kernels.MOE_RESOLVE, ref.moe_resolve),
                           (kernels.EXPERT_FETCH, ref.expert_fetch)):
            regs.ops.add(op.split("@")[0], oracle)
