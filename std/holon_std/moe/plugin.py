from . import kernels, offload, units
from . import reference as ref


class Plugin:
    def register(self, regs) -> None:
        regs.units.add("std.moe", units.MoE)
        regs.units.add("std.moe.resident", units.Resident)
        regs.units.add("std.moe.offload", offload.Offload)
        for op, oracle in ((kernels.MOE_TOPK, ref.moe_topk), (kernels.MOE_ROUTER, ref.moe_router),
                           (kernels.LINEAR_EXPERTS, ref.linear_experts),
                           (kernels.MOE_COMBINE, ref.moe_combine), (kernels.MOE_RESOLVE, ref.moe_resolve),
                           (kernels.EXPERT_FETCH, ref.expert_fetch), (kernels.EXPERT_PREFETCH, ref.expert_fetch),
                           (kernels.LINEAR_EXPERTS_ARENA, ref.linear_experts)):
            regs.ops.add(op.split("@")[0], oracle)
