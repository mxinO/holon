from . import kernels, offload, units
from . import reference as ref


class Plugin:
    @staticmethod
    def spill(options, checkpoint) -> tuple[dict[str, str], str] | None:
        """`holon serve`, after a start refused for host memory (docs/dev/v1-auto-config.md D1): the RAM-short mode of
        an offloaded MoE, named only (its share of RAM is the user's to choose)."""
        if options.get("moe.store") != "std.moe.offload@1" or "moe.ram" in options:
            return None
        return {}, ("or keep a share of the experts in RAM and read the rest from the checkpoint's file, 1.9–8.3× "
                    "slower (docs/dev/v1-offload.md): --plugin-options moe.ram='*=0.5' "
                    "moe.tiers=std.tiered.pinned@1,std.tiered.gguf_direct@1")

    def register(self, regs) -> None:
        regs.units.add("std.moe", units.MoE)
        regs.units.add("std.moe.resident", units.Resident)
        regs.units.add("std.moe.offload", offload.Offload)
        for op, oracle in ((kernels.MOE_TOPK, ref.moe_topk), (kernels.MOE_ROUTER, ref.moe_router),
                           (kernels.LINEAR_EXPERTS, ref.linear_experts),
                           (kernels.MOE_COMBINE, ref.moe_combine), (kernels.MOE_RESOLVE, ref.moe_resolve),
                           (kernels.EXPERT_FETCH, ref.expert_fetch), (kernels.EXPERT_PREFETCH, ref.expert_fetch),
                           (kernels.LINEAR_EXPERTS_ARENA, ref.linear_experts), (kernels.CPU_TASK, ref.cpu_task),
                           (kernels.CPU_EXPERTS, ref.cpu_experts), (kernels.MOE_PATCH_ROWS, ref.moe_patch_rows),
                           (kernels.MOE_TIERS, ref.moe_tiers), (kernels.ACTIVATION_LOAN, ref.ActivationLoan)):
            regs.ops.add(op.split("@")[0], oracle)
