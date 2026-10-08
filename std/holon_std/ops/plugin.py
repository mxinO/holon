from . import kernels, units
from . import reference as ref


class Plugin:
    def register(self, regs) -> None:
        for name, cls in [("rmsnorm", units.RMSNorm), ("rope", units.Rope), ("linear", units.Linear),
                          ("embedding", units.Embedding), ("lm_head", units.LMHead)]:
            regs.units.add(f"std.ops.{name}", cls)
        for target, oracle in ((kernels.RMSNORM, ref.rmsnorm), (kernels.ADD_RMSNORM, ref.rmsnorm),
                               (kernels.ROPE, ref.rope_neox), (kernels.QLINEAR, ref.qlinear),
                               (kernels.QEMBEDDING, ref.qembedding)):
            regs.ops.add(target.split("@")[0], oracle)
