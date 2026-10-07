from holon.api import Constraint, KernelSpec

from . import kernels, units
from . import reference as ref


class Plugin:
    def register(self, regs) -> None:
        for name, cls in [("rmsnorm", units.RMSNorm), ("rope", units.Rope), ("linear", units.Linear),
                          ("embedding", units.Embedding), ("lm_head", units.LMHead)]:
            regs.units.add(f"std.ops.{name}", cls)
        for target, impl, oracle in ((kernels.RMSNORM, kernels.RMSNorm, ref.rmsnorm),
                                     (kernels.ADD_RMSNORM, kernels.AddRMSNorm, ref.rmsnorm),
                                     (kernels.ROPE, kernels.Rope, ref.rope_neox)):
            op = target.split("@")[0]
            regs.ops.add(op, oracle)
            regs.kernels.add(op, KernelSpec(op=op, provider="std.ops", impl=impl,
                                            requires=Constraint(devices=frozenset({"cpu", "cuda"})),
                                            props=frozenset({"deterministic", "graph_safe"}), priority=-100))
