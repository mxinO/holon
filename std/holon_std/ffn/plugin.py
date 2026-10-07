from holon.api import Constraint, KernelSpec

from . import kernels, units
from . import reference as ref


class Plugin:
    def register(self, regs) -> None:
        regs.units.add("std.ffn.swiglu", units.SwiGLU)
        regs.ops.add("silu_mul", ref.silu_mul)
        regs.kernels.add("silu_mul", KernelSpec(op="silu_mul", provider="std.ffn", impl=kernels.SiluMul,
                                                requires=Constraint(devices=frozenset({"cpu", "cuda"})),
                                                props=frozenset({"deterministic", "graph_safe"}), priority=-100))
