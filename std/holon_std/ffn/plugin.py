from . import kernels, units
from . import reference as ref


class Plugin:
    def register(self, regs) -> None:
        regs.units.add("std.ffn.swiglu", units.SwiGLU)
        regs.ops.add(kernels.SILU_MUL.split("@")[0], ref.silu_mul)
