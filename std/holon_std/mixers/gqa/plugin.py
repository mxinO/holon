from holon.api import Constraint, KernelSpec

from . import kernels, units
from . import reference as ref


class Plugin:
    def register(self, regs) -> None:
        regs.units.add("std.mixers.gqa", units.GQA)
        for target in (kernels.PREFILL, kernels.DECODE, kernels.VERIFY):
            op = target.split("@")[0]
            regs.ops.add(op, ref.paged_attention)
            regs.kernels.add(op, KernelSpec(
                op=op, provider="std.mixers.gqa", impl=kernels.Reference, in_layouts={"kv": kernels.LAYOUT},
                requires=Constraint(devices=frozenset({"cpu", "cuda"})),
                supports=lambda f: f["layout"] == kernels.LAYOUT, props=frozenset({"deterministic"}),
                priority=-100))
