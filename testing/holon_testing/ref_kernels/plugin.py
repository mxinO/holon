from holon.api import Constraint, KernelSpec

from .kernels import BY_OP


class Plugin:
    def register(self, regs) -> None:
        for op, impl in BY_OP.items():
            regs.kernels.add(op, KernelSpec(op=op, provider="testing.ref_kernels", impl=impl,
                                            requires=Constraint(devices=frozenset({"cpu"})),
                                            props=frozenset({"deterministic", "graph_safe"})))
