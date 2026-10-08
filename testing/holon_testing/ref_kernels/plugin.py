from holon.api import Constraint, KernelSpec

from .kernels import BY_OP, PagedAttention

KV_LAYOUT = "kv.hnd"           # the only KV layout the paged reference reads


class Plugin:
    def register(self, regs) -> None:
        for op, impl in BY_OP.items():
            paged = impl is PagedAttention
            regs.kernels.add(op, KernelSpec(
                op=op, provider="testing.ref_kernels", impl=impl, requires=Constraint(devices=frozenset({"cpu"})),
                supports=(lambda f: f["layout"] == KV_LAYOUT) if paged else None,
                props=frozenset({"deterministic"} if paged else {"deterministic", "graph_safe"})))
