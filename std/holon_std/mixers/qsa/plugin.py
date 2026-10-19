from . import kernels
from . import reference as ref


class Plugin:
    def register(self, regs) -> None:
        for op, oracle in ((kernels.INDEX, ref.qsa_index), (kernels.SELECT, ref.qsa_select),
                           (kernels.ATTEND, ref.qsa_attend)):
            regs.ops.add(op.split("@")[0], oracle)
