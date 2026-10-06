from .kind import Paged


class Plugin:
    def register(self, regs) -> None:
        regs.state_kinds.add("paged", Paged)
