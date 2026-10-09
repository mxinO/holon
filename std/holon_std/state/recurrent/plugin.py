from .kind import Recurrent


class Plugin:
    def register(self, regs) -> None:
        regs.state_kinds.add("recurrent", Recurrent)
