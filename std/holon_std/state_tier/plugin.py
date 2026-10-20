from .kernels import STATE_COPY
from .reference import state_copy
from .tier import HostTier


class Plugin:
    def register(self, regs) -> None:
        regs.ops.add(STATE_COPY.split("@")[0], state_copy)

    def build(self, ctx) -> HostTier:
        """The engine's tier (`hooks.StateTierPlugin`)."""
        return HostTier(ctx)
