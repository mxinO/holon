from . import units


class Plugin:
    def register(self, regs) -> None:
        for name, cls in [("rmsnorm", units.RMSNorm), ("rope", units.Rope), ("linear", units.Linear),
                          ("embedding", units.Embedding), ("lm_head", units.LMHead)]:
            regs.units.add(f"std.ops.{name}", cls)
