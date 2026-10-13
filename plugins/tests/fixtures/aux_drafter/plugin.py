from holon.api import AuxSpec, Ref, WeightMap
from torch import nn


class AuxHead(nn.Module):
    """A copy of the target's first layer, built beside it as an auxiliary model."""

    def __init__(self, ctx, *, layers, hidden: int, block: Ref, norm: Ref) -> None:
        super().__init__()
        self.layers = nn.ModuleList(ctx.unit(block, layer=i, hidden=hidden, mixer=d.mixer, ffn=d.ffn, norm=norm)
                                    for i, d in enumerate(layers))


class Plugin:
    def register(self, regs) -> None:
        regs.units.add("tests.aux_head", AuxHead)

    def build(self, ctx, target):
        return Drafter(ctx, target)


class Drafter:
    """Runs the target's layer 0, loaded a second time as the aux model "aux", over every step's tokens: its KV (state
    `aux.kv`, its own pages) must equal the target's layer-0 KV at every computed position. It drafts nothing."""

    def __init__(self, ctx, target) -> None:
        w = target.spec.weights
        spec = AuxSpec(Ref("tests.aux_head@1", hidden=target.spec.hidden, block=target.spec.block,
                           norm=target.spec.norm),
                       target.spec.layers[:1], WeightMap(w.source, w.rules, w.root, only=("model.layers.0.",)))
        self.aux = ctx.aux_model("aux", spec, target.tensors())
        self.embed = target.shared("embed")

    def run(self, ctx, h) -> None:
        self.aux.layers[0](self.embed(ctx.batch.dev.input_ids), ctx.view("aux"))
        ctx.seq_table.n_draft.zero_()
