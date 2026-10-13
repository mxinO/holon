from holon.api import AuxSpec, Ref, WeightMap
from torch import nn

from ..oracle import ORACLE
from ..oracle_drafter.plugin import Drafter as OracleDrafter


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
    `aux.kv`, its own pages) must equal the target's layer-0 KV at every computed position. With ORACLE.ahead it also
    drafts the oracle's tokens and writes their aux KV ahead, one sub-step per draft (positions num_computed + i),
    the way a chained draft head does."""

    def __init__(self, ctx, target) -> None:
        w = target.spec.weights
        spec = AuxSpec(Ref("tests.aux_head@1", hidden=target.spec.hidden, block=target.spec.block,
                           norm=target.spec.norm),
                       target.spec.layers[:1], WeightMap(w.source, w.rules, w.root, only=("model.layers.0.",)))
        self.aux = ctx.aux_model("aux", spec, target.tensors())
        self.embed = target.shared("embed")
        self.oracle = OracleDrafter()

    def substeps(self, k: int) -> list[int]:
        return [1] * k if ORACLE.ahead else []

    def lookahead(self, k: int) -> int:
        return k if ORACLE.ahead else 0

    def run(self, ctx, h) -> None:
        aux = ctx.view("aux")
        self.aux.layers[0](self.embed(ctx.batch.dev.input_ids), aux)
        if not ORACLE.ahead:
            ctx.seq_table.n_draft.zero_()
            return
        self.oracle.run(ctx, h)
        table = ctx.seq_table
        for i in range(ctx.batch.plan.draft_k):
            sub = aux.substep(1, i)
            slots = sub.batch.dev.slots
            sub.batch.dev.input_ids.copy_(table.last_token[slots] if i == 0 else table.draft_tokens[slots, i - 1])
            self.aux.layers[0](self.embed(sub.batch.dev.input_ids), sub)
