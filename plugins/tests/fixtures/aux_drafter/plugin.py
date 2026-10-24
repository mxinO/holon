from holon.api import AuxSpec, Ref, WeightMap, rewind_modules
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
    the way a chained draft head does. Plugin options: `aux.write_kv` — the step's tokens only write their KV (the
    attention's `write_kv`, no attention); `aux.device_lens` — the aux model is built with `planner.device_lens`."""

    def __init__(self, ctx, target) -> None:
        w = target.spec.weights
        layer0 = next(r.pattern for r in w.rules if "{i}." in r.pattern).split("{i}.")[0] + "0."   # HF's, GGUF's
        spec = AuxSpec(Ref("tests.aux_head@1", hidden=target.spec.hidden, block=target.spec.block,
                           norm=target.spec.norm),
                       target.spec.layers[:1], WeightMap(w.source, w.rules, w.root, only=(layer0,)))
        options = ctx.options.get("plugins", {})
        self.write_only = bool(options.get("aux.write_kv"))
        self.aux = ctx.aux_model("aux", spec, target.tensors(),
                                 {"planner.device_lens": True} if options.get("aux.device_lens") else None)
        self.embed = target.shared("embed")
        self.oracle = OracleDrafter(ctx.options["max_seqs"] + 1, ctx.options["max_model_len"])

    def substeps(self, k: int) -> list[int]:
        return [1] * k if ORACLE.ahead else []

    def lookahead(self, k: int) -> int:
        return k if ORACLE.ahead else 0

    def rewind(self, ctx) -> None:
        self.oracle.rewind(ctx)

    def run(self, ctx, h) -> None:
        aux, layer = ctx.view("aux"), self.aux.layers[0]
        x = self.embed(ctx.batch.dev.input_ids)
        if self.write_only:
            layer.mixer.write_kv(layer.input_norm(x), aux)
        else:
            layer(x, aux)
        if (ctx.batch.plan.n_logits > 1).any():          # a verify step: roll the aux model back as the core did
            rewind_modules(self.aux, aux)                # the target (its recurrent layers keep only accepted tokens)
        if not ORACLE.ahead:
            self.oracle.rewound = None
            ctx.seq_table.n_draft.zero_()
            return
        self.oracle.run(ctx, h)
        table = ctx.seq_table
        for i in range(ctx.batch.plan.draft_k):
            sub = aux.substep(1, i)
            assert sub.sampler is ctx.sampler                  # a drafter samples on its sub-steps' rows
            slots = sub.batch.dev.slots
            sub.batch.dev.input_ids.copy_(table.last_token[slots] if i == 0 else table.draft_tokens[slots, i - 1])
            self.aux.layers[0](self.embed(sub.batch.dev.input_ids), sub)
            if len(slots):
                ORACLE.max_position = max(ORACLE.max_position, int(sub.batch.dev.positions.max()))
