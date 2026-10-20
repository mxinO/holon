"""Drafts with a DFlash / DFlash2 checkpoint (docs/m04-spec.md D4, D5): a block of B rows — the last token, then
B − 1 mask tokens — drafted in one forward of a small model that attends to the target's own features.

Every step (`run`), after the target's forward:
1. context: the target's residual streams after the checkpoint's `target_layer_ids` (taps, copied as the trunk passes
   them) become, through `fc` and `hidden_norm`, the draft's context keys and values at the step's positions in every
   draft layer (its own paged KV, `dflash.kv`, prefix-shared). A verify row's rejected positions are written too:
   they lie past the committed length, where this step's block overwrites them and later steps' context does;
2. for the rows that sampled: one sub-step of B rows at positions num_computed + j, the anchor (the token just
   emitted) then the mask token (the target's embedding row: this checkpoint ships no mask embedding). Its attention
   sees the context before it and the whole block (not causal, a sliding window); rows 1 … B − 1 predict the tokens
   at their own positions;
3. DFlash2: each row's top-k candidates under the target's `lm_head` are scored pairwise by the selector and walked
   from the anchor, ancestrally through the sampler at each request's own parameters on a uniform stream of the
   drafter's (greedy rows: the argmax, a one-hot law). The laws walked are published with the drafts (`q_support`),
   so `std.spec.rejection` verifies sampled requests losslessly.
"""
from __future__ import annotations

from typing import Any

import torch
from holon.api import AuxSpec, LayerDecl, Ref, WeightMap, WeightRule

NAMESPACE = "dflash"
STREAM = 8                                    # the drafter's uniforms (api.decoding.Sampler)

HF_WEIGHTS = WeightMap("hf_safetensors", rules=(
    WeightRule("fc.weight", "fc.weight"),
    WeightRule("hidden_norm.weight", "hidden_norm.weight"),
    WeightRule("norm.weight", "norm.weight"),
    WeightRule("layers.{i}.input_layernorm.weight", "layers.{i}.input_norm.weight"),
    WeightRule("layers.{i}.post_attention_layernorm.weight", "layers.{i}.post_norm.weight"),
    WeightRule("layers.{i}.self_attn.{rest}", "layers.{i}.mixer.{rest}"),
    WeightRule("layers.{i}.mlp.{rest}", "layers.{i}.ffn.{rest}"),
    WeightRule("layers.{i}.attention_conv.{rest}", "layers.{i}.attention_conv.{rest}"),
    WeightRule("layers.{i}.mlp_conv.{rest}", "layers.{i}.mlp_conv.{rest}"),
    WeightRule("candidate_selector.predecessor_codebook", "selector.predecessor.weight"),
    WeightRule("candidate_selector.successor_codebook", "selector.successor.weight"),
    WeightRule("candidate_selector.hidden_projection.weight", "selector.hidden_projection.weight"),
))
GGUF_WEIGHTS = WeightMap("gguf", rules=(         # llama.cpp's converter (conversion/qwen.py `DFlashModel`)
    WeightRule("fc.weight", "fc.weight"),
    WeightRule("enc.output_norm.weight", "hidden_norm.weight"),
    WeightRule("output_norm.weight", "norm.weight"),
    WeightRule("blk.{i}.attn_norm.weight", "layers.{i}.input_norm.weight"),
    WeightRule("blk.{i}.ffn_norm.weight", "layers.{i}.post_norm.weight"),
    *(WeightRule(f"blk.{{i}}.attn_{g}.weight", f"layers.{{i}}.mixer.{h}.weight")
      for g, h in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"), ("output", "o_proj"),
                   ("q_norm", "q_norm"), ("k_norm", "k_norm"))),
    *(WeightRule(f"blk.{{i}}.ffn_{g}.weight", f"layers.{{i}}.ffn.{g}_proj.weight") for g in ("gate", "up", "down")),
    *(WeightRule(f"blk.{{i}}.{g}_conv_base", f"layers.{{i}}.{h}_conv.base_kernel") for g, h in (("attn", "attention"),
                                                                                                  ("ffn", "mlp"))),
    *(WeightRule(f"blk.{{i}}.{g}_conv_proj.weight", f"layers.{{i}}.{h}_conv.kernel_projection.weight")
      for g, h in (("attn", "attention"), ("ffn", "mlp"))),
    WeightRule("selector_predecessor.weight", "selector.predecessor.weight"),
    WeightRule("selector_successor.weight", "selector.successor.weight"),
    WeightRule("selector_hidden.weight", "selector.hidden_projection.weight"),
))


def checkpoint(source) -> dict[str, Any]:
    """The draft's configuration, from an HF checkpoint's `config.json` or a GGUF's metadata."""
    c = source.config()
    if source.kind == "gguf":
        g = c["gguf"]
        return {"hidden": c["hidden_size"], "vocab": c["vocab_size"], "layers": c["num_hidden_layers"],
                "heads": c["num_attention_heads"], "kv_heads": c["num_key_value_heads"], "head_dim": c["head_dim"],
                "inter": c["intermediate_size"], "eps": c["rms_norm_eps"], "theta": c["rope_theta"],
                "block": g["block_size"], "taps": tuple(int(i) - 1 for i in g["target_layers"]),
                "mask": c["mask_token_id"], "causal": bool(g.get("attention.causal", True)),
                "window": int(g.get("attention.sliding_window", 0)), "conv": g.get("conv_kernel_size", 0),
                "group": g.get("conv_group_size", 0), "rank": g.get("selector_rank", 0),
                "top_k": g.get("selector_top_k", 0)}
    d = c.get("dflash_config", {})
    window = c.get("sliding_window") if c.get("use_sliding_window") else 0
    return {"hidden": c["hidden_size"], "vocab": c["vocab_size"], "layers": c["num_hidden_layers"],
            "heads": c["num_attention_heads"], "kv_heads": c["num_key_value_heads"],
            "head_dim": c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"],
            "inter": c["intermediate_size"], "eps": c["rms_norm_eps"],
            "theta": c.get("rope_parameters", {}).get("rope_theta") or c.get("rope_theta", 1e6),
            "block": d.get("block_size", 16), "taps": tuple(d["target_layer_ids"]), "mask": d["mask_token_id"],
            "causal": bool(c.get("is_causal", False)), "window": int(window or 0),
            "conv": d.get("conv_kernel_size", 0), "group": d.get("conv_group_size", 0),
            "rank": d.get("selector_rank", 0), "top_k": d.get("selector_top_k", 0)}


def tap_layers(cfg: dict[str, Any]) -> tuple[int, ...]:
    """The target layers whose output (the residual stream after them) the draft reads: `target_layer_ids`, HF's
    numbering (llama.cpp's GGUF stores id + 1, the next layer's input — the same tensor)."""
    return cfg["taps"]


class Plugin:
    def register(self, regs) -> None:
        from .units import Draft

        regs.units.add("drafters.dflash.draft", Draft)

    def build(self, ctx, target):
        path = ctx.options.get("plugins", {}).get("dflash.path")
        if not path:
            raise ValueError("drafters.dflash: set the plugin option dflash.path to the draft checkpoint")
        return DFlashDrafter(ctx, target, ctx.weight_source(path))


class DFlashDrafter:
    def __init__(self, ctx, target, source) -> None:
        cfg = self.cfg = checkpoint(source)
        if (r := target.spec.residual) is not None and r.width != target.spec.hidden:
            raise ValueError(f"drafters.dflash: the target's residual is {r.width} wide (ModelSpec.residual), not its "
                             f"hidden size {target.spec.hidden}: the draft's taps read hidden-size features")
        if (cfg["hidden"], cfg["vocab"]) != (target.spec.hidden, target.spec.vocab):
            raise ValueError(f"drafters.dflash: the draft's hidden size and vocabulary {cfg['hidden'], cfg['vocab']} "
                             f"are not the target's {target.spec.hidden, target.spec.vocab}")
        if not cfg["rank"]:
            raise ValueError("drafters.dflash: plain DFlash checkpoints (no candidate selector) are not supported")
        norm = Ref("std.ops.rmsnorm@1", eps=cfg["eps"])
        attn = Ref("std.mixers.gqa@1", heads=cfg["heads"], kv_heads=cfg["kv_heads"], head_dim=cfg["head_dim"],
                   qk_norm=norm, rope=Ref("std.ops.rope@1", theta=cfg["theta"]), causal=cfg["causal"],
                   window=cfg["window"])
        layers = tuple(LayerDecl(attn, Ref("std.ffn.swiglu@1", inter=cfg["inter"]), "draft")
                       for _ in range(cfg["layers"]))
        taps = tap_layers(cfg)
        unit = Ref("drafters.dflash.draft@1", hidden=cfg["hidden"], features=len(taps), norm=norm,
                   conv={"block": cfg["block"], "taps": cfg["conv"], "group": cfg["group"]} if cfg["conv"] else None,
                   vocab=cfg["vocab"], rank=cfg["rank"])
        # Every pass but the block's only writes context: its planners need device lengths only (verify kernels).
        weights = GGUF_WEIGHTS if source.kind == "gguf" else HF_WEIGHTS
        self.model = ctx.aux_model(NAMESPACE, AuxSpec(unit, layers, weights), source.tensors(),
                                   {"planner.device_lens": True})
        self.embed, self.lm_head = target.shared("embed"), target.shared("lm_head")
        self.block_size, self.top_k, self.q_support = cfg["block"], cfg["top_k"], cfg["top_k"]
        o = ctx.options
        tokens = max(o["max_step_tokens"], o["max_seqs"] * (1 + o["max_draft_tokens"]))
        hidden = cfg["hidden"]
        self.features = ctx.static("drafter.dflash.features", (tokens, len(taps) * hidden), ctx.dtype)
        for n, layer in enumerate(taps):
            ctx.tap(layer, lambda h, step, at=slice(n * hidden, (n + 1) * hidden): self.features[: len(h), at].copy_(h))
        self.mask = cfg["mask"]

    def substeps(self, k: int) -> list[int]:
        return [self.block_size] if k else []

    def lookahead(self, k: int) -> int:
        return self.block_size if k else 0

    def run(self, ctx, h: torch.Tensor) -> None:
        dev, table, plan = ctx.batch.dev, ctx.seq_table, ctx.batch.plan
        view = ctx.view(NAMESPACE)
        self.model.context(self.features[: len(h)], view)
        verify = bool((plan.n_logits > 1).any())
        if not plan.draft_k or not len(dev.logits_rows):
            table.n_draft.index_fill_(0, dev.slots if verify else dev.slots[dev.logits_rows], 0)
            return
        b = self.block_size
        sub = view.substep(b, 0)
        slots = sub.batch.dev.slots
        ids = sub.batch.dev.input_ids.view(-1, b)
        ids.fill_(self.mask)
        ids[:, 0] = table.last_token[slots]
        out = self.model.block(self.embed(sub.batch.dev.input_ids), sub).view(len(slots), b, -1)[:, 1:]
        unary, cand = self.lm_head(out.reshape(-1, out.shape[-1])).float().topk(self.top_k)
        cand, unary = cand.view(len(slots), b - 1, -1), unary.view(len(slots), b - 1, -1)
        drafts, laws = self.walk(self.model.selector(cand, unary, out, ids[:, 0]), cand, sub)
        n = min(plan.draft_k, b - 1)
        table.draft_tokens[slots, :n] = drafts[:, :n]
        table.draft_q_ids[slots, :n] = cand[:, :n]
        table.draft_q[slots, :n] = laws[:, :n]
        table.n_draft.index_fill_(0, slots, n)

    def walk(self, scores: torch.Tensor, cand: torch.Tensor, sub) -> tuple[torch.Tensor, torch.Tensor]:
        """One path through the selector's lattice, slot by slot from the anchor: each slot's candidate drawn from
        the law the sampler gives that row of scores (its temperature and filters; one-hot at the argmax for greedy
        requests) with the drafter's uniform at the drafted token's position. Returns the drafts and their laws [R,
        B − 1, (K)]: exactly the tensors drawn from."""
        r, slots_, k = scores.shape[:3]
        b = slots_ + 1
        u = sub.sampler.uniform(sub, STREAM).view(r, b)              # block row j draws the token at its position + 1
        prev = torch.zeros(r, dtype=torch.long, device=scores.device)
        first = torch.arange(r, device=scores.device) * b                   # each request's first block row
        drafts, laws = [], []
        for e in range(slots_):
            row = scores[:, e].gather(1, prev[:, None, None].expand(-1, 1, k))[:, 0]      # [R, K]: after prev
            law = sub.sampler.distribution(row, sub, rows=first + e)
            c = law.cumsum(-1)
            prev = torch.searchsorted(c, (u[:, e] * c[:, -1])[:, None], right=True)[:, 0].clamp_(max=k - 1)
            drafts.append(cand[:, e].gather(1, prev[:, None])[:, 0])
            laws.append(law)
        return torch.stack(drafts, 1), torch.stack(laws, 1)
