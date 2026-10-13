"""L0-drafter (docs/m04-spec.md §5): a draft head's teacher-forced logits judged like L2 — against an fp32 golden of
the published forward, with a floor (the same reference in the model's dtype) setting the budgets.

The MTP head of Qwen3.5 / Qwen3.8 (`mtp.*`): at every prompt position p, the head takes token p and the target's
final-norm hidden state of position p − 1 (zeros at p = 0; the drafter's pairing, D3) and predicts token p + 1, so its
logits form a recording of the same shape as the target's. The reference is transformers' own decoder layer and
norms loaded with the checkpoint's MTP weights, on the target's hidden states from transformers; the candidate is the
engine's drafter, its head's output over each prompt captured from its pass over a one-step prefill.
"""
from __future__ import annotations

import json
import pathlib

import torch
import transformers
from safetensors import safe_open

from .golden import hardware, model_meta, resolve_model, topk_rows
from .recording import Prompt, PromptSet, Recording


def _mtp_weights(path: pathlib.Path) -> dict[str, torch.Tensor]:
    index = path / "model.safetensors.index.json"
    files = sorted({f for n, f in json.loads(index.read_text())["weight_map"].items() if n.startswith("mtp.")}) \
        if index.exists() else ["model.safetensors"]
    out = {}
    for name in files:
        with safe_open(str(path / name), "pt") as f:
            out |= {n: f.get_tensor(n) for n in f.keys() if n.startswith("mtp.")}
    return out


class _Reference(torch.nn.Module):
    """transformers' modules for the MTP block: pre-fc norms, fc, one full-attention decoder layer, final norm."""

    def __init__(self, hf, weights: dict[str, torch.Tensor], dtype: torch.dtype) -> None:
        super().__init__()
        from transformers.models.qwen3_5 import modeling_qwen3_5 as q

        # The target's language model and head, held outside the module: only the MTP block is loaded here.
        lm = hf.model.language_model if hasattr(hf.model, "language_model") else hf.model
        self._target = (lm, hf.lm_head)
        cfg = lm.config
        norm = lambda: q.Qwen3_5RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)          # noqa: E731
        self.enorm, self.hnorm, self.norm = norm(), norm(), norm()
        self.fc = torch.nn.Linear(2 * cfg.hidden_size, cfg.hidden_size, bias=False)
        full = next(i for i, t in enumerate(cfg.layer_types) if t == "full_attention")
        self.layer = q.Qwen3_5DecoderLayer(cfg, layer_idx=full)
        state = {"enorm.weight": weights["mtp.pre_fc_norm_embedding.weight"],
                 "hnorm.weight": weights["mtp.pre_fc_norm_hidden.weight"],
                 "norm.weight": weights["mtp.norm.weight"], "fc.weight": weights["mtp.fc.weight"]}
        state |= {f"layer.{k.removeprefix('mtp.layers.0.')}": v for k, v in weights.items()
                  if k.startswith("mtp.layers.0.")}
        self.load_state_dict({k: v.to(dtype) for k, v in state.items()})
        self.to(dtype).eval()

    @torch.no_grad()
    def logits(self, ids: list[int]) -> torch.Tensor:
        lm, head = self._target
        x = torch.tensor([ids])
        hidden = lm(x).last_hidden_state[0]                                         # final-norm: the lm_head input
        prev = torch.cat([hidden.new_zeros(1, hidden.shape[1]), hidden[:-1]])
        h = self.fc(torch.cat([self.enorm(lm.embed_tokens(x[0])), self.hnorm(prev)], -1))[None]
        n = len(ids)
        positions = torch.arange(n)[None]
        pe = lm.rotary_emb(h, positions[None].expand(3, 1, n))
        mask = torch.full((n, n), float("-inf"), dtype=h.dtype).triu(1)[None, None]
        return head(self.norm(self.layer(h, position_embeddings=pe, attention_mask=mask,
                                              position_ids=positions)))[0].float()


def reference(model: str, prompts: PromptSet, dtype: torch.dtype = torch.float32, k: int = 20) -> Recording:
    """The MTP head's teacher-forced recording from transformers, on the CPU in `dtype`."""
    path = resolve_model(model)
    hf = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation="sdpa").eval()
    for param in hf.parameters():                          # transformers 5.6 ignores `dtype` for Qwen3.5 (golden.py)
        param.data = param.data.to(dtype)
    mtp = _Reference(hf, _mtp_weights(path), dtype)
    rows = [topk_rows(mtp.logits(p.ids)[:-1], torch.tensor(p.ids[1:]), k) for p in prompts.prompts]
    meta = model_meta(path) | prompts.meta() | {"source": "hf", "head": "mtp", "dtype": str(dtype).removeprefix(
        "torch."), "hardware": hardware(torch.device("cpu")), "transformers": transformers.__version__}
    return Recording.from_rows(rows, meta)


@torch.no_grad()
def candidate(engine, prompts: PromptSet, k: int = 20) -> Recording:
    """The engine drafter's head logits over each prompt (a one-step prefill, no prefix cache), as a recording."""
    from holon.api import SamplingParams

    drafter = engine.executor.drafter
    seen: list[torch.Tensor] = []
    forward = drafter.head.forward

    def capture(embedded, h, ctx):
        out = forward(embedded, h, ctx)
        seen.append(out)
        return out

    drafter.head.forward = capture
    rows = []
    try:
        for p in prompts.prompts:
            seen.clear()
            engine.generate([p.ids], SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True))
            out = next(o for o in seen if o.shape[0] == len(p.ids))[:-1]     # the pass over the prompt itself
            target = torch.tensor(p.ids[1:], device=out.device)
            blocks = [topk_rows(drafter.lm_head(out[a:a + 256]).float(), target[a:a + 256], k)   # bounds [n, vocab]
                      for a in range(0, len(target), 256)]
            rows.append(tuple(torch.cat(t) for t in zip(*blocks, strict=True)))
    finally:
        drafter.head.forward = forward
    return Recording.from_rows(rows, prompts.meta() | {"source": "holon", "head": "mtp",
                                                       "hardware": hardware(engine.device)})


# DFlash2 -------------------------------------------------------------------------------------------------------------
# The draft's rows 1 … B − 1 for an anchor at P predict positions P + 1 … P + B − 1, given the target's residual
# streams at positions < P. Each prompt is cut at a few lengths P; the engine prefills the cut and drafts once (its
# first draft, anchored at the token it sampled), and the judge records, besides the draft logits, the target's
# streams after the checkpoint's own `target_layer_ids` — its own taps, so a drafter tapping elsewhere shows. The
# reference (the plugin's `reference.py`) runs on those streams with the checkpoint's weights: fp32 for the golden,
# the model's dtype for the floor — for GGUF weights (the draft's and the target's head) dequantized by gguf-py, and
# in the floor every quantized matmul rounded as llama.cpp's kernel path for its row count does (q8_1).

# llama.cpp's DFlash tensor names (conversion/qwen.py `DFlashModel`, gguf-py tensor_mapping) -> the checkpoint's.
_DFLASH_GLOBAL = {"fc.weight": "fc.weight", "enc.output_norm.weight": "hidden_norm.weight",
                  "output_norm.weight": "norm.weight",
                  "selector_predecessor.weight": "candidate_selector.predecessor_codebook",
                  "selector_successor.weight": "candidate_selector.successor_codebook",
                  "selector_hidden.weight": "candidate_selector.hidden_projection.weight"}
_DFLASH_LAYER = {"attn_norm.weight": "input_layernorm.weight", "ffn_norm.weight": "post_attention_layernorm.weight",
                 "attn_q.weight": "self_attn.q_proj.weight", "attn_k.weight": "self_attn.k_proj.weight",
                 "attn_v.weight": "self_attn.v_proj.weight", "attn_output.weight": "self_attn.o_proj.weight",
                 "attn_q_norm.weight": "self_attn.q_norm.weight", "attn_k_norm.weight": "self_attn.k_norm.weight",
                 "ffn_gate.weight": "mlp.gate_proj.weight", "ffn_up.weight": "mlp.up_proj.weight",
                 "ffn_down.weight": "mlp.down_proj.weight", "attn_conv_base": "attention_conv.base_kernel",
                 "attn_conv_proj.weight": "attention_conv.kernel_projection.weight",
                 "ffn_conv_base": "mlp_conv.base_kernel", "ffn_conv_proj.weight": "mlp_conv.kernel_projection.weight"}


def dflash_prompts(prompts: PromptSet, cuts: int = 4, block: int = 8, shortest: int = 16) -> PromptSet:
    """Each prompt cut at `cuts` lengths P, evenly from `shortest` to its length − `block`: a sequence of the context
    ids[:P] and the block's targets ids[P + 1 : P + block] (its first P + block ids)."""
    out = []
    for p in prompts.prompts:
        last = len(p.ids) - block
        for c in sorted({shortest + (last - shortest) * i // max(cuts - 1, 1) for i in range(cuts)} if last >= shortest
                        else set()):
            out.append(Prompt(f"{p.id}@{c}", p.category, p.ids[:c + block]))
    return PromptSet(f"{prompts.name}:dflash{cuts}", prompts.sha256, out)


def _draft_rows(logits: torch.Tensor, ids: list[int], k: int) -> tuple[torch.Tensor, ...]:
    """The draft's rows for one cut: logits of rows 1 … B − 1 against the cut's block targets."""
    block = len(logits) + 1
    return topk_rows(logits.float().cpu(), torch.tensor(ids[-block + 1:]), k)


@torch.no_grad()
def dflash_candidate(engine, prompts: PromptSet, k: int = 20) -> tuple[Recording, list[tuple[torch.Tensor, int]]]:
    """The engine drafter's logits for each cut (one prefill, one draft), and what the reference needs: the target's
    streams [P, taps · H] at the checkpoint's layers and the anchor, per cut."""
    from holon.api import SamplingParams

    drafter, model = engine.executor.drafter, engine.executor.model
    taps: dict[int, torch.Tensor] = {}
    layers = drafter.cfg["taps"]
    for n, layer in enumerate(layers):
        model.taps.setdefault(layer, []).append(lambda h, ctx, n=n: taps.__setitem__(n, h.to("cpu", copy=True)))
    seen: list[torch.Tensor] = []
    head = drafter.lm_head
    drafter.lm_head = lambda h: seen.append(out := head(h)) or out
    rows, inputs = [], []
    try:
        for p in prompts.prompts:
            seen.clear()
            context = p.ids[:len(p.ids) - drafter.block_size]
            anchor = engine.generate([context], SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True))[0][0]
            rows.append(_draft_rows(seen[-1], p.ids, k))
            inputs.append((torch.cat([taps[n][:len(context)] for n in range(len(layers))], -1), anchor))
    finally:
        drafter.lm_head = head
        for layer in layers:
            model.taps[layer].pop()
    return Recording.from_rows(rows, prompts.meta() | {"source": "holon", "head": "dflash",
                                                       "hardware": hardware(engine.device)}), inputs


def _dflash_state(path: str) -> dict[str, torch.Tensor]:
    """The draft checkpoint's tensors by its own (HF) names: safetensors as stored, a GGUF dequantized by gguf-py."""
    p = pathlib.Path(path)
    if p.suffix != ".gguf":
        out = {}
        for f in sorted(p.glob("*.safetensors")):
            with safe_open(str(f), "pt") as st:
                out |= {n: st.get_tensor(n) for n in st.keys()}
        return out
    return {_dflash_name(t.name): x for t, x in _dequantized(p)}


def _dflash_name(name: str) -> str:
    if name in _DFLASH_GLOBAL:
        return _DFLASH_GLOBAL[name]
    layer, _, part = name.removeprefix("blk.").partition(".")
    return f"layers.{layer}.{_DFLASH_LAYER[part]}"


def _dequantized(path: pathlib.Path):
    import gguf
    import numpy as np

    for t in gguf.GGUFReader(path).tensors:
        x = gguf.quants.dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32, copy=False)
        yield t, torch.from_numpy(x.reshape([int(n) for n in reversed(t.shape)]).copy())


def _target_rows(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The target's embedding and output head (fp32), from its checkpoint (a GGUF dequantized by gguf-py)."""
    p = pathlib.Path(path)
    if p.suffix == ".gguf":
        import gguf
        import numpy as np

        found = {}
        for t in gguf.GGUFReader(p).tensors:
            if t.name in ("token_embd.weight", "output.weight"):
                x = gguf.quants.dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32, copy=False)
                found[t.name] = torch.from_numpy(x.reshape([int(n) for n in reversed(t.shape)]).copy())
        embed = found["token_embd.weight"]
        return embed, found.get("output.weight", embed)
    found = {}
    for f in sorted(p.glob("*.safetensors")):
        with safe_open(str(f), "pt") as st:
            for n in st.keys():
                if n.endswith(("embed_tokens.weight", "lm_head.weight")) and "mtp" not in n:
                    found["head" if n.endswith("lm_head.weight") else "embed"] = st.get_tensor(n).float()
    return found["embed"], found.get("head", found["embed"])


def _emulation(draft: str, target: str):
    """The floor's matmul for GGUF weights: llama.cpp's q8_1 rounding on the kernel path its row count selects."""
    import gguf
    import numpy as np
    from holon_plugins.kernels.ggml_cuda import extension, plugin

    from . import q8_1

    major, minor = torch.cuda.get_device_capability()
    lib, index = extension.load(10 * major + minor), torch.cuda.current_device()
    named = []                                   # (reference name, GGUF tensor) of every quantized matmul weight
    if draft.endswith(".gguf"):
        codebooks = ("selector_predecessor.weight", "selector_successor.weight")          # read by get_rows: exact
        named += [(_dflash_name(t.name), t) for t in gguf.GGUFReader(draft).tensors if t.name not in codebooks]
    if target.endswith(".gguf"):
        named += [("lm_head", t) for t in gguf.GGUFReader(target).tensors if t.name == "output.weight"]
    kinds = {}
    for name, t in named:
        kind = t.tensor_type.name.lower()
        if kind in ("f32", "f16", "bf16"):
            continue
        limit = max((n for n in range(1, 9) if lib.use_mmvq(index, plugin.TYPES[kind], n)), default=0)
        blocks = np.asarray(t.data)
        offs = {route: q8_1.offsets(blocks, kind, q8_1.blocking(kind, route).sum_group).float()
                for route in (q8_1.MMVQ, q8_1.MMQ) if q8_1.blocking(kind, route).sum_group}
        kinds[name] = (kind, limit, offs)

    def matmul(name: str, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if name not in kinds:
            return x @ w.T
        kind, limit, offs = kinds[name]
        route = q8_1.MMVQ if len(x) <= limit else q8_1.MMQ
        return q8_1.matmul(x.float(), w.float(), None, kind, route, offs.get(route), torch.float32)
    return matmul


def dflash_reference(draft: str, target: str, prompts: PromptSet, inputs: list[tuple[torch.Tensor, int]],
                     dtype: torch.dtype = torch.float32, k: int = 20, q8_1: bool = False) -> Recording:
    """The published forward's draft logits for each cut, on the recorded streams (CPU, `dtype`); `q8_1`: GGUF
    matmuls rounded as llama.cpp's kernels do (the floor; needs the GPU to ask which path a row count takes)."""
    from holon.core.assembly import open_source
    from holon.core.plugins import PluginSet, discover
    from holon_plugins.drafters.dflash.plugin import checkpoint
    from holon_plugins.drafters.dflash.reference import Reference

    cfg = checkpoint(open_source(PluginSet(discover()), draft))
    embed, head = _target_rows(target)
    ref = Reference(_dflash_state(draft), cfg, embed, head, dtype,
                    _emulation(draft, target) if q8_1 else None)
    rows = [_draft_rows(ref.logits(ref.block(features, anchor)), p.ids, k)
            for p, (features, anchor) in zip(prompts.prompts, inputs, strict=True)]
    return Recording.from_rows(rows, prompts.meta() | {"source": "reference", "head": "dflash", "dtype": str(
        dtype).removeprefix("torch."), "q8_1": "by rows" if q8_1 else None, "hardware": hardware(torch.device("cpu"))})
