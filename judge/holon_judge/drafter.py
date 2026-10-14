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


def cut_prompts(prompts: PromptSet, cuts: int = 4, block: int = 8, shortest: int = 16) -> PromptSet:
    """Each prompt cut at `cuts` lengths P, evenly from `shortest` to its length − `block`: a sequence of the context
    ids[:P] and a draft's targets ids[P + 1 : P + block] (its first P + block ids; `block` − 1 drafts)."""
    out = []
    for p in prompts.prompts:
        last = len(p.ids) - block
        for c in sorted({shortest + (last - shortest) * i // max(cuts - 1, 1) for i in range(cuts)} if last >= shortest
                        else set()):
            out.append(Prompt(f"{p.id}@{c}", p.category, p.ids[:c + block]))
    return PromptSet(f"{prompts.name}:cuts{cuts}x{block}", prompts.sha256, out)


def _draft_rows(logits: torch.Tensor, ids: list[int], k: int) -> tuple[torch.Tensor, ...]:
    """A draft's rows for one cut (`cut_prompts`): its logits [n, V] against the cut's last n targets."""
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
    import gguf

    return {_dflash_name(t.name): x for t, x in _dequantized(gguf.GGUFReader(p))}


def _dflash_name(name: str) -> str:
    if name in _DFLASH_GLOBAL:
        return _DFLASH_GLOBAL[name]
    layer, _, part = name.removeprefix("blk.").partition(".")
    return f"layers.{layer}.{_DFLASH_LAYER[part]}"


def _dequantized(reader, names=None):
    """(tensor, fp32 values) of a GGUF reader's tensors (`names`: which), dequantized by gguf-py."""
    import gguf
    import numpy as np

    for t in reader.tensors:
        if names is not None and not names(t.name):
            continue
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
    """The DFlash floor's matmul: the draft's GGUF weights (not the codebooks, read by get_rows: exact) and the
    target's GGUF head."""
    import gguf

    codebooks = ("selector_predecessor.weight", "selector_successor.weight")
    named = [(_dflash_name(t.name), t) for t in gguf.GGUFReader(draft).tensors if t.name not in codebooks] \
        if draft.endswith(".gguf") else []
    if target.endswith(".gguf"):
        named += [("lm_head", t) for t in gguf.GGUFReader(target).tensors if t.name == "output.weight"]
    return _q8_1_matmul(named)


def _q8_1_matmul(named: list):
    """A reference's matmul for the GGUF tensors `named` [(reference name, tensor)]: llama.cpp's q8_1 rounding on the
    kernel path its row count selects (the ggml plugin's own `mmvq_rows`, for this GPU's class); dense ones exact."""
    import numpy as np
    from holon.core import platform
    from holon_plugins.kernels.ggml_cuda import extension, plugin

    from . import q8_1

    major, minor = torch.cuda.get_device_capability()
    lib, fp = extension.load(10 * major + minor), platform.fingerprint(torch.device("cuda"))
    kinds = {}
    for name, t in named:
        kind = t.tensor_type.name.lower()
        if kind in ("f32", "f16", "bf16"):
            continue
        limit = plugin.mmvq_rows(fp, kind, "q8_1", int(t.shape[0]), int(t.shape[1]), lib)
        blocks = np.asarray(t.data)
        offs = {route: q8_1.offsets(blocks, kind, q8_1.blocking(kind, route).sum_group).float().cuda()
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
    ref = Reference(_dflash_state(draft), cfg, embed, head, dtype, _emulation(draft, target) if q8_1 else None,
                    "cuda" if torch.cuda.is_available() else "cpu")
    rows = [_draft_rows(ref.logits(ref.block(features, anchor)), p.ids, k)
            for p, (features, anchor) in zip(prompts.prompts, inputs, strict=True)]
    return Recording.from_rows(rows, prompts.meta() | {"source": "reference", "head": "dflash", "dtype": str(
        dtype).removeprefix("torch."), "q8_1": "by rows" if q8_1 else None, "hardware": hardware(ref.device)})



# MTP from a GGUF ------------------------------------------------------------------------------------------------------
# The Qwen3.5 / 3.8 head stored with the target (`blk.<n>.*`), judged per cut like DFlash: the engine prefills the cut
# (the head's pass over its tokens) and chains k drafts; the judge records the chain's logits and, with its own tap
# after the target's last layer, the stream whose final norm the head reads. The reference is the published forward
# (research note §M.2) in plain torch on the file's dequantized tensors, its chain teacher-forced with the
# candidate's drafts: step 1 takes the anchor and h(P − 1), step j the draft j − 1 and step j − 1's output. In the
# floor, the prompt's rows run as one matmul (mmq) and each chain step's single row as the engine's do (mmvq).

def mtp_candidate(engine, prompts: PromptSet, steps: int, k: int = 20
                  ) -> tuple[Recording, list[tuple[torch.Tensor, int, torch.Tensor]]]:
    """The engine's MTP chain logits for each cut (`cut_prompts(block=steps + 1)`, the policy drafting `steps`), and
    per cut the target's last-layer stream [P, H], the anchor and the drafts."""
    from holon.api import SamplingParams

    drafter, model = engine.executor.drafter, engine.executor.model
    last, stream = len(model.layers) - 1, {}
    model.taps.setdefault(last, []).append(lambda h, ctx: stream.__setitem__(0, h.to("cpu", copy=True)))
    seen: list[torch.Tensor] = []
    head = drafter.lm_head
    drafter.lm_head = lambda h: seen.append(out := head(h)) or out
    rows, inputs = [], []
    try:
        with torch.no_grad():
            for p in prompts.prompts:
                seen.clear()
                context = p.ids[:len(p.ids) - steps - 1]
                anchor = engine.generate([context], SamplingParams(temperature=0.0, max_tokens=1,
                                                                   ignore_eos=True))[0][0]
                logits = torch.cat(seen).float()
                rows.append(_draft_rows(logits, p.ids, k))
                inputs.append((stream[0][:len(context)], anchor, logits.argmax(-1).cpu()))
    finally:
        drafter.lm_head = head
        model.taps[last].pop()
    return Recording.from_rows(rows, prompts.meta() | {"source": "holon", "head": "mtp",
                                                       "hardware": hardware(engine.device)}), inputs


class MTPReference:
    """The MTP block of a Qwen3.5 / 3.8 GGUF in plain torch (research note §M.2): x = eh_proj([enorm(embed t);
    hnorm(h)]), one gated full-attention layer with a SwiGLU MLP, the head's norm, the target's output head. The
    converter stores every norm here as w + 1 (Gemma-style), so they multiply by the stored weight. `dtype`: the
    arithmetic; `q8_1`: the floor's rounding of quantized matmuls (needs the GPU, for the kernels' row limits)."""

    def __init__(self, path: str, dtype: torch.dtype = torch.float32, q8_1: bool = False, reader=None) -> None:
        """`reader`: the file's `gguf.GGUFReader`, if one is open (reading its metadata takes seconds)."""
        import gguf

        reader = reader or gguf.GGUFReader(path)
        value = lambda key: reader.fields[f"qwen35.{key}"].parts[-1][0]                      # noqa: E731
        n = int(value("block_count")) - int(value("nextn_predict_layers"))
        self.heads, self.kv = int(value("attention.head_count")), int(value("attention.head_count_kv"))
        self.dim, self.rot = int(value("attention.key_length")), int(value("rope.dimension_count"))
        self.theta, self.eps = float(value("rope.freq_base")), float(value("attention.layer_norm_rms_epsilon"))
        own = lambda name: name.startswith(f"blk.{n}.")                                       # noqa: E731
        local = lambda name: name.removeprefix(f"blk.{n}.").removeprefix("nextn.")           # noqa: E731
        shared = ("token_embd.weight", "output.weight", "output_norm.weight")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.w = {(local(t.name) if own(t.name) else t.name): x.to(self.device, dtype)
                  for t, x in _dequantized(reader, lambda name: own(name) or name in shared)}
        self.w.setdefault("output.weight", self.w["token_embd.weight"])
        self.dtype, self.q8_1 = dtype, q8_1
        self.matmul = _q8_1_matmul([(local(t.name), t) for t in reader.tensors if own(t.name)] + [
            ("lm_head", t) for t in reader.tensors if t.name == "output.weight"]) if q8_1 else (
            lambda name, x, w: x @ w.T)

    def _linear(self, name: str, x: torch.Tensor) -> torch.Tensor:
        return self.matmul(name, x, self.w[name]).to(self.dtype)

    def _norm(self, name: str, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps) * self.w[name].float()).to(self.dtype)

    def _rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        inv = 1.0 / self.theta ** (torch.arange(0, self.rot, 2, dtype=torch.float64) / self.rot)
        angle = positions.double()[:, None] * inv[None]
        cos, sin = angle.cos()[:, None].float(), angle.sin()[:, None].float()
        r, rest = x[..., :self.rot].float(), x[..., self.rot:]
        a, b = r.chunk(2, -1)
        return torch.cat([torch.cat([a * cos - b * sin, b * cos + a * sin], -1).to(self.dtype), rest], -1)

    def rows(self, tokens: torch.Tensor, hidden: torch.Tensor, cache: list[torch.Tensor]) -> torch.Tensor:
        """The head's output for rows at positions len(cache[0]) … (token, its predecessor's final-norm hidden),
        attending to the cached rows before them (the cache grows)."""
        start, t = (len(cache[0]) if cache else 0), len(tokens)
        positions = torch.arange(start, start + t)
        e = self._norm("enorm.weight", self.w["token_embd.weight"][tokens])
        x = self._linear("eh_proj.weight", torch.cat([e, self._norm("hnorm.weight", hidden.to(self.dtype))], -1))
        a = self._norm("attn_norm.weight", x)
        qg = self._linear("attn_q.weight", a).view(t, self.heads, 2 * self.dim)
        q, gate = qg[..., :self.dim], qg[..., self.dim:]
        k = self._linear("attn_k.weight", a).view(t, self.kv, self.dim)
        v = self._linear("attn_v.weight", a).view(t, self.kv, self.dim)
        q = self._rope(self._norm("attn_q_norm.weight", q), positions)
        k = self._rope(self._norm("attn_k_norm.weight", k), positions)
        if not cache:
            cache += [k[:0], v[:0]]
        cache[0], cache[1] = torch.cat([cache[0], k]), torch.cat([cache[1], v])
        keys, values = (c.repeat_interleave(self.heads // self.kv, 1).float() for c in cache)
        scores = torch.einsum("qhd,khd->hqk", q.float(), keys) / self.dim ** 0.5
        hidden_k = torch.arange(len(keys))[None] > positions[:, None]
        o = torch.einsum("hqk,khd->qhd", scores.masked_fill(hidden_k, float("-inf")).softmax(-1), values)
        o = (o * torch.sigmoid(gate.float())).to(self.dtype).reshape(t, -1)
        x = x + self._linear("attn_output.weight", o)
        b = self._norm("post_attention_norm.weight", x)
        m = (torch.nn.functional.silu(self._linear("ffn_gate.weight", b).float())
             * self._linear("ffn_up.weight", b).float()).to(self.dtype)
        return self._norm("shared_head_norm.weight", x + self._linear("ffn_down.weight", m))

    def chain(self, stream: torch.Tensor, context: list[int], anchor: int, drafts: torch.Tensor) -> torch.Tensor:
        """The chain's logits [k, V]: the prompt's rows (token p, h(p − 1)) in one pass, then step 1 (anchor, h(P −
        1)) and each step j (draft j − 1, step j − 1's output) one row at a time."""
        with torch.device(self.device):
            return self._chain(stream.to(self.device), context, anchor, drafts)

    def _chain(self, stream: torch.Tensor, context: list[int], anchor: int, drafts: torch.Tensor) -> torch.Tensor:
        h = self._norm("output_norm.weight", stream.to(self.dtype))                 # the target's final-norm hidden
        cache: list[torch.Tensor] = []
        self.rows(torch.tensor(context), torch.cat([torch.zeros_like(h[:1]), h[:-1]]), cache)
        x, out = h[-1:], []
        for token in [anchor, *drafts[:-1].tolist()]:
            x = self.rows(torch.tensor([token]), x, cache)
            out.append(self.matmul("lm_head", x, self.w["output.weight"]).float())
        return torch.cat(out)


    def record(self, prompts: PromptSet, inputs: list[tuple[torch.Tensor, int, torch.Tensor]], k: int = 20
               ) -> Recording:
        """The chain's logits for each cut on the recorded streams (CPU)."""
        rows = [_draft_rows(self.chain(stream, p.ids[:len(stream)], anchor, drafts), p.ids, k)
                for p, (stream, anchor, drafts) in zip(prompts.prompts, inputs, strict=True)]
        return Recording.from_rows(rows, prompts.meta() | {"source": "reference", "head": "mtp", "dtype": str(
            self.dtype).removeprefix("torch."), "q8_1": "by rows" if self.q8_1 else None,
            "hardware": hardware(self.device)})
