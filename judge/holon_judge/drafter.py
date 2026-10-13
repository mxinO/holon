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
from .recording import PromptSet, Recording


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
