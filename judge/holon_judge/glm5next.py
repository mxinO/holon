"""GLM-5.3-Flash (`glm5_next`) in the golden recorder (docs/v1-glm.md D2): what transformers' own model needs beyond its
forward, through `archs`' hooks, and its GGUF's names (`archs.GGUF_ARCHS`).

- **A causal LM over the text model**: transformers 5.19.0 ships only the multimodal `Glm5NextForConditionalGeneration`;
  `Glm5NextForCausalLM` is its text model and head, the same modules (`register` makes it the text config's
  `AutoModelForCausalLM`).
- **The tie rule** (D2's documented deviation from HF): the DSA indexer's `Tensor.topk` and the router's `torch.topk`
  are a stable descending sort for the duration of their own module's call (`ties.scoped`), so equal scores go to the
  lower pool and the lower expert (`stable_topk`, std.mixers.qsa's reference: one definition). The indexer's ReLU sums
  make exact 0.0 ties common; HF leaves their order unspecified.
- **The layer's own output**: a decoder layer returns `(streams, topk_indices)`; the streamed driver's stand-in and its
  per-layer step follow (`stand_in`, `streams`).
- **The final norm** is the stream mean, then the model's RMSNorm (`final_norm`).
- **The GGUF** (`glm5-next`, llama.cpp 27c7a76 conversion/glm.py): the mHC tensors, KDA's (`ssm_*`; `attn_q` / `k` /
  `v` / `output` as the generic names), the MLA and DSA indexer tensors, the shared expert and the selection bias
  (`GGUF_NAMES`); `gguf_tensor` undoes the conversion's value changes and splits (`ssm_a`, the KDA convs, MLA's
  `kv_b_proj`)."""
from __future__ import annotations

import functools
import pathlib
import re
from collections.abc import Callable

import torch
from holon_std.mixers.qsa.reference import stable_topk

from .ties import scoped

TIE_RULE = "dsa indexer and moe router: stable descending top-k, ties to the lower pool and the lower expert"


@functools.cache
def register() -> type:
    """`Glm5NextForCausalLM`, registered as `glm5_next_text`'s `AutoModelForCausalLM` (once)."""
    import transformers
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.models.glm5_next import modeling_glm5_next as hf

    class Glm5NextForCausalLM(hf.Glm5NextPreTrainedModel):
        """The text model and its head (what `Glm5NextForConditionalGeneration` runs on text)."""

        config_class = hf.Glm5NextTextConfig

        def __init__(self, config):
            super().__init__(config)
            self.model = hf.Glm5NextTextModel(config)
            self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.post_init()

        def forward(self, input_ids: torch.Tensor | None = None, **kwargs) -> CausalLMOutputWithPast:
            out = self.model(input_ids, **kwargs)
            return CausalLMOutputWithPast(logits=self.lm_head(out.last_hidden_state),
                                          past_key_values=out.past_key_values)

    transformers.AutoModelForCausalLM.register(hf.Glm5NextTextConfig, Glm5NextForCausalLM, exist_ok=True)
    return Glm5NextForCausalLM


def _ties(x: torch.Tensor, k: int, dim: int, largest: bool) -> int:
    """How many rows have their k-th and (k+1)-th values equal (not both masked at the dtype's minimum)."""
    if k >= x.shape[dim]:
        return 0
    values = torch.sort(x, dim=dim, descending=largest, stable=True).values
    kth, next_ = values.narrow(dim, k - 1, 1), values.narrow(dim, k, 1)
    return int(((kth == next_) & (next_ > torch.finfo(x.dtype).min)).sum())


def wrap_topk(model: torch.nn.Module, on_tie: Callable[[int], None] | None = None,
              on_router_tie: Callable[[int], None] | None = None) -> int:
    """Give every DSA indexer and MoE router of `model` the tie rule, scoped to its own call; `on_tie(n)` /
    `on_router_tie(n)` (optional) hear, per selection, how many rows had their k-th and (k+1)-th scores equal (the
    router's by its selection score, score + bias). Returns the number of modules wrapped."""
    def counted(hear):
        def topk(x: torch.Tensor, k: int, dim: int = -1, largest: bool = True, sorted: bool = True):
            if hear is not None:
                hear(_ties(x, k, dim, largest))
            return stable_topk(x, k, dim, largest, sorted)
        return topk

    n = 0
    for m in model.modules():
        if type(m).__name__ == "Glm5NextTextIndexer":
            m.forward, n = scoped(m.forward, torch.Tensor, "topk", counted(on_tie)), n + 1
        elif type(m).__name__ == "Glm5NextTextTopkRouter":
            m.forward, n = scoped(m.forward, torch, "topk", counted(on_router_tie)), n + 1
    return n


def prepare(model: torch.nn.Module, path: pathlib.Path | None, device: str, dtype: torch.dtype,
            on_tie: Callable[[int], None] | None = None) -> dict:
    """The tie rule on the indexers and routers. Every indexer must run its own selection (`indexer_types` all
    "full", as GLM-5.3-Flash's config: the streamed driver runs one layer at a time). Returns the golden metadata."""
    if any(t != "full" for t in model.config.indexer_types):
        raise ValueError("glm5_next: a shared-indexer layer reads the previous layer's selection; the golden driver "
                         "supports indexer_types all 'full'")
    return {"deviations": [TIE_RULE]} if wrap_topk(model, on_tie) else {}


def final_norm(model: torch.nn.Module) -> torch.nn.Module:
    """The stream mean, then the RMSNorm."""
    return torch.nn.Sequential(model.model.hc_head, model.model.norm)


def stand_in(hidden: torch.Tensor) -> tuple:
    """What a decoder layer returns: the streams and its selection (None: the next layer runs its own)."""
    return hidden, None


def streams(out: tuple) -> torch.Tensor:
    """A decoder layer's output streams."""
    return out[0]


# The GGUF's layer tensors (`blk.<i>.<part>`) that are not the generic ones, by part -> HF parameter; and those the
# conversion split or reshaped, fused back by `gguf_tensor` (part -> (what they make, their place in it)).
GGUF_NAMES = {f"hc_{g}_{p}.weight": f"{g}_hc.{p}" for g in ("attn", "ffn") for p in ("fn", "base", "scale")} | {
    "ssm_a": "self_attn.forget_gate.A_log", "ssm_dt.bias": "self_attn.forget_gate.dt_bias",
    "ssm_f_a.weight": "self_attn.forget_gate.f_a_proj.weight",
    "ssm_f_b.weight": "self_attn.forget_gate.f_b_proj.weight", "ssm_beta.weight": "self_attn.b_proj.weight",
    "ssm_g_a.weight": "self_attn.g_a_proj.weight", "ssm_g_b.weight": "self_attn.g_b_proj.weight",
    "ssm_norm.weight": "self_attn.o_norm.weight",
    "attn_q_a.weight": "self_attn.q_a_proj.weight", "attn_q_a_norm.weight": "self_attn.q_a_layernorm.weight",
    "attn_q_b.weight": "self_attn.q_b_proj.weight", "attn_kv_a_mqa.weight": "self_attn.kv_a_proj_with_mqa.weight",
    "attn_kv_a_norm.weight": "self_attn.kv_a_layernorm.weight",
    "indexer.attn_q_b.weight": "self_attn.indexer.wq_b.weight", "indexer.attn_k.weight": "self_attn.indexer.wk.weight",
    "indexer.k_norm.weight": "self_attn.indexer.k_norm.weight", "indexer.k_norm.bias": "self_attn.indexer.k_norm.bias",
    "indexer.proj.weight": "self_attn.indexer.weights_proj.weight",
    "indexer_compressor_ape.weight": "self_attn.indexer.index_kpool_compress_ape",
    "indexer_compressor_gate.weight": "self_attn.indexer.index_kpool_compress_gate",
    "exp_probs_b.bias": "mlp.gate.e_score_correction_bias",
    "ffn_gate_shexp.weight": "mlp.shared_experts.gate_proj.weight",
    "ffn_up_shexp.weight": "mlp.shared_experts.up_proj.weight",
    "ffn_down_shexp.weight": "mlp.shared_experts.down_proj.weight"}
GGUF_FUSED = {"ssm_conv1d_q.weight": ("conv", 0), "ssm_conv1d_k.weight": ("conv", 1),
              "ssm_conv1d_v.weight": ("conv", 2), "attn_k_b.weight": ("mla", 0), "attn_v_b.weight": ("mla", 1)}


def gguf_tensor(name: str, x: torch.Tensor, pending: dict) -> list[tuple[str, torch.Tensor]] | None:
    """Undo the conversion where it changed values or split a tensor (None: the tensor maps by name): `ssm_a` is
    −exp(A_log); the q / k / v convs ([1, C, 1, 4] each) are transformers' one depthwise `conv1d` [3C, 1, 4]; MLA's
    `kv_b_proj` [heads·(nope + v), kv_lora] was split into `k_b` (transposed, [heads, kv_lora, nope]) and `v_b`
    ([heads, v, kv_lora]). A fused tensor comes out once its parts are in (`pending`)."""
    m = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
    if m is None:
        return None
    if m[2] == "ssm_a":
        return [(f"model.layers.{m[1]}.{GGUF_NAMES['ssm_a']}", torch.log(-x))]
    if m[2] not in GGUF_FUSED:
        return None
    kind, at = GGUF_FUSED[m[2]]
    parts = pending.setdefault(f"{m[1]}.{kind}", {})
    parts[at] = x
    if len(parts) < (3 if kind == "conv" else 2):
        return []
    del pending[f"{m[1]}.{kind}"]
    if kind == "conv":
        return [(f"model.layers.{m[1]}.self_attn.conv1d.weight",
                 torch.cat([parts[i].reshape(-1, 1, parts[i].shape[-1]) for i in range(3)]))]
    k_b, v_b = parts[0].transpose(1, 2), parts[1]                      # [heads, nope, kv_lora], [heads, v, kv_lora]
    kv_b = torch.cat([k_b, v_b], dim=1)
    return [(f"model.layers.{m[1]}.self_attn.kv_b_proj.weight", kv_b.reshape(-1, kv_b.shape[-1]).contiguous())]
