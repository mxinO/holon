"""GLM-5.3-Flash (`glm5_next`) in the golden recorder (docs/v1-glm.md D2): what transformers' own model needs beyond its
forward, through `archs`' hooks.

- **A causal LM over the text model**: transformers 5.19.0 ships only the multimodal `Glm5NextForConditionalGeneration`;
  `Glm5NextForCausalLM` is its text model and head, the same modules (`register` makes it the text config's
  `AutoModelForCausalLM`).
- **The tie rule** (D2's documented deviation from HF): the DSA indexer's `Tensor.topk` and the router's `torch.topk`
  are a stable descending sort for the duration of their own module's call, so equal scores go to the lower pool and
  the lower expert (`stable_topk`, std.mixers.qsa's reference: one definition). The indexer's ReLU sums make exact
  0.0 ties common; HF leaves their order unspecified.
- **The layer's own output**: a decoder layer returns `(streams, topk_indices)`; the streamed driver's stand-in and its
  per-layer step follow (`stand_in`, `streams`).
- **The final norm** is the stream mean, then the model's RMSNorm (`final_norm`)."""
from __future__ import annotations

import pathlib
from collections.abc import Callable

import torch
from holon_std.mixers.qsa.reference import stable_topk

TIE_RULE = "dsa indexer and moe router: stable descending top-k, ties to the lower pool and the lower expert"


def _causal_lm():
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

    return Glm5NextForCausalLM


CausalLM = None


def register() -> type:
    """`Glm5NextForCausalLM` as `glm5_next_text`'s `AutoModelForCausalLM` (once); returns the class."""
    global CausalLM
    if CausalLM is None:
        import transformers
        from transformers.models.glm5_next import Glm5NextTextConfig

        CausalLM = _causal_lm()
        transformers.AutoModelForCausalLM.register(Glm5NextTextConfig, CausalLM, exist_ok=True)
    return CausalLM


def _scoped(forward, owner, name: str, topk) -> Callable:
    def call(*args, **kwargs):
        saved = getattr(owner, name)
        setattr(owner, name, topk)
        try:
            return forward(*args, **kwargs)
        finally:
            setattr(owner, name, saved)
    return call


def wrap_topk(model: torch.nn.Module, on_tie: Callable[[int], None] | None = None) -> int:
    """Give every DSA indexer and MoE router of `model` the tie rule, scoped to its own call; `on_tie(n)` (optional)
    hears, per indexer selection, how many query rows had their k-th and (k+1)-th scores equal. Returns the number of
    modules wrapped."""
    def indexer_topk(x: torch.Tensor, k: int, dim: int = -1, largest: bool = True, sorted: bool = True):
        if on_tie is not None and k < x.shape[dim]:
            values = torch.sort(x, dim=dim, descending=largest, stable=True).values
            tied = (values.narrow(dim, k - 1, 1) == values.narrow(dim, k, 1)) & (values.narrow(dim, k, 1) >
                                                                                 torch.finfo(x.dtype).min)
            on_tie(int(tied.sum()))
        return stable_topk(x, k, dim, largest, sorted)

    def router_topk(x: torch.Tensor, k: int, dim: int = -1, largest: bool = True, sorted: bool = True):
        return stable_topk(x, k, dim, largest, sorted)

    n = 0
    for m in model.modules():
        if type(m).__name__ == "Glm5NextTextIndexer":
            m.forward, n = _scoped(m.forward, torch.Tensor, "topk", indexer_topk), n + 1
        elif type(m).__name__ == "Glm5NextTextTopkRouter":
            m.forward, n = _scoped(m.forward, torch, "topk", router_topk), n + 1
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
