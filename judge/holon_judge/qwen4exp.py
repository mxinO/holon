"""Qwen3.8-Flash-Next (`qwen4_exp`) in the golden recorder (docs/v1-flash-next.md D2): what transformers' own model
needs beyond its forward, through `archs`' hooks.

- **The tie rule** (D2's documented deviation from HF): QSA's indexer breaks score ties toward the lower block. HF's
  `topk` leaves the order unspecified, and its ReLU sums make exact 0.0 ties common. One definition here; the fixture
  tool (`holon_testing.qwen4exp_files`) uses it too. Only an indexer's own call sees it: `Tensor.topk` is stable for
  the duration of that call, so the router's and the recording's top-k keep torch's.
- **PLE's table read lazily**: the whole table in fp32 is 205 GB, so a GGUF golden dequantizes only the hashed rows
  (`gguf.PleRows`), in the golden's dtype, after checking the file's hash constants against the ones HF computes.
- **The final norm** is the hyper-connection output mixer.
- **The MTP head's pooled reference** (D2's MTP oracle, D9): SGLang 0.5.21's `Qwen4ExpForCausalLMMTP`
  (`srt/models/qwen4_exp_mtp.py:91-112`) transcribed onto transformers' modules — `R_s = fc_embedding(enorm(e)) +
  fc_hidden(hnorm(h))_s` with hnorm one RMS over all streams, then one QSA + MoE layer without PLE and its output
  mixer — over a Strata-format MTP pack's own dequantized tensors (`pack_state`: HF names, fused `gate_up_proj`, BF16
  dense parts, F32 norms stored as w, Q2_0 experts)."""
from __future__ import annotations

import copy
import pathlib
from collections.abc import Callable

import torch

TIE_RULE = "qsa indexer: stable top-k, ties to the lower block"


def stable_topk(x: torch.Tensor, k: int, dim: int = -1, largest: bool = True, sorted: bool = True):
    """The tie rule: a stable sort, so equal scores keep their order (the lower block first)."""
    values, indices = torch.sort(x, dim=dim, descending=largest, stable=True)
    return torch.return_types.topk((values.narrow(dim, 0, k), indices.narrow(dim, 0, k)))


def wrap_indexers(model: torch.nn.Module, on_tie: Callable[[bool], None] | None = None) -> int:
    """Give every QSA indexer of `model` the tie rule, scoped to its own call; `on_tie(tied)` (optional) hears, per
    sparse selection, whether the k-th and (k+1)-th scores were equal. Returns the number of indexers wrapped."""
    def topk(x: torch.Tensor, k: int, dim: int = -1, largest: bool = True, sorted: bool = True):
        if on_tie is not None and x.dim() == 1 and k < x.numel():
            values = torch.sort(x, descending=largest, stable=True).values
            on_tie(bool(values[k - 1] == values[k]))
        return stable_topk(x, k, dim, largest, sorted)

    def scoped(forward):
        def call(*args, **kwargs):
            torch_topk, torch.Tensor.topk = torch.Tensor.topk, topk
            try:
                return forward(*args, **kwargs)
            finally:
                torch.Tensor.topk = torch_topk
        return call
    indexers = [m for m in model.modules() if type(m).__name__ == "Qwen4ExpTextQSAIndexer"]
    for m in indexers:
        m.forward = scoped(m.forward)
    return len(indexers)


class LazyRows(torch.nn.Module):
    """PLE's n-gram table in a streamed golden: the hashed rows dequantized from the GGUF on demand, in `dtype`."""

    def __init__(self, rows, device: str, dtype: torch.dtype):
        super().__init__()
        self.table, self.weight = rows, torch.empty(0, device=device, dtype=dtype)    # HF reads `.weight.device`

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.table.rows(ids).to(self.weight.device, self.weight.dtype)


def prepare(model: torch.nn.Module, path: pathlib.Path | None, device: str, dtype: torch.dtype,
            on_tie: Callable[[bool], None] | None = None) -> dict:
    """The tie rule on the indexers (`on_tie`: `wrap_indexers`'); with a GGUF `path`, PLE's table read lazily (its hash
    constants checked against HF's). Returns the golden metadata these changes add."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as hf

    meta = {"deviations": [TIE_RULE]} if wrap_indexers(model, on_tie) else {}
    embeddings = [m for m in model.modules() if isinstance(m, hf.Qwen4ExpTextNGramEmbedding)]
    if path is None or not embeddings:
        return meta
    from . import gguf

    rows, buffers = gguf.PleRows(path), gguf.ple_buffers(path)
    for m in embeddings:
        mine = {"layer_multipliers": hf._build_layer_multipliers(m.unigram_vocab_size, m.ngram_size,
                                                                 m.ple_layer_index, m.seed),
                "ngram_heads_offsets": torch.tensor(m.head_offsets),
                "ngram_heads_vocab_sizes": torch.tensor(m.head_vocab_sizes)}
        theirs = {n.rsplit(".", 1)[1]: t for n, t in buffers.items() if f".{m.layer_idx}.ple." in n}
        if any(not torch.equal(mine[n], theirs[n]) for n in mine):
            raise ValueError(f"{path.name}: PLE hash constants of layer {m.layer_idx} differ from HF's")
        m.ngram_embedding = LazyRows(rows, device, dtype)
    return meta | {"ple": "lazy rows"}


def final_norm(model: torch.nn.Module) -> torch.nn.Module:
    return model.model.hyper_connection_mixer


# ---- the MTP head's pooled reference ----

def pack_state(path: str | pathlib.Path) -> dict[str, torch.Tensor]:
    """A Strata-format MTP pack's tensors (fp32, HF names, as stored: norms are w, not 1 + w), dequantized by
    gguf-py."""
    import gguf as gguf_py
    import numpy as np

    out = {}
    for t in gguf_py.GGUFReader(path).tensors:
        x = gguf_py.quants.dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32, copy=False)
        out[t.name] = torch.from_numpy(x.reshape([int(n) for n in reversed(t.shape)]).copy())
    return out


@torch.inference_mode()
def mtp_pooled(state: dict[str, torch.Tensor], config, embeds: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    """The MTP head's output-mixer x [T, H] for token embeddings `embeds` [T, H] and the trunk's residual `hidden`
    [T, n·H] (before its output mixer), positions 0..T-1, over `state` (the pack's names, `mtp.*`). SGLang's form:
    `pre_fc_norm_hidden` is one RMS over the n·H features (pooled), then `fc_hidden` per stream."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as hf

    c = copy.deepcopy(config)
    c.num_hidden_layers, c.layer_types, c.ple_layer_ids = 1, ["indexed_attention"], []
    n, h = c.hc_count, c.hidden_size

    def load(module: torch.nn.Module, prefix: str) -> torch.nn.Module:
        module.load_state_dict({k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)},
                               strict=True)
        return module.eval()
    enorm = load(hf.Qwen4ExpTextRMSNorm(h, eps=c.rms_norm_eps), "mtp.pre_fc_norm_embedding.")
    hnorm = load(hf.Qwen4ExpTextRMSNorm(n * h, eps=c.rms_norm_eps), "mtp.pre_fc_norm_hidden.")
    fc_e = load(torch.nn.Linear(h, h, bias=False), "mtp.fc_embedding.")
    fc_h = load(torch.nn.Linear(h, h, bias=False), "mtp.fc_hidden.")
    r = (fc_e(enorm(embeds))[:, None, :] + fc_h(hnorm(hidden).view(-1, n, h))).reshape(1, -1, n * h)
    model = hf.Qwen4ExpTextModel(c).eval()
    load(model.layers[0], "mtp.layers.0.")
    load(model.hyper_connection_mixer, "mtp.hyper_connection_mixer.")
    wrap_indexers(model)
    hook = model.layers[0].register_forward_pre_hook(lambda m, args, kwargs: ((r,), kwargs), with_kwargs=True)
    try:                    # the model's own masks and rotary embedding; the layer's input is the fused residual
        return model(inputs_embeds=torch.zeros(1, len(embeds), h), ple_input_ids=torch.zeros(1, len(embeds),
                     dtype=torch.long)).last_hidden_state[0]
    finally:
        hook.remove()
