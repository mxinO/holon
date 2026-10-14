"""L2 recorder: HF `AutoModelForCausalLM`, teacher-forced, one forward per sequence (design §10.3, spike S5).

The golden is fp32 on CPU with a pinned thread count: it is bitwise reproducible run to run at the same thread
count, but not across thread counts (~1e-5 in logprob), so the count is part of the provenance. The floor for an
activation dtype is the same recorder run in that dtype (bf16 on the GPU class the engine targets).
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import re
import time

import torch
import transformers
from holon.api import Fingerprint
from holon_std.weights.hf import reference as hf_reference

from . import hf_quant, modes, moe, w4a4
from . import kv as kv_format
from . import q8_1 as q8_1_emulator
from .recording import PromptSet, Recording

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def resolve_model(model: str) -> pathlib.Path:
    """A checkpoint directory or file (a .gguf), or an HF model id resolved through the hub cache (downloaded if
    missing, unless the hub is offline)."""
    path = pathlib.Path(model)
    if path.exists():
        return path
    from huggingface_hub import snapshot_download

    return pathlib.Path(snapshot_download(model))


def model_meta(path: pathlib.Path) -> dict:
    """Provenance without local paths: the hub id and revision when `path` is a hub-cache snapshot, the sha256 of
    every weight file (the blob name for hub-cache symlinks, which is that sha256; hashed otherwise) and of
    config.json."""
    model, revision = path.name, None
    if path.parent.name == "snapshots" and path.parent.parent.name.startswith("models--"):
        model, revision = path.parent.parent.name.removeprefix("models--").replace("--", "/"), path.name
    weights = {f.name: _sha256(f) for f in sorted(path.glob("*.safetensors"))}
    config = path / "config.json"
    return {"model": model, "revision": revision, "weight_sha256": weights,
            "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest() if config.is_file() else None}


def _sha256(f: pathlib.Path) -> str:
    if f.is_symlink() and re.fullmatch(r"[0-9a-f]{64}", blob := os.path.basename(os.readlink(f))):
        return blob
    with open(f, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def hardware(device: torch.device) -> str:
    """The GPU's product name (it defines the card class), or just "cpu": host details stay out of records."""
    return torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"


@torch.inference_mode()
def teacher_force(model: torch.nn.Module, ids: list[int], k: int, rows: int = 256) -> tuple[torch.Tensor, ...]:
    """One sequence → (topk_ids [n-1, k], topk_lp, target_lp [n-1], tail_lp [n-1]) on the CPU, fp32."""
    x = torch.tensor([ids], device=model.device)
    return topk_rows(model(x).logits[0, :-1], x[0, 1:], k, rows)


def topk_rows(logits: torch.Tensor, target: torch.Tensor, k: int, rows: int = 256) -> tuple[torch.Tensor, ...]:
    """Next-token logits [n, V] and their targets [n] → a recording's rows (top-k ids and logprobs, the target's
    logprob, the tail's), on the CPU, fp32."""
    out = []
    for a in range(0, len(target), rows):  # row blocks bound the fp64 copies of [rows, vocab]
        # Normalize in fp64: an fp32 log-softmax over a 150k vocabulary is off by up to 1e-4 nats on CPU, the size
        # of the bf16 floor's KL p50.
        lp = logits[a:a + rows].double().log_softmax(-1)
        top_lp, top_ids = lp.topk(k, dim=-1)
        tgt = lp.gather(1, target[a:a + rows, None])[:, 0]
        # The tail straight from the full distribution: 1 − Σ top-k cancels catastrophically when it is tiny.
        tail = lp.scatter_(1, top_ids, float("-inf")).logsumexp(-1)
        out.append((top_ids.int(), top_lp.float(), tgt.float(), tail.float()))
    return tuple(torch.cat(t).cpu() for t in zip(*out, strict=True))


def record(model: str, prompts: PromptSet, dtype: torch.dtype = torch.float32, device: str = "cpu",
           k: int = 20, config: str | None = None, q8_1: str | None = None, pure_hf: bool = False,
           kv: str | None = None, act: str | None = None, stream: bool = False, rows: int | None = None,
           crossover: str | None = None) -> Recording:
    """Record top-k teacher-forced logprobs of `model` (hub id, checkpoint directory, or a .gguf file with the HF
    `config` it was made from: the dense model of its dequantized weights, M0.1 §4.6) over `prompts`. `q8_1` ("mmq"
    or "mmvq"; "prefill" for FP8): every quantized linear computes as the kernels on that path (Tier B's floor), its
    activations in mode `act` (`modes.MODES`; default q8_1, the only one for a GGUF; fp8, W8A8, for FP8 weights).
    `q8_1` "rows" (a dense GGUF): each linear on the path its matmul takes in steps of `rows` rows on the class
    `crossover` ("<gpu class>:<sm>", e.g. "rtx-5080:120"; `gguf.row_paths`): D2's small-step floors.
    `kv` ("q8_0" or "fp8"): attention reads K and V as a cache in that format stores them (`kv.emulate`).
    A GGUF, a quantized HF checkpoint (its weights as `hf_quant` reads them, docs/m06-lowp.md §4.8) or, with
    `stream`, a dense one runs one decoder layer at a time (`_streamed`), so its weights are never all held; a dense
    HF checkpoint otherwise loads through transformers. A floor (not fp32) of an MoE model runs the D3-mirroring MoE
    block (`moe.mirror`, M0.2 §4.6) unless `pure_hf`; the fp32 golden is always HF's own model."""
    path = resolve_model(model)
    if act is not None and not q8_1:
        raise ValueError(f"activation mode {act!r} without a kernel path to emulate it on")
    if (rows or crossover) and q8_1 != "rows":
        raise ValueError("--rows and --crossover go with --q8-1 rows (a per-shape floor)")
    if q8_1 == "rows" and path.suffix != ".gguf":
        raise ValueError(f"{path}: per-shape floors (--q8-1 rows) are for GGUF files")
    if path.suffix == ".gguf":
        from . import gguf

        if config is None:
            raise ValueError(f"{path}: a GGUF golden needs the HF config it was made from")
        if act not in (None, "q8_1") or q8_1 not in (None, "mmq", "mmvq", "rows"):
            raise ValueError(f"{path}: a GGUF floor emulates q8_1 activations on mmq or mmvq, not {act} on {q8_1}")
        hf_config = resolve_model(config)
        is_moe = _is_moe(hf_config)
        mirrored = dtype != torch.float32 and not pure_hf and is_moe
        routes, per_shape = q8_1, {}
        if q8_1 == "rows":
            if not rows or not crossover or is_moe:
                raise ValueError(f"{path}: a per-shape floor needs a dense GGUF, its step rows and the crossover class")
            gpu, sm = crossover.split(":")
            routes = gguf.row_paths(path, rows, Fingerprint(gpu=gpu, sm=int(sm), vram_gib=0, toolchain=""))
            per_shape = {"step_rows": rows, "crossover": gpu, "routes": {
                f"{t}/{k}x{o}": routes[name] for name, (t, k, o) in gguf.engine_shapes(path).items()}}
        experts = gguf.q8_1_experts(path, q8_1) if q8_1 and mirrored else None

        def prepare(module: torch.nn.Module, prefix: str) -> None:
            if q8_1:
                gguf.emulate_q8_1(module, path, routes, prefix)
            if mirrored:
                moe.mirror(module, experts=experts, prefix=prefix)

        config_sha256 = hashlib.sha256((hf_config / "config.json").read_bytes()).hexdigest()
        source = gguf.file_meta(path) | {"q8_1": q8_1, "config": model_meta(hf_config)["model"],
                                         "config_sha256": config_sha256} | _moe_meta(mirrored) | _emulation_meta(q8_1) \
            | per_shape
        with kv_format.emulate(kv):
            rows, wall = _streamed(_skeleton(hf_config), lambda select: dict(gguf.hf_state(path, select)), "blk.",
                                   prompts, k, dtype, device, mirrored, prepare)
    elif stream or hf_quant.quantized(path):
        is_moe = _is_moe(path)
        mirrored = dtype != torch.float32 and not pure_hf and is_moe
        quantized = hf_quant.quantized(path)
        with hf_quant.opened(path) as (get, names), kv_format.emulate(kv):
            kinds: dict[str, str] = {}

            def state(select):
                tensors, found = hf_quant.state(get, names, select)
                kinds.update(found)
                return tensors

            def prepare(module: torch.nn.Module, prefix: str) -> None:
                if q8_1:
                    modes.emulate(module, kinds, q8_1, act or "q8_1", prefix,
                                  lambda name: hf_quant.input_scale(get, names, name))
                if mirrored:
                    moe.mirror(module, None if quantized else dtype,
                               modes.experts(kinds, q8_1, act or "q8_1") if q8_1 else None, prefix)

            rows, wall = _streamed(_skeleton(path), state, "model.layers.", prompts, k, dtype, device, mirrored,
                                   prepare)
        source = model_meta(path) | _moe_meta(mirrored) | _emulation_meta(q8_1, act) | (
            {"hf_reference": _hash(hf_reference)} if quantized else {})
    else:
        if q8_1:
            raise ValueError(f"{path}: a dense checkpoint has no quantized linears to emulate")
        is_moe = _is_moe(path)
        mirrored = dtype != torch.float32 and not pure_hf and is_moe
        hf = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation="sdpa",
                                                               experts_implementation="eager").eval()
        # transformers 5.6 ignores `dtype` for a multimodal checkpoint's text model (Qwen3.5/3.8): cast the parameters
        # (only: the rotary frequency buffers stay f32, as `from_pretrained` keeps them).
        for param in hf.parameters():
            param.data = param.data.to(dtype)
        if mirrored:                       # the router F32 as the engine holds it; dense operands rounded to bf16
            for block in hf.modules():
                if type(block).__name__.endswith("SparseMoeBlock"):
                    block.gate.weight.data = block.gate.weight.data.float()
            moe.mirror(hf, operand_dtype=dtype)
        hf = place(hf, device)
        source = model_meta(path) | _moe_meta(mirrored)
        _torch_paths(hf)
        t0 = time.perf_counter()
        with kv_format.emulate(kv):
            rows = [teacher_force(hf, p.ids, k) for p in prompts.prompts]
        wall = time.perf_counter() - t0
    meta = source | prompts.meta() | {
        "source": "hf", "dtype": str(dtype).removeprefix("torch."), "device": torch.device(device).type,
        "attn": "sdpa", **({"experts": "eager"} if is_moe else {}),   # HF's expert path, pinned (a default may change)
        **({"kv": kv} if kv else {}),
        "hardware": hardware(torch.device(device)), "threads": torch.get_num_threads(), "torch": torch.__version__,
        "transformers": transformers.__version__, "wall_s": round(wall, 2)}
    return Recording.from_rows(rows, meta)


def _skeleton(config_dir: pathlib.Path) -> torch.nn.Module:
    """The HF model of `config_dir`'s config (a multimodal one: its text part; a quantized one: its dense model) on
    the meta device: weights come layer by layer (`_streamed`)."""
    config = transformers.AutoConfig.from_pretrained(config_dir).get_text_config()
    if hasattr(config, "quantization_config"):
        del config.quantization_config
    with torch.device("meta"):
        return transformers.AutoModelForCausalLM.from_config(config, attn_implementation="sdpa",
                                                             experts_implementation="eager").eval()


class _Inputs(torch.nn.Module):
    """A decoder layer's stand-in: keeps what the model passes the layer (per call) and returns its input."""

    def __init__(self, layer: torch.nn.Module, kept: list) -> None:
        super().__init__()
        self.kept = kept
        if hasattr(layer, "attention_type"):
            self.attention_type = layer.attention_type

    def forward(self, hidden: torch.Tensor, **kwargs) -> torch.Tensor:
        self.kept.append((hidden, kwargs))
        return hidden


@torch.inference_mode()
def _streamed(model: torch.nn.Module, state, layers: str, prompts: PromptSet, k: int, dtype: torch.dtype,
              device: str, router_f32: bool, prepare) -> tuple[list[tuple[torch.Tensor, ...]], float]:
    """Teacher forcing one decoder layer at a time: the model's own forward runs once per prompt with every layer
    replaced by `_Inputs` (so masks and position embeddings are its own), then each layer is loaded (`state(select)`:
    the fp32 state entries of the source names `select` keeps, those starting `layers` + its index), `prepare(module,
    name prefix)`d (a floor's emulation), applied to every prompt's hidden states and freed; the head (also
    `prepare`d) closes. The same operations as the whole model's forward, so the same bits (judge/tests). Weights are
    cast to `dtype` on `device`, MoE routers kept F32 if `router_f32`."""
    def load(module: torch.nn.Module, tensors: dict[str, torch.Tensor], strip: str = "") -> None:
        keep = lambda name: router_f32 and name.endswith(".mlp.gate.weight")          # noqa: E731
        module.load_state_dict({n.removeprefix(strip): (t if keep(n) else t.to(dtype)).to(device)
                                for n, t in tensors.items()}, strict=bool(strip), assign=True)

    config, blocks = model.config, model.model.layers
    load(model, state(lambda name: not name.startswith(layers)))
    if config.tie_word_embeddings:
        model.tie_weights()
    if missing := [n for n, p in model.named_parameters() if p.is_meta and not n.startswith("model.layers.")]:
        raise ValueError(f"no tensor for {missing[:4]}")
    model.model.rotary_emb = type(model.model.rotary_emb)(config=config).to(device)    # buffers, not weights
    prepare(model.lm_head, "lm_head")
    _torch_paths(model)
    t0 = time.perf_counter()
    kept: list[list] = [[] for _ in blocks]
    model.model.layers = torch.nn.ModuleList(_Inputs(b, kept[i]) for i, b in enumerate(blocks))
    for p in prompts.prompts:
        model.model(torch.tensor([p.ids], device=device), use_cache=False)
    model.model.layers = blocks
    hidden = [h for h, _ in kept[0]]
    for i, block in enumerate(blocks):
        prefix = f"model.layers.{i}."
        load(block, state(lambda name, start=f"{layers}{i}.": name.startswith(start)), prefix)
        prepare(block, prefix)
        hidden = [block(h, **kwargs) for h, (_, kwargs) in zip(hidden, kept[i], strict=True)]
        block.to("meta")
    rows = [topk_rows(model.lm_head(model.model.norm(h))[0, :-1], torch.tensor(p.ids[1:], device=device), k)
            for h, p in zip(hidden, prompts.prompts, strict=True)]
    return rows, time.perf_counter() - t0


def place(model: torch.nn.Module, device: str) -> torch.nn.Module:
    """On one device, or across the visible GPUs when it does not fit one (whole decoder layers per GPU)."""
    size = sum(p.numel() * p.element_size() for p in model.parameters())
    if device != "cuda" or size < 0.85 * torch.cuda.get_device_properties(0).total_memory:
        return model.to(device)
    import accelerate

    layer = type(model.model.layers[0]).__name__
    budget = {i: int(0.7 * torch.cuda.get_device_properties(i).total_memory)        # room for activations
              for i in range(torch.cuda.device_count())}
    placement = accelerate.infer_auto_device_map(model, max_memory=budget, no_split_module_classes=[layer])
    return accelerate.dispatch_model(model, placement)


def _torch_paths(model: torch.nn.Module) -> None:
    """The reference runs transformers' own torch code: a swappable hub kernel (e.g. flash-linear-attention's for a
    Gated DeltaNet layer) would make it another implementation of the same thing."""
    for name, m in model.named_modules():
        for attr in ("chunk_gated_delta_rule", "recurrent_gated_delta_rule", "causal_conv1d_fn"):
            fn = getattr(m, attr, None)
            if fn is not None and not getattr(fn, "__name__", "").startswith("torch_"):
                raise RuntimeError(f"{name}.{attr} is {fn!r}, not transformers' torch path")


def _is_moe(config_dir: pathlib.Path) -> bool:
    return transformers.AutoConfig.from_pretrained(config_dir).model_type.endswith("_moe")


def _moe_meta(mirrored: bool) -> dict:
    return {"moe_block": f"d3-reference@{moe.reference_hash()}"} if mirrored else {}


def _emulation_meta(path: str | None, act: str | None = None) -> dict:
    """A floor's emulated path and mode, and one hash over the modules emulating them (docs/m06-lowp.md §4.8): a change
    to any invalidates the floor."""
    if not path:
        return {}
    return {"q8_1": path, "act": act or "q8_1", "emulation": _hash(q8_1_emulator, modes, w4a4, hf_reference)}


def _hash(*modules) -> str:
    """The modules' sources' hash."""
    h = hashlib.sha256()
    for module in modules:
        with open(module.__file__, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]
