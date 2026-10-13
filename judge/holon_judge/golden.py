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

from . import kv as kv_format
from . import moe
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
           kv: str | None = None) -> Recording:
    """Record top-k teacher-forced logprobs of `model` (hub id, checkpoint directory, or a .gguf file with the HF
    `config` it was made from: the dense model of its dequantized weights, M0.1 §4.6) over `prompts`. `q8_1` ("mmq"
    or "mmvq", GGUF only): every quantized linear computes as llama.cpp's kernels on that path (Tier B's floor).
    `kv` ("q8_0" or "fp8"): attention reads K and V as a cache in that format stores them (`kv.emulate`).
    A floor (not fp32) of an MoE model runs the D3-mirroring MoE block (`moe.mirror`, M0.2 §4.6) unless `pure_hf`;
    the fp32 golden is always HF's own model."""
    path = resolve_model(model)
    if path.suffix == ".gguf":
        from . import gguf

        if config is None:
            raise ValueError(f"{path}: a GGUF golden needs the HF config it was made from")
        hf_config = resolve_model(config)
        is_moe = _is_moe(hf_config)
        mirrored = dtype != torch.float32 and not pure_hf and is_moe
        hf = gguf.hf_model(path, hf_config, dtype, device, router_f32=mirrored)
        if q8_1:
            gguf.emulate_q8_1(hf, path, q8_1)
        if mirrored:
            moe.mirror(hf, experts=gguf.q8_1_experts(path, q8_1) if q8_1 else None)
        config_sha256 = hashlib.sha256((hf_config / "config.json").read_bytes()).hexdigest()
        source = gguf.file_meta(path) | {"q8_1": q8_1, "config": model_meta(hf_config)["model"],
                                         "config_sha256": config_sha256} | _moe_meta(mirrored)
    else:
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
        "source": "hf", "dtype": str(dtype).removeprefix("torch."), "device": hf.device.type, "attn": "sdpa",
        **({"experts": "eager"} if is_moe else {}),      # HF's expert path, pinned (a default may change)
        **({"kv": kv} if kv else {}),
        "hardware": hardware(hf.device), "threads": torch.get_num_threads(), "torch": torch.__version__,
        "transformers": transformers.__version__, "wall_s": round(wall, 2)}
    return Recording.from_rows(rows, meta)


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
