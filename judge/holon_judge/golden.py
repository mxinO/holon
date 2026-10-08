"""L2 recorder: HF `AutoModelForCausalLM`, teacher-forced, one forward per sequence (design §10.3, spike S5).

The golden is fp32 on CPU with a pinned thread count: it is bitwise reproducible run to run at the same thread
count, but not across thread counts (~1e-5 in logprob), so the count is part of the provenance. The floor for an
activation dtype is the same recorder run in that dtype (bf16 on the GPU class the engine targets).
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import platform
import re
import time

import torch
import transformers

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
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    try:
        with open("/proc/cpuinfo") as f:
            return next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        return platform.processor() or platform.machine()


@torch.inference_mode()
def teacher_force(model: torch.nn.Module, ids: list[int], k: int, rows: int = 256) -> tuple[torch.Tensor, ...]:
    """One sequence → (topk_ids [n-1, k], topk_lp, target_lp [n-1], tail_lp [n-1]) on the CPU, fp32."""
    x = torch.tensor([ids], device=model.device)
    logits, target = model(x).logits[0, :-1], x[0, 1:]
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
           k: int = 20, config: str | None = None, q8_1: str | None = None) -> Recording:
    """Record top-k teacher-forced logprobs of `model` (hub id, checkpoint directory, or a .gguf file with the HF
    `config` it was made from: the dense model of its dequantized weights, M0.1 §4.6) over `prompts`. `q8_1` ("mmq"
    or "mmvq", GGUF only): every quantized linear computes as llama.cpp's kernels on that path (Tier B's floor)."""
    path = resolve_model(model)
    if path.suffix == ".gguf":
        from . import gguf

        if config is None:
            raise ValueError(f"{path}: a GGUF golden needs the HF config it was made from")
        hf_config = resolve_model(config)
        hf = gguf.hf_model(path, hf_config, dtype, device)
        if q8_1:
            gguf.emulate_q8_1(hf, path, q8_1)
        config_sha256 = hashlib.sha256((hf_config / "config.json").read_bytes()).hexdigest()
        source = gguf.file_meta(path) | {"q8_1": q8_1, "config": model_meta(hf_config)["model"],
                                         "config_sha256": config_sha256}
    else:
        hf = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=dtype, device_map=device,
                                                               attn_implementation="sdpa").eval()
        source = model_meta(path)
    t0 = time.perf_counter()
    rows = [teacher_force(hf, p.ids, k) for p in prompts.prompts]
    wall = time.perf_counter() - t0
    meta = source | prompts.meta() | {
        "source": "hf", "dtype": str(dtype).removeprefix("torch."), "device": hf.device.type, "attn": "sdpa",
        "hardware": hardware(hf.device), "threads": torch.get_num_threads(), "torch": torch.__version__,
        "transformers": transformers.__version__, "wall_s": round(wall, 2)}
    return Recording.from_rows(rows, meta)
