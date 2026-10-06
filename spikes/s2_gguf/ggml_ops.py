"""Build/load the S2 GGUF Torch extension for the current GPU's SM and expose checked wrappers."""
from __future__ import annotations

import os
from pathlib import Path

import gguf
import torch
from torch.utils.cpp_extension import load

HERE = Path(__file__).resolve().parent
VENDORED = HERE / "third_party" / "sglang_gguf"


def load_ext(sm: int | None = None):
    if sm is None:
        major, minor = torch.cuda.get_device_capability()
        sm = major * 10 + minor
    build = HERE / "out" / f"ext-sm{sm}"
    build.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{sm // 10}.{sm % 10}"
    return load(
        name=f"ggml_ext_sm{sm}",
        sources=[str(HERE / "csrc" / "ggml_ext.cu")],
        extra_include_paths=[str(HERE / "csrc"), str(VENDORED)],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17", "-lineinfo"],
        build_directory=str(build),
        verbose=False,
    )


def q8_1_words(rows: int, cols: int) -> int:
    """int32 words of the q8_1 activation workspace for a [rows, cols] input."""
    return rows * ((cols + 511) // 512 * 512) // 32 * 9


def nbytes(qtype: int, shape: tuple[int, ...]) -> int:
    block, size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(qtype)]
    n = 1
    for s in shape:
        n *= s
    assert shape[-1] % block == 0
    return n // block * size


def tile_weight(raw: torch.Tensor, qtype: int, shape: tuple[int, ...]) -> torch.Tensor:
    """A weight of `shape` (logical, last dim = K) built by tiling whole ggml blocks from a real tensor.

    ggml blocks are self-contained, so any sequence of real blocks is a valid tensor; used for timing shapes
    that no S2 file holds in that type."""
    block, size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(qtype)]
    blocks = raw.reshape(-1, size)
    need = nbytes(qtype, shape) // size
    reps = (need + blocks.shape[0] - 1) // blocks.shape[0]
    out = blocks.repeat(reps, 1)[:need]
    return out.reshape(*shape[:-1], shape[-1] // block * size).contiguous()


def moe_align(topk_ids: torch.Tensor, block: int, num_experts: int):
    """moe_align_block_size in torch (host-synchronizing; the spike only times the GEMM kernel).

    Returns sorted_token_ids (padding = T*k), expert_ids per block (-1 unused), num_tokens_post_padded[1]."""
    flat = topk_ids.flatten().to(torch.int64)
    n = flat.numel()
    max_len = n + num_experts * (block - 1)
    max_len = (max_len + block - 1) // block * block
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=num_experts)
    padded = (counts + block - 1) // block * block
    starts = torch.cumsum(padded, 0) - padded
    rank = torch.arange(n, device=flat.device) - (torch.cumsum(counts, 0) - counts)[flat[order]]
    sorted_ids = torch.full((max_len,), n, dtype=torch.int32, device=flat.device)
    sorted_ids[starts[flat[order]] + rank] = order.to(torch.int32)
    expert_ids = torch.full((max_len // block,), -1, dtype=torch.int32, device=flat.device)
    nblk = (padded // block).tolist()
    expert_ids[: sum(nblk)] = torch.repeat_interleave(
        torch.arange(num_experts, device=flat.device, dtype=torch.int32), torch.tensor(nblk, device=flat.device))
    total = padded.sum().to(torch.int32).reshape(1)
    return sorted_ids, expert_ids, total
