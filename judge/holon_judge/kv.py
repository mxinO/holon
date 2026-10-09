"""A floor with a quantized KV cache (M0.3 D7): HF's attention reads K and V as the engine's cache stores them.

The engine writes a step's K/V into the cache before attention and reads every token through it (`kv_write@1`), so
the floor quantize-dequantizes K and V (post-rope, as cached) once, before every attention read, independently of
the engine's code: q8_0 through gguf-py's own Q8_0 (per 32 values along the head dim, fp16 scale), fp8 as torch's
saturating e4m3 cast. The dequantized values are exact in f32 and enter the floor's attention in its dtype.
"""
from __future__ import annotations

import contextlib
from collections.abc import Iterator

import gguf
import numpy as np
import torch

FORMATS = ("q8_0", "fp8")
FP8_MAX = 448.0


def stored(x: torch.Tensor, fmt: str) -> torch.Tensor:
    """`x` (…, head_dim) as the cache in `fmt` returns it, in `x`'s dtype."""
    if fmt == "fp8":
        return x.float().clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).to(x.dtype)
    if fmt != "q8_0":
        raise ValueError(f"unknown KV format {fmt!r}; known: {FORMATS}")
    rows = x.detach().float().reshape(-1, x.shape[-1]).cpu().numpy()
    q = gguf.quants.quantize(rows, gguf.GGMLQuantizationType.Q8_0)
    back = gguf.quants.dequantize(q, gguf.GGMLQuantizationType.Q8_0).astype(np.float32)
    return torch.from_numpy(back).to(x.device).reshape(x.shape).to(x.dtype)


@contextlib.contextmanager
def emulate(fmt: str | None) -> Iterator[None]:
    """HF's sdpa attention reads K and V in `fmt` (None: the model dtype, unchanged) while inside."""
    if fmt is None:
        yield
        return
    if fmt not in FORMATS:
        raise ValueError(f"unknown KV format {fmt!r}; known: {FORMATS}")
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def attention(module, query, key, value, *args, **kwargs):
        return sdpa(module, query, stored(key, fmt), stored(value, fmt), *args, **kwargs)
    ALL_ATTENTION_FUNCTIONS["sdpa"] = attention
    try:
        yield
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa
