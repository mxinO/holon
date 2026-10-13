"""Reference grouped-query attention: causal softmax(q k^T / sqrt(d)) v per sequence, fp32 math."""
from __future__ import annotations

import torch


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, query_start: torch.Tensor,
                     causal: bool = True, window: int = 0) -> torch.Tensor:
    """q [T, Hq, D], k/v [T, Hkv, D]; sequences are the row ranges given by `query_start` [B+1]; `causal` and `window`
    as in `paged_attention`."""
    rep = q.shape[1] // k.shape[1]
    out = torch.empty_like(q)
    for s, e in zip(query_start[:-1].tolist(), query_start[1:].tolist(), strict=True):
        qs, ks, vs = q[s:e].float(), k[s:e].float(), v[s:e].float()
        ks, vs = ks.repeat_interleave(rep, dim=1), vs.repeat_interleave(rep, dim=1)
        scores = torch.einsum("qhd,khd->hqk", qs, ks) / q.shape[-1] ** 0.5
        pos = torch.arange(e - s, device=q.device)
        mask = (pos[None, :] > pos[:, None]) & causal
        if window:
            mask |= pos[:, None] - pos[None, :] >= window
        probs = scores.masked_fill(mask, float("-inf")).softmax(-1)
        out[s:e] = torch.einsum("hqk,khd->qhd", probs, vs).to(q.dtype)
    return out


def paged_attention(q: torch.Tensor, kv: torch.Tensor, table: torch.Tensor, rows: torch.Tensor, lens: torch.Tensor,
                    query_start: torch.Tensor, causal: bool = True, window: int = 0) -> torch.Tensor:
    """The `attention.paged_prefill@1` / `paged_decode@1` / `paged_verify@1` oracle: new rows attend over paged KV.

    q [T, Hq, D]; kv [units, 2, Hkv, page, D] (one layer, HND); `table` [slots, max_pages] unit ids; batch row b
    uses table row `rows[b]`, holds `lens[b]` tokens including this step's, and owns query rows
    `query_start[b]:query_start[b+1]`, the last ones of its sequence (bottom-right causal). Rows with length 0
    (graph padding) produce zeros.

    `causal=False`: a row's new tokens see each other both ways (and all the context before them). `window` W > 0: a
    query at position p sees no key at or before p − W (llama.cpp's `p − p_k < n_swa`, SGLang's `window_left` W − 1;
    HF's sliding mask admits one key more), keys after it (not causal) included.
    """
    if isinstance(kv, tuple):                     # q8_0: (values, scales), read through its dequantization
        kv = dequantize_q8_0(*kv)
    elif kv.dtype == FP8:
        kv = kv.float()
    rep, page = q.shape[1] // kv.shape[2], kv.shape[3]
    out = torch.zeros_like(q)
    bounds = zip(query_start[:-1].tolist(), query_start[1:].tolist(), rows.tolist(), lens.tolist(), strict=True)
    for s, e, row, n in bounds:
        if n == 0 or s == e:
            continue
        units = table[row, : -(-n // page)].long()
        k, v = (kv[units, i].transpose(1, 2).reshape(-1, *kv.shape[2:3], kv.shape[4])[:n].float()
                .repeat_interleave(rep, dim=1) for i in (0, 1))
        scores = torch.einsum("qhd,khd->hqk", q[s:e].float(), k) / q.shape[-1] ** 0.5
        pos = torch.arange(n - (e - s), n, device=q.device)[:, None]
        key = torch.arange(n, device=q.device)[None, :]
        hidden = (key > pos) if causal else torch.zeros(e - s, n, dtype=torch.bool, device=q.device)
        if window:
            hidden |= pos - key >= window
        probs = scores.masked_fill(hidden, float("-inf")).softmax(-1)
        out[s:e] = torch.einsum("hqk,khd->qhd", probs, v).to(q.dtype)
    return out


# KV formats (docs/m03-qwen38.md D6): the KV cache as stored, its write and its read.
Q8_BLOCK = 32
FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0


def quantize_q8_0(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """GGUF's q8_0 over the last dim: per 32 values d = amax/127 (stored fp16), q = round(x · (1/d)) half away from
    zero, with d and its inverse in f32 before the fp16 store (ggml's `quantize_row_q8_0_ref`)."""
    blocks = x.float().reshape(*x.shape[:-1], -1, Q8_BLOCK)
    amax = blocks.abs().amax(-1, keepdim=True)
    d = amax / torch.full_like(amax, 127)               # a true division (CUDA multiplies by 1/127 for a scalar)
    inv = torch.where(d == 0, 0.0, 1.0 / d)
    v = blocks * inv
    q = (v.sign() * (v.abs() + 0.5).floor()).clamp(-127, 127).to(torch.int8)
    return q.reshape(x.shape), d[..., 0].to(torch.float16)


def dequantize_q8_0(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    blocks = values.float().reshape(*values.shape[:-1], -1, Q8_BLOCK)
    return (blocks * scales.float()[..., None]).reshape(values.shape)


def to_fp8(x: torch.Tensor) -> torch.Tensor:
    """e4m3 with saturation (no inf in e4m3fn; out-of-range values clamp to ±448, as the kernels' writes do)."""
    return x.float().clamp(-FP8_MAX, FP8_MAX).to(FP8)


def kv_write(k: torch.Tensor, v: torch.Tensor, kv, unit: torch.Tensor, off: torch.Tensor) -> None:
    """The `kv_write@1` oracle: this step's k/v [T, Hkv, D] into the cache at (unit, off) per token, in its format."""
    if isinstance(kv, tuple):
        values, scales = kv
        for i, x in enumerate((k, v)):
            q, d = quantize_q8_0(x)
            values[unit, i, :, off], scales[unit, i, :, off] = q, d
    elif kv.dtype == FP8:
        kv[unit, 0, :, off], kv[unit, 1, :, off] = to_fp8(k), to_fp8(v)
    else:
        kv[unit, 0, :, off], kv[unit, 1, :, off] = k, v

