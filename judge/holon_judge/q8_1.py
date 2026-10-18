"""q8_1 emulation of llama.cpp's int8-dot matmuls (M0.1 design §4.6), the reference for L0 and for Tier B's floor.

llama.cpp quantizes the activations per block (`d = amax/127`, `q = round(x/d)`) and takes an integer dot product
with the weights' integer factors, so it computes `y = W x̃` with `x̃ = q·d`, up to one exception: for some types the
weights' per-group offset (`w = scale·k + b`, `k` the integer factor) multiplies a stored fp16 sum of the
*unquantized* activations instead of their quantized sum. Each path has its own block layout:

- `mmvq` (`block_q8_1`): 32 values per fp16 scale, fp16 sums; the offsets of Q4_0/Q4_1/Q5_0/Q5_1/IQ1_S use the
  sums, those of the k-quants the quantized values (`vecdotq.cuh`, e.g. `vec_dot_q4_K_q8_1_impl_vmmq`).
- `mmq` (`block_q8_1_mmq`): D4 (32 values per f32 scale; IQ1_M's too, its offset in its int8 weights: no sums,
  docs/v1-moe-prefill.md D9), DS4 (32 per fp16 scale, fp16 sums; the offsets of Q4_1/Q5_1/Q4_K/Q5_K/IQ1_S use
  them, `vecdotq.cuh:521-552` and `ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma`),
  D2S6 for Q2_K (64 per fp16 scale, fp16 sums of 16 for the first 96 of every 128 values).
  These are the MMA kernels', which llama.cpp runs on NVIDIA cc >= 7.5 (`turing_mma_available`), i.e. on every
  class we build (sm_86, sm_89, sm_120a); the dp4a kernels of older GPUs differ (e.g. Q4_0 there uses its sums).

A third path, `cpu`, is ggml-cpu's (docs/v1-offload.md F5b, the CPU experts, §6's path-aware floor): Q8_K activations
(per 256 values an f32 scale; Q8_0 for the types whose dot takes it) as std.moe.reference `quantize_cpu` emulates
them, exactly, and dots with no stored-sum correction (ggml-cpu's k-quant dots use the quantized values' sums).

So `y = W x̃ + Σ_g b_g (S_g − Σ_g x̃)` over the groups `g` whose offset uses a stored sum `S_g`. The offsets `b` are
read from the blocks by gguf-py with the integer factors zeroed. Everything else (fp16 products, summation order,
fast-math) is left to the L0 budget. Types are ggml names in lowercase (`"q4_k"`). Expert stacks (`linear_experts`,
M0.2 design §4.6) quantize each input row once and apply each slot's expert with the same per-path semantics.
"""
from __future__ import annotations

import dataclasses

import gguf
import numpy as np
import torch
from holon_std.moe.reference import Q8_0_DOT, quantize_cpu

MMVQ, MMQ, CPU = "mmvq", "mmq", "cpu"


@dataclasses.dataclass(frozen=True)
class Blocking:
    group: int                       # values per q8_1 scale
    scale: torch.dtype               # the stored scale's dtype
    sum_group: int = 0               # values per stored fp16 sum used by the offsets (0: none)
    sum_span: int = 128              # of every 128 values, the first `sum_span` use the stored sums


_D4, _DS4 = Blocking(32, torch.float32), Blocking(32, torch.float16)
_MMQ_DS4 = {"q4_0", "q4_1", "q5_1", "q4_k", "q5_k", "iq1_s"}
_MMQ_SUMS = {"q4_1", "q5_1", "q4_k", "q5_k", "iq1_s"}      # q4_0 is DS4 but its MMA tiles are signed: no offset
_MMVQ_SUMS = {"q4_0", "q4_1", "q5_0", "q5_1", "iq1_s"}

# Bytes of the integer factors in each block, zeroed to read the offsets b (types whose offsets use stored sums).
_FACTORS = {"q4_0": slice(2, 18), "q4_1": slice(4, 20), "q5_0": slice(2, 22), "q5_1": slice(4, 24),
            "q2_k": slice(16, 80), "q4_k": slice(16, 144), "q5_k": slice(16, 176)}
_IQ1S_DELTA = 0.125


def blocking(ggml_type: str, path: str) -> Blocking:
    """The q8_1 activation blocking of `path` for weights of `ggml_type` (`cpu`: ggml-cpu's Q8_K, or Q8_0)."""
    if path == CPU:
        return Blocking(32, torch.float16) if ggml_type in Q8_0_DOT else Blocking(256, torch.float32)
    if path == MMVQ:
        return Blocking(32, torch.float16, 32 if ggml_type in _MMVQ_SUMS else 0)
    if path != MMQ:
        raise ValueError(f"unknown path {path!r}")
    if ggml_type == "q2_k":
        return Blocking(64, torch.float16, 16, 96)
    if ggml_type in _MMQ_DS4:
        return Blocking(32, torch.float16, 32 if ggml_type in _MMQ_SUMS else 0)
    return _D4


def quantize(x: torch.Tensor, b: Blocking) -> torch.Tensor:
    """x̃ = q·d (float64), d = amax/127 per `b.group` values stored as `b.scale`, q rounded half away from zero."""
    xs = x.double().reshape(*x.shape[:-1], -1, b.group)
    d = xs.abs().amax(-1, keepdim=True) / 127
    v = torch.where(d == 0, 0.0, xs / d)
    q = v.sign() * (v.abs() + 0.5).floor()
    return (q * d.to(b.scale).double()).reshape(x.shape)


def offsets(blocks: np.ndarray, ggml_type: str, group: int) -> torch.Tensor:
    """b per `group` values [out, in/group] (float64): the weight when its integer factor is 0, as the kernels split
    it (IQ1_S: grid + 1 in {0, 1, 2}, offset d·(2s+1)·(δ − 1))."""
    if ggml_type == "iq1_s":
        q = np.ascontiguousarray(blocks).reshape(len(blocks), -1, 50)
        d = q[..., :2].copy().view(np.float16).astype(np.float64)                      # [out, nb, 1]
        qh = q[..., 34:50].copy().view(np.uint16).astype(np.int64)                     # [out, nb, 8]
        b = d * (2 * ((qh >> 12) & 7) + 1) * (np.where(qh & 0x8000, -_IQ1S_DELTA, _IQ1S_DELTA) - 1)
        return torch.from_numpy(b.reshape(len(blocks), -1))
    qtype = gguf.GGMLQuantizationType[ggml_type.upper()]
    size = gguf.GGML_QUANT_SIZES[qtype][1]
    zeroed = np.array(blocks).reshape(len(blocks), -1, size)
    zeroed[..., _FACTORS[ggml_type]] = 0
    w = gguf.quants.dequantize(zeroed.reshape(len(blocks), -1), qtype)
    return torch.from_numpy(w[:, ::group].astype(np.float64))


def matmul(x: torch.Tensor, weight: torch.Tensor, blocks: np.ndarray | None, ggml_type: str, path: str,
           offs: torch.Tensor | None = None, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """llama.cpp's `x @ weight.T` on `path` (in `dtype`, on x's device): `weight` [out, in] dequantized from `blocks`
    [out, row_bytes] (gguf.native) by gguf-py, `x` [T, in]. `offs`: `offsets(blocks, ggml_type, group)` computed
    once for many calls (Tier B's floor runs this in float32 on every quantized linear)."""
    b = blocking(ggml_type, path)
    if b.sum_group and offs is None:
        offs = offsets(blocks, ggml_type, b.sum_group)
    return _product(*_activations(x, b, dtype), weight, offs)


def linear_experts(x: torch.Tensor, weight: torch.Tensor, blocks: np.ndarray | None, ids: torch.Tensor,
                   ggml_type: str, path: str, glu: str = "none", offs: torch.Tensor | None = None,
                   dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """llama.cpp's MUL_MAT_ID on `path` (`linear_experts@2`, in `dtype`, on x's device): slot j of token t applies
    expert `ids[t, j]` of `weight` [E, out, in] (dequantized from `blocks` [E·out, row_bytes]) to `x[t, 0]` or
    `x[t, j]` (`x` [T, R, in], R = 1 or k) → [T, k, out_y]. Each input row is quantized once: for R = 1 (gate/up) a
    token's input once, scattered to its k slots; for R = k (down) each (token, slot) row, gathered. mmvq and mmq with
    ids use the dense paths' q8_1 layouts. `glu="swiglu"`: a merged gate/up stack, y = silu(gate)·up. `offs`:
    `offsets(blocks, ggml_type, group)` viewed [E, out, in/group], computed once for many calls."""
    t, k = ids.shape
    e, out = weight.shape[:2]
    b = blocking(ggml_type, path)
    if b.sum_group and offs is None:
        offs = offsets(blocks, ggml_type, b.sum_group).view(e, out, -1)
    if path == CPU:                                                        # T·R rows
        xt, corr = quantize_cpu(x.reshape(-1, x.shape[-1]), f"ggml.{ggml_type}/gguf.native").to(dtype), None
    else:
        xt, corr = _activations(x.reshape(-1, x.shape[-1]), b, dtype)
    source = torch.arange(t * k, device=x.device) // (k // x.shape[1])     # the input row of each (token, slot)
    flat = ids.reshape(-1).long()
    y = torch.zeros(t * k, out, dtype=dtype, device=x.device)
    for expert in flat.unique().tolist():
        rows = flat == expert
        src = source[rows]
        y[rows] = _product(xt[src], None if corr is None else corr[src], weight[expert],
                           None if offs is None else offs[expert])
    y = y.view(t, k, out)
    if glu == "swiglu":
        gate, up = y.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up
    if glu != "none":
        raise ValueError(f"unknown glu {glu!r}")
    return y


def _activations(x: torch.Tensor, b: Blocking, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor | None]:
    """x̃ [T, in] and, where the offsets use stored sums, each group's `S_g − Σ_g x̃` [T, in/group] (else None)."""
    xt = quantize(x, b).to(dtype)
    if not b.sum_group:
        return xt, None
    g = b.sum_group
    stored = x.float().reshape(len(x), -1, g).sum(-1).half().to(dtype)         # fp32 sums stored as fp16
    corr = stored - xt.reshape(len(x), -1, g).sum(-1)
    corr[:, (torch.arange(corr.shape[1], device=x.device) * g) % 128 >= b.sum_span] = 0
    return xt, corr


def _product(xt: torch.Tensor, corr: torch.Tensor | None, weight: torch.Tensor,
             offs: torch.Tensor | None) -> torch.Tensor:
    y = xt @ weight.to(xt.dtype).T
    if corr is not None:
        y += corr @ offs.to(xt.device, xt.dtype).T
    return y
