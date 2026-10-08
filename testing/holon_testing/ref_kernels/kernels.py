"""The std oracles wrapped as kernel instances, for CPU tests (tiny models end to end) and as oracles in GPU tests.

Each class has the instance API of the op it serves (see the owning unit's `kernels.py`).
"""
from __future__ import annotations

import gguf
import numpy as np
import torch
from holon_std.ffn import reference as ffn
from holon_std.mixers.gqa import reference as gqa
from holon_std.moe import reference as moe
from holon_std.ops import reference as ops


class RMSNorm:
    def __init__(self, ctx, *, dim: int, eps: float, offset: float, dtype: str) -> None:
        self.eps, self.offset = eps, offset

    def __call__(self, x, weight):
        return ops.rmsnorm(x, weight, self.eps, self.offset)


class AddRMSNorm(RMSNorm):
    def __call__(self, x, residual, weight):
        return ops.add_rmsnorm(x, residual, weight, self.eps, self.offset)


class Rope:
    def __init__(self, ctx, *, head_dim: int, rotary_dim: int, theta: float, max_positions: int, dtype: str) -> None:
        self.theta, self.rotary_dim = theta, rotary_dim

    def __call__(self, q, k, positions):
        return (ops.rope_neox(q, positions, self.theta, self.rotary_dim),
                ops.rope_neox(k, positions, self.theta, self.rotary_dim))


class SiluMul:
    def __init__(self, ctx, *, inter: int, dtype: str) -> None:
        self.inter = inter

    def __call__(self, gate_up):
        return ffn.silu_mul(gate_up[..., : self.inter], gate_up[..., self.inter:])


class PagedAttention:
    """`attention.paged_prefill/decode/verify@1` (any dtype)."""

    def __init__(self, ctx, **facts) -> None:
        pass

    def plan(self, batch, pages):
        return pages.table, batch.dev.slots, pages.lens, batch.dev.query_start

    def run(self, q, kv, handle):
        return gqa.paged_attention(q, kv, *handle)


class CausalAttention:
    def __init__(self, ctx, **facts) -> None:
        pass

    def __call__(self, q, k, v, query_start):
        return gqa.causal_attention(q, k, v, query_start)


class _Dequantized:
    """Quantized-op references for ggml layouts (`ggml.<type>/gguf.native`), dequantizing with gguf-py — the oracle
    the judge uses — once per weight."""

    def __init__(self, ctx, *, layout: str, **facts) -> None:
        self.type = gguf.GGMLQuantizationType[layout.split("/")[0].removeprefix("ggml.").upper()]
        self.facts, self._cache = facts, {}

    def dequantized(self, blocks: torch.Tensor) -> torch.Tensor:
        key = (blocks.data_ptr(), blocks.shape)
        if key not in self._cache:
            self._cache[key] = torch.from_numpy(gguf.quants.dequantize(np.asarray(blocks.cpu()), self.type))
        return self._cache[key]


class QLinear(_Dequantized):
    def __call__(self, x, blocks):
        return ops.qlinear(x, self.dequantized(blocks)).to(getattr(torch, self.facts["out_dtype"]))


class QEmbedding(_Dequantized):
    def __call__(self, ids, blocks):
        return ops.qembedding(ids, self.dequantized(blocks)).to(getattr(torch, self.facts["dtype"]))


class LinearExperts:
    """Expert stacks: ggml layouts dequantized by gguf-py once per stack; dense ones as stored, the activation rounded
    to their dtype (the contract)."""

    def __init__(self, ctx, *, layout: str, experts: int, out: int, glu: str, out_dtype: str, **facts) -> None:
        self.ggml = None if layout.startswith("dense/") else _Dequantized(ctx, layout=layout)
        self.experts, self.out, self.glu, self.out_dtype = experts, out, glu, getattr(torch, out_dtype)

    def __call__(self, x, parts, ids):
        if self.ggml is None:
            weight = parts["weight"]
            y = moe.linear_experts(x, weight, ids, self.glu, operand_dtype=weight.dtype)
        else:
            weight = self.ggml.dequantized(parts["blocks"]).view(self.experts, self.out, -1)
            y = moe.linear_experts(x, weight, ids, self.glu)
        return y.to(self.out_dtype)


class MoeTopk:
    def __init__(self, ctx, *, experts: int, top_k: int, renormalize: bool) -> None:
        self.top_k, self.renormalize = top_k, renormalize

    def __call__(self, logits):
        return moe.moe_topk(logits, self.top_k, self.renormalize)


class MoeCombine:
    def __init__(self, ctx, *, dtype: str, **facts) -> None:
        self.dtype = getattr(torch, dtype)

    def __call__(self, y, weights, out):
        out.copy_(moe.moe_combine(y, weights, self.dtype))


BY_OP = {"rmsnorm": RMSNorm, "add_rmsnorm": AddRMSNorm, "rope": Rope, "silu_mul": SiluMul,
         "attention.paged_prefill": PagedAttention, "attention.paged_decode": PagedAttention,
         "attention.paged_verify": PagedAttention, "attention.causal": CausalAttention,
         "qlinear": QLinear, "qembedding": QEmbedding, "linear_experts": LinearExperts, "moe_topk": MoeTopk,
         "moe_combine": MoeCombine}
