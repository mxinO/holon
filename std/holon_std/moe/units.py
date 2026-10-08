"""std.moe units: the MoE ffn (router, top-k, expert stacks, combine) and its expert stacks (docs/m02-moe.md §4.2)."""
from __future__ import annotations

import torch
from torch import nn

from . import kernels

CHUNK = 512                        # tokens per pass through the expert path: bounds its f32 activations


def _dtype(t: torch.dtype) -> str:
    return str(t).removeprefix("torch.")


class ExpertLinear(nn.Module):
    """A stack of E experts' [out, in] weights, applied per (token, slot) by `linear_experts@1`.

    Dense (an HF checkpoint) it is the parameter itself, layout `dense/<dtype>`; quantized (a GGUF file) `adopt`
    keeps the stored parts — ggml layouts have one, `blocks` [E·out, row_bytes], held in `weight`. `merge(gate, up)`
    builds one gate/up stack per expert that applies SwiGLU (`glu="swiglu"`); the kernel is bound after merges."""

    def __init__(self, ctx, *, experts: int, in_features: int, out_features: int, top_k: int,
                 rows_per_token: int) -> None:
        super().__init__()
        self.experts, self.in_features, self.out_features = experts, in_features, out_features
        self.top_k, self.rows_per_token, self.glu, self._ctx = top_k, rows_per_token, "none", ctx
        self.weight = nn.Parameter(torch.empty(experts, out_features, in_features, dtype=ctx.dtype),
                                   requires_grad=False)
        self.layout: str | None = None                       # set by `adopt`; None: dense
        self._kernel = None

    @property
    def role(self) -> str:
        return "weight" if self.layout is None else "blocks"

    def layout_fact(self) -> str:
        return self.layout or f"dense/{_dtype(self.weight.dtype)}"

    def adopt(self, ctx, name: str, q) -> bool:
        if name != "weight" or tuple(q.logical_shape) != (self.experts, self.out_features, self.in_features) \
                or set(q.parts) != {"blocks"}:
            return False
        self.layout = f"{q.layout.scheme.elem}/{q.layout.name}"
        torch.utils.swap_tensors(self.weight, nn.Parameter(q.parts["blocks"].to(ctx.device), requires_grad=False))
        return True

    @staticmethod
    def merge(gate: ExpertLinear, up: ExpertLinear) -> ExpertLinear | None:
        """One stack applying SwiGLU: per expert gate's rows, then up's (every stored row is a whole block run, so
        stacks of one layout concatenate per expert). None if the layouts or shapes differ; gate and up give up
        their weights."""
        if (gate.layout, gate.in_features, gate.out_features, gate.experts) != \
                (up.layout, up.in_features, up.out_features, up.experts) or gate.weight.dtype != up.weight.dtype:
            return None
        e = gate.experts
        weight = torch.cat([gate.weight.data.view(e, gate.out_features, -1),
                            up.weight.data.view(e, up.out_features, -1)], dim=1)
        with torch.device("meta"):
            merged = ExpertLinear(gate._ctx, experts=e, in_features=gate.in_features,
                                  out_features=2 * gate.out_features, top_k=gate.top_k,
                                  rows_per_token=gate.rows_per_token)
        if gate.layout is not None:
            weight = weight.view(-1, weight.shape[-1])               # back to the stored [E·out, row_bytes]
        torch.utils.swap_tensors(merged.weight, nn.Parameter(weight, requires_grad=False))
        merged.layout, merged.glu = gate.layout, "swiglu"
        gate.weight = up.weight = None
        return merged

    def bind_kernels(self, ctx) -> None:
        if self.weight is None:                                   # merged away
            return
        facts = {"layout": self.layout_fact(), "experts": self.experts, "in": self.in_features,
                 "out": self.out_features, "top_k": self.top_k, "rows_per_token": self.rows_per_token,
                 "glu": self.glu, "act_dtype": _dtype(ctx.dtype), "out_dtype": "float32"}
        site = "std.moe.experts[" + ",".join(f"{k}={v}" for k, v in facts.items()) + "]"
        self._kernel = ctx.select_kernel(kernels.LINEAR_EXPERTS, site=site, **facts).impl(ctx, **facts)

    def forward(self, x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        return self._kernel(x, {self.role: self.weight}, ids)


class MoE(nn.Module):
    """Mixture of experts: softmax top-k routing over an f32 router, gate/up/down expert stacks with SwiGLU, the
    weighted combine. The expert path runs in f32 (design D3), CHUNK tokens at a time."""

    def __init__(self, ctx, *, hidden: int, experts: int, top_k: int, inter: int, renormalize: bool,
                 layer: int = 0) -> None:
        super().__init__()
        self.layer = layer
        self.router = nn.Parameter(torch.empty(experts, hidden, dtype=torch.float32), requires_grad=False)
        stack = {"experts": experts, "top_k": top_k}
        self.gate = ExpertLinear(ctx, in_features=hidden, out_features=inter, rows_per_token=1, **stack)
        self.up = ExpertLinear(ctx, in_features=hidden, out_features=inter, rows_per_token=1, **stack)
        self.down = ExpertLinear(ctx, in_features=inter, out_features=hidden, rows_per_token=top_k, **stack)
        self._gate_up: ExpertLinear | None = None
        dtype = _dtype(ctx.dtype)
        topk = {"experts": experts, "top_k": top_k, "renormalize": renormalize}
        self._topk = ctx.select_kernel(kernels.MOE_TOPK, site=f"std.moe[experts={experts},top_k={top_k}]",
                                       **topk).impl(ctx, **topk)
        combine = {"hidden": hidden, "top_k": top_k, "dtype": dtype, "y_dtype": "float32"}
        self._combine = ctx.select_kernel(kernels.MOE_COMBINE, site=f"std.moe[hidden={hidden},top_k={top_k}]",
                                          **combine).impl(ctx, **combine)

    def finalize_weights(self, ctx) -> None:
        self._gate_up = ExpertLinear.merge(self.gate, self.up)
        if self._gate_up is None:
            raise ValueError(f"layer {self.layer}: gate and up experts must share a layout to merge "
                             f"(gate {self.gate.layout_fact()}, up {self.up.layout_fact()})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for a in range(0, x.shape[0], CHUNK):
            xc = x[a:a + CHUNK]
            weights, ids = self._topk(nn.functional.linear(xc.float(), self.router))
            h = self._gate_up(xc[:, None], ids)                       # [t, k, inter] f32, SwiGLU applied
            self._combine(self.down(h, ids), weights, out[a:a + CHUNK])
        return out
