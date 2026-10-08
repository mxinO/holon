"""llama.cpp's quantized CUDA kernels (vendored at a pinned commit, M0.1 design §4.4, M0.2 design §4.3) as
`qlinear@1`, `qembedding@1` and `linear_experts@1` kernels for ggml block layouts (`"ggml.<type>/gguf.native"`), and
its MoE routing and combine as `moe_topk@1` and `moe_combine@1` (op contracts: std.ops.kernels, std.moe.kernels).

`qlinear` dispatches like llama.cpp, from the row count: `mmvq` up to the type's threshold on this device (it writes
bf16 or f32 directly), else `mmq` in 512-row chunks (llama.cpp's ubatch) into a per-call f32 tensor that one cast
turns into `out_dtype` (D5). Both quantize the activations to q8_1 in a workspace shared by every instance with the
same `in`. `qembedding` is llama.cpp's `get_rows`. With `in` a multiple of 256 (all supported shapes) no kernel
reads past the weight blocks, so they need no padding (L0 poisons the bytes after them).

`linear_experts` is llama.cpp's MUL_MAT_ID per 512-token chunk: mmvq with ids up to the type's limit
(`get_mmvq_mmid_max_batch`), for `glu="swiglu"` through its fused gate on the merged stack, else mmq with ids into
f32, followed for swiglu by our f32 SwiGLU. Its workspace (`ggml_cuda.id[…]`) holds mmq's id maps besides q8_1.
`moe_topk` is `topk-moe.cu` (softmax, top k, renormalised) and `moe_combine` `moe-weighted-reduction.cu`, both
patched to the contracts' layouts (ids at row stride k; the combine stores bf16 into the caller's `out`).

The native extension is built ahead of time (`python -m holon_plugins.kernels.ggml_cuda.prebuild`); without it an
instance cannot be created and the error names that command.
"""
from __future__ import annotations

import functools

import torch
from holon.api import Constraint, KernelSpec

from . import extension

QLINEAR, QEMBEDDING = "qlinear", "qembedding"
LINEAR_EXPERTS, MOE_TOPK, MOE_COMBINE = "linear_experts", "moe_topk", "moe_combine"
CHUNK = 512                     # mmq rows per call: llama.cpp's ubatch; sizes the workspace and the f32 buffer
# Supported ggml types (design §4.6) and their ggml type ids.
TYPES = {"q4_0": 2, "q4_1": 3, "q5_0": 6, "q5_1": 7, "q8_0": 8, "q2_k": 10, "q3_k": 11, "q4_k": 12, "q5_k": 13,
         "q6_k": 14, "iq2_xxs": 16, "iq2_xs": 17, "iq3_xxs": 18, "iq1_s": 19, "iq4_nl": 20, "iq3_s": 21, "iq2_s": 22,
         "iq4_xs": 23}
DTYPES = ("bfloat16", "float32")


def ggml_type(layout: str) -> int | None:
    """The ggml type id of a `"ggml.<type>/gguf.native"` layout, None for any other layout."""
    scheme, _, name = layout.partition("/")
    return TYPES.get(scheme.removeprefix("ggml.")) if scheme.startswith("ggml.") and name == "gguf.native" else None


class Extension:
    """The prebuilt module, loaded once per SM on first use (never compiled here)."""

    def __init__(self) -> None:
        self.modules: dict[int, object] = {}

    def __call__(self, sm: int):
        if sm not in self.modules:
            self.modules[sm] = extension.load(sm)
        return self.modules[sm]


class QLinear:
    def __init__(self, ext: Extension, ctx, **facts) -> None:
        self.type, self.k, self.out = ggml_type(facts["layout"]), facts["in"], facts["out"]
        self.out_dtype = getattr(torch, facts["out_dtype"])
        self.lib, device = ext(ctx.fingerprint.sm), torch.device(ctx.device).index
        device = torch.cuda.current_device() if device is None else device
        # llama.cpp's choice from the row count (`ne11 <= threshold`; fixed per graph bucket): mmvq, else mmq
        self.mmvq_rows = max((n for n in range(1, 9) if self.lib.use_mmvq(device, self.type, n)), default=0)
        if not self.lib.use_mmq(device, self.type, self.mmvq_rows + 1):
            raise RuntimeError(f"kernels.ggml_cuda: llama.cpp uses neither mmvq nor mmq for {facts['layout']} here")
        self.workspace = ctx.workspace(f"ggml_cuda[in={self.k}]", self.lib.workspace_bytes(device, self.k, CHUNK))

    def __call__(self, x: torch.Tensor, blocks: torch.Tensor) -> torch.Tensor:
        x, rows = x.contiguous(), len(x)
        y = torch.empty(rows, self.out, dtype=self.out_dtype, device=x.device)
        if rows == 0:
            return y
        if rows <= self.mmvq_rows:
            self.lib.mmvq(self.type, blocks, x, y, self.workspace)
            return y
        f32 = None if y.dtype == torch.float32 else torch.empty(min(rows, CHUNK), self.out, device=x.device)
        for start in range(0, rows, CHUNK):
            xs, ys = x[start:start + CHUNK], y[start:start + CHUNK]
            self.lib.mmq(self.type, blocks, xs, ys if f32 is None else f32[:len(xs)], self.workspace)
            if f32 is not None:
                ys.copy_(f32[:len(xs)])
        return y


class QEmbedding:
    def __init__(self, ext: Extension, ctx, *, layout: str, vocab: int, hidden: int, dtype: str) -> None:
        self.type, self.hidden, self.dtype = ggml_type(layout), hidden, getattr(torch, dtype)
        self.lib = ext(ctx.fingerprint.sm)

    def __call__(self, ids: torch.Tensor, blocks: torch.Tensor) -> torch.Tensor:
        y = torch.empty(len(ids), self.hidden, dtype=self.dtype, device=blocks.device)
        if len(ids):
            self.lib.get_rows(self.type, blocks, ids.to(torch.int32), y)
        return y


class LinearExperts:
    def __init__(self, ext: Extension, ctx, **facts) -> None:
        self.type, self.k, self.out = ggml_type(facts["layout"]), facts["in"], facts["out"]
        self.experts, self.top_k, self.swiglu = facts["experts"], facts["top_k"], facts["glu"] == "swiglu"
        self.out_y = self.out // 2 if self.swiglu else self.out
        self.lib, device = ext(ctx.fingerprint.sm), torch.device(ctx.device).index
        device = torch.cuda.current_device() if device is None else device
        # llama.cpp's choice per chunk from its token count (fixed per graph bucket): mmvq with ids, else mmq
        self.mmvq_tokens = max((n for n in range(1, 9) if self.lib.use_mmvq_id(device, self.type, n)), default=0)
        if not self.lib.use_mmq(device, self.type, self.mmvq_tokens + 1):
            raise RuntimeError(f"kernels.ggml_cuda: llama.cpp uses neither mmvq nor mmq for {facts['layout']} here")
        # its size depends on in, k (the id maps, q8_1 per slot) and E (expert_bounds): all three name it
        self.workspace = ctx.workspace(f"ggml_cuda.id[in={self.k},top_k={self.top_k},experts={self.experts}]",
                                       self.lib.workspace_bytes_id(device, self.k, CHUNK, self.top_k, self.experts))
        self.calls = {"mmvq": 0, "mmq": 0}           # per path, for tests (counted when called, not on graph replay)

    def __call__(self, x: torch.Tensor, parts: dict[str, torch.Tensor], ids: torch.Tensor) -> torch.Tensor:
        x, ids, blocks = x.contiguous(), ids.contiguous(), parts["blocks"]
        tokens = len(ids)
        y = torch.empty(tokens, self.top_k, self.out_y, device=x.device)
        gate_up = None          # mmq's f32 gate/up output before our SwiGLU, a per-call activation
        for start in range(0, tokens, CHUNK):
            xs, ids_c, ys = x[start:start + CHUNK], ids[start:start + CHUNK], y[start:start + CHUNK]
            if len(ids_c) <= self.mmvq_tokens:
                self.calls["mmvq"] += 1
                self.lib.mmvq_id(self.type, blocks, xs, ids_c, ys, self.swiglu, self.workspace)
                continue
            self.calls["mmq"] += 1
            if not self.swiglu:
                self.lib.mmq_id(self.type, blocks, xs, ids_c, ys, self.workspace)
                continue
            if gate_up is None:
                gate_up = torch.empty(min(tokens, CHUNK), self.top_k, self.out, device=x.device)
            self.lib.mmq_id(self.type, blocks, xs, ids_c, gate_up[:len(ids_c)], self.workspace)
            self.lib.swiglu(gate_up[:len(ids_c)], ys)
        return y


class MoeTopk:
    def __init__(self, ext: Extension, ctx, *, experts: int, top_k: int, renormalize: bool) -> None:
        self.top_k, self.renormalize, self.lib = top_k, renormalize, ext(ctx.fingerprint.sm)

    def __call__(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = logits.contiguous()
        weights = torch.empty(len(logits), self.top_k, device=logits.device)
        ids = torch.empty(len(logits), self.top_k, dtype=torch.int32, device=logits.device)
        if len(logits):
            self.lib.topk_moe(logits, weights, ids, self.renormalize)
        return weights, ids


class MoeCombine:
    def __init__(self, ext: Extension, ctx, **facts) -> None:
        self.lib = ext(ctx.fingerprint.sm)

    def __call__(self, y: torch.Tensor, weights: torch.Tensor, out: torch.Tensor) -> None:
        if len(y):
            self.lib.moe_combine(y.contiguous(), weights.contiguous(), out)


def _supports_qlinear(f: dict) -> bool:
    return (ggml_type(f["layout"]) is not None and f["in"] % 256 == 0 and f["act_dtype"] in DTYPES
            and f["out_dtype"] in DTYPES)


def _supports_qembedding(f: dict) -> bool:
    return ggml_type(f["layout"]) is not None and f["hidden"] % 256 == 0 and f["dtype"] in DTYPES


def _supports_linear_experts(f: dict) -> bool:
    # out_y a multiple of 4: mmvq with ids at one token bounds a block's rows (up to 4) by the slot stride, not out_y
    swiglu = f["glu"] == "swiglu"
    return (ggml_type(f["layout"]) is not None and f["in"] % 256 == 0 and f["act_dtype"] in DTYPES
            and f["out_dtype"] == "float32" and f["glu"] in ("none", "swiglu") and f["out"] % (8 if swiglu else 4) == 0
            and f["rows_per_token"] in (1, f["top_k"]))


def _supports_moe_topk(f: dict) -> bool:
    e = f["experts"]
    return 0 < e <= 512 and e & (e - 1) == 0 and 0 < f["top_k"] <= e     # topk-moe.cu's instances


def _supports_moe_combine(f: dict) -> bool:
    return f["dtype"] == "bfloat16" and f["y_dtype"] == "float32"


class Plugin:
    def register(self, regs) -> None:
        ext = Extension()
        for op, impl, supports in ((QLINEAR, QLinear, _supports_qlinear),
                                   (QEMBEDDING, QEmbedding, _supports_qembedding),
                                   (LINEAR_EXPERTS, LinearExperts, _supports_linear_experts),
                                   (MOE_TOPK, MoeTopk, _supports_moe_topk),
                                   (MOE_COMBINE, MoeCombine, _supports_moe_combine)):
            regs.kernels.add(op, KernelSpec(
                op=op, provider="kernels.ggml_cuda", impl=functools.partial(impl, ext),
                requires=Constraint(sms=frozenset(extension.ARCHS)), supports=supports,
                props=frozenset({"graph_safe", "deterministic"}), priority=10))
