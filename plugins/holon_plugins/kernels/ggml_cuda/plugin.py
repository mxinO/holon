"""llama.cpp's quantized CUDA kernels (vendored at a pinned commit, M0.1 design §4.4) as `qlinear@1` and
`qembedding@1` kernels for ggml block layouts (`"ggml.<type>/gguf.native"`; op contracts: std.ops.kernels).

`qlinear` dispatches like llama.cpp, from the row count: `mmvq` up to the type's threshold on this device (it writes
bf16 or f32 directly), else `mmq` in 512-row chunks (llama.cpp's ubatch) into a per-call f32 tensor that one cast
turns into `out_dtype` (D5). Both quantize the activations to q8_1 in a workspace shared by every instance with the
same `in`. `qembedding` is llama.cpp's `get_rows`. With `in` a multiple of 256 (all supported shapes) no kernel
reads past the weight blocks, so they need no padding (L0 poisons the bytes after them).

The native extension is built ahead of time (`python -m holon_plugins.kernels.ggml_cuda.prebuild`); without it an
instance cannot be created and the error names that command.
"""
from __future__ import annotations

import functools

import torch
from holon.api import Constraint, KernelSpec

from . import extension

QLINEAR, QEMBEDDING = "qlinear", "qembedding"
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


def _supports_qlinear(f: dict) -> bool:
    return (ggml_type(f["layout"]) is not None and f["in"] % 256 == 0 and f["act_dtype"] in DTYPES
            and f["out_dtype"] in DTYPES)


def _supports_qembedding(f: dict) -> bool:
    return ggml_type(f["layout"]) is not None and f["hidden"] % 256 == 0 and f["dtype"] in DTYPES


class Plugin:
    def register(self, regs) -> None:
        ext = Extension()
        for op, impl, supports in ((QLINEAR, QLinear, _supports_qlinear),
                                   (QEMBEDDING, QEmbedding, _supports_qembedding)):
            regs.kernels.add(op, KernelSpec(
                op=op, provider="kernels.ggml_cuda", impl=functools.partial(impl, ext),
                requires=Constraint(sms=frozenset(extension.ARCHS)), supports=supports,
                props=frozenset({"graph_safe", "deterministic"}), priority=10))
