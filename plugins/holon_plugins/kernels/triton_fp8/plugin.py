"""Our Triton matmuls: FP8 ones as the `qlinear@2` kernel for `fp8_e4m3/hf` weights with 128 × 128 block scales (op
contract: std.ops; docs/m06-lowp.md D12, §4.6, §4.7), on sm ≥ 89 (e4m3 `tl.dot`; below it the decode kernel would run,
the prefill one would not: FP8 there is v1). Dense only: expert stacks stay on the reference provider.

A dense weight's few-row matmul as `linear_rows@1` (`rows.py`; std.ops: a verify pass's rows), on sm ≥ 80.

Paths by row count, each in its own activation mode (`config["act"]`, D3): `decode` up to 16 rows, W8A16 split-K
(the activations as given, `bf16`); `prefill` above, W8A8 (`fp8`: e4m3 per row and 128 columns). Calls are counted
per path and mode (`calls["decode/bf16"]`, `calls["prefill/fp8"]`; counted when called, not on graph replay).
"""
from __future__ import annotations

import collections

import torch
from holon.api import Constraint, KernelSpec, QTensor

from . import kernel, rows

OP = "qlinear"
ACT = {"decode": "bf16", "prefill": "fp8"}
ACT_QUANT = ("a4", "a8")            # the lowest precision a deployment allows; this kernel's lowest is 8 bits


class QLinear:
    def __init__(self, ctx, **facts) -> None:
        gpu, self.k, self.n = ctx.fingerprint.gpu, facts["in"], facts["out"]
        self.decode_cfg = {r: kernel.decode_config(gpu, self.n, self.k, r) for r in (1, kernel.DECODE_ROWS)}
        self.prefill_cfg = kernel.prefill_config(gpu, self.n, self.k)
        self.out_dtype = getattr(torch, facts["out_dtype"])
        self.calls: collections.Counter[str] = collections.Counter()

    def prepare(self, q: QTensor) -> QTensor:
        """The weight as adopted, once its scales are checked to be one per 128 × 128 block (the dialects allow any
        block size the config declares)."""
        w, s = q.parts["weight"], q.parts["scale"]
        if tuple(s.shape) != (w.shape[0] // kernel.BLOCK, w.shape[1] // kernel.BLOCK) or s.dtype != torch.float32:
            raise ValueError(f"kernels.triton_fp8 reads 128 × 128 f32 block scales, not {list(s.shape)} {s.dtype} for "
                             f"a weight {list(w.shape)}")
        return q

    def __call__(self, x: torch.Tensor, parts: dict[str, torch.Tensor]) -> torch.Tensor:
        x, w, s = x.contiguous(), parts["weight"], parts["scale"]
        if len(x) <= kernel.DECODE_ROWS:
            self.calls[f"decode/{ACT['decode']}"] += 1
            return kernel.decode(x, w, s, self.out_dtype, self.decode_cfg[1 if len(x) == 1 else kernel.DECODE_ROWS])
        self.calls[f"prefill/{ACT['prefill']}"] += 1
        return kernel.prefill(x, w, s, self.out_dtype, self.prefill_cfg)


def _supports(f: dict) -> bool:
    return (f["layout"] == "fp8_e4m3/hf" and f["in"] % kernel.BLOCK == 0 and f["out"] % kernel.BLOCK == 0
            and f["act_dtype"] == "bfloat16" and f["out_dtype"] in ("bfloat16", "float32")
            and f["act_quant"] in ACT_QUANT)


class LinearRows:
    """`linear_rows@1` (std.ops): a dense weight's few-row matmul (`rows.py`)."""

    def __init__(self, ctx, **facts) -> None:
        pass

    def __call__(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return rows.linear_rows(x, weight)


class Plugin:
    def register(self, regs) -> None:
        regs.kernels.add(OP, KernelSpec(op=OP, provider="kernels.triton_fp8", impl=QLinear,
                                        requires=Constraint(sm_min=89), supports=_supports,
                                        props=frozenset({"graph_safe", "deterministic"}), config={"act": ACT}))
        regs.kernels.add("linear_rows", KernelSpec(
            op="linear_rows", provider="kernels.triton_fp8", impl=LinearRows, requires=Constraint(sm_min=80),
            supports=lambda f: f["in"] % rows.BK == 0 and f["out"] <= rows.OUT_MAX
            and f["dtype"] in ("bfloat16", "float16"),
            props=frozenset({"graph_safe", "deterministic"})))
