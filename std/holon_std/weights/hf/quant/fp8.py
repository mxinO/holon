"""`quant_method: fp8` (the DeepSeek-V3 convention; Qwen's FP8 checkpoints): an e4m3 `weight` and `weight_scale_inv`,
one multiplier per `weight_block_size` block (a dequantization multiplier despite its name, BF16 or F32);
`modules_to_not_convert` stay dense. The activation precision is the deployment's (D3): `activation_scheme` is not
read."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import Dialect


def dialect(qc: Mapping[str, Any]) -> Dialect:
    block = qc.get("weight_block_size")
    if qc.get("fmt", "e4m3") != "e4m3" or not block or len(block) != 2:
        raise ValueError(f"fp8 quantization config {dict(qc)}: only e4m3 weights with 2-D block scales "
                         "(weight_block_size) are supported")
    return Dialect("fp8_e4m3", {"weight": "weight", "weight_scale_inv": "scale"},
                   ignore=tuple(qc.get("modules_to_not_convert") or ()), args={"block": tuple(block)})
