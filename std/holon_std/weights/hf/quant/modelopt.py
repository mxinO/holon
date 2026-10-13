"""NVIDIA ModelOpt (`quant_method: modelopt` in config.json, or hf_quant_config.json from its `producer`): NVFP4
only — a packed e2m1 `weight`, an e4m3 `weight_scale` per 16 and an F32 `weight_scale_2` per tensor; `ignore` (or
`exclude_modules`) stays dense. `input_scale`, the static W4A4 calibration, is dropped (no provider reads it, D2); the
FP8 KV cache's `k_scale` / `v_scale` must be 1.0, since the KV format is the deployment's (`kv.format`) and other
values would change the model."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import Dialect


def dialect(qc: Mapping[str, Any]) -> Dialect:
    if qc.get("quant_algo") != "NVFP4":
        raise ValueError(f"ModelOpt quant_algo {qc.get('quant_algo')!r}: only NVFP4 is supported")
    return Dialect("nvfp4", {"weight": "weight", "weight_scale": "scale", "weight_scale_2": "scale2"},
                   ignore=tuple(qc.get("ignore") or qc.get("exclude_modules") or ()),
                   dropped={"input_scale": None, "k_scale": 1.0, "v_scale": 1.0})
