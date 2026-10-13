"""compressed-tensors (`quant_method: compressed-tensors`; llm-compressor's and Red Hat's checkpoints): the format
names the parts — `nvfp4-pack-quantized`: `weight_packed`, an e4m3 `weight_scale` per 16 and `weight_global_scale`,
the reciprocal of ModelOpt's `weight_scale_2` (inverted at load); `mxfp4-pack-quantized`: `weight_packed` and an e8m0
`weight_scale` per 32; `float-quantized` with block weights: an e4m3 `weight` and a `weight_scale` per block. `ignore`
stays dense; `input_global_scale`, the static activation calibration, is dropped (D2)."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import Dialect

FORMATS = {
    "nvfp4-pack-quantized": ("nvfp4", {"weight_packed": "weight", "weight_scale": "scale",
                                       "weight_global_scale": "scale2"}),
    "mxfp4-pack-quantized": ("mxfp4", {"weight_packed": "weight", "weight_scale": "scale"}),
    "float-quantized": ("fp8_e4m3", {"weight": "weight", "weight_scale": "scale"}),
}


def dialect(qc: Mapping[str, Any]) -> Dialect:
    groups = (qc.get("config_groups") or {}).values()
    formats = ({qc.get("format")} | {g.get("format") for g in groups}) - {None}
    if len(formats) != 1 or (fmt := next(iter(formats))) not in FORMATS:
        raise ValueError(f"compressed-tensors format {sorted(formats)}: one of {sorted(FORMATS)} is supported")
    if qc.get("quantization_status", "compressed") != "compressed":
        raise ValueError(f"compressed-tensors quantization_status {qc['quantization_status']!r}: not compressed")
    elem, parts = FORMATS[fmt]
    args = {}
    if fmt == "float-quantized":
        blocks = {tuple((g.get("weights") or {}).get("block_structure") or ()) for g in groups
                  if (g.get("weights") or {}).get("strategy") == "block"}
        if len(blocks) != 1 or len(groups) != 1 or len(block := blocks.pop()) != 2:
            raise ValueError("compressed-tensors float-quantized: only block-scaled FP8 weights (one group, a 2-D "
                             "block_structure) are supported")
        args = {"block": block}
    return Dialect(elem, parts, ignore=tuple(qc.get("ignore") or ()), dropped={"input_global_scale": None},
                   inverted=frozenset({"scale2"}) if elem == "nvfp4" else frozenset(), args=args)
