"""AMD Quark (`quant_method: quark`): MXFP4 only — the global weight config `fp4`, groups of 32, `e8m0` scales: a
packed e2m1 `weight` and a U8 `weight_scale`; `exclude` stays dense. Quark packs element 2i in the low nibble, and its
`pack_method: reorder` moves nothing for fp4 (amd-quark 0.13 `Pack_fp4`). Per-layer configs are not supported."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import Dialect


def dialect(qc: Mapping[str, Any]) -> Dialect:
    weight = (qc.get("global_quant_config") or {}).get("weight") or {}
    scheme = (weight.get("dtype"), weight.get("group_size"), weight.get("scale_format"))
    if scheme != ("fp4", 32, "e8m0") or qc.get("layer_quant_config") or qc.get("layer_type_quant_config"):
        raise ValueError(f"Quark weights {scheme}: only MXFP4 (fp4, groups of 32, e8m0 scales) in the global config "
                         "is supported")
    return Dialect("mxfp4", {"weight": "weight", "weight_scale": "scale"}, ignore=tuple(qc.get("exclude") or ()))
