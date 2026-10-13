"""Quantized Hugging Face checkpoints (docs/m06-lowp.md D2, §4.4): one dialect per producer convention, picked by the
checkpoint's `quantization_config`, names one scheme's parts; the source yields each quantized module as one
`QTensor` of the layout `<elem>/hf` (normalized parts, `reference.py` dequantizes them) under `<module>.weight`, so
the recipes' weight rules map it unchanged.

A module is recognised by its tensors, not by the config's targets, which name classes (`Linear`) a weight source
cannot see: it is quantized when it holds a dialect part other than a float `weight`. A quantized module must hold
every part of its dialect and not be ignored by the config; an ignored module is then dense by construction.
"""
from __future__ import annotations

import dataclasses
import fnmatch
import re
from collections.abc import Callable, Mapping
from typing import Any

import torch
from holon.api import Layout, QTensor, Scheme

from . import compressed_tensors, fp8, modelopt, quark
from .base import NORMALIZE, Dialect

DIALECTS = {"fp8": fp8.dialect, "compressed-tensors": compressed_tensors.dialect, "modelopt": modelopt.dialect,
            "quark": quark.dialect}
_FLOAT = {"F16", "BF16", "F32", "F64"}         # safetensors dtypes of a dense weight


@dataclasses.dataclass(frozen=True)
class Module:
    """A quantized module: its name and its parts' checkpoint tensors by normalized role, the primary part first."""

    name: str
    parts: dict[str, str]


def dialect(qc: Mapping[str, Any]) -> Dialect:
    method = qc.get("quant_method")
    if method not in DIALECTS:
        raise ValueError(f"quantization_config quant_method {method!r}: not supported (one of {sorted(DIALECTS)})")
    return DIALECTS[method](qc)


def group(d: Dialect, header: list[tuple[str, str]]) -> tuple[list[str | Module], dict[str, float]]:
    """The checkpoint's tensors (name, safetensors dtype; in shard order) as what to yield, in shard order: a dense
    tensor's name, or a quantized module where its primary part is; and the dropped tensors whose value is checked
    (name -> value). Dropped tensors are never yielded."""
    dtype = dict(header)
    by_module: dict[str, dict[str, str]] = {}            # module -> {suffix: tensor name}
    for name, _ in header:
        module, _, suffix = name.rpartition(".")
        if suffix in d.parts or suffix in d.dropped:
            by_module.setdefault(module, {})[suffix] = name
    modules = {}
    for module, tensors in by_module.items():
        if not any(s in d.parts and (s != "weight" or dtype[n] not in _FLOAT) for s, n in tensors.items()):
            continue                                     # dense: a float `weight`, if any
        if ignored(module, d.ignore):
            raise ValueError(f"{module}: quantized, but the quantization config leaves it dense")
        if missing := [s for s in d.parts if s not in tensors]:
            raise ValueError(f"{module}: a quantized module without {missing} (has {sorted(tensors)})")
        modules[module] = Module(module, {role: tensors[s] for s, role in d.parts.items()})
    primary = next(iter(d.parts))
    plan: list[str | Module] = []
    for name, _ in header:
        module, _, suffix = name.rpartition(".")
        if suffix in d.dropped:
            continue
        if module not in modules:
            plan.append(name)
        elif suffix == primary:
            plan.append(modules[module])
        elif suffix not in d.parts:
            plan.append(name)                            # e.g. a bias: the recipe maps it, or refuses it
    checks = {n: v for m, tensors in by_module.items() for s, n in tensors.items()
              if (v := d.dropped.get(s)) is not None}
    return plan, checks


def load(d: Dialect, module: Module, get: Callable[[str], torch.Tensor]) -> QTensor:
    """The module's parts read (each once) and normalized: reciprocals inverted in f32, as llama.cpp's converter does
    (≤ 1 ulp from dividing by them), dtypes, shapes and bytes checked."""
    raw = {role: get(name) for role, name in module.parts.items()}
    raw |= {role: 1.0 / raw[role].float() for role in d.inverted}
    try:
        parts, shape = NORMALIZE[d.elem](**raw, **d.args)
    except ValueError as e:
        raise ValueError(f"{module.name}: {e}") from None
    return QTensor(Layout(Scheme(d.elem), "hf"), shape, parts)


def ignored(module: str, patterns: tuple[str, ...]) -> bool:
    """`module` is named by a pattern: itself, a dotted suffix or prefix of it (transformers' `modules_to_not_convert`
    rule), a `*` wildcard (ModelOpt, Quark) or `re:<regex>` (compressed-tensors)."""
    for p in patterns:
        if p.startswith("re:"):
            if re.match(p[3:], module):
                return True
        elif fnmatch.fnmatchcase(module, p) or module.endswith("." + p) or module.startswith(p + "."):
            return True
    return False
