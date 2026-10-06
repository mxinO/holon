"""Number semantics, byte layouts and typed weights (design §5.5).

`Scheme` says what the stored numbers mean; `Layout` says how the bytes are arranged; `QTensor` binds a layout to
concrete tensors. Repacking between layouts is done by registered converters.
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import torch


@dataclasses.dataclass(frozen=True, slots=True)
class Scale:
    dtype: str                       # e.g. "fp16", "e4m3"
    group: tuple[int, int]           # (rows, cols) covered by one scale; (-1, -1) = per tensor
    level: int = 1                   # 1 = applied to elements; 2 = a scale of scales (double quantization)


@dataclasses.dataclass(frozen=True, slots=True)
class Scheme:
    """What the stored numbers mean; `elem` names the element format, interpreted by its registered SchemeImpl."""

    elem: str
    scales: tuple[Scale, ...] = ()
    zero_point: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class Layout:
    """How the bytes are arranged, e.g. `Layout(Scheme("bf16"), "rowmajor")`. `params` are sorted (key, value)."""

    scheme: Scheme
    name: str
    params: tuple[tuple[str, Any], ...] = ()
    readable_by: frozenset[str] = frozenset({"cuda"})

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", tuple(sorted(self.params)))


@dataclasses.dataclass(slots=True, eq=False)
class QTensor:
    """A typed weight: logical shape + layout + the concrete tensors (parts) that hold it."""

    layout: Layout
    logical_shape: tuple[int, ...]
    parts: dict[str, torch.Tensor]


class SchemeImpl(Protocol):
    """Registered per `Scheme.elem`: the reference dequantization the judge's oracles rely on."""

    def dequant_ref(self, q: QTensor) -> torch.Tensor: ...


@dataclasses.dataclass(frozen=True, eq=False)
class Converter:
    """Repack a QTensor from one layout to another (registered by the plugin owning `dst`)."""

    src: str
    dst: str
    fn: Callable[..., QTensor]
    lossless: bool = True
    cost: float = 1.0


class Dialect(Protocol):
    """Provisional (settled with spike S2 / M0.1): a checkpoint's quantization conventions → source layouts."""

    name: str

    def detect(self, meta: Mapping[str, Any]) -> bool: ...

    def source_layout(self, tensor_name: str, meta: Mapping[str, Any]) -> Layout: ...
