"""Number semantics, byte layouts and typed weights (design §5.5).

`Scheme` says what the stored numbers mean; `Layout` says how the bytes are arranged; `QTensor` binds a layout to
concrete tensors. The plugins that read a layout interpret it; the core only carries it.
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclasses.dataclass(frozen=True, slots=True)
class Scheme:
    """What the stored numbers mean; `elem` names the element format (its scales included), interpreted by the
    plugins that read it."""

    elem: str


@dataclasses.dataclass(frozen=True, slots=True)
class Layout:
    """How the bytes are arranged, e.g. `Layout(Scheme("bf16"), "rowmajor")`."""

    scheme: Scheme
    name: str


@dataclasses.dataclass(frozen=True, slots=True)
class Extent:
    """Where a part's bytes lie unmodified in a checkpoint file, so they can be read again without the part."""

    path: str
    offset: int
    nbytes: int


@dataclasses.dataclass(slots=True, eq=False)
class QTensor:
    """A typed weight: logical shape + layout + the concrete tensors (parts) that hold it."""

    layout: Layout
    logical_shape: tuple[int, ...]
    parts: dict[str, torch.Tensor]
    extents: dict[str, Extent] = dataclasses.field(default_factory=dict)  # per part, from the weight source; none
                                                                          # for a part a kernel's `prepare` repacked
