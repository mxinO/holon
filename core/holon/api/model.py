"""Model plugin contract: recipes, ModelSpec, weight maps, units, BuildCtx (design §5.1).

A model is primarily a pure function `recipe(cfg, source) -> ModelSpec` that composes units by `Ref`. It imports only
`holon.api`; its identity is `canonical_hash(spec)`.
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Protocol

from .terms import Ref

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    import torch

    from .frontend import FrontendBundle
    from .memory import Pool
    from .state import StateSpec


@dataclasses.dataclass(frozen=True, slots=True)
class LayerDecl:
    mixer: Ref
    ffn: Ref
    kind: str = ""  # free label for selectors and telemetry, e.g. "full_attention", "linear_attention"


@dataclasses.dataclass(frozen=True, slots=True)
class DraftHeadDecl:
    """A model-provided drafter (e.g. a multi-token-prediction head): its layers, consumed taps, shared modules."""

    name: str
    layers: tuple[LayerDecl, ...]
    taps: tuple[str, ...] = ("pre_norm_last",)
    shares: tuple[str, ...] = ("embed", "lm_head")


@dataclasses.dataclass(frozen=True, slots=True)
class WeightRule:
    """Map checkpoint tensor names onto unit parameters, optionally through a registered value transform.

    `pattern` and `target` use `{i}` for the layer index and `{name}` for any other captured span.
    """

    pattern: str
    target: str
    transform: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class WeightMap:
    source: str                        # weight-source kind, e.g. "hf_safetensors"
    rules: tuple[WeightRule, ...]
    root: str = ""                     # prefix stripped from checkpoint names
    skip: tuple[str, ...] = ()         # prefixes ignored (e.g. a tied output head, a vision tower)


@dataclasses.dataclass(frozen=True, slots=True)
class ModelSpec:
    hidden: int
    vocab: int
    layers: tuple[LayerDecl, ...]
    block: Ref                          # residual wiring unit, e.g. Ref("std.blocks.prenorm@1")
    embed: Ref
    norm: Ref
    lm_head: Ref
    weights: WeightMap
    taps: tuple[str, ...] = ()          # extra named export points
    draft_heads: tuple[DraftHeadDecl, ...] = ()
    position_fn: str | None = None      # e.g. a multi-section rope; default: arange


class BuildCtx(Protocol):
    """Everything a unit may use while building. Units get other units only through `unit` (invariant 2)."""

    dtype: torch.dtype
    options: Mapping[str, Any]   # deployment options units may read (e.g. "page_size"); unknown keys → defaults

    def unit(self, ref: Ref, **site: Any) -> Unit: ...

    def state(self, spec_name: str, layer: int) -> tuple[StateSpec, Pool] | None:
        """The spec and pool holding state `spec_name` for `layer` (e.g. its KV pages); None in a model-only build,
        where units run their stateless reference path."""

    def static(self, name: str, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """A static device buffer, stable across graph replays (design §7.4)."""

    def workspace(self, name: str, nbytes: int) -> torch.Tensor:
        """Kernel scratch memory, accounted by the ledger (design §5.6 rule 1)."""


class Unit(Protocol):
    """A built component (op, norm, block, mixer, ffn …), typically a `torch.nn.Module`.

    If a unit defines `finalize_weights(ctx: BuildCtx)`, the core calls it once after loading and before graph
    capture (design §5.1); derived buffers are created there.
    """

    def forward(self, *args: Any) -> Any: ...


class WeightSource(Protocol):
    """Provisional (settled with spike S2 / M0.1): raw checkpoint tensors plus their metadata."""

    kind: str
    arch: str                          # resolved to the model plugin providing `arch.<arch>`
    eos_token_ids: frozenset[int]

    def config(self) -> Mapping[str, Any]: ...

    def tensors(self) -> Iterator[tuple[str, torch.Tensor]]: ...


class ModelPlugin(Protocol):
    """Declares the architectures it builds as `arch.<name>@1` resources in its manifest."""

    def recipe(self, cfg: Mapping[str, Any], source: str) -> ModelSpec: ...

    def bundle(self) -> FrontendBundle: ...
