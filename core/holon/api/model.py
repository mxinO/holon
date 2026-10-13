"""Model plugin contract: recipes, ModelSpec, weight maps, units, BuildCtx (design §5.1).

A model is primarily a pure function `recipe(cfg, source) -> ModelSpec` that composes units by `Ref`. It imports only
`holon.api`; its identity is `canonical_hash(spec)`.
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Protocol

from .terms import Ref

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping

    import torch

    from .batch import StepCtx
    from .frontend import FrontendBundle
    from .memory import Pool
    from .ops import KernelSpec
    from .platform import Fingerprint
    from .quant import QTensor
    from .state import StateSpec


@dataclasses.dataclass(frozen=True, slots=True)
class LayerDecl:
    mixer: Ref
    ffn: Ref
    kind: str = ""  # free label for selectors and telemetry, e.g. "full_attention", "linear_attention"


@dataclasses.dataclass(frozen=True, slots=True)
class WeightRule:
    """Map checkpoint tensor names onto unit parameters, optionally through a registered value transform.

    `pattern` and `target` use `{i}` for the layer index and `{name}` for any other captured span. With `stack`
    naming a capture (e.g. "e"), its integer indexes dim 0 of the target: one tensor per index fills a stack.
    """

    pattern: str
    target: str
    transform: str | None = None
    stack: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class WeightMap:
    source: str                        # weight-source kind, e.g. "hf_safetensors"
    rules: tuple[WeightRule, ...]
    root: str = ""                     # prefix stripped from checkpoint names
    skip: tuple[str, ...] = ()         # prefixes ignored (e.g. a tied output head, a vision tower)
    only: tuple[str, ...] = ()         # when set, the only prefixes read (a model sharing a checkpoint with another)


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
    position_fn: str | None = None      # e.g. a multi-section rope; default: arange


@dataclasses.dataclass(frozen=True, slots=True)
class AuxSpec:
    """A model built beside the target for a plugin (e.g. a draft head): `unit` builds it, given `layers` (numbered
    from 0 within it; their mixers' state is bound under the aux model's namespace); `weights` loads it."""

    unit: Ref
    layers: tuple[LayerDecl, ...]
    weights: WeightMap


class BuildCtx(Protocol):
    """Everything a unit may use while building. Units get other units only through `unit` (invariant 2)."""

    device: torch.device
    dtype: torch.dtype
    fingerprint: Fingerprint
    options: Mapping[str, Any]   # deployment options units may read (e.g. "page_size"); unknown keys → defaults

    def unit(self, ref: Ref, **site: Any) -> Unit: ...

    def select_kernel(self, op: str, *, site: str, **facts: Any) -> KernelSpec:
        """The kernel for versioned `op` (e.g. "attention.paged_decode@1") given canonical `facts` (shapes,
        dtypes and layouts as plain values): chosen among available kernel plugins by device, `supports(facts)`
        and priority, or pinned by the deployment; recorded for the lock under `site` (design §5.4). The caller
        creates its own instances with `spec.impl(ctx, **facts)` — instances hold plans, so they are not shared."""

    def state(self, spec_name: str, layer: int) -> tuple[StateSpec, Pool] | None:
        """The spec and pool holding state `spec_name` for `layer` (e.g. its KV pages); None in a model-only build,
        where units run without state (attention over each whole sequence)."""

    def static(self, name: str, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """A static device buffer, stable across graph replays (design §7.4)."""

    def workspace(self, name: str, nbytes: int) -> torch.Tensor:
        """Kernel scratch memory, accounted by the ledger (design §5.6 rule 1)."""

    def pool(self, name: str, *, unit_bytes: int, segments: int = 1, min_units: int = 0,
             pref_units: int = 0) -> Pool:
        """A plugin's own pool of units, sized by the ledger with the fixed pools (design §6.1): `pref_units` when
        the budget allows, else `min_units`."""

    def aux_model(self, namespace: str, spec: AuxSpec, tensors: Iterable[tuple[str, torch.Tensor | QTensor]]
                  ) -> torch.nn.Module:
        """Build, state-bind and load an auxiliary model (engine builds only). Its units see a child context in which
        state names and planner groups are `<namespace>.<name>`; at step time `StepCtx.view(namespace)` gives them the
        same mapping."""

    def tap(self, layer: int, consume: Callable[[torch.Tensor, StepCtx], None]) -> None:
        """Have the target's trunk call `consume(h, ctx)` every step with its residual stream after layer `layer` (h
        [T, H], every token of the step: the next layer's input, before its norm), inside the step's graph when it is
        captured (engine builds: a drafter registers its taps while it is built). `h` is valid during the call only
        (the next layer may reuse it): the consumer keeps what it needs in its own static buffers — a copy, or an fp32
        accumulation."""


class Unit(Protocol):
    """A built component (op, norm, block, mixer, ffn …), typically a `torch.nn.Module`.

    If a unit defines `finalize_weights(ctx: BuildCtx)`, the core calls it once after loading and before graph
    capture (design §5.1); derived buffers are created there. If it defines `bind_kernels(ctx: BuildCtx)`, the core
    calls it after every unit has finalized (merges included): kernels that depend on the loaded weights (e.g. a
    quantized layout adopted with `adopt(ctx, name, qtensor)`) are selected there.
    """

    def forward(self, *args: Any) -> Any: ...


class WeightSource(Protocol):
    """A checkpoint: its tensors plus the metadata to build and serve it. Its plugin's manifest declares `detect`
    patterns (`[support]`) so the core can tell which source reads a path without importing anything."""

    kind: str
    name: str                          # the display name, cheap (no hashing): the `<name>` of `identity()`
    arch: str                          # resolved to the model plugin providing `arch.<arch>`
    eos_token_ids: frozenset[int]      # the stop set

    def config(self) -> Mapping[str, Any]:
        """HF-style configuration (the keys recipes read)."""

    def tensors(self) -> Iterator[tuple[str, torch.Tensor | QTensor]]:
        """Checkpoint tensors by checkpoint name; quantized ones as `QTensor`s (adopted by their unit, design §5.5)."""

    def tokenizer(self) -> Any:
        """The checkpoint's tokenizer with its chat template (a transformers fast tokenizer)."""

    def identity(self) -> str:
        """The checkpoint's identity for plans and lock histories (design §4.3): `<name>@<build>`, the build a
        hash of the files that define the weights. Path segments joined by "/" (e.g. a hub id) are allowed."""


class ModelPlugin(Protocol):
    """Declares the architectures it builds as `arch.<name>@1` resources in its manifest."""

    def recipe(self, cfg: Mapping[str, Any], source: str) -> ModelSpec: ...

    def bundle(self, source: WeightSource) -> FrontendBundle:
        """Tokenizer, chat template and parsers for the checkpoint `source` reads."""

    # Optional: `draft_head(cfg, source, name) -> AuxSpec | None`, a draft head stored with the checkpoint (e.g. a
    # multi-token-prediction layer), built only when a drafter asks for it (`TargetView.draft_head`).
