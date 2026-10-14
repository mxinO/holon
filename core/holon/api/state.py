"""Per-request state: allocation shapes, state kinds, memory operations (design §6.2–§6.4).

The core allocates only two shapes — `Paged` (grows with tokens) and `Slot` (fixed size per request). What the state
*means* (prefix hits, rollback, forking) is defined by a `StateKind`, bound by the core to one spec and its pool, so
the core never names a layer type.
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    import numpy as np
    import torch

    from .batch import StepCtx
    from .memory import Pool
    from .model import BuildCtx


@dataclasses.dataclass(frozen=True, slots=True)
class Paged:
    unit_bytes: int
    tokens_per_unit: int


@dataclasses.dataclass(frozen=True, slots=True)
class Slot:
    """Fixed-size state per request: request slot s owns units s·(1 + draft_slots) … + draft_slots of its pool (the
    padding slot's units too, so graph padding rows have state to read and write; the padding slot is `max_seqs`,
    and its units are read only by padding rows, which start no request: a sink units may write to)."""

    unit_bytes: int
    draft_slots: int = 0         # extra slots for verify intermediates


@dataclasses.dataclass(frozen=True, slots=True)
class StateSpec:
    name: str                    # unique within the model, e.g. "attn.kv", "linear.state"
    kind: str                    # versioned StateKind, e.g. "paged@1"
    shape: Paged | Slot
    layers: tuple[int, ...]
    layout: str                  # negotiated with kernels
    segments: int = 1            # a unit is split into this many equal chunks, segment-major (e.g. one per layer)
    alignment: int = 1           # prefix-hit granularity when intrinsic (e.g. compressed-KV ratio)
    page_group: str | None = None  # specs sharing page ids, one page table and one unit count
    in_step: int = 0             # the grid (tokens) on which the spec's forward fills a payload inside a step
                                 # (`BatchPlan.publishes`); 0: payloads only where a step ends

    @property
    def group(self) -> str:
        """The key of this spec's page table and planner metadata (`StepCtx.meta`)."""
        return self.page_group or self.name


@dataclasses.dataclass(eq=False)
class Pages:
    """The core's view of one page group for one step (design §6.2).

    `table` is persistent — one row per request slot, a stable device address — and is what kernels index; the
    host mirror lets planners build kernel-specific metadata on the CPU (e.g. a CSR page list).
    """

    page: int                    # tokens per unit
    table: torch.Tensor          # [max_seqs, max_pages] int32 unit ids, row = request slot
    host_table: np.ndarray       # host mirror of `table`
    lens: torch.Tensor           # [B] tokens held per batch row after this step
    write_index: torch.Tensor    # [T] flat slot (unit * page + offset) where each new token's state is written


@dataclasses.dataclass(frozen=True, eq=False)
class MemOp:
    """An ordered device copy executed in-stream by the core (design §7.3)."""

    src_pool: str
    src_units: np.ndarray
    dst_pool: str
    dst_units: np.ndarray
    phase: Literal["pre_forward", "post_sample"] = "pre_forward"


@dataclasses.dataclass(frozen=True, slots=True)
class HitLen:
    length: int
    cost: float = 0.0            # tier cost: a hit on SSD is not scored like a hit in VRAM


@dataclasses.dataclass(frozen=True, slots=True)
class Payload:
    """What a StateKind publishes into the PrefixIndex for one block: units of the pool named `pool` (the index holds
    the publisher's reference on them), on some tier."""

    units: tuple[int, ...]
    pool: str
    tier: str = "vram"


@dataclasses.dataclass(frozen=True, slots=True)
class InStepPublish:
    """A payload this step's forward fills (`StateSpec.in_step`): spec `spec`'s state of batch row `row` after the
    row's first `offset` tokens of the step, into `payload`'s units (`StepCtx.pool(payload.pool)`)."""

    row: int
    spec: str
    offset: int
    payload: Payload


class SeqView(Protocol):
    """Read-only view of one request, as state kinds see it. `tokens` and `units()` are the live lists (no copies,
    this runs per request per step): kinds must not mutate them."""

    slot: int
    num_computed: int            # host mirror (upper bound while a step is in flight)
    tokens: tuple[int, ...]
    prompt_len: int

    def units(self, spec: str) -> tuple[int, ...]: ...


class PrefixView(Protocol):
    """A request's matched prefix chain as one kind sees it: `block` tokens per entry; `payload(i)` is the kind's
    payload at the end of entry i (position (i + 1)·block), `blocks(upto)` the entries within `upto` tokens."""

    block: int

    def payload(self, block: int) -> Payload | None: ...

    def blocks(self, upto: int) -> int: ...


def rewind_modules(module: torch.nn.Module, ctx: StepCtx) -> None:
    """Roll back every submodule of `module` that defines `rewind` (the core does this for the target; a drafter for
    the auxiliary models it runs, with `ctx.view(namespace)`)."""
    for m in module.modules():
        if callable(getattr(m, "rewind", None)):
            m.rewind(ctx)


class StateKind(Protocol):
    """The meaning of one bound state spec. Callbacks are per request; the core vectorises over SoA batches."""

    def units_needed(self, seq: SeqView, new_tokens: int) -> int: ...

    # Optional hooks (`holon.api.hooks`): KnowsCapacity, CutsPrefill, PublishesPayloads, PayloadPool,
    # RestoresPayloads.

    def reclaimable(self, seq: SeqView) -> list[int]: ...

    def fork(self, src: SeqView, dsts: list[SeqView]) -> list[MemOp]: ...


class StateKindFactory(Protocol):
    """Registered under the kind's name; the core binds one instance per spec."""

    def __call__(self, spec: StateSpec, pool: Pool, ctx: BuildCtx) -> StateKind: ...

