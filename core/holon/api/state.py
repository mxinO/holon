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
    padding slot's units too, so graph padding rows have state to read and write)."""

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


class Rewindable(Protocol):
    """Anything that speculative decoding must roll back past rejected drafts (design §7.5): a module of the target or a
    loaded slot object defining `rewind` (an auxiliary model's modules are rolled back by the drafter that runs them,
    `Drafter.run`). Paged state needs none — it rewinds by length."""

    def rewind(self, ctx: StepCtx, path: torch.Tensor | None = None) -> None:
        """Device-side, graph-capturable: after the verifier wrote `ctx.seq_table.n_accept` for the step's verify rows,
        keep only what their accepted tokens produced. `path` stays None until tree verification."""


class StateKind(Protocol):
    """The meaning of one bound state spec. Callbacks are per request; the core vectorises over SoA batches."""

    def units_needed(self, seq: SeqView, new_tokens: int) -> int: ...

    # Optional: `capacity(seq) -> int`, the tokens the request's current units hold. When every bound kind offers it,
    # the core skips `units_needed` for steps that fit (most decode steps), which keeps per-request host work O(1).
    #
    # Optional, for state reused through payloads (design §6.3; paged state is shared by reference instead):
    # `cut(seq, start, n) -> int`, the tokens (1 … n) a prefill row starting at `start` may take, so a step can end
    # where the kind publishes; `next_publish(seq) -> int | None`, the position the kind publishes at next, so the
    # core calls `publish` only when a row's step ends there or, for a spec with an `in_step` grid, when it lies
    # inside the step on that grid (`in_step=True`: the spec's forward fills it); `payload_pool`, the name of the
    # pool its payloads use, which the prefix index may evict payloads from to make room. `hit_length` answers for
    # state the kind can restore; a non-paged kind without it allows no prefix hit.

    def reclaimable(self, seq: SeqView) -> list[int]: ...

    def hit_length(self, idx: PrefixView, upto: int) -> HitLen: ...

    def publish(self, seq: SeqView, upto: int, in_step: bool = False) -> tuple[Payload, list[MemOp]] | None: ...
    # `in_step`: `upto` lies inside the step and the spec's forward fills the payload (`StateSpec.in_step`), so no
    # MemOps; otherwise `upto` is where the step ends and MemOps fill it.

    def adopt(self, seq: SeqView, payload: Payload) -> list[MemOp]: ...

    def fork(self, src: SeqView, dsts: list[SeqView]) -> list[MemOp]: ...


class StateKindFactory(Protocol):
    """Registered under the kind's name; the core binds one instance per spec."""

    def __call__(self, spec: StateSpec, pool: Pool, ctx: BuildCtx) -> StateKind: ...

