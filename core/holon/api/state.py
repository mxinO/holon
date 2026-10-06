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

    from .memory import Pool
    from .model import BuildCtx


@dataclasses.dataclass(frozen=True, slots=True)
class Paged:
    unit_bytes: int
    tokens_per_unit: int


@dataclasses.dataclass(frozen=True, slots=True)
class Slot:
    unit_bytes: int
    draft_slots: int = 0         # extra slots for verify intermediates


@dataclasses.dataclass(frozen=True, slots=True)
class StateSpec:
    name: str                    # unique within the model, e.g. "attn.kv", "linear.state"
    kind: str                    # StateKind registry key
    shape: Paged | Slot
    layers: tuple[int, ...]
    layout: str                  # negotiated with kernels; format settled with spike S1
    alignment: int = 1           # prefix-hit granularity when intrinsic (e.g. compressed-KV ratio)
    page_group: str | None = None  # specs sharing page ids, one page table and one unit count


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
    """What a StateKind publishes into the PrefixIndex for one block: units in its pool, on some tier."""

    units: tuple[int, ...]
    tier: str = "vram"


class SeqView(Protocol):
    """Read-only view of one request, as state kinds see it."""

    slot: int
    num_computed: int            # host mirror (upper bound while a step is in flight)
    tokens: tuple[int, ...]

    def units(self, spec: str) -> tuple[int, ...]: ...


class PrefixView(Protocol):
    def payload(self, block: int) -> Payload | None: ...

    def blocks(self, upto: int) -> int: ...


class CommitView(Protocol):
    """Host-side results of one step, after its device→host copy."""

    slots: np.ndarray            # [B]
    n_accept: np.ndarray         # [B] accepted draft tokens (0 without speculation)
    seqs: list[SeqView]


class Rewindable(Protocol):
    """Per-request state that speculative decoding may roll back (design §7.5)."""

    verify_mode: Literal["trivial", "checkpoint_slots", "replay", "snapshot"]

    def commit(self, slots: torch.Tensor, n_accept: torch.Tensor, path: torch.Tensor | None = None) -> None:
        """Device-side, graph-capturable; runs inside the sample graph."""

    def host_commit(self, view: CommitView) -> None:
        """CPU-side bookkeeping (frees, trims) after the step's results reach the host."""


class StateKind(Rewindable, Protocol):
    """The meaning of one bound state spec. Callbacks are per request; the core vectorises over SoA batches."""

    def units_needed(self, seq: SeqView, new_tokens: int) -> int: ...

    def reclaimable(self, seq: SeqView) -> list[int]: ...

    def hit_length(self, idx: PrefixView, upto: int) -> HitLen: ...

    def publish(self, seq: SeqView, upto: int) -> tuple[Payload, list[MemOp]] | None: ...

    def adopt(self, seq: SeqView, payload: Payload) -> list[MemOp]: ...

    def fork(self, src: SeqView, dsts: list[SeqView]) -> list[MemOp]: ...


class StateKindFactory(Protocol):
    """Registered under the kind's name; the core binds one instance per spec."""

    def __call__(self, spec: StateSpec, pool: Pool, ctx: BuildCtx) -> StateKind: ...

