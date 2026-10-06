"""Requests and steps: Seq, the device SeqTable, BatchPlan, StepCtx (design §7.1, §7.2).

The device `SeqTable` is the source of truth for per-slot progress; the host `Seq.num_computed` is an optimistic
mirror (an upper bound while steps are in flight). Host plans therefore carry bounds, and planners turn them into
exact device metadata inside the graph.
"""
from __future__ import annotations

import dataclasses
import enum
from collections.abc import Hashable
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np
    import torch

    from .state import MemOp


class Status(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    BLOCKED = "blocked"      # waiting on an async future (state load, transfer)
    PREEMPTED = "preempted"
    FINISHED = "finished"


class TokenSrc(enum.IntEnum):
    """Where a query row's input token comes from (overlap-safe placeholders)."""

    HOST = 0
    DEVICE_LAST = 1          # the token sampled by the previous step, still on device
    DEVICE_DRAFT = 2         # a draft token proposed on device


@dataclasses.dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    n: int = 1
    max_tokens: int | None = None
    ignore_eos: bool = False
    stop_token_ids: tuple[int, ...] = ()
    seed: int | None = None
    logprobs: int | None = None          # top-k logprobs per generated token
    prompt_logprobs: int | None = None   # top-k logprobs per prompt token
    ext: dict[str, Any] = dataclasses.field(default_factory=dict)  # namespaced plugin params


@dataclasses.dataclass(slots=True, eq=False)
class Seq:
    rid: str
    tokens: list[int]                    # committed tokens only
    params: SamplingParams
    slot: int = -1                       # persistent device row
    num_computed: int = 0                # host mirror; exact after commit
    status: Status = Status.WAITING
    finish_reason: str | None = None
    ext: dict[str, Any] = dataclasses.field(default_factory=dict)  # namespaced plugin state; core never reads


@dataclasses.dataclass(slots=True, eq=False)
class SeqTable:
    """Device-authoritative per-slot state, updated only inside the sample graph. Draft probabilities are added
    with speculative decoding (M0.4)."""

    num_computed: torch.Tensor           # [S] int32
    last_token: torch.Tensor             # [S] int64
    n_accept: torch.Tensor               # [S] int32
    draft_tokens: torch.Tensor           # [S, k_max] int64


@dataclasses.dataclass(frozen=True, slots=True)
class ShapeKey:
    """Graph bucket: padded sizes plus finite plugin-declared axes, e.g. (("spec.q", 4),)."""

    n_tokens: int
    n_seqs: int
    axes: tuple[tuple[str, Hashable], ...] = ()


@dataclasses.dataclass(slots=True, eq=False)
class BatchPlan:
    """Host-side plan for one step, struct-of-arrays over the step's rows (one row per scheduled Seq)."""

    step: int
    slots: np.ndarray                    # [B] int32
    n_query: np.ndarray                  # [B] query tokens per row (prefill chunk | 1 | 1+k)
    n_logits: np.ndarray                 # [B] rows needing logits (0 mid-prefill | 1 | 1+k)
    start_upper: np.ndarray              # [B] upper bound on the first query position; exact value in SeqTable
    token_src: np.ndarray                # [B] TokenSrc
    host_tokens: np.ndarray              # flat tokens for HOST rows
    shape: ShapeKey
    mem_ops: list[MemOp] = dataclasses.field(default_factory=list)
    ext: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(slots=True, eq=False)
class DeviceBatch:
    """Static device buffers for one ShapeKey; positions and lengths are exact (written on device)."""

    input_ids: torch.Tensor
    positions: torch.Tensor
    query_start: torch.Tensor            # [B+1]
    slots: torch.Tensor                  # [B]
    logits_index: torch.Tensor           # rows whose logits are computed
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)  # per planner


@dataclasses.dataclass(slots=True, eq=False)
class Batch:
    plan: BatchPlan
    dev: DeviceBatch


class StepCtx(Protocol):
    """What units see during one step."""

    batch: Batch
    seq_table: SeqTable

    def meta(self, group: str) -> Any:
        """Metadata prepared this step for a state group (a page group or state spec name)."""
