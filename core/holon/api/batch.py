"""Requests and steps: Seq, the device SeqTable, BatchPlan, StepCtx (design §7.1, §7.2).

The device `SeqTable` is the source of truth for per-slot progress; the host `Seq.num_computed` is an optimistic
mirror (an upper bound while steps are in flight). Host plans therefore carry bounds, and planners turn them into
exact device metadata inside the graph.
"""
from __future__ import annotations

import dataclasses
import enum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np
    import torch

    from .decoding import Sampler
    from .state import InStepPublish, MemOp


class Status(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    BLOCKED = "blocked"      # waiting on an async future (state load, transfer, its logits processor's `ready`)
    PREEMPTED = "preempted"
    FINISHED = "finished"


class RowKind(enum.IntEnum):
    """What a step's row computes (`BatchPlan.row_kind`)."""

    PREFILL = 0              # a prompt chunk (its last one may sample)
    DECODE = 1               # one input token, sampled
    VERIFY = 2               # an input token and its drafts, all sampled
    PADDING = 3              # graph padding past the step's rows (`padded`): computes nothing anyone reads


class TokenSrc(enum.IntEnum):
    """Where a query row's input token comes from (overlap-safe placeholders)."""

    HOST = 0
    DEVICE_LAST = 1          # the token sampled by the previous step, still on device


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


SEED_MASK = 2**63 - 1


def sequence_seed(seed: int, index: int) -> int:
    """The seed of sequence `index` of a request seeded `seed`: sequence 0 keeps it, the others get it mixed with
    their index by a fixed 64-bit function (never the interpreter's hash), so a sequence resubmitted alone (the
    supervisor) draws what it would have drawn beside its siblings."""
    base = seed & SEED_MASK
    if index == 0:
        return base
    x = (base + index + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF          # splitmix64
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    return (x ^ (x >> 31)) & SEED_MASK


@dataclasses.dataclass(slots=True, eq=False)
class Seq:
    rid: str
    tokens: list[int]                    # committed tokens only
    params: SamplingParams
    prompt_len: int = 0
    index: int = 0                       # which of the request's `n` sequences
    slot: int = -1                       # persistent device row
    num_computed: int = 0                # host mirror; exact after commit
    pending: int = 0                     # tokens sampled on the device, not yet committed (an upper bound)
    inexact: int = 0                     # uncommitted verify rows: num_computed is then only an upper bound
    n_draft: int = 0                     # real drafts its last commit wrote for its next step (host copy)
    drafts: list[int] = dataclasses.field(default_factory=list)   # and the drafts, when the logits processor masks it
    constrained: bool = False            # a logits processor masks its rows (launched only with every token committed)
    status: Status = Status.WAITING
    finish_reason: str | None = None
    branch_at: int = 0                   # set at first admission, kept on re-admissions (`SeqView.branch_at`)
    matched: int = 0                     # likewise: its prefix match before any trim (`SeqView.matched`)

    def __post_init__(self) -> None:
        if not self.prompt_len:
            self.prompt_len = len(self.tokens)


@dataclasses.dataclass(slots=True, eq=False)
class SeqTable:
    """Device-authoritative per-slot state, written by the step's sample stage (inside the graph when captured):
    while steps are in flight it, not the host mirror, holds the true lengths.

    Draft probabilities (with a drafter that samples its drafts, `Drafter.q_support` = Kq): draft j of a slot was drawn
    from q_j(t) = the sum of `draft_q[slot, j, c]` over the entries c with `draft_q_ids[slot, j, c]` = t, zero off
    them — exactly the law the drafter sampled it from, given the drafts before it. Rows past `n_draft` are never
    read. None: drafts are deterministic (q one-hot at the draft)."""

    num_computed: torch.Tensor           # [S] int64: tokens whose state is computed; the next input's position
    last_token: torch.Tensor             # [S] int64: the next input token
    n_accept: torch.Tensor               # [S] int64: drafts accepted by the slot's last sampled row (0 unless a verify)
    draft_tokens: torch.Tensor           # [S, k_max] int64: drafts for positions num_computed + 1 + j
    n_draft: torch.Tensor                # [S] int64: how many of draft_tokens are real (verifiers accept no more)
    draft_q_ids: torch.Tensor | None = None    # [S, k_max, Kq] int64: the support of each draft's law …
    draft_q: torch.Tensor | None = None        # [S, k_max, Kq] fp32: … and its probabilities


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
    mem_ops: list[MemOp] = dataclasses.field(default_factory=list)
    publishes: list[InStepPublish] = dataclasses.field(default_factory=list)   # payloads the forward fills
    draft_k: int = 0                     # drafts this step's rows verify (verify width − 1), or a prefill step's drafts
                                         # for its prompt-ending rows; 0: no speculation
    n_decode: int = 0                    # a mixed step's leading decode rows (the rest are prefill chunks); 0: unmixed
    row_kind: np.ndarray | None = None   # [B] RowKind per row, on every step the scheduler plans (graph padding:
                                         # PADDING); None: not known (startup profile and capture plans)


@dataclasses.dataclass(slots=True, eq=False)
class DeviceBatch:
    """Static device buffers for one step's shape; positions and lengths are exact (written on device)."""

    input_ids: torch.Tensor
    positions: torch.Tensor
    query_start: torch.Tensor            # [B+1]
    slots: torch.Tensor                  # [B]
    logits_index: torch.Tensor           # tokens whose logits are computed (the last of each sampled row)
    logits_rows: torch.Tensor            # [R] batch row of each logits_index entry


@dataclasses.dataclass(slots=True, eq=False)
class Batch:
    plan: BatchPlan
    dev: DeviceBatch


class StepCtx(Protocol):
    """What units see during one step."""

    batch: Batch
    seq_table: SeqTable
    sampler: Sampler             # the step's sampler (verifiers draw through it)

    def meta(self, group: str) -> Any:
        """Metadata prepared this step for a state group (a page group or state spec name)."""

    def pool(self, name: str) -> Any:
        """A memory pool by name (e.g. the pool of a payload this step's forward fills)."""

    def view(self, namespace: str) -> StepCtx:
        """This step as the units of auxiliary model `namespace` (`BuildCtx.aux_model`) see it: state groups and
        in-step publishes under their local names."""

    def substep(self, q: int, offset: int) -> StepCtx:
        """A drafter's next sub-step (in the order `Drafter.substeps` declared): q query tokens for each sampled row of
        this step at positions `num_computed[slot] + offset + j` (num_computed already advanced), with buffers and
        planned metadata of its own; the drafter fills its `batch.dev.input_ids`."""
