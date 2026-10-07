"""Scheduling, sampling and speculative decoding roles (design §7.3, §7.6).

Speculative decoding is four roles: a Drafter proposes, a DraftPolicy picks the source and k, a PlanTransform adds the
draft rows, and a Verifier replaces the Sampler for those rows. Rollback is the core's commit fan-out to Rewindables.
Ordering, overlap depth and shape axes of transforms and drafters are declared in their manifests.
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

if TYPE_CHECKING:
    from collections.abc import Hashable

    import numpy as np
    import torch

    from .batch import BatchPlan, SamplingParams, Seq, StepCtx


@dataclasses.dataclass(slots=True, eq=False)
class SampleOut:
    """Sampled tokens plus the logprobs requested by `SamplingParams.logprobs` (device tensors)."""

    tokens: torch.Tensor                       # [R]
    logprobs: torch.Tensor | None = None       # [R] logprob of each sampled token
    top_ids: torch.Tensor | None = None        # [R, k]
    top_logprobs: torch.Tensor | None = None   # [R, k]


class SchedulePolicy(Protocol):
    def order(self, waiting: list[Seq]) -> list[Seq]: ...

    def victim(self, running: list[Seq]) -> Seq | None: ...

    def mix_phases(self) -> bool:
        """Whether prefill and decode rows may share one step (v0 FCFS: False)."""


class PlanTransform(Protocol):
    def transform(self, plan: BatchPlan, seqs: list[Seq]) -> BatchPlan:
        """Host-side: rewrite the step's plan; `seqs` are the scheduled requests, row-aligned with the plan."""


class LogitsProcessor(Protocol):
    argmax_invariant: ClassVar[bool]   # greedy rows may skip this processor

    def host_input(self, plan: BatchPlan) -> Any | None:
        """Optional CPU-built input (e.g. a grammar bitmask), produced between forward and sample."""

    def apply(self, logits: torch.Tensor, host_input: Any, ctx: StepCtx) -> torch.Tensor: ...


class Sampler(Protocol):
    """One per engine, holding per-slot parameters (device rows once sampling is graph-captured)."""

    def admit(self, slot: int, params: SamplingParams) -> None:
        """A request now owns `slot`."""

    def sample(self, logits: torch.Tensor, ctx: StepCtx) -> SampleOut:
        """`logits` [R, V] fp32 for the plan rows with `n_logits > 0`, in row order."""

    def variant(self, slots: np.ndarray) -> Hashable:
        """The host-side choices `sample` makes for these slots — a step with the same variant runs the same device
        work, so it may replay a captured graph."""


class SamplerPlugin(Protocol):
    """The `sampler` slot's plugin object."""

    def sampler(self, max_seqs: int, device: torch.device) -> Sampler: ...


class Verifier(Protocol):
    def verify(self, logits: torch.Tensor, ctx: StepCtx) -> tuple[SampleOut, torch.Tensor]:
        """`logits` [R * (1+k), V]: the 1+k rows of each verify row in order (its last input token, then its k drafts,
        which the SeqTable's draft_tokens hold). Returns tokens [R * (1+k)] — position j is the token to emit after
        accepting j drafts — and `n_accept` [R] (device)."""


class Drafter(Protocol):
    def propose(self, ctx: StepCtx) -> None:
        """After each sample stage: write `seq_table.draft_tokens[slot, :k_max]` for the sampled rows' slots, the
        drafts for positions `num_computed[slot] + 1 + j` (num_computed is already advanced)."""


class DraftPolicy(Protocol):
    def choose(self, seq: Seq) -> tuple[str, int]:
        """(drafter name, k) from a finite declared set; ("", 0) disables drafting for this Seq."""

    def observe(self, seq: Seq, k: int, n_accept: int, step_ms: float) -> None: ...
