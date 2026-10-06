"""Scheduling, sampling and speculative decoding roles (design §7.3, §7.6).

Speculative decoding is four roles: a Drafter proposes, a DraftPolicy picks the source and k, a PlanTransform adds the
draft rows, and a Verifier replaces the Sampler for those rows. Rollback is the core's commit fan-out to Rewindables.
Ordering, overlap depth and shape axes of transforms and drafters are declared in their manifests.
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

if TYPE_CHECKING:
    import torch

    from .batch import BatchPlan, Seq, StepCtx


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
    def sample(self, logits: torch.Tensor, ctx: StepCtx) -> SampleOut: ...


class Verifier(Protocol):
    def verify(self, logits: torch.Tensor, ctx: StepCtx) -> tuple[SampleOut, torch.Tensor]:
        """Check the drafts in the SeqTable; return the emitted tokens and `n_accept` per row (device)."""


class Drafter(Protocol):
    def propose(self, ctx: StepCtx) -> None:
        """Write draft tokens into the SeqTable's draft buffers."""


class DraftPolicy(Protocol):
    def choose(self, seq: Seq) -> tuple[str, int]:
        """(drafter name, k) from a finite declared set; ("", 0) disables drafting for this Seq."""

    def observe(self, seq: Seq, k: int, n_accept: int, step_ms: float) -> None: ...
