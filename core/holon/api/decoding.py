"""Scheduling, sampling and speculative decoding roles (design §7.3, §7.6).

Speculative decoding is four roles: a Drafter proposes, a DraftPolicy picks the source and k, a PlanTransform adds the
draft rows, and a Verifier replaces the Sampler for those rows. Rollback is the core's `rewind` fan-out to Rewindables
(api.state). Ordering, overlap depth and shape axes of transforms and drafters are declared in their manifests.
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable

    import numpy as np
    import torch

    from .batch import BatchPlan, SamplingParams, Seq, StepCtx
    from .model import AuxSpec, BuildCtx, ModelSpec
    from .quant import QTensor


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
        accepting j drafts — and `n_accept` [R] (device), never more than the row's `SeqTable.n_draft`."""


class Drafter(Protocol):
    """Built by the `drafter` slot's plugin (`DrafterPlugin.build`); `admit`, `substeps`, `lookahead` are optional."""

    def admit(self, slot: int, tokens: list[int]) -> None:
        """Host: a request now owns `slot`; `tokens` is its whole history so far, prefix-hit tokens included (copy it
        with a pinned asynchronous copy: this is not per step). Resets the slot's carried state."""

    def run(self, ctx: StepCtx, h: torch.Tensor) -> None:
        """After every step's sample stage (prefill chunks too), inside the step's graph when captured. `h` [T, H] is
        the target trunk's output for every token of the step; for verify rows the SeqTable already holds `n_accept`
        and the advanced `num_computed`. Writes the drafter's own state for the step's tokens, then, for the rows that
        sampled, `seq_table.draft_tokens[slot, :n]` (positions `num_computed[slot] + 1 + j`) and `n_draft[slot] = n`,
        n at most the step's `plan.draft_k` if the drafter writes state per draft, else at most the engine's maximum."""

    def substeps(self, k: int) -> list[int]:
        """The query widths of the sub-steps `run` asks for at k drafts (fixed, so graphs can be captured)."""

    def lookahead(self, k: int) -> int:
        """Positions past a row's own tokens `run` writes state at, at k drafts (the scheduler reserves them)."""


class TargetView(Protocol):
    """What a drafter may use of the target it drafts for."""

    spec: ModelSpec

    def shared(self, name: str) -> torch.nn.Module:
        """A built target module (`embed`, `norm`, `lm_head`), shared rather than loaded twice."""

    def tensors(self) -> Iterable[tuple[str, torch.Tensor | QTensor]]:
        """The target checkpoint's tensors (e.g. a draft head stored with it)."""

    def draft_head(self, name: str) -> AuxSpec | None:
        """The draft head `name` the target's model plugin declares for this checkpoint, if any."""


class DrafterPlugin(Protocol):
    """The `drafter` slot's plugin object."""

    def build(self, ctx: BuildCtx, target: TargetView) -> Drafter: ...


class DraftPolicy(Protocol):
    def k(self, seqs: list[Seq]) -> int:
        """Drafts per row for the next step of `seqs` (the running requests), from a finite declared set; 0: a plain
        decode step. Per step, not per request: every row of a verify step has the same width."""

    def observe(self, seq: Seq, k: int, n_draft: int, n_accept: int) -> None:
        """After a commit: the step's k, the drafts written for the request's next step, and the drafts it accepted
        (results reach the host one step late at overlap depth 1)."""
