"""Scheduling, sampling and speculative decoding roles (design §7.3, §7.6).

Speculative decoding is three roles: a DraftPolicy picks k per step, a Drafter proposes (writing the SeqTable's draft
tokens and counts, and its own state, possibly through auxiliary models and sub-steps), and a Verifier replaces the
Sampler for the step's verify rows. Rollback is the core's `rewind` fan-out to Rewindables (api.state). Ordering,
overlap depth and shape axes of transforms and drafters are declared in their manifests.
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

    def distribution(self, logits: torch.Tensor, ctx: StepCtx) -> torch.Tensor:
        """The probabilities `sample` draws each row of `logits` from, in vocabulary order [R, V] (greedy rows:
        one-hot at the argmax) — what a verifier accepts drafts against."""

    def uniform(self, ctx: StepCtx, stream: int = 0) -> torch.Tensor:
        """Per logits row, a counter-based uniform at the position of the token the row draws; stream 0 is the one
        `sample` uses, others are independent (a verifier's coins)."""

    def logprobs(self, logits: torch.Tensor, tokens: torch.Tensor, ctx: StepCtx) -> SampleOut:
        """`tokens` (one per logits row) with the logprobs the step's requests asked for."""

    def variant(self, slots: np.ndarray) -> Hashable:
        """The host-side choices `sample` makes for these slots — a step with the same variant runs the same device
        work, so it may replay a captured graph."""


class SamplerPlugin(Protocol):
    """The `sampler` slot's plugin object."""

    def sampler(self, max_seqs: int, device: torch.device) -> Sampler: ...


class Verifier(Protocol):
    def verify(self, logits: torch.Tensor, ctx: StepCtx) -> tuple[SampleOut, torch.Tensor]:
        """`logits` [R * (1+k), V]: the 1+k rows of each verify row in order (its last input token, then its k drafts,
        which the SeqTable's draft_tokens hold). Returns `n_accept` [R] (device; never more than the row's
        `SeqTable.n_draft`) and tokens [R * (1+k)] whose first n_accept + 1 of each row are what it emits: the accepted
        drafts, then the token after them — with their logprobs when the step's requests ask for them. The step's
        sampler is `ctx.sampler`."""


class Drafter(Protocol):
    """Built by the `drafter` slot's plugin (`DrafterPlugin.build`); `admit`, `substeps`, `lookahead` and `q_support`
    are optional."""

    # Optional: `q_support: int`, Kq > 0 for a drafter that samples its drafts and publishes the laws it drew them
    # from, Kq entries per draft (`SeqTable.draft_q_ids` / `draft_q`); absent or 0: its drafts are deterministic.

    def admit(self, slot: int, tokens: list[int]) -> None:
        """Host: a request now owns `slot`; `tokens` is its whole history so far, prefix-hit tokens included (copy it
        with a pinned asynchronous copy: this is not per step). Resets the slot's carried state."""

    def run(self, ctx: StepCtx, h: torch.Tensor) -> None:
        """After every step's sample stage (prefill chunks too), inside the step's graph when captured. `h` [T, H] is
        the target trunk's output for every token of the step; for verify rows the SeqTable already holds `n_accept`
        and the advanced `num_computed`. Writes the drafter's own state for the step's tokens, then, for the rows that
        sampled, `seq_table.draft_tokens[slot, :n]` (positions `num_computed[slot] + 1 + j`; with `q_support`, their
        laws `draft_q_ids` / `draft_q[slot, :n]` too) and `n_draft[slot] = n`, n at most the step's `plan.draft_k` if
        the drafter writes state per draft, else at most the engine's maximum.
        An auxiliary model run here over a verify step's rows sees them after the verifier: the core's `rewind`
        fan-out (target modules, slot objects) has already run, so rolling the aux model back is the drafter's call
        (its modules' `rewind(ctx.view(namespace))` after their forward)."""

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
    """The `draft_policy` slot's plugin object, or what its optional `build(ctx: BuildCtx)` returns (for policies
    configured by the deployment's options)."""

    def k(self, seqs: list[Seq]) -> int:
        """Drafts per row for the next step of `seqs` (the running requests), from a finite declared set; 0: a plain
        decode step. Per step, not per request: every row of a verify step has the same width."""

    # Optional: `ks() -> set[int]`, the k values `k` may return (the engine captures verify graphs for those).

    def observe(self, seq: Seq, k: int, n_draft: int, n_accept: int) -> None:
        """After a commit: the step's k, the drafts written for the request's next step, and the drafts it accepted
        (results reach the host one step late at overlap depth 1)."""
