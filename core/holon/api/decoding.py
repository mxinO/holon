"""Scheduling, sampling and speculative decoding roles (design §7.3, §7.6).

Speculative decoding is three roles: a DraftPolicy picks k per step, a Drafter proposes (writing the SeqTable's draft
tokens and counts, and its own state, possibly through auxiliary models and sub-steps), and a Verifier replaces the
Sampler for the step's verify rows. Rollback is the core's `rewind` fan-out to Rewindables (api.hooks). Ordering and
shape axes of transforms and drafters are declared in their manifests.

A LogitsProcessor constrains requests (design §7.6): it follows each one on the host (`open`, `ready`, `commit`,
`close`), writes its masks at launch (`fill`) and applies them on the device before the sampler or verifier (`apply`).

Any of these slot objects (sampler, verifier, drafter, draft policy, logits processor) may count what it does:
`stats() -> dict[str, int]` (`holon.api.hooks.Reports`), reported in the engine's stats as `<plugin name>.<key>`.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

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


class RequestError(ValueError):
    """A request fails, the engine goes on: raised by a logits processor (`open`, `ready`, `commit`), it ends every
    sequence of the request with finish reason "error" and this `kind` — "client" (the request itself is wrong: a
    400 before its first token), "busy" (the server cannot take it now: a 503) or "engine" (it failed while
    generating)."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclasses.dataclass(frozen=True, slots=True)
class VocabView:
    """The vocabulary a logits processor masks."""

    size: int                          # logits per row (the model's padded vocabulary)
    stop_tokens: frozenset[int]        # the engine's stop set (a request may add its `stop_token_ids`)
    tokenizer: Callable[[], Any]       # returns the weight source's tokenizer (`WeightSource.tokenizer`; loads it)


@dataclasses.dataclass(slots=True, eq=False)
class MaskRow:
    """A constrained sequence's logits rows in a step: rows `start` … `start + n - 1` of the step's mask block —
    one, or a verify row's 1 + k: column j masks the token after `drafts[:j]`."""

    seq: Seq
    start: int
    n: int
    drafts: list[int]                  # the real drafts the row checks (at most n - 1)


class LogitsProcessor(Protocol):
    """Masks the logits of the requests it constrains (at most one processor per deployment, `EngineConfig.slots`).
    Its sequences run at overlap depth 0: a row is launched only once every token before it is committed, so `fill`
    sees them on the host. Host hooks run on the engine thread; `apply` runs on the device, in the step's graph."""

    mask_width: int                    # int32 words per mask row (the core stages the mask block, never reads it)

    def open(self, rid: str, params: SamplingParams, prompt: list[int]) -> bool:
        """At `Engine.add`, before anything is registered: whether it constrains request `rid`. Raising
        `RequestError` rejects the request; slow work (compiling) happens elsewhere, `ready` reports it."""

    def ready(self, rid: str) -> bool:
        """Polled each step while a constrained request waits to start (`Status.BLOCKED`): whether it may run.
        Raising `RequestError` ends the request."""

    def commit(self, seq: Seq, out: list[int]) -> None:
        """The tokens a step appends to `seq` (truncated at a stop), before the engine finishes or streams anything;
        once after `ready` for a resumed request (`EngineRequest.resume_output`). Raising `RequestError` ends the
        request and drops these tokens."""

    def close(self, rid: str) -> None:
        """Request `rid` ended (finished or aborted; every sequence)."""

    def fill(self, rows: list[MaskRow], buf: np.ndarray) -> None:
        """At launch: write each row's masks into `buf[row.start: row.start + row.n]` ([rows, mask_width] int32,
        stale until written). Never raises: a mask it cannot build allows every token, and the failure surfaces at
        the row's `commit`."""

    def apply(self, logits: torch.Tensor, mask: torch.Tensor, rows: torch.Tensor) -> None:
        """Device, in place, graph-capturable: mask logits row `rows[i]` with `mask[i]`; rows past the end of
        `logits` are padding (skip them)."""

    # Optional hooks (`holon.api.hooks`): ShutsDown, Reports.


class LogitsProcessorPlugin(Protocol):
    """The `logits_processor` slot's plugin object."""

    def build(self, ctx: BuildCtx, vocab: VocabView) -> LogitsProcessor: ...


class Sampler(Protocol):
    """One per engine, holding per-slot parameters (device rows once sampling is graph-captured).

    `distribution` and `uniform` serve any StepCtx — a step, an aux view, a drafter's sub-step — one row per entry of
    its `batch.dev.logits_rows`, and `distribution` any last-dim width (a drafter's candidate scores as well as the
    vocabulary). Uniform streams: 0 is the sampler's own draw, 1–2 the verifiers' coins, 8 and up a drafter's (a draft
    drawn on a verifier's coins is no longer independent of them, and verification stops being lossless)."""

    def admit(self, slot: int, params: SamplingParams) -> None:
        """A request now owns `slot`."""

    def sample(self, logits: torch.Tensor, ctx: StepCtx) -> SampleOut:
        """`logits` [R, V] fp32 for the plan rows with `n_logits > 0`, in row order."""

    def distribution(self, logits: torch.Tensor, ctx: StepCtx, rows: torch.Tensor | None = None) -> torch.Tensor:
        """The probabilities `sample` draws each row of `logits` from, in vocabulary order [R, V] (greedy rows:
        one-hot at the argmax) — what a verifier accepts drafts against. `rows`: which of ctx's logits rows `logits`
        holds (default: all, in order)."""

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
    """Built by the `drafter` slot's plugin (`DrafterPlugin.build`). Optional hooks (`holon.api.hooks`):
    AdmitsDrafts, DraftsInSubsteps, WritesAhead, SamplesDrafts."""

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
    configured by the deployment's options; `holon.api.hooks.BuildsPolicy`). Optional: DeclaresKs."""

    def k(self, seqs: list[Seq]) -> int:
        """Drafts per row for the next step of `seqs` (the running requests), from a finite declared set; 0: a plain
        decode step. Per step, not per request: every row of a verify step has the same width."""

    def observe(self, seq: Seq, k: int, n_draft: int, n_accept: int) -> None:
        """After a commit: the step's k, the drafts written for the request's next step, and the drafts it accepted
        (results reach the host one step late at overlap depth 1)."""
