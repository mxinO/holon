"""Optional hooks (release design §7): members a plugin object may define, which the core calls only if it does
(`hasattr` / `getattr` on the member's name). Every Protocol in this module is optional as a whole and implemented by
plugins; members of the Protocols elsewhere in `holon.api` are required. `scripts/ci/check_abi.py` fails on a name
the core looks up that no Protocol declares.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np
    import torch

    from .batch import BatchPlan, StepCtx
    from .decoding import DraftPolicy
    from .mixer import Planner
    from .model import AuxSpec, BuildCtx, Resolved
    from .quant import QTensor
    from .state import HitLen, MemOp, Payload, PrefixView, SeqView, StateSpec


# -- units (built modules) ------------------------------------------------------------------------------------------
class FinalizesWeights(Protocol):
    def finalize_weights(self, ctx: BuildCtx) -> None:
        """Once after loading, before graph capture (design §5.1): derived buffers are created here."""


class BindsKernels(Protocol):
    def bind_kernels(self, ctx: BuildCtx) -> None:
        """After every unit has finalized (merges included): select kernels that depend on the loaded weights."""


class AdoptsWeights(Protocol):
    def adopt_weights(self, ctx: BuildCtx, name: str, value: QTensor, index: int | None = None) -> bool:
        """Take the quantized checkpoint tensor for parameter `name` as is (design §5.5), or slice `index` of a stacked
        one; False: the unit cannot read its layout, a load error."""


# -- mixers (unit classes) and their planners -----------------------------------------------------------------------
class DeclaresState(Protocol):
    @classmethod
    def state_specs(cls, params: Mapping[str, Any], layers: tuple[int, ...], ctx: BuildCtx) -> list[StateSpec]:
        """The state the mixer keeps for `layers`; without it, none."""


class PlansState(Protocol):
    @classmethod
    def planner(cls, params: Mapping[str, Any], specs: list[StateSpec], ctx: BuildCtx) -> Planner:
        """The group's planner (without it, the core's `Pages`). With the deployment option `planner.device_lens` (a
        drafter's sub-steps) every batch it plans has its lengths on the device only: it must choose kernels that read
        them there, or declare `needs_host_lens` (the build then fails)."""


class NeedsHostLens(Protocol):
    needs_host_lens: bool     # the planner reads lengths on the host: the scheduler keeps them exact


class MixesRowKinds(Protocol):
    mixes_row_kinds: bool     # the planner runs a step of decode rows then prefill chunks (`BatchPlan.n_decode`)


class GraphKeyed(Protocol):
    graph_key: Any            # on planner metadata: steps whose keys differ get graphs of their own


# -- state kinds ------------------------------------------------------------------------------------------------------
class KnowsCapacity(Protocol):
    def capacity(self, seq: SeqView) -> int:
        """The tokens the request's current units hold. When every bound kind offers it, the core skips
        `units_needed` for steps that fit (most decode steps), which keeps per-request host work O(1)."""


class CutsPrefill(Protocol):
    def cut(self, seq: SeqView, start: int, n: int) -> int:
        """The tokens (1 … n) a prefill row starting at `start` may take, so a step can end where the kind publishes."""


class PublishesPayloads(Protocol):
    """State reused through payloads (design §6.3; paged state is shared by reference instead)."""

    def next_publish(self, seq: SeqView) -> int | None:
        """The position the kind publishes at next: the core calls `publish` only when a row's step ends there or, for
        a spec with an `in_step` grid, when it lies inside the step on that grid."""

    def publish(self, seq: SeqView, upto: int, in_step: bool = False) -> tuple[Payload, list[MemOp]] | None:
        """`in_step`: `upto` lies inside the step and the spec's forward fills the payload (`StateSpec.in_step`), so
        no MemOps; otherwise `upto` is where the step ends and MemOps fill it."""


class PayloadPool(Protocol):
    payload_pool: str         # the pool its payloads use, which the prefix index may evict payloads from


class KeepsTentatives(Protocol):
    keep_tentative: int       # a request's last this many tentative payloads (`Payload.tentative`) are kept; a newer
                              # one drops the oldest (docs/v1-kv-tiers.md D8)


class RestoresPayloads(Protocol):
    def hit_length(self, idx: PrefixView, upto: int) -> HitLen:
        """The prefix the kind can restore; a non-paged kind without it allows no prefix hit."""

    def adopt(self, seq: SeqView, payload: Payload) -> list[MemOp]:
        """MemOps copying `payload` from its pool — the kind's own, or a state tier's host copy — into the request's
        state, before its first forward."""


# -- drafters and draft policies --------------------------------------------------------------------------------------
class AdmitsDrafts(Protocol):
    def admit(self, slot: int, tokens: list[int]) -> None:
        """Host: a request now owns `slot`; `tokens` is its whole history so far, prefix-hit tokens included (copy it
        with a pinned asynchronous copy: this is not per step). Resets the slot's carried state."""


class DraftsInSubsteps(Protocol):
    def substeps(self, k: int) -> list[int]:
        """The query widths of the sub-steps `run` asks for at k drafts (fixed, so graphs can be captured)."""


class WritesAhead(Protocol):
    def lookahead(self, k: int) -> int:
        """Positions past a row's own tokens `run` writes state at, at k drafts (the scheduler reserves them)."""


class SamplesDrafts(Protocol):
    q_support: int            # Kq > 0: the drafter samples its drafts and publishes the laws, Kq entries per draft
    #                           (`SeqTable.draft_q_ids` / `draft_q`); absent or 0: its drafts are deterministic


class BuildsPolicy(Protocol):
    def build(self, ctx: BuildCtx) -> DraftPolicy:
        """On the `draft_policy` plugin object: the policy, configured by the deployment's options."""


class DeclaresKs(Protocol):
    def ks(self) -> set[int]:
        """On a draft policy: the k values `k` may return (the engine captures verify graphs for those)."""


# -- slot objects -----------------------------------------------------------------------------------------------------
class Rewindable(Protocol):
    """Anything that speculative decoding must roll back past rejected drafts (design §7.5): a module of the target or a
    loaded slot object defining `rewind` (an auxiliary model's modules are rolled back by the drafter that runs them,
    `Drafter.run`). Paged state needs none — it rewinds by length."""

    def rewind(self, ctx: StepCtx) -> None:
        """Device-side, graph-capturable: after the verifier wrote `ctx.seq_table.n_accept` for the step's verify rows,
        keep only what their accepted tokens produced."""


class Reports(Protocol):
    def stats(self) -> dict[str, int]:
        """A slot object's counters, reported in the engine's stats as `<plugin name>.<key>`."""


class ShutsDown(Protocol):
    def shutdown(self) -> None:
        """On a logits processor or a shared object (`BuildCtx.shared`), by `Engine.close` after the drain: stop its
        helpers."""


# -- shared objects (`BuildCtx.shared`) -------------------------------------------------------------------------------
# One that starts threads (CPU workers, I/O workers) sets their CPU affinity itself: otherwise they inherit the
# engine loop's, which `EngineConfig.loop_core` pins to one core.
class RequestsPools(Protocol):
    def before_materialize(self, ctx: BuildCtx) -> None:
        """Once, after every model (auxiliary models included) is built and loaded, before the pools are first
        materialized: the last point to request pools, host pools included (`BuildCtx.pool`; a state tier's host
        pools come at its `attach`, after the final materialize, `TierCtx.host_pool`)."""


class ResetsState(Protocol):
    def after_materialize(self, ctx: BuildCtx) -> None:
        """After every materialize of the pools (the startup profile's minimum one included), before graphs are
        captured: their buffers are new, so reset and seed the device state kept in them."""


class EditsBeforeLaunch(Protocol):
    def before_launch(self, step: BatchPlan) -> tuple[list[tuple[torch.Tensor, np.ndarray]], list[MemOp]]:
        """Before the step is launched: device writes — `(dst, values)`: `values` copied into the device tensor `dst`
        (the object's own static buffer, say) — and MemOps, which the core enqueues ahead of the step's forward in
        stream order. It returns them and enqueues no device work itself: what consumes the writes runs in the forward
        (e.g. a unit's apply kernel). An edit decided at the commit of step c applies at launch c + 2 on every path: a
        commit always precedes the launch two steps after it."""


class ObservesCommits(Protocol):
    telemetry: Mapping[str, torch.Tensor]   # device buffers a step writes, copied to the host with its tokens: static
                                            # (`BuildCtx.static`; read at the first launch, never reallocated)

    def after_commit(self, step: BatchPlan, telemetry: Mapping[str, torch.Tensor]) -> None:
        """At the step's commit, before its tokens are emitted: `telemetry` holds host copies of the declared buffers
        as the step left them (each launch has its own, valid during the call). Raising stops the engine, and steps
        launched after this one are discarded."""


# -- model plugins ----------------------------------------------------------------------------------------------------
class DeclaresDraftHead(Protocol):
    def draft_head(self, cfg: Mapping[str, Any], source: str, name: str) -> AuxSpec | None:
        """A draft head stored with the checkpoint (e.g. a multi-token-prediction layer), built only when a drafter
        asks for it (`TargetView.draft_head`)."""


class TrainedContext(Protocol):
    def trained_context(self, cfg: Mapping[str, Any], source: str) -> int | None:
        """The context length the checkpoint was trained for (`holon serve --max-model-len auto` serves at most
        that); None: unknown."""



# -- weight sources ---------------------------------------------------------------------------------------------------
class ResolvesRefs(Protocol):
    def resolve(self, ref: str) -> Resolved | None:
        """On a weight-source plugin: `ref` names a checkpoint that is not a local path (e.g. a hub id). None if this
        plugin does not fetch it. Claiming may look the reference up (a listing, a small index) but never downloads
        weights; a ValueError explains a reference recognised but not servable (`holon serve` shows it when no plugin
        claims the reference)."""
