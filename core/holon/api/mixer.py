"""Mixers: units that own per-request state (design §5.2).

A mixer declares its state from parameters alone (so the ledger can size pools before weights load), plans metadata
in two parts — host bounds, then exact device metadata inside the graph — and runs prefill, decode and verify rows.
Graph policy and overlap depth are declared in the manifest, not here.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from .model import Unit

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch

    from .batch import Batch, StepCtx
    from .model import BuildCtx
    from .state import Pages, StateSpec


class Planner(Protocol):
    """One per (mixer, state group), shared by all layers of that mixer. Its metadata is `StepCtx.meta(group)`;
    a mixer without a `planner` gets the core's `Pages` (or None for slot state) as its metadata."""

    def plan_bounds(self, batch: Batch, pages: Pages | None) -> Any:
        """Host side: shapes and upper bounds only (e.g. a kernel library's `plan()`); returns planner metadata."""

    def prepare_device(self, meta: Any, ctx: StepCtx) -> None:
        """Graph-capturable: write exact lengths/positions into static buffers from the SeqTable."""


class Mixer(Unit, Protocol):
    @classmethod
    def state_specs(cls, params: Mapping[str, Any], layers: tuple[int, ...], ctx: BuildCtx) -> list[StateSpec]: ...

    @classmethod
    def planner(cls, params: Mapping[str, Any], specs: list[StateSpec], ctx: BuildCtx) -> Planner:
        """The group's planner. With the deployment option `planner.device_lens` (a drafter's sub-steps) every batch
        it plans has its lengths on the device only: it must choose kernels that read them there, or declare
        `needs_host_lens` (the build then fails)."""

    def forward(self, x: torch.Tensor, ctx: StepCtx) -> torch.Tensor:
        """Prefill, decode and verify rows of one step."""
