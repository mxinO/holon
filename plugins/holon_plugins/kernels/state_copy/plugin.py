"""std.state_tier's `state_copy@1` (op contract: std.state_tier.kernels; docs/v1-kv-tiers.md D10): row copies between
device pools and mapped host pools. Loads by the SM gather (8 CTAs × 256 threads of 16-byte loads, no copy engine, so
an engine step's own copies never queue behind a load: K step 0's K-M2 and K-M3), stores by one
`cudaMemcpyBatchAsync` (copy-engine DMA, faster into host memory); the fact `copy` overrides either.

The native module is built ahead of time (`python -m holon_plugins.kernels.state_copy.prebuild`); without it an
instance cannot be created and the error names that command."""
from __future__ import annotations

import functools

import torch
from holon.api import Constraint, KernelSpec

from . import extension

STATE_COPY = "state_copy"
COPY = {"h2d": "sm", "d2h": "dma"}         # the default path per direction (fact `copy` overrides)
PATHS = ("sm", "dma")


class Modules:
    """The prebuilt module, loaded once per SM on first use (never compiled here)."""

    def __init__(self) -> None:
        self.loaded: dict[int, object] = {}

    def __call__(self, sm: int):
        if sm not in self.loaded:
            self.loaded[sm] = extension.load(sm)
        return self.loaded[sm]


class StateCopy:
    """`state_copy@1`. "sm": the SM gather; "dma": one batch of copy-engine DMA."""

    def __init__(self, modules: Modules, ctx, *, direction: str, chunk: int, copy: str | None = None) -> None:
        lib = modules(ctx.fingerprint.sm)
        self.copy = lib.state_copy_sm if (copy or COPY[direction]) == "sm" else lib.state_copy_dma

    def __call__(self, src: torch.Tensor, src_rows: torch.Tensor, dst: torch.Tensor, dst_rows: torch.Tensor) -> None:
        self.copy(src, src_rows, dst, dst_rows)


class Plugin:
    def register(self, regs) -> None:
        regs.kernels.add(STATE_COPY, KernelSpec(
            op=STATE_COPY, provider="kernels.state_copy", impl=functools.partial(StateCopy, Modules()),
            requires=Constraint(sms=frozenset(extension.ARCHS)), props=frozenset({"deterministic"}), priority=10,
            supports=lambda f: f["chunk"] % 16 == 0 and f["direction"] in COPY and f.get("copy", "sm") in PATHS))
