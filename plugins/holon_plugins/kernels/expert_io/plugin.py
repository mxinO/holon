"""The offload store's in-graph kernels as `moe_resolve@1` (one CTA: the layer's misses ranked by expert id, slots
for every entry) and `expert_fetch@1` (the zero-copy gather of missed units from a mapped host pool into staging
slots, 8 CTAs × 256 threads of 16-byte loads: step 0's M1, docs/v1-offload.md §13). Op contracts: std.moe.kernels.

The native module is built ahead of time (`python -m holon_plugins.kernels.expert_io.prebuild`); without it an
instance cannot be created and the error names that command."""
from __future__ import annotations

import functools

import torch
from holon.api import Constraint, KernelSpec

from . import extension

RESOLVE, FETCH = "moe_resolve", "expert_fetch"
ROLES, SHARED_BYTES = 8, 48 << 10          # a warp per role; `need` in static shared memory


class Modules:
    """The prebuilt module, loaded once per SM on first use (never compiled here)."""

    def __init__(self) -> None:
        self.loaded: dict[int, object] = {}

    def __call__(self, sm: int):
        if sm not in self.loaded:
            self.loaded[sm] = extension.load(sm)
        return self.loaded[sm]


class MoeResolve:
    def __init__(self, modules: Modules, ctx, *, experts: int, top_k: int, roles: int, staging: int) -> None:
        self.lib, self.roles, self.staging = modules(ctx.fingerprint.sm), roles, staging

    def __call__(self, ids, kinds, residency, host, counts, misses):
        tokens, k = ids.shape
        slots = torch.empty(self.roles, tokens, k, dtype=torch.int32, device=ids.device)
        fetch = torch.empty(self.roles, self.staging, dtype=torch.int32, device=ids.device)
        n_fetch = torch.empty(self.roles, dtype=torch.int32, device=ids.device)
        self.lib.resolve(ids, kinds, residency, host, counts, misses, slots, fetch, n_fetch)
        return slots, fetch, n_fetch


class ExpertFetch:
    def __init__(self, modules: Modules, ctx, *, unit_bytes: int, staging: int) -> None:
        self.lib = modules(ctx.fingerprint.sm)

    def __call__(self, fetch, n, src, dst, first: int) -> None:
        self.lib.fetch(fetch, n, src, dst, first)


class Plugin:
    def register(self, regs) -> None:
        modules = Modules()
        props = frozenset({"graph_safe", "deterministic"})
        requires = Constraint(sms=frozenset(extension.ARCHS))
        regs.kernels.add(RESOLVE, KernelSpec(
            op=RESOLVE, provider="kernels.expert_io", impl=functools.partial(MoeResolve, modules), requires=requires,
            supports=lambda f: f["roles"] <= ROLES and f["roles"] * f["experts"] * 4 <= SHARED_BYTES, props=props,
            priority=10))
        regs.kernels.add(FETCH, KernelSpec(
            op=FETCH, provider="kernels.expert_io", impl=functools.partial(ExpertFetch, modules), requires=requires,
            supports=lambda f: f["unit_bytes"] % 16 == 0, props=props, priority=10))
