"""The offload store's in-graph kernels as `moe_resolve@1` (one CTA: the layer's misses ranked by expert id, slots
for every entry), its variants `moe_resolve.lfu@1` (F11) and `moe_resolve.lru@1` (F4) — the same CTA first updates
the policy's state and admits misses into victims' slots —, and `expert_fetch@1` (the zero-copy gather of missed
units from a mapped host pool into their slots, 8 CTAs × 256 threads of 16-byte loads: step 0's M1, docs/v1-offload.md
§13). Op contracts: std.moe.kernels, and the placement plugins' (placement.lfu, placement.lru). A variant's `lend`
runs eagerly (ring steps are never graphed) in torch ops. `MoeResolve.plant` is judge-only: the judge's calibration
plants set its bits; production never does. After the std offload API's tag no new policy codes land here (a later
variant ships its own CUDA provider, docs/v1-offload.md D3).

The native module is built ahead of time (`python -m holon_plugins.kernels.expert_io.prebuild`); without it an
instance cannot be created and the error names that command."""
from __future__ import annotations

import functools

import torch
from holon.api import Constraint, KernelSpec

from . import extension

RESOLVE, FETCH, PREFETCH = "moe_resolve", "expert_fetch", "expert_prefetch"
POLICIES = {RESOLVE: 0, "moe_resolve.lfu": 1, "moe_resolve.lru": 2}     # the native module's policy codes
ROLES, SHARED_BYTES = 8, 48 << 10          # a warp per role; the resolve's tables in shared memory, beside
STATIC_BYTES = 7 * 4                       # its static shared words (and the variants' alignment word)


class Modules:
    """The prebuilt module, loaded once per SM on first use (never compiled here)."""

    def __init__(self) -> None:
        self.loaded: dict[int, object] = {}

    def __call__(self, sm: int):
        if sm not in self.loaded:
            self.loaded[sm] = extension.load(sm)
        return self.loaded[sm]


class MoeResolve:
    plant = 0                              # judge-only: the calibration bugs' bits (judge/holon_judge/planted.py)

    def __init__(self, modules: Modules, op: str, ctx, *, experts: int, top_k: int, roles: int, staging: int) -> None:
        self.lib, self.policy, self.roles, self.staging = modules(ctx.fingerprint.sm), POLICIES[op], roles, staging

    def __call__(self, ids, kinds, residency, host, limits, counts, routed, ring, stage, empty, n_empty, misses,
                 errors, delta, n_delta, group, **state):
        tokens, k = ids.shape
        slots = torch.empty(self.roles, tokens, k, dtype=torch.int32, device=ids.device)
        fetch = torch.empty(self.roles, 2, stage.shape[1], dtype=torch.int32, device=ids.device)
        n_fetch = torch.empty(self.roles, dtype=torch.int32, device=ids.device)
        self.lib.resolve(self.policy, ids, kinds, residency, host, limits, counts, routed, ring, stage, empty, n_empty,
                         misses, errors, delta, n_delta, group, [state[n] for n in STATE[self.policy]], self.plant,
                         slots, fetch, n_fetch, self.staging)
        return slots, fetch, n_fetch

    def lend(self, n: int, residency: torch.Tensor, classes: torch.Tensor, **state) -> torch.Tensor:
        """Whole resident experts, coldest (LFU: by score; LRU: oldest) first, ties by layer then expert id, until every
        class has min(n, its resident slots) of them."""
        whole = (residency >= 0).all(-1).view(-1)                           # [G·E], in (layer, expert) order
        key = (state["score"] if self.policy == 1 else state["last"]).reshape(-1)
        key = torch.where(whole, key, torch.full_like(key, torch.finfo(torch.float32).max if key.is_floating_point()
                                                          else torch.iinfo(key.dtype).max))
        order = torch.sort(key, stable=True).indices                        # stable: ties keep (layer, expert) order
        experts = residency.shape[1]
        slots = torch.zeros(len(key), int(classes.max()) + 1, dtype=torch.int64, device=key.device)
        for r in range(classes.shape[1]):                                   # [G·E, C]: the slots each expert holds
            slots.scatter_add_(1, classes[:, r].repeat_interleave(experts)[:, None], whole.long()[:, None])
        held = slots[order].cumsum(0)
        need = held[-1].clamp(max=n)
        reached = (held >= need).long().argmax(0)                           # each class's first index meeting its need
        take = int(torch.where(need > 0, reached + 1, 0).max())
        picked = torch.zeros(len(key), dtype=torch.bool, device=key.device)
        picked[order[:take]] = True
        return picked.view(residency.shape[:2])


STATE = {0: (), 1: ("score",), 2: ("last", "clock")}       # each policy's state, as its contract names it


class ExpertFetch:
    def __init__(self, modules: Modules, ctx, *, unit_bytes: int, staging: int) -> None:
        self.lib = modules(ctx.fingerprint.sm)

    def __call__(self, fetch, n, src, dst) -> None:
        self.lib.fetch(fetch, n, src, dst)


class ExpertPrefetch:
    """D8's ring copies: DMA on a stream of its own, each key's end an event the compute stream waits on."""

    def __init__(self, modules: Modules, ctx, *, unit_bytes: int) -> None:
        self.lib, self.stream, self.landed = modules(ctx.fingerprint.sm), torch.cuda.Stream(ctx.device), {}

    def start(self, key: int, pairs, src: torch.Tensor, dst: torch.Tensor) -> None:
        self.stream.wait_stream(torch.cuda.current_stream(dst.device))     # after the work enqueued so far
        pairs = torch.as_tensor(pairs, dtype=torch.int64).reshape(-1, 2).contiguous()
        self.lib.copy_pairs(src, dst, pairs, self.stream.cuda_stream)
        self.landed[key] = torch.cuda.Event()
        self.landed[key].record(self.stream)

    def wait(self, key: int) -> None:
        torch.cuda.current_stream().wait_event(self.landed.pop(key))


class Plugin:
    def register(self, regs) -> None:
        modules = Modules()
        props = frozenset({"graph_safe", "deterministic"})
        requires = Constraint(sms=frozenset(extension.ARCHS))
        for op, code in POLICIES.items():
            words = 0 if code == 0 else 9           # the variants' per-expert tables (and one word)

            def supports(f, w=words) -> bool:
                return f["roles"] <= ROLES and (f["roles"] + w) * f["experts"] * 4 + STATIC_BYTES <= SHARED_BYTES
            regs.kernels.add(op, KernelSpec(
                op=op, provider="kernels.expert_io", impl=functools.partial(MoeResolve, modules, op),
                requires=requires, props=props, priority=10, supports=supports))
        regs.kernels.add(PREFETCH, KernelSpec(
            op=PREFETCH, provider="kernels.expert_io", impl=functools.partial(ExpertPrefetch, modules),
            requires=requires, props=frozenset({"deterministic"}), priority=10))
        regs.kernels.add(FETCH, KernelSpec(
            op=FETCH, provider="kernels.expert_io", impl=functools.partial(ExpertFetch, modules), requires=requires,
            supports=lambda f: f["unit_bytes"] % 16 == 0, props=props, priority=10))
