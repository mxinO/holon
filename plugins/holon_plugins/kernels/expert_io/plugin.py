"""The offload store's in-graph kernels as `moe_resolve@1` (one CTA: the CPU's share of the misses, the rest ranked by
expert id, slots for every entry), its variants `moe_resolve.lfu@1` (F11) and `moe_resolve.lru@1` (F4) — the same CTA
first updates the policy's state and admits misses into victims' slots —, and `expert_fetch@1` (the zero-copy gather
of missed units from a mapped host pool into their slots, 8 CTAs × 256 threads of 16-byte loads: step 0's M1,
docs/v1-offload.md §13), `expert_prefetch@1` and `moe_patch_rows@1` (F5b: a token's row into the CPU's mapped input
rows, the CPU's output rows into the experts' output). Op contracts: std.moe.kernels, and the placement plugins'
(placement.lfu, placement.lru). A variant's `lend` (and F11's `lend_tail`)
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

RESOLVE, FETCH, PREFETCH, PATCH, TIERS = "moe_resolve", "expert_fetch", "expert_prefetch", "moe_patch_rows", "moe_tiers"
LOAN = "activation_loan"
POLICIES = {RESOLVE: 0, "moe_resolve.lfu": 1, "moe_resolve.lru": 2}     # the native module's policy codes
ROLES, SHARED_BYTES = 8, 48 << 10          # a warp per role; the resolve's tables in shared memory, beside
STATIC_BYTES = 9 * 4                       # its static shared words (and the variants' alignment word)


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
    splits = True                          # `split`: the fetched entries' slots apart (std.moe.kernels)

    def __init__(self, modules: Modules, op: str, ctx, *, experts: int, top_k: int, roles: int, staging: int) -> None:
        self.lib, self.policy, self.roles, self.staging = modules(ctx.fingerprint.sm), POLICIES[op], roles, staging

    def __call__(self, ids, kinds, residency, host, limits, counts, routed, ring, stage, empty, n_empty, misses,
                 errors, delta, n_delta, group, plan, jobs, split: bool = False, defer: bool = False, **state):
        tokens, k = ids.shape
        slots = torch.empty(self.roles, tokens, k, dtype=torch.int32, device=ids.device)
        fetch = torch.empty(self.roles, 2, stage.shape[1], dtype=torch.int32, device=ids.device)
        n_fetch = torch.empty(self.roles, dtype=torch.int32, device=ids.device)
        cpu = torch.empty(tokens, k, dtype=torch.int32, device=ids.device)
        missed = torch.empty_like(slots) if split else None
        later, n_later = (torch.empty_like(fetch), torch.empty_like(n_fetch)) if defer else (None, None)
        self.lib.resolve(self.policy, ids, kinds, residency, host, limits, counts, routed, ring, stage, empty, n_empty,
                         misses, errors, delta, n_delta, group, [state[n] for n in STATE[self.policy]], self.plant,
                         plan, jobs, slots, fetch, n_fetch, cpu, self.staging, missed, later, n_later)
        return (slots, fetch, n_fetch, cpu) + ((missed,) if split else ()) + ((later, n_later) if defer else ())

    def lend(self, need: torch.Tensor, residency: torch.Tensor, classes: torch.Tensor, **state) -> torch.Tensor:
        """Whole resident experts in the policy's order (LFU: by score; LRU: least recently routed), ties by expert id
        then layer; for each class c the first need[c] of them holding a slot of c. Device ops, no host sync."""
        groups, experts, _ = residency.shape
        whole = (residency >= 0).all(-1).t().reshape(-1)                      # [E·G], in (expert, layer) order
        key = (state["score"] if self.policy == 1 else state["last"]).t().reshape(-1)
        key = torch.where(whole, key, torch.full_like(key, torch.finfo(torch.float32).max if key.is_floating_point()
                                                          else torch.iinfo(key.dtype).max))
        order = torch.sort(key, stable=True).indices                        # stable: ties keep (expert, layer) order
        holds = torch.zeros(groups, len(need), dtype=torch.bool, device=key.device).scatter_(1, classes, True)
        held = holds.repeat(experts, 1)[order] & whole[order, None]          # [E·G, C] in the order
        take = (held & (held.cumsum(0) <= need.to(key.device)[None])).any(1)
        picked = torch.zeros_like(whole).scatter_(0, order, take)
        return picked.view(experts, groups).t()


STATE = {0: (), 1: ("score",), 2: ("last", "clock")}       # each policy's state, as its contract names it


class MoeResolveLfu(MoeResolve):
    """`moe_resolve.lfu@1` lends a tail too (placement.lfu's `lend_tail`, docs/v1-moe-prefill.md D11): F11 keeps no
    spares, so the experts holding the range are all it picks. Device ops, no host sync. It also `defers`: the CPU's
    experts are admitted as any miss, their pieces' fetch listed apart (`later`, `n_later`)."""

    defers = True

    def lend_tail(self, cls: int, n: int, residency: torch.Tensor, classes: torch.Tensor, limits: torch.Tensor,
                  **state) -> torch.Tensor:
        return ((classes[:, None, :] == cls) & (residency >= (limits - n)[:, None, :])).any(-1)


class ExpertFetch:
    """`expert_fetch@1`, and its optional `gather_roles`: a group's every role in one launch (item 2 of FN's step 7: a
    decode layer's two small gathers were two launches of 8 CTAs each, ≈ 27 GB/s on a 5080's link), on
    `expert_io.fetch_ctas` CTAs per role (plugin option, default 8); `expert_io.fetch_roles=off` withholds it."""

    parts = True                           # `gather_roles(..., part, parts)`: one part of the lists (std.moe.kernels)

    def __init__(self, modules: Modules, ctx, *, unit_bytes: int, staging: int) -> None:
        self.lib = modules(ctx.fingerprint.sm)
        opts = ctx.options.get("plugins", {})
        self.ctas = int(opts.get("expert_io.fetch_ctas", 8))
        if str(opts.get("expert_io.fetch_roles", "on")) == "off":       # a store then fetches role by role
            self.gather_roles = None

    def __call__(self, fetch, n, src, dst) -> None:
        self.lib.fetch(fetch, n, src, dst, False)

    def gather_roles(self, fetch, n, srcs, dsts, part: int = 0, parts: int = 1) -> None:
        """Role r's units srcs[r][fetch[r, 0, i]] into dsts[r][fetch[r, 1, i]] for i < min(n[r], F), one launch; with
        `parts`, only its part `part`: entries [part·q, (part + 1)·q), q = ⌈min(n[r], F) / parts⌉."""
        self.lib.fetch_roles(fetch, n, list(srcs), list(dsts), self.ctas * len(dsts), part, parts)


class ExpertPrefetch:
    """D8's ring copies on a stream of its own, after the work enqueued so far, each key's end an event the compute
    stream waits on: from the device's tables, the fetch kernel on few SMs (zero-copy reads of the mapped host pool);
    from the host's, copy-engine DMA from the pinned host pool (`cudaMemcpyBatchAsync`)."""

    def __init__(self, modules: Modules, ctx, *, unit_bytes: int, copy: str) -> None:
        self.lib, self.stream, self.landed = modules(ctx.fingerprint.sm), torch.cuda.Stream(ctx.device), {}
        self.copy = copy

    def start(self, key: int, fetch: torch.Tensor, n: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> None:
        if fetch.is_cuda != (self.copy == "sm"):
            raise ValueError(f"expert_prefetch@1 ({self.copy}): the copy list on the {fetch.device.type}")
        self.stream.wait_stream(torch.cuda.current_stream(dst.device))     # after the work enqueued so far
        if self.copy == "sm":                                              # the device's list: SM copies
            with torch.cuda.stream(self.stream):
                self.lib.fetch(fetch, n, src, dst, True)
        else:                                                              # the host's: copy engines
            self.lib.copy_batch(fetch, int(n[0]), src, dst, self.stream.cuda_stream)
        self.landed[key] = torch.cuda.Event()
        self.landed[key].record(self.stream)

    def wait(self, key: int) -> None:
        torch.cuda.current_stream().wait_event(self.landed.pop(key))


class MoePatchRows:
    def __init__(self, modules: Modules, ctx, *, hidden: int, top_k: int, dtype: str) -> None:
        self.lib = modules(ctx.fingerprint.sm)

    def publish(self, x, cpu, out) -> None:
        self.lib.publish(x, cpu, out)

    def __call__(self, y, rows, cpu) -> None:
        self.lib.patch(y, rows, cpu)


class MoeTiers:
    def __init__(self, modules: Modules, ctx, *, experts: int, top_k: int, roles: int) -> None:
        self.lib = modules(ctx.fingerprint.sm)

    def __call__(self, ids, kinds, group, fetch, n_fetch, stage, delta, start, n_delta, residency, host, piece, classes,
                 owner, capacity, heat, io, moves, n_moves) -> None:
        self.lib.tiers(ids, kinds, group, fetch, n_fetch, stage, delta, start, n_delta, residency, host, piece, classes,
                       owner, capacity, heat, io, moves, n_moves)


class ActivationLoan:
    """`activation_loan@1`: a MemPool of torch's caching allocator whose segments the native module's pluggable
    allocator cuts from the range `use` names — a new pool at each `use` (the last one's segments go back with it), so
    a step starts from an empty range; `held` is the current pool's live bytes."""

    def __init__(self, modules: Modules, ctx) -> None:
        self.lib = modules(ctx.fingerprint.sm)
        self.allocator = torch.cuda.memory.CUDAPluggableAllocator(self.lib.__file__, "holon_loan_malloc",
                                                                  "holon_loan_free")
        self.pool = None
        device = torch.device(ctx.device)
        self.index = device.index if device.index is not None else torch.cuda.current_device()

    def use(self, range_: torch.Tensor):
        self.release()
        self.lib.loan_set(range_.data_ptr(), range_.numel())
        self.pool = torch.cuda.MemPool(self.allocator.allocator())
        return torch.cuda.use_mem_pool(self.pool, self.index)

    def release(self) -> None:
        """The pool's segments back (none alive: `held`), the range free for the next `use`."""
        self.pool = None

    def held(self) -> int:
        return 0 if self.pool is None else sum(b["size"] for seg in self.pool.snapshot() for b in seg["blocks"]
                                               if b["state"] == "active_allocated")


class Plugin:
    def register(self, regs) -> None:
        modules = Modules()
        regs.kernels.add(LOAN, KernelSpec(
            op=LOAN, provider="kernels.expert_io", impl=functools.partial(ActivationLoan, modules),
            requires=Constraint(sms=frozenset(extension.ARCHS)), props=frozenset({"deterministic"}), priority=10))
        props = frozenset({"graph_safe", "deterministic"})
        requires = Constraint(sms=frozenset(extension.ARCHS))
        regs.kernels.add(TIERS, KernelSpec(
            op=TIERS, provider="kernels.expert_io", impl=functools.partial(MoeTiers, modules), requires=requires,
            supports=lambda f: f["roles"] <= ROLES, props=props, priority=10))
        regs.kernels.add(PATCH, KernelSpec(
            op=PATCH, provider="kernels.expert_io", impl=functools.partial(MoePatchRows, modules), requires=requires,
            props=props, priority=10))
        for op, code in POLICIES.items():
            words = 2 if code == 0 else 11          # the CPU share's per-expert tables, the variants' (and one word)

            def supports(f, w=words) -> bool:
                return f["roles"] <= ROLES and (f["roles"] + w) * f["experts"] * 4 + STATIC_BYTES <= SHARED_BYTES
            regs.kernels.add(op, KernelSpec(
                op=op, provider="kernels.expert_io",
                impl=functools.partial(MoeResolveLfu if op == "moe_resolve.lfu" else MoeResolve, modules, op),
                requires=requires, props=props, priority=10, supports=supports))
        regs.kernels.add(PREFETCH, KernelSpec(
            op=PREFETCH, provider="kernels.expert_io", impl=functools.partial(ExpertPrefetch, modules),
            requires=requires, props=frozenset({"deterministic"}), priority=10,
            supports=lambda f: f["copy"] in ("sm", "dma")))
        regs.kernels.add(FETCH, KernelSpec(
            op=FETCH, provider="kernels.expert_io", impl=functools.partial(ExpertFetch, modules), requires=requires,
            supports=lambda f: f["unit_bytes"] % 16 == 0, props=props, priority=10))
