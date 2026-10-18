"""`moe_route_transform.tail_skip@1` on CUDA: one Triton program per call — every expert's best routing rank by atomic
minima into a scratch word per expert, then each entry dropped or kept (reference.py) — graph-safe (no host sync, fixed
launch), deterministic (minima are order-free)."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK = 1024


@triton.jit(do_not_specialize=["n", "rank"])
def _tail_skip(ids, residency, best, out, n, rank, K: tl.constexpr, R: tl.constexpr, E: tl.constexpr,
               EB: tl.constexpr, BLOCK: tl.constexpr):
    e = tl.arange(0, EB)
    tl.store(best + e, K, mask=e < E)
    tl.debug_barrier()
    for start in range(0, n, BLOCK):
        i = start + tl.arange(0, BLOCK)
        x = tl.load(ids + i, mask=i < n, other=-1)
        tl.atomic_min(best + x, i % K, mask=(i < n) & (x >= 0))
    tl.debug_barrier()
    for start in range(0, n, BLOCK):
        i = start + tl.arange(0, BLOCK)
        x = tl.load(ids + i, mask=i < n, other=-1)
        live = (i < n) & (x >= 0)
        b = tl.load(best + x, mask=live, other=0, cache_modifier=".cg")
        missed = tl.zeros([BLOCK], dtype=tl.int1)
        for r in tl.static_range(R):
            missed = missed | (tl.load(residency + x * R + r, mask=live, other=0) < 0)
        tl.store(out + i, tl.where(live & missed & (b >= rank), -1, x), mask=i < n)


class TailSkip:
    def __init__(self, ctx, *, experts: int, top_k: int, rank: int) -> None:
        self.experts, self.top_k, self.rank = experts, top_k, rank
        self.best = torch.empty(experts, dtype=torch.int32, device=ctx.device)    # each expert's best rank

    def __call__(self, ids: torch.Tensor, weights: torch.Tensor,
                 residency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ids, residency = ids.contiguous(), residency.contiguous()
        out = torch.empty_like(ids)
        _tail_skip[(1,)](ids, residency, self.best, out, ids.numel(), self.rank, K=self.top_k,
                         R=residency.shape[1], E=self.experts, EB=triton.next_power_of_2(self.experts), BLOCK=BLOCK)
        return out, weights
