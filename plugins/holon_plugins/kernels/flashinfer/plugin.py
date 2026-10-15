"""FlashInfer 0.6 paged attention as `attention.paged_prefill@1` and `attention.paged_decode@1` kernels
(op contract: std.mixers.gqa; adoption per design §5.6 and spike S1).

Prefill uses the FA2 paged-prefill wrapper (bottom-right causal, so a chunk attends to the context before it; its
plan splits a long row's KV, so it also serves a mixed step's decode rows: `serves_decode_rows`, docs/m07-perf.md §3.1);
decode uses the tensor-core decode wrapper. Both plan on the host every step from the block table's host mirror.
Both read KV in the model dtype (`kv.hnd`) or in e4m3 (`kv.hnd.fp8`, with a bf16/fp16 q, from sm_89 on).
Each instance alternates two wrappers: a wrapper stages its plan through one pinned buffer with an asynchronous
copy, so with overlap the next step's plan() must not overwrite the buffer the previous step's copy still reads
(spike S1).
"""
from __future__ import annotations

import flashinfer
import numpy as np
import torch
from holon.api import Constraint, KernelSpec

from . import ops

# Float workspace for split-KV partials. It is only touched inside `run`, in stream order, so every instance of one
# size shares one (allocated by the core, zero-initialized as FlashInfer requires); each wrapper keeps its own plan
# buffers.
WORKSPACE_BYTES = 128 << 20
CTA_TILE_Q = 128            # FlashInfer's largest FA2 query tile (FA2DetermineCtaTileQ)


def workspace_bytes(device, heads: int, kv_heads: int, head_dim: int) -> int:
    """At least WORKSPACE_BYTES, and FlashInfer's split-KV partials at most: a split plan has up to 2 CTAs per SM per
    KV head (`num_blocks_per_sm`, fixed at 2; `max_batch_size_if_split`), each with a query tile of f32 outputs and
    log-sum-exps per head (scheduler.cuh PrefillPlan; the tensor-core decode plans the same way). Many SMs and few
    KV heads exceed the floor (e.g. Qwen3-30B-A3B's 4 KV heads on a 128-SM RTX 4090)."""
    ctas = 2 * torch.cuda.get_device_properties(device).multi_processor_count // kv_heads
    partials = heads * ctas * CTA_TILE_Q * (head_dim + 1) * 4 + 2 * 16          # tmp_v, tmp_s, their alignment
    return max(WORKSPACE_BYTES, partials)


class _Paged:
    def __init__(self, ctx, *, heads: int, kv_heads: int, head_dim: int, page: int, q_dtype: str, kv_dtype: str,
                 layout: str, max_seqs: int) -> None:
        self.geometry = (heads, kv_heads, head_dim, page)
        self.dtypes = {"q_data_type": getattr(torch, q_dtype), "kv_data_type": getattr(torch, kv_dtype)}
        nbytes = workspace_bytes(ctx.device, heads, kv_heads, head_dim)
        self.workspace = ctx.workspace(f"kernels.flashinfer[{nbytes}]", nbytes)

    def _pages(self, batch, pages) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """FlashInfer's CSR page list per batch row (host): indptr, unit ids, tokens in each row's last page."""
        page = self.geometry[3]
        lens = (batch.plan.start_upper + batch.plan.n_query).astype(np.int32)
        n_pages = -(-lens // page)
        held = pages.host_table[batch.plan.slots, : max(int(n_pages.max()), 1)]
        indices = held[np.arange(held.shape[1])[None, :] < n_pages[:, None]]      # row-major: per-row order
        indptr = np.concatenate([[0], np.cumsum(n_pages)]).astype(np.int32)
        last = np.where(n_pages > 0, lens - (n_pages - 1) * page, 0).astype(np.int32)   # 0-length padding rows
        return torch.from_numpy(indptr), torch.from_numpy(indices), torch.from_numpy(last)

    def run(self, q: torch.Tensor, kv: torch.Tensor, wrapper) -> torch.Tensor:
        return wrapper.run(q, kv)


class Prefill(_Paged):
    def __init__(self, ctx, **facts) -> None:
        super().__init__(ctx, **facts)
        self.wrappers = [flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.workspace, "HND") for _ in range(2)]

    def plan(self, batch, pages):
        self.wrappers.reverse()
        qo_indptr = torch.from_numpy(np.concatenate([[0], np.cumsum(batch.plan.n_query)]).astype(np.int32))
        heads, kv_heads, head_dim, page = self.geometry
        self.wrappers[0].plan(qo_indptr, *self._pages(batch, pages), heads, kv_heads, head_dim, page, causal=True,
                              **self.dtypes)
        return self.wrappers[0]


class Decode(_Paged):
    def __init__(self, ctx, **facts) -> None:
        super().__init__(ctx, **facts)
        self.wrappers = [flashinfer.BatchDecodeWithPagedKVCacheWrapper(self.workspace, "HND", use_tensor_cores=True)
                         for _ in range(2)]

    def plan(self, batch, pages):
        self.wrappers.reverse()
        heads, kv_heads, head_dim, page = self.geometry
        self.wrappers[0].plan(*self._pages(batch, pages), heads, kv_heads, head_dim, page, **self.dtypes)
        return self.wrappers[0]


def _supports(f: dict) -> bool:
    return (f["layout"] == "kv.hnd" and f["q_dtype"] == f["kv_dtype"] and f["q_dtype"] in ("bfloat16", "float16")
            and f["head_dim"] in (64, 128, 256) and _causal(f))


def _supports_fp8(f: dict) -> bool:
    # e4m3 KV with a half-precision q (scale 1: the cache holds the values themselves, saturated on write)
    return (f["layout"] == "kv.hnd.fp8" and f["kv_dtype"] == "float8_e4m3fn" and f["q_dtype"] in ("bfloat16", "float16")
            and f["head_dim"] in (64, 128, 256) and _causal(f))


def _causal(f: dict) -> bool:
    """Plain causal attention only: the `causal` / `window` facts (std.mixers.gqa) are not wired to the wrappers."""
    return not {"causal", "window"} & f.keys()


def _half(f: dict) -> bool:
    return f["dtype"] in ("bfloat16", "float16")


def _rope(f: dict) -> bool:
    # partial rotary (the first `rotary_dim` dims of each head, neox) comes from the cos/sin cache's width
    return (_half(f) and f["head_dim"] in (64, 128, 256) and 0 < f["rotary_dim"] <= f["head_dim"]
            and f["rotary_dim"] % 2 == 0)


class Plugin:
    def register(self, regs) -> None:
        # fp8 KV from sm_89 on, where e4m3 converts in hardware (on sm_86 FlashInfer converts in software, 4-5x slower
        # than bf16; spike S1)
        for supports, sm_min in ((_supports, 80), (_supports_fp8, 89)):         # kv.hnd, kv.hnd.fp8
            for op, impl in (("attention.paged_prefill", Prefill), ("attention.paged_decode", Decode)):
                regs.kernels.add(op, KernelSpec(
                    op=op, provider="kernels.flashinfer", impl=impl,
                    requires=Constraint(sm_min=sm_min, sm_max=129), supports=supports,
                    props=frozenset({"deterministic", "needs_host_lens"}
                                    | ({"serves_decode_rows"} if op == "attention.paged_prefill" else set())),
                    priority=10 if op == "attention.paged_prefill" else 5))
        for op, impl, supports in (("rmsnorm", ops.RMSNorm, _half), ("add_rmsnorm", ops.AddRMSNorm, _half),
                                   ("rope", ops.Rope, _rope), ("silu_mul", ops.SiluMul, _half)):
            regs.kernels.add(op, KernelSpec(
                op=op, provider="kernels.flashinfer", impl=impl, requires=Constraint(sm_min=80, sm_max=129),
                supports=supports, props=frozenset({"deterministic", "graph_safe"}), priority=10))
