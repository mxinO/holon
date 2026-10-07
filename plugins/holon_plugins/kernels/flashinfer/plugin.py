"""FlashInfer 0.6 paged attention as `attention.paged_prefill@1` and `attention.paged_decode@1` kernels
(op contract: std.mixers.gqa; adoption per design §5.6 and spike S1).

Prefill uses the FA2 paged-prefill wrapper (bottom-right causal, so a chunk attends to the context before it);
decode uses the tensor-core decode wrapper. Both plan on the host every step from the block table's host mirror.
Non-graph wrappers for now: CUDA-graph wrappers and two alternating wrappers per geometry (a wrapper reuses one
pinned staging buffer, so depth 1 needs two) come with graphs and overlap.
"""
from __future__ import annotations

import flashinfer
import numpy as np
import torch
from holon.api import Constraint, KernelSpec

# Float workspace for split-KV partials. It is only touched inside `run`, in stream order, so every instance shares
# one (allocated by the core, zero-initialized as FlashInfer requires); each wrapper keeps its own plan buffers.
WORKSPACE_BYTES = 128 << 20


class _Paged:
    def __init__(self, ctx, *, heads: int, kv_heads: int, head_dim: int, page: int, q_dtype: str, kv_dtype: str,
                 layout: str, max_seqs: int) -> None:
        self.geometry = (heads, kv_heads, head_dim, page)
        self.dtypes = {"q_data_type": getattr(torch, q_dtype), "kv_data_type": getattr(torch, kv_dtype)}
        self.workspace = ctx.workspace("kernels.flashinfer", WORKSPACE_BYTES)

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
        self.wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.workspace, "HND")

    def plan(self, batch, pages):
        qo_indptr = torch.from_numpy(np.concatenate([[0], np.cumsum(batch.plan.n_query)]).astype(np.int32))
        heads, kv_heads, head_dim, page = self.geometry
        self.wrapper.plan(qo_indptr, *self._pages(batch, pages), heads, kv_heads, head_dim, page, causal=True,
                          **self.dtypes)
        return self.wrapper


class Decode(_Paged):
    def __init__(self, ctx, **facts) -> None:
        super().__init__(ctx, **facts)
        self.wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(self.workspace, "HND", use_tensor_cores=True)

    def plan(self, batch, pages):
        heads, kv_heads, head_dim, page = self.geometry
        self.wrapper.plan(*self._pages(batch, pages), heads, kv_heads, head_dim, page, **self.dtypes)
        return self.wrapper


def _supports(f: dict) -> bool:
    return (f["layout"] == "kv.hnd" and f["q_dtype"] == f["kv_dtype"] and f["q_dtype"] in ("bfloat16", "float16")
            and f["head_dim"] in (64, 128, 256))


class Plugin:
    def register(self, regs) -> None:
        for op, impl in (("attention.paged_prefill", Prefill), ("attention.paged_decode", Decode)):
            regs.kernels.add(op, KernelSpec(
                op=op, provider="kernels.flashinfer", impl=impl, in_layouts={"kv": "kv.hnd"},
                requires=Constraint(sm_min=80, sm_max=129), supports=_supports, props=frozenset({"deterministic"}),
                priority=10 if op == "attention.paged_prefill" else 5))
