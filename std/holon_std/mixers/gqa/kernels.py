"""The attention op contract this mixer owns, and its reference kernel.

Ops `attention.paged_prefill@1` (ragged chunk rows) and `attention.paged_decode@1` (one row per sequence). A kernel
plugin registers a `KernelSpec` whose `impl(ctx, **facts)` returns an instance with:

- `plan(batch, pages) -> handle`: host side, once per step (e.g. a library's `plan()`), writing what it can into
  buffers the instance owns;
- `run(q [T, Hq, D], kv [units, 2, Hkv, page, D], handle) -> [T, Hq, D]`: one layer; this step's K/V are already
  written; rows of length 0 (graph padding) must be harmless.

Facts: heads, kv_heads, head_dim, page, q_dtype, kv_dtype (torch dtype names), layout ("kv.hnd"), max_seqs.
Kernel instances hold plans, so each planner creates its own.
"""
from __future__ import annotations

from . import reference as ref

PREFILL, DECODE = "attention.paged_prefill@1", "attention.paged_decode@1"
LAYOUT = "kv.hnd"


class Reference:
    """The oracle as a kernel (lowest priority): any device and dtype, so CPU and fp32 runs take the same path."""

    layout = LAYOUT

    def __init__(self, ctx, **facts) -> None:
        pass

    def plan(self, batch, pages):
        return pages.table, batch.dev.slots, pages.lens, batch.dev.query_start

    def run(self, q, kv, handle):
        return ref.paged_attention(q, kv, *handle)
