"""The attention op contracts this mixer owns. Their oracles are in `reference.py`; kernels come from kernel plugins.

Ops `attention.paged_prefill@1` (ragged chunk rows), `attention.paged_decode@1` (one row per sequence) and
`attention.paged_verify@1` (the same number of rows, 1 + k, for every sequence: the last input token and k drafts;
lengths only on the device, so verify kernels must not plan from host lengths). A kernel
plugin registers a `KernelSpec` whose `impl(ctx, **facts)` returns an instance with:

- `plan(batch, pages) -> handle`: host side, once per step (e.g. a library's `plan()`), writing what it can into
  buffers the instance owns;
- `run(q [T, Hq, D], kv [units, 2, Hkv, page, D], handle) -> [T, Hq, D]`: one layer; this step's K/V are already
  written; rows of length 0 (graph padding) must be harmless.

Optional: `prepare(handle)`, the device part of the plan, run after the core has written this step's lengths into
`pages.lens` (inside the graph when the step is captured); `graph_key(handle) -> hashable`, offered only by kernels
whose `prepare` and `run` may be captured in a CUDA graph and replayed for every step with the same key (all buffers
they read are stable; the key covers their launch shapes). `pages.lens` and `pages.write_index` are device-exact
only from `prepare` on; host planning uses the plan's bounds.

Facts: heads, kv_heads, head_dim, page, q_dtype, kv_dtype (torch dtype names), layout ("kv.hnd"), max_seqs.
Kernel instances hold plans, so each planner creates its own.

`attention.causal@1` (a model built without state, e.g. a whole-sequence comparison in tests): instance
`(q [T, Hq, D], k, v [T, Hkv, D], query_start [B+1]) -> [T, Hq, D]`. Facts: heads, kv_heads, head_dim, dtype.
"""
from __future__ import annotations

PREFILL, DECODE, VERIFY = "attention.paged_prefill@1", "attention.paged_decode@1", "attention.paged_verify@1"
CAUSAL = "attention.causal@1"
LAYOUT = "kv.hnd"
