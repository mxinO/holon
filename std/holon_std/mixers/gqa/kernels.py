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

Facts: heads, kv_heads, head_dim, page, q_dtype, kv_dtype, layout, max_seqs. Kernel instances hold plans, so each
planner creates its own.

KV layouts (the deployment's `kv.format` plugin option; docs/m03-qwen38.md D6), each a page of one layer:
- `kv.hnd`: `kv` [units, 2 (k/v), Hkv, page, D] in the model dtype (kv_dtype = q_dtype);
- `kv.hnd.fp8`: the same shape in float8_e4m3fn (kv_dtype "float8_e4m3fn"; written with saturation);
- `kv.hnd.q8_0`: `kv` is a pair — values [units, 2, Hkv, page, D] int8 and scales [units, 2, Hkv, page, D/32] fp16
  (GGUF's q8_0 per 32 values; kv_dtype "q8_0").
A step writes its K/V before attending (`kv_write@1`), so every read — the prefix and the step's own tokens — sees the
stored format. `kv_write@1`: instance `(k, v [T, Hkv, D], kv, unit [T], off [T]) -> None`, token t into page
`unit[t]` at `off[t]`. Facts: layout, kv_heads, head_dim, page, dtype.

`attention.causal@1` (a model built without state, e.g. a whole-sequence comparison in tests): instance
`(q [T, Hq, D], k, v [T, Hkv, D], query_start [B+1]) -> [T, Hq, D]`. Facts: heads, kv_heads, head_dim, dtype.
"""
from __future__ import annotations

PREFILL, DECODE, VERIFY = "attention.paged_prefill@1", "attention.paged_decode@1", "attention.paged_verify@1"
CAUSAL = "attention.causal@1"
KV_WRITE = "kv_write@1"
LAYOUT = "kv.hnd"
FORMATS = {None: LAYOUT, "fp8": "kv.hnd.fp8", "q8_0": "kv.hnd.q8_0"}      # `kv.format` -> layout
Q8_BLOCK = 32


def kv_dtype(layout: str, dtype: str) -> str:
    return {"kv.hnd.fp8": "float8_e4m3fn", "kv.hnd.q8_0": "q8_0"}.get(layout, dtype)


def page_bytes(layout: str, page: int, kv_heads: int, head_dim: int, itemsize: int) -> int:
    """Bytes of one layer's page in `layout`."""
    n = 2 * page * kv_heads * head_dim
    if layout == "kv.hnd.q8_0":
        return n + n // Q8_BLOCK * 2
    return n * (1 if layout == "kv.hnd.fp8" else itemsize)
