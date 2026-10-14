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
planner creates its own. Every attention op (`attention.causal@1` too) has two more, given only when not their
default (so the facts of every other site, which locks record, stay as they were), and a provider that does not
implement them must not support them:
- `causal` (default True): False — each row's 1 + k tokens see each other both ways (a block drafted at once), and
  all of the context before them;
- `window` (default 0, none): W > 0 — a query at position p sees no key at or before p − W (llama.cpp's
  `p − p_k < n_swa`; SGLang's `window_left` = W − 1), keys after p (`causal` False) included.

KV layouts (the deployment's `kv.format` plugin option; docs/m03-qwen38.md D6), each a page of one layer:
- `kv.hnd`: `kv` [units, 2 (k/v), Hkv, page, D] in the model dtype (kv_dtype = q_dtype);
- `kv.hnd.fp8`: the same shape in float8_e4m3fn (kv_dtype "float8_e4m3fn"; written with saturation);
- `kv.hnd.q8_0`: `kv` is a pair — values [units, 2, Hkv, page, D] int8 and scales [units, 2, Hkv, page, D/32] fp16
  (GGUF's q8_0 per 32 values; kv_dtype "q8_0").
A step writes its K/V before attending (`kv_write@1`), so every read — the prefix and the step's own tokens — sees the
stored format. `kv_write@1`: instance `(k, v [T, Hkv, D], kv, unit [T], off [T]) -> None`, token t into page
`unit[t]` at `off[t]`. Facts: layout, kv_heads, head_dim, page, dtype. A spec whose `config["fused"]` lists
`attention.qk_rope_kv@1` has that kernel from its provider for its facts and any the units below may answer.

`attention.qk_rope_kv@1` (docs/m07-perf.md D3): q/k norm, rotary embedding and the KV write in one kernel, used when the
mixer's injected q/k norm and rope units answer `fusion_facts()`. Instance `(q [T, Hq, Dq], k, v [T, Hkv, D],
positions [T], q_weight [D], k_weight [D], kv, unit [T], off [T]) -> q' [T, Hq, D]`: each head of q (its first D
values of Dq: D, or 2D with `gated`, the output gate interleaved after each head's queries) and of k is normed
(`norm` "rmsnorm@1": x / rms(x) · (weight + offset), eps) and rounded to `dtype`, then its first `rotary_dim`
dims rotated (`rope` "rope@1", neox) at `positions` with the static table `cos_sin` (rope@1's instance contract) and
rounded again; k and v (unnormed) are written as `kv_write@1` writes them, q' returned (contiguous). Any T ≥ 0, any
row strides. Facts: kv_write's, heads, gated, norm, eps, offset, rope, theta, rotary_dim, max_positions, cos_sin.
No oracle of its own: the reference composes the units' (`rmsnorm@1`, `rope@1`, `kv_write@1`) — the judge builds it
(holon_judge.fused). Bounded: a provider's norm reduction order is its own.

`attention.causal@1` (a model built without state, e.g. a whole-sequence comparison in tests): instance
`(q [T, Hq, D], k, v [T, Hkv, D], query_start [B+1]) -> [T, Hq, D]`. Facts: heads, kv_heads, head_dim, dtype.
"""
from __future__ import annotations

PREFILL, DECODE, VERIFY = "attention.paged_prefill@1", "attention.paged_decode@1", "attention.paged_verify@1"
CAUSAL = "attention.causal@1"
KV_WRITE, QK_ROPE_KV = "kv_write@1", "attention.qk_rope_kv@1"
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
