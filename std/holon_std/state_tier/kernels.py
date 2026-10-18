"""The op contract std.state_tier owns (docs/v1-kv-tiers.md D10). Its oracle is in `reference.py`; kernels come from
kernel plugins (`kernels.expert_io`); without one the tier runs the oracle, which is correct on every device, only
slower.

- `state_copy@1`: instance `(src [Rs, C] uint8, src_rows [n] int64, dst [Rd, C] uint8, dst_rows [n] int64) -> None`:
  `dst[dst_rows[i]] = src[src_rows[i]]` for every i < n, enqueued on the current stream (the tier's copy stream), in
  stream order. `src` and `dst` are views on the copy's device of whole pool buffers as rows of one chunk each: a
  device pool's buffer, or a mapped host pool's device view (`Pool.device_view`), `[segments · capacity, C]`, so unit u
  of segment s is row s · capacity + u. The row lists are on the host (CPU int64), n ≥ 0, the `dst_rows` distinct,
  and no row is both read and written by one call. `C` is a multiple of 16 and both views 16-byte aligned. Not graph
  safe (the lists change per call); deterministic (a byte copy). Facts: `direction` ("h2d": a load into device
  pools; "d2h": a store into host pools) and `chunk` (C); optionally `copy` ("sm": an SM gather or scatter, no copy
  engine; "dma": copy-engine DMA), else the provider's default per direction (rev 8.2: loads by the SM gather).
"""
STATE_COPY = "state_copy@1"
