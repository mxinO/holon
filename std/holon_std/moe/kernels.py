"""Op contracts std.moe owns (docs/m02-moe.md D2). Their oracles are in `reference.py`; kernels come from kernel
plugins. Kernel instances are stateless across steps and safe to call inside CUDA graphs; any `T` ≥ 0.

- `moe_topk@1`: instance `(logits [T, E] f32) -> (weights [T, k] f32, ids [T, k] int32)`, both contiguous: softmax
  over E in f32, the k largest (ties to the lower index), divided by their sum (clamped at 6.1e-5) when
  `renormalize`. Ids are unique and in range in every row; a row with any NaN or +inf logit gets such ids and
  unspecified weights. Facts: experts, top_k, renormalize.
- `linear_experts@2`: instance `(x [T, R, in], parts, ids [T, k]) -> y [T, k, out_y]` of `out_dtype`: row j of token
  t applies expert `ids[t, j]` to `x[t, 0]` (R = 1) or `x[t, j]` (R = k). `parts` holds the stack's tensors by role,
  as `QTensor.parts`, per expert (`{"blocks": [E, out, row_bytes]}` for ggml layouts, where @1 took [E·out,
  row_bytes]; `{"weight": [E, out, in]}` for `dense/<dtype>`). With `glu = "swiglu"` the stack is a merged gate/up
  stack (per expert `out/2` gate rows, then `out/2` up rows; a part with one value per expert is [2, E], gate's row
  then up's) and `y = silu(gate) · up` in f32 (out_y = out/2); with `glu = "none"`, out_y = out. A dense layout's
  activation operand is rounded to the weight dtype before the product, with f32 accumulation. Facts: layout,
  experts, in, out, top_k, rows_per_token, glu, act_dtype, out_dtype, act_quant (`ACT_QUANT` for quantized layouts,
  as `qlinear@2`'s; `a16` for dense ones). An instance may define `prepare` as `qlinear@2`'s (logical shape [E,
  out, in]); a spec's `config["act"]` is refused under mixed steps as `qlinear@2`'s.
- `moe_combine@1`: instance `(y [T, k, H] of y_dtype, weights [T, k] f32, out [T, H])`: writes `Σ_j w[t, j] · y[t,
  j]`, accumulated in f32 in slot order, into `out` (of `dtype`). Facts: hidden, top_k, dtype, y_dtype.

Expert offload (docs/v1-offload.md §5): an offload store keeps each slot class's experts in an arena `[S, unit]`
whose first `top_k` slots are null slots (zero weights; slot j for routing rank j, so a token's slots stay unique),
then `staging` slots, then cache slots.

- `linear_experts_arena@1`: `linear_experts@2` over an arena: instance `(x [T, R, in], parts, ids [T, k]) -> y [T, k,
  out_y]` as @2's, except that `parts` hold an arena of S slots per role, `ids` are arena slots (< S), and a call's
  ids name at most `experts + top_k` distinct slots (a layer's E experts, plus one null slot per rank). Its facts are
  @2's for the layer (`experts` = E) plus `slots`, the most slots an arena of the stack's class holds (a call's may
  hold fewer). An instance has `errors`, a device int32 word [1] its calls or violation bits into (zero: none; bits
  are the provider's), which the caller zeroes. Same results as @2 over the stack those slots hold: the oracle is
  @2's (`reference.linear_experts`, over the arena).

- `moe_resolve@1`: instance `(ids [T, k] int32, kinds [T] uint8, residency [E, R] int32, host [E, R] int32, limits [R]
  int32, counts [E] int32, routed [E] int32, ring [E, R] int32, stage [R, F] int32, empty [R, P] int32, n_empty [1]
  int32, misses [1] int32, errors [1] int32, delta [D, 3] int32, n_delta [1] int32, group: int) -> (slots [R, T, k]
  int32, fetch [R, 2, F] int32, n_fetch [R] int32)`, for one layer's R roles (gate/up, down: each in its own arena of
  `limits[r]` slots). `kinds` per token (`PADDING`, `DECODE` for decode and verify rows, `PREFILL`); `residency[e, r]`
  the arena slot of expert e's role r, or −1 − tier when it is not in VRAM; `ring[e, r]`, for this call, the slot a
  non-resident piece was streamed into (−1: none); `host[e, r]` its unit in the host tier the device reads. Entry (t,
  j) of a padding row, or with `ids[t, j]` < 0 (a dropped route), gets null slot j: no fetch, no count. Otherwise a
  resident expert gets its slot, a ring piece its ring slot — a cache slot, in [k + staging, limits[r]); any other is
  corrupt: the entry gets null slot j and `errors[0]` |= `RANGE` —; the layer's distinct missing pieces of role r
  (neither resident nor in the ring), by expert id ascending, are fetch entries i: `fetch[r, 0, i]` = `host[e, r]`,
  `fetch[r, 1, i]` the slot it is gathered into, here `stage[r, i]` (the entries read it; the caller's staging slots, or
  slots lent for the call: distinct). `n_fetch[r]` counts them all, even past F (the caller must prevent it:
  `misses[0]` becomes the largest `n_fetch` it has seen, and a miss past F reads `stage[r, F − 1]`). `counts[e]` += 1
  per entry of a `DECODE` row routed to e, resident or not (prefill rows never count); `routed[e]` |= 1 when any live
  entry, of any row kind, routes e (the step's needs). The calls of one step accumulate into `counts` and `routed`,
  which the caller zeroes per step. Deterministic; `fetch` past `n_fetch` unspecified. Facts: experts, top_k, roles,
  staging (the caller's own staging slots, [k, k + staging): the cache slots start past them). `empty` (the layer's
  empty cache slots per role, the first `n_empty` of them), `delta` (the layer's region of the delta log), `n_delta`
  (its count) and `group` (the layer's index in the map) serve the variants: @1 leaves them as they are.
- Variants (docs/v1-offload.md D3): `moe_resolve.<policy>@1`, a contract its placement policy's plugin owns (its own
  oracle and CPU provider), takes @1's arguments, then the policy's device state by name (`device_state`), and keeps
  @1's terms except where it says otherwise. It may change `residency` (the regions its policy owns) and the layer's
  `empty` slots, and then logs each change in the delta log: records appended at `n_delta`, which it advances (past D
  a record is dropped and `errors[0]` |= `OVERFLOW`), in a deterministic order — `(group, a, b)`: expert a took expert
  b's slot in every role (b's entries became −1, the RAM tier's code); `(group, a, −2 − j)`: a took the layer's empty
  slots `empty[:, j]` (popped from the top: j = `n_empty` − 1 first); `(group, −1, b)`: b was unmapped (lent, by the
  caller) — and a miss it admits is gathered straight into its slot (its fetch entry names that slot, not a staging
  slot), a ring piece too (its ring slot then goes unread). A call with `DECODE` rows (the one that picks victims)
  also sets `routed[e]` |= 2 for every expert its live entries route: its victims' scope. Its instance also has
  `lend(need [C] int64, residency [G, E, R] int32, classes [G, R] int64, **state of every group) -> picked [G, E]
  bool`, on the device without a host sync: the whole experts (resident in every role) in its policy's order, ties by
  expert id then group; for each class c (`classes[g, r]`: role r's slots of group g are in class c) the first
  `need[c]` of them holding a slot of c; the caller unmaps and logs them, and later returns their slots to their
  layers as empty.
- `moe_route_transform@1` (signature only; `approx`, bound by the explicit option `moe.route_transform = <name>`,
  never by kernel priority): instance `(ids [T, k] int32, weights [T, k] f32, residency [E, R] int32) -> (ids, weights)`
  between `moe_topk@1` and the resolve, with the layer's map as the resolve reads it (after the step's lending, its
  ring pieces at their ring slots); a negative id is dropped (the rank's null slot: no fetch,
  no count, a zero contribution). Each transform's semantics are a contract its plugin owns,
  `moe_route_transform.<name>@1`, with its own oracle and provider (docs/v1-offload.md D3). Facts: experts, top_k.
- `expert_fetch@1`: instance `(fetch [2, F] int32, n [1] int32, src [U, unit] uint8, dst [S, unit] uint8)`: copies unit
  `src[fetch[0, i]]` to slot `dst[fetch[1, i]]` for i < min(n, F), in the graph (`src` may be a mapped host pool's
  device view: a zero-copy gather); the slots are distinct. Facts: unit_bytes, staging.
- `expert_prefetch@1` (D8's ring, eager steps only): an instance has `start(key: int, fetch [2, F] int32, n [1] int32,
  src [U, unit] uint8, dst [S, unit] uint8)` — `expert_fetch@1`'s copy on the provider's own stream, after all the work
  enqueued on the current stream so far, returning at once —; and `wait(key)`: the current stream's later work waits
  until key's copies have landed (each key waited for once). Facts: unit_bytes, copy — the copy list's placement an
  instance serves: `sm`, `fetch` and `n` on the device and `src` a mapped host pool's device view (no host sync);
  `dma`, them on the host and `src` the pinned host pool (copy engines). A provider declares the modes it supports.
"""
from __future__ import annotations

MOE_TOPK, LINEAR_EXPERTS, MOE_COMBINE = "moe_topk@1", "linear_experts@2", "moe_combine@1"
MOE_RESOLVE, EXPERT_FETCH, LINEAR_EXPERTS_ARENA = "moe_resolve@1", "expert_fetch@1", "linear_experts_arena@1"
EXPERT_PREFETCH = "expert_prefetch@1"
PADDING, DECODE, PREFILL = 0, 1, 2   # `moe_resolve@1`'s row kinds
RANGE, OVERFLOW = 1, 2               # error bits: a resident slot outside the arena's cache slots; a delta record past
                                     # the log (the variants)
ACT_QUANT = ("a4", "a8")            # `act_quant` of quantized stacks (`a16` is not offered for them)
