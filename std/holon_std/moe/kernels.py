"""Op contracts std.moe owns (docs/m02-moe.md D2). Their oracles are in `reference.py`; kernels come from kernel
plugins. Kernel instances are stateless across steps and safe to call inside CUDA graphs; any `T` ≥ 0.

- `moe_topk@1`: instance `(logits [T, E] f32) -> (weights [T, k] f32, ids [T, k] int32)`, both contiguous: softmax
  over E in f32, the k largest (ties to the lower index), divided by their sum (clamped at 6.1e-5) when
  `renormalize`. Ids are unique and in range in every row; a row with any NaN or +inf logit gets such ids and
  unspecified weights. Facts: experts, top_k, renormalize.
- `moe_router@1`: instance `(x [T, H] of dtype, router [E, H] f32, or bf16 under the fact router_dtype = bfloat16 —
  its values, exactly) -> (weights [T, k] f32, ids [T, k] int32)`: the
  router and `moe_topk@1` in one op — `moe_topk@1`'s result, in its terms, for the logits `x · routerᵀ` summed in f32
  from x's exact f32 values, in an order the provider fixes (deterministic; it may depend on the call's row count, as a
  linear's kernel path does), not necessarily the f32 `linear`'s bits (bounded against the composition). std.moe uses it
  where a provider serves its facts, else composes the f32 `linear` and `moe_topk@1`. Facts: hidden, experts, top_k,
  renormalize, dtype. An instance may define `route_prepare(x, router, experts) -> (weights, ids, x_experts)`: its
  result and the experts' input for `x`, `x[:, None]` or a handle standing for it that the `linear_experts@2` (or
  `linear_experts_arena@1`) instance `experts` reads — x made ready for it in the router's own launch (as qlinear@2's
  `prepare_input`, std.ops.kernels: the provider's, keyed and generation-checked, falling back to the handle's x).
  It may expose `pass_rows` (a multiple of `CHUNK`; else `CHUNK`): a row's result is the same in any call of whole
  `CHUNK`s up to that many rows as in its own chunk's call, so std.moe routes an expert pass in one call
  (docs/v1-moe-prefill.md D3′).
- `linear_experts@2`: instance `(x [T, R, in], parts, ids [T, k]) -> y [T, k, out_y]` of `out_dtype`: row j of token t
  applies expert `ids[t, j]` to `x[t, 0]` (R = 1) or `x[t, j]` (R = k). `parts` holds the stack's tensors by role, as
  `QTensor.parts`, per expert (`{"blocks": [E, out, row_bytes]}` for ggml layouts, where @1 took [E·out, row_bytes];
  `{"weight": [E, out, in]}` for `dense/<dtype>`). With `glu = "swiglu"` the stack is a merged gate/up stack (per expert
  `out/2` gate rows, then `out/2` up rows; a part with one value per expert is [2, E], gate's row then up's) and `y =
  silu(gate) · up` in f32 (out_y = out/2); with `glu = "none"`, out_y = out. A dense layout's activation operand is
  rounded to the weight dtype before the product, with f32 accumulation. An instance may accept, in place of `x`, a
  handle its own provider's `moe_router@1` prepared for it (`route_prepare`; once stale, it reads the handle's x).
  Every such handle exposes `.x`, the tensor it stands for, so a consumer that does not know the handle (another
  provider's instance, a composition) falls back on `handle.x`.
  Facts: layout, experts, in, out, top_k, rows_per_token, glu, act_dtype, out_dtype, act_quant (`ACT_QUANT` for
  quantized layouts, as `qlinear@2`'s; `a16` for dense ones). An instance may define `prepare` as `qlinear@2`'s (logical
  shape [E, out, in]); a spec's `config["act"]` is refused under mixed steps as `qlinear@2`'s.
  An instance may expose `pass_rows` (an int, a multiple of `CHUNK`; `CHUNK` without it), the most rows a call may
  carry at its `CHUNK`-row calls' bits (docs/v1-moe-prefill.md D0, §4): for every n ≥ 1 with n·`CHUNK` ≤ `pass_rows`, a
  call of n·`CHUNK` rows gives each row the bits the `CHUNK`-row call of that row's own chunk gives it. Nothing is
  promised for a call that is not a whole number of `CHUNK`s, nor across placements (`linear_experts_arena@1`'s own
  term). std.moe passes at most the smaller of a layer's two stacks' `pass_rows` per call.
  Passed buffers (docs/v1-moe-prefill.md D10): an instance that defines `scratch_bytes(tokens) -> int` (the bytes of
  scratch a call of `tokens` tokens takes; 0 where it needs none) accepts the optional keywords `out` (the result, [T,
  k, out_y] f32, contiguous: written and returned) and `scratch` (a contiguous uint8 buffer of at least
  `scratch_bytes(T)` bytes, 256-byte aligned, for the call's transients); std calls every other instance without them.
  The instance keeps no reference to either past the call; both are read and written on the current stream, after the
  work enqueued before the call; one `scratch` may serve the successive calls of a pass (gate/up, then down); neither
  `out` nor `scratch` may overlap `x`, nor each other. Where the call uses a buffer, a short, misaligned or
  overlapping one is refused (ValueError) before anything is launched (a call that needs no `scratch`, e.g. a
  decode-size one, may leave it unchecked). The results are bitwise those of the call without them, wherever they
  live. Without `scratch` the instance allocates what it needs (a buffer passed inside a captured graph is the
  graph's: static); with it, it may still use its own persistent workspaces (its calls run in order on the stream).
- `moe_combine@1`: instance `(y [T, k, H] of y_dtype, weights [T, k] f32, out [T, H])`: writes `Σ_j w[t, j] · y[t,
  j]`, accumulated in f32 in slot order, into `out` (of `dtype`). Facts: hidden, top_k, dtype, y_dtype. An instance
  may define `add(y, weights, residual [T, H] of dtype, out)`: writes `residual + s`, s that sum rounded to `dtype`,
  added in f32 and rounded to `dtype` again — what `residual + out` gives after the call, so the layer's residual add
  costs no launch of its own (docs/v1-offload.md §14, step 3c).

Expert offload (docs/v1-offload.md §5): an offload store keeps each slot class's experts in an arena `[S, unit]`
whose first `top_k` slots are null slots (zero weights; slot j for routing rank j, so a token's slots stay unique),
then `staging` slots, then cache slots.

- `linear_experts_arena@1`: `linear_experts@2` over an arena: instance `(x [T, R, in], parts, ids [T, k]) -> y [T, k,
  out_y]` as @2's, except that `parts` hold an arena of S slots per role, `ids` are arena slots (< S), and a call's
  ids name at most `experts + top_k` distinct slots (a layer's E experts, plus one null slot per rank). Its facts are
  @2's for the layer (`experts` = E) plus `slots`, the most slots an arena of the stack's class holds (a call's may
  hold fewer). It may accept a prepared handle in place of `x` as @2's (exposing `.x`, the tensor it stands for, which
  a consumer that does not know the handle reads instead). It may expose `pass_rows` as @2's (the same promise, at the
  bits of its `CHUNK`-row calls over the same slots). An instance has `errors`, a device int32 word [1] its
  calls or violation bits into (zero: none; bits are the provider's), which the caller zeroes. Same results as @2
  over the stack those slots hold: the oracle is @2's (`reference.linear_experts`, over the arena). An entry on a null
  slot may get its rows of 0 without the slot being read (its zero weights' product, for finite activations).

- `moe_resolve@1`: instance `(ids [T, k] int32, kinds [T] uint8, residency [E, R] int32, host [E, R] int32, limits [R]
  int32, counts [E] int32, routed [E] int32, ring [E, R] int32, stage [R, F] int32, empty [R, P] int32, n_empty [1]
  int32, misses [1] int32, errors [1] int32, delta [D, 3] int32, n_delta [1] int32, group: int, plan [E + 1] int32,
  jobs [1 + J, 4 + T'] int32) -> (slots [R, T, k] int32, fetch [R, 2, F] int32, n_fetch [R] int32, cpu [T, k] int32)`,
  for one layer's R roles (gate/up, down: each in its own arena of `limits[r]` slots). `kinds` per token (`PADDING`,
  `DECODE` for decode and verify rows, `PREFILL`); `residency[e, r]` the arena slot of expert e's role r, or −1 − tier
  when it is not in VRAM; `ring[e, r]`, for this call, the slot a
  non-resident piece was streamed into (−1: none); `host[e, r]` its unit in the host tier the device reads. Entry (t,
  j) of a padding row, or with `ids[t, j]` < 0 (a dropped route), gets null slot j: no fetch, no count. Otherwise a
  resident expert gets its slot, a ring piece its ring slot — a cache slot, in [k + staging, limits[r]); any other is
  corrupt: the entry gets null slot j and `errors[0]` |= `RANGE` —; the layer's distinct missing pieces of role r
  (neither resident nor in the ring), by expert id ascending, are fetch entries i: `fetch[r, 0, i]` = `host[e, r]` (or
  −2 − e where it is −1: a piece no device-readable tier holds, for `moe_tiers@1` to provide),
  `fetch[r, 1, i]` the slot it is gathered into, here `stage[r, i]` (the entries read it; the caller's staging slots, or
  slots lent for the call: distinct). `n_fetch[r]` counts them all, even past F (the caller must prevent it:
  `misses[0]` becomes the largest `n_fetch` it has seen, and a miss past F reads `stage[r, F − 1]`). `counts[e]` += 1
  per entry of a `DECODE` row routed to e, resident or not (prefill rows never count); `routed[e]` |= 1 when any live
  entry, of any row kind, routes e (the step's needs). The calls of one step accumulate into `counts` and `routed`,
  which the caller zeroes per step. Deterministic; `fetch` past `n_fetch` unspecified. Facts: experts, top_k, roles,
  staging (the caller's own staging slots, [k, k + staging): the cache slots start past them). `empty` (the layer's
  empty cache slots per role, the first `n_empty` of them), `delta` (the layer's region of the delta log), `n_delta`
  (its count) and `group` (the layer's index in the map) serve the variants: @1 leaves them as they are.
  The CPU's share (F5b, docs/v1-offload.md D4): the call's CPU candidates are the experts a live entry routes that are
  missing in every role (neither resident nor in the ring) and held by the host tier the device reads (tier code −1)
  in every role; of its n of them the CPU takes c = min(`plan[n]`, n) — the last c by (best routing rank, expert id)
  ascending, the best rank being the least j of the expert's live entries (Strata's rule: the lowest-ranked misses).
  They are not fetched (no fetch entry; a variant never admits them, though they count), `routed[e]` |= 4 marks them,
  their entries read null slot j and get `cpu[t, j]` = t·k + j (every other entry −1), and they are written to
  `jobs` (`cpu_experts@1`'s table, in mapped host memory) by expert id ascending: `jobs[0, 0]` = c, zero included,
  then row 1 + i = (expert, `host[e, 0]`, `host[e, 1]`, n_tok, its entries t·k + j ascending). A call without the
  CPU's share passes a zero `plan` (and a table of its own); the caller sizes `jobs` for its calls with a share (T ≤
  T', c ≤ J): past it `errors[0]` |= `JOBS` and the table's contents are unspecified.
- Optional `split` (an instance with `splits = True`): `split=True` also returns `miss_slots` [R, T, k] — each entry
  whose piece the call fetches (staged or admitted) reads its slot there and its null slot j in `slots`, every other
  entry the reverse — so a caller can run the entries already resident while the fetch lands (FN step 7: the hits
  first), the two passes' outputs summing to the one pass's (a null slot's output is zero).
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
  layers as empty. Its instance may also have `lend_tail(cls: int, n: int, residency [G, E, R] int32, classes [G, R]
  int64, limits [G, R] int32, **state of every group) -> picked [G, E] bool` (docs/v1-moe-prefill.md D11), on the
  device without a host sync: the experts holding any of class `cls`'s last `n` slots (a role r of class `cls` at a
  slot ≥ `limits[g, r]` − n, the class's arena size less n), whole — the caller evicts them in every role, unmaps,
  logs and returns them as `lend`'s —, after the policy has released its own slots in that range that no expert maps
  (spares: such a slot stops being one). The caller lends the range itself to the step's expert passes (D10's `out`
  and `scratch`), and the experts' other slots and the layer's empty slots outside it to the step's ring and staging,
  until the step's end. Its class is the store's choice, the same under every policy: the largest class arena — the
  class of the first role (gate/up) that the most layers have, ties to the lower class index. A policy without
  `lend_tail` cannot lend a tail (the store refuses `moe.ring_lend = tail` under it).
- `moe_route_transform@1` (signature only; `approx`, bound by the explicit option `moe.route_transform = <name>`,
  never by kernel priority): instance `(ids [T, k] int32, weights [T, k] f32, residency [E, R] int32) -> (ids, weights)`
  between `moe_topk@1` and the resolve, with the layer's map as the resolve reads it (after the step's lending, its
  ring pieces at their ring slots); graph padding rows arrive with every id −1; a negative id is dropped (the rank's
  null slot: no fetch, no count, a zero contribution). Each transform's semantics are a contract its plugin owns,
  `moe_route_transform.<name>@1`, with its own oracle and provider (docs/v1-offload.md D3). Facts: experts, top_k.
- `expert_fetch@1`: instance `(fetch [2, F] int32, n [1] int32, src [U, unit] uint8, dst [S, unit] uint8)`: copies unit
  `src[fetch[0, i]]` to slot `dst[fetch[1, i]]` for i < min(n, F), in the graph (`src` may be a mapped host pool's
  device view: a zero-copy gather); the slots are distinct. Facts: unit_bytes, staging. Optional member
  `gather_roles(fetch [R, 2, F], n [R], srcs, dsts)`: every role's copy of a group at once (srcs[r] → dsts[r] as the
  instance's copy, role r's list), so a store issues one call for a group's roles.
- `expert_prefetch@1` (D8's ring, eager steps only): an instance has `start(key: int, fetch [2, F] int32, n [1] int32,
  src [U, unit] uint8, dst [S, unit] uint8)` — `expert_fetch@1`'s copy on the provider's own stream, after all the work
  enqueued on the current stream so far, returning at once —; and `wait(key)`: the current stream's later work waits
  until key's copies have landed (each key waited for once). Facts: unit_bytes, copy — the copy list's placement an
  instance serves: `sm`, `fetch` and `n` on the device and `src` a mapped host pool's device view (no host sync);
  `dma`, them on the host and `src` the pinned host pool (copy engines). A provider declares the modes it supports.

- `activation_loan@1` (FN step 7: a tail step's activations from its lent tail, the core's ActivationLender): an
  instance has `use(range [N] uint8)` — a context in which every device allocation of the calling thread is cut from
  `range` (device memory the caller lends for the span; its earlier allocations released first, none of them alive)
  — `held() -> int`, the bytes allocated in it still alive, and `release()`: the range free again (nothing held). An
  allocation the range cannot hold fails.

Host tiers past RAM (F9 and the exclusive RAM tier, docs/v1-offload.md D10): when RAM holds fewer pieces than the
store, a piece lives in exactly one of VRAM, RAM (map code −1, `host` its unit) and the SSD (code −2: the checkpoint's
file, always valid), and each layer call runs `moe_resolve → moe_tiers → [the SSD reads: cpu_task@1's copy task;
signal; wait] → expert_fetch → expert_fetch (demotions) → expert_fetch (moves)`:

- `moe_tiers@1`: instance `(ids [T, k] int32, kinds [T] uint8, group: int, fetch [R, 2, F] int32, n_fetch [R] int32,
  stage [R, F] int32, delta [D, 3] int32, start [1] int32, n_delta [1] int32, residency [G, E, R] int32, host [G, E, R]
  int32, piece [G, E, R] int32, classes [G, R] int32, owner [C, U] int32, capacity [C] int32, heat [G, E] f32 | None,
  io [R, 1 + F, 2] int32, moves [R, 4, F] int32, n_moves [R, 2] int32)`, one layer call after its resolve (the
  resolve's `fetch`, its records `delta[start : n_delta]`): every layer's map, RAM units and class piece indices,
  each layer's role classes, each class's RAM units (`owner[c, u]`: the piece g·E + e, −1 free; units [0,
  `capacity[c]`), then the class's I/O staging units capacity[c] + i), and the policy's heat (None: no displacement).
  It provides the call's pieces no device-readable tier holds and keeps its RAM tier exclusive, in this order (ties
  by expert id; one writer: a call's decisions see the earlier ones):
  1. Its admissions (the records (group, a, b ≥ 0), in order): a is staged — its fetch entry gathers into a staging
     slot `stage[r, n + j]` past the call's other misses, then moves (`moves[r, 2:4]`: that slot → b's slot, after the
     demotions) —, and b, leaving VRAM, is placed: a's RAM units if a came from RAM (b's demotion `moves[r, 0:2]`:
     b's slot → that unit, after the fetch read it), else a free unit of each of its classes, else the units of RAM's
     coldest expert of its classes (least heat, then group, then id; not routed by the call, not placed by it) when b
     is strictly hotter — that expert goes to the SSD —, else the SSD. An admission into an empty slot (b ≤ −2) needs
     no move. An admitted a's RAM units go to b, or are freed for the next call (the gather still reads them).
  2. Its SSD misses (fetch sources −2 − e) routed by a `DECODE` row and not admitted, in a call with one: kept in RAM
     (D10's keep rule) as b would be placed (free units, else the coldest when strictly colder), else not.
  3. The SSD reads: per role the pieces (`piece`) and the units they land in — a kept piece its RAM unit, any other
     the staging unit capacity[c] + i of its fetch entry i — in `io[r]` (n, then (piece, unit) pairs); those fetch
     entries read that unit.
  The map, `host` and `owner` take the call's result (a resident, b and kept pieces in RAM or on the SSD, displaced
  experts on the SSD). `n_moves[r]` = (demotions, moves). Deterministic. Facts: experts, top_k, roles.

The CPU's share of the misses (F5b, docs/v1-offload.md D5–D7): per layer the graph runs `moe_resolve → [publish the
CPU jobs; signal] → expert_fetch → linear_experts_arena (gate/up, down) → [wait] → moe_patch_rows → moe_combine`.

- `cpu_task@1` (D6): an instance (facts: `pool`, the worker's kind: `compute`, `io`) has `bind(task) -> slot` (at
  build), `signal(slot)` and `wait(slot)` (graph ops on the current stream: the slot's task runs once the stream
  reaches the signal, the device's writes before it visible to the task; work after the wait runs once the task has
  answered, its writes visible), and `errors`, a device int32 word [1] that waits OR bits into (1: a task failed, 2: a
  wait timed out), which the caller zeroes per step and checks at the step's commit. Fail-open: a task error answers
  at once, a wait gives up after a time bound, and once either happened every later wait returns at once (so a graph
  never waits forever and a drain ends). A task is the CPU kernel's (`cpu_experts@1`'s `task`): callable (runs it now),
  with `native` its handle for a native worker — `(ABI version, function, argument)` of the task ABI
  (kernels.cpu_task's csrc/holon_cpu_task.h), absent where only a synchronous run is possible. An instance also has
  `copy(roles) -> task`, the generic copy task (F9): per role `(descriptors [P, X, 4] int64, units [U, unit] uint8,
  requests [1 + F, 2] int32)` (host tensors; a piece's extents (fd, O_DIRECT, offset, nbytes), nbytes 0 past them;
  the request table `moe_tiers@1`'s `io[r]`): each requested piece's extents read in order into its unit. The oracle
  runs a task.
- `cpu_experts@1` (a CPU kernel): instance `(jobs [1 + J, 4 + T] int32, x [T, hidden] f32, gate_up [U, gu_unit] uint8,
  down [V, down_unit] uint8, rows [T·k, hidden] f32, threads: int = 1)`, host tensors: for each job (`moe_resolve@1`'s
  table) the expert whose gate/up piece is unit `gate_up[u]` (gate's `inter` rows, then up's, as the GGUF stores them)
  and down piece `down[v]`, applied to x[entry // k] for each of its entries, into `rows[entry]`; rows of other entries
  are left as they are. Each input row is quantized to gate/up's ggml-cpu `vec_dot_type` (Q8_K for the k- and
  i-quants, Q8_0 for Q4_0, Q5_0, Q8_0, IQ4_NL), each (weight row, token) one ggml-cpu dot, `silu(g) · u` in f32 as
  `g / (1 + exp(−g)) · u`, quantized to down's `vec_dot_type`, one dot per (row, token). A token's row is bitwise the
  same whatever the other tokens of its job, the other jobs and the thread count (rows are split across threads,
  never K): D7. `task(jobs, x, gate_up, down, rows)`: a `cpu_task@1` task over those buffers (host memory the device
  maps, fixed for the engine's life). Facts: gate_up, down (their layouts), hidden, inter, top_k, tokens (T), jobs
  (J). Providers carry the CPU's tier (`Constraint.cpu_isa`): ggml-cpu's reductions differ by ISA. The oracle
  (`reference.cpu_experts`) dequantizes the weights and quantizes the activations as ggml-cpu does, accumulating in
  float64: it bounds a kernel within the L0 budget; ggml-cpu itself is the bitwise reference.
- `moe_patch_rows@1`: instance `publish(x [T, H], cpu [T, k] int32, out [T', H] f32)`: `out[t]` = x[t] in f32 for every
  token with an entry `cpu[t, j]` ≥ 0 (others untouched; `out` the CPU's mapped input rows); `__call__(y [T, k, H]
  f32, rows [T'·k, H] f32, cpu [T, k] int32)`: `y[t, j]` = `rows[cpu[t, j]]` where it is ≥ 0, in place (`rows` the
  CPU's mapped output rows), before `moe_combine@1`. Facts: hidden, top_k, dtype (x's).
"""
from __future__ import annotations

MOE_TOPK, LINEAR_EXPERTS, MOE_COMBINE = "moe_topk@1", "linear_experts@2", "moe_combine@1"
MOE_ROUTER = "moe_router@1"
MOE_RESOLVE, EXPERT_FETCH, LINEAR_EXPERTS_ARENA = "moe_resolve@1", "expert_fetch@1", "linear_experts_arena@1"
EXPERT_PREFETCH, ACTIVATION_LOAN = "expert_prefetch@1", "activation_loan@1"
CPU_TASK, CPU_EXPERTS, MOE_PATCH_ROWS = "cpu_task@1", "cpu_experts@1", "moe_patch_rows@1"
MOE_TIERS = "moe_tiers@1"
PADDING, DECODE, PREFILL = 0, 1, 2   # `moe_resolve@1`'s row kinds
RANGE, OVERFLOW, JOBS = 1, 2, 8      # error bits: a resident slot outside the arena's cache slots; a delta record past
                                     # the log (the variants); the CPU's share past its job table
ACT_QUANT = ("a4", "a8")            # `act_quant` of quantized stacks (`a16` is not offered for them)
CHUNK = 512                          # tokens per router call, and the expert path's unit of rows (`pass_rows`)
