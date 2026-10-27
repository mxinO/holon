# In-step recurrent checkpoints: an agent turn's prefill in one step

Status: implemented (bbfeeb1, review fixes 66085f5; design rev 3, reviewed; code reviewed clean). Measured on the
RTX 5080 agent trace at the deployment (`evidence/bench_agent-27b-iq3s-prefix-*_rtx-5080_2026-10-12.json` vs
`…_2026-10-09.json`): TTFT p50 8.42 s vs 8.92 s with the cut (−5.6%), p99 16.0 vs 16.5 s, throughput and tokens saved
unchanged; the 27B bounded checks pass, prefix resume tighter (KL p99 0.022 vs 0.039). Builds on docs/dev/m03-qwen38.md (§4.3 seams, D4 checkpoints) and its §7 deferred
item "an in-step snapshot (fla writing the fp32 state at one chunk index), which would remove the extra step a cut
causes".

## 1. Problem (measured)

A recurrent checkpoint (`std.state.recurrent`, policy `prompt_end`) is published where a prefill step **ends**, so
the kind cuts every prompt's prefill at its last 64-aligned position: an agent turn computes its new tokens in two
steps, the second one a remainder of < 64 tokens.

- On the agent trace at the RTX 5080 deployment, 119 of the 289 prefill steps (41%) are such remainders (mean 33
  tokens).
- A remainder step costs about a full step anyway:
  - On the GPU, the quantized matmuls read every weight once whatever the row count, and llama.cpp's mmq is slow
    at small row counts: a 32-token step is 34.8 ms of GPU on the 5080 and 59 ms on an A5000. Decode is 15.7 ms.
  - On the host, an eager step costs about 12 ms (5080 node) or 36 ms (4090 node).
- So a turn pays about 47 ms (5080) or 72 ms (4090) for the cut alone.

Prefill graphs would remove only the host part; this design removes the step.

## 2. Goal and acceptance

1. A checkpoint position strictly inside a prefill row's step is published **by that step's forward**: no cut, one
   step per turn. The payload is the state after exactly `at` tokens.
2. Exactness unchanged: a hit on such a payload resumes to the same state as recomputing — bitwise for the
   reference kernels (CPU tests) and for fla (L0: the snapshot equals the final state of a split ending there,
   bitwise), as for today's end-of-step payloads.
3. Rows whose chunk grid does not meet the position keep today's cut: same results, one extra step. A row starts
   unaligned only after a budget-truncated step (a second prefill row in a step gets what the first left; a
   `max_step_tokens` that is not a multiple of 64): the kind now rounds such a step's end down to the alignment
   (§3.3), which costs < 64 tokens rather than a step and keeps later starts aligned.
4. Core diff: one generic seam (§3.4) under the M0.3 generality rule, proven by the toy kind; no model, mixer or
   kernel names.
5. Measured: the agent trace on the RTX 5080 (TTFT, steps per turn) and the bounded checks on the 27B; the
   payload plants still fail. Removing the cut changes how prefill splits into steps, so the 27B's L2, snapshot
   traces and its (unwritten) locks are re-run; the HF goldens are unaffected.

## 3. Design

### 3.1 Kernel (`kernels.fla`, vendored patch)

`chunk_gated_delta_rule_fwd_kernel_h` keeps each sequence's running state `b_h` in fp32 registers. It already stores
the state at the start of every chunk to `h`, but in bf16 (`k.new_empty`), so `h` cannot serve as the payload. The
patch adds three optional arguments:

- `snap`: the payload pool's layer view, fp32 `[units, H, V, K]`;
- `snap_unit` `[N]` int32: the destination unit per sequence, −1 for none;
- `snap_chunk` `[N]` int32: the chunk index within the sequence.

At the top of iteration `i_t == snap_chunk[n]` the kernel stores fp32 `b_h` to `snap[snap_unit[n]]`, with the same
blocking as its `ht` store. The state at the start of chunk `i` is the state after `start + 64·i` tokens. The
argument is constexpr-gated (`SNAPSHOT`), so calls without it compile to today's kernel. It is part of
`vendor.patch`, so `vendor.sh --check` keeps passing.

The op contract `gated_delta_rule@1` is amended (no released users; fla and the reference are its only
providers): every provider **must** take snapshots: `plan(lens, snapshots)` receives per row an offset (a multiple of the chunk
size, `0 < offset < the row's length`) and a destination unit, once per step (one host→device upload, not one per
layer), and asserts `0 < chunk < NT` on the host; each call takes the layer's payload-pool view. The reference kernel
(CPU) stores the recurrence state after that offset and asserts the same. The fla glue asserts the pool view is
contiguous (`input_guard` would otherwise make a temporary copy and the store would be lost) and remaps the per-row
snapshot arrays when it drops zero-token rows.

### 3.2 Conv state

The conv state after `at` is the K − 1 conv inputs before `at`. Since `at − start ≥ 64`, they all come from this
step's own input rows, and the GDN mixer copies them into the conv payload unit (an index copy per layer, eager
prefill only).

### 3.3 Mixer and kind

- **The GDN unit** declares that its forward fills its specs' payloads in-step on its chunk grid:
  `StateSpec(..., in_step=64)` — the grid in tokens, 0 for none (not `alignment`, which is the prefix-hit
  granularity: a mixer could resume anywhere yet snapshot only on its grid) — a capability of every
  `gated_delta_rule@1` provider (§3.1), so it can be declared before a kernel is chosen. Its
  planner turns the step's in-step publishes for its specs into the kernel's snapshots and the conv copy rows,
  reaching the payload pool through the step context (§3.4).
- **`Recurrent.cut`** allows **one** in-step position per row: the first publish position `at1` inside the step
  when `spec.in_step` is set and `(at1 − start) % spec.in_step == 0` (the position itself stays on an index block,
  `lcm(alignment, page)`); where the grid does not meet `at1`, it cuts at `at1` as today. If a second position
  `at2` (`every:<n>`) also falls inside, the step ends at `at2 − spec.in_step`, so the next step starts on the grid
  and takes `at2` in-step (with `every == in_step` that end is `at1` itself: today's end-of-step publish). Only a
  budget-truncated step (`start + n < len(seq.tokens)`) whose end is off the grid is rounded down to it, and only
  if the rounded end is still past `start`; otherwise `n` is unchanged.
- **`Recurrent.publish(seq, at, in_step)`** returns the payload with no MemOps when `in_step` (the forward fills
  it), and today's post-sample copy otherwise.

### 3.4 Core seam

`StateManager.step_ops` today publishes only when a row's step ends at `next_publish`. It extends to
`start < at ≤ end`:

- `at == end`: unchanged — the MemOps run after the sample.
- `start < at < end`: an **in-step publish**, made only when the spec declares an `in_step` grid, the row is not a
  verify row (`n_logits ≤ 1`: decode rows have one token and never meet the condition; preempted recomputes past
  the prompt are prefill rows too), and `(at − start) % spec.in_step == 0`; otherwise skipped, as today (a kind
  that wanted it cut). The core
  calls `publish(view, at, in_step=True)`, rejects any MemOps it returns, checks the payload pool's unit layout
  (unit bytes, segments) equals the spec's, and adds `InStepPublish(row, spec, offset=at − start, payload)` to a new
  `BatchPlan.publishes` list (copied by `padded()` like `mem_ops`), which planners read through `batch.plan`.
  `StepCtx.pool(name)` gives units the pools by name (the executor holds them), so the mixer reaches the payload's
  pool without knowing the kind. Registration at commit is unchanged: filed only if committed tokens cover `at`.

The core does not know who fills the payload — the spec's mixer, by declaration. ABI additions:
`StateSpec.in_step: int = 0`, the `in_step` argument of `publish` (default False: today's kinds unchanged),
`BatchPlan.publishes`, `InStepPublish`, `StepCtx.pool`. The toy kind gains a toy mixer that fills payloads in-step,
to prove the seam in core tests (including commit coverage for an in-step payload).

Steps with in-step publishes are prefill steps, which run eagerly; they never meet a decode graph.

### 3.5 What stays

Other reasons for a cut remain (an unaligned chunk grid, a second position in one step); the cut hook stays in
the core. `every:<n>` saves at most one step per row per step (one in-step position), not all of them. The deferred
prefill graphs would need static snapshot buffers.

## 4. Tests and evidence

- **L0 (sm_86, then sm_89/sm_120):**
  - the snapshot equals the final state of a split at that chunk, bitwise, at chunk 1 and at the last full chunk
    with a partial tail, both head orders;
  - the call's output and final state are bitwise those of a call without snapshots;
  - rows without a snapshot and zero-token rows (through the plugin) are untouched;
  - the host assertion rejects chunk 0 and chunk ≥ NT;
  - plants: a snapshot one chunk late fails; and so does "store the bf16 `h`".
- **CPU (tiny model):**
  - an aligned turn prefills in one step;
  - a hit on an in-step payload is bitwise equal to recomputing;
  - a conv window one token early fails (the SSM check does not cover the conv copy);
  - `every:<n>` with two positions in one step: payloads exist at both (one in-step, the next step's in-step);
  - a budget-truncated second row: its step end rounds down, the next starts aligned; a prompt's last step under
    `every:<n>` is not rounded (no extra remainder step); a truncation that would round to nothing is not rounded;
  - one of the two payload pools full: no hit, no wrong state;
  - no in-step publish for specs without `in_step`;
  - the payload-one-block-early plant fails.
- **Core:** the toy kind and toy mixer, in-step publish, commit coverage.
- **27B:**
  - the bounded checks (prefix resume vs recompute, chunked);
  - the agent trace on the RTX 5080 at the deployment, before and after: steps per turn, TTFT;
  - the full suite and CI.

## 5. Risks

- The fla glue drops zero-token rows: the per-row snapshot arrays must follow (an L0 case). `snap_chunk` is the
  sequence's own chunk index (`i_t`), so fla's `chunk_indices` need nothing.
- Register pressure from the extra store: the store is only taken at one iteration, so expected negligible. The
  kernel time is checked in L0 timing.
- The conv copy reads the step's pre-conv activations: they must still be live when the mixer copies (the copy is
  issued right after the conv, in the same layer).

## 6. Review of rev 1, resolved

H1 the kind cannot tell an in-step publish → `publish(seq, at, in_step)`, the core rejects MemOps for in-step;
H2 the mixer cannot reach the payload pool → `InStepPublish` carries the payload, `StepCtx.pool(name)`, layout check;
M3 `every:<n>` loses positions → one in-step position per row, cut at the second; M4 unfillable requests → the core
publishes in-step only for `in_step` specs, prefill rows and grid-aligned offsets, the plugin asserts the chunk
range; M5 fla glue → contiguity assert, zero-token remap; M6 unaligned starts enumerated, end rounding, `alignment`
not the lcm for the grid; L7 snapshot support mandatory for every provider, uploaded once per step in `plan`;
L8 `padded()` copies publishes, offsets carried. Rev 2: `every:<n>` lost its second position → end at
`at2 − in_step`; end rounding too broad → only budget-truncated steps, never to nothing; `in_step` an int grid,
not a bool; `plan` takes (offset, unit), the call takes the layer view; a prefill row is `n_logits ≤ 1`. Confirmed: the snapshot index has no off-by-one, the split-equality
claim holds, the conv copy is exact; the typed seam is preferred to a MemOp phase or `plan.ext`.
