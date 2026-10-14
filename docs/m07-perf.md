# M0.7 — performance: fused glue, mixed steps, a gated megakernel experiment

Status: **rev 3.3, step 1 built** (2026-10-14; §12). Step 0 reviewed and re-checked. Revision 3.1 was reviewed clean by a panel of three (correctness,
architecture, numerics / performance / tests) over four rounds. Step 0 ran (§11), and the panel's review of it and the
lead's decisions are in §10's "Step 0 panel round"; §1, §3.1, §4, §5, §6, §7 and §11 carry them. The panel re-checks
rev 3.2 before step 1; rev 3.3 fixes its findings (§10, "rev 3.2 re-check") and awaits the panel's final
confirmation.

Scope: plan §2 (M0.7: "fused prefill glue (KV-store kernel, add+norm, q/k norm) and a model-specific megakernel
experiment (design §5.7)"), narrowed and extended by what M0.3–M0.5 recorded for M0.7: the verify-step glue
(m04-spec §13), mixed prefill/decode steps (M0.5 measured streams silent up to 36 s behind a long prefill), and the
mmvq/mmq crossover. The small-batch quantized matmul (plan §6.1) leaves M0.7 as a recorded follow-up with a measured
gate (D9); the forward/sample split stays deferred (M0.5 D2: 1.061× / 1.026×, under its 1.10× rule). Measurements and
prior art are in the local research note `docs/brainstorm/m07-perf-research.md` ("R §n", taken on an RTX A5000) and in
the reviewers' re-derivations recorded in §10; the consumer classes are re-measured first (step 0).

M0.5 and M0.6 (Gates A and B) are merged into `main`; this branch is rebased onto `main` at 4a44fdc, which includes
Gate B's `kernels.triton_fp8` and `std.weights.hf`'s merge-order change. Step 0 ran at Gate A's ggml pin `cbe28bb3`
(before Gate B's merge; no step-0 measurement touches Gate B's code), and Gate B's merge-order change was propagated
to the development-class HF locks in a commit of its own (§11.13). The plan's M0.7 row is updated with this
revision; design §7.1 / §4.2 and plan §10 (mixing as a core budget) are updated with step 4's code.

## 1. Goal and acceptance

Faster steps and shorter stalls on the consumer classes, with every change bitwise where it can be and judged on the
paths it touches. Acceptance:

1. **Fusions shipped only where they pay** (D3–D6): each is measured incrementally (D3 on top of the bf16 `kv_write`;
   verify rows by per-step wall time, not TPOT) and kept only if its saving is ≥ max(0.5%, 3 sd of its row) on some
   release-class row (RTX 5080 / 4090 / 5090; graphed decode B1/B4/B8, verify steps, prefill 512/2048), else reverted
   and recorded. Classification (D10): **bitwise** — D4, D5, D6's shared quantization, the bf16 `kv_write`, D11;
   **bounded** — D3, D6's gate/up + SwiGLU and GDN gating, judged by L0 and the L2 of §6, bounded where decode paths
   change. A bitwise fusion that is kept runs on every class (the 4090 included); D3, if kept, runs on every step,
   prefill included. Per-class keep or drop applies only to the bounded items and to D2.
2. **Crossover entries keyed per shape, shipped only where they pay** (D2): an entry per (class, type, in, out,
   activation mode) where mmq beats mmvq at a row count graphed steps run (4, 8, the drafter's 7) by ≥ 3% per call
   and by > 3 sd across repeated sweeps; the entries ship where their sum is ≥ 1% of some lock's decode or verify
   row; the shipped set delivers ≥ half its re-stated saving, and rows 4 and 8 (`holon_tg` batch 4 / 8, verify steps)
   are guarded on every lock whose shapes receive an entry, with no regression beyond noise. (This replaces rev 3.1's
   per-(class, type, mode) reading, and the per-file reading proposed in §11.12; step 0 panel.)
3. **Mixed steps** (D7), on plain (non-speculative) locks of the RTX 5080 (27B) and 4090, ≥ 4 rounds per arm: the
   per-request maximum inter-token gap improves ≥ 2× at p99 and max (p50 recorded); the paired TTFT ratio (the median
   over (session, turn) of TTFT_mixed / TTFT_base) is at most 1 + max(15%, 2 sd of the baseline rounds), and the
   long-prompt scenario's TTFT p99 regresses by at most 25%; throughput by at most max(5%, 2 sd of the baseline
   rounds); TPOT p99 recorded, not gated; the mixed-steps counter > 0. Speculative locks record the alternation's gap
   and long-prompt TTFT only. D7 serves formats whose decode-row and chunk paths share an activation mode (§3.1): the
   a8 GGUF locks; a deployment where they differ refuses `chunk_with_decode > 0` at bind time (§3.1).
4. **Megakernel experiment** (D8): measured against a fixed speed-of-light gate on the RTX 5090 and 5080 after D3–D6;
   a one-layer prototype only on "go"; recorded either way. No core seam for it unless it goes.
5. **Locks:** propagated once, at the close (bitwise changes keep snapshots byte-identical; D2 / D3 / D6's bounded
   fusions / D7 move them — approvals delegated for passing verdicts); the snapshot trace gains a capped chunk beside
   decoders when D7 ships.
6. **Core:** only the seams of §3, toy-proven; ≤ 6000 code lines. **Suite time** does not grow (lean tests).

Non-goals: mixing verify rows or drafting into prefill steps (§3.1; a follow-up with its seam list); the small-batch
quantized matmul (D9); a lower-precision GDN state; prefill graphs for slow hosts; the residual fold in `Model.trunk`
and the "add+norm" fold of the old plan row (R #10: ≤ 0.2%, recorded); the legibility benchmark (plan §9, design D9 —
run before the release freezes the ABI, not in M0.7).

## 2. Starting point (R §1, corrected by review)

- **Glue share** (A5000, B1, graphed): non-GEMM 3.8% of bf16 decode, 15% on 8B Q4_K_M, 12% on the 27B; quantized prefill
  pays 6–7% for quantization and the f32 → bf16 cast (A5000-specific in part: its 6 MB L2 sends the f32 buffers to DRAM;
  the consumer classes' 64–96 MB L2 keep most on chip — the likely saving there is 1.5–3%).
- **Crossover:** llama.cpp's rule is already tuned per type on the RTX 4090 (Ada: mmvq ≤ 8 for Q4_K/Q5_K/Q6_K) and the
  RTX 5090 (Blackwell, which covers the 5080: Q4_K ≤ 5, Q5_K ≤ 6, Q6_K ≤ 7) (`vendor/.../mmvq.cu:318-346`); the A5000
  finding (mmq faster from 5–7 rows) applies to sm_86 (the rtx-3090 release class) and to untuned i-quants on Ada /
  Blackwell. On the 4090 the 8-column mmvq costs only +3.6 ms on a 32.7 ms verify step (m04-spec §13): the bound is
  ≈ 11%, the expected k-quant saving ≈ 0. The judge carries a copy of the rule (`judge/holon_judge/drafter.py:274`,
  `gguf.py:200-213`).
- **Verify steps:** 1.72× a decode step (5080, MTP) and ≈ 1.5× (4090, DFlash2); the GDN rewind ≈ 960 small kernels.
- **Scheduling:** prefill and decode never share a step (`sched.py`); mixing per row kind is not supported today:
  GQA picks one kernel per step (`gqa/units.py:33-51`), GDN one path per step and refuses mixed verify rows
  (`gdn/units.py:57-64`), the executor and verifiers view logits as uniform verify rows (`executor.py:292-296, 354-359`),
  and the scheduler computes a k for prompt-ending prefill rows (`sched.py:99-106`).
- **Graph timing:** the profiler inflates graph-replayed kernel sums by ≈ 2%; the rest of the 2–19% excess appears only
  where eager steps leave the GPU idle (a clock/power effect), and not on the 4090; graphed traces convert ≈ 1:1 to wall.

## 3. Core seams (today → change)

1. **Mixed steps without speculation.**
   - Today: prefill-only or decode-only steps.
   - Change: `EngineConfig.chunk_with_decode` (an engine setting beside `max_step_tokens`, recorded in locks'
     `[engine]`; **0 = today's behaviour exactly, the default**). When it is > 0 and the deployment has no drafter, a
     step that has a plain decode row may also carry prefill chunks: decode rows are charged first, then prefill
     chunks up to `chunk_with_decode` tokens, at least `min(64, chunk_with_decode)` per step so decoders cannot starve
     prefill; build validates `max_step_tokens ≥ max_seqs + chunk_with_decode`. Prompt-ending rows get k = 0 (no
     drafter). Constrained rows with a token in flight sit out, as today. With a drafter and the setting > 0, steps
     alternate a capped prefill step with a graphed decode/verify step when both kinds are schedulable, else take the
     schedulable kind; no row ever mixes with a verify row.
   - The scheduler states the split: `BatchPlan.n_decode` (a holon.api minor; 0 for unmixed steps), the count of
     leading decode rows — a 1-token prefill row looks like a decode row otherwise.
   - Planners: each attention / mixer planner exposes `mixes_row_kinds` (a kernel prop surfaced by the planner and read
     at build, like `needs_host_lens`, `engine.py:145`); `chunk_with_decode > 0` without a drafter and with a planner
     that cannot mix fails at startup. The activation-mode refusal lives in std, not the core: `Linear` and the expert
     linear unit (`ExpertLinear`), at bind time, read `chunk_with_decode` from `BuildCtx.options` and refuse when it is
     > 0 and their bound kernel's activation mode (its spec's `config["act"]`) differs between the path decode rows
     take and the path a chunk takes — for `qlinear@2` (mmvq vs mmq) and `linear_experts@2` (mmvq_id vs mmq_id: the 30B
     NVFP4 under `fp4` runs W4A8 vs W4A4). The check compares modes generically and names no format; it keeps a mixed
     step from running a row at a precision its unmixed step would not. `chunk_with_decode` joins the options the
     core lists for `BuildCtx.options` in the ABI 0.2 commit. Only steps with `n_decode > 0` take the per-row-kind path: decode rows first,
     so a mixed step is two contiguous token slices; GQA's prefill kernel takes both when its prop says it accepts 1-token rows, GDN runs
     chunked over the prefill slice and recurrent over the decode slice, re-basing published `row` indices for the
     chunked subset. Unmixed steps (all steps at the default) keep today's paths, so snapshots at 0 are unchanged.
   - Mixed steps run eagerly (graphs stay uniform). The worst-step memory profile runs a mixed step when the setting is
     > 0 (a chunk's activations alive through sampling).
   - The dead `SchedulePolicy.mix_phases()` protocol (`core/holon/api/decoding.py:39`) is deleted.
   - **ABI:** one commit at step 1 moves the ABI to 0.2 and carries every break and addition of M0.7 (`mix_phases`
     removed; `BatchPlan.n_decode`, `EngineConfig.chunk_with_decode` and its `BuildCtx.options` key, the kernel prop
     `mixes_row_kinds` and its planner hook added), so the release's step 6 sets 1.0 from it. As built (§12.1): the
     std.ops minors of D3 and D6 land with their code at step 2 — std contracts, which no core listing renders and
     which are free to add. The guide's `core_abi` example names `>=0.2,<0.3` (the legibility gate is retired).
   - Toy proof: the toy state kinds across mixed steps with chunked prefill, constrained rows and preemption, outputs
     equal to today's schedule on CPU tiny models; startup failure for a non-mixing planner; a bounded check "mixed
     schedule vs today's" on GPU.
2. **No other core change** unless the megakernel goes (D8).

Line budget ≈ +90 (scheduler, `n_decode`, executor planning per row kind, validation, profile).

## 4. Decisions

- **D1. Re-measure before building** (step 0, at the ggml pin `cbe28bb3`, on the RTX 5080 / 4090 / 5090): the
  graph-timed per-call crossover sweep per (type, shape in the locks, rows 1–9, activation mode) with and without
  today's cast; the glue share per row (decode B1/B4/B8, verify, prefill 512/2048) including the cast and quantize
  share; the verify-step breakdown; two timing ratios per class (unprofiled graphed wall / eager kernel sum, profiled
  graphed kernel sum / unprofiled wall; clocks investigated only if the first exceeds 1.05, with `nvidia-smi
  --query-gpu=clocks.sm,clocks.mem,power.draw,clocks_event_reasons.active -lms 100`); D8's f; the mixed-steps simulator
  recalibrated per class (eager floors from `prefill_idle` evidence, checkpoint-limited prefix reuse, checked against
  the measured plain agent run). §5 is re-stated per class after step 0, before any code; a candidate below 1% of its
  step is dropped and recorded. Step 0 also builds the release benchmark runner the close and the release reuse (the
  per-request max inter-token gap added to `bench/driver.py`'s rows; a long-prompt scenario; TTFT paired per turn).
- **D2. The crossover is a per-class table inside `kernels.ggml_cuda`, keyed per shape**: (class, type, in, out,
  activation mode), with llama.cpp's rule as the fallback, behind one exported function `mmvq_rows(fingerprint, type,
  mode, in, out)` that `QLinear` calls with its `in` and `out`, as do both judge sites (`drafter.py`, `gguf.py`). An
  entry is chosen per shape at the row counts graphed steps run — 4, 8 and the drafter's 7 — where mmq is faster than
  mmvq by ≥ 3% per call and by > 3 sd across repeated sweeps (step 1 repeats step 0's sweep for the candidate shapes);
  the entries ship where their sum on some lock's decode or verify row is ≥ 1% (§11.5 lists the candidates). The
  entries are chosen against today's mmq, which casts f32 → bf16; step 1's repeat sweep times both mmq variants (with
  and without the cast), and once D4 is kept the entries are re-derived from the no-cast column before the single
  propagation at the close (if D4 is reverted, they stay as chosen). Rows 4
  and 8 are guarded on every lock whose shapes receive an entry. Step 0's per-type reading and §11.12's per-file
  proposal are replaced by this (step 0 panel). llama.cpp's fallback takes an explicit compute
  capability (today's C API reads the current device's, `csrc/api.cu:111`), and the existing ggml plugin test asserts
  that `QLinear`'s actual dispatch per row count matches `mmvq_rows` for every lock class. The table is pinned by the plugin's `src` hash (as `triton_attn`'s class configs are), no
  lock-to-kernel core path. A range check per entry (≤ 8, mmvq for every n up to it). Under M0.6's `fp4` mode a lower crossover moves 5–8-row steps from W4A8 to W4A4: a precision change,
  judged by the small-step L2 of §6. The kernel is chosen by the row count the step actually runs (eager real rows,
  graphed bucket rows) — recorded.
- **D3. Fused q/k-norm + RoPE + KV store** is a std op `attention.qk_rope_kv@1` under `std.mixers.gqa`, used only when
  the injected `qk_norm` and `rope` units answer a new optional `fusion_facts()` (std.ops minor: eps, offset, theta,
  `rotary_dim`, `max_positions`, the gate interleave, the reference identity, and the cos/sin table's static name and
  layout, which become part of the `rope@1` instance contract in `std/holon_std/ops/kernels.py` — a rope provider
  without such a table makes the unit answer None) and a provider supports those facts — never by unit names; a future RoPE variant answers None and keeps the composition. Its reference is built by
  the judge from the composed units' references (unit isolation); the Triton provider covers bf16 and q8_0 KV, rounds
  to bf16 after the norm and after RoPE (before q8_0), handles T = 0; the model-only `_causal` build keeps the
  composition. Bounded (not bitwise vs FlashInfer's rmsnorm reduction). If kept, it runs on every step, prefill
  included (not per row kind). The bf16 Triton `kv_write@1` provider (one launch for k and v) is separate and
  bitwise.
- **D4. mmq writes bf16** through `vendor.patch`, keeping the stream-k fixup's partials in f32 until the final single
  rounding: bitwise equal to today's f32 + cast (`torch.equal` inside the existing `test_mmq_schedules`, which forces
  the fixup). Under M0.6, NVFP4's `scale2` multiplies after the fixup reduction (in the fixup kernel for fixup tiles,
  in the write-back otherwise), before the rounding; `test_mmq_schedules` gains an NVFP4 fixup case.
- **D5. The GDN rewind's glue is fused per step** (masking, window restore, gather / index_copy: ≈ 1.9 of 3.7 ms),
  with the provider calls (`gated_delta_rule`, `causal_conv1d`) kept as locked. At the first GDN layer's rewind (the
  layer's index in its group is 0 — `rewind_modules` visits layers in order; no counter, so graph warm-up and capture
  behave alike) one launch masks every layer's saved gates and writes every layer's accepted-end conv window into the
  pool straight from `window ‖ saved_x`; each layer's replay then calls `causal_conv1d` with the saved window buffer as
  its state (slots `arange(rows)`), so the provider never re-dirties the pool. The replay buffers gain a layer axis.
  L1.5 stays unconditionally bitwise against 1 + n decode calls, eager and graphed.
- **D6. Fusions in the ggml and fla plugins**, each classified:
  - **gate/up + SwiGLU** (bounded): a std op `qlinear_glu@1` (owned by std.ops; reference `silu(x @ Wg.T) · (x @
    Wu.T)`, computed from f32 accumulators) and an optional `Linear.glu(x)` (std.ops minor) that returns None when
    unsupported, SwiGLU then composing as today; SwiGLU's `finalize_weights` marks its merged gate/up `Linear`, so
    `qlinear_glu@1` is selected in `bind_kernels` and recorded as a lock site, never lazily. The ggml provider uses the
    vendored mmvq `fusion.gate` / `glu_op` at one column and the composition otherwise. Judged by L0 (the ggml case on
    q8_1-emulated accumulators, as the qlinear L0 is) and the `--max-step-tokens 1` L2 against the mmvq floor.
  - **shared q8_1 quantization** (bitwise; step 0 shows 1.5–1.9% of the 27B's decode on the 5080 / 5090; kept, it
    runs on every class, the 4090's 0.96% included): explicit, never a memo — an optional
    `Linear.prepare_input(x)` (std.ops minor; an optional `prepare_input` and handle-accepting calls in the `qlinear@2`
    instance contract, documented in `std/holon_std/ops/kernels.py`; reference kernels return x) returning a
    provider-opaque handle that carries x, its key (path, ds layout, activation mode, shape) and a host-side
    generation; a caller with several consumers of one input (GDN's in-projections) prepares once and drops the handle
    after them. The handle lives in the provider's own buffer (one per layout), never the workspace shared per `in`; a
    later `prepare_input` of the same layout bumps the generation, and a consumer whose key or generation does not
    match quantizes from the carried x (decided on the host, so graph capture records the right path). No identity or
    `_version` key, so in-place writes cannot make it stale.
  - **GDN gating** in `kernels.fla` (bounded unless it matches torch's `exp` / `softplus` / `sigmoid` and the bf16 β
    cast; prefill only): L0 against the torch composition and the standard L2.
- **D7. Mixed steps** (§3.1): default 0; plain locks opt in through `[engine]` with a cap chosen per class by step 4's
  measurement on the recalibrated simulator and real runs (candidates 256 / 512; the simulator fails cap 256 on the
  4090, and leaves the 5090's 256 undecided until step 4). Only formats whose decode-row and chunk paths share an
  activation mode (the a8 GGUF locks) can opt in (§3.1). On q8_0 KV a mixed step's decode rows go through
  `_paged_prefill`, which has no KV split, so the simulator under-costs them; step 4 measures it, and may route the
  decode slice through the decode kernel instead.
- **D8. Megakernel experiment, gated on speed of light.** After D3–D6, per class (5090, 5080), on 8B Q4_K_M B1 decode:
  f = 1 − (bytes read per step) / (B × step time), with bytes = the weights read (the embedding table excluded) + KV,
  and B = the bandwidth the step's GEMV kernels achieve, from unprofiled GEMV time — each call priced from the
  per-call sweep (graph-timed), less the activation quantization each sweep call includes (the step's profiled q8_1
  quantize time), so B is the GEMV kernels' alone; not from the profile: profiled kernels overlap under programmatic
  dependent launch, so D1's second ratio above 1 is overlap, not profiler inflation — so f excludes
  the GEMVs' own gap to peak, which a megakernel does not recover; a
  read-only stream over ≥ 1 GB is recorded beside it. Projected gain 1 / (1 − f·c) with c = 0.4–0.55 (the
  share of the gap to speed of light the published megakernels closed: Cohere 0.38, Hazy 0.56), reported at both
  ends. Stop if the projection at c = 0.55 is under 1.15× (f < 24%); otherwise build a one-layer prototype in
  `models/qwen3/fast/` and go only if its L0 against the judge-built recipe reference passes and it projects ≥ 1.15×
  at B1 with no regression at B4–8. A go means a design addendum (core seams: one composite bound to many layers, a
  decode-loop override, taps written by the megakernel), reviewed by this panel. Step 0 (rev 3.3 method): f is 26.9%
  on the 5090 (1.174× at c = 0.55: prototype territory today), 15.8% on the 5080 (stop) and 14.5% on the 4090; step 5
  decides after D3–D6 with the same method.
- **D9. The small-batch quantized matmul leaves M0.7** as a follow-up gated by step 0's per-call sweep: the target's
  4-row / 1-row GEMM time ratio after D2 > 1.25 on a release class (it would be a row-dispatch path inside
  `kernels.ggml_cuda` with its own `config["act"]` entry, not a new provider). Step 0, at the entries that ship:
  1.248 on the 5080 (the mean of four sweeps; not fired), 1.01 on the 4090, 1.11 on the 5090; recomputed once step
  1's repeat sweep fixes the shipped set. Its upside is
  recorded: 3.5 ms of the 5080's 27B 4-row calls over their 1-row time, about 12.8% of its MTP verify step (§11.7).
  D9 stays outside M0.7.
- **D10. Classify every change** bitwise or bounded before building (§1.1); target bitwise; propagate once at the close.
- **D11. NVFP4's `scale2` inside mmvq for more than one column** (bitwise; step 0: 3.3–7.2% of NVFP4 decode B4/B8,
  where the q8_1 path writes f32 and multiplies afterwards, `plugin.py:227-230`). A `vendor.patch` change to mmvq's
  epilogue: the multiply per output element after the full reduction, as the one-column epilogue does, with the same
  launch variant as today's unfused launch (nwarps, rows_per_cuda_block, small_k, halve_iters unchanged), writing
  bf16 with one round-to-nearest-even rounding — so the result equals today's f32 × `scale2` → bf16. L0 inside the
  existing qlinear loop at rows 2–8 with a `scale2` that is not a power of two, compared by value. The MoE path
  (`mmvq_id`, `scale2` [2, E]) already applies the expert scales in llama.cpp's fused epilogue for any token count
  (`plugin.py`'s docstring, "mmvq in llama.cpp's fused epilogue (one row dense; any tokens with ids)"); it is included
  only if step 2's profile of the 30B NVFP4 at decode B4/B8 shows separate scale multiplies. Subject to §1.1's
  keep-or-revert rule. Native W4A4's bf16 → f32 activation copy (6.7–7.0% of its prefill) is recorded only.
- **`triton_attn` shared memory (a bug fix in step 1, not subject to keep or drop).** DFlash2's width-8 verify with bf16
  KV fails graph capture on the RTX 5090: `paged_attention` needs 106,752 B of shared memory against sm_120's 101,376 B
  (§11.4). The kernel caps BLOCK_M at 64 and tiles the query rows over the grid, with each row's arithmetic unchanged
  (checked bitwise against today's kernel where today's compiles); if the compiled `metadata.shared` still exceeds the
  device limit, `num_stages` steps down to 1 (on sm_120 at BLOCK_M 64, head_dim 256 and bf16 only one stage fits),
  and only then `block_n` halves, bounded. The bitwise check covers the stage fallback too (3 stages vs 1 at a shape
  where both compile). The effective config is a deterministic function of (class, the site's facts, the declared
  widths), decided at build: the class configs are pinned by the plugin's `src` hash and the widths (verify widths,
  draft block) come from the lock's `[engine]`, so the lock pins it without a field of its own. The existing
  `triton_attn` L0 is parametrized over every class config × declared width × KV format × head_dim (head_dim 256,
  G = 6, NQ = 8 and 16, bf16 KV included), compiled offline for each target arch, and asserts the compiled shared
  memory is within that target's limit.
- **M0.6 follow-ups.** The FP4 repack holes (`kernels.ggml_cuda`'s `prepare` leaves ≈ 0.63 GiB reserved but unallocated
  on the 30B NVFP4) are fixed in M0.7 by repacking into a destination allocated before the sources are freed: a
  memory-only change, no numerics. The allocator-setting investigation is out of scope.

## 5. Predictions (re-stated per class after step 0)

Rev 3.1's provisional predictions are replaced by step 0's measurements (§11; per class in §11.11). A saving is the
largest over the release rows, as a share of its step; D3, D5 and gate/up + SwiGLU are upper bounds (the fused kernel
is taken to cost nothing).

| Item | 5080 | 4090 | 5090 |
|---|---|---|---|
| Crossover (D2, per shape) | 27B decode B8 −5.7%, DFlash verify −4.8% | none | 27B decode B8 −4.1%, DFlash verify −4.6% |
| bf16 `kv_write` | 8B Q4_K_M B1 −2.2% | −1.5% | −3.5% |
| `qk_rope_kv` (D3), on top (upper bound) | 8B B1 −1.9%; 27B ≤ 0.4%; prefill ≤ 1.0% | −1.7% | −2.9% |
| mmq bf16 out (D4) | GGUF prefill −2.4–3.2%; NVFP4 −5.0% | −4.8–5.1%; −6.0% | −2.4–2.6%; −4.6% |
| GDN rewind glue (D5, upper bound) | verify −3.4–4.2% | −3.4–4.1% | −5.3–6.1% |
| gate/up + SwiGLU (D6, upper bound) | 8B B1 −1.6% | −1.1% | −2.4% |
| shared q8_1 (D6) | 27B decode −1.5% | −0.96% | −1.9% |
| GDN gating (D6) | prefill −0.3%: dropped | −0.4%: dropped | −0.4%: dropped |
| NVFP4 `scale2` in mmvq (D11) | NVFP4 decode B4/B8 −3.5–4.2% | −3.3–3.6% | −6.2–7.2% |
| Mixed steps (D7), plain 27B locks (simulated) | worst gap 2.7 s → 127 / 232 ms (cap 256 / 512); paired TTFT 1.02 / 1.00; long prompt +10% / +2% | → 94 / 151 ms; 1.26 (fails) / 1.01; +22% / +0.7% | → 71 / 123 ms; 1.12 (undecided) / 1.00; +20% / +5% |
| Megakernel (D8), f today | 15.8%: stop | 14.5% (not a D8 class) | 26.9%: prototype territory (1.174×); step 5 decides after D3–D6 |

## 6. Judge

- **Bitwise checks inside existing L0 loops:** D4 (`test_mmq_schedules`, fixup forced, plus the NVFP4 case), D5 (L1.5),
  D6's shared quantization (a mixed-type pair of consumers through one handle), the bf16 `kv_write`; D11 in the
  qlinear loop at rows 2–8 with a `scale2` that is not a power of two (by value); `triton_attn`'s tiled query rows
  bitwise against today's kernel (the stage fallback included, 3 vs 1 stage), and its L0 parametrized over class config × declared width × KV format × head_dim
  with the compiled shared memory checked against each target's limit (offline compiles for sm_86 / 89 / 120a).
- **Bounded L0** (sm_86 / 89 / 120a + memcheck): D3 against the judge-composed reference, incl. partial rotary, the
  gated q layout and q8_0 1-token rows; `qlinear_glu@1` against its reference; GDN gating against the torch composition.
- **Small-step L2 for the paths propagate never runs** (it scores prefill only and bounded runs batch 48): `holon_judge
  l2 --max-step-tokens 1` against the `*-mmvq-floor` goldens (gate/up + SwiGLU, D3's decode rows); `--max-step-tokens
  8`, 7 and 4 against the floor of the path `mmvq_rows` picks (D2), on every lock whose shapes receive an entry; floors recorded where missing (the 27B
  `mmvq-kv-q8_0` floor for q8_0 locks; M0.6's fp4-mode floor for W4A4 entries). Bounded at `max_seqs` 1 and 8, the
  baseline at the same value.
- **Plants (two new):** a KV slot off by one only at page boundaries (D3); a RoPE position off by one on decode rows
  only (D3; bounded catches it). Re-verified with a "fired" count > 0: `ggml_stale_q8_1` (after D6), `replay_off_by_one`
  and `verify_writes_state` (after D5). Plants are verified once, not kept as tests.
- **Locks:** propagated once at the close; the snapshot trace gains a capped chunk beside decoders when D7 ships.

## 7. Steps

| # | Work | Checks | Days |
|---|---|---|---|
| 0 | Re-measure on 5080 / 4090 / 5090 (D1); the benchmark runner with max-gap rows; §5 re-stated | records | 2 |
| 1 | The ABI 0.2 commit (§3.1); the `triton_attn` shared-memory fix; `mmvq_rows` per shape and D2's entries from a repeated sweep of the candidate shapes | `triton_attn` L0 (widths × configs × archs) and the bitwise check; agreement test; guarded rows 4 / 8; small-step L2 | 2 |
| 2 | bf16 `kv_write`, D4, D11, D6 (bitwise first), D3 (bounded); the FP4 repack holes | L0 ×3, L2, bounded; keep-or-revert per §1.1; the 30B NVFP4 fit | 4.5 |
| 3 | D5 glue | L1.5 bitwise; bounded spec | 1.5 |
| 4 | Mixed steps (§3.1, D7; the activation-mode startup check); design §7.1 / §4.2 and plan §10 | core toy proof; bounded mixed-vs-today; agent + long prompt on 5080 / 4090 with A/A; the q8_0 decode rows' cost | 3 |
| 5 | Megakernel gate (D8, GEMV time from the sweep or unprofiled); prototype only on "go" | records | 1–5 |
| 6 | Close: locks once, bars, panel code review, plan status | review clean | 1.5 |

## 8. Memory and suite time

New persistent buffers: D6's prepared-input buffer (one per layout, bounded by the step's largest activation), D5's
layer axis on the replay buffers (bounded by the verify step), and the mixed step's activations (in the worst-step
profile). Lean tests: bitwise checks live in existing L0 loops; two new plants, verified once.

## 9. Risks

- **Mixed steps and speculative deployments:** alternation keeps every speculative invariant but helps less and costs
  long-prompt TTFT (§5); mixing verify rows is the follow-up (its seam list: GQA / GDN per-row-kind planners and rewind,
  the executor's and verifiers' uniform-width views, both drafters, the scheduler's single k, a device-length prefill
  kernel).
- **Approvals:** D2 (where shipped), D3, D6's bounded fusions and D7 move snapshots; one propagation at the close keeps
  it to one approval round per class (plan §9: > 1 approval per day is a trigger).
- **The recalibrated simulator** still models, not measures: step 4's real runs decide.

## 10. Review findings, resolved

**Rev 3.2 re-check (rev 3.3, 2026-10-14).**

| Finding | Resolution |
|---|---|
| Correctness (major): D8 priced the activation quantize as GEMV | the GEMV kernels' time = the sweep's calls less the step's profiled q8_1 quantize time (`step0_report.d8`); f 26.9% on the 5090 (1.174×, prototype territory today), 15.8% on the 5080, 14.5% on the 4090; §5, §11.6 restated; step 5 decides after the fusions with the same method (D8) |
| Correctness m1: D9 at every candidate, not the shipped set | computed at `shipped`: 1.2477 on the 5080 (not fired); recomputed after step 1's repeat sweep fixes the set (D9, §11.7) |
| Correctness m2: `triton_attn`'s stage fallback stopped too early | `num_stages` steps down to 1 before `block_n` halves (one stage fits on sm_120 at BLOCK_M 64, head_dim 256, bf16); the bitwise check covers 3 vs 1 stage (§4, §6) |
| Architecture m1: the activation-mode refusal belongs in std | `Linear` and `ExpertLinear` refuse at bind time from `BuildCtx.options`' `chunk_with_decode` (added to the listed options in the ABI 0.2 commit), for `qlinear@2` and `linear_experts@2` (§3.1) |
| Architecture m2: "recorded in the lock entry" | dropped: the effective config is a deterministic function of (class, facts, declared widths), pinned by the plugin's `src` hash, the widths from the lock's `[engine]` (§4) |
| Architecture m3: stale header | updated: Gate B is merged; the branch is rebased on 4a44fdc (header) |
| Numerics m1: D2's entries assume today's cast, which D4 removes | step 1's repeat sweep times both mmq variants; with D4 kept the entries are re-derived from the no-cast column before the close's propagation; with D4 reverted they stay (D2) |

**Step 0 panel round (rev 3.2, 2026-10-14).** No blocker; the lead's decisions on each finding:

| Finding | Resolution |
|---|---|
| Numerics MAJOR: D2 per type (or per file) mixes shapes that win and lose | D2 keyed per shape (class, type, in, out, mode); `mmvq_rows` gains `in` / `out`, `QLinear` passes them; entries at rows 4, 8 and 7 with ≥ 3% per call and > 3 sd across repeats; shipped where their sum is ≥ 1% of some lock's row; rows 4 / 8 guarded on every lock with an entry; §1.2 reworded, the per-file reading recorded as replaced (§4 D2, §11.5) |
| Architecture M1: mixed steps can run a row at another activation mode | `chunk_with_decode > 0` refuses at startup when any `qlinear` site's `config["act"]` differs between the decode-row and chunk paths (generic, no format names); D7 therefore serves the a8 GGUF locks (§1.3, §3.1, D7) |
| Both MAJORs: `triton_attn` exceeds sm_120's shared memory (DFlash2 / bf16 KV on the 5090) | a step-1 bug fix, not keep-or-drop: BLOCK_M ≤ 64 with query rows tiled over the grid (per-row arithmetic unchanged, checked bitwise), then fewer `num_stages`, then a bounded `block_n` halving; decided at build for the declared widths and recorded in the lock entry; L0 over class config × width × KV format × head_dim with the compiled shared memory checked per arch (§4, §6, §7) |
| Architecture m2, numerics m3: bitwise fusions dropped per class | bitwise fusions stay on everywhere once kept (shared q8_1 on the 4090 too); D3 runs on every step, prefill included; per-class keep / drop only for bounded items and D2 (§1.1, D3, D6, §11.11) |
| New: NVFP4 `scale2` after mmvq above one column (3.3–7.2% of NVFP4 B4/B8) | D11 (bitwise): the multiply in mmvq's epilogue after the full reduction, today's launch variant, one RNE rounding to bf16; L0 in the qlinear loop at rows 2–8 with a non-power-of-two `scale2`, by value; the MoE path already fused (included only if step 2's profile says otherwise); keep-or-revert; the W4A4 activation copy recorded only (§4 D11) |
| Numerics m2: D8's B from the profiled GEMV time / r2 (superseded by rev 3.3) | r2 > 1 is PDL overlap, not inflation; GEMV time priced from the per-call sweep: f 22.1% (5090) and 12.7% (5080), both stop today; step 5 uses the sweep or unprofiled time (§11.2, §11.6, D8) |
| Numerics m4: D3, D5 and gate/up savings read as measured | labelled upper bounds; "other glue" on eager rows includes host idle (§5, §11.3, §11.11) |
| Correctness m1: D9 at a table that will not ship (superseded by rev 3.3) | recomputed at the per-shape table: 1.245 over four sweeps, not fired; upside 3.4 ms (12.6% of the 5080's MTP verify step) recorded; D9 stays outside M0.7 (§11.7) |
| Correctness m2: the clock sampler picked the GPU by `CUDA_VISIBLE_DEVICES` | `Sampler` names torch's current device by UUID (`bench/clock_check.py`). No test: `bench/` is outside the suite and a test would need a GPU and a second device order; checked by hand with `CUDA_VISIBLE_DEVICES=6,3` and device 1, which now samples physical GPU 3 (it sampled GPU 6 before) |
| Correctness m3: q8_0 KV decode rows in mixed steps go through `_paged_prefill` (no KV split) | the simulator under-costs them; step 4 measures, and may route the decode slice through the decode kernel; the 5090's cap 256 is undecided until step 4 (D7, §11.8) |
| Architecture m4: M0.6's follow-ups | the FP4 repack holes fixed in M0.7 (memory only, step 2); the allocator-setting investigation out of scope; one ABI 0.2 commit at step 1 with every break and addition (`mix_phases` removed, `n_decode`, `chunk_with_decode`, the std.ops minors), the release's step 6 sets 1.0; the frozen guide's `core_abi >=0.1,<0.2` is updated after the legibility runs (§3.1, §4) |

**Rev 3** (no blocker or major; minors):

| Finding | Resolution |
|---|---|
| A-m1: a per-step counter breaks under warm-up + capture | trigger on the layer's index in its group; L1.5 graphed too (D5) |
| A-m2 / B-m2: planners cannot tell decode rows from 1-token prefill rows | `BatchPlan.n_decode` (holon.api minor), in the seam list and budget (§3.1) |
| A-m3: alternation with nothing schedulable | alternate when both kinds are schedulable, else the schedulable kind (§3.1) |
| A-m4 / B-m4: one buffer per layout invalidates earlier handles; contract | key + host generation; mismatch re-quantizes from the carried x; `qlinear@2` instance-contract minor (D6) |
| B-m1: the cos/sin table belongs to the kernel, not the unit | part of the `rope@1` instance contract (D3) |
| B-m3: `glu` fallback direction and selection time | `glu` returns None, SwiGLU composes; marked in `finalize_weights`, selected in `bind_kernels` (D6) |
| B-m5: startup check refuses alternation | only without a drafter (§3.1) |
| B-m6: tautological agreement test; one device's rule | dispatch-vs-`mmvq_rows` test; explicit compute capability (D2) |
| C-m1 / A-nit: A5000 f range | 16–17.5%, 1.07–1.11×; B from unprofiled time (D8, §5) |
| C-m2: A/A from one difference; paired ratio | 2 sd of the ≥ 4 baseline rounds; the ratio defined (§1.3) |
| C-m3: no plain long-prompt prediction | added; the bar selects cap 512 on the 4090 (§5) |
| A-nit: "restores the state rows" | reworded: the launch masks gates and writes windows (D5) |

**Rev 2** (A: correctness, B: architecture, C: numerics / performance / tests).

| Finding | Resolution |
|---|---|
| A-major: D5 "once per step" lets later layers' conv replays re-dirty the pool | one launch writes all accepted-end windows from `window ‖ saved_x`; replays use the saved window buffer as state; host-side trigger; layer axis (D5) |
| B-N1: the judge carries its own crossover rule | one exported `mmvq_rows` used by `QLinear` and both judge sites; agreement test (D2, §2) |
| B-N2 / A-m3 / C-MAJOR-2: gate/up + SwiGLU needs an op and is bounded; gating bounded | `qlinear_glu@1` with reference and `Linear.glu`; both classified bounded with named checks; §1.1, §1.5, §9 updated (D6) |
| B-N3 / A-m4 / C-m6: the memo key is unsound (`_version`, raw-pointer writes, capture, column count) | no memo: an explicit `Linear.prepare_input` handle per (path, layout, mode, shape), lifetime owned by the caller; built only if step 0 shows ≥ 1% (D6) |
| C-MAJOR-1 / B-m5 / C-m5: c optimistic; f includes the GEMVs' own gap; B undefined | c = 0.4–0.55 from the sources, both ends reported; B = the GEMVs' achieved bandwidth, stream bandwidth recorded; f ≥ 24% stated; A5000 projection stops (D8, §5) |
| A-m1 / B-m2: k for prompt-ending rows; alternation must not change the default | mixing only without a drafter (prompt-ending rows k = 0); alternation only when the setting > 0 (§3.1) |
| A-m2: the per-row GDN split changes default prefill steps | per-row-kind paths only in steps the scheduler mixed; decode rows first, two slices, `row` re-based (§3.1) |
| A-m5: NVFP4 `scale2` vs the fixup | multiply after the fixup reduction; NVFP4 fixup case (D4) |
| A-m6: minimum chunk unspecified | `min(64, chunk_with_decode)`; build validates `max_step_tokens ≥ max_seqs + chunk_with_decode` (§3.1) |
| A-m7 / B-m2: "facts accept 1-token rows" is no fact; silent non-mixing | `mixes_row_kinds` prop read at build; startup failure (§3.1) |
| B-m1: lock recording described wrongly | the table is pinned by the plugin's `src` hash (D2) |
| B-m3: D3 reads another plugin's cos/sin buffer | the table's static name and layout are part of `fusion_facts()` (D3) |
| B-m4: D5's trigger and buffers | host-side per-step counter; layer axis (D5) |
| B-m6: promised plan / design edits missing | plan row updated in this commit; design §7.1 / §4.2 and plan §10 with step 4's code (header, §7) |
| C-m1: gap p50 within seed noise; rounds | gate on gap p99 / max ≥ 2×, p50 recorded; ≥ 4 rounds per arm; TTFT paired per turn (§1.3, D1) |
| C-m2: long-prompt TTFT bound has no number; which locks | p99 ≤ +25% on plain locks; speculative alternation recorded with its prediction (§1.3, §5) |
| C-m3: missing floors; `max_seqs` values | floors named and recorded; bounded at 1 and 8 with a matching baseline (§6) |
| C-m4: D3's saving includes `kv_write`; verify rows by TPOT | incremental measurement; §5 split; per-step wall for verify rows (§1.1, §5) |
| C-m7: plant count; redundant fixup plant | two new plants, fixup plant dropped (the bitwise check covers it) (§6, §8) |

**Rev 1**: B1 (verify rows cannot mix) → k = 0 only; B2 → an engine budget, default 0, `mix_phases` deleted; B3 / H1 →
a per-class table, entries only where step 0 pays; M3 → keyed by activation mode; B4 / M2 → no format leak (now
explicit handles); B5 → `fusion_facts()`, judge-built reference; B6 / M7 → speed-of-light gate; B7 / M6 → small-batch
matmul out (D9); B8 / M1 → D4 bitwise, classify, propagate once; M4 / B9 → rewind glue only, providers kept; H2 → gap
metric, A/A, recalibrated simulator; H3 → small-step L2; C-M3 → reworked plants; C-M4 → two timing ratios; C-M5 →
restated savings; minors as listed in rev 2's table (git `c03c370`).

## 11. Step 0 results (2026-10-14)

Step 0 ran on the RTX 5080, 4090 and 5090 at the ggml pin `cbe28bb3` (the fixed CI worktree at 474674e; no core, std
or plugin change). The tables are computed from the records (§11.10) by `bench/step0_report.py` (glue, candidates,
timing ratios, D8, D2, D9) and `bench/mixed_sim.py` (D7); the method of each estimate is in their docstrings.
- **Files:** Qwen3-8B Q4_K_M (rtx-5080 lock 7, every class); Qwen3.8-27B at its class's plain lock (UD-IQ3_S with q8_0
  KV on the 5080; UD-Q4_K_M with bf16 KV on the 4090, borrowed by the 5090); the 27B with MTP (k 3 at one row, 2 at
  four) and with DFlash2 (k 7); Qwen3-8B NVFP4 (q8_1 mode, and native W4A4 on sm_120); Qwen3-8B bf16 (4090, 5090).
- **Rows:** decode B1/B4/B8 at a 512-token context; verify B1 (and B4 for MTP); prefill 512 / 2048, one request, one
  step.

### 11.1 Desktops and clocks

Each desktop was screened before its evidence was recorded (`bench/clock_check.py`: 60 s of 27B decode at batch 4;
nvidia-smi every 100 ms; the record keeps the GPU clock and the slowdown share only):

| class | SM clock p10 / p50 / p90 (MHz) | thermal / hardware slowdown | at the power cap | verdict |
|---|---|---|---|---|
| RTX 5080 | 2670 / 2677 / 2685 | 0 | all samples | steady |
| RTX 4090 | 2715 / 2715 / 2730 | 0 | none | steady |
| RTX 5090 | 2565 / 2655 / 2707 | 0 | all samples | steady (p10 / p50 0.97) |

"Steady" means p10 ≥ 0.9 × p50. The first RTX 4090 desktop lost its GPU a minute into its screen (it fell off the bus:
`nvidia-smi` could no longer find the device). It was released with no record kept; every 4090 record comes from a
second desktop.

### 11.2 The two timing ratios (D1)

- r1 = the unprofiled graphed wall / the eager kernel sum.
- r2 = the profiled graphed kernel sum / the unprofiled graphed wall.
- Both are per decode and verify row (`ratio_*` in the profile records). The SM clock was sampled over a 2 s graphed
  window and a 2 s eager window.

| class | r1 | r2 | SM clock, graphed vs eager windows |
|---|---|---|---|
| RTX 5080 | 0.966–1.005 | 0.995–1.036 | 2632–2760 vs 2655–2752 MHz |
| RTX 4090 | 0.969–1.004 | 0.990–1.002 | equal, 2715–2730 MHz |
| RTX 5090 | 0.948–1.010 | 0.990–1.042 | 2617–2872 vs 2752–2872 MHz |

- **r1 never exceeds 1.05, so no clock investigation was due.** The 2–19% graphed excess of R §1.1 (open question 6)
  is an A5000 clock effect. Its decode B1 rows (`m07-step0_profile-*_rtx-a5000`) give r1 = 1.01–1.13, with the SM
  clock at 1395–1500 MHz graphed against 1755–1905 MHz eager, at the power cap. On the consumer classes, graphed
  kernel time converts 1:1 into wall time.
- **r2 is 0.99–1.04, and above 1 it is overlap, not profiler inflation** (step 0 panel): under programmatic
  dependent launch a kernel starts before its predecessor ends, so the profiled kernel sum exceeds the wall. The
  shares below are profiled kernel time divided by r2 (each category's share of the overlapped sum); D8's GEMV time
  comes from the per-call sweep instead (§11.6).

### 11.3 Glue share per row

Share of the step, in %: graphed for decode and verify, eager wall for prefill. "—" means under 0.05%.
- **cast:** mmq's f32 → bf16 cast, plus NVFP4's f32 × `scale2` → bf16 multiply (on mmq, and on mmvq above one row).
- **bf16→f32:** native W4A4's activation copy.
- **other glue:** everything else (residual adds, gates, copies, index ops, sampling, and the verify step's drafter and
  rewind glue). On the eager prefill rows the step is the wall, so "other glue" there also holds the host's idle time.

| class | model | row | step ms | GEMM | attention | GDN | act quant | cast | bf16→f32 | KV store | q/k norm + rope | norms | silu·mul | other glue |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5080 | 8b-q4_k_m | decode b1 | 6.54 | 82.6 | 3.6 | — | 3.0 | — | — | 4.2 | 1.9 | 1.8 | 1.6 | 1.4 |
| 5080 | 8b-q4_k_m | decode b8 | 8.61 | 74.8 | 11.7 | — | 1.8 | 1.7 | — | 3.5 | 1.4 | 1.7 | 0.9 | 2.5 |
| 5080 | 8b-q4_k_m | prefill 512 | 61.49 | 85.2 | 3.5 | — | 1.7 | 3.2 | — | 0.8 | 0.7 | 0.9 | 0.8 | 3.3 |
| 5080 | 8b-q4_k_m | prefill 2048 | 251.64 | 82.4 | 6.5 | — | 2.2 | 3.1 | — | 0.5 | 0.5 | 0.7 | 2.4 | 1.7 |
| 5080 | 27b-plain | decode b1 | 15.88 | 87.3 | 0.8 | 3.5 | 3.2 | — | — | 0.2 | 0.4 | 2.1 | 1.1 | 1.5 |
| 5080 | 27b-plain | decode b4 | 20.81 | 85.2 | 1.2 | 5.7 | 2.6 | — | — | 0.1 | 0.3 | 1.9 | 0.8 | 2.2 |
| 5080 | 27b-plain | decode b8 | 29.41 | 86.1 | 1.4 | 6.4 | 2.1 | 0.2 | — | 0.1 | 0.2 | 1.4 | 0.6 | 1.6 |
| 5080 | 27b-plain | prefill 2048 | 838.00 | 83.7 | 1.8 | 3.8 | 2.3 | 2.4 | — | — | 0.1 | 0.9 | 1.9 | 2.9 |
| 5080 | 27b-mtp | verify b1 | 27.21 | 81.2 | 0.8 | 5.8 | 2.1 | — | — | 0.2 | 0.3 | 1.6 | 0.7 | 7.3 |
| 5080 | 27b-mtp | verify b4 | 36.08 | 79.3 | 1.4 | 8.6 | 1.3 | 1.2 | — | 0.1 | 0.2 | 1.1 | 0.4 | 6.2 |
| 5080 | 27b-dflash2 | verify b1 | 34.76 | 80.2 | 1.0 | 6.2 | 1.9 | 0.2 | — | 0.1 | 0.3 | 1.2 | 0.5 | 8.3 |
| 5080 | 8b-nvfp4 | decode b1 | 6.97 | 84.0 | 3.4 | — | 2.5 | — | — | 3.9 | 1.8 | 1.7 | 1.5 | 1.3 |
| 5080 | 8b-nvfp4 | decode b8 | 9.38 | 74.3 | 10.4 | — | 2.2 | 3.5 | 0.1 | 3.2 | 1.3 | 1.6 | 1.1 | 2.2 |
| 5080 | 8b-nvfp4 | prefill 2048 | 305.42 | 83.6 | 5.2 | — | 1.6 | 5.0 | — | 0.4 | 0.4 | 0.6 | 1.8 | 1.3 |
| 5080 | 8b-nvfp4-a4 | prefill 2048 | 148.51 | 57.9 | 10.8 | — | 4.5 | 10.2 | 7.0 | 0.9 | 0.8 | 1.3 | 3.7 | 3.1 |
| 4090 | 8b-q4_k_m | decode b1 | 6.90 | 85.2 | 3.5 | — | 2.4 | — | — | 2.9 | 1.7 | 1.8 | 1.1 | 1.4 |
| 4090 | 8b-q4_k_m | decode b8 | 8.47 | 76.3 | 12.2 | — | 2.3 | — | — | 2.5 | 1.5 | 1.9 | 1.0 | 2.4 |
| 4090 | 8b-q4_k_m | prefill 512 | 43.91 | 77.3 | 5.7 | — | 1.9 | 4.8 | — | 0.7 | 0.9 | 1.3 | 1.1 | 6.2 |
| 4090 | 8b-q4_k_m | prefill 2048 | 164.90 | 74.3 | 7.0 | — | 3.5 | 5.1 | — | 0.5 | 0.7 | 1.0 | 4.2 | 3.9 |
| 4090 | 27b-plain | decode b1 | 21.48 | 90.5 | 0.6 | 2.7 | 2.2 | — | — | 0.4 | 0.2 | 1.6 | 0.7 | 1.1 |
| 4090 | 27b-plain | decode b4 | 23.73 | 86.3 | 1.1 | 5.6 | 2.2 | — | — | 0.4 | 0.2 | 1.7 | 0.7 | 1.9 |
| 4090 | 27b-plain | decode b8 | 27.10 | 83.9 | 1.7 | 8.0 | 2.1 | — | — | 0.4 | 0.2 | 1.5 | 0.6 | 1.7 |
| 4090 | 27b-plain | prefill 2048 | 570.29 | 74.8 | 1.3 | 5.5 | 3.9 | 4.9 | 0.1 | 0.1 | 0.1 | 1.5 | 2.9 | 5.0 |
| 4090 | 27b-mtp | verify b1 | 30.87 | 83.1 | 0.8 | 5.1 | 1.8 | — | — | 0.4 | 0.2 | 1.4 | 0.5 | 6.6 |
| 4090 | 27b-mtp | verify b4 | 40.30 | 80.9 | 1.0 | 8.0 | 1.5 | 1.1 | — | 0.3 | 0.2 | 1.1 | 0.4 | 5.5 |
| 4090 | 27b-dflash2 | verify b1 | 31.51 | 78.5 | 1.1 | 6.4 | 1.9 | — | — | 0.5 | 0.3 | 1.4 | 0.5 | 9.3 |
| 4090 | 8b-nvfp4 | decode b1 | 7.35 | 86.3 | 3.2 | — | 2.0 | — | — | 2.7 | 1.6 | 1.7 | 1.1 | 1.3 |
| 4090 | 8b-nvfp4 | decode b8 | 8.90 | 74.5 | 11.4 | — | 1.9 | 3.3 | 0.1 | 2.4 | 1.4 | 1.8 | 0.9 | 2.3 |
| 4090 | 8b-nvfp4 | prefill 2048 | 189.73 | 76.4 | 6.1 | — | 2.8 | 6.0 | — | 0.5 | 0.6 | 0.8 | 3.6 | 3.2 |
| 4090 | 8b-bf16 | decode b1 | 19.15 | 95.6 | 1.2 | — | — | — | — | 1.0 | 0.6 | 0.7 | 0.4 | 0.4 |
| 4090 | 8b-bf16 | decode b8 | 21.36 | 91.6 | 4.7 | — | — | — | — | 1.0 | 0.6 | 0.8 | 0.4 | 1.0 |
| 4090 | 8b-bf16 | prefill 2048 | 194.06 | 86.2 | 5.9 | — | — | — | — | 0.4 | 0.6 | 1.2 | 3.0 | 2.7 |
| 5090 | 8b-q4_k_m | decode b1 | 4.14 | 73.4 | 5.3 | — | 4.6 | — | — | 6.6 | 2.9 | 2.8 | 2.4 | 2.0 |
| 5090 | 8b-q4_k_m | decode b8 | 5.66 | 68.8 | 10.6 | — | 2.8 | 2.5 | — | 5.3 | 2.2 | 2.5 | 1.5 | 3.7 |
| 5090 | 8b-q4_k_m | prefill 512 | 33.90 | 81.4 | 5.8 | — | 1.9 | 2.4 | — | 1.0 | 1.0 | 0.9 | 0.9 | 4.8 |
| 5090 | 8b-q4_k_m | prefill 2048 | 134.00 | 82.4 | 7.1 | — | 2.4 | 2.4 | — | 0.6 | 0.7 | 0.8 | 2.0 | 1.7 |
| 5090 | 27b-plain | decode b1 | 11.76 | 83.2 | 1.0 | 3.9 | 4.3 | — | — | 1.1 | 0.5 | 2.7 | 1.4 | 2.0 |
| 5090 | 27b-plain | decode b4 | 13.88 | 79.7 | 1.4 | 6.3 | 3.8 | — | — | 1.0 | 0.4 | 2.9 | 1.2 | 3.3 |
| 5090 | 27b-plain | decode b8 | 16.77 | 78.6 | 1.8 | 8.0 | 3.0 | 1.3 | — | 0.8 | 0.4 | 2.3 | 1.0 | 2.8 |
| 5090 | 27b-plain | prefill 2048 | 428.38 | 83.7 | 1.4 | 3.6 | 2.6 | 2.6 | 0.1 | 0.1 | 0.1 | 0.7 | 1.8 | 3.2 |
| 5090 | 27b-mtp | verify b1 | 18.90 | 73.5 | 1.0 | 7.0 | 3.1 | — | — | 1.0 | 0.4 | 2.4 | 1.0 | 10.7 |
| 5090 | 27b-mtp | verify b4 | 23.35 | 72.3 | 1.4 | 9.5 | 2.1 | 1.8 | — | 0.8 | 0.3 | 1.8 | 0.7 | 9.3 |
| 5090 | 27b-dflash2-kvq8 | verify b1 | 21.52 | 69.5 | 1.8 | 8.5 | 2.5 | 1.2 | — | 0.2 | 0.4 | 1.9 | 0.8 | 13.1 |
| 5090 | 8b-nvfp4 | decode b1 | 4.38 | 75.2 | 5.0 | — | 3.9 | — | 0.1 | 6.3 | 2.8 | 2.6 | 2.3 | 1.9 |
| 5090 | 8b-nvfp4 | decode b8 | 6.09 | 65.3 | 9.7 | — | 3.6 | 6.2 | 0.1 | 4.8 | 2.0 | 2.3 | 2.4 | 3.4 |
| 5090 | 8b-nvfp4 | prefill 2048 | 159.95 | 83.0 | 5.9 | — | 1.8 | 4.6 | — | 0.5 | 0.5 | 0.6 | 1.5 | 1.7 |
| 5090 | 8b-nvfp4-a4 | prefill 2048 | 79.78 | 57.8 | 11.7 | — | 5.3 | 9.2 | 6.7 | 1.0 | 1.1 | 1.2 | 3.0 | 3.2 |
| 5090 | 8b-bf16 | decode b1 | 10.17 | 92.2 | 1.9 | — | — | — | — | 2.1 | 1.1 | 1.0 | 0.7 | 0.9 |
| 5090 | 8b-bf16 | decode b8 | 11.47 | 87.5 | 5.2 | — | — | — | 0.1 | 2.4 | 1.1 | 1.2 | 0.7 | 1.9 |
| 5090 | 8b-bf16 | prefill 2048 | 152.03 | 88.8 | 6.3 | — | — | — | — | 0.5 | 0.6 | 0.6 | 1.7 | 1.6 |

- **The GEMVs dominate decode:** 69–90% on the quantized files, 92–96% on bf16. The glue that a cheap fusion touches
  is 1–7% per item.
- **The glue share grows with the card.** On 8B Q4_K_M B1 decode, the KV store is 4.2% of the step on the 5080, 2.9%
  on the 4090 and 6.6% on the 5090, and q/k norm + rope is 1.9%, 1.7% and 2.9%. A faster GEMV shrinks the step while
  the glue stays the same.
- **The cast costs 2.4–3.2% of quantized prefill on the 5080 and 5090, and 4.8–5.1% on the 4090.** On a 64–96 MB L2
  card this is about half the A5000's share (R §1.2); the 4090 pays more than the Blackwell cards. On the 27B
  (GDN) prefill rows the casts include a few outside mmq (510 casts per step against 463 matmul calls), so the
  estimate is an upper bound of about 10% there.
- **NVFP4 (q8_1 mode) pays for `scale2` above one row:** 3.3–6.2% of decode B8. The q8_1 path multiplies after mmvq
  above one row (`plugin.py:227-230`); llama.cpp fuses it into mmvq at one row only. Native W4A4 prefill also pays
  6.7–7.0% for the bf16 → f32 activation copy, on top of a 9–10% `scale2` share.
- **Classifier note:** the records were made at 474674e, where GDN gating's `neg` kernels count as "other".

### 11.4 Verify-step breakdown

Region times are from the eager trace, scaled to the graphed step. "x decode" is the verify step over the plain decode
step at the same batch.

| class | drafter | row | step ms | x decode | trunk (its GEMM) | drafter | head | rewind (its glue) | verifier + rest |
|---|---|---|---|---|---|---|---|---|---|
| 5080 | 27b-mtp | verify b1 | 27.21 | 1.71 | 19.44 (16.70) | 4.74 | 0.96 | 1.95 (1.15) | 0.11 |
| 5080 | 27b-mtp | verify b4 | 36.08 | 1.73 | 28.42 (24.20) | 3.62 | 1.04 | 2.89 (1.32) | 0.12 |
| 5080 | 27b-dflash2 | verify b1 | 34.76 | 2.19 | 27.42 (24.14) | 3.49 | 1.01 | 2.30 (1.20) | 0.55 |
| 4090 | 27b-mtp | verify b1 | 30.87 | 1.44 | 21.50 (18.49) | 5.86 | 1.26 | 2.11 (1.26) | 0.13 |
| 4090 | 27b-mtp | verify b4 | 40.30 | 1.70 | 31.12 (26.63) | 4.34 | 1.61 | 3.10 (1.37) | 0.13 |
| 4090 | 27b-dflash2 | verify b1 | 31.51 | 1.47 | 23.40 (20.01) | 3.86 | 1.32 | 2.36 (1.28) | 0.57 |
| 5090 | 27b-mtp | verify b1 | 18.90 | 1.61 | 13.22 (10.52) | 3.09 | 0.66 | 1.83 (1.15) | 0.11 |
| 5090 | 27b-mtp | verify b4 | 23.35 | 1.68 | 17.79 (14.07) | 2.40 | 0.69 | 2.36 (1.24) | 0.11 |
| 5090 | 27b-dflash2-kvq8 | verify b1 | 21.52 | 1.83 | 15.69 (12.48) | 2.60 | 0.65 | 2.06 (1.15) | 0.51 |

- **A verify step costs 1.44–2.19× a decode step.** The extra cost is mostly the trunk's wider GEMMs (+3.8 ms MTP and
  +11.2 ms DFlash on the 5080), the drafter (2.4–5.9 ms) and the rewind (1.8–3.1 ms, 960 kernels).
- **The rewind's glue (masking, the window rebuild, gather and index_copy) is 1.15–1.37 ms on every class:** 3.4–6.1%
  of the step.
- **DFlash2 with the borrowed lock's bf16 KV fails on the 5090.** Graph capture of the width-8 verify stops in
  `triton_attn`'s `paged_attention`, which needs 106,752 B of shared memory against a 101,376 B limit. The row above
  ran with q8_0 KV (`-kvq8`), so its decode ratio compares against a bf16-KV decode step. The 4090 runs the same
  bf16-KV verify. On sm_120 it fails (§11.11).

### 11.5 The crossover sweep (D2), keyed per shape

`bench/qmm_sweep.py` timed today's `QLinear`, graph-timed, for every (type, in, out, mode) of the `qlinear` sites in the
latest lock of every GGUF and FP4 checkpoint (107–113 shapes per class; real GGUF rows, random codes with valid scales
for the HF NVFP4 layouts, enough copies to defeat L2). It records mmvq at rows 1–8, mmq with the cast and without it
(rows 1–9), and llama.cpp's choice. Graphed steps run at bucket rows (1, 2, 4, 8, …) and drafts at their block, so
mmvq and mmq compete at rows 4 (decode B3–B4, MTP verify B1), 8 (decode B5–B8, DFlash verify) and 7 (DFlash's draft).

**Per-shape entries (rev 3.2, D2).** A shape gets an entry — the smallest limit in {3, 6, 7} below llama.cpp's for
which every graphed row it moves to mmq is ≥ 3% faster there — or keeps llama.cpp's rule
(`step0_report.py`, `entry`). The entries ship where their sum on some lock's decode or verify row is ≥ 1%. Step 0 has
one sweep per class (the 5080's 27B shapes four at rows 1 and 4), so the > 3 sd condition is applied at step 1 on a
repeated sweep of these candidates.

| class | shapes with an entry | candidates on a ≥ 1% row (by type) | row savings, with today's cast (after D4) |
|---|---|---|---|
| 5080 | 45 | 26: iq4_xs 7, iq3_s 6, iq2_s 4, iq2_xxs 4, iq2_xs 3, iq1_s 1, iq3_xxs 1 | 27B decode B8 −5.7% (−6.8%); DFlash verify B1 −4.8% (−5.7%); decode B4 −0.24% |
| 4090 | 4 (iq1_s, iq2_xxs) | none | under 0.1% on every row |
| 5090 | 41 | 11: iq4_xs 5, iq4_nl 3, iq3_s 2, q6_k 1 (the 27B's 248320-row head) | 27B decode B8 −4.1% (−4.6%); DFlash verify B1 −4.6% (−5.0%); decode B4 −0.64% |

- MTP verify at one row (4 rows) gains at most 0.5% (the 5090) on any class: mmvq still wins at 4 rows for nearly
  every shape. The 4090 keeps llama.cpp's rule (its Ada tuning holds), and native W4A4 and NVFP4 at q8_1 keep 8 on
  every class.
- Rev 3.1 keyed the table per (class, type, mode); the per-type reading of this section's first version (iq3_s → 7,
  iq4_xs → 6 on the 5080, …) and §11.12's per-file proposal are replaced by the per-shape table (step 0 panel).
- The 5090 has no GGUF lock yet; its entries come from the files it borrows.

### 11.6 D8's f (8B Q4_K_M, B1 decode)

- bytes = the matmuls' weight bytes (4.67 GB, the embedding table excluded), plus the KV read at the graphed window's
  context (0.088 GB).
- B = weight bytes / the GEMV kernels' time: the step's calls priced from the per-call sweep (graph-timed,
  unprofiled), less the activation quantization each sweep call includes (the step's profiled q8_1 quantize time).
  (Rev 3.2 kept the quantize inside the GEMV time, which lowered f; rev 3.1's first version took the profiled GEMV
  sum divided by r2, which is not the GEMVs' time when kernels overlap.)
- f = 1 − bytes / (B × step).

| class | step ms | GEMV ms (sweep − quantize) | B (GB/s) | read-only stream, 2 GiB (GB/s) | f | projected gain, c = 0.4 / 0.55 | gate today |
|---|---|---|---|---|---|---|---|
| 5080 | 6.54 | 5.61 − 0.20 = 5.41 | 864 | 913 | 15.8% | 1.067 / 1.095 | stop |
| 4090 | 6.90 | 5.95 − 0.17 = 5.79 | 807 | 844 | 14.5% | 1.062 / 1.087 | (not a D8 class) |
| 5090 | 4.14 | 3.17 − 0.20 = 2.97 | 1572 | 1683 | **26.9%** | 1.121 / **1.174** | prototype territory |

**The 5090 is in prototype territory today; the 5080 stops.** D8 is defined after D3–D6: if the surviving fusions
saved their upper bounds on this step (9.3% of it: kv_write 3.5%, D3 2.9%, gate/up 2.4%, shared q8_1 0.5%), f would
fall to about 19.4% (1.088 / 1.12: stop), but those are upper bounds. Step 5 decides after the fusions, with this
same method.

### 11.7 D9's trigger

D9 is the target's 4-row / 1-row matmul time over its decode step's calls, at the D2 entries that ship (`shipped`,
per shape; recomputed once step 1's repeat sweep fixes the set), with today's cast; each call's time is the mean over
the sweeps available.

| class | target | ratio | bar 1.25 | upside (4-row calls over their 1-row time) |
|---|---|---|---|---|
| 5080 | 27B UD-IQ3_S | 1.248 (mean of four sweeps: the full one and three repeats; sd of the per-sweep ratios 0.003) | not fired | 3.5 ms, 12.8% of its 27.2 ms MTP verify step |
| 4090 | 27B UD-Q4_K_M | 1.014 | not fired | 0.3 ms |
| 5090 | 27B UD-Q4_K_M | 1.11 | not fired | 1.1 ms |

D9 stays outside M0.7; the upside is recorded for the follow-up. (Rev 3.2 priced it at every candidate entry, 1.245;
at the shipped set it is 1.2477, as the panel computed.)

### 11.8 Mixed steps: the measured plain runs and the recalibrated simulator (D7)

**Measured** (`bench_m07-agent-plain_<class>`): 4 rounds per class of `bench/run.py --lock <the class's plain 27B
lock> --backends holon --corpus judge/prompts/v1-qwen3_5.jsonl --traces agent long_prompt`. The two traces ran in
one worker, so the agent trace ran at `max_model_len` 30208 (the long prompt's), not the lock's 32768 nor the agent
trace's own 6400.

Throughput is steady across rounds:
- RTX 5080: 68.5–69.8 tok/s (sd 0.65, so 2 sd = 1.9%; §1.3's throughput bar stays 5%).
- RTX 4090: 70.2–70.4 tok/s.
- RTX 5090: 73.8 tok/s.

**Simulator** (`bench/mixed_sim.py`):
- **Calibrated per class:**
  - graphed decode walls at contexts 512 and 4096 (`profile-27b-plain*`);
  - eager prefill walls with their host floor, at histories 4096 and 16384 (`prefill-idle-27b*`; at 64 new tokens
    the GPU idles 43% on the 4090 desktop, 38% on the 5090's and 3% on the 5080's);
  - the KV pool's measured tokens.
- **Prefix reuse is limited by the recurrent checkpoints**, modelled on a real engine run (an A5000 run logging every
  admission's hit):
  - one checkpoint per prompt, at its last aligned position, 4 units;
  - a matched chain refreshes its checkpoints;
  - a turn-0 prompt appending fewer than 64 tokens publishes the shared system prompt's checkpoint, which every later
    turn then hits;
  - own-session checkpoints are mostly evicted.
- **Not modelled:** overlap's lag and admission order. The simulator has no fitted parameter.

| class | trace | run | TTFT p50 / p99 ms | paired TTFT | TPOT p50 / p99 ms | max gap p50 / p99 / max ms | tok/s | prefix tokens saved |
|---|---|---|---|---|---|---|---|---|
| 5080 | agent | measured | 1266 / 7851 | — | 32.1 / 76.2 | 755 / 1862 / 1864 | 69.1 | 169k |
| 5080 | agent | today | 1377 / 7968 | — | 32.9 / 75.0 | 1008 / 2672 / 2672 | 68.8 | 167k |
| 5080 | agent | mixed:256 | 1213 / 9586 | 1.019 | 32.3 / 88.9 | 125 / 127 / 127 | 69.8 | 170k |
| 5080 | agent | mixed:512 | 1073 / 8162 | 1.0 | 31.2 / 68.4 | 228 / 232 / 232 | 69.4 | 169k |
| 5080 | long prompt | measured | 313 / 15032 | — | 35.5 / 35.5 | 14290 / 14293 / 14293 | 84.7 | 0k |
| 5080 | long prompt | today | 345 / 14927 | — | 34.3 / 34.3 | 15394 / 15394 / 15394 | 87.5 | 0k |
| 5080 | long prompt | mixed:256 | 345 / 16429 | 1.0 | 33.7 / 33.7 | 170 / 170 / 170 | 89.3 | 0k |
| 5080 | long prompt | mixed:512 | 345 / 15225 | 1.0 | 33.5 / 33.5 | 318 / 318 / 318 | 89.6 | 0k |
| 4090 | agent | measured | 541 / 6406 | — | 30.4 / 72.0 | 460 / 1298 / 1599 | 70.4 | 174k |
| 4090 | agent | today | 508 / 6005 | — | 30.4 / 56.4 | 511 / 1778 / 1778 | 69.9 | 181k |
| 4090 | agent | mixed:256 | 700 / 8550 | 1.262 | 31.5 / 54.1 | 92 / 94 / 94 | 69.2 | 168k |
| 4090 | agent | mixed:512 | 550 / 6448 | 1.006 | 29.0 / 53.0 | 149 / 151 / 151 | 70.0 | 172k |
| 4090 | long prompt | measured | 217 / 9498 | — | 32.8 / 32.8 | 8967 / 8988 / 8988 | 91.9 | 0k |
| 4090 | long prompt | today | 230 / 9405 | — | 32.6 / 32.6 | 9699 / 9699 / 9699 | 92.5 | 0k |
| 4090 | long prompt | mixed:256 | 230 / 11519 | 1.0 | 32.1 / 32.1 | 112 / 112 / 112 | 94.0 | 0k |
| 4090 | long prompt | mixed:512 | 230 / 9468 | 1.0 | 31.3 / 31.3 | 188 / 188 / 188 | 96.2 | 0k |
| 5090 | agent | measured | 318 / 1744 | — | 14.8 / 46.4 | 84 / 1634 / 1676 | 73.8 | 196k |
| 5090 | agent | today | 244 / 1795 | — | 15.3 / 48.9 | 133 / 1947 / 1966 | 73.9 | 189k |
| 5090 | agent | mixed:256 | 278 / 1637 | 1.118 | 15.3 / 40.2 | 70 / 71 / 71 | 73.6 | 206k |
| 5090 | agent | mixed:512 | 242 / 1335 | 1.002 | 14.5 / 40.3 | 118 / 122 / 123 | 74.0 | 207k |
| 5090 | long prompt | measured | 181 / 8082 | — | 22.2 / 22.2 | 7680 / 7692 / 7692 | 135.8 | 0k |
| 5090 | long prompt | today | 187 / 7338 | — | 20.6 / 20.6 | 7562 / 7562 / 7562 | 145.7 | 0k |
| 5090 | long prompt | mixed:256 | 187 / 8827 | 1.0 | 20.6 / 20.6 | 87 / 87 / 87 | 145.8 | 0k |
| 5090 | long prompt | mixed:512 | 187 / 7672 | 1.0 | 20.2 / 20.2 | 153 / 153 / 153 | 148.7 | 0k |

- **Calibration ("today" against "measured"):**
  - TTFT p50 within 6–23% and TTFT p99 within 6%;
  - TPOT p50 within 4%; TPOT p99 within 6%, except the 4090 (56 vs 72 ms);
  - throughput and prefix reuse within 4%;
  - long-prompt TTFT within 2–7%.
  - The simulator overstates the worst gap by 11–43%: it has no overlap, and it counts a whole prefill step as a stall.
- **Predictions for D7** (gate: gap p99 and max ≥ 2× better; paired TTFT ≤ 1.15; long-prompt TTFT p99 ≤ +25%;
  throughput within 5%):

  | class | cap 256 | cap 512 |
  |---|---|---|
  | 5080 | gap max 2672 → 127 ms (21×); paired TTFT 1.02; long TTFT +10.1%; passes | gap max 232 ms (11×); paired 1.00; long +2.0%; passes |
  | 4090 | gap max 94 ms (19×); **paired TTFT 1.26: fails** (the slow host's eager floor); long +22.5% | gap max 151 ms (12×); paired 1.01; long +0.7%; passes |
  | 5090 | gap max 71 ms (28×); paired 1.12; long +20.3%; passes, close to the TTFT bar | gap max 123 ms (16×); paired 1.00; long +4.6%; passes |

  The long prompt stalls the decoders for 14.3 s (5080), 9.0 s (4090) and 7.7 s (5090) today. Every cap brings that
  to one chunk: 87–318 ms in the simulator.
- **Step 0 panel:** the 5090's cap 256 (paired TTFT 1.12, near the bar) is undecided until step 4's runs. On q8_0 KV
  (the 5080's lock) a mixed step's decode rows go through `_paged_prefill`, which has no KV split, so the simulator
  under-costs them; step 4 measures it and may route the decode slice through the decode kernel. D7 serves the a8 GGUF
  locks only: a deployment whose decode-row and chunk paths differ in activation mode refuses mixed steps (§3.1).

### 11.9 The release cells re-measured at the pin (RTX 5090)

`bench/bars.py`, 3 rounds, Holon's ggml at `cbe28bb3` (records `bars_release-m07-*_rtx-5090`; same desktop as §11.1):

| file | pp512 | pp512 @ d8192 | tg128 | tg128 @ d8192 |
|---|---|---|---|---|
| Qwen3-8B Q4_K_M | 1.05 | **0.91** | **0.97** | 0.99 |
| Qwen3-30B-A3B Q4_K_M | 1.09 | **0.94** | **0.93** | 0.98 |
| Qwen3.8-27B UD-Q4_K_M | 1.11 | ≈ 0.96 | 1.03 | 1.05 |

**The cells release step 3 flagged are unchanged at the pin.** The pp512 @ 8K and 5090 tg128 gaps do not come from the
ggml commit, which is now the same on both sides. pp512 @ 8K is attention at depth: Holon's prefill attention against
llama.cpp's flash attention, which grows from 7% of the 8B's prefill at no depth. tg128 0.93 on the 30B-A3B is the
MoE decode path. Neither is an M0.7 candidate (§11.11).

### 11.10 What step 0 built and recorded

- **The release benchmark runner** (`bench/`, used by the close and the release):
  - `driver.py` rows carry each request's maximum inter-token gap.
  - `run.py` adds:
    - arms: `holon@<label>` with `--arm <label>='<flags>'`, merged per key over the lock's or the environment's
      flags; a label without flags is an A/A arm;
    - paired TTFT, the median over (round, session, turn) against the first backend, also per round;
    - gap p50 / p99 / max per trace;
    - ABBA order across rounds;
    - arm flags made public in the record.
  - `report.py` prints the gap p99 and the paired ratio.
  - `workloads.py` adds the `long_prompt` trace: three streams decoding from t = 0 (256 + 1024), then a 30,000-token
    prompt at t = 2 s.
- **Measurement tools:**
  - `step_profile.py`: per-row categories, verify regions, both ratios with clocks (sampled by the device's UUID
    since the step 0 panel), and matmul calls, weight bytes and distinct quantized inputs per step;
  - `qmm_sweep.py`;
  - `clock_check.py`;
  - `mixed_sim.py`;
  - `step0_report.py`.
- **Records**, `evidence/` (GPU names only, flags repo-relative):
  - `m07-step0_{clock, qmm-sweep, profile-<row set>, prefill-idle-27b[-h16k], sim}_<class>_2026-10-14`;
  - `bench_m07-agent-plain_<class>`;
  - `bars_release-m07-*_rtx-5090`;
  - `m07-step0_qmm-sweep-d9-repeat_rtx-5080`;
  - the A5000 `profile-*_rtx-a5000` records: distinct quantized inputs per decode step, a property of the model.
- **Code review:** one reviewer, two rounds. The findings, all fixed:
  - NVFP4's `scale2` was credited to D4 on mmvq rows;
  - "D2 after D4" was the sum of D2 and D4;
  - an arm's dict flags replaced the environment's without a lock;
  - arm flags were unscrubbed;
  - paired TTFT lost its round ids;
  - the simulator's chain refresh order and context slope were wrong;
  - D8's KV context was wrong;
  - bad flags without a lock aborted the run mid-way.
- **No core, std or plugin change; no test added:** the bench is outside the suite, and no fix touched engine code.

### 11.11 §5 re-stated per class

A bounded candidate survives on a class if it saves at least 1% of its step on some release row there (graphed decode
B1/B4/B8, verify, prefill 512/2048). A bitwise candidate that survives on any class runs on every class (rev 3.2,
§1.1). Shares are the largest over the rows; D3, D5 and gate/up + SwiGLU are upper bounds (the fused kernel is taken
to cost nothing).

| item | kind | 5080 | 4090 | 5090 | verdict |
|---|---|---|---|---|---|
| D2 crossover, per shape | bounded | 27B B8 −5.7%, DFlash −4.8% (26 candidate entries) | none | B8 −4.1%, DFlash −4.6% (11) | **keep** on 5080, 5090 (entries confirmed by step 1's repeated sweep); none on the 4090 |
| bf16 `kv_write` | bitwise | 8B B1 −2.2% (B8 −1.8%) | −1.5% (−1.3%) | −3.5% (−2.8%) | **keep**, every class |
| D3 `qk_rope_kv` (on top; upper bound) | bounded | 8B B1 −1.9% | −1.7% | −2.9% | **keep** on every class (each ≥ 1%), on every step, prefill included |
| D4 mmq bf16 out | bitwise | prefill −2.4–3.2% (NVFP4 −5.0%, W4A4 −10.2%) | −4.8–5.1% (NVFP4 −6.0%) | −2.4–2.6% (NVFP4 −4.6%, W4A4 −9.2%); 27B B8 −1.3% | **keep**, every class |
| D5 rewind glue (upper bound) | bitwise | verify −3.4–4.2% | −3.4–4.1% | −5.3–6.1% | **keep**, every class |
| D6 gate/up + SwiGLU (upper bound) | bounded | 8B B1 −1.6%, 27B B1 −1.1% | −1.1%, −0.7% | −2.4%, −1.35% | **keep** on every class |
| D6 shared q8_1 | bitwise | 27B B1 −1.5% | −0.96% | −1.9% | **keep**, every class (the 4090 included) |
| D6 GDN gating | bounded | prefill −0.3% | −0.4% | −0.4% | **drop** everywhere: about 2 ms of an 838 ms step |
| D11 NVFP4 `scale2` in mmvq above one column | bitwise | NVFP4 B4/B8 −3.5–4.2% | −3.3–3.6% | −6.2–7.2% | **new, keep-or-revert at step 2**, every class |
| D7 mixed steps | — | cap 256 or 512 | cap 512 (256 fails paired TTFT) | cap 512; 256 undecided | **keep**; step 4's real runs pick the cap; a8 GGUF locks only |
| D8 megakernel | — | f 15.8% → stop | (14.5%) | f 26.9% → prototype territory today | step 5 decides after D3–D6, same method |
| D9 small-batch matmul | — | 1.248 | 1.014 | 1.11 | not triggered; outside M0.7; recomputed after step 1 |
| `triton_attn` shared memory | bug fix | — | — | DFlash2 / bf16 KV fails capture | **fix at step 1** |

**Measured, not in §5:** native W4A4's bf16 → f32 activation copy, 6.7–7.0% of NVFP4 W4A4 prefill on sm_120 (recorded
only); pp512 @ 8K (0.91–0.96) and the 30B-A3B's tg128 (0.93) are not ggml (§11.9).

### 11.12 For the lead (step 0's report; answered by §10's step 0 panel round)

Step 0 asked the lead about D2's per-entry rule (proposing a per-file table), D9 at its bar, D8 on the 5090 (then read
as 25.2%), the D7 cap per class, the items outside §5 and the lost desktop. The answers and corrections are §10's step
0 panel round and the rev 3.2 re-check: D2 per shape (§11.5), D9 not fired (§11.7), D8 26.9% on the 5090 and
decided at step 5 (§11.6), the cap decided at step 4 with
256 open on the 5090 (§11.8), D11 and the `triton_attn` fix added (§4), and bitwise fusions on every class (§11.11).

### 11.13 After the step 0 panel (2026-10-14)

- **Rebased onto `main` at 4a44fdc** (M0.6 Gate B: `kernels.triton_fp8`, `std.weights.hf`'s merge order). The rebase
  had no conflicts; `uv sync --group dev` re-synced the venv. Full GPU suite (`pytest core std plugins judge testing
  tools`, two A5000s): 1155 passed, 63 skipped, 18 min 55 s. `scripts/ci/run_all.sh` green.
- **M0.6's merge-order change propagated on its own** (commit 9b1db33, `evidence/propagate_m06-merge-order_2026-10-14.json`):
  every development-class lock that pins `std.weights.hf` — rtx-4090 Qwen3-8B bf16 (lock 11), nvidia Qwen3-8B NVFP4
  (2), nvidia Qwen3-30B-A3B NVFP4 (2); rtx-5080 nvidia Qwen3-8B NVFP4 (2). Each passes L2 against its floor (the
  bf16 floor; the W4A8 mmq floor for NVFP4), its propagate trace is byte-identical to the previous lock's, and the
  snapshot ×3 at the new lock's settings and pins is run-identical. Approved under the delegation. The rtx-5090 bf16
  lock (a release class) waits for the release's step 7. Desktops screened first (RTX 5080 2625 / 2662 MHz p10 /
  p50, RTX 4090 2670 / 2670, no slowdown). The first RTX 4090 desktop of this run lost its GPU at its screen — the
  same desktop that failed in step 0 — and was released; a third desktop ran the propagation.


## 12. As built (step 1, 2026-10-14)

Commits 85942db (`mmvq_rows`), 68bf31a (ABI 0.2), 1e51ba2 (`triton_attn`), d5c7e6d (std's refusal), fac1225 (D2's
table and the judge's per-shape floors) and the review fixes after them. Desktops screened first (`bench/clock_check.py`,
60 s of 27B decode at batch 4): RTX 5080 SM clock p10 / p50 2685 / 2715 MHz, RTX 5090 2565 / 2655 MHz, no thermal or
hardware slowdown, both steady. The 4090 has no D2 candidate (§11.5) and was not used.

### 12.1 ABI 0.2

One commit (68bf31a). Breaks: `SchedulePolicy.mix_phases` removed (nothing called or implemented it); `EngineConfig`'s
fields after `max_step_tokens` and `BatchPlan.ext` move by one index. Additions: `BatchPlan.n_decode` (0 unmixed),
`EngineConfig.chunk_with_decode` (default 0, checked >= 0) and its `BuildCtx.options` key, the kernel prop
`mixes_row_kinds` and the optional planner hook `MixesRowKinds`. `core/abi.txt` regenerated; `check_abi --base main`
accepts it as a bump (0.1 → 0.2). Every manifest's `core_abi`, the test fixtures' included (38), is `>=0.2,<0.3`; the
guide's example and design §4.1's follow. Behaviour is today's: nothing reads the new fields but std's refusal (§12.5),
so `--chunk-with-decode` is a flag and a lock field that does nothing until step 4.

- **Deviation: the std.ops minors of D3 and D6** (`qk_rope_kv@1`, `qlinear_glu@1`, `fusion_facts`, `glu`,
  `prepare_input`) are not in the ABI commit: they land with their code at step 2. They are std contracts and unit
  members, which no core listing renders and which are free to add (`check_abi`'s contract rule rejects only
  removals), so step 2 needs no second core bump; adding them before their keep-or-revert would make a revert a
  contract removal.
- Core: 4129 code lines (4123 at the base; +6).

### 12.2 `mmvq_rows` (D2)

`kernels.ggml_cuda.mmvq_rows(fingerprint, type, mode, in, out)`: the class's `CROSSOVER` entry for the shape and
mmq's activation mode, else llama.cpp's threshold at the class's compute capability. The C API's `use_mmvq` takes the
capability explicitly (`ggml_cuda_should_use_mmvq` is host code), so any SM's module answers for any class, no device
needed (`rule_module`). An entry must lie in 1 … llama.cpp's limit (an error otherwise), so mmvq runs every row count
up to it. Users: `QLinear` (its `mmvq_rows`), the judge's DFlash floor (`drafter._q8_1_matmul`), `gguf.mmvq_step`
(dense matmuls by shape, plus every class entry of their type and `in`: a merged matmul's shape is not the tensor's)
and the per-shape floors (§12.6). `test_qlinear_dispatch_is_mmvq_rows` runs `QLinear` for every lock class (the
build's fingerprint names the class) on every shape with an entry and a k-quant without, rows 1–9, and asserts the
counted path is `mmvq_rows`'s; it fails when `QLinear` ignores the entries. It also checks llama.cpp's rule at another
class's capability (Q4_K: 8 rows on Ada, 5 on Blackwell).

### 12.3 D2's table, the guard and D9

`bench/qmm_sweep.py --shapes` re-swept step 0's candidates (§11.5: 26 on the 5080, 11 on the 5090) at rows 1, 4, 7,
8, graph-timed, mmvq and both mmq variants: ten sweeps per class, five, then five more about 15 minutes later; and the
27B lock whole at rows 1 and 4 (D9). `bench/d2_table.py` takes a row as passing when mmq's mean is >= 3% under mmvq's
and the mean per-sweep difference exceeds 3 sd of the differences; an entry is the smallest limit in {3, 6, 7} whose
moved graphed rows all pass. No entry moves row 4. Records: `evidence/m07-step1_qmm-sweep-repeats_<class>`,
`m07-step1_d2_<class>` (with the cast), `m07-step1_d2-no-cast_<class>` (D4's re-derivation).

| class | entries shipped (of candidates) | limits | moved rows' margin, min–max | row saving (with the cast) |
|---|---|---|---|---|
| RTX 5080 (27B UD-IQ3_S) | 25 of 26 (iq3_xxs 5120x34816 drops: 1.8% at row 8) | 6: 6 shapes, 7: 19 | row 8: 3.0–20.5% (7–128 sd); row 7 (limit 6): 4.6–12.2% (12–89 sd) | decode B8 −4.9%, DFlash verify −4.1% |
| RTX 5090 (27B UD-Q4_K_M, borrowed) | 10 of 11 (iq4_nl 5120x17408 drops: 3.9%, 3 sd at row 8) | 6: 4 (q6_k head included), 7: 6 | row 8: 4.0–17.5% (> 3–17 sd); row 7 (limit 6): 4.9–28.1% (4–10 sd) | decode B8 −3.5%, DFlash verify −3.9% |

- The five-sweep table differed on the 5090 only (it kept iq4_nl 5120x17408 at 7 and gave iq4_xs 5120x34816 6): the
  ten-sweep table ships. Step 0's re-stated savings were −5.7% / −4.8% (5080) and −4.1% / −4.6% (5090).
- **Without the cast** (D4 kept): 25 entries on the 5080 (iq2_xs 5120x17408 and iq4_xs 5120x12288 move from 7 to 6;
  B8 −5.5%, DFlash −4.7%), 11 on the 5090 (iq3_s / iq4_xs 5120x17408 from 7 to 6, iq4_nl 5120x17408 gains 7). The
  table is re-derived from that column before the close's propagation if D4 is kept.
- **Guard** (rows 4 and 8, ABBA, two runs per arm, the entries against `--llama-crossover`): `holon_tg` tg128 at batch 4
  and 8 on the lock's flags; `step_profile`'s graphed verify step of MTP (4 rows) and DFlash2 (8 rows). No regression:

  | class | tg128 b4 (tok/s) | tg128 b8 (tok/s) | MTP verify B1 (ms) | DFlash2 verify B1 (ms) |
  |---|---|---|---|---|
  | RTX 5080 | 196.4 vs 195.8 (+0.3%; sd 0.6) | 289.7 vs 277.1 (**+4.5%**) | 26.93 vs 26.95 (0.0%) | 32.65 vs 34.18 (**−4.5%**) |
  | RTX 5090 | 291.2 vs 290.7 (+0.2%; sd 0.7) | 497.7 vs 482.7 (**+3.1%**) | 18.84 vs 18.86 (−0.1%) | 20.88 vs 21.64 (**−3.5%**) |

  The delivered saving is at least half the re-stated one on every row (§1.2). Records `m07-step1_guard_<class>`.
- **D9 at the shipped set:** 1.2423 on the 5080 (15 sweeps: step 0's four and this step's eleven; was 1.2477; upside
  3.4 ms), 1.1116 on the 5090 (12 sweeps), 1.014 on the 4090 (no entry): not fired.
- **D8 prep:** `qmm_sweep.py` now times mmvq's q8_1 quantize alone per `in`, graph-timed (a new benchmark-only C entry
  point, `quantize_q8_1`), so step 5's GEMV time is a sweep's call less its own quantize, with no profile. In these
  sweeps: 0.66–0.98 µs per call at one row (in 5120–17408).

### 12.4 `triton_attn`'s shared memory

- BLOCK_M is capped at 64 (query row, head) pairs; wider steps tile the rows over the grid (program 0 is (row,
  tile)), every tile walking the same KV range.
- `fitted` lowers the class's pinned config at build until the kernel, compiled for the device's SM by Triton's
  compiler alone (`shared_bytes`: the JIT's own specializer, Triton internals pinned by uv.lock), fits the SM's shared
  memory (`SHARED`, the lock classes' 101,376 B; another SM's device) at one split and several: stages down to 1,
  then block_n halved to 16. It is fitted per BLOCK_M at the widest declared width (decode 1; verify 1 + k for k up
  to `max_draft_tokens`), a pure function of (class, facts, widths); an undeclared width (a drafter's block) is
  fitted by the same function when it first runs. Offline, it reproduces §11.4's 106,752 B. Only head_dim 256 / bf16
  KV / BLOCK_M 64 on sm_120 changes: the 5090's and 5080's (64, 4|8, 3) run 1 stage (73,728 B).
- **L0** (`test_triton_attn_configs_fit_every_class`): every lock class's config × KV format (bf16, fp8, q8_0) ×
  head_dim (128 with 8 heads per KV head, 256 with 6) × width (1, 4, 8, 16), compiled for the class's SM, at one
  split and several, within its shared memory; it fails with the pinned configs (5080 and 5090, head_dim 256, bf16).
  First run ≈ 3 min of compiles, then Triton's cache.
- **Bitwise:** 3 stages = 1 stage, one split and several. `holon_judge.kernel_identity` against 68bf31a: 48 / 48
  bitwise on sm_86, sm_120 (5080 and 5090) — every shape a lock runs today (≤ 64 pairs). **Deviation:** tiling is
  bitwise against the single tile at head_dim 256 (the 27B's 16-row blocks) but not at head_dim 128: Triton lays a
  chained dot's warps along the rows when rows ≥ head_dim, so a 128-pair tile and a 64-pair tile reduce the row sums
  in another order — one bf16 ulp in 1 of 262,144 values at 16 rows × 8 heads. No lock runs more than 64 pairs
  (every triton_attn lock has `max_draft_tokens` 0); a future lock that does (Qwen3-30B-A3B verify at k ≥ 8) is
  tuned on the tiled kernel. The test asserts bitwise at 256, one ulp at 128.
- **5090, DFlash2 with bf16 KV** (the borrowed 4090 lock, k 7): captures and runs, verify B1 21.56 ms graphed
  (`m07-step1_profile-27b-dflash2_rtx-5090`); step 0 needed q8_0 KV (21.52 ms).

### 12.5 std's activation-mode refusal

`Linear` (through `_Quantized.bind`) and `ExpertLinear` refuse at bind a kernel whose spec's `config["act"]` names
more than one mode when `chunk_with_decode` > 0 and the deployment has no drafter (with one, steps alternate and never
mix — rev 3's B-m5; a review finding). Generic: no format is named; it catches `kernels.ggml_cuda`'s native FP4
(mmvq q8_1, mmq fp4) and `kernels.triton_fp8` (decode bf16, prefill fp8). Toy proof: `tests.toy_quant` registers a
one-mode spec and, preferred under `quant.act = a4`, a two-mode one; the linear and a merged expert stack refuse the
latter at 256 without a drafter and bind it at 0 or with one (the test fails without the check).

### 12.6 The small-step L2 and the per-shape floors

`holon_judge golden --q8-1 rows --rows N --crossover <class>:<sm>` records D2's floor: each dense GGUF linear on the
path its matmul takes at N rows on the class (`gguf.row_paths`), its shape the engine's after std's merges
(`engine_shapes`: GQA's q/k/v, else q/k; SwiGLU's gate/up; same type and `in`). The floor records the path per shape
(`routes`); the candidate records each ggml kernel's `mmvq_rows` per shape and its GPU class; compare requires the
class, the step size and every shape's path at it to agree, and every call in the floor's mode. A tiny GGUF with mixed
types checks `engine_shapes` against the engine's sites. At 4 rows no entry moves, so the existing mmvq floors apply.

| class | `--max-step-tokens` | floor | top-1 | KL p99 (floor) | ΔPPL vs floor | verdict |
|---|---|---|---|---|---|---|
| RTX 5080 | 4 | mmvq, q8_0 KV | 98.07% (≥ 97.78%) | 3.18e-1 (3.06e-1; ≤ 6.11e-1) | −0.12% ± 0.41% | PASS |
| RTX 5090 | 4 | mmvq | 98.26% (≥ 97.94%) | 2.28e-1 (2.34e-1; ≤ 4.69e-1) | −0.36% ± 0.42% | PASS |
| RTX 5080 | 8 | per shape (38 mmq, 12 mmvq), q8_0 KV | 97.50% (≥ 97.30%) | 6.93e-1 (7.04e-1; ≤ 1.41) | −0.57% ± 0.40% | PASS |
| RTX 5090 | 8 | per shape | 97.07% (≥ 96.76%) | 8.66e-1 (7.58e-1; ≤ 1.52) | −0.49% ± 0.84% | PASS |

At 8 rows the candidate and its floor both sit well above the 4-row ones (KL p99 ≈ 0.7–0.9 against 0.2–0.3): the
mmq path's rounding, which the floor emulates; the verdict is against the floor of the same paths. `--max-seqs 4` (the
locks' value). The 8-row floors took ≈ 27 min each on an RTX A5000.

Records `m07-step1_l2-<n>_<class>`; the floors are
`judge/goldens/qwen3.8-27b-ud-iq3_s/v1-bfloat16-rows8-rtx-5080-kv-q8_0-floor` and
`judge/goldens/qwen3.8-27b-ud-q4_k_m/v1-bfloat16-rows8-rtx-5090-floor`.

### 12.7 Review

One reviewer (correctness, architecture, numerics / tests) on the aggregate change. Round 1: no blocker; one major
(the guard and the small-step L2 had not run yet: run above), six minors and nine nits. Fixed: per-shape floors tied
to the candidate's per-shape paths (they checked only the set of paths); `d2_table --variant` re-derives the whole
no-cast table (it kept only the with-cast set's keys); `mmvq_rows` and the floors need no GPU; the refusal exempts
deployments with a drafter; `triton_attn` fits to the class SM's table, not the test device's; the head's calls on a
per-shape floor are checked by mode only; docs (§3.1, this section, the guide's `core_abi`). Recorded, not changed:
the DFlash floor's `_q8_1_matmul` keys limits by unmerged tensor shapes (latent: its draft tensors have no entry);
a step shorter than the floor's rows (a prompt's last) may take the other path; `--crossover`'s SM is not checked
against the class. Round 2: no blocker, major or minor; four nits (fixed or recorded: the D9 label on the no-cast
table, the floor flags' error, step 4's drafter predicate in §12.8; the per-shape floor test still skips without a
GPU). Round 3 (the 8-row L2, the floors' routes recomputed without a GPU, §12 against the evidence): none.

Suite at 34235ea (code as committed; two RTX A5000s): `pytest core std plugins judge testing tools` 1185 passed, 65
skipped in 16 min 29 s (step 0's base: 1155 / 63, 18 min 55 s; the new L0's compiles are cached by Triton after the
first run, ≈ 3 min cold); `scripts/ci/run_all.sh main` green.

### 12.8 For the lead

- The std.ops minors' deferral to step 2 (§12.1), and `chunk_with_decode` living as an inert flag until step 4.
- The refusal's drafter exemption (§12.5) reads rev 3's B-m5 over §1.3's literal text.
- `triton_attn`'s tiling is bitwise only where Triton keeps its warp layout (§12.4); no lock changes.
- For step 4: the refusal stands down whenever `max_draft_tokens` > 0, but the engine builds a drafter only if a
  draft policy is also installed (`engine.py:145`); step 4 mixes only when `max_draft_tokens` is 0, or both use the
  engine's own predicate.
- `main` gained 899b661 (the legibility gate retired) after this branch's base; the branch is not rebased.
