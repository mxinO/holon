# M0.7 — performance: fused glue, mixed steps, a gated megakernel experiment

Status: **closed, acceptance met** (2026-10-15; §17: the close panel's findings fixed, the single propagation, acceptance
and deviations). Rev 3.3, steps 0–5 built (§11 step 0, §12 step 1, §13 step 2, §14 step 3, §15 step 4, §16 step 5: D8
stops). Step 0 reviewed and re-checked. Revision 3.1 was reviewed clean by a panel of three (correctness,
architecture, numerics / performance / tests) over four rounds. Step 0 ran (§11), and the panel's review of it and the
lead's decisions are in §10's "Step 0 panel round"; §1, §3.1, §4, §5, §6, §7 and §11 carry them. The panel re-checks
rev 3.2 before step 1; rev 3.3 fixes its findings (§10, "rev 3.2 re-check").

Scope: plan §2 (M0.7: "fused prefill glue (KV-store kernel, add+norm, q/k norm) and a model-specific megakernel
experiment (design §5.7)"), narrowed and extended by what M0.3–M0.5 recorded for M0.7: the verify-step glue
(m04-spec §13), mixed prefill/decode steps (M0.5 measured streams silent up to 36 s behind a long prefill), and the
mmvq/mmq crossover. The small-batch quantized matmul (plan §6.1) leaves M0.7 as a recorded follow-up with a measured
gate (D9); the forward/sample split stays deferred (M0.5 D2: 1.061× / 1.026×, under its 1.10× rule). Measurements and
prior art are in the local research note `docs/brainstorm/m07-perf-research.md` (not published) ("R §n", taken on an RTX A5000) and in
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
  decides after D3–D6 with the same method. Step 5 (§16, the quantize isolated and graph-timed): 15.6% on the 5090
  (1.094×) and 8.6% on the 5080 (1.050×): **stop**.
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
| Megakernel (D8), step 5 (§16) | 8.6% (1.050×): stop | — | 15.6% (1.094×): **stop** |

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
  only (D3; predicted caught by bounded; it is not: §13.7). Re-verified with a "fired" count > 0: `ggml_stale_q8_1`
  (after D6), `replay_off_by_one`
  and `verify_writes_state` (after D5). Plants are verified once, not kept as tests.
- **Decode consistency at the attention unit** (the lead's decision after §13.7's judge gap, step 4):
  `plugins/tests/test_attention_kv_quant.py::test_a_decode_row_is_the_last_prefill_row`, in the GQA L0 module. The
  GQA unit of a tiny Qwen3.5 on the GPU kernels (D3's fused op selected; partial rotary, the gated q layout, a norm
  offset), bf16 and q8_0 KV: prefill T tokens against prefill T − 1 then the last as a one-token decode row, T 17 (the
  decode row opens a page) and 40. The last token's output agrees within the attention budget (another kernel computes
  it: usage 0.006 of 1), and the KV it wrote is bitwise (inputs in {−1, 0, 1} and projection weights in {−1, 0, 1}
  / 8 make the projections exact whatever the matmul's order; review round 1). Verified once: the decode-row plant
  `qk-rope-kv-decode-position` fails it on both formats (with random inputs the KV differed by up to 0.75 in bf16 and
  the output alone would pass, usage 0.24, since only 8 of 64 dims rotate), the clean build passes.
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

## 13. As built (step 2, 2026-10-14)

Commits 35560a8 (`kv_write`), 0cea1e5 (D4, D11), 5156f24 and 48b5534 (D6, D3), 83bd092 (D2 re-derived), fe0fbc1
(the RTX 5090 rows-8 floor), 1d96857 (`GLU_TYPES`) and the review and measurement fixes after them. No core change:
core 4129 code lines, as at step 1. No ABI change: the std.ops minors (`qlinear_glu@1`, `attention.qk_rope_kv@1`,
`Linear.prepare_input` / `use_glu` / `glu`, `fusion_facts`, rope@1's `cos_sin`) are std contracts (`std/contracts.txt`
gains the two ops). Desktops were clock-screened before their records (`bench/clock_check.py`): RTX 5080 p10 / p50
2662 / 2685 MHz, RTX 4090 2670 / 2685 MHz, both steady, no thermal or hardware slowdown. RTX 5090: see §13.6.

### 13.1 What was built

- **`kv_write@1`, model-dtype layout:** a Triton copy of k and v in one launch (`triton_attn` `write_model`; it was two
  index_put kernels). Bitwise: `test_kv_write_is_the_reference_bitwise` now covers `kv.hnd` too.
- **D4, mmq writes bf16** (`vendor.patch`, mmq.cuh): every store goes through `mmq_store` — NVFP4's per-tensor
  `scale2` multiplied in f32 (`ggml_cuda_mul_rn`: `mul.rn.f32` without fast math's flush to zero, as torch's multiply),
  then one rounding to the dst's type. A dense stream-k tile split over blocks is never rounded early: the block that
  finishes a tile it did not start (`stage_end`, exactly the blocks the fixup kernel adds partials for) keeps its part
  in f32 in a second half of the fixup buffer, and the fixup kernel rounds `end + sum` once. The ids (MoE) path is
  unchanged. The workspace formula sizes two fixup tiles per SM for dense calls, one with ids. `QLinear` lost its
  per-call f32 tensor and cast.
- **D11, `scale2` in mmvq above one column** (mmvq.cu): a scale-only fusion runs today's unfused kernel (same nwarps,
  rows per block, small_k, halve_iters), whose epilogue multiplies after the full reduction (`mul_rn`), then rounds
  once; `scale2` is loaded before the K loop (§13.4). The MoE path: a profile of the 30B NVFP4's eager decode steps at
  B4 / B8 (A5000, its rtx-4090 lock) shows no separate multiply (mmvq with ids fuses its expert scales up to 8 tokens:
  `get_mmvq_mmid_max_batch` is 8 for NVFP4 from Turing on), so D11 stays dense, as D11 says.
- **D6, gate/up + SwiGLU:** `qlinear_glu@1` (std.ops; oracle `reference.qlinear_glu`, CPU provider in
  `testing.ref_kernels`). `SwiGLU.finalize_weights` calls `use_glu("swiglu")` on its merged gate/up; `Linear.bind_kernels`
  then selects `qlinear_glu@1` beside `qlinear@2` when the qlinear spec lists it in `config["fused"]` and the chosen glu
  kernel is the same provider's (it reads the parts that provider's `prepare` made). `glu(x)` returns None where the
  kernel does not fuse the call, and SwiGLU composes. ggml's `QLinearGlu` is mmvq's fused gate at one row
  (`holon_ggml_mmvq`'s `gate`), for `GLU_TYPES` = {Q4_K, Q5_K, NVFP4} only (§13.4; Q5_K dropped at the close, §17),
  halves of a multiple of 4 rows.
- **D6, shared q8_1:** `Linear.prepare_input(x)` returns the kernel's handle (ggml: `Prepared(x, key, buffer,
  generation)`): on the mmvq path `x` is quantized once (`quantize_q8_1`, as mmvq does) into `ggml_cuda.q8_1[in=…]`, a
  buffer of its own per `in`; every `prepare_input` into it takes the next generation (`Extension`, by the buffer's
  address), and a call reads the buffer only if the handle's key (path, mode, rows, `in`) is its own and the
  generation is the buffer's last, else it quantizes `x` (mmvq with a pre-quantized src1, `vendor.patch`). Users: GQA's
  unmerged projections, SwiGLU's unmerged gate and up, and GDN's four in-projections (three lines in
  `gdn/units.py`'s forward: the one edit step 2 makes in GDN; see §13.9). Dense linears and other providers take the
  handle's `x`.
- **D3, `attention.qk_rope_kv@1`** (std.mixers.gqa's contract; no oracle of its own, the judge composes the units'
  references in `holon_judge/fused.py`): GQA uses it when `q_norm` and `k_norm` answer the same `fusion_facts()`
  (norm op, eps, offset), `rope` answers (op, rotary_dim, theta, max_positions, and its kernel's `cos_sin` table, now part
  of rope@1's instance contract: FlashInfer's rope names it), and the kv_write spec lists the op in `config["fused"]`
  (kernel props are a closed core vocabulary, so the promise lives in the spec's config; triton_attn lists it for the
  model-dtype and q8_0 layouts, not fp8). Then on every step, prefill included, one Triton launch (`_qk_rope_kv`, one
  program per token and head) norms and rotates q (the gated layout read in place: each head's first D of 2D) and k,
  rounding to the model dtype after the norm and after the rotation, writes k and v (q8_0 through the same
  `_store_q8_0` as `kv_write`), and returns q. The drafters' `write_kv` takes the same path; the model-only build keeps
  the composition.

### 13.2 Checks

- **Bitwise, in the existing L0 loops** (sm_86 locally, sm_89 on the RTX 4090, sm_120a on the RTX 5080; plus the RTX
  5090 at eb9aeaa): `test_mmq_schedules` asserts mmq's bf16 output is its f32 output cast under each schedule (the
  fixup forced at out 300), and with NVFP4's `scale2` (an HF layout, native W4A4 too on sm_120) the f32 output times
  scale2, cast; a mutation that rounds the staged partial before the fixup fails all 20 types. D11: in
  `check_qlinear`, rows 2–8 equal by value the f32 mmvq times scale2 (scale2 not a power of two). Shared q8_1: in
  `check_qlinear`, one `prepare_input` serves the kernel and a Q4_K (or Q6_K) consumer of the same `in` bitwise as their
  own calls, and a handle a later `prepare_input` overwrote is not read (ignoring the generation fails). `kv_write`:
  the model-dtype copy in the kv_write loop. Also, mmq against the step-1 build on sm_86: 81 cases (9 types, out 300 /
  6144 / 12288, rows 9 / 64 / 512) bitwise in bf16 and f32.
- **Bounded L0:** `qlinear_glu` at one row against `silu(gate) · up` of the q8_1-emulated sums (`l0.check`, bf16 budget)
  for the fused types, both FP4 layouts' `scale2`; the other types and halves not of 4-row blocks compose (asserted).
  `test_qk_rope_kv_is_the_composition`: three geometries (head_dim 128 full rotary; 256 with rotary 64 and the gated q
  layout, offset 1; 64), bf16 and fp16, T = 1, 300 (rows across pages, any order) and 0: q' and the model-dtype cache
  within the L0 budget of a norm over head_dim, v copied exactly, no other byte touched, and the q8_0 write bitwise the
  reference write of the kernel's own model-dtype k.
- **L0 runs** (`test_ggml_cuda`, `_moe`, `test_attention_kv_quant`): RTX 5080 322 passed / 1 skipped and RTX 4090
  314 / 9 at 74574e0, then `test_ggml_cuda` again at e14b311: 143 passed (5080), 137 / 6 (4090); RTX 5090 322 / 1 at
  eb9aeaa (`evidence/m07-step2_l0_2026-10-14.json`).
- **memcheck** (`compute-sanitizer`): `memcheck_ggml_cuda` (now also bf16 / f32 outputs at 513 rows, the fused gate,
  prepared inputs) found the fused gate reading past the up half at out 300 (halves of 150 rows: mmvq reads whole
  4-row blocks at small K) — fixed (0308bbc), then 0 errors on sm_86, sm_89 and sm_120a (5080, 5090).
  `memcheck_triton_attn` (now also the model-dtype write and the fused kernel): 0 errors on sm_89 and on sm_120a (the
  5090); on the 5080 it ran out of memory under the sanitizer before reaching them (16 GB; not a finding).
- **L2** (RTX A5000, `holon_judge l2`, against the floor of the path; `--max-seqs 4` for small steps):

  | model | steps | floor | top-1 (floor) | KL p99 (floor) | ΔPPL vs floor | verdict |
  |---|---|---|---|---|---|---|
  | Qwen3-8B Q4_K_M | 1 token (D3 decode rows, glu) | mmvq | 99.06% (99.08%) | 9.7e-3 (1.10e-2) | +0.008% ± 0.072% | PASS |
  | Qwen3-8B Q4_K_M | whole prompts (D3 prefill, D4) | mmq | 98.44% (98.36%) | 3.5e-2 (3.6e-2) | −0.089% ± 0.126% | PASS |
  | Qwen3.8-27B UD-IQ3_S, q8_0 KV | 1 token (D3 gated q, partial rotary, q8_0 1-token rows; shared q8_1) | mmvq, q8_0 KV | 97.92% (98.03%) | 0.330 (0.306) | +0.37% ± 0.39% | PASS |
  | Qwen3.8-27B UD-IQ3_S, q8_0 KV | whole prompts | mmq, q8_0 KV | 97.51% (97.54%) | 0.628 (0.712) | +0.20% ± 0.59% | PASS |
  | Qwen3-8B NVFP4 | 1 token | mmvq (W4A8) | 98.72% (98.78%) | 1.2e-2 (1.4e-2) | −0.041% ± 0.040% | PASS |
  | Qwen3-8B NVFP4 | whole prompts | mmq (W4A8) | 98.74% (98.69%) | 1.3e-2 (1.2e-2) | −0.057% ± 0.060% | PASS |

  The 27B's 1-token run predates `GLU_TYPES` (its i-quant gate/ups were still fused): the shipped code composes them.

### 13.3 Keep or revert (§1.1)

`bench/fusion_ab.py` on one engine per deployment: the arms add the fusions in order (`base`: every step-1 path
restored in-process; then `kv_write`, `qk_rope_kv`, `mmq_bf16`, `scale2`, `glu`, `q8_1`), each arm's saving against
the previous one; six rounds in ABBA order; graphed decode / verify rows by per-step wall over 64 steps (graphs
re-captured per arm), eager prefill by wall (median of 3). The sd is that of the six per-round differences; a row
passes at mean ≥ max(0.5%, 3 sd). Two approximations of the `base` arm: its f32 mmq is the new stream-k path (bitwise
the old one, §13.2) and its crossover is the re-derived table. Deployments: Qwen3-8B Q4_K_M (rtx-5080 lock 7), the
class's 27B plain lock (UD-IQ3_S with q8_0 KV on the 5080, UD-Q4_K_M on the 4090) with MTP and DFlash2 verify steps,
Qwen3-8B NVFP4 (rtx-5080 lock 2; native W4A4 prefill on sm_120), Qwen3-8B bf16 (4090). The best release row per
class, in % of the step (− = faster), with its sd:

| fusion | kind | RTX 5080 | RTX 4090 | RTX 5090 | verdict |
|---|---|---|---|---|---|
| `kv_write` (bf16 copy) | bitwise | 8B Q4_K_M B1 −3.73 ± 0.10 (NVFP4 B1 −3.38) | 8B B1 −2.40 ± 0.04; 8B bf16 B1 −0.81 | 8B B1 −6.37 ± 0.04 (NVFP4 B1 −5.80; bf16 B4 −2.18) | **kept**, every class |
| D3 `qk_rope_kv` (on top of the copy) | bounded | 8B B4 −3.02 ± 0.17; NVFP4 B4 −3.30; 27B B1 −0.51; prefill 512 −0.61 | 8B B4 −2.92 ± 0.03; bf16 B4 −1.04; 27B B1 −0.33 | 8B B4 −5.15 ± 0.34; NVFP4 B4 −5.44; bf16 B4 −2.21; 27B B1 −0.73; DFlash verify −0.72 | **kept**: every class (each saves ≥ 2%) |
| D4 mmq bf16 | bitwise | 8B prefill 512 −3.52 ± 0.10; NVFP4 2048 −5.19; W4A4 −11.02; 27B prefill −3.47, B8 −0.94; MTP B4 −1.15 | 8B prefill −6.48 ± 0.10; NVFP4 −7.19; 27B prefill −7.03; MTP B4 −1.35 | 8B prefill 2048 −4.64 ± 0.12; NVFP4 −5.40; W4A4 −10.18; 27B B8 −1.76; DFlash verify −1.58; MTP B4 −1.75 | **kept**, every class |
| D11 `scale2` in mmvq | bitwise | NVFP4 B4 −4.97 ± 0.04, B8 −3.62 ± 0.08 | NVFP4 B4 −3.37 ± 0.03, B8 −1.94 | NVFP4 B4 −7.24 ± 0.32, B8 −4.91 | **kept**, every class |
| D6 gate/up + SwiGLU (`GLU_TYPES`) | bounded | 8B B1 −0.78 ± 0.07; NVFP4 B1 −0.94 ± 0.03; 27B UD-IQ3_S ±0 | 8B B1 −0.65 ± 0.03; NVFP4 −0.55 ± 0.02; 27B UD-Q4_K_M B1 −0.17 | 8B B1 −1.58 ± 0.04; NVFP4 B1 −2.06; 27B UD-Q4_K_M B1 −0.14 ± 0.17 (noise) | **kept**: every class (8B B1 −0.65 to −1.58%) |
| D6 shared q8_1 | bitwise | 27B B1 −0.65 ± 0.01, B4 −0.62; MTP verify B1 −0.56 | 27B B4 −0.94 ± 0.003, B1 −0.90; MTP / DFlash verify B1 −0.71 | 27B B1 −0.88 ± 0.14; MTP verify B1 −0.49 | **kept**, every class |

- Every other cell is within noise of zero, as expected where a fusion has no work (the copy under q8_0 KV, D11
  without NVFP4, glu above one row, shared q8_1 on the 8B whose projections merge). Records
  `evidence/m07-step2_ab_<class>_2026-10-14.json` (all arms at 74574e0; the NVFP4 deployment again after §13.4's
  fix at e14b311; glu with `GLU_TYPES` at 1d96857).
- Against §11.11 (5080 / 4090): the copy saves 3.7 / 2.4% of 8B B1 (step 0's KV-store share 2.2 / 1.5%: the
  index_put pair cost more wall than its kernel time), D3 2.0 / 1.7% of 8B B1 (bound 1.9 / 1.7%), D4 3.5–5.2 /
  6.1–7.2% of
  prefill (bound 2.4–5.0 / 4.8–6.0%), D11 5.0 / 3.4% of NVFP4 B4 (bound 3.5–4.2 / 3.3–3.6%), glu 0.8 / 0.7% of 8B B1
  (bound 1.6 / 1.1%), shared q8_1 0.65 / 0.9% of 27B B1 (bound 1.5 / 0.96%).

### 13.4 Found on the way

- **D11 slowed 8-row NVFP4 steps by 8.5% on the RTX 5080** as first built (−0.5% at B4), against an expected 3.5–4.2%
  saving: the epilogue loaded `scale2` from global memory per output element, a dependent load on every row's store.
  Loaded once before the K loop (bf30ced), the same step saves 5.0% at B4 and 3.6% at B8 (and 3.4 / 1.9% on the 4090).
  The by-value check covers the move (the same f32 into the same multiply).
- **mmvq's fused gate is not a free win.** `bench/glu_sweep.py` (graph-timed per call, real GGUF rows) against the
  matmul plus FlashInfer's `silu_and_mul`, at the locks' gate/up shapes (8B 4096 × 24576, 27B 5120 × 34816), RTX
  5080 / 4090: Q4_K +2.4 / +2.1% (8B), +0.1 / +2.1% (27B); Q5_K +0.6 / +1.5%; NVFP4 +2.8 / +1.6%; iq4_nl +0.5 / +1.3%;
  iq2_s +0.3 / −2.2%; iq4_xs −2.9 / +1.1%; iq3_s −2.5 / −1.2%; iq3_xxs −8.1 / −2.1%; iq2_xs −5.9 / −4.4%; iq2_xxs
  −8.3 / −4.4%; Q3_K −3.9 / −9.8%. On the RTX 5090: Q4_K +4.2% (8B) and −0.1% (27B), Q5_K −0.3%, NVFP4 +5.8%,
  iq4_nl +0.6%, the other i-quants −1.0 to −5.3%, Q3_K −3.8%. (The records also hold other shapes whose out is twice
  some `in`, e.g. the 27B's
  5120 × 12288; the sweep picked them by a looser rule, since fixed; they are not gate/ups and are not read here.) On
  the 5080's 27B UD-IQ3_S lock the fusion as first built cost 0.47% of the B1 step. Hence `GLU_TYPES` = {Q4_K, Q5_K,
  NVFP4} (Q5_K dropped at the close, §17), chosen on the 5080 and 4090: the types clearly faster on both (iq4_nl,
  +0.5% on the 5080, composes as
  marginal; Q2_K has no gate/up in the locks swept and composes). The 5090's sweep, which came after, keeps them within
  0.4% of the composition at the 27B's shape and faster at the 8B's. With the set, the 27B row is within noise on the
  5080 and the 5090 (−0.14 ± 0.17%), and −0.17% on the 4090. Each sweep is one run without an sd: Q5_K (+0.6 / +1.5 /
  −0.3%) and iq4_nl (+0.5 / +1.3 / +0.6%) are within its noise and could go either way; the set stands on the step-
  level A/B of the 8B rows. Records `evidence/m07-step2_glu-sweep_<class>_2026-10-14.json`.
- **memcheck caught the fused gate reading past the up half** of a gate/up whose halves are not a multiple of 4 rows
  (§13.2). Real models' intermediate sizes are multiples of 8; such halves now compose.
- **The CPU reference provider fuses gate/up at every row count** (`testing.ref_kernels`' `qlinear` lists
  `qlinear_glu@1`), so CPU engine tests run `Linear.glu`; the toy provider (`tests.toy_quant`, one row like ggml)
  covers the composed path and the handles.
- **D11's rounding of subnormals** differs by row count as before: one column keeps llama.cpp's fused epilogue
  (fast-math, flushes), more columns multiply with `mul_rn` (keeps them, as torch's multiply did).

### 13.5 D2 re-derived from the no-cast column

D4 is kept, so `CROSSOVER` is step 1's no-cast table (`evidence/m07-step1_d2-no-cast_<class>`; 83bd092): 25 RTX 5080
entries (iq2_xs 5120x17408 and iq4_xs 5120x12288 move from 7 to 6: row 7 only) and 11 RTX 5090 entries (iq3_s /
iq4_xs 5120x17408 from 7 to 6, and iq4_nl 5120x17408 gains 7 — its 8-row path moves to mmq). The 5080's rows-8 floor
is unchanged (same 8-row paths); the 5090's was re-recorded (`holon_judge golden --q8-1 rows --rows 8 --crossover
rtx-5090:120`, fe0fbc1). `judge/tests/test_gguf.py::test_a_per_shape_floor_routes_as_the_table_does` (CPU) now ties
every committed per-shape floor to the table: it failed on the old 5090 floor. The 8-row L2 on the RTX 5080 at
48ef045 (D3 and D4 in): top-1 97.44% (≥ 97.30%), KL p99 0.592 (floor 0.704), ΔPPL vs floor +0.28% ± 0.40%: PASS
(`evidence/m07-step2_l2-8_rtx-5080_2026-10-14.json`). On the RTX 5090 at eb9aeaa, against the re-recorded floor:
top-1 97.15% (≥ 96.77%), KL p99 0.714 (floor 0.713), ΔPPL vs floor −1.04% ± 0.80%: PASS
(`evidence/m07-step2_l2-8_rtx-5090_2026-10-14.json`). The 5090's guard again (ABBA, the entries against
`--llama-crossover`): tg128 at batch 4 295.6 against 295.6 tok/s (0.0%), at batch 8 513.1 against 496.1 (**+3.4%**),
DFlash2 verify B1 20.16 against 20.89 ms (**−3.5%**). Step 1 had +3.1% and −3.5%
(`evidence/m07-step2_guard_rtx-5090_2026-10-14.json`). The 5080's moved entries change row 7 only, which no guard
row runs.

### 13.6 The RTX 5090

Three desktops failed before a fourth ran it:
- one throttled at its clock screen (SM p10 / p50 1927 / 2340 MHz, not steady) and was released;
- one throttled too (1687 / 2077) and then lost its GPU (`CUDA unknown error`) — no record kept;
- one passed L0, memcheck and its screen, then fell off the bus two rounds into the 8B's A/B (Xid 79).

The fourth (SM p10 / p50 2617 / 2670 MHz, steady, at the power cap) ran everything at eb9aeaa, the final code: the
L0 (322 passed, 1 skipped), both memchecks (0 errors), every A/B arm (§13.3), the glu sweep (§13.4), the 8-row L2 and
the guard (§13.5). The 27B plain lock's rows are noisier on the 5090 than on the other classes (sd up to 1–4% on
B4 and prefill), but every verdict rests on rows with sd ≤ 0.35%.

### 13.7 Plants (verified once, not kept as tests)

| plant | check | prediction | result |
|---|---|---|---|
| `qk-rope-kv-page-slot` (D3: a token opening a page written one slot late) | L2, Qwen3-8B Q4_K_M, whole prompts, mmq floor | FAIL | FAIL: top-1 74.8% (≥ 98.11%), KL p99 6.25; fired on 2520 calls |
| `qk-rope-kv-decode-position` (D3: decode rows rotated one position late) | `bounded`, Qwen3-8B Q4_K_M, `--max-seqs 8` | FAIL | **PASS** (prediction missed): every configuration PASS; fired on 443232 calls |
| `ggml-stale-q8_1` (re-verified after D6: a stale workspace, now also on prepared inputs) | L2, 27B UD-IQ3_S, 1-token steps, q8_0 KV, mmvq floor | FAIL | FAIL: top-1 77.8% (≥ 97.78%), KL p99 9.68; fired 56162 times on plain mmvq calls and 103362 on prepared inputs |

- **The decode-row plant is not caught.** It shifts every decode row's rotation, and `bounded` judges each
  configuration against a baseline that runs the same decode rows. So the bug moves the floor itself: the
  baseline's agreement with the teacher-forced reference falls from 98.66% to 98.33% (± 0.12%), its mean KL rises
  from 0.0020 to 0.0117 (× 5.7; regret 0.0017 → 0.0121), and its KL p50 / p99 rise from 7.8e-5 / 0.024 to 1.6e-4 /
  0.037. Every configuration still sits within the margin of that shifted floor. Two other checks pass with it too:
  - `holon_judge l2 --max-step-tokens 1` (8B: top-1 99.06% and KL p99 9.6e-3, the clean run's; fired 1181952 times).
    Rotation is relative, and in 1-token steps every token, keys included, shifts by the same one position.
  - `bounded --spec-vs-plain` on the 27B UD-IQ3_S with MTP (fired 110298 times). Its checks are one-sided (a
    configuration may only be worse than the floor), and the spec configurations' verify rows are not decode rows:
    only the drafter's 1-token rows carry the bug, while the plain baseline carries it on every decode row. So the
    configurations are cleaner than their floor (KL p99 0.018 against 0.032) and pass by construction.
  No existing check should have caught it: D3's L0 takes positions as given, and the GPU engine test's greedy
  agreement (≥ 75% of 16 tokens) is coarse. An absolute floor at today's margins would not catch it either (98.33%
  clears 98.66% less the bounded margin; both KL quantiles pass). What separates it is the mean KL (0.0020 ± 0.0001
  clean against 0.0117) or the regret, or a GPU check of GQA on T tokens as one prefill against prefill(T − 1) plus
  one decode row. For the lead (§13.12).
- `bounded` without a plant (§6: at `max_seqs` 8 and 1, the baseline at the same value): every configuration PASS,
  at both (8 / 1: baseline top-1 98.66% / 98.63%; prefix on, depth 1: 98.54% / 98.57%).
- Records: `evidence/m07-step2_l2_rtx-a5000_2026-10-14.json` (the L2 runs, the plants and the bounded reports).

### 13.8 The FP4 repack holes (M0.6 follow-up): measured, not fixed

- **The fit, RTX 4090** (the nvidia 30B NVFP4 at its rtx-4090 lock 2's flags, 48ef045): 34192 KV tokens (the lock
  records 34128); at the engine's free-memory read the allocator holds **0.355 GiB** reserved but unallocated
  (0.332 GiB right after the weights load; M0.6's 0.63 GiB was read at another point of the build). Record
  `evidence/m07-step2_fit-30b-nvfp4_rtx-4090_2026-10-14.json`.
- **Where the holes come from** (A5000, the allocator's history after load, by the site that freed the block now
  inside a hole): 192 MiB from the bind of the MoE gate/up stacks (`_hold` releasing a merged stack after `prepare`'s
  repack), 108 + 30 MiB from the merges themselves (`ExpertLinear.merge`, `Linear.merge` releasing their sources),
  53 MiB from attention linears' bind, 52 + 12 + 5 MiB from the repack's chunk transients. Without the merges (a
  probe that refuses them) the holes are 0.08 GiB. Each phase transforms N same-size tensors allocate-then-free, which
  leaves about one tensor's worth unfilled; and torch's expandable segments return only page-aligned free blocks to
  the driver (a probe: a 25 MiB free block between two live ones at an unaligned offset stays mapped after
  `empty_cache`; whole 2 MiB-aligned blocks are released).
- **Two fixes tried, both worse, reverted:** (1) repacking to the host and uploading after the unit releases the
  source (free before allocate, a std `_hold` change): holes 0.552 GiB, KV −2.1K tokens on the A5000; (2) padding
  each repacked buffer to whole 2 MiB pages: holes 0.269 GiB but 0.087 GiB more allocated, net worse.
- **Not fixed.** The remaining hole is about one MoE layer's merge-and-repack (≈ 0.3 GiB, ≈ 3.4K KV tokens at the
  30B's 96 KiB per token). A real fix needs either the merge and the repack in one allocation (the merge handing
  `prepare` its sources) or free-before-allocate in std's merges as well — std and kernel contract changes beyond
  "memory only" — or an allocator setting (out of M0.7's scope, §4). For the lead (§13.12).

### 13.9 Deviations and decisions

- **How a unit learns a fused kernel exists.** The design says "a provider supports those facts", but `BuildCtx` has no
  optional selection (`select_kernel` raises when nothing serves), and kernel props are a closed core vocabulary
  (`KERNEL_PROPS`). The base op's spec says it instead: `config["fused"]` lists the fused ops its provider serves for
  its facts (qlinear@2 → `qlinear_glu@1`; kv_write@1 → `attention.qk_rope_kv@1`, model-dtype and q8_0 layouts). That
  is std's documented convention (`std/holon_std/ops/kernels.py`, `mixers/gqa/kernels.py`), with no core change.
- **D3's reference** lives in the judge (`holon_judge/fused.py`), composed from the units' oracles. std.mixers.gqa
  registers no oracle for the op. There is no CPU provider: the reference rope kernel names no table, so CPU builds
  keep the composition. fp8 KV composes too.
- **glu is kept per type as well as per class.** `GLU_TYPES` is one set for every class: the types whose fused gate
  was faster on every class swept (§13.4). For a merged gate/up of another type the provider still binds a
  `qlinear_glu@1` site that never fuses: the lock records it, and the call composes.
- **The GDN mixer is edited** (5156f24, three lines in `forward`) although step 3 works in `gdn/units.py`: D6's shared
  quantization pays on the 27B through GDN's four in-projections. The diff is separate and small, for step 3 to
  rebase over.
- **D11's MoE clause** does not apply: the 30B NVFP4's B4 / B8 decode runs no separate scale multiply (§13.1).
- **The FP4 repack holes are not fixed** (§13.8).
- **Step 2's measurement method** is in-process A/B on one engine (`bench/fusion_ab.py`), not separate builds per
  commit.
  Each fusion is switched off by restoring the code it replaced, with the two approximations named in §13.3.
- **`bench/`** gains `fusion_ab.py` and `glu_sweep.py`. `qmm_sweep.py`'s "mmq_cast" variant now times D4's path
  (mmq writes bf16).
- **Tests added:**
  - `test_qk_rope_kv_is_the_composition` (D3's L0);
  - the toy proof in `test_quantized_units`;
  - `test_a_per_shape_floor_routes_as_the_table_does` (CPU; ≈ 19 s when the 27B files are cached, else skipped).
  The rest are assertions inside existing loops.

### 13.10 Review

One reviewer (correctness, architecture, numerics / tests) on the aggregate change.
- **Round 1:** no blocker, two majors, three minors, five nits.
  - The majors: the 5090 floor no longer matched the re-derived table (fixed: re-recorded, plus the CPU test that
    failed on the old floor); `qlinear_glu`'s support was narrower than `config["fused"]` promised, so a merged
    gate/up with halves not of 4-row blocks would fail to build (fixed: any even `out` is served, and those halves
    compose).
  - The minors: GQA at T = 0; another provider's handle reaching ggml; a glu kernel from another provider than the
    qlinear's. All fixed.
  - The nits were fixed or are recorded in §13.3 / §13.4.
- **Round 2:** no blocker or major; one minor (evidence records for the round's new claims: written,
  `evidence/m07-step2_*`) and four nits: the glu provider checked before its activation-mode check (fixed); the
  per-type glu sites (recorded in §13.9); `GLU_TYPES` per class (§13.4); the docs (this section).
- **Round 3** (§13, the evidence, the engine_shapes cache): no blocker or major; three minors.
  - The §13.7 metrics and mechanism, now corrected: mean KL against regret, the one-sided spec-vs-plain check, the
    fired counts and the plant's 1-token L2 added to the record, and §6 amended.
  - `glu_sweep`'s gate/up rule was too loose; fixed (eb9aeaa), and §13.4 reads only the real gate/up shapes.
  - The lead list: the 5090 items, the corrected judge-gap options, lock churn.
  - Nits: the cache key (documented); the committed placeholders (filled).

### 13.11 Suite

Full GPU suite at 9de4222 (`pytest core std plugins judge testing tools`, one RTX A5000 while the other ran a
bounded check): 1200 passed, 65 skipped, 18 min 46 s. Step 1 had 1185 / 65 in 16 min 29 s on two GPUs. The added
tests are the D3 L0 (6 cases, ≈ 13 s), the CPU per-shape floor test (≈ 19 s) and the toy proof; the existing loops
gained assertions. `scripts/ci/run_all.sh main` is green.

### 13.12 For the lead

- **FP4 repack holes** (§13.8): not fixed, measured. The fix needs a std / kernel contract change (merge and repack in
  one allocation) or an allocator setting. Decide whether to take it into M0.7 or leave it as the follow-up it
  remains.
- **A judge gap: the decode-row RoPE plant passes** (§13.7). §6 predicted that `bounded` catches it. It does not:
  the check is relative to a baseline that carries the same bug, and the small-step L2 is blind to a uniform shift.
  An absolute floor at today's margins would not catch it either. What would: a gate on `bounded`'s mean KL or regret
  against a recorded clean value (0.0020 → 0.0117 here), or a GPU check of GQA prefilling T tokens against
  prefill(T − 1) plus one decode row. Both are judge changes, not taken here.
- **Lock churn at the close:** every `qlinear@2` and `kv_write@1` site's recorded config gains `fused`, every merged
  gate/up gains a `qlinear_glu@1` site (also for types that compose: §13.9; a cross-provider glu, which no installed
  plugin produces, would also leave a site that never runs), and every GQA layer an `attention.qk_rope_kv@1` site.
  The single propagation touches every GGUF and HF lock.
- **GDN** (§13.9): step 2's three-line edit to `gdn/units.py` needs a rebase with step 3's D5.
- **The RTX 5090 desktops:** three of the four 5090 desktops tried failed (§13.6); every 5090 verdict comes from the
  fourth, at the final code.
- **glu is small** (0.55–1.0% of the 8B's B1 step, nothing on the 27B) and type-dependent. It is kept by §1.1's rule.
  Dropping it would remove `qlinear_glu@1`, `Linear.use_glu` / `glu` and `QLinearGlu`.
- **A latent read in plain mmvq:** at small K, llama.cpp's one-column kernel reads whole blocks of up to 4 rows, so a
  quantized linear whose `out` is not a multiple of 4 reads past its weights. llama.cpp's own buffers are padded;
  Holon's are not. No supported model has such a shape. The fused gate's case was fixed (§13.2); the plain qlinear
  is unchanged and its L0 uses out 300.
- **D9** stays on step 1's with-cast table (step 1's review). It is not recomputed on the no-cast table.
- **UX follow-up** (after the UX branch merges): the manifest `[options]` declarations for `std.mixers.gqa`
  (`kv.format`), `std.ops` and `std.moe` (`quant.act`). Step 2 adds no option keys.


## 14. As built (step 3, 2026-10-14): D5

Commits 016903f (the glue once per step, as torch ops in std), bcdfbc7 (`gdn_rewind@1`, one launch), 12bb917 (review
nits), eb1756b (measured pools; reverted) and 8f3827e (expandable segments from the session's start, §13.5), and this
section's. Desktops screened first (`bench/clock_check.py`, 60 s of 27B decode at batch 4): RTX 5080
SM clock p10 / p50 2677 / 2707 MHz, RTX 4090 2715 / 2715, RTX 5090 2497 / 2587 (0.7% of samples with a slowdown
reason); the 5080 and the 5090 at the power cap in every sample, as in §11.1; all steady. (The torch composition's
session ran on another 5090 desktop at p50 2647 MHz; each saving is a ratio within its own session.)

### 14.1 What changed

- **The replay buffers have a layer axis:** one static buffer per name (`gdn.replay.{x,a,b,window}`, [layers, …]),
  each GDN layer's slice at its index in the state group; the same total as the per-layer buffers they replace.
- **`gdn_rewind@1`** (std.mixers.gdn contract and oracle; a new contract, so `std/contracts.txt` gains a line and no
  version moves): after a verify step, for every layer of the group at once, the saved gates a and b ← −1e4 past each
  row's 1 + n accepted tokens and the conv state at the accepted end ← the last K − 1 inputs of window ‖ saved x,
  written into the pool. The planner selects it at the group's site for speculative deployments (a lock site
  `recurrent/gdn_rewind@1` beside the delta rule's; no lock under `plans/` drafts today, so none changes before the
  close's propagation). Providers: `kernels.fla`'s `rewind.py` (ours, Triton, bf16: one launch, a program per
  (layer, row, 256 channels), reading K − 1 tokens per row and writing the pool directly) and `testing.ref_kernels`
  (the oracle).
- **`GDN.rewind`:** the group's first layer (index 0; `rewind_modules` visits a model's layers in order, as the design
  says, no counter) calls `gdn_rewind`; then every layer runs `causal_conv1d` over its saved x with the saved window
  buffer as the state (rows `arange(R)`, a static buffer), so the provider dirties only that buffer, and the delta rule
  as before. The provider calls are today's; results are bitwise today's for every unit but the padding sink.
- **Deviation, then back to the design:** the first build (016903f) ran the per-step glue as ≈ 11 torch ops in std,
  no new contract. The review found that it held every layer's window ‖ x at once (n · R · C · (2(K − 1) + W) · 2
  bytes: ≈ 79 MB at 8 rows and k 3 on the 27B, ≈ 0.9 GB at 64 rows and k 7, charged by the startup profile against the
  state pools) and read all of saved x; bcdfbc7 is the design's one launch. Both measure alike (§13.2), the kernel
  0–0.4 points better.

### 14.2 Keep-or-revert (§1.1): kept, every class

Per-step graphed wall of the verify step (`bench/step_profile.py`, 32 steps per window, the lock's flags; 27B
UD-IQ3_S with q8_0 KV on the 5080, UD-Q4_K_M with bf16 KV on the 4090 and the 5090, the 4090's lock borrowed): eight
rounds per desktop, A B B A B A A B, A = std at the base 1b9d38e (by `PYTHONPATH`), B = 12bb917, four of each. The bar
is max(0.5%, 3 sd), sd the larger arm's over its four rounds. Records `evidence/m07-step3_d5_<class>_2026-10-14`
(with the torch composition's same rounds at 016903f).

| class | row | base ms | D5 ms | saving | 3 sd | torch composition | rewind kernels (eager) | rewind glue ms (eager) |
|---|---|---|---|---|---|---|---|---|
| RTX 5080 | MTP verify B1 | 26.914 | 25.979 | **3.5%** | 0.26% | 3.3% | 960 → 289 | 1.16 → 0.22 |
| RTX 5080 | MTP verify B4 | 35.472 | 34.504 | **2.7%** | 0.14% | 2.3% | 960 → 289 | 1.30 → 0.23 |
| RTX 5080 | DFlash2 verify B1 | 32.661 | 31.715 | **2.9%** | 0.07% | 2.7% | 960 → 289 | 1.19 → 0.22 |
| RTX 4090 | MTP verify B1 | 30.878 | 29.975 | **2.9%** | 0.34% | 2.8% | 960 → 289 | 1.29 → 0.25 |
| RTX 4090 | MTP verify B4 | 40.284 | 39.432 | **2.1%** | 0.32% | 2.1% | 960 → 289 | 1.40 → 0.27 |
| RTX 4090 | DFlash2 verify B1 | 31.459 | 30.586 | **2.8%** | 0.52% | 2.7% | 960 → 289 | 1.32 → 0.25 |
| RTX 5090 | MTP verify B1 | 18.848 | 17.912 | **5.0%** | 0.19% | 4.9% | 960 → 289 | 1.15 → 0.22 |
| RTX 5090 | MTP verify B4 | 23.310 | 22.295 | **4.4%** | 0.15% | 4.0% | 960 → 289 | 1.23 → 0.23 |
| RTX 5090 | DFlash2 verify B1 | 20.872 | 19.946 | **4.4%** | 0.08% | 4.4% | 960 → 289 | 1.16 → 0.22 |

- **Every row passes its bar;** against the design's upper bounds (§5, §11.11: −3.4–4.2% on the 5080, −3.4–4.1% on
  the 4090, −5.3–6.1% on the 5090), measured 2.7–3.5%, 2.1–2.9% and 4.4–5.0%: the bounds took the fused launch to cost
  nothing, and the rewind keeps the provider's own copies (below).
- **The rewind's 289 kernels** are `gdn_rewind` and, per layer, the conv (2), the delta rule (1) and 3 copies: the
  vendored `fused_sigmoid_gating_delta_rule_update` wrapper copies q, k and v (`.contiguous()` of split views of the
  conv's output, contiguous only at one token) on every call of more than one token — the rewind, the verify trunk
  and decode above one row. The rewind's 0.22–0.27 ms of glue left is those copies (0.21–0.25 ms) and the fused
  launch itself (4–18 µs; `step_profile` leaves `_rewind` unclassified). Outside D5 (the provider calls stay as
  locked); recorded for the lead (§13.6).

### 14.3 Judge

- **L1.5, bitwise, eager and graphed** (`plugins/tests/test_gdn_rewind.py::test_rewind_is_the_decode_steps_bitwise`):
  every GDN layer of a tiny Qwen3.5 runs a verify step, then the model's rewind (`rewind_modules`), n accepted of 3 for
  n < k and n = k, rows on two slots and a third untouched. CPU (reference kernels): every layer's conv window and
  delta-rule state bitwise 1 + n one-token decode calls. GPU (64-wide heads on `kernels.fla` / `gdn_triton`, bf16):
  conv windows bitwise one-token calls, the delta rule bitwise one per-token call over the accepted tokens (m04-spec
  §1.3's fallback), and the verify step plus rewind captured into a CUDA graph at another acceptance and replayed:
  bitwise the eager run on every unit but the sink. It checked the first layer alone before; mutations each fail it
  (windows written for layer 0 only, gates masked for layer 0 only, the replay conv on the pool, no trigger); the
  base passes it.
- **L0** (`test_gdn_kernels.py::test_rewind_glue_is_its_oracle`): the Triton kernel bitwise its oracle on three
  layers, rows on slot 0, the last unit and another, accepting 0, some and width − 1 drafts, at widths 4 and 8, on the
  27B's shape (C 10240, Hv 48) and a small one; the oracle leaves every other unit, row and gate as it was.
- **Bounded spec vs plain** (RTX A5000, 12bb917; the 27B UD-IQ3_S with q8_0 KV, `--max-seqs 8`, M0.4's
  configurations; the judge's default 48 sessions × 4 turns, 24,576 scored positions per configuration, where M0.4
  ran 8 sessions). Top-1 of the speculating configurations (prefix on at depth 1; depth 0) against the plain baseline's
  99.036%, and the plants' fired counts (verify steps whose first GDN layer reached the planted path). Eager: every
  verify step of both configurations' logprob runs (logprob steps are not graphed), plus each graph's warm-up and the
  startup profiles; captured: the verify graphs of both engines, whose replays in the plain runs are not counted — the
  plain runs' top-1 and the logprob runs' KL both fail. Records
  `judge/results/bounded-12bb917-qwen3.8-27b-ud-iq3_s-spec-*`.

  | drafter | clean | `replay-off-by-one` | `verify-writes-state` |
  |---|---|---|---|
  | MTP (k 1:3, 3:2) | 99.129% / 99.162%: **PASS** | 75.63% / 77.45%: **FAIL**; fired 3014 eager, 60 captured | 95.56% / 95.55%: **FAIL**; fired 2528 eager, 60 captured |
  | DFlash2 Q4_K_M (k 7) | 99.129% / 99.137%: **PASS** | 82.30% / 82.59%: **FAIL**; fired 2321 eager, 30 captured | 89.51% / 89.27%: **FAIL**; fired 1908 eager, 30 captured |

  Every plant fails on top-1, confident top-1 and both KL percentiles; the clean runs pass every check.

### 14.4 Review

One reviewer on the aggregate change. Round 1 (016903f): no blocker; one major — the torch composition's transient
(§13.1) — fixed by the one-launch op; two minors fixed (the plants counted every layer and the startup profile without
saying so: they now count the group's first layer per verify step, profile and warm-up included; `GDN.rewind` names
its dependence on `rewind_modules`' order — the core's docstring is left alone, step 3 touches no core); nits fixed
(the shared `arange` written once, explicit layer counts in the pool views). Round 2 (bcdfbc7, with the kernel run
bitwise against the oracle under Triton's interpreter over f32 / f16 / bf16, widths 2–8, K 2–5 and channel counts off
the block): no blocker, major or minor; nits fixed in 12bb917 (a unit shared by padding rows is unspecified in the
contract; the provider serves bf16, its L0's dtype). Round 3 (this section and the records, every summary recomputed
from its runs): no blocker or major; two minors fixed (the design's predictions misquoted; one 3 sd cell) and three
nits (the glue left split into copies and launch; the screen's power-cap share; the 5090 sessions' clocks).
Round 4 (the bounded table, its fired counts, and eb1756b): no blocker or major; two minors fixed (one top-1
rounding; the fired counts' reading) and nits; its residual risk (small state pools in a cached segment) then
happened, and 8f3827e takes its suggested fix. Round 5 (8f3827e and the §13 edits): no blocker, major or minor;
nits (line wrapping, eb1756b marked reverted, §13.5 names the allocator fix). Round 6 (the suite line, against its
three logs): one minor fixed (the new cases are the 3 GPU rewind and 4 L0 cases, +7) and two nits; nothing open.

### 14.5 Suite

Full GPU suite at 8f3827e (two RTX A5000s): `pytest core std plugins judge testing tools` 1192 passed, 65 skipped in
18 min 13 s (step 1: 1185 / 65, 16 min 29 s; the new cases: 3 GPU rewind cases and 4 L0 cases, +7; the 3 CPU rewind
cases now check every layer). The first two full runs failed the 3 GPU rewind cases: the tiny engine's small state
pools landed in a segment earlier tests had cached before the engine's in-process allocator setting (with fixed
pools, then with measured ones in eb1756b); 8f3827e sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in
`plugins/tests/conftest.py` at collection, as launchers do. `scripts/ci/run_all.sh main` green.

### 14.6 For the lead

- **D5 is kept on every class** (2.1–3.5% of the 5080's and 4090's verify steps, 4.4–5.0% of the 5090's).
- **The vendored delta-rule wrapper's q / k / v copies** (§13.2): measured in the rewind, 0.21–0.25 ms per verify step
  on the 27B (≈ 1% of the step); the verify trunk's and decode's calls above one row pay likewise. A candidate for
  step 2's or a later fusion pass, which would touch `kernels.fla`'s `vendor.patch` (strided q / k / v in the fused
  recurrent kernel).
- **A new lock site** `recurrent/gdn_rewind@1` (speculative builds only) is recorded at the close's propagation.
- **Files outside the lead's list:** `testing/holon_testing/ref_kernels` (the CPU provider every op needs),
  `testing/holon_testing/qwen3_5_files.py` (`tiny_config`'s head size, for the GPU case), `plugins/tests/conftest.py`
  (expandable allocator segments from the session's start, §13.5) and `judge/holon_judge/planted.py` (the two plants'
  fired counts).


## 15. As built (step 4, 2026-10-15): mixed steps, option declarations, the decode-row check

Commits c793674 (the UX follow-up: plugin options declared), fc8b1d8 (the decode-row check), aec742f (mixed steps),
cd6fb5c (`bench/mixed_step.py`, the backends' counters), 870ae2c (review round 1), f567fe8 (`bench/mixed_bars.py`) and
this section's. Desktops screened first (`bench/clock_check.py`, 60 s of 27B decode at batch 4): RTX 5080 SM clock
p10 / p50 2737 / 2737 MHz, RTX 4090 2670 / 2685, RTX 5090 2565 / 2655; no thermal or hardware slowdown; the 5080 and
the 5090 at the power cap in every sample, as in §11.1; all steady. A first RTX 5090 desktop (screened steady, p10 /
p50 2467 / 2542) fell off the bus (Xid 79) in round 2 of the plain run: released, no record kept; a second desktop
ran everything. Its screen after the runs read p10 / p50 1927 / 2512 MHz (power cap, no thermal slowdown: not
steady); in its records the long prompt's per-round throughput spreads up to 2.2% (today's arm; 1.6% at 512), the
agent trace's (paced by arrivals) 0.1%; the cap-512 verdict holds with that margin (paired TTFT 1.02, long TTFT p99
+15.8%). The 5080 (2737 / 2737) and the 4090 (2670 / 2670) were steady after their runs too.

Every record ran at cd6fb5c, before review round 1's fixes (870ae2c), which change no measured path: the minimum
chunk was unreachable under the build check, and the alternation's clamp binds only when C exceeds `max_step_tokens`
(the locks' 2048 against 512).

### 15.1 What was built

- **Plugin options (the UX step's follow-up, release §14.14).** `std.mixers.gqa` declares `kv.format` (choices
  `q8_0`, `fp8`; default the model dtype) and `std.ops` `quant.act` (`a4`, `a8`; default `a8`; `std.moe` already
  did). No other key an engine-side plugin reads was undeclared, so `holon serve` warns on none. `check_locks.py`
  fails a committed lock that sets an option no installed plugin declares, or a value outside its declarers'
  `choices` (it failed on the rtx-5080 27B locks' `kv.format` before). ABI unchanged.
- **The scheduler** (`sched.py`, §3.1): with `chunk_with_decode` C > 0 and no drafter, the decode rows are reserved
  first (they may preempt lower-ranked requests, as today's decode step does), then prefill chunks of up to
  min(C, `max_step_tokens` − decode rows) tokens; `BatchPlan.n_decode` is the count of leading decode rows, set only
  when the step also has a chunk (a step whose chunks find nothing is today's graphed decode step). A step with
  nothing to decode prefills up to `max_step_tokens`, as today. With a drafter: a prefill step of min(C,
  `max_step_tokens`) tokens alternates with a decode / verify step while a running request could decode, else the
  step takes the kind that can (a prefill step with no decoder waiting takes the full budget).
- **The drafter predicate** (§12.8): the engine builds a drafter when `max_draft_tokens` > 0 and a draft policy is
  installed; units see `max_draft_tokens` 0 otherwise (`BuildCtx.options`), so the scheduler, std's activation-mode
  refusal and GDN's replay buffers share the engine's predicate.
- **Build checks:** with mixing, `max_step_tokens` ≥ `max_seqs` + C (a `ValueError` naming both), and every planner
  must have `mixes_row_kinds` (a `PluginError` naming the groups); the core's default planner has it (kernels read
  `Pages` alike in every step).
- **The executor:** a step with `n_decode` > 0 is never graphed; the worst-step profile runs, when steps mix,
  `max_seqs` − 1 decode rows beside a C-token chunk, sampled with the chunk's activations alive. The engine's
  `stats()` counts `mixed_steps`.
- **GQA** (`std.mixers.gqa`): its planner always mixes. A mixed step's decode rows go through the prefill kernel in
  one call when the prefill kernel's spec has the `mixes_row_kinds` prop (FlashInfer's prefill, which splits a long
  row's KV), else (triton_attn's q8_0 prefill, one program per row and head over the whole KV) through the decode
  kernel, beside the prefill kernel's call on the chunks (`_Split`, the rows sliced by `_rows`). Unmixed steps keep
  today's paths.
- **GDN** (`std.mixers.gdn`): a mixed step's delta rule is planned per row kind — the decode rows as a decode step
  (the recurrent kernel), the chunks as a prefill step (chunked), with their in-step checkpoints' rows re-based —
  and run as two calls; the convolution runs over all rows at once, as before.
- **The judge:** `holon_judge bounded --mixed C` judges `chunk_with_decode` C (prefix off, depth 0; and prefix on,
  depth 1) against today's schedule; a configuration that set C but mixed no step fails. **The bench:**
  `bench/mixed_step.py` (a mixed step's wall by GQA's path), `bench/mixed_bars.py` (§1.3's bars from a `run.py`
  record); the in-process and HTTP backends report `mixed_steps` and `preemptions`.

Core: 4366 code lines (4321 at the step's base; +45). Docs: design §7.1 and §4.2, plan §10.

### 15.2 Checks

- **Toy proof, CPU** (`core/tests`): the scheduler's liveness test runs three modes (today, mixed, alternating) over
  40 random workloads each and checks every step's shape — decode rows first, chunks within C, never mixed with a
  drafter, a capped prefill step followed by a decode step whenever a request could decode, every step within
  `max_step_tokens` (it failed on review round 1's alternation bug); mutations that put chunks first, ignore the cap
  or never alternate each fail it. `test_mixed_steps_need_room_mixing_planners_and_no_drafter`: the build's two
  refusals (a toy planner without `mixes_row_kinds`), and drafts without a draft policy mix (`max_draft_tokens` 0
  for the units).
- **Outputs equal to today's, CPU** (`plugins/tests/test_qwen3_5_engine.py::test_mixed_steps_compute_what_todays_steps_do`):
  a tiny Qwen3.5 (GQA and GDN, recurrent checkpoints), six requests — chunked prompts beside decoding ones, two
  constrained by `testing.toy_mask`, one sampled — with prefix hits on a second pass (roomy) or preemption: tokens
  equal to today's and logprobs within 1e-5, for C 8 (GQA's two calls) and 72 (one call; a chunk then holds an
  in-step checkpoint). Mutations that drop GDN's row re-base, the chunk call's snapshot, or GQA's `query_start` re-base
  each fail it.
- **GPU bounded, mixed vs today** (RTX A5000, `--mixed 256 --max-seqs 8`, 48 sessions × 4 turns; records
  `judge/results/bounded-cd6fb5c-*-mixed-256-rtx-a5000.json`):

  | model (path) | baseline top-1 | mixed 256 | mixed 256, prefix on, depth 1 | mixed steps | verdict |
  |---|---|---|---|---|---|
  | Qwen3.8-27B UD-IQ3_S, q8_0 KV (GQA two calls; GDN per row kind) | 99.284% | 99.190% | 99.268% | 648 / 568 | PASS |
  | Qwen3-8B Q4_K_M, bf16 KV (GQA one call, FlashInfer) | 98.657% | 98.454% | 98.433% | 648 / 18 | PASS |

  Every KL quantile passes too. The 8B's prefix-on, depth-1 configuration mixed only 18 steps (its turns hit the
  cache), so FlashInfer's one-call path with prefix hits at depth 1 is thinly covered on the GPU; the CPU test covers
  it for correctness. These runs size the pools by `--memory-fraction 0.9`, so they ran the mixed
  worst-step profile; so did every desktop run below.
- **The decode-row check (B)** is §6's last item: the decode-position plant fails it on both KV formats.

### 15.3 GQA's decode rows on q8_0 KV (§10 correctness m3): decided by measurement

`bench/mixed_step.py` on each class's plain 27B model with its lock's `[engine]` values as flags (no kernel pins): three requests decode while an 8192-token prompt prefills in
512-token chunks beside them (16 mixed steps), or, "long", decoders at 8K / 16K / 24K beside a 4096-token prompt;
each mixed step timed synchronously; four rounds per arm, ABBA, the arms flipping the planners' `prefill_mixes`.
Records `evidence/m07-step4_mixed-step*_<class>_2026-10-15.json`.

| class (KV) | contexts 2K–6K: one call / two calls (ms) | contexts 8K–24K: one call / two calls (ms) | path kept |
|---|---|---|---|
| RTX 5080 (q8_0, triton_attn) | 234.6 / 234.9 | 256.9 / **228.8** (−10.9%) | two calls (no prop) |
| RTX 4090 (bf16, FlashInfer) | **161.4** / 163.0 | **160.3** / 163.3 | one call (prop) |
| RTX 5090 (bf16, FlashInfer) | 126.4 / 127.0 | **125.0** / 127.1 | one call (prop) |

On q8_0 the decode rows cost nothing extra in one call while their contexts are short, and 28 ms of a 229 ms step
when they are long (one program walks each row's whole KV); FlashInfer's prefill serves them as well as or up to 2%
better than two calls (the 5090's short case, 0.5%, is within its sd). In two calls the decode rows' attention is the decode kernel's, as in an unmixed decode step.

### 15.4 §1.3's bars on the plain locks (cap per class)

`bench/run.py --lock <the class's plain 27B lock> --backends holon holon@m256 holon@m512 --arm m256='--chunk-with-decode
256' --arm m512='--chunk-with-decode 512' --rounds 4 --traces agent long_prompt` (each arm four rounds, ABBA; the
27B UD-IQ3_S with q8_0 KV on the 5080, the 4090's UD-Q4_K_M with bf16 KV on the 4090 and the 5090), then
`bench/mixed_bars.py`. The paired TTFT bar is 1 + max(15%, 2 sd of today's per-round TTFT medians) — 1.15, and 1.18
on the 4090; throughput's max(5%, 2 sd) is 5% everywhere. Records `evidence/bench_m07-step4-plain_<class>_*` and
`m07-step4_bars_<class>_2026-10-15.json`.

| class | cap | agent: max gap p99 / max (×) | paired TTFT | agent tok/s | long prompt: gap max (×) | long TTFT p99 | mixed steps | verdict |
|---|---|---|---|---|---|---|---|---|
| RTX 5080 | today | 1793 / 1794 ms | — | 71.0 | 13650 ms | 14332 ms | 0 | |
| RTX 5080 | 256 | 139 / 292 (12.9× / 6.1×) | **1.298** | 69.0 (−2.8%) | 204 (67×) | **+34.6%** (tok/s −6.5%) | 3017 | fails |
| RTX 5080 | **512** | 240 / 246 (7.5× / 7.3×) | 1.103 | 70.8 (−0.3%) | 338 (40×) | +14.3% | 1560 | **passes** |
| RTX 4090 | today | 2144 / 2146 ms | — | 70.3 | 8712 ms | 9182 ms | 0 | |
| RTX 4090 | 256 | 107 / 403 (20× / 5.3×) | **1.196** | 69.3 (−1.4%) | 112 (78×) | **+26.9%** | 3083 | fails |
| RTX 4090 | **512** | 165 / 165 (13× / 13×) | 1.036 | 70.3 (0.0%) | 204 (43×) | +14.5% | 1620 | **passes** |
| RTX 5090 | today | 1217 / 1218 ms | — | 74.0 | 7166 ms | 7528 ms | 0 | |
| RTX 5090 | 256 | 78 / 80 (16× / 15×) | **1.203** | 73.8 (−0.3%) | 93 (77×) | **+28.0%** | 2155 | fails |
| RTX 5090 | **512** | 134 / 327 (9.1× / 3.7×) | 1.021 | 73.9 (−0.1%) | 174 (41×) | +15.8% | 1148 | **passes** |

- **Cap 512 on every class** (the 4090 as predicted, §11.8; the 5090's undecided 256 fails; the 5080 too, where the
  simulator passed it). 256 fails the paired TTFT everywhere and the long prompt's TTFT p99 everywhere: a prompt
  takes twice the mixed steps, each paying the eager step's fixed cost (§11.8's host floor; not measured here).
- **Against the simulator** (§11.8): the gaps land 3–9% above its predictions at 512 (p99 240 against 232 ms on the
  5080, 165 against 151 on the 4090, 134 against 123 on the 5090), and the costs are larger: paired TTFT 1.10 / 1.04 / 1.02 at 512 against
  1.00 / 1.01 / 1.00, and the long prompt's TTFT p99 +14–16% against +2.0 / +0.7 / +4.6%. The simulator charged a
  mixed step the prefill of chunk + decode rows; the measured step also pays eager launches and the decode rows'
  attention, which it under-costed (its own note).
- TPOT p99 (recorded, not gated) at 512: 70 against 98 ms on the 5080, 60 against 70 on the 4090, 33 against 32 on
  the 5090.
- At 512 the agent trace's throughput is within 0.3% of today's on every class; the long prompt's is −2.1% on the
  5080 and within 0.1% elsewhere.

### 15.5 Speculative deployments: alternation (recorded)

The same runs with MTP over each plain lock (`--max-draft-tokens 3`, drafter `drafters.mtp`, `fixed_k`, greedy
verifier, `spec.k=1:3,3:2`), two rounds per arm, ABBA; records `evidence/bench_m07-step4-mtp_<class>_*`. Steps never
mix: a prefill step capped at C alternates with a verify step.

| class | C | agent: max gap p99 / max (ms) | paired TTFT | long prompt: gap max | long TTFT p99 | long prompt tok/s |
|---|---|---|---|---|---|---|
| RTX 5080 | today | 1908 / 2060 | — | 14061 ms | 14748 ms | 110.7 |
| RTX 5080 | 256 | 767 / 777 | 1.233 | 223 ms | +44.3% | 101.4 |
| RTX 5080 | 512 | 799 / 799 (2.4× / 2.6×) | 1.002 | 360 ms (39×) | +18.6% | 107.7 |
| RTX 4090 | today | 1683 / 1916 | — | 8900 ms | 9394 ms | 125.3 |
| RTX 4090 | 256 | 506 / 566 | 1.274 | 144 ms | +63.9% | 118.2 |
| RTX 4090 | 512 | 342 / 554 (4.9× / 3.5×) | 1.007 | 227 ms (39×) | +25.8% | 124.4 |
| RTX 5090 | today | 880 / 1061 | — | 7110 ms | 7461 ms | 198.4 |
| RTX 5090 | 256 | 201 / 301 | 1.019 | 106 ms | +46.8% | 186.4 |
| RTX 5090 | 512 | 369 / 386 (2.4× / 2.7×) | 1.001 | 182 ms (39×) | +22.7% | 192.7 |

- Alternation brings the long prompt's stall to one capped step (39× at 512) but the agent trace's gap only 2.4–4.9×:
  a verify step waits for a whole capped prefill step, and decoders still wait behind it every other step.
- The long prompt's TTFT p99 rises 19–26% at 512 (44–64% at 256): every other step is a verify step. Its throughput
  falls 0.7–2.9% at 512. §5 predicted that alternation helps
  less and costs long-prompt TTFT; mixing verify rows (§9) is the follow-up that would close it.

### 15.6 Deviations and decisions

- **The minimum chunk is gone.** §3.1's "at least min(64, `chunk_with_decode`)" can never bind: the build's
  `max_step_tokens` ≥ `max_seqs` + C leaves every mixed step room for the whole C (the review found the clamp dead;
  the simulator's rule is the same). The chunk is min(C, `max_step_tokens` − decode rows) = C.
- **GQA's planner always declares `mixes_row_kinds`:** its two-call path serves any decode and prefill kernels, so
  the kernel prop chooses one call or two rather than gating the build; std's planners never refuse. The refusal is
  the core's, for planners that lack the attribute (proved with a toy planner).
- **`max_draft_tokens` in `BuildCtx.options` is the engine's effective value** (0 without a drafter); the lock and
  `EngineConfig` keep the configured one.
- **The alternation's prefill step without waiting decoders takes the full budget**, as §3.1's "else take the
  schedulable kind" reads; with a drafter its budget is min(C, `max_step_tokens`) (review round 1's fix).
- **The A/A, from the records** (§1.3, §7 name one; no separate arm ran): today's arm against itself, round against
  round, per (session, turn), each of the 6 pairs of its 4 rounds once (`bench/mixed_bars.py --aa`;
  `evidence/m07-close_aa_2026-10-15.json`). The per-pair medians lie within 0.9978–1.0010 on the agent trace (sd 0.0003
  / 0.0002 / 0.0015 on the 5080 / 4090 / 5090) and 0.9924–1.0073 on the long prompt (sd 0.0004 / 0.0007 / 0.0054) — far
  inside the 1.15 bar. The paired-TTFT and throughput bars' sd is today's arm's over its four rounds (§10's rev-3 C-m2).
- **The 5090** borrows the 4090's lock (no 5090 GGUF lock yet), as at step 0.

### 15.7 Review

One reviewer on the aggregate change (A, B, C). Round 1: no blocker; one major — with a drafter, the alternation's
capped prefill step could exceed `max_step_tokens` when C did (no build check applies with a drafter), crashing the
step's buffers (reproduced on CPU) — fixed with a test that fails before; five minors, all fixed or recorded: the dead
minimum chunk (dropped; the liveness test now draws only configurations the engine accepts and checks every step's
size), the decode-row check's bitwise branch depending on cuBLAS (now exact by construction), the mixed worst-step
profile untested on the CPU (recorded: the GPU bounded and desktop runs exercise it, §15.2), the bounded test's mixed
configuration in every run (now in one, with the "no step mixed" failure shown), `check_locks` not checking
`choices` (now does); nits fixed. Round 2: nothing at blocker, major or minor; the documentation items (this
section) and one test nit, fixed. Round 3 (this section against its records, every cell recomputed): no blocker or
major; four minors in the text (the 5090's round spread, the gaps against the simulator, the long prompt's
throughput at 512, the records' commit) and nits, all fixed.

### 15.8 Suite

Full GPU suite at f567fe8 (two RTX A5000s; `pytest core std plugins judge testing tools`): 1320 passed, 65 skipped in
17 min 54 s (step 3: 1192 / 65 in 18 min 13 s; 1233 at this step's base). The new cases: the scheduler's liveness test
in three modes (40 → 120 cases, ≈ 2 s on the CPU), the build test, the mixed-vs-today engine test (4 cases, ≈ 11 s),
the decode-row L0 (2 cases, ≈ 3 s), and two bounded configurations in one existing CPU test. `scripts/ci/run_all.sh
main` is green.

### 15.9 For the lead

- **Cap 512 on every class** for the close's propagation (`[engine] chunk_with_decode = 512` on the plain locks
  that serve a8 GGUF: the 5080's and the 4090's 27B; the 5090 borrows). Cap 256 fails §1.3 everywhere, the 5080
  included, where the simulator passed it.
- **The simulator under-costs mixed steps:** paired TTFT and long-prompt TTFT p99 cost 2–14 points more than it
  predicted at 512 (§15.4), still inside the bars.
- **GQA's q8_0 decode rows go through the decode kernel** (two calls) by measurement (§15.3); FlashInfer keeps one call.
- **The drafter predicate** is now one: units see `max_draft_tokens` 0 without a drafter (§15.6), which also drops
  GDN's replay buffers and rewind site for deployments that set drafts without a policy (a lock difference at the
  close's propagation for any such lock; none is committed).
- **Another RTX 5090 desktop fell off the bus** (Xid 79) mid-run; a second one ran every 5090 row.
- **Speculative alternation** helps the gaps less and costs long-prompt TTFT (§15.5): recorded, not gated, as §1.3 says.

## 16. As built (step 5): D8's gate

### 16.1 Method and prediction (fixed before measuring)

D8's f on 8B Q4_K_M decode B1 (the rtx-5080 lock 7, every class, as at step 0), at the head with every kept fusion,
profile-free (`bench/d8_gate.py`, from paired rounds of `step_profile.py`, `qmm_sweep.py` and `glu_sweep.py`):

- **bytes** = the weights the step's matmuls read (the fused gate/up's whole stack and the head included) + the
  embedding rows it gathers (one row: 2,304 bytes) + the KV read at the graphed window's context;
- **GEMV time** = each call of the step priced from the per-call sweep (graph-timed, unprofiled; the fused gate/up
  call from the glu sweep's fused time) less the isolated, graph-timed q8_1 quantize of its `in` (`qmm_sweep.py`'s
  `quantize`), which every swept call includes; B = weight bytes / GEMV time;
- f = 1 − bytes / (B × step), step the unprofiled graphed wall; projection 1 / (1 − f·c) at c = 0.4 and 0.55; stop
  under 1.15× at c = 0.55 (f < 24%).

`step_profile.py`'s counter now counts the fused gate/up calls (`QLinearGlu` has its own `__call__`, which it missed:
at the head it would have dropped the 36 gate/up calls, 2.04 GB of the step's 4.67 GB) and the embedding bytes.

**Prediction.** Step 2's A/B measured this step with every fusion at 3.684 ms on the 5090 and 6.09 ms on the 5080
(§13.3's records), against step 0's 4.14 and 6.54 ms; the GEMVs are nearly unchanged (glu only fuses the SwiGLU into
the gate/up GEMV), and the isolated quantize (≈ 0.7–1 µs per call, §12.3) is a little under the profiled one. So
f ≈ 1 − (≈ 3.06 ms of bytes at B) / 3.68 ms ≈ **17% on the 5090 (≈ 1.10× at c = 0.55: stop)** and ≈ **9% on the
5080 (stop)**: the fusions took most of the glue a megakernel would have recovered.

### 16.2 Results (b66b002, 2026-10-15)

Each desktop was clock-screened before and after (`bench/clock_check.py`, 27B decode at batch 4): RTX 5090 SM p10 /
p50 2565 / 2655 MHz before and 2542 / 2595 after, RTX 5080 2737 / 2737 both times; no thermal or hardware slowdown, at
the power cap in every sample (as in §11.1); steady. Three paired rounds per class (a profile, a sweep and a glu sweep
each); the table gives their means. Records `evidence/m07-step5_d8_<class>_2026-10-15.json` (the raw records of
every round, the screens and `d8_gate.py`'s output).

| class | step ms | GEMV ms (swept − isolated quantize) | B (GB/s) | read-only stream, 2 GiB (GB/s) | f (per round) | projected gain, c = 0.4 / 0.55 | gate |
|---|---|---|---|---|---|---|---|
| RTX 5080 | 6.087 | 5.579 − 0.118 = 5.461 | 855 | 913 | **8.6%** (8.61 / 8.58 / 8.59) | 1.036 / **1.050** | **stop** |
| RTX 5090 | 3.705 | 3.187 − 0.117 = 3.069 | 1522 | 1682 | **15.6%** (15.65 / 15.62 / 15.54) | 1.067 / **1.094** | **stop** |

- bytes per step: weights 4.6705 GB (163 calls, 36 of them the fused gate/up), KV 0.0882 GB (598 tokens of context in
  the graphed window), the embedding row 2,304 bytes.
- The isolated quantize is 0.72–0.73 µs per call at one row on both classes (145 of the 163 calls quantize in the
  step; every swept call does), against the profile's 0.16 ms per step. The profiled GEMV sum (3.153 / 5.519 ms) is
  2.7% / 1.1% above the swept GEMV time: the PDL overlap of §11.2 (r2 1.026 / 1.015 here).
- **Where step 0's f went.** Step 0's records under this method (its sweep's calls less the isolated quantize):
  5090 f 25.0% (1.159×), 5080 14.5% (1.087×) — the method alone moved step 0's 26.9% to 25.0%. The GEMV time is
  unchanged (3.05 → 3.07 ms on the 5090), and the step fell from 4.14 to 3.71 ms (−10.6%: the bf16 `kv_write`, D3,
  glu; §13.3): the fusions took the glue a megakernel would have recovered.
- **Even against the stream's bandwidth** (B = 1682 GB/s, which credits the megakernel with the GEMVs' own gap to
  peak, which D8 excludes), the 5090's f is 23.6% (1.104 / 1.149×) and the 5080's 14.4% (1.061 / 1.086×): under
  the bar either way.
- Against the prediction (§16.1: ≈ 17% and ≈ 9%, stop): as predicted; the 5090's f is ≈ 1.4 points lower (its GEMVs
  took 3.07 ms against the estimate's ≈ 3.0, and its step 3.705 ms against §13.3's 3.684).

### 16.3 Verdict: stop

f is under 24% on both D8 classes (the projection at c = 0.55 is 1.094× on the 5090 and 1.050× on the 5080, under
1.15×), so D8 stops here: no prototype in `models/qwen3/fast/`, no design addendum and no core seam (§1.4: "recorded
either way"). What is left beside the GEMVs on the 5090's B1 step, profiled: 0.65 of 3.80 ms of kernel time
(attention 0.23 ms, quantize 0.16, norms 0.12, rope and residual 0.09).

### 16.4 What changed

- `bench/d8_gate.py` (new): D8's f from paired records, profile-free (§16.1).
- `bench/step_profile.py`: the counter counts `QLinearGlu`'s fused calls (marked `glu`) and the embedding bytes;
  a composed glu call is counted once, as the plain qlinear it runs. Step 0's records are unaffected (no glu then).
- No engine, core, std or plugin change; no test (the bench is outside the suite).

## 17. Close (2026-10-15)

### 17.1 The close panel's findings, fixed

Commits 03692ce (numerics), 6959f22 (architecture), 94bb0d9 (correctness) and f0b9100 (review round 1). Every code fix
has a test that fails without it (run against the code before it).

**Numerics.**
- **MAJOR, graph replay of the fused decode kernels.** `test_cuda_graph_replays_bitwise` (`plugins/tests/test_ggml_cuda.py`)
  gains the fused gate/up at one row (Q4_K, and NVFP4's HF layout with its `scale2`) and one `prepare_input` read by
  two consumers of the same `in` (a Q4_K and a Q6_K linear), whose host-side decision to read the prepared buffer is
  captured; each replays bitwise its eager calls, also on new inputs. The decode-row check
  (`test_attention_kv_quant.py::test_a_decode_row_is_the_last_prefill_row`) gains a captured variant: one request's
  decode row of the GQA unit (D3's fused `qk_rope_kv` selected) captured at position 31, replayed at 31, 32 (opens a
  page) and 33, each step staged on the host as the engine stages a graphed step; output and KV bitwise another
  request's eager decode rows, on bf16 and q8_0 KV.
- **Graph-only plants, verified once (not kept):** the GQA planner deciding decode rows' write slots on the host at
  plan time (correct eagerly, stale on replay) fails the captured decode-row check at position 32 on both KV formats,
  while its eager part passes; `prepare_input` memoising by input address (skips the quantize on a re-used address)
  fails the new prepared graph case on new inputs while the eager L0 (`check_qlinear`, Q4_K) passes. The clean code
  passes both.
- **Minors.** `GLU_TYPES` = {Q4_K, NVFP4}: Q5_K (+0.6 / +1.5 / −0.3% in §13.4's sweeps, within a sweep's noise, no L2
  of its own) composes; it is a table change, the L0 asserts each type fuses or composes as the set says. §15.6's "no
  A/A arm" is replaced by the A/A from the records (`bench/mixed_bars.py --aa`). `test_triton_attn_configs_fit_every_class`
  compiles each fitted config once, at the widest width that runs it (shared memory grows with BLOCK_M). The scheduler
  liveness test is 3 cases (one per mode) looping over its 40 workloads.
- **MTP acceptance is unchanged** by M0.7's fusions: tokens per verify step on the agent trace (k 3 and 2; step 4's MTP
  records, today's arm) 2.78 on the RTX 5080 and 2.85 on the 4090, against M0.4's 2.78 / 2.82 (m04-spec §10).

**Architecture.**
- **`_Quantized._select`** chooses `qlinear_glu@1` among `BuildCtx.candidates` of the qlinear's provider before
  anything is recorded: when only another provider serves it, no site is recorded and SwiGLU composes (before, the
  lock recorded a binding that never ran); when the deployment would choose another provider's (a pin or a higher
  priority), the build refuses, naming the pin that fixes it. `candidates(op, facts)` joins the `BuildCtx` protocol
  (ABI 0.2, unmerged: an addition), read-only and recording nothing; `ChildContext` delegates it. Toy proof:
  `test_quantized_units.py::test_a_glu_kernel_of_another_provider_binds_no_site` (both cases).
- **The kernel prop is `serves_decode_rows`** (FlashInfer's prefill; GQA runs a mixed step's decode rows in one call
  when its prefill kernel has it). The planner attribute stays `mixes_row_kinds`. `core/abi.txt` regenerated at 0.2.
- **`BuildCtx.options["chunk_with_decode"]` is the engine's effective value** (0 under a drafter, as
  `max_draft_tokens` is 0 without one); std's activation-mode refusal (`std.ops`, `std.moe`) reads that key alone.
  `test_plans.py::test_mixed_steps_need_room_mixing_planners_and_no_drafter` builds an n-gram deployment (options 0,
  the scheduler keeps 6); the toy refusal test's (256, drafts 3) case now refuses (it bound before).
- **Design:** §4.4 and §5.7 (class and shape tables in kernel plugins are tuned defaults pinned by `src`; retuning moves
  every lock that uses the plugin); §5.3 (fused ops over several units: `fusion_facts()`, `config["fused"]`, the
  judge's oracle `holon_judge.fused` for `qk_rope_kv`, which has no `reference.py`, provider-filtered selection and
  `prepare_input` handles); §5.5 (`holon serve`'s model-name grammar — a path, `<org>/<repo>`, `<org>/<repo>:<TAG>`,
  `<org>/<repo>/<file>` — part of the CLI contract; release §7 lists it with the frozen surface); §5.6 (the crossover
  from the per-class, per-shape table with llama.cpp's rule as fallback; mmq writes bf16 with the `scale2` epilogue);
  §7.1 (the options key and the prop).
- **Unused ABI names** — `SLOTS` entries no plugin fills, the dead `SchedulePolicy`, the parsed-but-unused
  `[tunables]` / `Binding.tunables` — are cut at release step 6 (with ABI 1.0), not here.

**Correctness.**
- **Alternation inherited from a plain lock.** A plain lock's `chunk_with_decode` 512 turns on alternation when drafting
  flags are layered over it (with a drafter the units see 0 and steps never mix; the scheduler alternates only when
  C > 0 and a drafter exists). The README says so (Options, "Mixed steps"), with `--chunk-with-decode 0` to turn it off;
  release step 7's per-lock MTP decision sets `chunk_with_decode` explicitly in every speculative lock.
- **HF hub resolver:** with `model.safetensors.index.json`, only the safetensors files it names (plus the config,
  tokenizer and template files) are fetched and sized; the index alone is fetched to know them (the hook's contract:
  "never downloads weights"); if it cannot be fetched, every top-level weight file is, as before. Test cases: a
  consolidated copy beside the shards is left out; an unfetchable index keeps it.
- **Draft floors route by the engine's merged shapes:** `drafter.row_limits` asks `mmvq_rows` for the matmul the engine
  runs (`gguf.merged`, the same `MERGES` as `engine_shapes`; `gguf.module_name` maps a draft's or MTP block's GGUF
  names), not per tensor. Test: on the RTX 5090 an IQ4_XS gate/up merges to 5120x34816 (7 rows), its own 5120x17408
  is 6. No committed floor moves (the DFlash and MTP floors were recorded on an A5000, which has no entry).
- **`hint()`** names `[::1]` for a `::` bind (an IPv6-only socket answers it; 127.0.0.1 may not).

**Review.** One reviewer on the aggregate change. Round 1: no blocker or major; six minors (the A/A median pooled r with
1/r and was 1 by construction — unordered pairs now; an unfetchable index raised past `claim_ref`'s hints — falls back;
design §4.4 claimed locks record the tables' choices; stale §13 notes and §17 citations; `_select`'s docstring; README
ahead of the locks — lands with them) and four nits (`candidates` takes a dict where `select_kernel` takes keywords;
the captured check's output rests on cuBLAS picking one algorithm in capture and eager; one unused consumer per
graph case; GQA's `_fusion` raises instead of composing when the kernel's `supports` refuses the facts — pre-existing,
recorded for the release's review, not changed).

### 17.2 The single propagation (§7 step 6)

Every development-class lock propagated once, at the final code (f0b9100), in the release uv environment (`uv sync
--frozen --group dev` from the committed `uv.lock`: CPython 3.12.12, torch 2.11.0+cu130, Triton 3.6.0; the plans'
`uv_lock` sha256:922c0a6d…), in a CUDA 13.0 container on one RTX 5080 and one RTX 4090 desktop. Each desktop was
clock-screened before and after: RTX 5080 SM p10 / p50 2737 / 2745–2752 MHz, at the power cap, no slowdown; RTX 4090
2685 / 2685 MHz, no slowdown; both steady. Every lock gains `slots logits_processor = grammar.xgrammar` (release D3);
the plain 27B locks gain `chunk_with_decode = 512` (§15.9); the others record `chunk_with_decode = 0` (a new
`[engine]` key, M0.7's ABI 0.2). Per lock: `holon_tools propagate --approve` (L2 against the golden and the floor of its
path and KV format, the trace twice), then `holon_judge snapshot --runs 3` at the new lock's settings and pins. M0.6
Gate B's 8B FP8 locks are new: `tune --new-model` on the RTX 5080, then on the RTX 4090 `tune` from the 5080's lock
(its first for the class, as M0.6's second-class locks; a first try with `--new-model` was refused because the 5080's
lock existed by then). Record `evidence/propagate_m07-close_2026-10-15.json`.

| class | lock | floor | top-1 (floor; budget) | KL p99 (floor; budget) | ΔPPL vs floor (budget) | snapshot ×3 | approved | s |
|---|---|---|---|---|---|---|---|---|
| RTX 5080 | Qwen3-30B-A3B UD-IQ3_XXS → 6 | mmq | 98.25% (98.33; ≥ 98.08) | 5.68e-2 (5.93e-2; ≤ 0.119) | +0.035% ± 0.127% (≤ 0.568%) | identical | ✓ | 141 |
| RTX 5080 | Qwen3-8B Q4_K_M → 8 | mmq | 98.43% (98.36; ≥ 98.11) | 3.54e-2 (3.58e-2; ≤ 7.17e-2) | −0.056% ± 0.091% (≤ 0.477%) | identical | ✓ | 73 |
| RTX 5080 | Qwen3-8B UD-IQ3_XXS → 8 | mmq | 98.90% (98.87; ≥ 98.62) | 1.33e-2 (1.53e-2; ≤ 3.06e-2) | +0.055% ± 0.077% (≤ 0.442%) | identical | ✓ | 66 |
| RTX 5080 | Qwen3.8-27B UD-IQ3_S, q8_0 KV → 5 (C 512) | mmq, q8_0 KV | 97.50% (97.54; ≥ 97.29) | 0.639 (0.712; ≤ 1.42) | +0.009% ± 0.805% (≤ 2.001%) | identical | ✓ | 143 |
| RTX 5080 | nvidia Qwen3-8B NVFP4 → 3 | mmq (W4A8) | 98.77% (98.69; ≥ 98.44) | 1.29e-2 (1.24e-2; ≤ 2.49e-2) | −0.096% ± 0.053% (≤ 0.359%) | identical | ✓ | 72 |
| RTX 5080 | **Qwen3-8B FP8 → 1 (new)** | W8A8 prefill | 98.14% (98.17; ≥ 97.92) | 4.26e-2 (4.18e-2; ≤ 8.37e-2) | −0.066% ± 0.127% (≤ 0.516%) | identical | ✓ | 107 |
| RTX 4090 | Qwen3-8B bf16 → 12 | bf16 | 98.99% (98.93; ≥ 98.68) | 4.70e-3 (5.76e-3; ≤ 1.15e-2) | −0.076% ± 0.037% (≤ 0.343%) | identical | ✓ | 148 |
| RTX 4090 | Qwen3-30B-A3B Q4_K_M → 6 | mmq | 98.26% (98.22; ≥ 97.97) | 6.11e-2 (5.88e-2; ≤ 0.118) | −0.083% ± 0.139% (≤ 0.588%) | identical | ✓ | 163 |
| RTX 4090 | Qwen3.8-27B UD-Q4_K_M → 5 (C 512) | mmq | 97.11% (96.95; ≥ 96.70) | 0.893 (0.648; ≤ 1.30) | −1.822% ± 0.749% (≤ 2.123%) | identical | ✓ | 184 |
| RTX 4090 | nvidia Qwen3-8B NVFP4 → 3 | mmq (W4A8) | 98.70% (98.69; ≥ 98.44) | 1.22e-2 (1.24e-2; ≤ 2.49e-2) | −0.100% ± 0.065% (≤ 0.359%) | identical | ✓ | 82 |
| RTX 4090 | nvidia Qwen3-30B-A3B NVFP4 → 3 | mmq (W4A8) | 98.40% (98.36; ≥ 98.11) | 4.85e-2 (4.92e-2; ≤ 9.84e-2) | +0.104% ± 0.116% (≤ 0.503%) | identical | ✓ | 155 |
| RTX 4090 | Qwen3-30B-A3B-2507 MXFP4_MOE → 2 | mmq (W4A8) | 98.43% (98.47; ≥ 98.22) | 2.64e-2 (2.81e-2; ≤ 5.63e-2) | +0.089% ± 0.089% (≤ 0.471%) | identical | ✓ | 162 |
| RTX 4090 | **Qwen3-8B FP8 → 1 (new)** | W8A8 prefill | 98.19% (98.17; ≥ 97.92) | 4.17e-2 (4.18e-2; ≤ 8.37e-2) | −0.127% ± 0.092% (≤ 0.481%) | identical | ✓ | 178 |

- **Every verdict passes:** L2 within budget against the floor of the lock's path and KV format, and the snapshot
  trace run-identical ×3 (and twice inside `propagate`). **13 approvals** under the owner's delegation, one per lock
  (11 propagations, 2 new locks), each committed with this section.
- **What moved, as expected:** the core's and most plugins' `src` (ABI 0.2, the UX step, M0.7's code); every GQA
  layer gains an `attention.qk_rope_kv@1` site and its `kv_write@1` config the `fused` list; FlashInfer's prefill sites
  their props (`serves_decode_rows`); GGUF merged gate/ups a `qlinear_glu@1` site (Q4_K fuses; other types compose,
  §13.9), and `kernels.ggml_cuda`'s `src` pins its per-shape crossover and glu types; the speculative-only
  `gdn_rewind@1` is not recorded (no lock drafts). Every trace moved (logprob deltas from the second step on: the
  fusions' numerics). The 27B's 4090 lock (UD-Q4_K_M) drifted most: ΔPPL against the fp32 golden −1.80% ± 0.69%
  (a deviation, not an improvement; M0.6's previous lock already had −1.43% ± 1.36%, within one SE), and KL p99
  0.740 → 0.893 (+21%; p50 4.26e-4 → 4.49e-4), inside the 2× floor budget (≤ 1.30); the 5080's 27B barely moved (KL
  p99 0.628 → 0.639). D3 is the only bounded change on that lock's prefill path; an L2 A/B with `qk_rope_kv` composed
  would attribute it (not run).
  KV pools are measured afresh where the previous lock's snapshot did not record its pool bytes (the GGUF and bf16
  locks), else reused, as `propagate` does.
- **The trace carries mixed steps at C 512:** the snapshot trace at the rtx-5080 27B lock's flags ran 61 mixed steps
  (engine stats, on an RTX A5000), so §1.5's "capped chunk beside decoders" is in every plain 27B lock's trace.
- **Sweep time and approvals against plan §9's triggers:** RTX 5080 10.0 min (6 locks), RTX 4090 18.0 min (7 locks,
  the FP8 retry included); the classes ran in parallel, ≈ 20 min of wall time. Under the 30-minute trigger per class.
  **The approvals trigger fires:** 13 approvals in one day against "> 1 / day". Recorded, not a blocker (release §6:
  it schedules async propagation for v1). The count is what one propagation per milestone close costs with every
  changed plan needing an approval while changed-line coverage is not recorded (design §4.6); release step 7 adds
  another round of the same size.
- **Release classes untouched** (release step 7): the rtx-5090 Qwen3-8B bf16 lock (6) pins pre-M0.7 code and is
  outdated; no rtx-3090 lock exists.

### 17.3 Acceptance (§1)

| # | item | status |
|---|---|---|
| 1 | fusions shipped only where they pay | **met**: the bf16 `kv_write`, D3, D4, D11, D6 (glu and shared q8_1) and D5 kept on every class by §1.1's rule (§13.3, §14.2); GDN gating dropped at step 0 (< 1%); glu's types narrowed to Q4_K and NVFP4 (§13.4, §17.1) |
| 2 | crossover entries per shape where they pay | **met**: 25 RTX 5080 and 11 RTX 5090 entries (the no-cast column), rows 4 / 8 guarded, ≥ half the re-stated saving delivered, small-step L2 passing (§12.3, §12.6, §13.5); the 4090 has none |
| 3 | mixed steps on plain locks | **met at cap 512** on the RTX 5080 and 4090 (and the 5090), ≥ 4 rounds per arm: gap p99 / max 7.3–13×, paired TTFT 1.02–1.10 (bar 1.15 / 1.18), long-prompt TTFT p99 +14–16% (≤ 25%), throughput within 0.3% / 2.1%; cap 256 fails everywhere (§15.4). Speculative alternation recorded (§15.5). The A/A comes from the records, not an arm (§15.6) |
| 4 | megakernel experiment | **met: stop**, recorded (f 8.6% / 15.6%, 1.050× / 1.094× at c = 0.55; §16) |
| 5 | locks propagated once, at the close | **met** (§17.2): 11 propagated, 2 new (M0.6 Gate B's FP8), all approved under the delegation; the trace gains a capped chunk beside decoders (61 mixed steps) |
| 6 | core only §3's seams, toy-proven, ≤ 6000 lines; suite time does not grow | **met with a deviation**: core 4369 / 6000 lines; besides §3.1's seams, `BuildCtx.candidates` (close panel, toy-proven, §17.1); full GPU suite 17 min 59 s at the close against step 0's base 18 min 55 s |

### 17.4 Deviations from the design, all recorded where they were made

- §12.1: the std.ops minors landed with their code at step 2, not in the ABI commit.
- §12.4: `triton_attn`'s row tiling is bitwise only where Triton keeps its warp layout (head_dim 256; one ulp at 128).
- §12.5 / §15.6: std's activation-mode refusal stands down under a drafter — at the close by reading the engine's
  effective `chunk_with_decode` (§17.1), no longer `max_draft_tokens`.
- §13.8: the FP4 repack holes (an M0.6 follow-up) measured, not fixed.
- §13.9: fused kernels are offered through `config["fused"]`; D3's oracle lives in the judge; glu kept per type.
- §13.7 / §6: the decode-row RoPE plant passed `bounded` (prediction missed); the decode-row check (§6, step 4) and,
  at the close, its captured variant cover it.
- §14.1: D5 first built as torch ops, then the design's one launch.
- §15.6: the minimum chunk is gone (it could never bind); GQA's planner always mixes (the prop picks one call or two).
- D9 stays on step 1's with-cast table (§13.12).
- §17.1: a core seam beyond §3 (`BuildCtx.candidates`, read-only); the build refuses a cross-provider glu choice
  instead of composing.

### 17.5 Decisions taken at the close (lead, under the owner's delegation)

- The close panel's findings fixed as §17.1 records; the panel re-checks before the merge.
- The 13 locks approved and committed, every verdict passing; the approvals trigger recorded (§17.2).
- `chunk_with_decode` 512 on the plain 27B locks, 0 recorded on every other lock; release step 7's MTP decision sets
  it explicitly for speculative locks.
- Unused ABI names cut at release step 6 (§17.1).
- M0.6 Gate B's acceptance 15 (FP8 locks) closed here (`docs/dev/m06-lowp.md`).

### 17.6 Suite and CI

Full GPU suite at 94bb0d9 (two RTX A5000s; `pytest core std plugins judge testing tools`): 1209 passed, 65 skipped in
16 min 50 s (step 4: 1320 / 65 in 17 min 54 s; the liveness test's 120 cases are 3 now, −117; +6 new cases: 3 graph
replay, 2 glu-provider and the draft-floor test; the captured decode row is inside an existing test). Final, at
442d3ab (the round-1 fixes and the locks in): 1210 passed, 65 skipped in 17 min 59 s (+1: the unindexed hub case);
`scripts/ci/run_all.sh main` green.

### 17.7 For the lead

- **13 approvals in one day** (plan §9's "> 1 / day" fires; recorded, not a blocker); sweep times under 30 min.
- **README's lock table is stale** for M0.6's files ("TBD (M0.6 merge): … not in this tree yet"): the NVFP4,
  MXFP4_MOE and now FP8 locks are in `plans/`. Release step 10 rewrites the table; noted, not changed here.
- **GQA's `_fusion`** raises when the KV writer offers `qk_rope_kv` but the fused kernel refuses the units' facts (a
  pre-existing pattern of the one fixed in `_select`; review nit): a `candidates` check there would compose. For the
  release review.
- **The rtx-5090 bf16 lock is outdated** until release step 7.
