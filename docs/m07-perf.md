# M0.7 — performance: fused glue, mixed steps, a gated megakernel experiment

Status: **revision 3.1, reviewed clean** (2026-10-14) by a panel of three (correctness, architecture, numerics / performance / tests) over four rounds; every finding is
resolved in §10.

Scope: plan §2 (M0.7: "fused prefill glue (KV-store kernel, add+norm, q/k norm) and a model-specific megakernel
experiment (design §5.7)"), narrowed and extended by what M0.3–M0.5 recorded for M0.7: the verify-step glue
(m04-spec §13), mixed prefill/decode steps (M0.5 measured streams silent up to 36 s behind a long prefill), and the
mmvq/mmq crossover. The small-batch quantized matmul (plan §6.1) leaves M0.7 as a recorded follow-up with a measured
gate (D9); the forward/sample split stays deferred (M0.5 D2: 1.061× / 1.026×, under its 1.10× rule). Measurements and
prior art are in the local research note `docs/brainstorm/m07-perf-research.md` ("R §n", taken on an RTX A5000) and in
the reviewers' re-derivations recorded in §10; the consumer classes are re-measured first (step 0).

Implementation starts after M0.5 (merged) and M0.6's Gate A merge into `main` (both change the core, the scheduler and
the ggml plugin; M0.7 needs Gate A's ggml pin `cbe28bb3` and fp4 mode, not Gate B's `triton_fp8`); step 0 runs on that
pin. If Gate B merges after M0.7's close, it brings one more shared-change propagation (release design §9). The plan's M0.7 row is updated with this revision; design §7.1 / §4.2 and
plan §10 (mixing as a core budget) are updated with step 4's code.

## 1. Goal and acceptance

Faster steps and shorter stalls on the consumer classes, with every change bitwise where it can be and judged on the
paths it touches. Acceptance:

1. **Fusions shipped only where they pay** (D3–D6): each is measured incrementally (D3 on top of the bf16 `kv_write`;
   verify rows by per-step wall time, not TPOT) and kept only if its saving is ≥ max(0.5%, 3 sd of its row) on some
   release-class row (RTX 5080 / 4090 / 5090; graphed decode B1/B4/B8, verify steps, prefill 512/2048), else reverted
   and recorded. Classification (D10): **bitwise** — D4, D5, D6's shared quantization, the bf16 `kv_write`; **bounded**
   — D3, D6's gate/up + SwiGLU and GDN gating, judged by L0 and the L2 of §6, bounded where decode paths change.
2. **Crossover entries shipped only where the sweep shows a gain** (D2): per (class, type, activation mode), only where
   step 0's sweep shows ≥ 1% of a decode or verify step; each shipped entry delivers ≥ half its re-stated saving and no
   guarded row (`holon_tg` batch 2 / 4 / 8, verify steps) regresses beyond its noise.
3. **Mixed steps** (D7), on plain (non-speculative) locks of the RTX 5080 (27B) and 4090, ≥ 4 rounds per arm: the
   per-request maximum inter-token gap improves ≥ 2× at p99 and max (p50 recorded); the paired TTFT ratio (the median
   over (session, turn) of TTFT_mixed / TTFT_base) is at most 1 + max(15%, 2 sd of the baseline rounds), and the
   long-prompt scenario's TTFT p99 regresses by at most 25%; throughput by at most max(5%, 2 sd of the baseline
   rounds); TPOT p99 recorded, not gated; the mixed-steps counter > 0. Speculative locks record the alternation's gap and long-prompt TTFT only.
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
     that cannot mix fails at startup. Only steps with `n_decode > 0` take the per-row-kind path: decode rows first,
     so a mixed step is two contiguous token slices; GQA's prefill kernel takes both when its prop says it accepts 1-token rows, GDN runs
     chunked over the prefill slice and recurrent over the decode slice, re-basing published `row` indices for the
     chunked subset. Unmixed steps (all steps at the default) keep today's paths, so snapshots at 0 are unchanged.
   - Mixed steps run eagerly (graphs stay uniform). The worst-step memory profile runs a mixed step when the setting is
     > 0 (a chunk's activations alive through sampling).
   - The dead `SchedulePolicy.mix_phases()` protocol (`core/holon/api/decoding.py:39`) is deleted (a holon.api minor).
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
- **D2. The crossover is a per-class table inside `kernels.ggml_cuda`**, keyed by (class, type, activation mode), with
  llama.cpp's rule as the fallback, behind one exported function `mmvq_rows(fingerprint, type, mode)` used by
  `QLinear` and by both judge sites (`drafter.py`, `gguf.py`); llama.cpp's fallback takes an explicit compute
  capability (today's C API reads the current device's, `csrc/api.cu:111`), and the existing ggml plugin test asserts
  that `QLinear`'s actual dispatch per row count matches `mmvq_rows` for every lock class. The table is pinned by the plugin's `src` hash (as `triton_attn`'s class configs are), no
  lock-to-kernel core path. Entries only where D1's sweep shows ≥ 1% of a step; a range check (≤ 8, mmvq for every n
  up to it). Under M0.6's `fp4` mode a lower crossover moves 5–8-row steps from W4A8 to W4A4: a precision change,
  judged by the small-step L2 of §6. The kernel is chosen by the row count the step actually runs (eager real rows,
  graphed bucket rows) — recorded.
- **D3. Fused q/k-norm + RoPE + KV store** is a std op `attention.qk_rope_kv@1` under `std.mixers.gqa`, used only when
  the injected `qk_norm` and `rope` units answer a new optional `fusion_facts()` (std.ops minor: eps, offset, theta,
  `rotary_dim`, `max_positions`, the gate interleave, the reference identity, and the cos/sin table's static name and
  layout, which become part of the `rope@1` instance contract in `std/holon_std/ops/kernels.py` — a rope provider
  without such a table makes the unit answer None) and a provider supports those facts — never by unit names; a future RoPE variant answers None and keeps the composition. Its reference is built by
  the judge from the composed units' references (unit isolation); the Triton provider covers bf16 and q8_0 KV, rounds
  to bf16 after the norm and after RoPE (before q8_0), handles T = 0; the model-only `_causal` build keeps the
  composition. Bounded (not bitwise vs FlashInfer's rmsnorm reduction). The bf16 Triton `kv_write@1` provider (one
  launch for k and v) is separate and bitwise.
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
  - **shared q8_1 quantization** (bitwise; built only if step 0 shows ≥ 1%): explicit, never a memo — an optional
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
  measurement on the recalibrated simulator and real runs (candidates 256 / 512).
- **D8. Megakernel experiment, gated on speed of light.** After D3–D6, per class (5090, 5080), on 8B Q4_K_M B1 decode:
  f = 1 − (bytes read per step) / (B × step time), with bytes = the weights read (the embedding table excluded) + KV,
  and B = the bandwidth the step's GEMVs achieve, from unprofiled GEMV time (profiled times make B ≈ 2% low; D1's
  second ratio corrects them) — so f excludes the GEMVs' own gap to peak, which a megakernel does not recover; a
  read-only stream over ≥ 1 GB is recorded beside it. Projected gain 1 / (1 − f·c) with c = 0.4–0.55 (the
  share of the gap to speed of light the published megakernels closed: Cohere 0.38, Hazy 0.56), reported at both
  ends. Stop if the projection at c = 0.55 is under 1.15× (f < 24%); otherwise build a one-layer prototype in
  `models/qwen3/fast/` and go only if its L0 against the judge-built recipe reference passes and it projects ≥ 1.15×
  at B1 with no regression at B4–8. A go means a design addendum (core seams: one composite bound to many layers, a
  decode-loop override, taps written by the megakernel), reviewed by this panel.
- **D9. The small-batch quantized matmul leaves M0.7** as a follow-up gated by step 0's per-call sweep: the target's
  4-row / 1-row GEMM time ratio after D2 > 1.25 on a release class (it would be a row-dispatch path inside
  `kernels.ggml_cuda` with its own `config["act"]` entry, not a new provider).
- **D10. Classify every change** bitwise or bounded before building (§1.1); target bitwise; propagate once at the close.

## 5. Predictions (provisional; re-stated per class after step 0)

| Item | Prediction |
|---|---|
| Crossover (D2) | 4090 k-quants ≈ 0 (bound 3.6 ms per DFlash verify step); 5080 already tuned; gains possible on sm_86 (rtx-3090) and i-quants (UD files) — sized by step 0 |
| bf16 `kv_write` | 8B Q4_K_M B1 decode −2.5–3% (A5000) |
| `qk_rope_kv` (D3), on top of `kv_write` | 8B Q4_K_M B1 −1.7–2.3%; 27B with q8_0 KV ≤ 0.3% (likely dropped by D1's rule); prefill 2048 −1–1.5% |
| mmq bf16 out (D4) | GGUF prefill −1.5–3% on the consumer classes (−4–5% on the A5000) |
| GDN rewind glue (D5) | verify step −2–3% (27B) |
| D6 | 8B Q4_K_M B1 −2% (gate/up + SwiGLU); 27B decode ≈ −1% (shared quantization, decode only); prefill −0.6% (gating) |
| Fusions combined | 8B Q4_K_M B1 decode −6–7%; bf16 −2%; GGUF prefill −3–5% |
| Mixed steps (D7), plain locks | worst gap: 4090 511 → 97–154 ms, 5080 710–974 → 120–220 ms (gap p50 −13 to −35%, within seed noise — recorded only); TPOT p99 within noise; TTFT p50 +4–49% at cap 256 (cap 512 smaller) |
| Long prompt, plain locks | 30K-token prompt behind decoders: 4090 9.3 s → ≈ 11.7 s at cap 256 (+26%), ≈ 9.1 s at cap 512; 5080 ≈ +8% at cap 256 — §1.3's +25% bar selects cap 512 on the 4090 |
| Alternation, speculative locks | 2048-token prompt TTFT on the 4090: 618 ms → ≈ 1008 ms at cap 256 (+63%), ≈ 736 ms at cap 512 (+19%) |
| Megakernel (D8) | A5000: weights 4.67 GB + KV 0.075 GB, GEMVs at 655–666 GB/s, step 8.64 ms → f ≈ 16–17.5% → 1.07–1.11× at c = 0.4–0.55: stop; consumer classes decided by their f |

## 6. Judge

- **Bitwise checks inside existing L0 loops:** D4 (`test_mmq_schedules`, fixup forced, plus the NVFP4 case), D5 (L1.5),
  D6's shared quantization (a mixed-type pair of consumers through one handle), the bf16 `kv_write`.
- **Bounded L0** (sm_86 / 89 / 120a + memcheck): D3 against the judge-composed reference, incl. partial rotary, the
  gated q layout and q8_0 1-token rows; `qlinear_glu@1` against its reference; GDN gating against the torch composition.
- **Small-step L2 for the paths propagate never runs** (it scores prefill only and bounded runs batch 48): `holon_judge
  l2 --max-step-tokens 1` against the `*-mmvq-floor` goldens (gate/up + SwiGLU, D3's decode rows); `--max-step-tokens
  8` and 4 against the floor of the path `mmvq_rows` picks (D2); floors recorded where missing (the 27B
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
| 1 | `mmvq_rows` and D2 table entries where the sweep pays | agreement test; guarded rows; small-step L2 | 1 |
| 2 | bf16 `kv_write`, D4, D6 (bitwise first), D3 (bounded) | L0 ×3, L2, bounded; keep-or-revert per §1.1 | 4 |
| 3 | D5 glue | L1.5 bitwise; bounded spec | 1.5 |
| 4 | Mixed steps (§3.1, D7); design §7.1 / §4.2 and plan §10 | core toy proof; bounded mixed-vs-today; agent + long prompt on 5080 / 4090 with A/A | 3 |
| 5 | Megakernel gate (D8); prototype only on "go" | records | 1–5 |
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
