# M0.7 — performance: fused glue, mixed steps, a gated megakernel experiment

Status: **revision 3.1, reviewed clean** (2026-10-14) by a panel of three (correctness, architecture, numerics /
performance / tests) over four rounds; every finding is resolved in §10. **Step 0 is done (§11); §5 is re-stated per
class in §11.11, for the panel's review before steps 1–6.**

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
- **r2 is 0.99–1.04.** The profiler inflates graphed kernel sums by at most 4%. The shares below are profiled kernel
  time divided by r2, and D8's B uses the unprofiled GEMV time.

### 11.3 Glue share per row

Share of the step, in %: graphed for decode and verify, eager wall for prefill. "—" means under 0.05%.
- **cast:** mmq's f32 → bf16 cast, plus NVFP4's f32 × `scale2` → bf16 multiply (on mmq, and on mmvq above one row).
- **bf16→f32:** native W4A4's activation copy.
- **other glue:** everything else (residual adds, gates, copies, index ops, sampling, and the verify step's drafter and
  rewind glue).

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

### 11.5 The crossover sweep (D2)

`bench/qmm_sweep.py` timed today's `QLinear`, graph-timed, for every (type, in, out, mode) of the `qlinear` sites in the
latest lock of every GGUF and FP4 checkpoint (107–113 shapes per class; real GGUF rows, random codes with valid scales
for the HF NVFP4 layouts, enough copies to defeat L2). It records mmvq at rows 1–8, mmq with the cast and without it
(rows 1–9), and llama.cpp's choice.
- **Per (type, mode) table limit:** the largest n with mmvq at least as fast as mmq for every row up to n, summed over
  the type's shapes weighted by their per-step calls.
- **Step saving:** the profiled decode and verify steps' calls priced at that limit against llama.cpp's.
- **Graph buckets:** graphed steps run at bucket rows (1, 2, 4, 8, …), so rows 8 (decode B5–B8, DFlash verify) and 4
  (decode B3–B4, MTP verify B1) are the ones that matter.

| class | type (q8_1) | llama.cpp's limit | measured (with cast / without) | step saving where it pays (with cast) |
|---|---|---|---|---|
| 5080 | iq3_s | 8 | 7 / 7 | 27B decode B8 −2.45%; DFlash verify B1 −2.07% |
| 5080 | iq4_xs | 8 | 6 / 6 | decode B8 −1.36%; DFlash verify −1.15% |
| 5080 | iq2_s, iq2_xxs, iq2_xs, iq3_xxs, iq1_s | 8 | 6–7 | decode B8 −0.84, −0.49, −0.33, −0.30, −0.10% (each < 1%) |
| 5080 | q2_k, q3_k / q5_k / q6_k | 5 / 6 / 7 | 3 / 5 / 5 | ≤ 0.17% (rows 4–7 only) |
| 4090 | every type | 8 (k-quants 4–8) | the same or higher | 0.0% on every row (llama.cpp's Ada rule holds) |
| 5090 | iq4_xs | 8 | 6 / 6 | 27B decode B8 −3.76%; DFlash verify −2.93% |
| 5090 | q6_k | 7 | 4 / 4 | DFlash verify −1.34% (its 5–7-row calls) |
| 5090 | iq3_s, iq4_nl, iq2_xxs, iq3_xxs, q3_k | 8 / 5 | 6–7 / 3 | ≤ 0.2% each |

- The whole table saves 5.9% of the 5080's 27B decode B8 and 5.0% of its DFlash verify step. On the 5090 it saves 4.1%
  and 4.5%, and on the 4090 nothing.
- MTP verify at one row (4 rows) gains ≤ 0.15% everywhere: mmvq still wins at 4 rows for every type that matters.
- Without the cast (after D4), the 5080's 27B B8 saving is 6.8% and the 5090's 4.6%. D4 itself is counted separately
  (§11.11).
- Native W4A4 (`nvfp4/fp4`) and NVFP4 at q8_1 keep llama.cpp's 8 on every class.
- The 5090 has no GGUF lock yet; its entries come from the files it borrows.

### 11.6 D8's f (8B Q4_K_M, B1 decode)

- bytes = the matmuls' weight bytes (4.67 GB, the embedding table excluded), plus the KV read at the graphed window's
  context (0.088 GB).
- B = weight bytes / the unprofiled GEMV time.
- f = 1 − bytes / (B × step). The step's idle is outside f, and r2 cancels.

| class | step ms | GEMV ms (unprofiled) | B (GB/s) | read-only stream, 2 GiB (GB/s) | f | projected gain, c = 0.4 / 0.55 | today's gate |
|---|---|---|---|---|---|---|---|
| 5080 | 6.54 | 5.40 | 865 | 913 | 15.9% | 1.068 / 1.096 | stop |
| 4090 | 6.90 | 5.88 | 795 | 844 | 13.2% | 1.056 / 1.078 | (not a D8 class) |
| 5090 | 4.14 | 3.04 | 1535 | 1683 | **25.2%** | 1.112 / **1.161** | prototype (f ≥ 24%) |

- **D8 is defined after D3–D6.** On the 5090, the surviving fusions are worth 9.3% of this step: kv_write 3.5%, D3
  2.9%, gate/up + SwiGLU 2.4% and shared q8_1 0.5%. With them, f falls to 1 − 0.748 / 0.907 = **17.5%**, which
  projects 1.075–1.107×. That predicts "stop" at step 5.
- On the 5080, f falls from 15.9% to about 10.5% after the same fusions (6.0% of the step).
- Step 5 measures the 5090 gate again after D3–D6, as designed.

### 11.7 D9's trigger

D9 is the target's 4-row / 1-row matmul time over its decode step's calls, at D2's table, with today's cast:

| class | target | ratio | bar 1.25 |
|---|---|---|---|
| 5080 | 27B UD-IQ3_S | 1.250 in the full sweep; 1.243 / 1.245 / 1.246 in three repeats (mean of all four 1.246, sd 0.003) | at the bar; not fired on the mean |
| 4090 | 27B UD-Q4_K_M | 1.014 | not fired |
| 5090 | 27B UD-Q4_K_M | 1.108 | not fired |

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
  - `step_profile.py`: per-row categories, verify regions, both ratios with clocks, and matmul calls, weight bytes and
    distinct quantized inputs per step;
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

A candidate survives on a class if it saves at least 1% of its step on some release row there (graphed decode
B1/B4/B8, verify, prefill 512/2048). Shares are the largest over the rows; "—" means it does not apply.

| item | 5080 | 4090 | 5090 | verdict |
|---|---|---|---|---|
| D2 crossover | 27B B8 −5.9%, DFlash −5.0% (iq3_s → 7, iq4_xs → 6) | 0 | B8 −4.1%, DFlash −4.5% (iq4_xs → 6, q6_k → 4) | **keep** on 5080, 5090; **drop** on 4090 (llama.cpp's rule is already optimal there); per-type entries under 1% dropped (§11.5) |
| bf16 `kv_write` | 8B B1 −2.2% (B8 −1.8%) | −1.5% (−1.3%) | −3.5% (−2.8%) | **keep** (8B, bf16-KV rows; the 27B's bf16-KV rows are 0.2%) |
| D3 `qk_rope_kv` (on top) | 8B B1 −1.9% | −1.7% | −2.9% | **keep** for decode; prefill 0.5–1.0% and the 27B 0.2–0.4% (as predicted: ≤ 0.3%) |
| D4 mmq bf16 out | prefill −2.4–3.2% (NVFP4 −5.0%, W4A4 −10.2%) | −4.8–5.1% (NVFP4 −6.0%) | −2.4–2.6% (NVFP4 −4.6%, W4A4 −9.2%); 27B B8 −1.3% | **keep** |
| D5 rewind glue | verify −3.4–4.2% | −3.4–4.1% | −5.3–6.1% | **keep** |
| D6 gate/up + SwiGLU | 8B B1 −1.6%, 27B B1 −1.1% | −1.1%, −0.7% | −2.4%, −1.35% | **keep** |
| D6 shared q8_1 | 27B B1 −1.5% | −0.96% | −1.9% | **keep** on 5080, 5090; **drop** on 4090 (under 1%); 8B 0.2–0.3%, NVFP4 has no repeat |
| D6 GDN gating | prefill −0.3% | −0.4% | −0.4% | **drop** everywhere: about 2 ms of an 838 ms step |
| D7 mixed steps | cap 256 or 512 | cap 512 (256 fails paired TTFT) | cap 512 (256 at 1.12) | **keep**; step 4's real runs pick the cap |
| D8 megakernel | f 15.9% → stop | — | f 25.2% today, 17.5% predicted after D3–D6 | gate re-measured at step 5 |
| D9 small-batch matmul | 1.246 (at the bar) | 1.014 | 1.108 | not triggered |

**Measured, not in §5 (for the lead):**
- **NVFP4 `scale2` after mmvq above one row:** 3.3–7.2% of NVFP4 decode B4/B8 on every class. llama.cpp fuses it into
  mmvq at one row only; doing it above one row would be a vendor patch, as D4 is.
- **Native W4A4's bf16 → f32 activation copy:** 6.7–7.0% of NVFP4 W4A4 prefill on sm_120. A quantizer that reads bf16
  would remove it.
- **The 5090's DFlash2 failure** with bf16 KV (§11.4): a `triton_attn` config for sm_120 at width 8.
- **pp512 @ 8K (0.91–0.96) and the 30B-A3B tg128 (0.93)** are not ggml (§11.9).

### 11.12 For the lead

1. **Re-stated §5 above, for the panel.** Every fusion survives on some class except GDN gating; shared q8_1 and the
   D2 table drop on the 4090.
2. **D2's per-entry rule.** Of the 5080's i-quant entries, only iq3_s and iq4_xs reach 1% alone (3.8% of decode B8
   together). The other five add 2.0% between them, at under 1% each. Ship them as one entry per class ("i-quants →
   6/7") or drop them? I recommend shipping the measured limits for every i-quant of the 27B file. They all come from
   one sweep, and D2's small-step L2 covers them anyway.
3. **D9 sits at its bar on the 5080:** 1.243–1.250 over four sweeps, mean 1.246. By the rule it does not fire. The
   lead decides whether a ratio at the bar is a trigger.
4. **D8 on the 5090 is in prototype territory today** (f 25.2%, 1.16× at c = 0.55). After the surviving fusions it is
   predicted at 17.5% (stop). Step 5 decides after D3–D6, as designed.
5. **The D7 cap per class (simulated):** 512 on the 4090, whose slow host makes cap 256 fail the paired-TTFT bar
   (1.26); 256 or 512 on the 5080 and 5090. The predicted worst-gap gains are 11–28×.
6. **New items** (not in §5): the NVFP4 `scale2` epilogue above one row, the bf16 W4A4 quantizer, the 5090
   DFlash2/bf16-KV failure, and the attention-at-depth gap. Each is a follow-up or another milestone's work, not
   M0.7's.
7. **One desktop lost:** the first RTX 4090 desktop fell off the bus; all evidence comes from screened desktops.
