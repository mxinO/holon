# M0.7 — performance: fused glue, mixed steps, a gated megakernel experiment

Status: **revision 2, for review** (2026-10-14). Rev 1 was reviewed by a panel of three; every finding is resolved in
§10.

Scope: plan §2 (M0.7: "fused prefill glue (KV-store kernel, add+norm, q/k norm) and a model-specific megakernel
experiment (design §5.7)"), narrowed and extended by what M0.3–M0.5 recorded for M0.7: the verify-step glue
(m04-spec §13), mixed prefill/decode steps (M0.5 measured streams silent up to 36 s behind a long prefill), and the
mmvq/mmq crossover. The small-batch quantized matmul (plan §6.1) leaves M0.7 as a recorded follow-up with a measured
gate (D9); the forward/sample split stays deferred (M0.5 D2: 1.061× / 1.026×, under its 1.10× rule). Measurements and
prior art are in the local research note `docs/brainstorm/m07-perf-research.md` ("R §n", taken on an RTX A5000) and in
the reviewers' re-derivations recorded in §10; the consumer classes are re-measured first (step 0).

Implementation starts after M0.5 and M0.6 merge into `main` (both change the core, the scheduler and the ggml plugin);
step 0 runs on M0.6's ggml pin (`cbe28bb3`). The plan's M0.7 row is updated with this revision (plan §2).

## 1. Goal and acceptance

Faster steps and shorter stalls on the consumer classes, with every change bitwise where it can be and judged on the
paths it touches. Acceptance:

1. **Fusions shipped only where they pay** (D3–D6): each fusion is kept only if its measured wall saving is ≥ max(0.5%,
   3 sd of its row) on some release-class row (RTX 5080 / 4090 / 5090; graphed decode B1/B4/B8, verify steps, prefill
   512/2048), else reverted and recorded. Bitwise fusions (D4, D5's glue, D6's shared quantization, the bf16 `kv_write`)
   pass bitwise checks inside the existing L0 loops; bounded ones (D3) pass L0, the small-step L2 of §6 and bounded.
2. **Crossover entries shipped only where the sweep shows a gain** (D2): per (class, type, activation mode), only where
   step 0's sweep shows ≥ 1% of a decode or verify step; each shipped entry delivers ≥ half its re-stated saving and no
   guarded row (`holon_tg` batch 2 / 4 / 8, verify steps) regresses beyond its noise.
3. **Mixed steps** (D7): on the agent trace and a long-prompt scenario, the per-request maximum inter-token gap (p50,
   p99, max) improves on the RTX 5080 (27B) and 4090; TTFT p50 and throughput regress by at most max(15%, 2× the A/A
   spread of the same session); TPOT p99 recorded, not gated; a mixed-steps counter > 0 in every run that claims them.
4. **Megakernel experiment** (D8): measured against a fixed speed-of-light gate on the RTX 5090 and 5080 after D3–D6;
   a one-layer prototype only on "go"; recorded either way. No core seam for it unless it goes.
5. **Locks:** propagated once, at the close (bitwise changes keep snapshots byte-identical; D2 / D3 / D7 move them —
   approvals delegated for passing verdicts); the snapshot trace gains a capped chunk beside decoders when D7 ships.
6. **Core:** only the seams of §3, toy-proven; ≤ 6000 code lines. **Suite time** does not grow (lean tests).

Non-goals: mixing verify rows into prefill steps (§3.1; a follow-up with its seam list); the small-batch quantized
matmul (D9); a lower-precision GDN state; prefill graphs for slow hosts; the residual fold in `Model.trunk` and the
"add+norm" fold of the plan row (R #10: ≤ 0.2%, recorded); the legibility benchmark (plan §9, design D9 — before the
release freezes the ABI, not in M0.7).

## 2. Starting point (R §1, corrected by review)

- **Glue share** (A5000, B1, graphed): non-GEMM 3.8% of bf16 decode, 15% on 8B Q4_K_M, 12% on the 27B; quantized prefill
  pays 6–7% for quantization and the f32 → bf16 cast (A5000-specific in part: its 6 MB L2 sends the f32 buffers to DRAM;
  the consumer classes' 64–96 MB L2 keep most on chip — the likely saving there is 1.5–3%).
- **Crossover:** llama.cpp's rule is already tuned per type on the RTX 4090 (Ada: mmvq ≤ 8 for Q4_K/Q5_K/Q6_K) and the
  RTX 5090 (Blackwell, which covers the 5080: Q4_K ≤ 5, Q5_K ≤ 6, Q6_K ≤ 7) (`vendor/.../mmvq.cu:318-346`); the A5000
  finding (mmq faster from 5–7 rows) applies to sm_86 (the rtx-3090 release class) and to untuned i-quants on Ada /
  Blackwell. On the 4090 the 8-column mmvq costs only +3.6 ms on a 32.7 ms verify step (m04-spec §13): the bound is
  ≈ 11%, the expected k-quant saving ≈ 0.
- **Verify steps:** 1.72× a decode step (5080, MTP) and ≈ 1.5× (4090, DFlash2); the GDN rewind ≈ 960 small kernels.
- **Scheduling:** prefill and decode never share a step (`sched.py`); mixing per row kind is not supported today:
  GQA picks one kernel per step (`gqa/units.py:33-51`), GDN one path per step and refuses mixed verify rows
  (`gdn/units.py:57-64`), the executor and verifiers view logits as uniform verify rows (`executor.py:292-296, 354-359`).
- **Graph timing:** the profiler inflates graph-replayed kernel sums by ≈ 2%; the rest of the 2–19% excess appears only
  where eager steps leave the GPU idle (a clock/power effect), and not on the 4090; graphed traces convert ≈ 1:1 to wall.

## 3. Core seams (today → change)

1. **Mixed steps without speculation.**
   - Today: prefill-only or decode-only steps.
   - Change: when the step's k is 0 (no verify rows), a step may carry plain decode rows plus prefill chunks: decode
     rows are charged first, then prefill chunks up to `EngineConfig.chunk_with_decode` tokens (an engine setting
     beside `max_step_tokens`, recorded in locks' `[engine]`; **0 = today's behaviour, the default**), with a minimum
     chunk so decoders cannot starve prefill. "While decoders run" means "this step has a decode row". Rows whose
     host length is inexact (an uncommitted verify row) and constrained rows with a token in flight sit out, as today.
     Speculative deployments alternate a capped prefill step with a graphed decode/verify step (no verify row ever
     mixes). Mixed steps run eagerly (graphs stay uniform); planners get a per-row-kind split: GQA runs its prefill
     kernel over the whole step only if the kernel's facts accept 1-token rows (else the step does not mix), GDN runs
     chunked for prefill rows and recurrent for 1-token rows within the step.
   - The worst-step memory profile runs a mixed step (a chunk's activations alive through sampling).
   - The dead `SchedulePolicy.mix_phases()` protocol is deleted; design §7.1 / §4.2 and plan §10 are updated (mixing is
     a core budget, not a policy plugin, until a second policy exists).
   - Toy proof: the toy state kinds across mixed steps with chunked prefill, depth 1, constrained rows and preemption,
     outputs equal to today's schedule on CPU tiny models; a bounded check "mixed schedule vs today's" on GPU.
2. **No other core change** unless the megakernel goes (D8).

Line budget ≈ +70 (scheduler, executor planning per row kind, profile).

## 4. Decisions

- **D1. Re-measure before building** (step 0, at the ggml pin `cbe28bb3`, on the RTX 5080 / 4090 / 5090): the
  graph-timed per-call crossover sweep per (type, shape in the locks, rows 1–9, activation mode) with and without
  today's cast; the glue share per row (decode B1/B4/B8, verify, prefill 512/2048) including the cast and quantize
  share; the verify-step breakdown; two timing ratios per class (unprofiled graphed wall / eager kernel sum, profiled
  graphed kernel sum / unprofiled wall; clocks investigated only if the first exceeds 1.05, with `nvidia-smi
  --query-gpu=clocks.sm,clocks.mem,power.draw,clocks_event_reasons.active -lms 100`); the speed-of-light fraction of
  decode steps (D8); the mixed-steps simulator recalibrated per class (eager floors from `prefill_idle` evidence,
  checkpoint-limited prefix reuse, checked against the measured plain agent run). §5 is re-stated per class after
  step 0, before any code; a candidate below 1% of its step is dropped and recorded. Step 0 also builds the release
  benchmark runner the close and the release reuse (the per-request max inter-token gap added to `bench/driver.py`'s
  rows; a long-prompt scenario).
- **D2. The crossover is a per-class table inside `kernels.ggml_cuda`** keyed by (class, type, activation mode), with
  llama.cpp's rule as the fallback; the effective value is recorded in each site's `config` (as `triton_attn`'s class
  configs are) — no lock-to-kernel core path. Entries only where D1's sweep shows ≥ 1% of a step; a range check
  (≤ 8, mmvq for every n up to it). Under M0.6's `fp4` mode a lower crossover moves 5–8-row steps from W4A8 to W4A4: a
  precision change, judged by the small-step L2 of §6, not a rounding change. The kernel is chosen by the row count the
  step actually runs (eager real rows, graphed bucket rows) — recorded.
- **D3. Fused q/k-norm + RoPE + KV store** is a std op `attention.qk_rope_kv@1` under `std.mixers.gqa`, used only when
  the injected `qk_norm` and `rope` units answer a new optional `fusion_facts()` (std.ops minor: eps, offset, theta,
  `rotary_dim`, `max_positions`, the gate interleave, the reference identity) and a provider supports those facts —
  never by unit names; a future RoPE variant answers None and keeps the composition. Its reference is built by the
  judge from the composed units' references (unit isolation); the Triton provider covers bf16 and q8_0 KV, rounds to
  bf16 after the norm and after RoPE (before q8_0), reuses the shared `rope.cos_sin` static, handles T = 0; the
  model-only `_causal` build keeps the composition. Bounded (not bitwise vs FlashInfer's rmsnorm reduction). The bf16
  Triton `kv_write@1` provider (one launch for k and v) is separate and bitwise.
- **D4. mmq writes bf16** through `vendor.patch`, keeping the stream-k fixup's partials in f32 until the final single
  rounding: bitwise equal to today's f32 + cast (`torch.equal` inside the existing `test_mmq_schedules`, which forces
  the fixup). Under M0.6, NVFP4's `scale2` multiply moves into the kernel before the rounding.
- **D5. The GDN rewind's glue is fused per step** (masking, window restore, gather / index_copy: ≈ 1.9 of 3.7 ms),
  triggered once from the step's shared `GDNMeta` (no core change); the provider calls (`gated_delta_rule`,
  `causal_conv1d`) stay as locked. L1.5 stays unconditionally bitwise against 1 + n decode calls.
- **D6. Provider-internal fusions in `kernels.ggml_cuda`**, no contract change: gate/up + SwiGLU through the vendored
  mmvq `fusion.gate` / `glu_op` for single-column same-type gate/up (its op and owner: std.ffn's `glu_linear` site,
  any row count handled, the fusion only at one column); a shared q8_1 quantization memoised by (input identity —
  strong reference, `data_ptr`, stride, `_version` — path, mmq ds layout, activation mode) in its own buffer, never the
  workspace shared per `in`; bitwise; one GDN gating kernel in `kernels.fla`.
- **D7. Mixed steps** (§3.1): default 0; each lock opts in through `[engine]` with a cap chosen per class by step 4's
  measurement on the recalibrated simulator and real runs (candidates 256 / 512), long-prompt TTFT p99 bounded.
- **D8. Megakernel experiment, gated on speed of light.** After D3–D6, per class (5090, 5080): f = 1 − (Σ weight + KV
  bytes) / (measured stream bandwidth × step time) on 8B Q4_K_M B1 decode; the projected gain 1 / (1 − f·c) with c the
  capture fraction (0.6–0.7 from the published megakernels). Stop if the projection is under 1.15× (f·c < 13%);
  otherwise build a one-layer prototype in `models/qwen3/fast/` and go only if its L0 against the judge-built recipe
  reference passes and it projects ≥ 1.15× at B1 with no regression at B4–8. A go means a design addendum (core seams:
  one composite bound to many layers, a decode-loop override, taps written by the megakernel), reviewed by this panel.
- **D9. The small-batch quantized matmul leaves M0.7** as a follow-up gated by step 0's per-call sweep: the target's
  4-row / 1-row GEMM time ratio after D2 > 1.25 on a release class (it would be a row-dispatch path inside
  `kernels.ggml_cuda` with its own `config["act"]` entry, not a new provider).
- **D10. Classify every change** bitwise or bounded before building; target bitwise; propagate once at the close.

## 5. Predictions (provisional; re-stated per class after step 0)

| Item | Prediction |
|---|---|
| Crossover (D2) | 4090 k-quants ≈ 0 (bound 3.6 ms per DFlash verify step); 5080 already tuned; gains possible on sm_86 (rtx-3090) and i-quants (UD files) — sized by step 0 |
| `qk_rope_kv` (D3) | 8B Q4_K_M B1 decode −4.6% (A5000), bf16 −1.7%; prefill 2048 −1.5–2% |
| mmq bf16 out (D4) | GGUF prefill −1.5–3% on the consumer classes (−4–5% on the A5000) |
| GDN rewind glue (D5) | verify step −2–3% (27B) |
| D6 fusions | 8B Q4_K_M B1 −2% (gate/up+SwiGLU); 27B decode −1% (shared quantization, decode only); prefill −0.6% (gating) |
| D3–D6 combined | 8B Q4_K_M B1 decode −6–7%; bf16 −2%; GGUF prefill −3–5% |
| Mixed steps (D7) | worst gap: 4090 511 → 97–154 ms, 5080 710–974 → 120–220 ms; TPOT p99 within noise; TTFT p50 +4–49% at cap 256 (cap 512 smaller) |
| Megakernel | ceiling ≤ 17% at B1 (A5000), lower after D3–D6; decision by D8's projection |

## 6. Judge

- **Bitwise checks inside existing L0 loops:** D4 (`test_mmq_schedules`, fixup forced), D5 (L1.5), D6 shared
  quantization (a mixed-type pair of consumers), the bf16 `kv_write`.
- **D3:** L0 vs the judge-composed reference (sm_86 / 89 / 120a + memcheck) incl. partial rotary and the gated q layout.
- **Small-step L2 for the paths propagate never runs** (it scores prefill only and bounded runs batch 48): `holon_judge
  l2 --max-step-tokens 1` against the `*-mmvq-floor` goldens for single-column paths (D6's gate/up), `--max-step-tokens
  8` and 4 against the floor of the path a tuned crossover takes (D2), and bounded with a small `max_seqs` for decode.
- **Plants:** a KV slot off by one only at page boundaries (D3); a RoPE position off by one on decode rows only (D3;
  bounded catches it); mmq's fixup rounding the partial to bf16 (D4; fails the bitwise check). Re-verified with a
  "fired" count > 0: `ggml_stale_q8_1` (after D6), `replay_off_by_one` and `verify_writes_state` (after D5).
- **Locks:** propagated once at the close; the snapshot trace gains a capped chunk beside decoders when D7 ships.

## 7. Steps

| # | Work | Checks | Days |
|---|---|---|---|
| 0 | Re-measure on 5080 / 4090 / 5090 (D1); the benchmark runner with max-gap rows; §5 re-stated | records | 2 |
| 1 | D2 table entries where the sweep pays | guarded rows; small-step L2 | 1 |
| 2 | D4, D6, bf16 `kv_write` (bitwise), D3 (bounded) | L0 ×3 (bitwise in-loop), L2, bounded; keep-or-revert per §1.1 | 4 |
| 3 | D5 glue | L1.5 bitwise; bounded spec | 1.5 |
| 4 | Mixed steps (§3.1, D7) | core toy proof; bounded mixed-vs-today; agent + long prompt on 5080 / 4090 with A/A | 3 |
| 5 | Megakernel gate (D8); prototype only on "go" | records | 1–5 |
| 6 | Close: locks once, bars, panel code review, plan status | review clean | 1.5 |

## 8. Memory and suite time

No new persistent buffers beyond D6's quantization buffer (one per (input, layout), bounded by the step's largest
activation) and the mixed step's activations (in the worst-step profile). Lean tests: bitwise checks live in existing
L0 loops; two new plants replace five proposed.

## 9. Risks

- **Mixed steps and speculative deployments:** alternating capped prefill with decode/verify steps keeps every
  speculative invariant but helps speculative deployments less; mixing verify rows is the follow-up (its seam list:
  GQA / GDN per-row-kind planners and rewind, the executor's and verifiers' uniform-width views, both drafters,
  the scheduler's single k, a device-length prefill kernel).
- **Approvals:** D2 (where shipped), D3 and D7 move snapshots; one propagation at the close keeps it to one approval
  round per class (plan §9: > 1 approval per day is a trigger).
- **The recalibrated simulator** still models, not measures: step 4's real runs decide.

## 10. Review findings, resolved

**Rev 1** (A: correctness, B: architecture, C: numerics / performance / tests).

| Finding | Resolution |
|---|---|
| A-B1 / B-B1: verify rows cannot mix (GDN refuses, rewind widths, uniform logits views, drafters, the scheduler's k, FlashInfer host lengths) | mixed steps only when k = 0; inexact and in-flight constrained rows sit out; speculative deployments alternate; verify mixing a follow-up with its seam list (§3.1, §9) |
| B-B2: the mixing knob is no tunable; default; dead protocol; design text | `EngineConfig.chunk_with_decode`, default 0, locks opt in; `mix_phases()` deleted; design §7.1 / §4.2, plan §10 updated (§3.1) |
| B-B3 / C-H1: llama.cpp's crossover is already tuned on Ada / Blackwell; no lock-to-kernel path; per-model duplication | a per-class table in the ggml plugin, entries only where step 0 shows ≥ 1%; predictions restated (≈ 0 on 4090 k-quants) (D2, §5) |
| A-M3: the crossover vs M0.6's activation modes | keyed by activation mode; a precision change judged by small-step L2 (D2) |
| B-B4 / A-M2 / C-M2: a pre-quantized activation leaks the format; layouts differ by type; shared workspace | provider-internal memo keyed by (input, path, ds layout, mode) in its own buffer; no contract change; bitwise; prefill saving restated (D6) |
| B-B5 / A-m1: the fused op needs a seam between units; reference under unit isolation; parameters | `fusion_facts()` on the injected units; judge-built reference; the parameter list; T = 0; `_causal` composition; bounded (D3) |
| B-B6 / C-M7: the megakernel gate measures the wrong quantity; "L2 passes" uncheckable | speed-of-light f, projection 1 / (1 − f·c), c 0.6–0.7, stop under 1.15×; L0 vs the recipe reference for "go" (D8) |
| B-B7 / C-M6: D10 malformed, fires regardless | the small-batch matmul leaves M0.7, gated by the 4-row / 1-row GEMM ratio (D9) |
| B-B8 / A-M1 / C-M1: churn; D4 can be bitwise; fixup double rounding | f32 fixup partials, bitwise `torch.equal`; classify bitwise vs bounded; propagate once (D4, D10, §1.5) |
| A-M4 / B-B9: a cross-layer rewind bypasses locked providers; `out=` | glue fused per step from `GDNMeta`; provider calls kept; L1.5 bitwise (D5) |
| C-H2: mixed-steps bars inside the noise; simulator calibrated where mixing is free; no gap metric | gate on max inter-token gap (driver rows gain it), TTFT / throughput bars vs A/A spread, TPOT p99 recorded only; simulator recalibrated per class (D1, D7, §1.3) |
| C-H3: propagate's L2 never runs the changed small-row paths | small-step L2 (`--max-step-tokens 1 / 4 / 8`) and small-`max_seqs` bounded (§6) |
| C-M3: plants too gross or not bugs; replaced targets | two subtle plants; three existing plants re-verified with fired counts (§6) |
| C-M4: graph timing mostly answered | two ratios per class, clocks only if > 1.05 (D1) |
| C-M5: mmq-bf16 saving A5000-specific; crossover with the cast | re-stated 1.5–3%; the sweep times calls as dispatched, re-run after D4 (§5, D1) |
| C-L1–L4, A-m2–m6, B-B10–B12: keep-or-revert rule; guarded rows; non-goal in the combined figure; tune's resolution; mixed counter; partial rotary; min chunk; profile; "decoders run"; fla β; q8_0 1-token rows; padding vs crossover; benchmark runner; D9 naming; plan row; long-prompt TTFT | §1.1 keep-or-revert; §1.2 guarded rows; §5 combined without R #10; D1 per-call sweep; §1.3 counter; D3 cases; §3.1 minimum chunk, profile, definition; GDN per-row path stated; L0 case for q8_0 1-token prefill rows (step 2); D2 row-count note; D1 runner; the forward/sample split no longer numbered D9 here; plan row updated; D7 long-prompt TTFT p99 |
