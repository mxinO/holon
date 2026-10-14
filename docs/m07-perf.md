# M0.7 — performance: fused glue, tuned crossovers, mixed steps, a megakernel experiment

Status: **revision 1, for review** (2026-10-14).

Scope: plan §2 (M0.7: "fused prefill glue (KV-store kernel, add+norm, q/k norm) and a model-specific megakernel
experiment (design §5.7)") plus the follow-ups M0.3–M0.5 recorded as M0.7's: speculation's verify-step cost
(m04-spec §10, §13), the small-batch quantized matmul (plan §6.1), mixed prefill/decode scheduling (M0.5 measured
streams silent up to 36 s behind a long prefill), the forward/sample split (deferred by M0.5's D2 measurement:
1.061× / 1.026×, under its 1.10× rule — stays deferred, D9). Measurements and prior art are in the local research note
`docs/brainstorm/m07-perf-research.md` ("R §n"), taken on an RTX A5000; the consumer classes are re-measured first
(step 0) because the A5000 under-states the glue share and the crossovers move with bandwidth and SM count.

Implementation starts after M0.5 and M0.6 merge into `main` (both change the core and the scheduler, which M0.7's
mixed steps touch). This design is written now so the panel can challenge it in parallel.

## 1. Goal and acceptance

Faster steps on the consumer classes without moving numerics beyond bounded checks. Acceptance:

1. **Crossover tuning** (D2). The mmvq/mmq crossover is a tunable of `kernels.ggml_cuda` per class and type, swept
   by `holon tune`; DFlash2 verify steps on the RTX 4090 and MTP verify on the RTX 5080 get faster by at least half of
   the predicted saving (§5), and no decode or prefill row gets slower (llama-bench-style rows, ±1%).
2. **Fusions** (D3–D6): the fused q/k-norm + RoPE + KV-store op, mmq writing bf16, the fused GDN rewind, gate/up +
   SwiGLU in mmvq, one q8_1 quantization per shared input, and one GDN gating kernel — each with an L0 against its
   composed reference on sm_86 / sm_89 / sm_120a (+ memcheck), L2 passing, bounded checks passing; combined effect
   measured per class against §5's predictions.
3. **Mixed steps** (D7): decode rows ride along with prefill chunks under a chunk cap while decoders run; the agent
   trace's TPOT p99 and largest silent gap per stream improve on the RTX 5080 (27B) and RTX 4090 against today; TTFT
   p50 regresses by at most 15%; bounded (chunked vs one-shot already is) and snapshots run-identical.
4. **Megakernel experiment** (D8): re-measured after 1–3 on the RTX 5090 and 5080 against a fixed go/no-go rule; a
   one-layer prototype only on "go", recorded either way. No core seam for it unless it goes.
5. **Bars and locks:** every lock re-propagated (bounded changes need approval: delegated for passing verdicts); bars vs
   llama.cpp re-measured on the 5080 and 4090; nothing regresses beyond its noise.
6. **Core:** only the seams of §3, toy-proven; ≤ 6000 code lines.
7. **Suite time** does not grow (lean tests): new L0 cases extend existing sweeps.

Non-goals: lower-precision GDN state (a numerics change, R §6 Q4); prefill CUDA graphs for slow hosts (R #11, deferred:
the fusions remove 25% of the 27B's eager launches first); the residual fold in `Model.trunk` (R #10: ≤ 0.2%);
the forward/sample split (D9).

## 2. Starting point (R §1)

- **Glue share** (A5000, batch 1, graphed): non-GEMM 3.8% of a bf16 decode step, 15% on 8B Q4_K_M, 12% on the 27B
  UD-IQ3_S; quantized prefill pays 6–7% for activation quantization and the f32 → bf16 cast after each mmq.
- **Against FreeToken** (8B bf16): 169 / 89 more kernels per prefill / decode step, 0.7–1.8% of prefill, unmeasurable at
  decode; equal wall times. **llama.cpp's fusions** are worth 3–5% (`GGML_CUDA_DISABLE_FUSION=1`), and Holon already
  beats llama.cpp on every A5000 row.
- **Crossover:** llama.cpp runs mmvq up to 8 rows; on the A5000 mmq wins from 5–7 rows and by 14–31% at 8 (the DFlash2
  verify width).
- **Verify steps:** 1.76× a decode step (MTP) and 2.51× (DFlash2) on the A5000; the wider target GEMMs +8 / +23.5 ms,
  the draft model 6–8 ms, the GDN rewind 3–3.7 ms in ≈ 960 small kernels.
- **Scheduling:** prefill and decode never share a step (`sched.py`); a 30K-token prompt silences every stream for its
  whole prefill.
- **Graph timing:** kernels replayed in graphs measured 2–19% slower than eager (clocks? R §6 Q6).

## 3. Core seams (today → change)

1. **Mixed steps.** Today the scheduler builds prefill-only or decode-only steps (`sched.py`). Change: a step may carry
   decode (and verify) rows plus prefill chunks; the chunk budget while any decoder runs is the scheduler tunable
   `sched.chunk_with_decode` (tokens; 0 = today's behaviour); decode rows are charged against `max_step_tokens` first.
   Mixed steps run eagerly (graphs stay decode/verify-only); planners already plan per row kind (GQA prefill vs
   decode kernels, GDN chunked vs recurrent rows), and the batch keeps its row order (decode rows first). Toy proof:
   the toy state kinds and the oracle drafter across mixed steps, chunked prefill, depth 1, verify rows and constrained
   rows (M0.5's gate) — outputs equal today's schedule's on CPU tiny models.
2. **No other core change** unless the megakernel goes (D8: composite binding and a loop override, designed then).

Line budget ≈ +60.

## 4. Decisions

- **D1. Re-measure before building** (step 0). On the RTX 5080, 4090 and 5090: the per-type × rows crossover sweep,
  the glue share, the verify-step breakdown, and the graph-timing question (`nvidia-smi dmon` clocks during graph
  replay vs eager). §5's predictions are re-stated per class after it, before any code; a candidate whose re-measured
  saving falls below 1% of its step is dropped and recorded.
- **D2. The crossover is a tunable of `kernels.ggml_cuda`** per (class, type) — an explicit lock entry (level 1), not
  a startup measurement — swept by `holon tune`. It changes rounding (mmq vs mmvq), so snapshots move: propagate with
  approval, bounded checks pass.
- **D3. Fused q/k-norm + RoPE + KV store is a std op** `attention.qk_rope_kv@1` under `std.mixers.gqa`, used when a
  provider supports the site's facts (every Qwen3 / 3.5 / 3.8 GQA site; general, not a model fast path), with the
  composition as its reference and a Triton provider for the bf16 and q8_0 KV layouts; its first piece is a Triton
  `kv_write` (one launch for k and v).
- **D4. mmq writes bf16** (and its stream-k fixup), through `vendor.patch` (design §5.6 borrowed-kernel patch); one
  rounding less than f32 → bf16, so bounded, not bitwise.
- **D5. The GDN rewind is one Triton launch per verify step** over all GDN layers (the conv state is one pool with a
  layer axis), and the replay buffers are written by the projections, not copied.
- **D6. Gate/up + SwiGLU in mmvq** for single-column same-type gate/up (the vendored `fusion.gate` / `glu_op`), and one
  q8_1 quantization per shared input (`qlinear` takes a pre-quantized activation; the contract gains an optional
  argument, a minor version), and one GDN gating kernel in `kernels.fla`.
- **D7. Mixed steps** (§3.1): the cap is a tunable with default chosen by step 4's measurement (candidates 256 / 512);
  eager first; the graphed decode half split from an eager prefill half is deferred.
- **D8. Megakernel experiment, gated.** After steps 1–3, on the RTX 5090 and 5080: if the 8B Q4_K_M batch-1 decode
  step's non-GEMM + bubble share is under 15%, stop (record). Otherwise build a one-layer prototype in
  `models/qwen3/fast/` and go only if it projects ≥ 1.15× at batch 1, no regression at batch 4–8, and L2 passes. A go
  means a design addendum (core seams: one composite bound to many layers, a decode-loop override; taps written by the
  megakernel so speculation composes), reviewed by this panel before code.
- **D9. The forward/sample split stays deferred** (M0.5 D2 measured 1.061× / 1.026×, under its rule); revisit if mixed
  steps or the fusions change the host's share.
- **D10. The small-batch quantized matmul (R #6) is conditional**: built only if, after D2's tuning, MTP verify on the
  5080 still costs more than 1.5× a decode step or Q4_K_M batch-32 decode trails llama.cpp; k-quants first.

## 5. Predictions (A5000-derived; re-stated per class after step 0)

| Item | Prediction |
|---|---|
| Crossover tuning | DFlash2 verify step −10–16% (4090); MTP verify (width 4) small or none |
| Fused q/k-norm + RoPE + KV | 8B Q4_K_M B1 decode −4.6%, bf16 −1.7%; prefill 2048 −1.5–2% |
| mmq bf16 out | GGUF prefill −4–5% |
| Fused GDN rewind | verify step −2–4% (27B) |
| Gate/up + SwiGLU, single quantization, GDN gating | 8B Q4_K_M B1 −2%; 27B decode −1.3%, prefill −0.6–0.9% |
| Cheap fusions combined | 8B Q4_K_M B1 decode −7–8%; bf16 −2%; GGUF prefill −6–8% |
| Mixed steps | agent TPOT p99 −9–17%; worst silent gap 351 → 51–95 ms; TTFT p50 +0–13% |
| Megakernel ceiling | ≤ 17% (8B Q4_K_M B1, A5000), ≤ 10% after the fusions |

## 6. Judge

- L0 per new kernel vs its composed reference (sm_86 / 89 / 120a), memcheck.
- L2 and bounded on Qwen3-8B (bf16, Q4_K_M) and the 27B for every changed path; plants: a q/k-norm applied after RoPE,
  a KV write to the wrong page, mmq bf16 rounding before the fixup reduction, a rewind mask off by one, a crossover
  tunable that picks mmq for the wrong type.
- Locks: every lock re-propagated per class; bars vs llama.cpp per class.

## 7. Steps

| # | Work | Checks | Days |
|---|---|---|---|
| 0 | Re-measure on 5080 / 4090 / 5090 (D1); predictions re-stated | records | 1 |
| 1 | Crossover tunable + `holon tune` sweep (D2) | L0 unchanged; bars; locks bounded | 1 |
| 2 | `qk_rope_kv@1` + Triton provider; mmq bf16 out; gate/up+SwiGLU; single quantization; GDN gating (D3, D4, D6) | L0 ×3, L2, bounded, bars | 4 |
| 3 | Fused GDN rewind (D5) | L1.5 bitwise vs today's rewind where defined; bounded spec | 2 |
| 4 | Mixed steps (§3.1, D7) | core toy proof; agent trace on 5080 / 4090 | 3 |
| 5 | Megakernel gate (D8); prototype only on "go" | records | 1–5 |
| 6 | (conditional, D10) small-batch quantized matmul | L0, bars | 5 |
| 7 | Close: locks, bars, panel code review, plan status | review clean | 1.5 |

## 8. Risks

- **Crossover tuning moves every GGUF lock's snapshots** (mmq vs mmvq rounding) — approvals per class, bounded.
- **Mixed steps and speculation/grammar interplay** (verify rows and constrained rows beside prefill chunks): covered
  by the toy proof's matrix; eager mixed steps cost graph benefits on those steps.
- **Graph-timing inflation** (R §6 Q6): if replayed kernels run slower because of clocks, kernel-time savings convert
  to less wall time; step 0 measures it.

## 9. Review findings, resolved

(none yet)
