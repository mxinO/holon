# Holon — Implementation plan

Status: rev 3 (after an adversarial review) · Date: 2026-10-06 ·
Companion to [design.md](design.md)

---

## 1. Environment and hardware

| Role | Hardware | Use |
|---|---|---|
| Development | RTX A5000 24 GB GPUs (**sm_86**); CUDA 13.0, Python 3.12, torch 2.11.0+cu130, Triton 3.6, uv | coding, functional tests, golden recording (CPU fp32 fits in RAM) |
| Consumer evidence | consumer desktop machines (1 GPU each): **RTX 5080 16 GB** and **RTX 5090 32 GB** (sm_120), **RTX 4090 24 GB** (sm_89), RTX 3090 (sm_86) | all performance numbers, tuned defaults per fingerprint class, consumer memory-fit checks |
| Blackwell dev (optional) | workstation with RTX PRO 6000 Blackwell Max-Q 96 GB (sm_120) | sm_120 kernel development and debugging (S1–S3 on Blackwell), large-memory work (GPU golden recording, large models), NVFP4 later; not a consumer-performance proxy |
| Datacenter (later) | H100/H200, GB200/GB300 | v2 datacenter path (TP/EP/PD, NVFP4) |

**Working method:** on a shared test machine, start one persistent container and run builds, tests and benchmarks
inside it repeatedly rather than as many short batch jobs. The A5000 host (16 memory channels, server CPUs) is not a
faithful consumer proxy — CPU-side and PCIe-bound tunables are only measured on consumer desktops.

**Quantization priority (D9):** for consumer and edge GPUs the formats that matter are llama.cpp's GGUF sub-4-bit
types (k-quants Q2_K–Q6_K, i-quants IQ1–IQ4, e.g. `unsloth/Qwen3.8-27B-GGUF`); they are v0. HF-ecosystem formats
(GPTQ/AWQ INT4) follow in v1. FP8, NVFP4 and MXFP4 checkpoints (dense and MoE) have their own milestone, M0.6,
developed on Blackwell (RTX PRO 6000, sm_120: native FP8 and FP4 tensor cores) with FP8 also on sm_89.

Runtime dependencies (v0; declared per distribution, pinned in `uv.lock`, release design D1): the core: `torch`
(2.11, CUDA 13), `numpy`, `msgspec`, `pyzmq`; std: `gguf` (the GGUF reader), `safetensors`, `huggingface-hub` (hub
references), **`transformers`** (tokenizers and chat templates, the GGUF tokenizer included; a released version,
5.6.2); plugins: `flashinfer-python` (JIT modules prebuilt by `holon prebuild`), `triton`, `xgrammar`,
`fastapi` / `uvicorn`, `transformers`. Vendored, as our own plugins: GDN ops from vLLM's
`third_party/flash_linear_attention` (subset, Apache-2.0 / MIT, `vllm.*` imports stripped; **not** pip `fla-core`)
and llama.cpp's ggml CUDA kernels (dequant, mmvq, mmq, MoE; MIT, built as our own Torch extension). Judge and tests
only (the `dev` group): `accelerate`, `pandas` / `pyarrow`, `compressed-tensors`, `scipy`, `jsonschema`, `pytest`;
golden recording runs transformers in an environment **without** fla. Baselines in separate pinned environments
under `bench/` (`bench/README.md`): llama.cpp (`llama-bench`, `llama-server`), SGLang and FreeToken for the release
suite; mini-sglang and nano-vllm (bf16 paths) from M0.

**Implementation policy (D6):** references are studied, not copied. Each work item names what it learns from;
the code is our own plugin-native implementation. Only kernels may be borrowed, under design §5.6.

## 2. Milestone ladder (v0)

| Milestone | Model | Seam proven | Key plugins added |
|---|---|---|---|
| M0 | Qwen3-8B bf16 | core end to end: overlap, graphs, prefix cache, device SeqTable, MemOps/quiesce, locks, snapshots, judge L0–L2 | qwen3, gqa, paged_kv, flashinfer attn, std ops, sampler, FCFS, openai frontend |
| M0.1 | Qwen3-8B GGUF (Q8_0, Q4_K_M, IQ4_XS, IQ3_XXS) | quant pipeline: GGUF weight source, ggml dialect, borrowed ggml kernels chosen per class; first 16 GB consumer data point (RTX 5080) | gguf weight source, ggml dialect, ggml_cuda kernels |
| M0.2 | Qwen3-30B-A3B GGUF (UD-IQ3_XXS 12.0 GiB, Q4_K_M; UD-Q3_K_XL optional) + bf16 HF (correctness, RTX PRO 6000) | model isolation; format-general MoE unit and ops; a 30B MoE on a 16 GB card | qwen3_moe recipe, std.moe, GGUF MoE kernels, stacked loading, a Triton grouped GEMM |
| M0.3 | Qwen3.8-27B: GGUF (UD-IQ3_S 11.2 GiB on 16 GB, UD-Q4_K_M) + bf16 HF (correctness, RTX PRO 6000) | heterogeneous state, publish/adopt MemOps, checkpoint prefix reuse, quantized KV cache (q8_0 ours, fp8 FlashInfer) | qwen3_5 recipe, gdn mixer + kernels, recurrent state kind, checkpoint policies, quantized-KV layouts |
| M0.4 | Qwen3.5 / Qwen3.8 (built-in MTP, DFlash2 draft) + Qwen3-8B | spec decoding at depth 1 on the device SeqTable; taps, auxiliary layers, Rewindable incl. GDN replay; verify graphs | mtp, dflash and gpu n-gram drafters, greedy + rejection verifiers, fixed-k policy (`docs/m04-spec.md`) |
| M0.5 | Qwen3-8B, Qwen3.8-27B | logits-processor lifecycle, host-token depth gate, drafts to the host, per-request errors, resume (`docs/m05-grammar.md` §3; the forward/sample split only if measured necessary) | xgrammar processor, parsers, one HTTP frontend with OpenAI, Anthropic and Responses routers |
| M0.6 | Qwen3-8B / Qwen3-30B-A3B in FP8 (block-scaled), NVFP4 and MXFP4 | low-precision HF checkpoints, dense and MoE: weight-source dialects over `QTensor` parts, `qlinear` / `linear_experts` providers, judge tiers per format; RTX PRO 6000 (sm_120), FP8 also on sm_89 | fp8 / nvfp4 / mxfp4 dialects and kernels |
| M0.7 | Qwen3-8B, Qwen3.8-27B GGUF | fused glue (q/k-norm + RoPE + KV store, bf16 KV store, mmq bf16 output, GDN rewind glue, gate/up + SwiGLU), the mmvq/mmq crossover per class, mixed prefill/decode steps without speculation, and a megakernel experiment gated on speed of light (design §5.7; `docs/m07-perf.md`) | fused kernels in std and the ggml plugin; an engine budget for mixed steps |
| Release | every v0 model | packaging, quickstart, rtx-5090 / rtx-3090 locks, an agent demo; the benchmark suite (decode, prefill, agent trace vs llama.cpp, FreeToken and SGLang where each runs) on the release classes | — |

Order (user, 2026-10-13): M0.4 → M0.5 (whole: its grammar seam builds on verify); M0.6 runs in parallel, in its own
worktree; then M0.7 and the release, which needs the benchmark suite first.

**M0.6 status (2026-10-14): Gate A met** (`docs/m06-lowp.md` §8; panel-clean at its close). FP4 (NVFP4, MXFP4) HF and
GGUF checkpoints load through `qlinear@2` / `linear_experts@2` at W4A8 by default. Tier A, Tier B and the plants pass
as recorded. Bars run 0.99–1.05× of llama-bench on the same bytes. Four new locks (rtx-5080 8B NVFP4; rtx-4090 8B and 30B NVFP4,
MXFP4_MOE GGUF), with every existing snapshot byte-identical. W4A4 (`quant.act = a4`) stays opt-in: its pre-registered default rule is
not met (KL p99 42–145× the W4A8 floor's, for ≈ 2.1× prefill speed), and acceptance 7 is met with a recorded deviation.
Gate B (FP8 on `kernels.triton_fp8`) is off the v0 critical path (release design §9); M0.7 starts on Gate A. **Gate B met** (2026-10-15): its FP8 locks
(acceptance 15) came with M0.7's closing propagation (§8a.1).

Expert caching/offload, KV offload, VMM and the optional self-improvement layer follow in v1 (§10); the M0 seams they
build on and the additive core seams they still need are listed in design D8.

**Acceptance vocabulary** (design §7.10): **run-identical** = byte-identical run to run on the model's snapshot
trace (fixed multi-Seq composition covering all graph buckets, chunked prefill, depth 1); **bounded** = within judge
L2 budgets. Every criterion names its mode.

**Memory fit** is computed: weights + KV bytes/token × tokens + recurrent state × Seqs + ~1.5 GiB overhead.

| Model | Weights | KV / token | A5000 / 4090 (24 GB) | RTX 5090 (32 GB) | RTX 5080 (16 GB) |
|---|---|---|---|---|---|
| Qwen3-8B bf16 | 15.26 GiB | 144 KiB | 32K tokens ≈ 21.3 GiB ✓ | ✓ (more KV) | ✗ |
| Qwen3-8B GGUF Q4_K_M | 4.7 GiB | 144 KiB | ✓ | ✓ | 64K ≈ 15.2 GiB ✓ |
| Qwen3-30B-A3B GGUF UD-IQ3_XXS | 12.0 GiB | 96 KiB | ✓ (Q4_K_M 17.3 GiB) | ✓ | 16K ≈ 15.0 GiB ✓ (more with q8_0 KV) |
| Qwen3.5-4B bf16 text + MTP | ≈ 8.5 GiB | full-attn layers only | ✓ | ✓ | ✓ (not measured: Qwen3.5 is not an M0.3 target) |
| Qwen3.5-9B bf16 text + MTP | 17.13 GiB | full-attn layers only | ≈ 21–22 GiB ✓ | ✓ | ✗ (GGUF builds ✓) |
| Qwen3.8-27B GGUF UD-IQ3_S / UD-Q4_K_M | 11.2 / 15.3 GiB | 64 KiB (16 full-attn layers; ≈ 34 KiB q8_0) + GDN state ≈ 147 MiB/Seq (+ 147 MiB per prefix checkpoint) | ✓ (UD-Q4_K_M) | ✓ | UD-IQ3_S, 2 Seqs, 2 checkpoints: ≈ 23K tokens bf16 KV / ≈ 43K q8_0 (estimate, `docs/m03-qwen38.md` §4.8) |

Global acceptance for every milestone ≥ M0.1:
- Every earlier model's lock is byte-unchanged, or advanced by `holon propagate` (auto-advance on a
  run-identical snapshot, or one approval) — design §4.6.
- New plugins pass judge L0–L2 (+ L0-dist for samplers/verifiers/processors, + L1.5 where state is involved).
- Memory fit met; perf numbers recorded on the relevant consumer classes; CI lints green.

**Size budget** (first-party, excluding tests and vendored kernels; ±30%):

| Part | Estimate |
|---|---|
| Core runtime + plugin API | 3.5–5k (cap 6k) |
| std units (ops, gqa, paged KV, sampling, blocks, ffn, moe) incl. `reference.py` | 2–2.5k |
| Model recipes + Qwen bundles (templates, reasoning/tool parsers) | ~0.9k |
| Own kernels + kernel adapters, GGUF reader + ggml dialect, i-quant batched kernels, quantized-KV attention (ggml CUDA kernels vendored, not counted) | 2.5–3.5k |
| GDN mixer/state/policies, spec decoding, grammar | ~2.1k |
| Frontends (OpenAI chat/completions, Responses, Anthropic) | ~1.5k |
| Judge harness (L0, L0-dist, L1, L2 recorder/comparator, snapshot traces, oracle provenance) | ~2–2.5k |
| Propagation + tune tooling, bench (trace replay, ABAB, remote runners), CI lints | ~1.5–2k |
| Supervisor (v0 part: lifecycle, crash attribution) | 0.5–0.8k |
| **v0 total** | **≈ 16–20k** (tests typically +0.5–1×; vendored kernels separate) |

---

## 3. M0 — core end to end (Qwen3-8B bf16)

### 3.1 Spikes (first, time-boxed, results recorded as evidence)

| Spike | Question | Exit |
|---|---|---|
| S1 Attention | FlashInfer paged prefill/decode on sm_86/sm_89/sm_120 + cu130 + torch 2.11, head_dim 128/256; JIT pre-build; graph capture; run-to-run byte identity; an attention path that reads lengths **on the device** from the SeqTable (for spec Seqs at depth 1) | names, per fingerprint class, the non-spec kernel (FlashInfer with host `plan()` in `plan_bounds`), the spec-at-depth-1 kernel (FlashInfer device-length path or our own Triton paged attention), and which KV layouts each kernel supports (bf16, fp8; q8_0/q4_0 need our own kernel) |
| S2 GGUF kernels | build the ggml-derived CUDA kernels (dequant, mmvq, mmq, MoE) standalone as our own Torch extension, wrappers on Ledger buffers, on sm_86/sm_89/sm_120; correctness vs gguf-py for every type; speed vs llama.cpp on the same card measured **separately for decode (1 and many Seqs) and prefill, dense and MoE, per type** (IQ3_XXS, IQ3_S, IQ4_XS, Q3_K, Q4_K, Q8_0); graph capture | per type and path: adopt, port llama.cpp's newer i-quant `mmq` / MoE `mmq`, or write our own Triton kernel — with the effort sized for M0.1/M0.2 |
| S3 GDN | vendored chunk + recurrent kernels on sm_86/sm_120 vs HF torch reference; no `vllm` imports | within budget, no NaN at T=1..8192, run-to-run identical |
| S4 Determinism | every production kernel candidate is run-to-run deterministic with pinned configs (no split-K/atomics); which sites must be `uncovered` | list of deterministic kernels + uncovered sites |
| S5 Golden | HF fp32 recording on CPU (or `device_map` over GPUs) for 8B; noise floor HF bf16 vs fp32 on frozen prompts | τ and KL/ΔPPL budgets calibrated |
| S6 CPU handshake | stream-memop flags (`cuStreamWriteValue`/`WaitValue`) + persistent native CPU worker inside a FULL graph; `quiesce()` + double-buffered resolver map under depth 1 (learning from FreeToken's `cpu_executor`) | per-step overhead measured on a desktop host; seam confirmed before the ABI freezes |

**Spike outcomes (2026-10-07; details in `spikes/s*/README.md`, design §5.6, §7.4, §10.3):**

| Spike | Outcome |
|---|---|
| S1 | FlashInfer 0.6.12: FA2 paged prefill + tensor-core decode (host `plan()`, two wrappers alternated per geometry at depth ≥ 1); own Triton paged attention for spec-at-depth-1 and q8_0 KV (≈ parity with FlashInfer); CUDA-core decode and the no-split graph wrapper rejected; all paths run-to-run deterministic |
| S2 | Port llama.cpp's current ggml-cuda quantized kernels via a ≈ 200-line shim (≈ 5 days, M0.1–M0.2); SGLang set rejected (IQ3_S grid bug; 0.05–0.8× llama.cpp off single-row decode); 10 ggml types needed; llama-bench baselines recorded |
| S3 | Vendored GDN subset (vLLM `6dd813e0`, ≈ 2.75k code lines) passes on sm_86/sm_120: error ≈ HF bf16, no NaN to 8K, resume at 64-multiples and verify bitwise exact; `v_head_tiled` switch for GGUF; GDN prefill needs graph capture (≈ 0.5 ms/layer eager launch overhead) |
| S4 | Folded into S1–S3: every adopted kernel is run-to-run deterministic with pinned configs; none is batch-invariant (as designed) |
| S5 | Top-20 fp32 CPU golden (Qwen3-8B: 4–8 min, 49 GB RAM, 5.4 MB); prompt set v1 (24 prompts, 4 categories); budgets per design §10.3 with KL p99 as the discriminating metric; injected RoPE bug caught, legit variants pass |
| S6 | Handshake seam sound: spin-kernel wait 2.8 µs / memop 9.7 µs / host func 31.7 µs per handshake; zero-copy pinned I/O; overlap works; 1.5 M handshakes without deadlock; stream-ordered map swaps need no quiesce |

Consumer evidence (RTX 4090 / 5080 / 5090 desktops, 2026-10-06; "Consumer cards" in the spike READMEs):
S1 decisions hold (Triton ≈ FlashInfer, median 0.99–1.01× time; configs pinned per class; q8_0 KV tuning still
open on sm_120). S2: all 10 types pass on sm_89/sm_120; dequant + cuBLAS reaches only 0.44–0.73× llama.cpp prefill
on GeForce (int8 ≈ 4× bf16 throughput), so the int8 `mmq` port comes first in M0.1; llama-bench baselines recorded
per class. S6: spin-kernel wait by default (1.9–2.8 µs), memop fallback for watchdog/laptop GPUs; CPU workers =
physical cores − 1 with the Python loop pinned to the free core.

### 3.2 Work items

**A. Repo bootstrap** — uv workspace (`core/`, `std/*`, `plugins/*`, `judge/`, `supervisor/`, `bench/`, `plans/`);
CI scripts (§3.5); `AGENTS.md` (no claims without measurement; failing-then-passing test per bug fix; never edit
`judge/` or locks by hand); a pinned container image for builds/tests and scripts to run it on a shared
test machine (one persistent container, repeated runs inside — D7).

**B. `holon.api`** — dataclasses/Protocols only: Manifest schema, resources, slots + cardinality, Plan + lock
format (`recipe`, `[units]` src/sem, `[deploy]`, `[snapshot]`), recipe terms (`Ref`, `ModelSpec` with
`block`/`weights`) + canonical hash, `Plugin.register(regs)`, ParamSpec, `finalize_weights`, Op/composite markers,
KernelSpec/Constraint, Scheme/Layout/QTensor/Dialect/converter, StateSpec/StateKind (incl. `fork`, `HitLen`,
MemOp-returning publish/adopt), page groups, Ledger/Pool, PrefixView + eviction contract, Seq/SeqTable/
SamplingParams, BatchPlan (bounds) / ShapeKey / DeviceBatch, Planner (`plan_bounds`, `prepare_device`,
`needs_host_lens`), MemOp, Rewindable, LogitsProcessor, Drafter/DraftPolicy/Verifier/PlanTransform,
FrontendBundle, Event, Platform (incl. stream memops), launcher slot.

**C. `holon.core` runtime** (our own implementation; learns from mini-sglang's overlap loop and graph runner,
vLLM Model Runner V2's device-side state, nano-vllm's minimal loop)
1. `plugins/`: manifest loading, discovery, resource matching, resolver → Plan (recipe hash, per-unit src/sem, one
   version per unit), lock history read/write/diff, fingerprint classes + nearest-class `unverified` fallback, lazy
   import, Protocol checks, hard errors.
2. `platform/cuda.py`: streams, events, pinned alloc, graph capture/replay, stream memops, fingerprint class,
   single-rank launcher.
3. `memory/`: Ledger (fixed mode; startup profiling; budget split incl. page groups), Pool, slot allocator.
4. `prefix/`: page-granular hash-chain PrefixIndex, namespace, fixed-point hit loop with `HitLen`, joint chain
   eviction (builtin LRU).
5. `sched/`: Seq lifecycle, token-budget mechanism, FCFS builtin that never mixes prefill and decode,
   preemption by recompute, per-Seq depth gating (`needs_host_lens`, depth tolerance).
6. `exec/`: device **SeqTable**; Batch builder (SoA bounds → static buffers via `prepare_device`); StepLoop with
   depth 0/1; ordered MemOps with pins; `quiesce()`; device Rewindable commit inside the sample graph; deferred
   frees; ShapeKey bucketing; graph manager (FULL decode/verify, EAGER prefill).
7. `control.py`: abort, pause/drain/resume, flush_prefix (all through `quiesce`).
8. `telemetry.py`: Event ring buffer, CUDA-event phase timers (diagnostic).
9. `ipc.py`: frontend ↔ engine (zmq + msgspec structs as the language-neutral schema), token-id protocol.

**D. std units** (each numeric unit ships `reference.py`)
- `std.ops`: linear (bf16), rmsnorm, add_rmsnorm composite, rope (neox), silu_mul, embedding, lm_head.
  (Knobs `rmsnorm.offset` and `rope.rotary_dim` exist; M0.3 uses them for Qwen3.5/3.8.)
- `std.blocks.prenorm` + `hf_weights`; `std.ffn.swiglu`; `std.mixers.gqa` (Planner with `plan_bounds` /
  `prepare_device`; `output_gate` knob added in M0.3); `std.state.paged_kv` (truncate, fork); `std.sampling`
  (greedy/temperature/top-k/top-p/min-p, n>1, logprobs incl. prompt logprobs; seeded per (Seq, position)).

**E. plugins** — `models/qwen3` (`recipe.py` ≤ 80 LOC + `bundle/`); `kernels/flashinfer` (prefill + non-spec decode)
and `kernels/triton_attn` (spec-at-depth-1, quantized KV) per S1; `frontends/openai`.

**F. judge harness** (separate process, client of the engine) — L0 + `budget(dtype, K)`; L0-dist for the sampler;
L1 contracts (incl. run-to-run determinism of bound kernels); L2 recorder (CPU fp32 / multi-GPU; quantized models
via `dequant_ref` into the dense HF class) + comparator with calibrated budgets; snapshot-trace recorder/comparator;
oracle provenance (`reference.py` at reviewed commits).

**G. bench** — trace replay (ShareGPT-like + **mixed agent multi-turn tool traces**), TTFT/TPOT/throughput,
interleaved ABAB, client-side timing; runs inside the persistent container on consumer desktops; baselines
in pinned environments; container image digest recorded in evidence.

**H. propagation tooling** (outside core, ~400 LOC; changed-line coverage captured during snapshot traces) — `holon impact`, `holon propagate` (3 outcomes,
design §4.6), `propagate --new-model`, `holon tune` (benchmarks tunables/kernel configs per fingerprint class
into the lock); CI `isolation-check` re-runs propagate on a trusted GPU runner (A5000) and on consumer-desktop
runners for the maintained consumer classes: a scheduled, renewable allocation per class running the
pinned container (on shared-change PRs where capacity allows, nightly at least).

**I. supervisor (v0 part)** (~0.5–0.8k) — engine lifecycle on the newest lock, health checks, restart on the same
lock, crash attribution with poison-request quarantine, rate-limited resubmission from committed tokens, rollback to
the previous lock after repeated non-attributed crashes, restart-time measurement.

### 3.3 Order

A → B → spikes S1–S6 (parallel) → C1–C3 → D + E eager, depth 0, no prefix → F (L0/L2) green → C4 prefix →
C6 depth 1 + SeqTable + MemOps/quiesce → graphs → frontend + IPC → G bench (A5000 + consumer desktops) → perf work →
H (propagate, tune) → I (supervisor) → tuned default locks for each class.

### 3.4 Acceptance

1. L2 bounded vs HF fp32 golden within calibrated budgets.
2. Decode throughput within 10% of mini-sglang at batch 1/8/64 **and** TTFT/TPOT within 10% on the mixed agent
   trace — on A5000 and RTX 5090 (same model, same GPU).
3. Engine CPU time per step at B=256 ≤ nano-vllm's; if not met, move per-token input prep to GPU kernels before M0.1.
4. Snapshot trace run-identical across 3 runs; prefix cache on vs off and depth 1 vs 0 bounded; tokens_saved > 0
   on the multi-turn trace.
5. SeqTable seam: a test drafter proposing fixed tokens runs at depth 1 with correct positions/lengths (bounded vs
   depth 0) — before any real drafter exists.
6. Robustness: restart time measured and an SLO set; a poison-request test is quarantined without rollback;
   `n > 1` and prompt logprobs served.
7. Memory fit per §2; tuned default locks for `rtx-a5000`, `rtx-4090`, `rtx-5090` (and `rtx-5080` from M0.1).
8. Core LOC ≤ 6000; lints green.

### 3.5 CI checks (from M0)

`lint-core-denylist`, `import-contracts`, `manifest-deps`, `core-loc-cap`, `no-env-config`,
`no-torch-cuda-outside-platform`, `no-arch-branches`, `reference-provenance`, `isolation-check` (propagate),
`core-diff` (informational in v0, enforcing from v1), `unit`, `judge-L0`, `judge-L2-goldens` (GPU runner),
`snapshot` (GPU runner).

### 3.6 Status (2026-10-08): acceptance met

Measured on Qwen3-8B bf16 unless noted, with the final M0 core (ee12f42) unless noted; records in `evidence/` and
`judge/results/`, locks in `plans/`.

| # | Result |
|---|---|
| 1 | L2 passes for Qwen3-0.6B and Qwen3-8B (`judge/results/l2-ee12f42-*`), and for Qwen3-8B on rtx-a5000, rtx-4090 and rtx-5090 when their locks were written |
| 2 | Decode tok/s Holon / mini-sglang — A5000: B1 40.6 / 41.5, B8 310.5 / 315.0, B64 2033.8 / 2041.5; RTX 5090: B1 98.6 / 97.7, B8 708.3 / 727.3, B64 4776.6 / 4124.2. Agent trace TTFT / TPOT p50 — A5000: 111.9 / 28.4 ms vs 107.6 / 27.5 ms; RTX 5090: 35.4 / 11.2 ms vs 37.0 / 10.9 ms. All within 10%. (On sm_120 mini-sglang runs with a one-line patch that auto-selects FlashInfer attention, as it does on sm_8x.) |
| 3 | CPU per step at B256 (A5000, blocking sync, Qwen3-0.6B): Holon 2.45 ms median / 2.60 ms mean vs nano-vllm 2.94 / 2.84 (both rounds below on both). The mean first came out at 4.0 ms: page-boundary steps with a full pool evicted by copying the whole prefix index per request; fixed in c322ed2 |
| 4 | Snapshot trace (every graph bucket to 64, a 2048-token prefix hit, n = 2, logprobs, prompt logprobs) run-identical across 3 fresh engines (at 52f0f1a; the final locks re-ran it byte-identically). Prefix cache on/off and depth 1 vs 0 bounded (`holon_judge bounded`, design §7.10, at a302e9e): all configurations within margin of the baseline; planted bugs in prefix adoption and the depth-1 token feed fail exactly the configurations using their path. Prefix cache saves 155K tokens on the 48-session multi-turn trace |
| 5 | SeqTable seam: a fixed-token test drafter runs at depth 1 with correct positions and lengths (`plugins/tests/test_spec.py`) |
| 6 | Restart (A5000, 5 kills mid-stream, every stream completes; exact continuation is tested in fp32): median 11.5 s, max 11.6 s for Qwen3-8B (6.7 s for 0.6B), at 0f50514. **SLO: restart ≤ 15 s for models up to 8B on a 24 GB card.** A poison request is quarantined without rollback, also alone on a quiet server; `n > 1` and prompt logprobs are served |
| 7 | Memory fit per §2 (Qwen3-8B on a 24 GB A5000: 42K KV tokens at the default budget, mini-sglang 37K). Tuned default locks for rtx-a5000, rtx-4090, rtx-5090, written at ee12f42 and re-propagated after the engine split (c488f64, snapshot traces byte-identical) (`holon_tools tune`, then propagated: Triton paged decode/verify, FlashInfer prefill and fused per-token ops on all three) |
| 8 | Core 3035 / 6000 code lines (counted with the tokenizer since 2026-10-08: the earlier line-based check skipped code after docstrings that close mid-line, reporting about 2000); lints, import contracts and the static checks green |

Carried into M0.1 and later:
- CI jobs: `manifest-deps`, `no-arch-branches`, `reference-provenance` and the GPU-runner jobs (`isolation-check`,
  `judge-L2-goldens`, `snapshot`) exist as commands (`holon_tools propagate`, `holon_judge l2 | snapshot`), not yet as
  CI jobs on scheduled runners.
- Propagation: changed-line coverage (auto-advance stays opt-in until then); approval's "no L2 regression vs the
  current lock" (needs L2 metrics in the lock); the engine/supervisor starting from the newest lock.
- Graph replay with logprobs (logprob steps run eagerly at depth 1) and the graph-vs-eager bounded comparison.
- rtx-5080 lock (M0.1, per §3.4 item 7).
- Watch: one rtx-4090 lock run failed with a CUDA "unknown error" in FlashInfer `silu_and_mul` during memory
  profiling (the largest prefill); the same step ran cleanly on another rtx-4090 node and in two earlier runs, so it
  was taken as a node fault. A recurrence would point at the kernel (an out-of-bounds access) instead.
- The two agent-trace records at ee12f42 carry `-dirty`: an earlier run in the same worktree had rewritten a tracked
  evidence file (no code changed); `bench/run.py` now ignores `evidence/` when judging dirtiness.
- Prefix eviction in O(1): an ordered set of evictable leaves (eviction now scans from the LRU end, which stays
  cheap while unused leaves sit there; long contexts with many pinned blocks would make the scan long).

---

## 4. M0.1 — quant pipeline with GGUF (Qwen3-8B GGUF)

Work:
- `weight_sources/gguf`: memory-mapped GGUF reader; model config from GGUF metadata (cross-checked against the HF
  config, which is authoritative for fields GGUF lacks); `recipe(cfg, source)` selects `gguf_weights` (names +
  value transforms) so the bf16 recipe term is unchanged; `model_id` includes the checkpoint build, so each quant
  has its own lock history. Tokenizer from the HF base bundle with a token-table hash check; a GGUF-embedded chat
  template takes precedence (differences logged).
- `dialects/ggml`: ggml tensor types → `Scheme`s for k-quants (Q2_K…Q6_K), i-quants (IQ1_S…IQ4_XS, IQ4_NL) and
  Q4_0/Q4_1/Q5_x/Q8_0, with `dequant_ref` following llama.cpp's gguf-py reference; layouts `gguf.native` (block
  structs as stored, no repack).
- `kernels/ggml_cuda`: llama.cpp's current ggml-cuda quantized kernels (`mmvq`, `mmq` incl. i-quants) ported behind
  a ≈ 200-line shim that allocates from our workspaces (per S2, ≈ 5 developer-days across M0.1–M0.2), as our own
  Torch extension; parity tests against llama.cpp on the same shapes. The int8 `mmq` prefill path is ported first:
  on GeForce cards a dequantize + cuBLAS fallback is too slow even as a stopgap (S2 consumer results).

Acceptance: `git diff plugins/models/qwen3` limited to adding `gguf_weights` to the recipe; L0 for every
supported type vs `dequant_ref`; L2 bounded vs a golden recorded from `dequant_ref` weights in the dense HF class
(CPU fp32), with quantization error vs the bf16 model reported separately; Qwen3-8B bf16 locks unchanged;
**first 16 GB consumer data point**: Qwen3-8B Q4_K_M and IQ3_XXS on RTX 5080 with ≥ 64K KV tokens; decode
tok/s vs llama.cpp (`llama-server`) on the same card and quant recorded.

### 4.1 Status (2026-10-08): acceptance met

Design and evidence: `docs/m01-gguf.md` (rev 3.1); records in `evidence/` and `judge/results/`, locks in `plans/`.
Qwen3-8B, unsloth `Qwen3-8B-GGUF` Q4_K_M and UD-IQ3_XXS. Built differently from the work list above, by decision
in the design: the GGUF file is self-contained (no HF base model or config), there is no `dialects/ggml` (the
kernels read the stored blocks, `gguf.native`; gguf-py dequantizes for the judge and L0), and references never
serve (test-only `holonlm-testing`).

| # | Result |
|---|---|
| 1 | `models/qwen3` adds only the GGUF weight map, `arch.gguf.qwen3` and a bundle built from the weight source; the bf16 recipe hash is unchanged (the bf16 locks re-propagated with no `recipe` change). GGUF is self-contained: configuration, tokenizer, chat template and stop set come from the file and match the HF checkpoint (tokenization, agent transcripts with tools and reasoning, stop set) |
| 2 | L0 for the 18 supported types (q8_1-emulated references per kernel path, mmq schedules asserted by the kernels that launch, mmvq rows 1–9, bit-exact `convert`/`get_rows`, compute-sanitizer clean with no weight padding): sm_86 and sm_89 118 passed (the PDL test needs sm_90+), sm_120a 119 passed |
| 3 | Per-tensor gate: all 399 tensors of both files ≤ 0.5 NMSE (worst 0.0065 Q4_K_M, 0.104 UD-IQ3_XXS). Tier A (dense dequantized weights, A5000): both pass (KL p99 4.62e-3 / 4.73e-3 vs floors 5.47e-3 / 5.35e-3); rope θ 9e5, transposed `attn_output`, gate/up swap and Q/K permutation all fail. Tier B (production kernels; floors = HF bf16 with q8_1 emulated per path, recorded on the A5000): prefill (mmq) and decode (mmvq: 8 tokens a step on sm_86, 5 on sm_120) pass for both files on the A5000 and the RTX 5080; one tensor read as the wrong type and a stale q8_1 workspace (every 97th call) fail; a rare stale workspace (every 9973rd call) fails the decode run and passes prefill, where it fired once — the mmq floor is 3× looser than mmvq's (llama.cpp's mmq pairs the Q4_K min-term with unquantized sums), so prefill sensitivity comes from Tier A and L0. Quantization error vs the bf16 checkpoint (reported): Q4_K_M KL p50 1.3e-3, top-1 95.7%, ΔPPL +1.6%; UD-IQ3_XXS 1.1e-2, 88.8%, +9.5% |
| 4 | bf16 Qwen3-8B locks on rtx-a5000, rtx-4090, rtx-5090: snapshot traces byte-identical, L2 passing, changes limited to `[units]` source pins; approved |
| 5 | GGUF snapshot trace (Q4_K_M) run-identical over 3 fresh engines on rtx-a5000 and rtx-5080 |
| 6 | RTX 5080: both files serve 64K-token contexts — UD-IQ3_XXS with 75K KV tokens at `max_seqs` 64, Q4_K_M with 65.5K at `max_seqs` 48 (64 leaves 64.1K: the graph pool is 0.50 GiB, not the estimated 0.25). In-process bars vs llama-bench at the same commit: pp512 1.04× / 1.06×, tg128 0.977× / 0.972× (llama-bench's warm-up repetition excluded, as in ours) (Q4_K_M 8291 / 155.0 t/s, UD-IQ3_XXS 8595 / 206.0); host CPU 0.36 ms per decode step. Serving vs `llama-server` (Holon over its own HTTP frontend; same settings): batch-1 TPOT equal (6.52 ms Q4_K_M), TTFT lower on every trace, decode throughput 1.27× at batch 8 and 2.3× at batch 48–64; agent trace TTFT / TPOT p50 57 / 7.4 ms vs 228 / 9.0 ms (Q4_K_M) |
| 7 | GGUF locks for rtx-a5000 and rtx-5080 (`unsloth/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M@120307ba529eb243`, `…/Qwen3-8B-UD-IQ3_XXS@495e28b50d2373d0`); the 5080 locks plan its deployment settings (`max_step_tokens` 2048, `max_seqs` 48 / 64; `memory_fraction` 0.98 shapes no output, so it is a deployment flag outside the plan); all four tuned to Triton paged decode |

Found on the way:
- Any prompt longer than `max_step_tokens` failed on the GPU: a step of mid-prompt chunks ran the head on zero rows,
  which FlashInfer's rmsnorm rejects; the CPU reference kernels accepted it, so no M0 test saw it (fixed: no head
  without sampled rows, with a regression test).
- `llama-server` streams no chunk for a token whose text ends in an incomplete UTF-8 character; the bench counts its
  tokens by `tokens_predicted`.

Carried forward:
- The design's KV estimate for the 5080 missed the CUDA-graph pool; quantized KV (M0.3) is the remedy if a larger
  `max_seqs` is needed at 64K.
- A Tier B prefill run cannot see rare kernel faults (its floor is loose by construction); L0 and Tier A cover them.

## 5. M0.2 — model isolation + MoE (Qwen3-30B-A3B GGUF)

Work: `models/qwen3_moe` — `recipe.py` copied from qwen3 with `derived_from` (diff ≤ ~15 lines: MoE ffn via
`Ref("std.ffn.moe_topk@1", …)`, `mlp_only_layers`, expert weight map); `std.ffn.moe_topk` coarse Op (router, top-k,
experts, combine; reference); GGUF MoE through the ported llama.cpp `mul_mat_id` path (batched for all types,
incl. i-quants); pinned configs, provenance.

Acceptance: qwen3 locks unchanged or advanced only by propagate; MoE L0 vs reference; L2 bounded vs a golden from
`dequant_ref` weights (CPU fp32, ≈ 122 GB in RAM); **30B MoE on a 16 GB card**: UD-IQ3_XXS / UD-Q3_K_XL on RTX 5080
with ≥ 16K KV tokens, Q4_K_M on 4090/A5000 (the 5080 covers sm_120); decode tok/s vs llama.cpp on the same card and quant recorded;
core diff: the stacking weight rule only (`docs/m02-moe.md` D7).

MoE is format-general: M0.2 also serves the bf16 HF checkpoint as a correctness path on an RTX PRO 6000 node
(stacked loading of HF's per-expert tensors, a Triton grouped GEMM, L2; no performance bar). FP8, NVFP4 and MXFP4
are M0.6. Design: `docs/m02-moe.md` (D1, D7, D8).

### 5.1 Status (2026-10-09): acceptance met

Design and evidence: `docs/m02-moe.md` (rev 5 and its implementation review); records in `evidence/` and
`judge/results/`, locks in `plans/`. Qwen3-30B-A3B, unsloth `Qwen3-30B-A3B-GGUF` UD-IQ3_XXS and Q4_K_M; the bf16
HF checkpoint on an RTX PRO 6000. Built differently from the work list above, by decision in the design: the MoE is
its own package (`std.moe`: `moe_topk@1`, `linear_experts@1` with a layout fact, `moe_combine@1`), the expert path
runs in f32 between the stacks with an f32 router (D3), and dense bf16 stacks run on our Triton grouped GEMM.

| # | Result |
|---|---|
| 1 | `models/qwen3_moe` copied from qwen3 (`derived_from`): ~25 changed recipe lines (the `std.moe` `Ref` and its facts, five expert weight rules). Core diff: the stacking weight rule only (`WeightRule.stack`, 42 lines in `build.py`/`model.py`). Qwen3-8B locks re-propagated — bf16 on rtx-a5000 and rtx-4090, GGUF on rtx-5080 and rtx-a5000: snapshot traces byte-identical, L2 passing, only the core's, `kernels.flashinfer`'s and (GGUF) `kernels.ggml_cuda`'s source pins changed; approved (the rtx-a5000 locks were then removed: lock classes are rtx-4090 and rtx-5080, design §4.4) |
| 2 | L0 for the MoE ops (`linear_experts` per type, path and `glu`; `moe_topk`; `moe_combine`; our SwiGLU; the padding plant) with the dense kernels' and `kernels.triton_experts`' (`evidence/l0_m02_2026-10-09.json`): at the final code sm_86 and sm_120a (RTX 5080) 221 passed each, memcheck 0 errors on both; sm_89 219 passed (4d2cfa2, before the last tests were added); the RTX PRO 6000 (sm_120a, 188 SMs) 216 passed at d8c26c2. The router precision test passes at the fp32 budget, and a bf16 router and TF32 fail it |
| 3 | Gate: every tensor of both files ≤ 0.5 NMSE per expert (worst 0.092 UD-IQ3_XXS, 0.0094 Q4_K_M), routers exact. Tier A (fp32 CPU, production GGUF source): KL p99 1.8e-5 / 2.2e-5 under the 1e-4 sanity ceiling; its 7 plants fail (renormalisation off in one layer at 3.94× the floor's KL p99 against a predicted 1.5–3×: the layer's expert output shrinks by its top-8 mass; δ unchanged). Tier B on the A5000 at the final engine code (c07efdc; floors: the D3-mirroring bf16 block with q8_1 per path) passes prefill and decode for both files, every expert call on the floor's path; also on the RTX 5080 (UD-IQ3_XXS, 4d2cfa2). Wrong type, stale q8_1 (dcf8902) and the up-first merge (8157e92) fail; the rare stale plant passes the decode run (1.07×, as reported). Quantization error vs the bf16 checkpoint: Q4_K_M KL p50 1.0e-3, top-1 95.9%, ΔPPL +1.0%; UD-IQ3_XXS 2.8e-3, 93.2%, +3.0% |
| 4 | Snapshot trace run-identical over 3 fresh engines (UD-IQ3_XXS on rtx-5080, Q4_K_M on rtx-a5000). Graphs vs eager (`bounded --graphs-vs-eager`) passes at decode batches 3, 5, 12 and 20 for both files; the padding plant fails exactly where the padded bucket's expert path is mmq (batches 12 and 20, and batch 5 for UD-IQ3_XXS on sm_86) and passes on mmvq, as predicted |
| 5 | RTX 5080, UD-IQ3_XXS, 16K contexts at `max_seqs` 32: 18.6K KV tokens at `memory_fraction` 0.9, the class's deployment setting (a 16K prompt served); 20.8K at 0.98 (22.1K at `max_seqs` 16), where prompt-logprob steps run out of memory (below). The bars and the serving runs used 0.98 (no prompt logprobs). In-process vs llama-bench (median of repetitions 2–5): pp512 / tg128 1.11× / 0.97× (8199 / 240.2 t/s; bar 5970 / 199) on the 5080 (4d2cfa2); Q4_K_M 1.02× / 1.01× on the 4090 (63941b4) and 1.10× / 1.02× on the A5000 (UD-IQ3_XXS 1.04× / 1.00×; dcf8902). Serving vs `llama-server` on the 5080: TPOT 4.21 vs 4.26 ms at batch 1 and 10.1 vs 11.6 ms at batch 8, lower TTFT; agent trace TTFT / TPOT p50 62 / 4.9 ms vs 209 / 6.0 ms (`llama-server` has no KV admission control: at Holon's KV size it runs 3 slots, its queue holding the rest) |
| 6 | GGUF locks: UD-IQ3_XXS on rtx-5080 (`max_seqs` 32, `max_step_tokens` 2048, `max_model_len` 16384), Q4_K_M on rtx-4090; tuned to Triton paged decode, L2 passing (an rtx-a5000 Q4_K_M lock was also written and then removed with the lock-class decision) |
| 7 | The bf16 HF checkpoint on an RTX PRO 6000 (sm_120, 96 GB; c21762a): it loads through the stacking rule, and the structural test finds every expert equal to the checkpoint's (two swapped experts: 6 mismatches; `evidence/structure_qwen3-30b-a3b_rtx-pro-6000_2026-10-09.json`). L2 against its fp32 golden passes (KL p99 3.32e-2 vs the D3 bf16 floor's 3.42e-2, itself below the pure-HF floor's 4.37e-2, as predicted); the stack off by one, gate/up swapped and renormalisation ignored fail. The M0.1 chat set reads correct (facts, code, Chinese, a tool call). `kernels.triton_experts` L0 passes on sm_86 and sm_120a (row 2) |

Found on the way:
- `kernels.flashinfer`'s fixed 128 MiB workspace overflowed for Qwen3-30B-A3B (4 KV heads) on the 128-SM RTX 4090:
  FlashInfer splits the KV into up to 2 CTAs per SM per KV head. Sized per device and geometry now, with a test.
- A Tier B MoE floor emulates one kernel path; the verdict now checks that every expert call took it.

Carried forward:
- At `memory_fraction` 0.98 the 30B on the 16 GB card keeps ~50 MB beyond the measured worst step, below allocator
  slack for prompt-logprob steps (L2 ran out of memory). The 5080's deployment setting is therefore 0.9 (the lock was
  tuned there; it plans the same kernels), and the lock does not record it: a deployer passes it. Remedy: an
  absolute headroom floor in the KV budget.
- The 30B GGUF goldens and floors predate the `experts` meta field; their expert path was pinned to eager anyway
  (`gguf.hf_model`).
- Lock classes (2026-10-09 decision, design §4.4): rtx-4090 and rtx-5080 during development; rtx-5090 and rtx-3090
  at release (their cards borrow the nearest lock, unverified, until then; the outdated rtx-5090 Qwen3-8B lock stays
  as history); rtx-a5000 has no locks (an sm_86 L0 host, uncommon among users). The rtx-4090 has no Qwen3-8B GGUF locks yet (M0.1 wrote them for rtx-a5000 and rtx-5080).
- UD-Q3_K_XL (optional) not recorded.

## 6. M0.3 — Qwen3.8-27B: heterogeneous state, prefix reuse, quantized KV (merged with the former M0.3b)

Merged 2026-10-09 by user decision: Qwen3.8-27B (the current, more used family) is the target; Qwen3.5 shares its
architecture (`Qwen3_5ForConditionalGeneration`); a tiny random-weight model of it is the fast test fixture. Design and plan:
`docs/m03-qwen38.md`.

Work: `models/qwen3_5` (Qwen3.5 and Qwen3.8, HF and GGUF, text-only); `std.mixers.gdn` with `causal_conv1d`,
`gated_delta_rule`, `gated_rmsnorm` ops; `kernels.fla` (vendored per S3) and our conv1d / gated-norm kernels;
`std.state.recurrent` (Slot state with 64-aligned checkpoint payloads) and checkpoint policies; the core's designed state
seams (SeqTable-slot addressing, prefill cuts, MemOps, payloads in plugin pools, the fixed-point prefix hit); quantized KV as KV layouts (q8_0 through our
Triton kernels, fp8 through FlashInfer); the std knob `gqa.output_gate` (partial rotary and the norm offset reuse existing knobs).

Acceptance (design §1): L2 for the 27B on every path (bf16 HF on an RTX PRO 6000; GGUF UD-IQ3_S / UD-Q4_K_M Tier A
and B); bounded checks (chunked vs one-shot prefill, prefix resume vs recompute, graphs vs eager, depth 1 vs 0); prefix tokens
saved on agent traces with depth-1-safe checkpoint publishing; q8_0 / fp8 KV judged against floors that emulate them;
RTX 5080 UD-IQ3_S serving a 32K context (q8_0 KV), RTX 4090 UD-Q4_K_M, bars vs llama.cpp ≥ 0.8×; locks for rtx-5080 and
rtx-4090; Qwen3 / Qwen3-MoE snapshots byte-identical, propagated with one approval; core diff limited to the
designed state seams, all general.

### 6.1 Status (2026-10-12): acceptance met

Design and evidence: `docs/m03-qwen38.md` (rev 3 and its implementation review, re-reviewed clean) and
`docs/m03-in-step-snapshot.md` (in-step checkpoints, reviewed); records in `evidence/` and `judge/results/`, locks in
`plans/`. Qwen3.8-27B, unsloth `Qwen3.8-27B-GGUF` UD-IQ3_S and UD-Q4_K_M; the bf16 HF checkpoint on an RTX PRO 6000.
Results are at the commit in their names; the final verdicts at the paired ΔPPL guard (65a82cb) and later.

| # | Result |
|---|---|
| 1 | `models/qwen3_5` from qwen3 (`derived_from`), HF and GGUF maps; the tiny model matches transformers (HF and llama.cpp-converted GGUF) on the CPU. Gates: 851 tensors per file, none over 0.5 (worst 0.296 UD-IQ3_S, 0.026 UD-Q4_K_M; norms, `A_log`, conv exact). L2 passes on every path: bf16 HF on the RTX PRO 6000 (top-1 98.45% vs floor 98.36%; its floor with f32 RoPE frequencies); GGUF Tier A (PRO 6000) and Tier B prefill and decode (A5000) for both files. The fp32 goldens run in fp32 (transformers 5.6 ignores `dtype` for a multimodal checkpoint's text model: the judge casts). Quantization error vs bf16: UD-IQ3_S top-1 88.2%, KL p50 1.25e-2; UD-Q4_K_M 94.6%, 2.3e-3. Plants: all seven L2 plants fail on Tier B, the four HF-path ones on the PRO 6000 |
| 2 | Bounded on the 27B (A5000): chunked 100 vs one-shot, prefix resume vs recompute, depth 1 vs 0, graphs vs eager at batch 5 pass; a payload filed one block early, the conv carry dropped between prefill chunks and padding on a real slot fail. Snapshot trace run-identical over 3 fresh engines at each lock's deployment |
| 3 | Prefix reuse on the RTX 5080 agent trace: recurrent checkpoints taken **in-step** (the turn's prefill in one step: fla stores its f32 state at the checkpoint's chunk; exact on resume, L0 bitwise) — TTFT p50 8.42 s vs 8.92 s with the former cut at max_seqs 2; at the deployment, max_seqs 4 with four checkpoints, TTFT p50 / p99 0.93 / 7.1 s vs 5.3 / 13.8 s with the cache off, throughput 69.5 tok/s. Registration needs committed tokens (core tests) |
| 4 | q8_0 KV (our Triton write, decode, verify, prefill) and fp8 KV (FlashInfer prefill at bf16's speed, our Triton decode and verify) pass L2 against floors emulating them (q8_0: A5000, both files; fp8: the 4090); the KV error vs bf16 KV floors: KL p50 2.9e-4, top-1 97.8%. KV units 1.88× smaller with q8_0 |
| 5 | RTX 5080 UD-IQ3_S, q8_0 KV: 76.8K KV tokens at max_seqs 2, 58K at max_seqs 4 (the deployment), a 32.5K prompt served. Bars vs llama-bench at the same KV format: pp512 1.15–1.20×, tg128 1.03–1.05×, tg at 8K 1.03–1.07× (5080 and 4090). Serving vs `llama-server` on the 5080: TTFT 106 vs 297 ms for one request; agent TTFT p50 9.3 vs 10.7 s. Holon vs FreeToken (Qwen3-8B bf16, A5000): decode 0.98–0.99×, prefill 0.93–0.98×, agent TTFT within 2% |
| 6 | Locks (approved): Qwen3.8-27B UD-IQ3_S on rtx-5080 and UD-Q4_K_M on rtx-4090, both at max_seqs 4 with four checkpoints (q8_0 KV on the 5080) |
| 7 | Qwen3 / Qwen3-MoE: all five development-class locks propagated with one approval (rtx-4090: Qwen3-8B bf16, 30B Q4_K_M; rtx-5080: 30B UD-IQ3_XXS, 8B Q4_K_M, 8B UD-IQ3_XXS) — snapshot traces byte-identical, L2 passing |
| 8 | L0 for the GDN and quantized-KV kernels: sm_86, sm_89 (4090) and sm_120a (5080) 193 passed each, fp8 included from sm_89; memcheck 0 errors on all; the in-step snapshot's L0 on sm_86. The GDN bound is pinned to the judge's transformers |
| 9 | Core diff: the §4.3 seams plus the in-step publish seam (`StateSpec.in_step`, `BatchPlan.publishes`, `StepCtx.pool`), all general and proven by the toy state kind |

Decisions taken at the close (2026-10-12): the in-step checkpoints are part of M0.3; L2's ΔPPL guard is paired with
the floor (|ΔPPL vs floor| ≤ 0.25% + 2.5 SE, sequences as clusters, the SE capped at √2 × the floor's own so an error
confined to a few sequences cannot widen its budget — the unpaired margin had no noise term and failed runs whose
top-1 and KL matched the floor; the guard stays coarse, ≈ 2.5% on v1-qwen3_5); the 5080 deploys at max_seqs 4 with
four checkpoints.

Found on the way: the fp32-golden dtype and RoPE-buffer issues above; a misplaced conv plant (fixed to drop only the
prefill-chunk carry); NVIDIA's CuTe DSL parsing `-d` from any process's argv (bench `--depths`); prefill cut with
nothing to publish (fixed in the review); the scheduler never mixes prefill and decode in a step.

Next (optimization, after M0.3): fuse the prefill glue (a KV-store kernel, fused add+norm and q/k norm: 167 extra
kernels per Qwen3-8B step vs FreeToken), then a model-specific megakernel experiment (design §5.7) and prefill graphs
for slow hosts; a small-batch quantized matmul (9–64 rows) to beat llama.cpp's mmq on short steps; mixed
prefill/decode scheduling (prefill steps stall every decoder).

## 7. M0.4 — speculative decoding

Work: `drafters/mtp` (consumes ModelSpec `draft_heads`), `drafters/ngram_gpu`, `drafters/lookup_cpu` (depth
tolerance 0 — exercises per-Seq gating), `verifiers/greedy_exact`, `verifiers/rejection`, `draft_policy/fixed_k`,
`draft_policy/cost_model` (Strata-style choose/observe), `transforms/spec_rows` (pads mixed k to the step's
`spec.q`; caps verify windows at checkpoint boundaries); GDN `verify_mode="replay"` (default) and
`"checkpoint_slots"` (opt-in, `draft_slots ≤ 2`); drafter KV as a Rewindable in the target's page group.

Acceptance:
- MTP is exercised on Qwen3.5 and on Qwen3.8-27B GGUF, loading the separate MTP GGUF (`MTP/mtp-Qwen3.8-27B-Q4_0.gguf`,
  1.28 GiB; gguf-py's QWEN35 list has no NEXTN tensors) as a second weight source into `draft_heads`. On RTX 5080
  only with UD-IQ3_S (MTP takes ~half of the remaining ≈ 2.6 GiB); UD-IQ4_XS + MTP on 4090/5090.
- MTP and n-gram Seqs actually run at **depth 1** on the device SeqTable, using the spec-at-depth-1 attention
  kernel named by S1 (measured: fraction of spec steps at depth 1).
- Spec vs non-spec greedy **bounded** on Qwen3-8B (n-gram) and Qwen3.5 (MTP); greedy self-consistency exact
  (every accepted token equals the verify step's target argmax); L1.5 "verify + commit(k) vs k decode steps"
  bounded for `replay` and `checkpoint_slots`; L0-dist for `rejection`.
- Acceptance length and speedup recorded on agent traces (A5000 + consumer desktops); graphs captured for the `spec.q` set.
- Core diff target zero.

### 7.1 Status (2026-10-13): acceptance met (Gate A: MTP, n-gram; Gate B: DFlash2)

Design and evidence: `docs/m04-spec.md` (rev 3, reviewed clean; §10 is Gate A's status with every result, decision
and deviation); records `evidence/bench_m04nat-*` (natural text), `fit_m04-*`, `judge/results/` (bounded, L0-drafter,
snapshot); locks in `plans/`. The scope moved from this section's first draft: MTP from the target checkpoint itself
(no separate MTP GGUF), DFlash2 added, `checkpoint_slots` dropped for GDN replay, the CPU lookup drafter and the
cost-model policy deferred.

| # | Result |
|---|---|
| 1 | Drafters `drafters.mtp` (Qwen3.5 / Qwen3.8 HF and GGUF heads) and `drafters.ngram` (device prompt lookup); verifiers `std.spec.greedy` (exact-match against the sampler's own draw, greedy or sampled) and `std.spec.rejection` (L0-dist); policy `std.spec.fixed_k`; core seams with toy proofs (core 3643 / 6000 lines) |
| 2 | Lossless: bounded spec vs plain passes for Qwen3-8B + n-gram, Qwen3.5-9B + MTP, the 27B GGUF + MTP; GDN rollback by masked replay, bitwise at the unit; L0-drafter passes on Qwen3.5-9B |
| 3 | 27B + MTP, natural text, single stream: 1.49× on the RTX 5080 (10.63 ms, llama.cpp MTP 10.53), 1.85× on the RTX 4090 (11.52 ms, llama.cpp 12.63); agent trace TPOT p50 1.71× / 1.75× (19.1 / 16.7 ms vs llama.cpp MTP 28.1 / 34.6); acceptance equal to llama.cpp's drafts; 2.6–2.8 tokens per verify step |
| 4 | Verify steps 100% graphed, 99.6–100% at depth 1; n-gram on Qwen3-8B: agent 1.10×, single stream 0.97× (below break-even) |
| 5 | All eight locks propagated, snapshots byte-identical (speculation off changes nothing) — at Gate A and again at Gate B |
| 6 | Gate B (`docs/m04-spec.md` §11–§13): taps, attention `causal` / `window` facts, sparse draft laws (`std.spec.rejection` exact for sampled drafts), `drafters.dflash` (HF and GGUF checkpoints); L0-drafter passes for DFlash2 (bf16 and Q4_K_M) and the GGUF MTP head with teacher-forced chain steps; bounded spec vs plain on the 27B passes, snapshot run-identical |
| 7 | 27B + DFlash2 on the RTX 4090 (q8_0 KV): single stream 2.01× (10.59 ms; llama-server with DFlash 12.75), agent TPOT p50 2.14× (14.06 ms); acceptance equal to llama.cpp's on the same drafter file; 2.93 tokens per verify step single stream (predicted 3.5–5: the drafter's on this target); 54K KV tokens; llama-server with DFlash does not fit the agent trace on 24 GB |

Decisions: speculation is benchmarked on natural text (`--corpus`; uniform-id prompts inflate acceptance); MTP's
k table stays `1:3,3:2`; lock approvals delegated for passing verdicts. Found on the way: the sampler's uniform
could round to 1.0 (fixed); a stale MTP carry at position 0 (fixed by the L0-drafter judge); `repeat_interleave`
without an output size broke capture; the startup profile missed the drafter's work on verify steps; the snapshot
judge and `propagate` gave later engines smaller pools; a draft checkpoint was named by host path in plans (now by
identity). The drafter judge's top-1 margin became max(0.25 points, 2.5 paired SE) after the first verdicts (a fixed
0.25-point bar fails a correct drafter 8–40% of the time at 300–2000 rows; reviewer simulation). Next: M0.7's
speculation items (a verify step costs ≈ 1.5–1.8 decode steps: multi-column mmvq, the draft heads, rewind's index
ops).

## 8. M0.5 — structured output and tool calls

Work: `grammar/xgrammar` LogitsProcessor (`host_input` bitmask; Rewindable matcher advancing through drafts and
rolling back), parsers with `channel()` in the qwen3/qwen3_5 bundles, `frontends/anthropic` (messages,
count_tokens), `frontends/openai_responses`.

Acceptance: 100% schema conformance on the JSON-schema suite (truncated responses reported separately); tool-call
round trips via OpenAI and Anthropic APIs (Claude-Code-style client on `/v1/messages`); grammar + spec vs grammar
without spec bounded; core diff target zero.

Amended (2026-10-13, `docs/m05-grammar.md` D1, D7–D9): "core diff target zero" became "only the general seams of
the design's §3, each with a toy proof" — grammar needs a request lifecycle, a depth rule, drafts on the host and
per-request errors; the three API dialects are routers of one frontend package (`holon serve` runs one frontend);
conformance is defined for strict requests against a published reject list (xgrammar silently ignores some JSON
Schema keywords); `tool_choice: "auto"` stays unconstrained unless a tool is strict.

### 8.1 Status (2026-10-14): acceptance met, one recorded shortfall

Design and evidence: `docs/m05-grammar.md` (rev 5, reviewed clean; §11 is the build record with every result,
deviation and the panel round after step 5); records `evidence/conformance_*`, `gate_cost_*`, `gate_sizing_*`,
`round_trips_*`, `constrained_bounded_*`, `claude_code_session_*`.

| # | Result |
|---|---|
| 1 | Conformance on JSONSchemaBench (strict, against the published reject list): Qwen3-8B, the full filtered split (1625 accepted): greedy 1582 of 1582 valid, sampled 1587 / 1584 of 1587 / 1584, 0 request-time errors; Qwen3.8-27B + MTP, 500 sampled (301 accepted): 293 of 293 valid. Declared coverage 60.6% / 59.1% of the split (152k / 248k vocabularies). The model-free oracle finds no accepted schema that admits an invalid instance (its hole list grew with the reject list: lookarounds, backreferences, mid-pattern anchors) |
| 2 | **Shortfall, accepted by the owner:** the 27B's truncated share is +2.33 points over unconstrained (2.66% vs 0.33%, n = 301, SE ≈ 1.0) against the 2-point gate; 6 of its 8 truncations are grammar-admitted runaways (whitespace, digits, a repeated string). The 8B is within the gate (−1.73 points). Runaway guards deferred |
| 3 | Tool calls: OpenAI chat, Anthropic Messages and OpenAI Responses round trips on both models (9 of 9 calls schema-valid, streamed = whole); a Claude Code session against the 27B on an RTX 5080 completed a multi-tool task |
| 4 | Grammar with speculation: exact on CPU tiny models with an oracle drafter; the constrained bounded check on the 27B + MTP passes (first divergences ≤ 0.124 nats under the calibrated 0.229); refused drafts 1.1–1.7% |
| 5 | A′ (no constrained row scheduled with one in flight); one graph set with a processor selected (27B + MTP: KV −0.12%); nothing changes with none selected; the forward/sample split deferred (1.061× / 1.026×, under its 1.10× rule) |
| 6 | Core: the logits-processor seams of the design's §3, a request lifecycle with per-request errors, and slot objects' `stats()`, each with a toy proof (core 3868 / 6000 lines) |

Decisions: conformance is defined for strict requests against a reject list; no whitespace cap (S2-1: a cap of 16
costs 17–30× the compiled size and 3–10× the mask fill); the truncation gate kept as written and its 27B shortfall
accepted (owner, 2026-10-14). Found on the way: GGUF kernels are not batch-invariant (the judge's reference scores one
sequence at a time); xgrammar silently ignores lookarounds and mid-pattern anchors (refused under strict); Claude Code
sends `system` messages inside `messages`. Next: M0.7 (mixed prefill/decode steps answer the 34–36 s stream stalls
measured here).

## 8a. M0.7 — performance

Work (`docs/m07-perf.md`): fused glue (bf16 KV store; q/k-norm + RoPE + KV store, `attention.qk_rope_kv@1`; mmq writing
bf16; NVFP4's `scale2` in mmvq's epilogue; gate/up + SwiGLU; one q8_1 quantization per shared input; the GDN rewind's
glue in one launch), the mmvq/mmq crossover per class and shape, mixed prefill/decode steps without speculation
(`chunk_with_decode`), and a megakernel experiment gated on speed of light.

### 8a.1 Status (2026-10-15): acceptance met

Design and evidence: `docs/m07-perf.md` (rev 3.3, reviewed clean; §11–§16 as built per step, §17 the close: the close
panel's findings, the propagation, acceptance and deviations); records `evidence/m07-step*_*`, `bench_m07-step4-*`,
`m07-close_aa_*`, `propagate_m07-close_*`.

| # | Result |
|---|---|
| 1 | Fusions kept where they pay, on every class: bf16 `kv_write` (8B B1 −2.4 to −6.4%), D3 `qk_rope_kv` (8B B4 −2.9 to −5.2%), mmq bf16 (prefill −3.5 to −7.2%), NVFP4 `scale2` in mmvq (B4 −3.4 to −7.2%), gate/up + SwiGLU (8B B1 −0.65 to −1.6%; Q4_K and NVFP4 only), shared q8_1 (27B −0.5 to −0.9%), GDN rewind glue (verify −2.1 to −5.0%); GDN gating dropped (< 1%) |
| 2 | Crossover per (class, type, in, out, mode): 25 RTX 5080 and 11 RTX 5090 entries; decode B8 +4.5% / +3.4%, DFlash verify −4.5% / −3.5%; small-step L2 passing; none on the 4090 |
| 3 | Mixed steps at cap 512 on the plain 27B locks (RTX 5080, 4090; 5090 too): agent max-gap p99 7.5–13× better, paired TTFT 1.02–1.10, long-prompt TTFT p99 +14–16%, throughput within 0.3% (long prompt −2.1% on the 5080); cap 256 fails the TTFT bars everywhere |
| 4 | Megakernel gate: **stop** — f 8.6% (RTX 5080) and 15.6% (RTX 5090) after the fusions, 1.050× / 1.094× at c = 0.55, under 1.15× |
| 5 | One propagation at the close: 11 development-class locks and 2 new (M0.6 Gate B's 8B FP8 on the 5080 and 4090), every lock with the grammar processor, the plain 27B locks with `chunk_with_decode` 512; L2 passing and snapshots run-identical ×3 on all 13; 13 approvals (delegated) |
| 6 | Core: ABI 0.2 (`BatchPlan.n_decode`, `EngineConfig.chunk_with_decode`, the planner hook `MixesRowKinds`, the kernel prop `serves_decode_rows`, `BuildCtx.candidates`; `SchedulePolicy.mix_phases` removed), each with a toy proof; 4369 / 6000 lines; full GPU suite 1210 passed in 17 min 59 s (1155 in 18 min 55 s at the start) |

Decisions: D2 keyed per shape, not per type (step 0 panel); D7's cap 512 by measurement (the simulator under-costs
mixed steps); D8 stops; D9 (small-batch quantized matmul) stays outside M0.7 (1.24 on the 5080, under 1.25); the FP4
repack holes measured, not fixed; `chunk_with_decode` in `BuildCtx.options` is the engine's effective value (0 under a
drafter). Found on the way: D11's first build slowed 8-row NVFP4 steps by 8.5% (a per-element `scale2` load); mmvq's
fused gate is slower than the composition for most i-quants; the decode-row RoPE plant passed every relative judge
(`bounded`'s floor moves with it) — a decode-row check at the attention unit, eager and captured, now covers it; three
of four RTX 5090 desktops failed in a step (thermal throttling, Xid 79). Plan §9: 13 approvals in one day at the close
(the "> 1 / day" trigger fires, recorded: async propagation is v1's); sweeps 10 and 18 min per class. Next: the release
(ABI 1.0 at its step 6, the release-class locks and the MTP decision at step 7). Follow-up, not covered by D7's
measurement: the a8 8B / 30B GGUF locks (the 8B Q4_K_M on the RTX 5080 is where M0.5 measured the 36 s stall) stay at
`chunk_with_decode = 0` because only the 27B was measured; release step 7 measures cap 512 on them against §1.3's bars
or records why not. Measured at release step 7 (`docs/release.md` §16.5): the long prompt's stall falls 23–37× at 512, but the
agent trace's gap bar (≥ 2×) fails on every such lock, whose agent gaps are already short, so they keep 0.

## 8b. Release

Work (`docs/release.md`): packaging and `holon serve` from the lock, ABI 1.0, the public root, the release-class locks,
the benchmark suite, the agent demo, the clean install. Publishing (step 11) waits for the owner's go.

### 8b.1 Status (2026-10-16): ready for the owner's go, after the lead's final panel review

Design and evidence: `docs/release.md` (rev 3.2, reviewed clean; §13–§19 as built per step); records
`evidence/release7_*`, `bench_release*`, `bars_release*`, `release8_clock_*`, `agent_demo_*`, `release10_*`.

| # | Acceptance (release §1) | Result |
|---|---|---|
| 1 | Legibility | retired as a gate by the owner (§4.0); no claim |
| 2 | ABI 1.0 | declared, enforced at load, listed in `core/abi.txt` and checked by `check_abi.py` (§15) |
| 3 | Locks | every cell written and approved in the release environment (§16, §19.1–§19.2); `holon serve` reports a plan equal to the lock on each class, no clamp (§19.3) |
| 4 | Packaging | a clean install by the README on each class served a chat completion and an Anthropic message; after the prebuild fix a first start compiles only Triton kernels (and FlashInfer's CuTe DSL norms in memory); the Dockerfile builds (§19.3–§19.4) |
| 5 | Benchmarks | all four classes; every GGUF cell ≥ 0.8× llama.cpp at the vendored pin; every cell published, losses included (§17) |
| 6 | Agent demo | 3 of 3 on the RTX 5080 and 3 of 3 on the RTX 4090 (§18) |
| 7 | History | superseded by the owner (2026-10-16): `main` keeps the full history; the squashed line is archived (release §20). The history's private strings were scrubbed in place (§20); v0.1.0 released privately (§21) |

**Carried CI items** (plan §3.5, §3.6; release §9 step 10):

| item | disposition |
|---|---|
| `no-arch-branches` (design invariant 10) | **deferred to v1.** The core is covered by `check_core_denylist.py`; std and other shared units by review and the import contracts (no plugin imports another). A scan of std at the release found no `model_type` or architecture branch; the GGUF tokenizer keeps a table of pre-tokenizers keyed by the file's `tokenizer.ggml.pre`, as llama.cpp does — data, not a branch. A lint over std (the denylist's names, with an allow-list for that table and docstrings) is v1's |
| `reference-provenance` | **closed for v0.** Each unit's oracle (`reference.py`) is inside its unit directory, so every lock's `src` pin (and `sem` where declared) covers it; `check_locks.py` re-reads every lock, and `--release` fails when a newest lock does not pin this tree's sources, a changed oracle included. Loading the oracle from a reviewed commit by `sem` (design §4.6) matters once an untrusted candidate can edit code: it comes with v1's self-improvement layer |
| `isolation-check` (propagate on trusted GPU runners) | **deferred to v1** (it needs scheduled GPU runners per class). In v0 it is a manual step: every lock change was propagated on the class's desktop with snapshots ×3 (§16, §19), and `check_locks.py --release` holds every newest lock to this tree. Step 10 found what such a runner must handle: one class's trace can differ between two desktops of the same GPU (the RTX 3090's 27B, §19.1) |
| the IPC IDL (design §7.8) | **deferred to its trigger** (a native frontend, design §12). The IPC records (`EngineRequest`, `TokenDelta`, `Abort`) are frozen in ABI 1.0's listing, so `check_abi.py` already fails on any change to them; the generated IDL file adds nothing until a second implementation reads it |
| CPU-only GitHub Actions run | **prepared:** `.github/workflows/ci.yml` runs `run_all.sh` and the CPU tests (707 passed, 579 skipped locally without a GPU); it runs once the repository is published (step 11), not before |

Decisions: the 27B locks' context raised to 49,152 (lead, after §18's ≈ 31K side requests); the FlashInfer prebuild
fixed and every lock re-propagated (§19.2); MTP off and `chunk_with_decode` 0 on the 8B / 30B locks stay as decided at
step 7 (README, Known limitations). Found on the way: the prebuild left every FlashInfer module to compile again at first
use; an RTX 3090 desktop records another 27B trace than the one the lock was written on; the hub listing failed once
with an HTTP error while four desktops listed it within minutes. Next: step 11 on the owner's go (§8 of `docs/release.md`).

## 9. Kill / pivot criteria

| When | Measure | Pivot if |
|---|---|---|
| S6 spike | CPU handshake inside a FULL graph on a desktop host | seam does not hold → redesign the graph/CPU seam before the core ABI freezes |
| M0 | decode + mixed-trace TTFT/TPOT vs mini-sglang; CPU/step vs nano-vllm; core LOC | < 0.9× after two weeks of perf work, or core > 6k → move per-token input prep to GPU kernels and cut core scope before continuing |
| M0.2 | core changes needed for MoE; approvals per week from propagation | > 1 core ABI change, or > 3 approvals/week → abstraction or machinery is wrong; simplify |
| ~~Before the release freezes the ABI~~ — **retired by the owner (2026-10-14)** | legibility benchmark (agent cost to add a model vs a mini-sglang fork). An exploratory 1 + 1 pilot passed on both sides at ≈ 1.4–1.8× Holon's cost (one run each, within noise); Holon's advantage is isolation (no shared code touched), not agent cost per model (`docs/release.md` §4.0; record `evidence/legibility_pilot_2026-10-14.json`) | — |
| Any | propagation sweep time; approvals/day | > 30 min or > 1/day → async propagation (design §4.6 trigger) |
| M0.1 / M0.2 / M0.3 | GGUF decode **and prefill** tok/s vs llama.cpp on the same card and quant (RTX 5080), dense and MoE | < 0.8× on either after a perf pass → fix the GGUF kernel path (ported i-quant `mmq` or own Triton kernels for the slow types) before adding features |
| v1 offload | MoE larger than VRAM on RTX 5080: decode tok/s vs FreeToken and Strata on the same node | < 0.8× → rework the offload path before adding features |

---

## 10. v1 and v2 (outline; detailed plans written when v0 lands)

**v1 — consumer-GPU story**
- **A full pip package** (owner, 2026-10-14: v0 ships on GitHub only, as a tagged source release; pip comes in v1 or
  v2). It needs prebuilt wheels with Holon's CUDA extensions compiled for the pinned torch / CUDA stack and every
  supported SM, a CI build and test matrix on each card class, and `pip install holonlm` serving without a local CUDA
  toolkit.
- **Expert caching/offload** on the M0 seams (SeqTable, MemOps, `drain`) plus additive core seams (design D8; the S6 stream-memop CPU worker was
  spiked only):
  TieredStore, in-graph DeviceResolver with double-buffered residency, placement policies (LRU, profile-seeded
  decayed LFU), CPU miss compute (GGUF experts computed on the CPU, Strata-style, with ggml CPU kernels) — MoE
  models larger than VRAM on RTX 5080 16 GB (consumer desktop): Qwen3.5-35B-A3B at Q4_K_M and above, then larger
  MoE (e.g. Qwen3.5-122B-A10B, Qwen3.8-Flash-Next) — compared with FreeToken and Strata on the same nodes.
- KV offload to pinned RAM / SSD (connector-chosen BLOCKED vs layer-wise, cancellable futures, demote/pin eviction).
- VMM ledger + live rebalance KV ↔ experts through `quiesce` (run-identical after shrink/regrow).
- PIECEWISE graphs for mixed prefill/decode steps (mixing itself is M0.7's core budget `chunk_with_decode`, design
  §7.1; mixing verify rows is its follow-up, `docs/m07-perf.md` §9); depth 2; radix PrefixIndex if profiles warrant.
- HF-ecosystem quantization: GPTQ/AWQ INT4 (W4A16), and FP8 on sm_86 (W8A16 via uint8 bit-decode; native FP8,
  NVFP4 and MXFP4 are M0.6); TP=2 on 2× A5000 (zero core diff); `update_weights`; LoRA; Qwen3.5 VLM.
- **Optional self-improvement layer** (design §10): supervisor, improve API (first-version tool set), sandbox
  container, spare-GPU evaluation, L4 + L4-confirm, canary with sampled live re-scoring. Acceptance: (1) a scripted
  or external agent promotes a tunable and a policy variant with evidence whose gain CI excludes 0; (2) a served-agent
  demo makes a judged submission end to end; (3) injected variants — wrong output, edited `reference.py`, `sem`
  mismatch at a site, `build`/`weights` override, multi-site composite — are rejected or routed to proposals; (4)
  sandbox escape tests (forbidden write/read/network) fail.
- Reuse machinery and native modules only at their triggers (design §4.6, §12).
- **A recurrent-state checkpoint at the end of each answer** (GDN hybrids, `std.state.recurrent`): today the
  `prompt_end` policy checkpoints only where a prompt ends, so an agent's next turn reuses the previous prompt but
  prefills the previous answer again, even when it is sent back verbatim. In the pi test on an RTX 5090 that answer
  was 44% of the uncached input tokens and ≈ 2% of wall time (`docs/agent-test-pi-rtx5090.md` §2); the gain grows
  where templates or agents keep the reasoning in later prompts and answers are long. It needs the answer's tokens
  as the template re-renders them (a checkpoint at the last aligned position the next prompt shares) and a payload
  unit per finished request.
- **Locks pin only the code that can change outputs** (kernels, model math, scheduling, sampling), not version
  strings, comments or response-reporting fields. Today a unit's `src` hashes every source file of its directory, the
  core's included, so v0.1.1 (a `TokenDelta` field for usage reporting, a comment, the version string) outdated all 22
  locks and needed a propagation round on four card classes for byte-identical traces. `src_hash` and the `holon`
  unit pin are frozen in ABI 1.0 and in every committed lock, so narrowing them (e.g. hashing the modules a model's
  step imports, without docstrings and comments, and leaving the frontend-facing records out) is a deliberate v1
  change with its own lock migration.

**v2 — breadth**
- DeepSeek-class: DSA indexer (page groups), V4 compressed KV (`tokens_per_unit`) — correctness on reduced-layer
  configs locally, performance on datacenter GPUs.
- PD disaggregation, EP, DP-attention, PP on datacenter GPUs (H100/GB200); draft-model drafter (nested ModelPlugin).
- Tree verification, KVMem-style retrieval window, batch-invariant deterministic Plan (if needed), L3 quality judge,
  ROCm/CPU platforms, multi-model serving.

---

## 11. Working agreements

- One branch per milestone: `m0-core`, `m0.1-fp8`, …
- One reviewer subagent per milestone for complex/risky code; fix-and-re-review until clean.
- Every perf claim comes with an evidence record (command, fingerprint class, plan_id, raw numbers); consumer
  claims only from consumer desktops.
- Every bug fix comes with a test that fails before and passes after.
- A PR that changes shared code includes the `holon propagate` output.
