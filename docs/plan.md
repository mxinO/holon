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

Runtime dependencies (v0): `torch`, `triton`, `safetensors`, `tokenizers`, `jinja2`, `flashinfer` (JIT caches
pre-built), vendored GDN ops from vLLM's `third_party/flash_linear_attention` (subset, Apache-2.0, `vllm.*` imports
stripped; **not** pip `fla-core`), ggml-derived GGUF CUDA kernels (dequant, mmvq, mmq, MoE — from SGLang's
copy of vLLM's adaptation of llama.cpp ggml-cuda, MIT/Apache-2.0, built as our own Torch extension),
`gguf` (reader reference), `xgrammar`, `fastapi`/`uvicorn`, `pyzmq` + `msgspec`.
Judge/test only: `transformers` (golden recording, in an environment **without** fla), `hypothesis`, `numpy`,
`scipy`. Baselines in separate pinned environments under `bench/`: mini-sglang, nano-vllm, vLLM (bf16 paths) and
llama.cpp `llama-server` + kvmem-llama.cpp (GGUF paths).

**Implementation policy (D6):** references are studied, not copied. Each work item names what it learns from;
the code is our own plugin-native implementation. Only kernels may be borrowed, under design §5.6.

## 2. Milestone ladder (v0)

| Milestone | Model | Seam proven | Key plugins added |
|---|---|---|---|
| M0 | Qwen3-8B bf16 | core end to end: overlap, graphs, prefix cache, device SeqTable, MemOps/quiesce, locks, snapshots, judge L0–L2 | qwen3, gqa, paged_kv, flashinfer attn, std ops, sampler, FCFS, openai frontend |
| M0.1 | Qwen3-8B GGUF (Q8_0, Q4_K_M, IQ4_XS, IQ3_XXS) | quant pipeline: GGUF weight source, ggml dialect, borrowed ggml kernels chosen per class; first 16 GB consumer data point (RTX 5080) | gguf weight source, ggml dialect, ggml_cuda kernels |
| M0.2 | Qwen3-30B-A3B GGUF (UD-IQ3_XXS 12.0 GiB, Q4_K_M; UD-Q3_K_XL optional) + bf16 HF (correctness, RTX PRO 6000) | model isolation; format-general MoE unit and ops; a 30B MoE on a 16 GB card | qwen3_moe recipe, std.moe, GGUF MoE kernels, stacked loading, a Triton grouped GEMM |
| M0.3 | Qwen3.8-27B: GGUF (UD-IQ3_S 11.2 GiB on 16 GB, UD-Q4_K_M) + bf16 HF (correctness, RTX PRO 6000) | heterogeneous state, publish/adopt MemOps, checkpoint prefix reuse, quantized KV cache (q8_0 ours, fp8 FlashInfer) | qwen3_5 recipe, gdn mixer + kernels, recurrent state kind, checkpoint policies, quantized-KV layouts |
| M0.4 | Qwen3.5 / Qwen3.8 (built-in MTP) + Qwen3-8B | spec decoding at depth 1 on the device SeqTable; Rewindable incl. GDN rollback | mtp drafter, gpu n-gram drafter, verifiers, draft policies |
| M0.5 | Qwen3-8B, Qwen3.5-9B | forward/sample split, host_input, Rewindable grammar | xgrammar processor, parsers, anthropic + responses frontends |
| M0.6 | Qwen3-8B / Qwen3-30B-A3B in FP8 (block-scaled), NVFP4 and MXFP4 | low-precision HF checkpoints, dense and MoE: weight-source dialects over `QTensor` parts, `qlinear` / `linear_experts` providers, judge tiers per format; RTX PRO 6000 (sm_120), FP8 also on sm_89 | fp8 / nvfp4 / mxfp4 dialects and kernels |

Expert caching/offload, KV offload, VMM and the optional self-improvement layer follow in v1 (§10); their core
seams are built and spiked in M0 (design D8).

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

## 8. M0.5 — structured output and tool calls

Work: `grammar/xgrammar` LogitsProcessor (`host_input` bitmask; Rewindable matcher advancing through drafts and
rolling back), parsers with `channel()` in the qwen3/qwen3_5 bundles, `frontends/anthropic` (messages,
count_tokens), `frontends/openai_responses`.

Acceptance: 100% schema conformance on the JSON-schema suite (truncated responses reported separately); tool-call
round trips via OpenAI and Anthropic APIs (Claude-Code-style client on `/v1/messages`); grammar + spec vs grammar
without spec bounded; core diff target zero.

## 9. Kill / pivot criteria

| When | Measure | Pivot if |
|---|---|---|
| S6 spike | CPU handshake inside a FULL graph on a desktop host | seam does not hold → redesign the graph/CPU seam before the core ABI freezes |
| M0 | decode + mixed-trace TTFT/TPOT vs mini-sglang; CPU/step vs nano-vllm; core LOC | < 0.9× after two weeks of perf work, or core > 6k → move per-token input prep to GPU kernels and cut core scope before continuing |
| M0.2 | core changes needed for MoE; approvals per week from propagation | > 1 core ABI change, or > 3 approvals/week → abstraction or machinery is wrong; simplify |
| After M0.3 (design D9) | legibility benchmark: ≥ 3 runs per side, same agent and prompt, adding a model whose units already exist, from a branch point before its model package | Holon not ≥ 30% cheaper for the agent than a mini-sglang fork → collapse slots and nouns |
| Any | propagation sweep time; approvals/day | > 30 min or > 1/day → async propagation (design §4.6 trigger) |
| M0.1 / M0.2 / M0.3 | GGUF decode **and prefill** tok/s vs llama.cpp on the same card and quant (RTX 5080), dense and MoE | < 0.8× on either after a perf pass → fix the GGUF kernel path (ported i-quant `mmq` or own Triton kernels for the slow types) before adding features |
| v1 offload | MoE larger than VRAM on RTX 5080: decode tok/s vs FreeToken and Strata on the same node | < 0.8× → rework the offload path before adding features |

---

## 10. v1 and v2 (outline; detailed plans written when v0 lands)

**v1 — consumer-GPU story**
- **Expert caching/offload** on the M0 seams (SeqTable, MemOps, `quiesce`, S6 stream-memop CPU worker):
  TieredStore, in-graph DeviceResolver with double-buffered residency, placement policies (LRU, profile-seeded
  decayed LFU), CPU miss compute (GGUF experts computed on the CPU, Strata-style, with ggml CPU kernels) — MoE
  models larger than VRAM on RTX 5080 16 GB (consumer desktop): Qwen3.5-35B-A3B at Q4_K_M and above, then larger
  MoE (e.g. Qwen3.5-122B-A10B, Qwen3.8-Flash-Next) — compared with FreeToken and Strata on the same nodes.
- KV offload to pinned RAM / SSD (connector-chosen BLOCKED vs layer-wise, cancellable futures, demote/pin eviction).
- VMM ledger + live rebalance KV ↔ experts through `quiesce` (run-identical after shrink/regrow).
- Mixed prefill/decode scheduling policy + PIECEWISE graphs; depth 2; radix PrefixIndex if profiles warrant.
- HF-ecosystem quantization: GPTQ/AWQ INT4 (W4A16), and FP8 on sm_86 (W8A16 via uint8 bit-decode; native FP8,
  NVFP4 and MXFP4 are M0.6); TP=2 on 2× A5000 (zero core diff); `update_weights`; LoRA; Qwen3.5 VLM.
- **Optional self-improvement layer** (design §10): supervisor, improve API (first-version tool set), sandbox
  container, spare-GPU evaluation, L4 + L4-confirm, canary with sampled live re-scoring. Acceptance: (1) a scripted
  or external agent promotes a tunable and a policy variant with evidence whose gain CI excludes 0; (2) a served-agent
  demo makes a judged submission end to end; (3) injected variants — wrong output, edited `reference.py`, `sem`
  mismatch at a site, `build`/`weights` override, multi-site composite — are rejected or routed to proposals; (4)
  sandbox escape tests (forbidden write/read/network) fail.
- Reuse machinery and native modules only at their triggers (design §4.6, §12).

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
