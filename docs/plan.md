# Holon — Implementation plan

Status: rev 3 (after an adversarial review) · Date: 2026-10-06 ·
Companion to [design.md](design.md)

---

## 1. Environment and hardware

| Role | Hardware | Use |
|---|---|---|
| Development | 8× RTX A5000 24 GB (**sm_86**), 96 cores, 251 GB RAM; CUDA 13.0, Python 3.12, torch 2.11.0+cu130, Triton 3.6, uv | coding, functional tests, golden recording (CPU fp32 fits in RAM) |
| Consumer evidence | consumer desktop machines (AM5 desktop, 8 cores, 1 GPU): **RTX 5080 16 GB** and **RTX 5090 32 GB** (sm_120), **RTX 4090 24 GB** (sm_89), RTX 3090 (sm_86) | all performance numbers, tuned defaults per fingerprint class, consumer memory-fit checks |
| Blackwell dev (optional) | workstation with RTX PRO 6000 Blackwell Max-Q 96 GB (sm_120) | sm_120 kernel development and debugging (S1–S3 on Blackwell), large-memory work (GPU golden recording, large models), NVFP4 later; not a consumer-performance proxy |
| Datacenter (later) | H100/H200, GB200/GB300 | v2 datacenter path (TP/EP/PD, NVFP4) |

**Working method:** on a shared test machine, start one persistent container and run builds, tests and benchmarks
inside it repeatedly rather than as many short batch jobs. The A5000 host (16 memory channels, server CPUs) is not a
faithful consumer proxy — CPU-side and PCIe-bound tunables are only measured on consumer desktops.

**Quantization priority (D9):** for consumer and edge GPUs the formats that matter are llama.cpp's GGUF sub-4-bit
types (k-quants Q2_K–Q6_K, i-quants IQ1–IQ4, e.g. `unsloth/Qwen3.8-27B-GGUF`); they are v0. HF-ecosystem formats
(GPTQ/AWQ INT4, FP8) follow in v1; NVFP4/MXFP4 are datacenter formats (large-memory GPUs) in v2.

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
| M0.2 | Qwen3-30B-A3B GGUF (UD-IQ3_XXS 12.0 GiB, UD-Q3_K_XL, Q4_K_M) | model isolation + coarse MoE op; a 30B MoE on a 16 GB card | qwen3_moe recipe, moe op, GGUF MoE kernels |
| M0.3 | Qwen3.5-4B → 9B, bf16, text-only | heterogeneous state, page groups, publish/adopt MemOps, checkpoint prefix reuse; legibility benchmark | qwen3_5 recipe, gdn mixer, ssm_checkpoint kind, checkpoint policy |
| M0.3b | Qwen3.8-27B GGUF (UD-IQ3_S 11.2 GiB, UD-IQ4_XS 13.3 GiB) | large-model target on 16 GB; quantized KV cache (fp8 via FlashInfer, then q8_0/q4_0 via our own kernel; per-class support set by S1) | qwen3_8 recipe, quantized-KV layouts |
| M0.4 | Qwen3.5 / Qwen3.8 (built-in MTP) + Qwen3-8B | spec decoding at depth 1 on the device SeqTable; Rewindable incl. GDN rollback | mtp drafter, gpu n-gram drafter, verifiers, draft policies |
| M0.5 | Qwen3-8B, Qwen3.5-9B | forward/sample split, host_input, Rewindable grammar | xgrammar processor, parsers, anthropic + responses frontends |

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
| Qwen3.5-4B bf16 text + MTP | ≈ 8.5 GiB | full-attn layers only | ✓ | ✓ | ✓ (exact numbers in M0.3) |
| Qwen3.5-9B bf16 text + MTP | 17.13 GiB | full-attn layers only | ≈ 21–22 GiB ✓ | ✓ | ✗ (GGUF builds ✓) |
| Qwen3.8-27B GGUF UD-IQ3_S / UD-IQ4_XS | 11.2 / 13.3 GiB | 64 KiB (16 full-attn layers; ≈ 34 KiB q8_0) + GDN state ≈ 147 MiB/Seq | ✓ (UD-Q4_K_M 15.3 GiB) | ✓ | 4 Seqs: IQ3_S ≈ 43K tokens bf16 KV / ≈ 80K q8_0; IQ4_XS ≈ 9K / ≈ 17K |

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
  (Knobs `rmsnorm.offset`, `rope.rotary_dim` are added in M0.3 with old defaults.)
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

### 3.6 Status (2026-10-07): acceptance met

Measured on Qwen3-8B bf16 unless noted; records in `evidence/` and `judge/results/`, locks in `plans/`.

| # | Result |
|---|---|
| 1 | L2 passes for Qwen3-0.6B and Qwen3-8B on the final engine (`judge/results/l2-52f0f1a-*`), and for Qwen3-8B on rtx-a5000, rtx-4090 and rtx-5090 when their locks were written |
| 2 | Decode tok/s Holon vs mini-sglang — A5000: B1 40.9 / 41.5, B8 311.8 / 315.2, B64 2042.2 / 2042.9; RTX 5090: B1 99.0 / 97.8, B8 708.3 / 725.9, B64 4707.7 / 3905.3. Agent trace TTFT / TPOT p50 — A5000: 109.9 / 28.3 ms vs 106.4 / 27.5 ms; RTX 5090: 40.4 / 11.2 ms vs 42.4 / 10.9 ms. All within 10% |
| 3 | CPU per step at B256 (A5000, blocking sync): 2.45 ms vs nano-vllm 2.98 ms |
| 4 | Snapshot trace (every graph bucket to 64, a 2048-token prefix hit, n = 2, logprobs, prompt logprobs) run-identical across 3 fresh engines. Prefix cache on/off and depth 1 vs 0 bounded (`holon_judge bounded`, design §7.10): all configurations within margin of the baseline; planted bugs in prefix adoption and the depth-1 token feed fail exactly the configurations using their path. Prefix cache saves 155K tokens on the 48-session multi-turn trace |
| 5 | SeqTable seam: a fixed-token test drafter runs at depth 1 with correct positions and lengths (`plugins/tests/test_spec.py`) |
| 6 | Restart (A5000, 5 kills mid-stream, every stream completes; exact continuation is tested in fp32): median 11.5 s, max 11.6 s for Qwen3-8B (6.7 s for 0.6B). **SLO: restart ≤ 15 s for models up to 8B on a 24 GB card.** A poison request is quarantined without rollback, also alone on a quiet server; `n > 1` and prompt logprobs are served |
| 7 | Memory fit per §2 (Qwen3-8B on a 24 GB A5000: 42K KV tokens at the default budget, mini-sglang 37K). Tuned default locks for rtx-a5000, rtx-4090, rtx-5090 (`holon_tools tune`: Triton paged decode/verify, FlashInfer prefill and fused per-token ops on all three) |
| 8 | Core 2038 / 6000 lines; lints, import contracts and the static checks green |

Carried into M0.1 and later:
- CI jobs: `manifest-deps`, `no-arch-branches`, `reference-provenance` and the GPU-runner jobs (`isolation-check`,
  `judge-L2-goldens`, `snapshot`) exist as commands (`holon_tools propagate`, `holon_judge l2 | snapshot`), not yet as
  CI jobs on scheduled runners.
- Propagation: changed-line coverage (auto-advance stays opt-in until then); approval's "no L2 regression vs the
  current lock" (needs L2 metrics in the lock); the engine/supervisor starting from the newest lock.
- Graph replay with logprobs (logprob steps run eagerly at depth 1) and the graph-vs-eager bounded comparison.
- rtx-5080 lock (M0.1, per §3.4 item 7).

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

## 5. M0.2 — model isolation + MoE (Qwen3-30B-A3B GGUF)

Work: `models/qwen3_moe` — `recipe.py` copied from qwen3 with `derived_from` (diff ≤ ~15 lines: MoE ffn via
`Ref("std.ffn.moe_topk@1", …)`, `mlp_only_layers`, expert weight map); `std.ffn.moe_topk` coarse Op (router, top-k,
experts, combine; reference); GGUF MoE through the ported llama.cpp `mul_mat_id` path (batched for all types,
incl. i-quants); pinned configs, provenance.

Acceptance: qwen3 locks unchanged or advanced only by propagate; MoE L0 vs reference; L2 bounded vs a golden from
`dequant_ref` weights (CPU fp32, ≈ 122 GB in RAM); **30B MoE on a 16 GB card**: UD-IQ3_XXS / UD-Q3_K_XL on RTX 5080
with ≥ 16K KV tokens, Q4_K_M on 4090/5090/A5000; decode tok/s vs llama.cpp on the same card and quant recorded;
core diff target zero.

## 6. M0.3 — heterogeneous state (Qwen3.5-4B → 9B, text-only)

Qwen3.5 specifics (from HF config/modeling; record deviations found during implementation):
- 32 layers: 3 GDN : 1 full attention (`full_attention_interval=4`); weights under `model.language_model.*`;
  `model.visual.*` skipped when deployment field `text_only = true`; vocab 248320; 4B ties embeddings, 9B does not.
- Full attention: head_dim 256, `attn_output_gate=True` (`q_proj` output doubled, sigmoid gate), partial rotary
  (`partial_rotary_factor=0.25` → 64 of 256 dims, neox), interleaved mrope [11,11,10] reducing to 1-D rope for
  text — declare `position_fn="mrope"` and test the equivalence.
- GDN: q/k heads `repeat_interleave` 16→32, `use_qk_l2norm`, `RMSNormGated(silu(z))`, conv1d kernel 4 over 8192
  channels, fp32 state, `-exp(A_log)` in `finalize_weights`; zero-centred RMSNorm as offset-gamma.
- MTP (used in M0.4): `mtp.{fc, pre_fc_norm_hidden, pre_fc_norm_embedding, layers.0, norm}`; its KV joins the target
  KV's page group.

Work: `models/qwen3_5` (recipe copied from qwen3 with `derived_from`); `mixers/gdn` (vendored kernels per S3, plus
conv1d and gated RMSNorm kernels; GDN prefill graph-captured per length bucket;
pure-torch recurrent `reference.py`); `state/ssm_checkpoint` (`Slot`, `hit_length` = deepest checkpoint ≤ L,
publish/adopt as MemOps, ledger-budgeted checkpoint pool, joint chain eviction); `policies/checkpoint_prompt_end`
(default for agent traffic) and `policies/checkpoint_every_n`.

Acceptance:
- L1.5 bounded: chunked vs one-shot prefill; prefix-hit resume vs recompute; graph replay vs eager. Snapshot trace
  run-identical. L2 bounded vs HF fp32 golden (CPU, without fla).
- Multi-turn agent trace: prefill tokens saved vs cache-off measured; checkpoints published only at boundaries
  covered by committed tokens (depth 1 test).
- New std knobs (`gqa.output_gate`, `rope.rotary_dim`, `rmsnorm.offset`) land with old defaults and `propagate`
  auto-advances Qwen3 and Qwen3-MoE (run-identical snapshots, zero approvals) — first real test of reuse.
- Qwen3.5-4B on RTX 5080, 9B on 5090/4090/A5000; memory fit recorded.
- **Legibility benchmark:** a fixed coding agent adds Qwen3.5 in Holon vs in a mini-sglang fork (≥ 3 runs per
  side, same agent and prompt, from a branch point before `qwen3_5` exists); measure agent time/tokens, lines
  outside the model package, human interventions.
- Core diff target zero.

## 6b. M0.3b — large-model target (Qwen3.8-27B GGUF) + quantized KV cache

GGUF builds of Qwen3.5/3.8 differ from HF: norms stored as `w+1` (except `ssm_norm`), `ssm_a` = `-exp(A_log)`,
conv1d squeezed, and GDN value heads in tiled order (affects in_proj qkv/z/a/b, dt_bias, A_log, conv1d rows and the
input columns of `out_proj`). `gguf_weights` carries these value transforms and the GDN mixer uses
`v_head_order="tiled"` (design §5.1). GGUF goldens are additionally recorded through transformers' own GGUF loader
as an independent check of the mapping.

Qwen3.8-27B: dense GDN hybrid, 64 layers (48 GDN + 16 full attention), hidden 5120, one MTP layer — the model kvmem
targets on 16 GB cards. unsloth GGUF builds: UD-IQ3_S 11.2 GiB, UD-IQ4_XS 13.3 GiB, UD-Q4_K_M 15.3 GiB.

Work: `models/qwen3_8` recipe (copied from qwen3_5 with `derived_from`); quantized KV cache as a KV `Layout`
negotiated with the attention kernel — first `fp8` through FlashInfer (dequantized in-kernel; whether this path
applies on sm_86 classes is decided per fingerprint class by S1), then `q8_0` (and
`q4_0` for V, as kvmem uses) through **our own Triton paged attention for quantized KV** (head_dim 256; decode,
prefill and verify; reads lengths on the device for spec at depth 1; L0 reference); GGUF builds of the Qwen3.5
models; GDN `v_head_order="tiled"`; a checkpoint pool budgeted explicitly (≈ 7 GDN checkpoints per GiB for 27B).

Acceptance:
- Qwen3.8-27B UD-IQ3_S / UD-IQ4_XS on RTX 5080 16 GB and UD-Q4_K_M on 4090/5090; memory fit per the §2 table
  (weights + overhead + GDN state ≈ 147 MiB/Seq + KV 64 KiB/token bf16, ≈ 34 KiB q8_0) recorded; quantized KV at
  least doubles the KV tokens that fit vs bf16 KV.
- GGUF weight-map transforms verified: golden via `dequant_ref` and via transformers' GGUF loader agree.
- L2 bounded vs the `dequant_ref` golden; KV quantization error reported separately (bf16 vs q8_0 KV, same weights).
- Decode tok/s and prefill tok/s vs llama.cpp and vs kvmem-llama.cpp on the same RTX 5080 and quant, short and long
  context.
- Core diff target zero.

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
| M0.3 | legibility benchmark: ≥ 3 runs per side, same agent and prompt, both starting from a branch point before `qwen3_5` exists | Holon not ≥ 30% cheaper for the agent than a mini-sglang fork → collapse slots and nouns |
| Any | propagation sweep time; approvals/day | > 30 min or > 1/day → async propagation (design §4.6 trigger) |
| M0.1 / M0.2 / M0.3b | GGUF decode **and prefill** tok/s vs llama.cpp on the same card and quant (RTX 5080), dense and MoE | < 0.8× on either after a perf pass → fix the GGUF kernel path (ported i-quant `mmq` or own Triton kernels for the slow types) before adding features |
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
- HF-ecosystem quantization: GPTQ/AWQ INT4 (W4A16) and FP8 (W8A16 via uint8 bit-decode on sm_86, native FP8 on
  sm_89/120); TP=2 on 2× A5000 (zero core diff); `update_weights`; LoRA; Qwen3.5 VLM.
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
- PD disaggregation, EP, DP-attention, PP on datacenter GPUs (H100/GB200); NVFP4 and MXFP4 for large-memory datacenter
  GPUs; draft-model drafter (nested ModelPlugin).
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
