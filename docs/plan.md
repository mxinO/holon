# plug-serve — Implementation plan

Status: rev 3 (after an adversarial review) · Date: 2026-10-06 ·
Companion to [design.md](design.md)

---

## 1. Environment and hardware

| Role | Hardware | Use |
|---|---|---|
| Development | 8× RTX A5000 24 GB (**sm_86**), 96 cores, 251 GB RAM; CUDA 13.0, Python 3.12, torch 2.11.0+cu130, Triton 3.6, uv | coding, functional tests, golden recording (CPU fp32 fits in RAM) |
| Consumer evidence | consumer desktop machines (AMD B650, 8 cores, 128 GB RAM, 1 GPU): **RTX 5080 16 GB** and **RTX 5090 32 GB** (sm_120), **RTX 4090 24 GB** (sm_89), RTX 3090 (sm_86) | all performance numbers, tuned defaults per fingerprint class, consumer memory-fit checks |
| Datacenter (later) | H100/H200, GB200/GB300 | v2 datacenter path (TP/EP/PD) |

The A5000 host (16 memory channels, server CPUs) is not a
faithful consumer proxy — CPU-side and PCIe-bound tunables are only measured on consumer desktops.

Consequences: sm_86 has no FP8/FP4 tensor cores and Triton rejects `fp8e4nv` there → weight-only W8A16 (uint8
bit-decode) / W4A16 on sm_86; native FP8 on sm_89/sm_120 and NVFP4 on sm_120 are verified on consumer desktops.

Runtime dependencies (v0): `torch`, `triton`, `safetensors`, `tokenizers`, `jinja2`, `flashinfer` (JIT caches
pre-built), vendored GDN ops from vLLM's `third_party/flash_linear_attention` (subset, Apache-2.0, `vllm.*` imports
stripped; **not** pip `fla-core`), vendored Triton fused-MoE, `xgrammar`, `fastapi`/`uvicorn`, `pyzmq` + `msgspec`.
Judge/test only: `transformers` (golden recording, in an environment **without** fla), `hypothesis`, `numpy`,
`scipy`. Baselines (mini-sglang, nano-vllm, vLLM) in separate pinned venvs under `bench/venvs/`.

**Implementation policy (D6):** references are studied, not copied. Each work item names what it learns from;
the code is our own plugin-native implementation. Only kernels may be borrowed, under design §5.6.

## 2. Milestone ladder (v0)

| Milestone | Model | Seam proven | Key plugins added |
|---|---|---|---|
| M0 | Qwen3-8B bf16 | core end to end: overlap, graphs, prefix cache, device SeqTable, MemOps/quiesce, locks, snapshots, judge L0–L2 | qwen3, gqa, paged_kv, flashinfer attn, std ops, sampler, FCFS, openai frontend |
| M0.1 | Qwen3-8B-FP8 (block 128×128) | quant pipeline; per-class kernel selection (W8A16 on sm_86, native FP8 on sm_89/120); first 16 GB consumer data point (RTX 5080) | fp8_block dialect, triton W8A16, native FP8 GEMM |
| M0.2 | Qwen3-30B-A3B-GPTQ-Int4 (16.9 GB, `desc_act=False`) | model isolation + coarse MoE op + 2nd quant format | qwen3_moe recipe, moe op, gptq dialect, W4A16 + fused MoE |
| M0.3 | Qwen3.5-4B → 9B, bf16, text-only | heterogeneous state, page groups, publish/adopt MemOps, checkpoint prefix reuse; legibility benchmark | qwen3_5 recipe, gdn mixer, ssm_checkpoint kind, checkpoint policy |
| M0.4 | Qwen3.5 (built-in MTP) + Qwen3-8B | spec decoding at depth 1 on the device SeqTable; Rewindable incl. GDN rollback | mtp drafter, gpu n-gram drafter, verifiers, draft policies |
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
| Qwen3-8B-FP8 | 8.79 GiB | 144 KiB | 64K ≈ 19.3 GiB ✓ | ✓ | 32K ≈ 14.8 GiB ✓ |
| Qwen3-30B-A3B-GPTQ-Int4 | 15.8 GiB | 96 KiB | 32K ≈ 20.3 GiB ✓ | ✓ | ✗ (v1 expert offload) |
| Qwen3.5-4B bf16 text + MTP | ≈ 8.5 GiB | full-attn layers only | ✓ | ✓ | ✓ (exact numbers in M0.3) |
| Qwen3.5-9B bf16 text + MTP | 17.13 GiB | full-attn layers only | ≈ 21–22 GiB ✓ | ✓ | ✗ |

Global acceptance for every milestone ≥ M0.1:
- Every earlier model's lock is byte-unchanged, or advanced by `plugserve propagate` (auto-advance on a
  run-identical snapshot, or one approval) — design §4.6.
- New plugins pass judge L0–L2 (+ L0-dist for samplers/verifiers/processors, + L1.5 where state is involved).
- Memory fit met; perf numbers recorded on the relevant consumer classes; CI lints green.

**Size budget** (first-party, excluding tests and vendored kernels; ±30%):

| Part | Estimate |
|---|---|
| Core runtime + plugin API | 3.5–5k (cap 6k) |
| std units (ops, gqa, paged KV, sampling, blocks, ffn, moe) incl. `reference.py` | 2–2.5k |
| Model recipes + Qwen bundles (templates, reasoning/tool parsers) | ~0.9k |
| Own kernels + kernel adapters | 1.5–2k |
| GDN mixer/state/policies, spec decoding, grammar | ~2.1k |
| Frontends (OpenAI chat/completions, Responses, Anthropic) | ~1.5k |
| Judge harness (L0, L0-dist, L1, L2 recorder/comparator, snapshot traces, oracle provenance) | ~2–2.5k |
| Propagation + tune tooling, bench (trace replay, ABAB, remote runners), CI lints | ~1.5–2k |
| Supervisor (v0 part: lifecycle, crash attribution) | 0.5–0.8k |
| **v0 total** | **≈ 15–19k** (tests typically +0.5–1×; vendored kernels separate) |

---

## 3. M0 — core end to end (Qwen3-8B bf16)

### 3.1 Spikes (first, time-boxed, results recorded as evidence)

| Spike | Question | Exit |
|---|---|---|
| S1 Attention | FlashInfer paged prefill/decode on sm_86/sm_89/sm_120 + cu130 + torch 2.11, head_dim 128/256; JIT pre-build; graph capture; run-to-run byte identity; an attention path that reads lengths **on the device** from the SeqTable (for spec Seqs at depth 1) | names, per fingerprint class, the non-spec kernel (FlashInfer with host `plan()` in `plan_bounds`) and the spec-at-depth-1 kernel (FlashInfer device-length path or our own Triton paged attention) |
| S2 W8A16 | Triton GEMM loading FP8 e4m3fn as `uint8`, exact bit-decode to bf16 in the K-loop (sm_86) | L0 exact decode vs torch `float8_e4m3fn` cast |
| S3 GDN | vendored chunk + recurrent kernels on sm_86/sm_120 vs HF torch reference; no `vllm` imports | within budget, no NaN at T=1..8192, run-to-run identical |
| S4 Determinism | every production kernel candidate is run-to-run deterministic with pinned configs (no split-K/atomics); which sites must be `uncovered` | list of deterministic kernels + uncovered sites |
| S5 Golden | HF fp32 recording on CPU (or `device_map` over GPUs) for 8B; noise floor HF bf16 vs fp32 on frozen prompts | τ and KL/ΔPPL budgets calibrated |
| S6 CPU handshake | stream-memop flags (`cuStreamWriteValue`/`WaitValue`) + persistent native CPU worker inside a FULL graph; `quiesce()` + double-buffered resolver map under depth 1 (learning from FreeToken's `cpu_executor`) | per-step overhead measured on a desktop host (consumer desktops); seam confirmed before the ABI freezes |

### 3.2 Work items

**A. Repo bootstrap** — uv workspace (`core/`, `std/*`, `plugins/*`, `judge/`, `supervisor/`, `bench/`, `plans/`);
CI scripts (§3.5); `AGENTS.md` (no claims without measurement; failing-then-passing test per bug fix; never edit
`judge/` or locks by hand); job templates for consumer classes.

**B. `plugserve.api`** — dataclasses/Protocols only: Manifest schema, resources, slots + cardinality, Plan + lock
format (`recipe`, `[units]` src/sem, `[deploy]`, `[snapshot]`), recipe terms (`Ref`, `ModelSpec` with
`block`/`weights`) + canonical hash, `Plugin.register(regs)`, ParamSpec, `finalize_weights`, Op/composite markers,
KernelSpec/Constraint, Scheme/Layout/QTensor/Dialect/converter, StateSpec/StateKind (incl. `fork`, `HitLen`,
MemOp-returning publish/adopt), page groups, Ledger/Pool, PrefixView + eviction contract, Seq/SeqTable/
SamplingParams, BatchPlan (bounds) / ShapeKey / DeviceBatch, Planner (`plan_bounds`, `prepare_device`,
`needs_host_lens`), MemOp, Rewindable, LogitsProcessor, Drafter/DraftPolicy/Verifier/PlanTransform,
FrontendBundle, Event, Platform (incl. stream memops), launcher slot.

**C. `plugserve.core` runtime** (our own implementation; learns from mini-sglang's overlap loop and graph runner,
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

**E. plugins** — `models/qwen3` (`recipe.py` ≤ 80 LOC + `bundle/`), attention kernel plugin per S1, `frontends/openai`.

**F. judge harness** (separate process, client of the engine) — L0 + `budget(dtype, K)`; L0-dist for the sampler;
L1 contracts (incl. run-to-run determinism of bound kernels); L2 recorder (CPU fp32 / multi-GPU; quantized models
via `dequant_ref` into the dense HF class) + comparator with calibrated budgets; snapshot-trace recorder/comparator;
oracle provenance (`reference.py` at reviewed commits).

**G. bench** — trace replay (ShareGPT-like + **mixed agent multi-turn tool traces**), TTFT/TPOT/throughput,
interleaved ABAB, client-side timing; remote runners for consumer classes; baselines in pinned venvs.

**H. propagation tooling** (outside core, ~400 LOC; changed-line coverage captured during snapshot traces) — `plugserve impact`, `plugserve propagate` (3 outcomes,
design §4.6), `propagate --new-model`, `plugserve tune` (benchmarks tunables/kernel configs per fingerprint class
into the lock); CI `isolation-check` re-runs propagate on a trusted GPU runner (A5000) and on remote
consumer-desktop runners for the maintained consumer classes (on shared-change PRs where capacity allows, nightly at
least).

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
7. Memory fit per §2; tuned default locks for `a5000`, `rtx-4090`, `rtx-5090`.
8. Core LOC ≤ 6000; lints green.

### 3.5 CI checks (from M0)

`lint-core-denylist`, `import-contracts`, `manifest-deps`, `core-loc-cap`, `no-env-config`,
`no-torch-cuda-outside-platform`, `no-arch-branches`, `reference-provenance`, `isolation-check` (propagate),
`core-diff` (informational in v0, enforcing from v1), `unit`, `judge-L0`, `judge-L2-goldens` (GPU runner),
`snapshot` (GPU runner).

---

## 4. M0.1 — quant seam (Qwen3-8B-FP8)

Work: `dialects/fp8_block` (roles `weight`/`weight_scale_inv`; source layout `fp8_block.rowmajor`),
`Scheme(elem="e4m3fn", scales=(block 128×128,))` with `dequant_ref`; `kernels/triton_wxa16` W8A16 (uint8 load +
bit-decode, S2) for sm_86; a native FP8 GEMM kernel for sm_89/sm_120 (borrowed under §5.6 or our own Triton).

Acceptance: `git diff plugins/models/qwen3` empty; L0 for both kernels; the resolver picks W8A16 on `a5000` and
native FP8 on `rtx-4090`/`rtx-5080`/`rtx-5090` from the same lock format; L2 bounded vs the golden recorded from
`dequant_ref` weights; Qwen3-8B bf16 locks unchanged; **first 16 GB consumer data point**: Qwen3-8B-FP8 on RTX 5080
with ≥ 32K KV tokens, perf recorded.

## 5. M0.2 — model isolation + MoE (Qwen3-30B-A3B-GPTQ-Int4)

Work: `models/qwen3_moe` — `recipe.py` copied from qwen3 with `derived_from` (diff ≤ ~15 lines: MoE ffn via
`Ref("std.ffn.moe_topk@1", …)`, `mlp_only_layers`, expert weight map); `std.ffn.moe_topk` coarse Op (router, top-k,
experts, combine; reference); `dialects/gptq` (`desc_act=False` only; GPTQ v1 `qzeros` +1 convention in
`dequant_ref`; fp16 scales); `kernels/triton_wxa16` W4A16; `kernels/triton_moe` (vendored fused MoE W4A16 path,
pinned configs, provenance).

Acceptance: qwen3 locks unchanged (no shared unit changed) or advanced only by propagate; MoE L0 vs reference; L2
bounded vs a golden recorded on CPU from `dequant_ref` weights (≈ 122 GB fp32 in RAM); memory fit per §2; perf on
A5000, 4090, 5090; core diff target zero.

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

Work: `models/qwen3_5` (recipe copied from qwen3 with `derived_from`); `mixers/gdn` (vendored kernels per S3;
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
- **Legibility benchmark:** a fixed coding agent adds Qwen3.5 in plug-serve vs in a mini-sglang fork (≥ 3 runs per
  side, same agent and prompt, from a branch point before `qwen3_5` exists); measure agent time/tokens, lines
  outside the model package, human interventions.
- Core diff target zero.

## 7. M0.4 — speculative decoding

Work: `drafters/mtp` (consumes ModelSpec `draft_heads`), `drafters/ngram_gpu`, `drafters/lookup_cpu` (depth
tolerance 0 — exercises per-Seq gating), `verifiers/greedy_exact`, `verifiers/rejection`, `draft_policy/fixed_k`,
`draft_policy/cost_model` (Strata-style choose/observe), `transforms/spec_rows` (pads mixed k to the step's
`spec.q`; caps verify windows at checkpoint boundaries); GDN `verify_mode="replay"` (default) and
`"checkpoint_slots"` (opt-in, `draft_slots ≤ 2`); drafter KV as a Rewindable in the target's page group.

Acceptance:
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
| M0.3 | legibility benchmark: ≥ 3 runs per side, same agent and prompt, both starting from a branch point before `qwen3_5` exists | plug-serve not ≥ 30% cheaper for the agent than a mini-sglang fork → collapse slots and nouns |
| Any | propagation sweep time; approvals/day | > 30 min or > 1/day → async propagation (design §4.6 trigger) |
| v1 offload | Qwen3-30B-A3B decode tok/s on RTX 5080 vs FreeToken on the same node and settings | < 0.8× → rework the offload path before adding features |

---

## 10. v1 and v2 (outline; detailed plans written when v0 lands)

**v1 — consumer-GPU story**
- **Expert caching/offload** on the M0 seams (SeqTable, MemOps, `quiesce`, S6 stream-memop CPU worker):
  TieredStore, in-graph DeviceResolver with double-buffered residency, placement policies (LRU, profile-seeded
  decayed LFU), CPU miss compute — Qwen3-30B-A3B on RTX 5080 16 GB (consumer desktop) and bf16 on a 24 GB card;
  then Qwen3.5-35B-A3B. Compared with FreeToken on the same nodes.
- KV offload to pinned RAM / SSD (connector-chosen BLOCKED vs layer-wise, cancellable futures, demote/pin eviction).
- VMM ledger + live rebalance KV ↔ experts through `quiesce` (run-identical after shrink/regrow).
- Mixed prefill/decode scheduling policy + PIECEWISE graphs; depth 2; radix PrefixIndex if profiles warrant.
- NVFP4 (sm_120) and GGUF dialects; TP=2 on 2× A5000 (zero core diff); `update_weights`; LoRA; Qwen3.5 VLM.
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
- A PR that changes shared code includes the `plugserve propagate` output.
