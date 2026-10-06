# plug-serve — Implementation plan

Status: draft for review (rev 2) · Date: 2026-10-06 · Companion to [design.md](design.md)

---

## 1. Environment and constraints

| Item | Value |
|---|---|
| Dev GPUs | 8× NVIDIA RTX A5000, 24 GB, **sm_86** (Ampere), driver <version> |
| Host | 96 cores, 251 GB RAM |
| Toolchain | CUDA 13.0, Python 3.12, torch 2.11.0+cu130, Triton 3.6, uv |
| Consequences | No FP8/FP4 tensor cores on sm_86, and Triton rejects `fp8e4nv` on sm_86 → v0 quantized paths are weight-only (W8A16 with uint8 bit-decode, W4A16). Native FP8 (sm_89+) / NVFP4 (sm_120) are v1 and need other GPUs. Served models must fit one 24 GB GPU. |

Runtime dependencies (v0): `torch`, `triton`, `safetensors`, `tokenizers`, `jinja2`, `flashinfer` (0.7, JIT caches
pre-built), vendored GDN ops from vLLM's `fla/ops` copy (Apache-2.0; **not** pip `fla-core`), vendored Triton
fused-MoE, `xgrammar`, `fastapi`/`uvicorn`, `pyzmq` + `msgspec`, `mcp`.
Judge/test only: `transformers` (golden recording; environment **without** fla installed so HF uses its torch
reference path), `hypothesis`, `numpy`, `scipy`.
Baselines: separate pinned venvs under `bench/venvs/` (mini-sglang pins torch < 2.10; nano-vllm needs a flash-attn
build); vLLM as fallback baseline. Baseline torch/CUDA versions are recorded in evidence.

## 2. Milestone ladder (v0)

Each milestone stresses one seam. v0 is pre-1.0: core diffs are allowed but each is listed in the PR under "Core
changes" with its reason; target **zero core diff from M0.2 on**.

| Milestone | Model | Seam proven | Key plugins added |
|---|---|---|---|
| M0 | Qwen3-8B bf16 | core end to end, overlap, graphs, prefix cache, lockfiles, judge L0–L2 | qwen3, gqa, paged_kv, flashinfer attn, std ops, sampler, FCFS, openai frontend, det kernel set |
| M0.1 | Qwen3-8B-FP8 (block 128×128, served W8A16) | quant pipeline; model file untouched | fp8_block dialect, triton W8A16 (uint8 decode) |
| M0.2 | Qwen3-30B-A3B-GPTQ-Int4 (16.9 GB, `desc_act=False`) | model isolation + coarse MoE op + 2nd quant format | qwen3_moe, moe op, gptq dialect, triton W4A16 + fused MoE |
| M0.3 | Qwen3.5-4B → 9B, bf16, text-only | heterogeneous state, checkpoint prefix reuse, weight finalization | qwen3_5, gdn mixer, ssm_checkpoint kind, checkpoint policies |
| M0.4 | Qwen3.5 (built-in MTP) + Qwen3-8B | Rewindable incl. GDN rollback, spec roles, ShapeKey axis, per-Seq depth | mtp drafter, gpu n-gram drafter, verifiers, draft policies |
| M0.5 | Qwen3-8B, Qwen3.5-9B | forward/sample split, host_input, Rewindable grammar | xgrammar processor, parsers, anthropic + responses frontends |
| M0.6 | Qwen3.5-9B agent served by plug-serve | self-improvement end to end, trust enforcement | supervisor, registry, judge L4, improve API, OS identities |

**Acceptance vocabulary** (design §7.10): **run-identical** = byte-identical run to run with fixed batch
composition (regression Plan); **det-identical** = byte-identical across batch-shape/path changes (deterministic
Plan — only where it resolves; not for Qwen3.5 in v0); **bounded** = within judge L2 budgets (production Plan).
Every criterion below names its mode.

**Memory-fit numbers** are computed, not guessed: weights + KV bytes/token × tokens + recurrent state × Seqs +
~1.5 GiB overhead (graphs, workspaces, activations) ≤ 23.99 GiB.

| Model | Weights | KV / token | Target | Total (est.) |
|---|---|---|---|---|
| Qwen3-8B bf16 | 15.26 GiB | 144 KiB | 32K tokens | ≈ 21.3 GiB |
| Qwen3-8B-FP8 (W8A16) | 8.79 GiB | 144 KiB | 64K tokens | ≈ 19.3 GiB |
| Qwen3-30B-A3B-GPTQ-Int4 | 15.8 GiB | 96 KiB | 32K tokens | ≈ 20.3 GiB |
| Qwen3.5-9B bf16 text + MTP | 17.13 GiB | full-attn layers only | 32K tokens + 8 Seqs GDN state + 1 GiB checkpoints | ≈ 21–22 GiB (exact numbers recorded in M0.3) |

Global acceptance for every milestone ≥ M0.1:
- Every earlier model's lock is byte-unchanged, or advanced by `plugserve propagate` with a **run-identical**
  regression snapshot (design §4.6); no re-baseline without approval (model isolation; the regression Plan
  resolves for every model incl. Qwen3.5).
- New plugins pass judge L0–L2 (+ L0-dist for samplers/verifiers/processors, + L1.5 where state is involved).
- Memory-fit criterion of the milestone met; CI lints green (§3.5).

---

## 3. M0 — core end to end (Qwen3-8B bf16)

### 3.1 Spikes (first, time-boxed, results recorded as evidence)

| Spike | Question | Exit |
|---|---|---|
| S1 FlashInfer | paged prefill/decode on sm_86 + cu130 + torch 2.11, head_dim 128/256; JIT pre-build; graph capture with `plan()` in `prepare`; run-to-run byte identity | adopt, or fall back to vendored Triton paged attention |
| S2 W8A16 | Triton GEMM loading FP8 e4m3fn as `uint8`, exact bit-decode to bf16 in the K-loop | L0 exact decode vs torch `float8_e4m3fn` cast on CPU |
| S3 GDN | vLLM-vendored `fla/ops` chunk + recurrent kernels on sm_86 vs HF torch reference; strip `vllm.*` imports (e.g. `vllm.triton_utils`) | max error within budget, no NaN at T=1..8192; no `vllm` import; run-to-run byte identity |
| S4 Determinism | batch-invariant Triton GEMM (vendored vLLM `batch_invariant`) + attention with fixed split, same reduction order for 1-row decode, k+1-row verify and prefill rows | det-identical outputs across batch compositions on Qwen3-8B layers; pinned GEMM/MoE configs free of split-K/atomics |
| S5 Golden | HF fp32 recording on CPU (or `device_map` over GPUs) for 8B; noise floor HF bf16 vs fp32 on frozen prompts | τ and KL/ΔPPL budgets calibrated |

### 3.2 Work items

**A. Repo bootstrap**
1. uv workspace: `core/`, `std/*`, `plugins/*`, `judge/`, `supervisor/`, `bench/` as packages; `plans/` dir.
2. CI scripts: core denylist grep; import-linter contracts + manifest `[dependencies]` check; core LOC cap 6000;
   no `os.environ` in core/std; no `torch.cuda` outside platform/kernels; `core/` diff check
   (`scripts/check_core_diff.sh <base>`); `isolation-check` (= `plugserve propagate` on the PR, design §4.6;
   subsumes lockfile drift); `no-arch-branches` (no `model_type`/arch branches in std/shared units);
   `reference-provenance` (numeric units have `reference.py` + `sem`); pytest.
3. `AGENTS.md`: no claims without measurement; failing-then-passing test per bug fix; never edit `judge/`,
   `plans/`, `registry/`; variants only in `workspace/`.

**B. `plugserve.api`** — dataclasses/Protocols only: Manifest schema (incl. `[dependencies]`), resources, slots +
cardinality (design §4.2), Plan + lockfile format (recipe hash, `[units]` src/sem, `[deploy]`, `[snapshot]`), recipe term
types (`Ref`, `ModelSpec` with `block`/`weights`) + canonical term hash, `Plugin.register(regs)` + Registries, ModelSpec/LayerDecl/
ParamSpec, `finalize_weights`, Op/composite, KernelSpec (+ `config`)/Constraint, Scheme/Layout/QTensor/Dialect/
converter, StateSpec(Paged/Slot)/StateKind, Ledger/Pool, PrefixView + eviction contract (drop/demote/pin), Seq/
SamplingParams, BatchPlan/ShapeKey/DeviceBatch, Rewindable, LogitsProcessor (sharded-vs-gathered flag), Drafter/
DraftPolicy/Verifier/PlanTransform (depth tolerance), FrontendBundle, Event, Platform, launcher slot.

**C. `plugserve.core` runtime**
1. `plugins/`: manifest loading, entry-point discovery, resource matching, resolver → Plan (emits recipe hash and
   per-unit `src`/`sem`, enforces one version per unit), lockfile read/write/diff, lazy import, Protocol checks,
   hard errors.
2. `platform/cuda.py`: streams, events, pinned alloc, graph capture/replay, static fingerprint, single-rank launcher.
3. `memory/`: Ledger (fixed mode; startup profiling; budget split per design §6.1), Pool, slot allocator.
4. `prefix/`: page-granular hash-chain PrefixIndex, namespace, fixed-point hit loop, eviction slot (builtin LRU).
5. `sched/`: Seq lifecycle, token-budget scheduler (phase-free, chunked prefill), FCFS builtin, preemption by
   recompute, per-Seq depth gating.
6. `exec/`: Batch builder (SoA → static buffers), StepLoop with depth 0/1 (design §7.3; depth 2 is v1), device
   Rewindable commit inside the sample graph, deferred frees, ShapeKey bucketing, graph manager (FULL decode/
   verify, EAGER prefill).
7. `control.py`: abort, pause/drain/resume, flush_prefix.
8. `telemetry.py`: Event ring buffer (shared memory), CUDA-event phase timers (diagnostic).
9. `ipc.py`: frontend ↔ engine (zmq + msgspec), token-id protocol with a language-neutral schema file (design
   §7.8). PrefixIndex, allocator and batch builder keep narrow SoA/numpy interfaces (design §3).

**D. std packages**
- Every numeric unit ships its oracle in a separate `reference.py` (defines `sem`).
- `std.ops`: linear (bf16), rmsnorm, add_rmsnorm composite, rope (neox), silu_mul, embedding, lm_head — reference
  + default kernel. (Knobs `rmsnorm.offset` and `rope.rotary_dim` are added in M0.3 with old defaults.)
- `std.blocks.prenorm` (residual wiring that builds from `ModelSpec`, ~150 LOC) and `hf_weights` (weight-map
  helper); `ModelPlugin.build`/`weights` default to them.
- `std.ffn.swiglu`.
- `std.mixers.gqa`: GQA mixer, Planner (FlashInfer `plan()` in `prepare`), q/k-norm option. (`output_gate` knob
  added in M0.3 with default off.)
- `std.state.paged_kv`: paged_kv StateKind (Rewindable: truncate).
- `std.sampling`: sampler (greedy/temperature/top-k/top-p/min-p; seeded per (Seq, position)), penalties.

**E. plugins**
- `models/qwen3`: `recipe.py` (pure, ≤ 80 LOC) + `bundle/` (tokenizer, template, reasoning + tool parsers); default
  `build`/`weights`. Target ≤ 150 LOC excluding the bundle.
- `kernels/flashinfer` (per S1), `kernels/det` (S4 deterministic set), `frontends/openai`.

**F. judge v0** (separate process, client of the engine)
- L0 harness + `budget(dtype, K)`; L0-dist for sampler; L1 contracts; L2 recorder (CPU fp32 / multi-GPU; quantized
  models via `dequant_ref` into the dense HF class) + comparator with calibrated budgets (S5).
- Engine snapshot recorder/comparator (regression Plan bit hash; regression Plan = production bindings at batch 1,
  depth 0, `uncovered` sites recorded — design §7.10) with the ratchet rule; judge-owned oracle store
  (`sem → reference.py`, filled from reviewed commits); per-site closure-diff classifier (design §10.3).

**H. propagation tooling** (supervisor/CLI, outside core, ~400 LOC)
- `plugserve impact`, `plugserve propagate` (classification table incl. `uncovered` sites and additive-knob
  auto-advance, design §4.6), `plugserve propagate --new-model`, `stale` marking for fingerprints without a CI
  runner, `plugserve drift` (report-only), holdback forks under `plugins/legacy/` with 14-day expiry (blocks new
  deployments/promotions, never restarts of the running record). Local `propagate --fingerprint=self` for hosts
  without a CI runner lands with the M0.6 supervisor.
- CI `isolation-check` re-runs propagate on a trusted GPU runner and rejects locks/snapshots it cannot reproduce.
- Deployment-record materialization (worktree/venv cache) is deferred to M0.6, where the supervisor first needs it.

**G. bench**
- `bench/serve.py`: trace replay (ShareGPT-like + agent multi-turn tool traces), TTFT/TPOT/throughput, interleaved
  ABAB runner, client-side timing. Baselines in pinned venvs.

### 3.3 Order

A → B → spikes S1–S5 (parallel) → C1–C3 → D + E with eager, depth 0, no prefix → F (L0/L2) green → C4 prefix →
C6 depth 1 → graphs → frontend + IPC → G bench → perf work. H (impact/propagate, snapshots) lands before M0 closes
so that M0.1+ can use it.

### 3.4 Acceptance

1. L2 bounded vs HF fp32 golden within calibrated budgets.
2. Decode throughput within 10% of mini-sglang (same GPU, model, batch 1/8/64), production Plan.
3. Engine CPU time per step at B=256 ≤ nano-vllm's; if not met, move input prep to GPU before M0.1.
4. Prefix cache on vs off: det-identical outputs on a multi-turn trace; tokens_saved > 0.
5. Depth 1 vs depth 0: det-identical.
6. Deterministic Plan resolves with a candidate for every op (S4 kernels); production Plan lockfile checked in.
7. Memory fit: ≥ 32 concurrent Seqs with ≥ 32K total KV tokens on one A5000 (bf16 weights 15.26 GiB).
8. Core LOC ≤ 6000; lints green.

### 3.5 CI checks (from M0)

`lint-core-denylist`, `import-contracts`, `manifest-deps`, `core-loc-cap`, `no-env-config`,
`no-torch-cuda-outside-platform`, `no-arch-branches`, `reference-provenance`, `isolation-check` (propagate),
`core-diff` (informational in v0, enforcing from v1), `unit`,
`judge-L0`, `judge-L2-goldens` (GPU runner).

---

## 4. M0.1 — quant seam (Qwen3-8B-FP8 → W8A16)

Work: `dialects/fp8_block` (roles `weight`/`weight_scale_inv`; source layout `fp8_block.rowmajor`),
`Scheme(elem="e4m3fn", scales=(block 128×128,))` with `dequant_ref`, `kernels/triton_wxa16` W8A16 (uint8 load +
bit-decode, per S2), deterministic variant with pinned config.

Acceptance: `git diff plugins/models/qwen3` empty; L0 exact decode + GEMM within budget; L2 bounded vs the golden
recorded from `dequant_ref` weights in the dense HF class; Qwen3-8B bf16 lockfile + goldens run-identical; memory fit ≥
64 Seqs / ≥ 64K KV tokens; throughput recorded (no target — weight-only on sm_86).

## 5. M0.2 — model isolation + MoE (Qwen3-30B-A3B-GPTQ-Int4)

Work: `models/qwen3_moe` — `recipe.py` copied from qwen3 with `derived_from` (diff vs qwen3 ≤ ~15 lines: MoE ffn
via `Ref("std.ffn.moe_topk@1", …)`, `mlp_only_layers`, expert weight map); `std.ffn.moe_topk` coarse Op (router,
top-k, experts, combine; reference); `dialects/gptq` (`desc_act=False` only — reject act-order; GPTQ v1 `qzeros`
+1 convention in `dequant_ref`; fp16 scales); `gptq4.rowmajor` layout (+ converter if the kernel needs another);
`kernels/triton_wxa16` W4A16; `kernels/triton_moe` (vendored fused MoE W4A16 path, pinned configs, provenance).

Acceptance: qwen3 locks byte-unchanged (no shared unit changed) or advanced only by propagate; MoE L0 vs reference; L2 bounded vs golden recorded on CPU from
`dequant_ref` weights (≈ 122 GB fp32 — CPU RAM); memory fit ≥ 16 Seqs / ≥ 32K KV tokens; core diff target zero.

## 6. M0.3 — heterogeneous state (Qwen3.5-4B → 9B, text-only)

Qwen3.5 specifics (from HF config/modeling; record any deviation found during implementation):
- 32 layers: 3 GDN : 1 full attention (`full_attention_interval=4`); weights under `model.language_model.*`;
  `model.visual.*` skipped when deployment field `text_only = true`; vocab 248320 (logits buffers sized for it);
  4B ties embeddings, 9B does not.
- Full attention: head_dim 256, `attn_output_gate=True` (`q_proj` output doubled, sigmoid gate on attention
  output), **partial rotary** (`partial_rotary_factor=0.25` → 64 of 256 dims, neox), interleaved mrope [11,11,10]
  which reduces to 1-D rope for text — declare `position_fn="mrope"` and test the equivalence.
- GDN: q/k heads `repeat_interleave` 16→32, `use_qk_l2norm`, `RMSNormGated(silu(z))` output, conv1d kernel 4 over
  8192 channels, `mamba_ssm_dtype=float32` state, `-exp(A_log)` precomputed in `finalize_weights`.
- Zero-centred RMSNorm (`x·(1+w)`): offset-gamma kernel parameter, not folded into bf16 weights.
- MTP (used in M0.4): `mtp.{fc, pre_fc_norm_hidden, pre_fc_norm_embedding, layers.0, norm}`, its own
  full-attention KV StateSpec (namespaced `draft.*`).

Work: `models/qwen3_5`; `mixers/gdn` (vendored kernels per S3; pure-torch recurrent reference); `state/ssm_checkpoint`
(`Slot`, `alignment=1`, `hit_length` = deepest checkpoint ≤ L, publish/adopt with copy into an active slot,
ledger-budgeted checkpoint pool with its own LRU); `policies/checkpoint_every_n`, `policies/checkpoint_turn_anchor`
(anchor from a commit tap; PlanTransform cuts prefill at `floor(anchor/page)·page`).

Acceptance:
- L1.5 bounded (no deterministic GDN kernel in v0): chunked vs one-shot prefill; prefix-hit resume vs recompute;
  graph replay vs eager is **run-identical** (same kernels, same batch composition).
- L2 bounded vs HF fp32 golden (CPU, without fla installed).
- Multi-turn agent trace: prefill tokens saved vs cache-off measured.
- Memory fit (9B bf16 ≈ 19 GB): ≥ 8 Seqs, ≥ 32K KV tokens, checkpoint pool ≥ 1 GiB — record achieved numbers;
  4B: ≥ 32 Seqs.
- New std knobs (`gqa.output_gate`, `rope.rotary_dim`, `rmsnorm.offset`) land with old defaults, and
  `plugserve propagate` auto-advances the Qwen3 and Qwen3-MoE locks with **run-identical** snapshots and zero
  human re-baselines — the first real test of the reuse mechanism (design §4.6).
- `models/qwen3_5/recipe.py` is a copy of qwen3's with `derived_from`; recipe diff recorded.
- Core diff target zero.

## 7. M0.4 — speculative decoding

Work: `drafters/mtp` (consumes ModelSpec `draft_heads`), `drafters/ngram_gpu` (device prompt lookup),
`drafters/lookup_cpu` (declares depth tolerance 0 — exercises per-Seq depth gating), `verifiers/greedy_exact`,
`verifiers/rejection`, `draft_policy/fixed_k`, `draft_policy/cost_model` (Strata-style choose/observe),
`transforms/spec_rows`; GDN `verify_mode="replay"` (default) and `"checkpoint_slots"` (opt-in, `draft_slots ≤ 2`);
drafter KV as a Rewindable.

Acceptance:
- Qwen3-8B + n-gram drafter: spec greedy vs non-spec greedy **det-identical** (deterministic attention handles
  verify rows identically to decode rows, S4).
- Qwen3.5 + MTP: spec greedy vs non-spec greedy **bounded** (GDN path), plus the L1.5 invariant
  "verify + commit(k) vs k decode steps" bounded, for both `replay` and `checkpoint_slots`.
- L0-dist for `rejection` verifier vs plain target sampling.
- Acceptance length and speedup recorded on agent traces; graphs captured for `spec.q ∈ {1, k+1}`.
- CPU lookup drafter runs alongside depth-1 Seqs without stalling them.
- Core diff target zero.

## 8. M0.5 — structured output and tool calls

Work: `grammar/xgrammar` LogitsProcessor (`host_input` bitmask; Rewindable matcher advancing through drafts and
rolling back), parsers with `channel()` in the qwen3/qwen3_5 bundles, `frontends/anthropic` (messages,
count_tokens), `frontends/openai_responses`.

Acceptance: 100% schema conformance on the JSON-schema suite, excluding responses truncated by `max_tokens`
(reported separately); tool-call round trips via OpenAI and Anthropic APIs (Claude-Code-style client on
`/v1/messages`); grammar + n-gram spec vs grammar without spec det-identical on Qwen3-8B, bounded on Qwen3.5 + MTP;
core diff target zero.

## 9. M0.6 — self-improvement MVP

Work:
- `supervisor`: engine lifecycle on baseline/promoted lockfiles (`plans/` vs `registry/plans/`) via deployment records
  `(commit, uv.lock, lock, variant unit srcs)` with cached worktrees/venvs; promoted variant code in
  `registry/units/<src>/`; rebase of active promoted locks after each propagation (design §4.6); health checks,
  Registry (model-scoped), evidence ledger, evaluation scheduler (spare-GPU mode default; in-process A/B for `host`
  policies and `apply="live"` tunables, only after sandbox L0–L1.5 pass, with judge-driven replay traffic and other
  requests held; drain mode as stretch), canary on frontend-measured TTFT/TPOT per request-shape bucket, automatic
  rollback to last-known-good lockfile, quarantine.
- improve API (design §10.2) with token auth, localhost only.
- judge: L4 (client-side timing, trace portfolio, ABAB N ≥ 5, bootstrap CI, teacher-forced re-scoring of sampled
  outputs against the reference Plan, argmax self-consistency, response-count checks), L4-confirm on holdout,
  tunable search inside declared spaces.
- OS identities `ps-agent`, `ps-sandbox`, `ps-engine`, `ps-frontend`, `ps-judge`, `ps-supervisor` with the permissions of design
  §10.6 (separate uids, sandbox network namespace, read-only mounts, per-run scratch).
- Seeded opportunities on **Qwen3-8B** (deterministic Plan resolves → policy variants judged det-identical): one
  deliberately mis-tuned tunable (`apply="live"`, e.g. scheduler prefill token budget) and one deliberately weak
  policy (`draft_policy/fixed_k` with k = 4 for the n-gram drafter).

Acceptance:
1. **Mechanism test** on Qwen3-8B (scripted or external agent): promotes (a) a tunable via in-process A/B and (b) a
   policy variant via the spare-GPU sandbox, each with evidence whose L4 gain CI excludes 0 and whose holdout
   confirm passes; policy variant judged det-identical vs parent.
2. **Served-agent demo**: a Qwen3.5-9B agent served by plug-serve performs at least one judged submission through
   MCP end to end, judged **bounded** (no deterministic Plan for GDN in v0); promotion not required.
3. Injected variant that edits its `reference.py` → rejected (oracle provenance); injected variant that changes a
   `sem` hash or recipe → classified as a proposal, not auto-promoted. Injected variants that (a) bind a kernel
   whose `implements` matches a `sem` present elsewhere in the Plan but not at its own site, (b) override
   `build`/`weights`, or (c) submit a multi-site composite → classified as proposals. Injected wrong-output variant → rejected by
   the judge (L2 or L4 output checks). Injected slow-in-production
   variant that passes on dev traces → rolled back by the canary with no agent action.
4. OS-level tests: every forbidden write/read/network action of design §10.6 fails for `ps-agent`, `ps-sandbox`
   and `ps-engine`; ptrace across identities fails; an in-process A/B is refused for a variant without a sandbox
   L0–L1.5 pass, and the live engine restarts from its lockfile after every in-process A/B.

---

## 10. v1 and v2 (outline; detailed plans written when v0 lands)

**v1 — consumer-GPU story**
- Graph host-node contract spike (native function pointer), then TieredStore + expert offload with CPU miss compute:
  Qwen3-30B-A3B **bf16** (61 GB) on one 24 GB A5000; then Qwen3.5-35B-A3B.
- KV offload to pinned RAM / SSD via `BLOCKED` + per-layer events (demote/pin eviction contract).
- VMM ledger + live rebalance KV ↔ experts (det-identical after shrink/regrow).
- Depth 2, PIECEWISE graphs; radix PrefixIndex if profiles warrant.
- TP=2 on 2× A5000 with zero core diff (launcher plugin, NCCL collectives, sharded logits).
- `update_weights` control-plane op; LoRA; Qwen3.5 VLM (encoder plugin + encoder cache).
- Native FP8 (sm_89+) and NVFP4 (sm_120) kernels — needs additional GPUs.
- Reuse machinery, each at its trigger (design §4.6): async propagation (sweep > 30 min or > 1 re-baseline/day),
  unit store (> 2 holdbacks/unit or > 30 days), closed hook set in `std.blocks` (first sandwich-norm/embed-scale
  model), enforced 100% changed-line coverage (when agents may upstream to std).
- Native (Rust/PyO3) hot structures or Rust frontend only if the profiling triggers in design §12 fire.
- Judge: prompt perturbation, signed goldens, registry fallback hierarchy and `shape_class` scopes; repacked-weight
  artifact cache; drain-window evaluation hardened for single-GPU hosts.

**v2 — breadth**
- DeepSeek-class: DSA indexer (`co_index`), V4 compressed KV (`tokens_per_unit`) — correctness on reduced-layer
  configs locally, perf on datacenter hardware.
- PD disaggregation, EP, DP-attention, PP; draft-model drafter (nested ModelPlugin).
- Tree verification, KVMem-style retrieval window, L3 quality judge, per-plugin process isolation, ROCm/CPU
  platforms, multi-model serving.

---

## 11. Working agreements

- One branch per milestone: `m0-core`, `m0.1-fp8`, …
- One reviewer subagent per milestone for complex/risky code; fix-and-re-review until clean.
- Every perf claim comes with an evidence record (command, hardware fingerprint, plan_id, raw numbers).
- Every bug fix comes with a test that fails before and passes after.
- A PR that changes shared code includes the `plugserve propagate` output; holdbacks expire after 14 days.
