# M0.3 — Qwen3.8-27B: heterogeneous state, prefix reuse, quantized KV: design and implementation plan

Status: **revision 3, reviewed clean** (2026-10-09; revisions 1 and 2 reviewed, every finding resolved, §8). Scope: plan §6,
which merges the former M0.3 (Qwen3.5, heterogeneous state) and M0.3b (Qwen3.8-27B GGUF, quantized KV) by user
decision: Qwen3.8-27B is the only target. Builds on design §5.2, §5.6, §6, §7.3, §7.5, M0.1 (GGUF, `kernels.ggml_cuda`,
the judge's tiers), M0.2 (std.moe, the HF bf16 correctness path) and spikes S1 (quantized KV) and S3 (GDN kernels).

## 1. Goal and acceptance

Serve Qwen3.8-27B — 48 Gated-DeltaNet layers and 16 full-attention layers — from GGUF on a 16 GB RTX 5080 and a 24 GB
RTX 4090, with prefix reuse for agent traffic and a quantized KV cache, and the bf16 HF checkpoint as a correctness
path on an RTX PRO 6000. Acceptance:

1. **Model.** `models/qwen3_5` serves `Qwen3_5ForConditionalGeneration` text-only (Qwen3.5 and Qwen3.8 alike; the
   vision tower and the MTP block are skipped, MTP is M0.4), copied from `models/qwen3` with `derived_from`. L2 passes
   for Qwen3.8-27B: bf16 HF on an RTX PRO 6000 against its fp32 golden; GGUF UD-IQ3_S and UD-Q4_K_M in Tier A and
   Tier B (prefill and decode) against the files' fp32 goldens; per-tensor gates pass; quantization error vs the bf16
   checkpoint reported. A tiny random-weight model of the architecture matches transformers in CPU tests (§4.7).
2. **Bounded checks** (L1.5, the 27B GGUF): chunked vs one-shot prefill; prefix-hit resume vs recompute; graphs vs
   eager on decode; overlap depth 1 vs 0; snapshot trace run-identical over 3 fresh engines.
3. **Prefix reuse.** On the multi-turn agent trace on the RTX 5080: prefill tokens saved and TTFT vs cache off are
   measured; a checkpoint produced by prefill and adopted at its boundary resumes to the same state as recomputing to
   it (bitwise, same path; decode-produced ones are bounded); the core registers a payload only once committed tokens
   cover it (proven in core tests; the end-to-end plant needs speculation, M0.4).
4. **Quantized KV.** q8_0 KV (our Triton kernels, every class) passes L2 against floors that emulate it, and fp8 KV
   (FlashInfer, sm_89 / sm_120) does where its prefill path measures usable (§4.5); the KV error vs bf16 KV is reported
   separately; KV tokens per byte grow by the format's ratio (q8_0 ≥ 1.85×, fp8 2×), and the 5080 fits the 32K
   context that bf16 KV does not (§4.8).
5. **Consumer classes.** RTX 5080: UD-IQ3_S serves one 32K-token context with q8_0 KV at `max_seqs` 2 and two
   checkpoints (estimate in §4.8; the fit records the other configurations). RTX 4090: UD-Q4_K_M. In-process pp512 /
   tg128 and decode at depth 8K against llama-bench at the same commit and KV format (`-ctk/-ctv q8_0` when ours is
   q8_0; both medians of repetitions 2–5), bar ≥ 0.8× (plan §9); decode and agent traces against `llama-server` on
   the 5080.
6. **Locks** for rtx-5080 (UD-IQ3_S) and rtx-4090 (UD-Q4_K_M), the development lock classes (design §4.4).
7. **Reuse.** The one new std knob (the attention output gate, D2) and the core's seams leave every Qwen3 and
   Qwen3-MoE lock's snapshot trace byte-identical and its L2 passing; `propagate` advances them with one approval
   covering every lock (changed lines the Qwen3 traces cannot reach — the gate branch, the KV write op, the new core
   seams — keep auto-advance off, design §4.6).
8. **Kernels.** L0 for every new kernel on sm_86, sm_89 and sm_120a; memcheck clean.
9. **Core diff:** the designed-but-unbuilt state seams of §4.3 only, under its generality rule.

Non-goals: MTP and speculative decoding (M0.4); the vision tower; q4_0 KV (§7); Qwen3.8-Flash-Next (`qwen4_exp`,
plan §10); performance of the bf16 HF path.

## 2. Starting point

- **The model** (`Qwen3_5ForConditionalGeneration`; Qwen3.8-27B: 64 layers, hidden 5120, FFN 17408, vocab 248320,
  untied head). Every 4th layer (3, 7, …) is full attention, the rest Gated DeltaNet (GDN). Pre-norm blocks with
  zero-centred RMSNorm (`x̂·(1 + w)`, fp32); SwiGLU MLP.
  - **Full attention:** 24 q heads, 4 kv heads, head_dim 256 (GQA group 6); `q_proj` emits q and an output gate
    **interleaved per head** (`view(…, H, 512).chunk(2)`); q/k zero-centred RMSNorm over 256; RoPE on the first 64 of
    256 dims, NeoX, θ = 1e7 — the config's interleaved mrope [11,11,10] is plain 1-D RoPE for text (T = H = W = pos);
    out = `o_proj(attn · sigmoid(gate))`. KV: 64 KiB / token (16 layers, bf16).
  - **GDN:** `in_proj_qkv` → q, k (16 heads × 128) and v (48 heads × 128); a depthwise causal conv1d, kernel 4, no
    bias, SiLU, over all 10240 qkv channels; q/k l2-normalised (eps 1e-6), q scaled by 128^-½; q/k heads repeated
    16 → 48 (ratio 3; Qwen3.5: 2); `β = sigmoid(in_proj_b x)`, `g = −exp(A_log) · softplus(in_proj_a x + dt_bias)` in
    fp32; recurrence `S ← e^g S; S += k ⊗ β(v − Sᵀk); o = Sᵀq` with fp32 `S` [48, 128, 128]; output
    `out_proj(w · rmsnorm(o) · silu(z))` per 128-dim head (a plain-`w` gated norm), `z = in_proj_z x`. Recurrent
    state per sequence: SSM 48 × 128 × 128 fp32 (144 MiB for 48 layers) and conv 3 × 10240 in the activation dtype
    (HF; bf16 inputs make bf16 storage lossless; 2.8 MiB) — ≈ 147 MiB.
  - **`output_gate_type: "swish"`** (Qwen3.8 only) is read by neither transformers' `qwen3_5` nor llama.cpp; in
    transformers' `qwen4_exp` it names the GDN norm gate's activation, i.e. the `silu(z)` above. Qwen3.8 is therefore
    numerically Qwen3.5's architecture; the recipe asserts the key is absent or `swish`/`silu` (§4.4).
- **The files** (unsloth `Qwen3.8-27B-GGUF`, arch `qwen35`): UD-IQ3_S 11.21 GiB (the 16 GB target; ≈ 0.3 GiB of it is
  the MTP block we skip), UD-Q4_K_M 15.33 GiB (the 24 GB target). `qwen35.block_count` is **65**: the MTP block is
  `blk.64` (`nextn_predict_layers = 1`). llama.cpp's converter writes every zero-centred norm as `w + 1` in f32 (all
  `*norm.weight` but the GDN norm), stores `ssm_a = −exp(A_log)`, squeezes conv1d, and reorders value heads from
  grouped to **tiled** (v-head j uses q/k head `j mod 16`) in `in_proj_qkv`'s v rows, `in_proj_z/b/a`, `A_log`,
  `dt_bias`, conv1d's v channels and `out_proj`'s input columns. Every ggml type in both files is in
  `kernels.ggml_cuda`'s set.
- **Kernels.** S3 adopted a vendored subset of vLLM's flash-linear-attention copy (vLLM `6dd813e0`, 14 files):
  `chunk_gated_delta_rule` for prefill, `fused_sigmoid_gating_delta_rule_update` for decode; correct vs HF's fp32
  recurrence within 1e-2 relative (HF bf16 itself: 6.9e-3); chunk-aligned (64) resume and graph replay bitwise equal
  to one-shot and eager; a `V_HEAD_TILED` constexpr reads GGUF's order with no repacking; pinned configs; measured on
  sm_86 and sm_120 (sm_89 is new in M0.3). Open from S3: outputs and scratch in our buffers, conv1d and the gated
  norm, the chunk kernel's lack of state indices (gather/scatter), an eager host floor of 0.44–0.65 ms per GDN layer
  per prefill step. S1 measured q8_0 KV in a spike Triton paged attention (int8 + an fp16 scale per 32): at the
  memory roofline on sm_89, ALU-bound on sm_120 (0.60–0.90× bf16 time); fp8 via FlashInfer's tensor-core decode on
  sm_89 / sm_120 (0.54–0.56×), no fast sm_86 fp8 path, fp8 prefill unmeasured.
- **Holon.** The state ABI is designed in full (design §6.2–6.3, §7.3), built for paged state only: Slot specs are
  bound and sized but have no null unit, no MemOp is executed, no kind hook is called, the prefix index switches off
  when any spec is not paged, and prefill is never graph-captured. `RMSNorm(offset)` and `Rope(rotary_dim)` already
  exist (FlashInfer's rope only rejects partial rotary in its `supports`).
- **Baselines** (step 0): llama-bench and `llama-server` at 51ce9c11 on the 5080 (UD-IQ3_S) and the 4090 (UD-Q4_K_M),
  bf16 and q8_0 KV; llama.cpp has its own `qwen35` CUDA path.

## 3. Decisions

- **D1 — Qwen3.8-27B is the only target.** One recipe (`models/qwen3_5`) serves the architecture, so Qwen3.5
  checkpoints load too, but nothing in M0.3 depends on them: no Qwen3.5 lock or acceptance item (a quick Qwen3.5-4B
  run is allowed as an ad-hoc check). The fast inner loop is a **tiny random-weight model** (§4.7). End-to-end
  development runs the 27B GGUF on the A5000 (UD-IQ3_S fits 24 GB); full-size bf16 runs on an RTX PRO 6000.
- **D2 — One new std knob; the existing ones are reused.** `std.mixers.gqa` gains `output_gate: none | sigmoid`
  (q_proj emits per-head-interleaved q and gate; `o_proj(attn · σ(gate))`); default `none` is today's path. Partial
  rotary is `Rope(rotary_dim=64)` (existing; FlashInfer's rope `supports` is relaxed to `rotary_dim ≤ head_dim`,
  its kernel already handles it). Zero-centred norms are `RMSNorm(offset=1)` (existing) for both sources: the GGUF
  map transforms the stored `w + 1` back by `w − 1` in f32 (exact for |w| ≥ 2⁻¹⁶; the result then rounds to bf16
  exactly as HF's checkpoint), so one recipe term serves both sources and no gain loses precision to a bf16 `1 + w`.
- **D3 — GDN is a std mixer with its own ops; kernels come from kernel plugins.** `std.mixers.gdn@1` composes three
  ops, each with a pure-torch reference (the fp32 recurrence is the oracle, as HF's torch path):
  `causal_conv1d@1` (depthwise, kernel K, SiLU, reading and updating the per-sequence conv state),
  `gated_delta_rule@1` (prefill chunks and decode rows over per-sequence state, `v_head_order: grouped | tiled`),
  `gated_rmsnorm@1` (`w · rmsnorm(o) · silu(z)`). Both stateful ops take a per-row **has-initial-state** flag (the
  row's start > 0, computed by the planner on the device): without it they start from zero state, so a slot reused by
  a new or preempted sequence never reads its previous occupant's state and nothing is cleared per admission.
  `kernels.fla` provides `gated_delta_rule` (the vendored S3 subset, its `vendor.patch`, provenance as
  `kernels.ggml_cuda`; the patch also moves the recurrent kernels' skip sentinel from index ≤ 0 to < 0, because state
  units are SeqTable slots and slot 0 is a real sequence, §4.3); `kernels.gdn_triton` provides our conv1d and gated-norm kernels. The tiled order is a fact the
  GGUF map sets; the kernels read it, nothing is repacked. GGUF's `ssm_a` becomes `A_log = log(−ssm_a)` by a value
  transform (HF stores `A_log` in bf16, so the round trip recovers it exactly after the bf16 cast; the gate compares
  after that cast).
- **D4 — Recurrent state is a Slot spec; checkpoints are payloads in a plugin pool.** The GDN mixer declares one
  `Slot` spec (`recurrent`: per layer conv + SSM state, alignment 64 — the chunk size, the granularity at which S3
  showed resume to be bitwise), served by a new state kind `std.state.recurrent`. Its prefix reuse is design §6.3's
  checkpoint: a copy of a sequence's whole recurrent state at a 64-aligned position, held in the kind's own
  ledger-budgeted pool (budget from the deployment), with `hit_length` = the deepest checkpoint ≤ L, `publish` /
  `adopt` as MemOps, and per-payload eviction joint with the chain. Where checkpoints are taken is a policy plugin:
  `checkpoint_prompt_end` (default; at `floor(prompt_end/64)·64`) and `checkpoint_every_n`. A checkpoint is ≈ 147 MiB
  for the 27B, so on the 5080 it competes with KV: the fit records both (acceptance 5). No compression codec (§7).
- **D5 — The core completes the designed state seams** (§4.3), each general: Slot pools indexed by the SeqTable slot
  with a null unit; a scheduler cut hook; MemOps run around the step; the kind hooks called at planning and commit;
  the prefix index over mixed kinds with payloads in their own pools; plugin pools from the build context. ABI
  additions are small and generic (§4.3); no new concept.
- **D6 — Quantized KV is a layout of the KV spec, negotiated with the kernels.** `kv.hnd` gains `kv.hnd.q8_0` (int8
  values + fp16 scale per 32) and `kv.hnd.fp8` (e4m3, saturating cast, scale 1). The attention unit's KV append becomes
  a `kv_write@1` op so the format lives in kernels. Prefill **writes the chunk's K/V first and attends through the
  cache**, so every read (prefix and the chunk's own tokens) sees the stored format, independent of chunking — and one
  floor emulates it exactly (D7). q8_0: our Triton kernels (write, decode, prefill; verify is M0.4) on every class.
  fp8: FlashInfer decode and prefill on sm_89 / sm_120 (its `supports` gains `q_dtype ≠ kv_dtype`); if fp8 prefill
  measures unusable in step 7, fp8 is recorded as decode-only and q8_0 is the deployment's format.
- **D7 — The judge mirrors deployment numerics, as in M0.1/M0.2.** fp32 goldens on the CPU through HF's torch GDN
  path (the judge asserts transformers' swappable hub kernels are off, as S3's harness did); bf16 floors on GPUs,
  torch path; GGUF Tier A on the RTX PRO 6000 (bf16 engine on the file's dequantized weights, as M0.1's, with its own
  bf16 floor — 54 GB fits 96 GB, neither the A5000 nor a CPU run is practical at 27B) and Tier B on the A5000
  (production kernels; floors with q8_1 emulation per path); KV formats emulated in the floor's attention exactly as
  D6 stores and reads them; the KV error itself reported against the bf16-KV floor. Plants in §4.7.
- **D8 — Bars are llama.cpp's.** The GDN kernel source is fla (S3), the oracle is HF, the bar is llama.cpp on the same
  card, file and KV format.
- **D9 — The legibility benchmark runs later, on a model whose parts then exist** (user, 2026-10-09). The old M0.3
  acceptance (plan §9's pivot criterion) had a coding agent add Qwen3.5 to Holon and to a mini-sglang fork from a
  branch point before `qwen3_5` existed; that would measure kernel porting (GDN, recurrent state) on both sides, hours
  per run. It is not an M0.3 gate; plan §9 keeps it, to run on a later model whose units then exist, so it measures
  the plugin structure.

## 4. Design

### 4.1 Packages

| Package | Contents |
|---|---|
| `std/holon_std/mixers/gqa` | `output_gate` (D2) |
| `std/holon_std/mixers/gdn` (new) | `unit.std.mixers.gdn@1`, op contracts `causal_conv1d@1`, `gated_delta_rule@1`, `gated_rmsnorm@1`, `reference.py` |
| `std/holon_std/state/recurrent` (new) | the `recurrent` Slot kind with checkpoint payloads (`hit_length`, `cut`, `publish`, `adopt`) |
| `std/holon_std/policies/checkpoint` (new) | `checkpoint_prompt_end`, `checkpoint_every_n` |
| `plugins/.../models/qwen3_5` (new) | recipe (`derived_from = "models.qwen3@…"`), HF and GGUF weight maps, bundle |
| `plugins/.../kernels/fla` (new) | vendored S3 subset + patch: `gated_delta_rule` provider |
| `plugins/.../kernels/gdn_triton` (new) | our conv1d and gated-RMSNorm kernels |
| `plugins/.../kernels/triton_attn` | `kv_write`; q8_0 KV decode and prefill |
| `plugins/.../kernels/flashinfer` | partial-rotary rope; fp8 KV decode / prefill |
| `core/holon` | §4.3 |
| `judge/holon_judge` | GDN-aware goldens/floors (torch path), KV-format emulation, plants, the prefix bounded checks |

### 4.2 The GDN mixer

`unit.std.mixers.gdn@1(heads_k, heads_v, head_k_dim, head_v_dim, conv_kernel, v_head_order)` at (layer, hidden).
Forward over a step's rows (prefill chunks and decode rows mixed): `in_proj_qkv`, `in_proj_z`, `in_proj_b`,
`in_proj_a` (`qlinear`/`linear` sites, merged where the layouts allow, M0.1's rule) → `causal_conv1d` → 
`gated_delta_rule` (l2-norm, scale, β/g gating inside) → `gated_rmsnorm` → `out_proj`. A row's state is the
`recurrent` pool's unit at the row's SeqTable slot (§4.3); padding rows use the null unit (their writes land there,
as paged padding writes land in the null page) with has-initial-state false. The chunk kernel takes no state indices
(S3), so the mixer gathers the step's prefill rows' states into a contiguous scratch and scatters them back (≈ 12 MiB
per prefill row per layer, plus launches counted in §4.6); slot-indexing the chunk kernel is a perf item (§7). The
planner (design §5.2) builds fla's chunk indices and offsets from the host plan (else the kernel syncs), the
has-initial-state flags, and the decode `graph_key`.

### 4.3 Core: completing the state seams

Every item is a seam design §6–7 already specifies. **Generality rule** (user, 2026-10-09: the core holds only
general mechanisms; the plugin design depends on it):

1. No model, mixer, state or format names in the core (CI's `check_core_denylist` already bans gdn, deltanet, mamba,
   …). Each mechanism serves any `Slot` state (Mamba, a sliding-window ring, a draft model's state) and any mix of
   state kinds; the core never says "checkpoint" — it handles **payloads** a kind publishes.
2. Each mechanism is proven in the core's own tests by a **toy state kind** (a per-sequence running sum with
   payloads at every n tokens) before GDN uses it.
3. No policy in the core: what a payload holds and where it is taken (the kind and its policy plugin), its pool's
   budget (the deployment) and eviction preferences come from plugins; the core calls the hooks and moves the bytes.

| Seam | Today | M0.3 (ABI change in **bold**) |
|---|---|---|
| Slot addressing | Slot pools sized `max_seqs·(1+draft)`, no null unit | a Slot pool's unit for a sequence **is its SeqTable slot** (with draft slots, M0.4: `slot·(1+k)+j`; the core owns the mapping); the pool gets the padding slot's unit as its null unit (all pools expose `pool.null`); no new batch field |
| Prefill cuts | `n = min(remaining, budget)` | before `reserve`, the scheduler asks every kind **`cut(seq, start, n) → 1 ≤ n' ≤ n`** and takes the minimum (a kind asks its policy; default: no cut; cuts follow the kind's alignment); **`SeqView.prompt_len`** is added for policies |
| Publish | never called | optional: a kind that publishes exposes **`next_publish(seq) → position \| None`**; the core compares it per row (O(1), nothing for kinds that never publish, so Qwen3's decode path is unchanged) and, when a row's step ends there, asks `publish(seq, p)` → (payload, MemOps) or None; ops run in the step's `post_sample` phase; at `host_commit` the payload is registered after that commit's paged blocks are inserted (one payload per entry and spec; a duplicate is freed), only if committed tokens cover p, else dropped — before `n > 1` siblings are released |
| Adopt | never called | `admit` takes the fixed-point hit over every kind's `hit_length` (design §6.3); the admitting request holds a reference on the adopted payload unit until its `adopt` MemOp has been enqueued (`pre_forward` of its first scheduled step), so best-effort publishes cannot evict it in between |
| MemOps | `BatchPlan.mem_ops` never executed | run eagerly on the step's stream, before the forward / after the sample (or around a graph replay, never inside a graph: their count varies per step); `Pool.copy` copies unit by unit across pools with no temporary (a 147 MiB gather would break the profiled peak) |
| Prefix index | off when any spec is not paged; refcounts by spec name | entries hold paged units and payloads; **`Payload.pool`** names the payload's pool for refcounts; **per-payload drop** (design §6.3's tombstone) with per-pool victim choice, so a full payload pool evicts its own payloads (interior ones included) and never the KV; publish is best-effort (no free unit: no payload) |
| Plugin pools | `BuildContext` has `static` and `workspace` | **`ctx.pool(name, unit_bytes, segments, min_units, pref_units)`**, design §6.1's plugin pool (step 3 of the ledger's split); a cross-pool unit copy copies the unit's segments in order and requires equal unit shapes |
| Prefill graphs | decode only | optional, per token bucket, if §4.6's measurement says so (the planner's `graph_key` for prefill) |

Not built in M0.3: `fork` (siblings of `n > 1` wait and adopt the first's payloads when there is a free unit, as
today's paged path). The paged path's behaviour is unchanged, so Qwen3 snapshots stay byte-identical (acceptance 7).
The core grows by an estimated ≈ 500 lines (3134 / 6000 now).

### 4.4 Model plugin

`models/qwen3_5`: `derived_from = "models.qwen3@<commit>"`; provides `model.qwen3_5@1`,
`arch.Qwen3_5ForConditionalGeneration@1` (reads `text_config`; skips `model.visual.*` and `mtp.*`) and
`arch.gguf.qwen35@1`. Facts per source: HF from the config; GGUF from `qwen35.*` with `n_layer = block_count −
nextn_predict_layers` and layer types derived from `full_attention_interval` (the GGUF map skips `blk.n_layer..`, the
MTP block). Per layer: `Ref("std.mixers.gqa@1", output_gate="sigmoid", …)` with `Rope(rotary_dim=64)`, or
`Ref("std.mixers.gdn@1", v_head_order=…)`; norms `RMSNorm(offset=1)` except the GDN norm (`offset=0`). Value
transforms in the GGUF map: norms `w − 1` (f32), `A_log = log(−ssm_a)`, conv1d unsqueeze. The recipe rejects what it
does not implement: `output_gate_type` other than absent / swish / silu, `attn_output_gate: false`, an mrope section
that is not text-equivalent. Bundle: tokenizer, chat template (thinking), stop set — from the file for GGUF.

### 4.5 Quantized KV

`std.state.paged` carries a `layout` per spec; GQA declares `kv.hnd.<format>` from the deployment's `kv_dtype`
(`bfloat16` default, `q8_0`, `fp8`), and unit bytes follow the format. `kv_write@1` writes a step's K/V into the cache
(bf16: today's copy; q8_0: per-32 quantize; fp8: saturating cast) before attention, which then reads everything
through the cache (D6). Readers: `triton_attn` (q8_0 decode and prefill, head_dim 256, GQA 6 — S1's spike kernel
productionised, with an L0 reference that dequantizes the cache) and `flashinfer` (fp8 decode and prefill). Step 7
measures fp8 prefill before fp8 enters acceptance.

### 4.6 Performance

- Decode: graphed as today; GDN decode is bandwidth-bound (S3: ~1.46 TB/s on sm_120); the attention part is 16
  layers of GQA-6 at head_dim 256.
- Prefill: eager launches overlap GPU work, so the host floor matters only where host time exceeds GPU time. Step 8
  measures the **GPU-idle fraction** of eager prefill steps (wall time minus summed kernel time) at agent-turn sizes,
  including the extra step a cut causes (§4.3; ≈ 21–31 ms of host floor plus one weight pass ≈ 12 ms on the 5080);
  if idle ≥ 20% at those sizes, prefill is graph-captured per token bucket, else deferred.
- Bars: pp512 / tg128 and decode at depth 8K vs llama-bench; agent and decode traces vs `llama-server` (5080).

### 4.7 Numerics and the judge (thresholds and predictions fixed before measuring)

- **Tiny model.** Built at test time from a small `Qwen3_5` text config (2 layers of each type, ≥ 2 key heads so the
  tiled and grouped orders differ, head ratios 2 and 3) with random weights saved in bf16 (so `(w + 1) − 1` is exact);
  its GGUF is produced by llama.cpp's own converter at 51ce9c11 (`conversion/qwen.py`, `--outtype f32`; `mtp.*`
  tensors added to the checkpoint by hand, since transformers' `Qwen3_5` has no MTP module, so `block_count =
  n_layer + 1` is exercised), and cross-checked against transformers' own GGUF loader where it supports `qwen35`. The engine
  (reference kernels, CPU) must match transformers on the HF weights and on the GGUF.
- **Goldens:** fp32 on the CPU through HF's torch GDN path, streamed per layer (M0.2's `hf_state`; ≈ 108 GB of fp32
  weights each, recorded once per file); dequantized GGUF weights for the GGUF goldens. HF bf16 floors on GPUs
  (`device_map`), torch path. Tier A runs in bf16 (as M0.1's, not M0.2's fp32 CPU run), so it is judged against its
  own bf16 floor and has no fp32 sanity ceiling.
- **L0 for GDN:** against the f64 reference recurrence, relative Frobenius error within 2× that of HF's bf16 torch
  path on the same inputs (measured in the test, not a constant), plus exactness: chunk-aligned resume and graph
  replay bitwise equal to one-shot and eager, verify-free decode equal to the recurrence step by step within the same
  bound.
- **KV formats in the floor:** K and V quantize-dequantized exactly as `kv_write` stores them (q8_0 per 32 with fp16
  scales; fp8 e4m3 saturating), every attention read through them; the bf16-KV floor stays the reference for the KV
  error report.
- **Plants**, verdicts predicted before running:
  - L2 must fail: value heads read grouped when tiled (gross); `A_log` sign (gross); q/k not l2-normalised; the
    attention gate split as one block instead of per head; norm offset 0 on zero-centred norms; rotary over all 256
    dims; stale state on slot reuse (has-initial-state forced true).
  - Bounded must fail: the conv state dropped between prefill chunks (chunked vs one-shot); a payload adopted one
    block early (resume vs recompute); padding rows writing a real slot (graphs vs eager).
  - Core tests must fail (toy kind): a payload registered with a fabricated commit whose accepted tokens do not cover
    it (without speculation every computed token is committed, so the end-to-end plant waits for M0.4); an adopted
    payload unpinned before its copy and evicted by a publish.
  - L0 must fail: conv1d without the state carry; has-initial-state ignored; a row on slot 0 skipped (the sentinel).
- **Bounded checks** (design §7.10, `holon_judge bounded`): chunked vs one-shot prefill (chunk sizes 64·k and not),
  prefix-hit resume vs recompute, graphs vs eager on decode at padded batches, overlap depth 1 vs 0.

### 4.8 Memory on the 16 GB card

From the 30B's measured fit on the same card (2.52 GiB free after 12.0 GiB of weights): UD-IQ3_S's ≈ 10.9 GiB of
loaded weights (MTP skipped) leaves ≈ 3.6 GiB, × 0.9 (the 5080's fraction) ≈ 3.2 GiB, minus the worst step ≈ 0.8 GiB
(the prompt-logprob chunk alone is ≈ 0.47 GiB at the 248K vocab; GDN prefill scratch ≈ 84 KB per token per layer
call) and graphs ≈ 0.3 GiB ≈ 2.1 GiB for state. Recurrent state 147 MiB per Slot unit (`max_seqs` + 1 for the null
unit), a payload 147 MiB, KV 64 KiB / token bf16, 34 KiB q8_0, 32 KiB fp8:

| `max_seqs` | payloads | KV bytes | bf16 KV tokens | q8_0 KV tokens |
|---|---|---|---|---|
| 4 | 4 | ≈ 0.8 GiB | ≈ 13K | ≈ 25K |
| 4 | 0 | ≈ 1.4 GiB | ≈ 23K | ≈ 43K |
| 2 | 2 | ≈ 1.4 GiB | ≈ 23K | ≈ 43K |

So acceptance 5's single 32K context is planned at `max_seqs` 2 with two payloads and q8_0 KV. A generic option to
recover the prompt-logprob chunk's 0.47 GiB (sizing that chunk by bytes rather than rows) is noted for the fit, not
assumed. The fit records KV tokens per (`max_seqs`, payloads, KV format); the deployment's choice goes into the lock.

## 5. Implementation plan

| Step | Work | Checks | Days |
|---|---|---|---|
| 0 | Files (done: 27B HF, UD-IQ3_S, UD-Q4_K_M); baselines on 5080 and 4090 (bf16 and q8_0 KV); the GGUF name map and hparams checked against the checkpoint | baselines | 0.5 |
| 1 | `gqa.output_gate`; FlashInfer partial rotary; their unit tests | the knob's default path byte-identical | 0.5 |
| 2 | `kernels.fla` (vendor S3's subset, patch, provenance), `kernels.gdn_triton` (conv1d, gated norm), op contracts and references, has-initial-state | L0 on sm_86 / sm_89 / sm_120a, both head ratios and orders, graph capture, memcheck | 3 |
| 3 | Core seams (§4.3), proven by the toy state kind in core tests; `std.state.recurrent` | core tests (incl. the commit-coverage and pinning plants); denylist; Qwen3 snapshots unchanged | 4 |
| 4 | GDN mixer, `models/qwen3_5` (HF and GGUF maps), the tiny model fixture; the 27B GGUF end to end on the A5000 | tiny model vs transformers; gates; bounded checks; plants | 3 |
| 5 | 27B goldens and floors; Tier B (A5000), Tier A and bf16 HF L2 + chat on an RTX PRO 6000 | L2 all paths | 3 |
| 6 | Payloads end to end: policies, the agent trace | prefix bounded checks, tokens saved and TTFT | 2 |
| 7 | Quantized KV (D6): `kv_write`, q8_0 Triton decode/prefill, fp8 FlashInfer; floors with KV emulation | L0, L2, KV error report | 3 |
| 8 | Consumer classes: fit, bars, serving, prefill-graph decision (§4.6), tune + locks (5080, 4090), snapshots; Qwen3 / Qwen3-MoE propagation (one approval) | evidence, locks | 3 |
| 9 | Docs: design §5.2, §6.2 (GDN alignment 64, not 1), §6.3; plan §6 status | — | 0.5 |

## 6. Risks

- **The core seams** are the riskiest part: MemOps, cuts and the fixed-point hit across kinds have no production code
  yet. Step 3 lands them against the toy kind before any model uses them.
- **Memory on the 5080:** 147 MiB per Slot unit and per payload; the planned 32K context needs `max_seqs` 2 and q8_0
  KV (§4.8). A payload codec is the follow-up (§7).
- **Cut cost:** each agent turn's prefill becomes two steps (§4.6); measured against the hit's savings.
- **fla numerics:** per-chunk states are bf16 inside the chunk kernel (S3); L2 and the resume check decide; fp32
  per-chunk states are the fallback.
- **27B goldens and Tier A** need ≈ 108 GB of RAM per golden and a PRO 6000 for Tier A: queue time.

## 7. Deferred, deliberately

- q4_0 KV; q8_0 / fp8 verify (M0.4).
- A payload codec (bf16 or quantized checkpoints), judged bounded.
- A slot-indexed chunk kernel (no gather/scatter); an in-step snapshot (fla writing the fp32 state at one chunk
  index), which would remove the extra step a cut causes; PIECEWISE graphs; `fork`.
- MTP (`blk.64`, M0.4; its verify/commit index rows use −1, not 0, for "don't store" under the patched sentinel); the
  vision tower (v1).

## 8. Review findings, resolved

Revision 2: M1 slot 0 skipped by fla's ≤ 0 sentinel → patched to < 0, an L0 plant (D3, §4.7); M2 the depth-1
plant cannot fail without speculation → proven in core tests, end-to-end in M0.4 (acceptance 3, §4.7); M3 an adopted
payload evicted before its copy → the admitting request pins it (§4.3); M4 q8_0 cannot double KV tokens → byte
ratios (acceptance 4). Minors: `SeqView.prompt_len`; the `cut` contract (≥ 1, before `reserve`, aligned); an optional
`next_publish` so decode pays nothing; registration order and duplicates; `ctx.pool` segments; the draft-slot
mapping; the tiny model's MTP tensors and bf16 weights; acceptance 3's bitwise scope; plan wording.

Revision 1 (one reviewer: architecture, kernels, numerics). Blockers: B1 stale state on slot reuse → per-row
has-initial-state in both stateful ops, a plant (D3, §4.7); B2 no step ends at a boundary → the generic `cut` hook
(§4.3, §4.6). Majors: M1 `offset` and `rotary_dim` exist → one new knob (D2); M2 GGUF norm precision → `w − 1` in f32,
`offset=1` for both sources (D2, §4.4); M3 auto-advance impossible → one approval (acceptance 7); M4 payloads in other
pools, unevictable interior payloads, KV evicted for payloads, no plugin pool API → `Payload.pool`, per-payload drop,
best-effort publish, `ctx.pool` (§4.3); M5 MemOps in graphs, gather temporaries → eager around replays, unit copies
(§4.3); M6 fla's null convention and checkpoint words in the core → SeqTable-slot addressing with `pool.null`,
payloads (§4.3); M7 the prefill rule → GPU-idle fraction (§4.6); M8 the depth-1 plant → `checkpoint_every_n` with
stops near a boundary (§4.7); M9 unsupported plants → dropped / moved to bounded (§4.7); M10 27B Tier A → on the PRO
6000, goldens streamed (D7, §4.7); M11 the tiny GGUF → llama.cpp's converter (§4.7); M12 `block_count` 65 (§2, §4.4);
M13 KV-format path → write first, read through the cache, saturating fp8, fp8 prefill measured (D6, §4.5). Minors:
alignment 64 vs design §6.2 (step 9); `fork` not built (§4.3); §4.8 redone (null unit, the prompt-logprob chunk, the
skipped MTP block); bars at the same KV format; sm_89 fla is new evidence; hub kernels asserted off; a GDN L0
budget defined; D-order. Facts corrected: conv state dtype, `A_log` exactness, `block_count`, plan §2 / §6 / §9.
