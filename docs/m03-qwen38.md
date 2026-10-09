# M0.3 — Qwen3.8-27B: heterogeneous state, prefix reuse, quantized KV: design and implementation plan

Status: **revision 1, draft for review** (2026-10-09). Scope: plan §6, which now merges the former M0.3 (Qwen3.5,
heterogeneous state) and M0.3b (Qwen3.8-27B GGUF, quantized KV) by user decision: Qwen3.8-27B is the target, Qwen3.5
shares its architecture. Builds on design §5.2, §5.6, §6, §7.3, §7.5, M0.1 (GGUF, `kernels.ggml_cuda`, the judge's
tiers), M0.2 (std.moe, the HF bf16 correctness path) and spikes S1 (quantized KV) and S3 (GDN kernels).

## 1. Goal and acceptance

Serve Qwen3.8-27B — 48 Gated-DeltaNet layers and 16 full-attention layers — from GGUF on a 16 GB RTX 5080 and a 24 GB
RTX 4090, with prefix reuse for agent traffic and a quantized KV cache, and the bf16 HF checkpoint as a correctness
path on an RTX PRO 6000. Acceptance:

1. **Model.** `models/qwen3_5` serves `Qwen3_5ForConditionalGeneration` text-only (Qwen3.5 and Qwen3.8 alike; the
   vision tower and the GGUF's MTP block are skipped, MTP is M0.4), copied from `models/qwen3` with `derived_from`.
   L2 passes for Qwen3.8-27B: bf16 HF on an RTX PRO 6000 against its fp32 golden; GGUF UD-IQ3_S and UD-Q4_K_M in Tier A
   and Tier B (prefill and decode) against the files' fp32 goldens; per-tensor gates pass; quantization error vs the
   bf16 checkpoint reported. A tiny random-weight model of the architecture (both head ratios) matches transformers
   in CPU tests.
2. **Bounded checks** (L1.5, on Qwen3.8-27B GGUF): chunked vs one-shot prefill; prefix-hit resume vs
   recompute; graphs vs eager on decode; snapshot trace run-identical over 3 fresh engines.
3. **Prefix reuse.** On the multi-turn agent trace, prefill tokens saved vs cache off are measured on the RTX 5080;
   recurrent-state checkpoints are published only at boundaries covered by committed tokens (an overlap-depth-1
   test); a checkpoint adopted at a boundary resumes to the same state as recomputing it (bitwise on the same path).
4. **Quantized KV.** q8_0 KV (our Triton kernels, every class) and fp8 KV (FlashInfer, sm_89 / sm_120) pass L2 with
   floors that emulate the KV format; their error vs bf16 KV is reported separately; on the 5080 they at least
   double the KV tokens that fit vs bf16 KV.
5. **Consumer classes.** RTX 5080: UD-IQ3_S serving a 32K-token context at `max_seqs` 4 with one checkpoint per
   sequence (q8_0 KV; estimate in §4.8). RTX 4090: UD-Q4_K_M. In-process
   pp512 / tg128 and long-context decode against llama-bench at the same commit (both medians of repetitions 2–5),
   bar ≥ 0.8× (plan §9); decode and agent traces against `llama-server` on the 5080; memory fit recorded.
6. **Locks** for rtx-5080 (UD-IQ3_S) and rtx-4090 (UD-Q4_K_M), the development lock classes (design §4.4).
7. **Reuse.** New std knobs (attention output gate, partial rotary, RMSNorm offset) land with defaults that keep the
   old behaviour, and `propagate` auto-advances every Qwen3 and Qwen3-MoE lock (byte-identical snapshots, no approval).
8. **Kernels.** L0 for every new kernel on sm_86, sm_89 and sm_120a; memcheck clean.
9. **Core diff:** the designed-but-unbuilt state seams of §4.3 only, no new concepts (the plan's "zero" was written
   before it was known that these seams were deferred to this milestone; see §4.3).

Non-goals: MTP and speculative decoding (M0.4); the vision tower; q4_0 KV (§7); Qwen3.8-Flash-Next (its own
architecture, `qwen4_exp`, 512 experts: after M0.3, see plan §10); performance of the bf16 HF path.

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
    16 → 48 (ratio 3; Qwen3.5: 16 → 32, ratio 2); `β = sigmoid(in_proj_b x)`, `g = −exp(A_log) · softplus(in_proj_a x
    + dt_bias)` in fp32; recurrence `S ← e^g S; S += k ⊗ β(v − Sᵀk); o = Sᵀq` with fp32 `S` [48, 128, 128];
    output `out_proj(w · rmsnorm(o) · silu(z))` per 128-dim head (a plain-`w` gated norm), `z = in_proj_z x`.
    Recurrent state per sequence: conv 3 × 10240 and SSM 48 × 128 × 128, fp32 in HF and llama.cpp — ≈ 150 MiB for the
    48 layers (Qwen3.5-4B / 9B: ≈ 50 MiB).
  - **`output_gate_type: "swish"`** (Qwen3.8 only) is read by neither transformers' `qwen3_5` nor llama.cpp; in
    transformers' `qwen4_exp` it names the GDN norm gate's activation, i.e. the `silu(z)` above. Qwen3.8 is therefore
    numerically Qwen3.5's architecture; the recipe asserts the key is absent or `swish`/`silu` (§4.4).
- **The files** (unsloth `Qwen3.8-27B-GGUF`, arch `qwen35`): UD-IQ3_S 11.21 GiB (the 16 GB target), UD-Q4_K_M
  15.33 GiB (the 24 GB target). llama.cpp's converter adds 1 to every norm weight except the GDN norm's, stores
  `ssm_a = −exp(A_log)`, squeezes conv1d, and reorders value heads from grouped to **tiled** (v-head j uses q/k head
  `j mod 16`) in `in_proj_qkv`'s v rows, `in_proj_z/b/a`, `A_log`, `dt_bias`, conv1d's v channels and `out_proj`'s
  input columns. The MTP block is in the main file as `blk.64.*` (`nextn.*`). Hparams `qwen35.ssm.*`,
  `qwen35.full_attention_interval`, `qwen35.rope.dimension_count = 64`.
- **Kernels.** S3 adopted a vendored subset of vLLM's flash-linear-attention copy (vLLM `6dd813e0`, 14 files):
  `chunk_gated_delta_rule` for prefill, `fused_sigmoid_gating_delta_rule_update` for decode; correct vs HF's fp32
  recurrence within 1e-2 relative (HF bf16 itself: 6.9e-3); chunk-aligned resume and graph replay bitwise equal to
  one-shot and eager; a `V_HEAD_TILED` constexpr reads GGUF's order with no repacking; pinned configs (no autotune).
  Open from S3: outputs and scratch in our buffers, conv1d and the gated norm (HF torch in the spike), slot indexing
  in the chunk kernel (else gather/scatter), padded rows zeroed, an eager host floor of 0.44–0.65 ms per GDN layer per
  prefill step. S1 measured q8_0 KV in our Triton paged attention (int8 + an fp16 scale per 32): at the memory
  roofline on sm_89 (0.49–0.57× bf16 time), ALU-bound on sm_120 (0.60–0.90×; 1.1–1.7× FlashInfer fp8's time); fp8
  via FlashInfer's tensor-core decode on sm_89 / sm_120 (0.54–0.56×), sm_86 has no fast fp8 path.
- **Holon.** The state ABI is designed in full (design §6.2–6.3, §7.3): `StateSpec` with `Paged | Slot` shapes,
  `StateKind` with `hit_length / publish / adopt / fork / commit`, MemOps run by the core around the step, a
  PrefixIndex whose hit is the fixed point over every kind. Built so far: paged state only. `bind_state` binds Slot
  specs and the ledger sizes their pools, but the batch exposes no slot units, the pool has no null unit for graph
  padding rows, no MemOp is ever executed, no kind hook is called, and the prefix index switches itself off when any
  spec is not paged. Prefill is never graph-captured. These are this milestone's core work (§4.3).
- **Baselines** (step 0): llama-bench and `llama-server` at 51ce9c11 on the 5080 (UD-IQ3_S) and the 4090
  (UD-Q4_K_M); llama.cpp has its own `qwen35` CUDA path (GDN and SSM-conv ops).

## 3. Decisions

- **D1 — Qwen3.8-27B is the only target.** One recipe (`models/qwen3_5`) serves the architecture, so Qwen3.5
  checkpoints load too, but nothing in M0.3 depends on them: no Qwen3.5 lock or acceptance item (a quick Qwen3.5-4B run
  is allowed as an ad-hoc check). The fast inner loop is a **tiny random-weight model** built at test time from a
  small `Qwen3_5` config (two layers of each type; v/k head ratios 2 and 3; both value-head orders through a GGUF
  writer, as M0.2's tiny MoE fixture), checked against transformers on the CPU in seconds. End-to-end development
  runs the 27B GGUF on the A5000 (UD-IQ3_S fits 24 GB); the bf16 HF path at full size runs on an RTX PRO 6000.
- **D2 — Three std knobs, defaults unchanged (the reuse test).** `std.mixers.gqa` gains `output_gate: none |
  sigmoid` (q_proj emits per-head-interleaved q and gate, `o_proj(attn · σ(gate))`) and `rotary_dim` (partial NeoX
  RoPE; default = head_dim); `std.ops` RMSNorm gains `offset` (`x̂·(offset + w)`; default 0). Defaults reproduce
  Qwen3 exactly, so Qwen3 / Qwen3-MoE locks auto-advance (acceptance 7). The HF map uses `offset=1`; the GGUF map
  uses `offset=0` because the file stores `w + 1` — the source picks the knob, no weight transform. A per-head
  interleaved gate is a fact of the unit, not a weight permutation, so the gqa unit and the GGUF map need none.
- **D3 — GDN is a std mixer with its own ops; kernels come from a kernel plugin.** `std.mixers.gdn@1` composes
  three ops, each with a pure-torch reference (the fp32 recurrence is the oracle, as HF's torch path):
  `causal_conv1d@1` (depthwise, kernel K, SiLU, reading and updating the per-sequence conv state),
  `gated_delta_rule@1` (prefill chunks and decode rows over per-sequence state, `v_head_order: grouped | tiled`),
  `gated_rmsnorm@1` (`w · rmsnorm(o) · silu(z)`). Facts carry heads, head dims, the head ratio and the order.
  `kernels.fla` provides `gated_delta_rule` (the vendored S3 subset, its `vendor.patch`, provenance as
  `kernels.ggml_cuda`) and `kernels.gdn_triton` provides our conv1d and gated-norm kernels (fused with the β / g
  gating where it saves a launch). The tiled order is a fact the GGUF map sets (`v_head_order="tiled"`); the
  kernels read it, nothing is repacked. GGUF's `ssm_a = −exp(A_log)` becomes `A_log = log(−ssm_a)` by a value
  transform in the GGUF map (one f32 rounding; the gate checks it against the checkpoint).
- **D4 — Recurrent state is a Slot spec; checkpoints are a second pool of the same kind.** The GDN mixer declares
  one `Slot` spec per layer group (`recurrent`: conv + SSM state, fp32, alignment = the chunk size 64), served by a
  new state kind `std.state.recurrent`. Prefix reuse for it is the design's `ssm_checkpoint`: a checkpoint is a copy
  of a sequence's whole recurrent state at a page-aligned position, held in a ledger-budgeted checkpoint pool
  (bytes from the deployment, default: one checkpoint per running sequence), with `hit_length` = the deepest
  checkpoint ≤ L on the chain, `publish` / `adopt` as device MemOps, and eviction joint with the paged KV's chain
  (design §6.3). Placement is a policy: `checkpoint_prompt_end` (default; cut at `floor(end/page)·page`) and
  `checkpoint_every_n`. A checkpoint is 150 MiB for the 27B, so on the 5080 it competes with KV for the same bytes:
  the fit records both (acceptance 5). No compression codec in M0.3 (§7).
- **D5 — The core completes the designed state seams, nothing more** (§4.3): slot units per row in the batch and a
  null slot unit for padding rows; the MemOp runner (adopt before the forward, publish after the sample, both in
  stream order and graph-safe); the kind hooks called where design §6.3 places them; the prefix index over mixed
  kinds with the fixed-point hit; and prefill graphs per token bucket if step 8's measurement needs them (§4.6).
  No new ABI type; `DeviceBatch` gains one field (`slot_units`), the planner one optional key (prefill `graph_key`).
- **D6 — Quantized KV is a layout of the KV spec, negotiated with the kernels.** `kv.hnd` gains `kv.hnd.q8_0` (int8
  values + fp16 scale per 32, as S1) and `kv.hnd.fp8` (e4m3, per-tensor scale 1). q8_0: our Triton kernels — the
  quantizing KV write, paged decode and prefill (verify is M0.4) — on every class. fp8: FlashInfer decode and
  prefill on sm_89 / sm_120 (no fast sm_86 path; the layout is simply unsupported there). The deployment chooses the
  layout (`kv_dtype`); `tune` picks providers per class. q4_0 is deferred (§7).
- **D7 — The judge mirrors deployment numerics, as in M0.1/M0.2.** fp32 goldens on the CPU (HF's torch GDN path,
  never fla); bf16 floors on GPUs with HF's torch path; GGUF Tier A (production weight source, reference kernels,
  CPU fp32) and Tier B (production kernels; floors with q8_1 emulation per path, M0.1); KV formats emulated in the
  floor's attention (quantize-dequantize K and V as stored) so a q8_0 or fp8 run is judged against a floor that pays
  the same rounding; the KV error itself is reported against the bf16-KV floor. Plants in §4.7.
- **D9 — Open (user): the legibility benchmark.** The old M0.3 acceptance (and the plan §9 pivot criterion) has a
  coding agent add Qwen3.5 to Holon and to a mini-sglang fork from a branch point before `qwen3_5` exists, ≥ 3 runs
  per side. Before this milestone that means implementing GDN, recurrent state and its kernels on both sides —
  hours of agent time per run, ≥ 6 runs. Options: (a) run it as specified, after M0.3's code is final, from the
  pre-M0.3 branch point; (b) run it on a model whose parts then exist (e.g. a GDN model variant after M0.3) so it
  measures the plugin structure rather than kernel porting; (c) drop it as an M0.3 gate and keep it in §9 for a
  later milestone. Not in §1's acceptance until decided.
- **D8 — Bars are llama.cpp's.** The kernel source for GDN is fla (S3), the oracle is HF, the bar is llama.cpp on the
  same card and file (its own `qwen35` CUDA path).

## 4. Design

### 4.1 Packages

| Package | Contents |
|---|---|
| `std/holon_std/mixers/gqa` | `output_gate`, `rotary_dim` (D2) |
| `std/holon_std/ops` | RMSNorm `offset` (D2) |
| `std/holon_std/mixers/gdn` (new) | `unit.std.mixers.gdn@1`, op contracts `causal_conv1d@1`, `gated_delta_rule@1`, `gated_rmsnorm@1`, `reference.py` |
| `std/holon_std/state/recurrent` (new) | the `recurrent` Slot kind with checkpoints (`hit_length`, `publish`, `adopt`, `fork`) |
| `std/holon_std/policies/checkpoint` (new) | `checkpoint_prompt_end`, `checkpoint_every_n` |
| `plugins/.../models/qwen3_5` (new) | recipe (`derived_from = "models.qwen3@…"`), HF and GGUF weight maps, bundle |
| `plugins/.../kernels/fla` (new) | vendored S3 subset + patch: `gated_delta_rule` provider |
| `plugins/.../kernels/gdn_triton` (new) | our conv1d, gated RMSNorm (+ gating) kernels |
| `plugins/.../kernels/triton_attn` | q8_0 KV: quantizing write, decode, prefill |
| `plugins/.../kernels/flashinfer` | fp8 KV decode / prefill |
| `core/holon` | §4.3 |
| `judge/holon_judge` | GDN-aware goldens/floors (torch path), KV-format emulation, plants, the checkpoint bounded check |

### 4.2 The GDN mixer

`unit.std.mixers.gdn@1(heads_k, heads_v, head_k_dim, head_v_dim, conv_kernel, v_head_order)` at (layer, hidden).
Forward over a step's rows (prefill chunks and decode rows mixed): `in_proj_qkv`, `in_proj_z`, `in_proj_b`,
`in_proj_a` (four `qlinear`/`linear` sites, merged where the layouts allow, M0.1's rule) → `causal_conv1d` (reads
and writes the conv state of each row's slot) → `gated_delta_rule` (reads the SSM state of each row's slot, writes
the final state back; l2-norm, scale, β/g gating inside) → `gated_rmsnorm` → `out_proj`. State is addressed by the
batch's `slot_units[recurrent]` (§4.3): one unit per running sequence, unit 0 the null unit padding rows read and
write. The chunk kernel takes no state indices (S3), so the mixer gathers the step's prefill states into a
contiguous scratch and scatters them back — a few MiB per prefill row per layer; slot-indexing the chunk kernel is a
perf item (§7). Facts: `v_head_order` (`grouped` for HF, `tiled` for GGUF), head counts, dims, and `dtype`.

The planner (design §5.2) builds the chunk indices and offsets the fla prefill kernel needs from the host plan
(else it syncs), and a `graph_key` for decode (and, if adopted, prefill buckets).

### 4.3 Core: completing the state seams

Every item is a seam design §6–7 already specifies; none adds a concept.

| Seam | Today | M0.3 |
|---|---|---|
| Slot units in the batch | `DeviceBatch` has no per-spec slot index | `slot_units: dict[spec, Tensor[rows]]`, filled like block tables |
| Null slot unit | Slot pools have none | unit 0 reserved; graph padding rows use it (as paged pools' null unit) |
| MemOps | `BatchPlan.mem_ops` never executed | the executor runs `pre_forward` ops (adopt) before the forward and `post_sample` ops (publish) after the sample, in stream order, captured with the step's graph when the step is graphed |
| Kind hooks | never called | `admit` asks every kind's `hit_length` and takes the fixed point (design §6.3); `adopt` MemOps on a hit; `publish` at the policy's boundaries, registered at `host_commit` only once committed tokens cover them (depth 1) |
| Prefix index | off when any spec is not paged | entries carry each kind's payload (KV pages, a checkpoint unit); joint chain eviction |
| Prefill graphs | decode only | optional, per token bucket, if §4.6's measurement says so |

`StateManager.admit` and `publish`, `PrefixIndex` and the executor change; the paged path is unchanged and its
snapshots stay byte-identical (Qwen3 / Qwen3-MoE propagate with zero approvals; checked in step 3). Estimated
≈ 400 core lines; the core stays under its LOC cap (3134 / 6000 now).

### 4.4 Model plugin

`models/qwen3_5`: `derived_from = "models.qwen3@<commit>"`; provides `model.qwen3_5@1`,
`arch.Qwen3_5ForConditionalGeneration@1` (reads `text_config`; skips `model.visual.*` and `mtp.*`) and
`arch.gguf.qwen35@1` (skips `blk.<n_layer>.*`, the MTP block). Per layer: `layer_types[i]` selects
`Ref("std.mixers.gqa@1", output_gate="sigmoid", rotary_dim=64, …)` or `Ref("std.mixers.gdn@1", …)`; norms
`Ref(rmsnorm, offset=1 if source is HF else 0)` except the GDN norm (plain `w`). Facts read per source (HF config or
`qwen35.*` hparams). The recipe rejects configs it does not implement: `output_gate_type` other than absent / swish /
silu, `attn_output_gate: false`, an mrope section that is not text-equivalent. Bundle: tokenizer, chat template
(thinking), stop set — from the file for GGUF (M0.1 D3).

### 4.5 Quantized KV

`std.state.paged` already carries a `layout` per spec. GQA declares `kv.hnd.<format>` from the deployment's
`kv_dtype` (`bfloat16` default, `q8_0`, `fp8`); unit bytes follow the format. Writes: the attention unit's KV append
becomes a `kv_write@1` op (bf16: today's copy; q8_0: quantize per 32; fp8: cast) so the format lives in kernels, not
in the unit. Readers: `triton_attn` (q8_0 decode and prefill — S1's kernel productionised, head_dim 256, GQA 6) and
`flashinfer` (fp8 decode and prefill on sm_89 / sm_120). L0 against a reference that dequantizes the cache.

### 4.6 Performance

- Decode: graphed as today; GDN decode is bandwidth-bound (S3: ~1.46 TB/s on sm_120), the attention part is 16
  layers of GQA-6 with head_dim 256.
- Prefill: S3's eager host floor (0.44–0.65 ms per GDN layer, ≈ 21–31 ms per step for the 27B) matters for agent
  turns of a few hundred tokens. Step 8 measures TTFT on the agent trace; if the floor is ≥ 20% of TTFT, prefill is
  graph-captured per token bucket (D5) — else deferred.
- Bars: pp512 / tg128 and tg128 at depth 8K vs llama-bench; agent and decode traces vs `llama-server` (5080).

### 4.7 Numerics and the judge (thresholds and predictions fixed before measuring)

- **Goldens:** fp32 on the CPU through HF's torch GDN path (the 27B ≈ 108 GB of RAM; dequantized GGUF weights for the
  GGUF goldens, M0.1). HF bf16 floors on GPUs (`device_map`), torch path. A sanity ceiling for Tier A: KL p99 ≤ 1e-4.
- **KV formats in the floor:** K and V quantize-dequantized as stored (q8_0 per 32 with fp16 scales; fp8 e4m3 cast)
  in the floor's attention; the bf16-KV floor stays the reference for reporting the KV error.
- **Plants**, verdicts predicted before running:
  - L2 must fail: value heads read grouped when tiled (gross); `A_log` sign (gross); the conv state dropped between
    prefill chunks; q/k not l2-normalised; the attention gate split as one block instead of per head; RMSNorm offset
    0 on the HF source; rotary over all 256 dims.
  - Bounded must fail: a checkpoint adopted one page early (resume vs recompute); a checkpoint published before its
    tokens are committed (the depth-1 test); padding rows writing a real slot (graphs vs eager).
  - L0 must fail: conv1d without the state carry; the delta rule with bf16 state.
- **Bounded checks** (design §7.10, `holon_judge bounded`): chunked vs one-shot prefill (chunk sizes 64·k and not),
  prefix-hit resume vs recompute, graphs vs eager on decode at padded batches.

### 4.8 Memory on the 16 GB card

Estimated from the 30B's measured fit on the same card (2.52 GiB free after 12.0 GiB of weights; worst step
0.31 GiB, graphs 0.26 GiB): UD-IQ3_S's 11.21 GiB leaves ≈ 3.3 GiB, × 0.9 (the 5080's fraction) ≈ 2.97 GiB, minus
≈ 0.6 GiB for the worst step (GDN prefill scratch ≈ 84 KB per token per layer call) and graphs ≈ 2.4 GiB for state.
Per running sequence 150 MiB of recurrent state; per checkpoint 150 MiB; KV 64 KiB / token bf16, 34 KiB q8_0, 32 KiB
fp8. At `max_seqs` 4:

| checkpoints | KV bytes | bf16 KV tokens | q8_0 KV tokens |
|---|---|---|---|
| 0 | ≈ 1.8 GiB | ≈ 29K | ≈ 55K |
| 4 (one per sequence) | ≈ 1.2 GiB | ≈ 20K | ≈ 37K |

So acceptance 5's 32K context needs q8_0 (or fp8) KV once checkpoints are on. The fit records KV tokens per
(`max_seqs`, checkpoints, KV format), and the deployment's choice goes into the lock's engine settings.

## 5. Implementation plan

| Step | Work | Checks | Days |
|---|---|---|---|
| 0 | Files (27B HF, UD-IQ3_S, UD-Q4_K_M); llama-bench / llama-server baselines on 5080 and 4090; GGUF name map and hparams checked against the checkpoint | baselines | 0.5 |
| 1 | Std knobs (D2) and their unit tests; `propagate` Qwen3 / Qwen3-MoE | auto-advance, zero approvals | 1 |
| 2 | `kernels.fla` (vendor S3's subset, patch, provenance), `kernels.gdn_triton` (conv1d, gated norm), op contracts and references | L0 on sm_86 / sm_89 / sm_120a, both head ratios, both orders, graph capture, memcheck | 3 |
| 3 | Core seams (§4.3) with core tests; `std.state.recurrent` | core tests; Qwen3 snapshots unchanged | 3 |
| 4 | GDN mixer, `models/qwen3_5` (HF and GGUF maps), the tiny model fixture; the 27B GGUF end to end on the A5000 | the tiny model vs transformers (CPU); 27B Tier A on the CPU; bounded checks; plants | 3 |
| 5 | Qwen3.8-27B: gates, Tier B (A5000 floors), plants; bf16 HF on an RTX PRO 6000 (L2, chat) | L2 all paths | 2.5 |
| 6 | Checkpoints: kind hooks, policies, agent trace | prefix bounded checks, depth-1 test, tokens saved | 2 |
| 7 | Quantized KV (D6): `kv_write`, q8_0 Triton decode/prefill, fp8 FlashInfer; floors with KV emulation | L0, L2, KV error report | 3 |
| 8 | Consumer classes: fit, bars, serving, prefill-graph decision (§4.6), tune + locks (5080, 4090), snapshots | evidence, locks | 3 |
| 9 | Docs: design §5.2 / §6 / plan §6 status | — | 0.5 |

## 6. Risks

- **The core seams** are the riskiest part: MemOps inside graphs and the fixed-point hit across kinds have no
  production code yet. Step 3 lands them with core tests before any model uses them.
- **Checkpoint memory:** 150 MiB each on a 16 GB card; the default (one per running sequence) may leave too little
  KV at 4 sequences — the fit decides, and a compressed checkpoint codec is the follow-up (§7).
- **Prefill host floor** (§4.6) could cost agent TTFT; the decision is measured.
- **fla numerics:** the chunk kernel keeps per-chunk states in bf16 (S3); L2 and the bounded resume check decide
  whether that is within the floor; fp32 per-chunk states are the fallback.
- **CPU goldens for the 27B** (≈ 108 GB RAM, torch recurrence): hours, not minutes; recorded once per file.

## 7. Deferred, deliberately

- q4_0 KV (S1: not implemented anywhere), and q8_0 verify (M0.4).
- A checkpoint codec (bf16 or quantized checkpoints) — judged bounded, after M0.3's measurements.
- Slot-indexed chunk kernel (removes the gather/scatter), PIECEWISE graphs.
- MTP (the GGUF's `blk.64`, M0.4); the vision tower (v1).

## 8. Review findings, resolved

(none yet)
