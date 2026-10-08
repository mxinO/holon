# M0.2 — Mixture of experts on a 16 GB card: design and implementation plan

Status: **revision 3.1, for review** (2026-10-08). Revisions 1 and 2 were challenged by three reviewers
(architecture/API, kernels/performance, numerics/judge); §8 maps every finding to its resolution. Scope: plan §5. Builds on M0.1
(`docs/m01-gguf.md`: the GGUF source, `kernels.ggml_cuda`, the judge's tiers) and spike S2 (MoE sections).

## 1. Goal and acceptance

Serve Qwen3-30B-A3B (128 experts, 8 per token, ≈ 3B active) from llama.cpp GGUF files with llama.cpp's own MoE CUDA
kernels, on a 16 GB RTX 5080 and on 24–32 GB cards. Acceptance:

1. `models/qwen3_moe` is `models/qwen3` copied with `derived_from`; its recipe diff to qwen3 is ≤ ~15 lines. Core
   diff: zero. The bf16 Qwen3 locks do not change. The GGUF Qwen3 locks change only in `kernels.ggml_cuda`'s `src`
   pin: `propagate` shows a byte-identical snapshot trace, L2 passing, `[sites]`, `[engine]` and the recipe hash
   unchanged; approved once.
2. L0 for the MoE ops on sm_86, sm_89, sm_120a, every supported type on both kernel paths (§4.6), memcheck clean.
3. L2 for UD-IQ3_XXS and Q4_K_M: Tier A and Tier B (prefill and decode) pass; each planted bug fails at the level
   and margin recorded in §4.6; the gates pass; the quantization error vs bf16 is reported.
4. Snapshot trace run-identical over 3 fresh engines (UD-IQ3_XXS on rtx-5080, Q4_K_M on rtx-a5000); graphs vs eager
   bounded on MoE decode, on both kernel paths.
5. RTX 5080: UD-IQ3_XXS with ≥ 16K KV tokens. RTX 4090, 5090, A5000: Q4_K_M. In-process pp512 / tg128 against
   llama-bench at the same commit on every class, both sides as the median of repetitions 2–5 (bar on the 5080:
   ≥ 0.8× S2's 7462 / 248.2, i.e. pp512 ≥ 5970, tg128 ≥ 199); decode and agent traces against `llama-server` on the
   5080. UD-Q3_K_XL (plan §5) is optional: recorded on the 5080 if time allows, not required.
6. GGUF locks for rtx-5080 (UD-IQ3_XXS) and rtx-a5000, rtx-4090, rtx-5090 (Q4_K_M).

Non-goals: serving the bf16 HF checkpoint (§3 D1); expert offload / expert cache (v1, §3 D2); expert parallelism;
Qwen3.5 (M0.3).

## 2. Starting point

- **The model.** 48 layers, all MoE; hidden 2048; attention 32 / 4 heads × 128 with q/k norm (as Qwen3); 128
  experts, top-8, expert width 768, softmax routing renormalised over the top-8; vocab 151936, untied head. KV:
  96 KiB / token.
- **The files** (unsloth `Qwen3-30B-A3B-GGUF`): UD-IQ3_XXS 12.0 GiB, Q4_K_M 17.28 GiB (both cached), UD-Q3_K_XL
  12.88 GiB (optional). UD-IQ3_XXS: experts IQ3_XXS / IQ3_S (gate, up), IQ3_XXS / IQ4_XS / IQ3_S (down); attention
  IQ4_XS / Q5_K / Q6_K; router `ffn_gate_inp` **F32** [128, 2048]; embedding Q4_K, head Q6_K. Gate and up share a type
  in every layer, as do q / k / v. Expert stacks are `[E, out, in]` in HF's expert order with no permutation
  (measured: NMSE vs HF 0.035 per expert; 2.0 for gate↔up; 2.3–2.6 for a neighbouring expert). The GGUF lacks the
  gating function and `norm_topk_prob`: llama.cpp hard-codes softmax + renormalisation for `qwen3moe`.
- **llama.cpp's MoE path** is in our vendored tree (`ggml_cuda_mul_mat_id`, `ggml-cuda.cu:1992-2031`): mmvq with ids
  up to `get_mmvq_mmid_max_batch` (8 on sm_89 / sm_120; on sm_86 Q3_K 5, IQ3_S 6, Q2_K / IQ2_S / IQ3_XXS 7, else 8),
  else mmq with ids (device-side compaction per expert, `mmid.cu`; grid z = E), else a host-syncing fallback no
  supported type reaches. Our bf16 quantizer patch already covers the gather and scatter paths. Grids depend only on
  (T, E, k, out): capture-safe. Fused routing (`topk-moe.cu`) and the weighted combine (`moe-weighted-reduction.cu`)
  are not yet vendored.
- **Holon.** `adopt` receives a `QTensor` of any rank (a 3-D stack arrives as `QTensor(layout, (E, out, in),
  blocks [E·out, row_bytes])`), and the block calls `ffn(x)` without a step context, which MoE does not need.
- **Baselines** (llama-bench at 51ce9c11, median of repetitions 2–5): rtx-a5000 UD-IQ3_XXS 3694 / 159.2, Q4_K_M
  3458 / 172.6 (measured for M0.2); rtx-5080 UD-IQ3_XXS 7462 / 248.2 (S2). S2 rejected the SGLang-carried MoE kernels
  (0.07–0.45× llama.cpp at 512 tokens).

## 3. Decisions

**How references are chosen** (M0.1's rule, now written down). A reference has one of three jobs: the *oracle* is the
model's own definition — HF transformers with the original checkpoint — plus the format owner's decoding (gguf-py),
never another serving engine; the *kernel source* is the best-maintained implementation of the exact workload,
chosen by measurement (S2: llama.cpp's MoE kernels over SGLang's stale fork of them); the *bar* is the engine users
run for that workload on the target hardware (llama.cpp for GGUF on consumer cards).

- **D1 — GGUF only.** The model plugin declares `arch.gguf.qwen3moe@1` alone. The bf16 checkpoint (57 GiB) fits no
  target card, and loading HF's 18,867 per-expert tensors into stacks needs a stacking weight rule (a core change):
  both wait for a deployment that can hold the model. The judge builds the HF model itself (§4.6).
- **D2 — Its own package, three ops.** Everything MoE lives in `std/holon_std/moe` (a sibling of `std.ffn`, so no
  existing unit's `src` or `sem` changes and no bf16 lock moves):
  - `unit.std.moe@1(experts, top_k, inter, renormalize)` — the MoE ffn, and the **v1 seam**: an expert cache (a
    residency map, miss handling, a CPU path for missed experts, router telemetry) is a variant of this unit; the
    kernel ops stay stateless. How a deployment selects the variant (a recipe knob, or a second provider chosen by
    pin — today one provider serves each unit resource) is v1's decision.
  - `moe_topk@1`: `logits [T, E] f32 → (weights [T, k] f32, ids [T, k] int32)`, both contiguous: softmax over E in
    f32, the k largest (ties to the lower index; NaN as −FLT_MAX), divided by their sum when `renormalize` (the sum is
    clamped at 6.1e-5 as llama.cpp, a guard softmax never reaches: the top-8 sum is ≥ 8/128). Ids are unique and in
    range for every row, padding included. Facts: experts, top_k, renormalize.
  - `qlinear_experts@1`: `(x [T, R, in], blocks, ids [T, k]) → y [T, k, out_y] f32` for any T ≥ 0; row j of token
    t applies expert `ids[t, j]` to `x[t, 0]` (R = 1) or `x[t, j]` (R = k). With `glu = "swiglu"` the stack is a
    merged gate/up stack (per expert: `inter` gate rows, then `inter` up rows) and `y = silu(gate) · up` in f32
    (out_y = inter); with `glu = "none"`, out_y = out. Facts: layout, experts, in, out, top_k, rows_per_token, glu,
    act_dtype.
  - `moe_combine@1`: `(y [T, k, H] f32, weights [T, k] f32, out [T, H])`: writes `Σ_j w[t, j] · y[t, j]`, accumulated
    in f32 in slot order, into `out` (the activation dtype; so a chunk lands in place). Facts: hidden, top_k, dtype.
  The split follows the kernels (three vendored kernels, three L0 suites), not a cache's needs; plan §5's "one coarse
  op" becomes one unit over three ops. `ExpertLinear` (a 3-D quantized stack: adopt, merge, bind) is an internal class
  of the unit, not a `Ref`'d unit.
- **D3 — f32 inside the expert path, as llama.cpp.** Expert matmuls write f32, SwiGLU runs in f32, the down
  projection quantizes the f32 hidden; the combine stores the activation dtype. The router multiplies the bf16 hidden
  state cast to f32 by the F32 weight in IEEE fp32 (no TF32: torch's default, asserted by a test); HF instead
  rounds router logits to bf16 before an f32 softmax.
- **D4 — Gate and up merge per expert** when their layouts match (every layer of both files): one stack, one scatter
  quantization, one compaction and one matmul instead of two — valid on both paths (the scatter mode depends only on
  R = 1, the compaction never reads `out`) and the layout llama.cpp's own merged path uses (gate rows first). On mmvq
  the merged stack feeds llama.cpp's fused gate kernel directly (`fusion{gate, SWIGLU}`, gate and up at equal
  strides), so decode needs no separate SwiGLU; on mmq a small f32 SwiGLU kernel follows. Gate and up of different
  layouts — in no file in scope — are a startup error naming the layer and both layouts, not a slower path.
- **D5 — The router is the unit's own F32 parameter**, applied with `F.linear` in the unit (as dense `Linear` does).
- **D6 — Configuration in the model plugin.** The recipe reads the MoE keys from the file's raw `qwen3moe.*` metadata
  (`c["gguf"]["expert_count"]`, `expert_used_count`, `expert_feed_forward_length`), so `std.weights.gguf` is
  untouched; `renormalize=True` is written with llama.cpp's reason (hard-coded for `qwen3moe`).

## 4. Design

### 4.1 Packages

| package | change |
|---|---|
| `plugins/holon_plugins/models/qwen3_moe` | new: recipe copied from qwen3 (`derived_from`), `arch.gguf.qwen3moe@1`, GGUF weight map, bundle (copied) |
| `std/holon_std/moe` | new: the unit and `ExpertLinear`; `moe_topk@1`, `qlinear_experts@1`, `moe_combine@1` contracts and `reference.py` |
| `plugins/holon_plugins/kernels/ggml_cuda` | the three kernels; `topk-moe.cu` and `moe-weighted-reduction.cu` vendored (`vendor.sh`, `SHA256SUMS`, `[provenance]`, `THIRD_PARTY_NOTICES.md`); `vendor.patch`: topk ids at stride k (the write and its stride assert), the reduction's bf16 store; csrc: f32 SwiGLU on llama.cpp's `ggml_cuda_op_silu_single`, the shim's `ggml_get_unary_op` |
| `testing` | reference providers for the three ops (gguf-py dequantization, cached); tiny MoE GGUF writer |
| `judge` | MoE name map, streamed golden, gates, the MoE block of the floors (§4.6), emulator expert paths, planted bugs |

### 4.2 The MoE unit

```python
class MoE(nn.Module):                      # unit.std.moe@1(experts, top_k, inter, renormalize) at (layer, hidden)
    router: Parameter [E, H] f32
    gate_up, down: ExpertLinear             # gate_up = merge(gate, up) in finalize_weights; it carries glu="swiglu"
    def forward(x [T, H]):
        out = empty_like(x)
        for c in chunks(T, 512):                                       # bounds the f32 expert path (≈ 45 MiB)
            w, ids = topk(F.linear(x[c].float(), router))               # moe_topk@1
            h = gate_up(x[c][:, None], ids)                             # [t, k, I] f32, SwiGLU applied
            combine(down(h, ids), w, out[c])                            # [t, k, H] f32 → bf16, in place
        return out
```

`ExpertLinear` mirrors `Linear`: `adopt` takes a quantized `(E, out, in)` stack, `merge(gate, up)` interleaves the two
per expert (`view(E, out, rb)` concatenated on dim 1, in `finalize_weights`, one layer at a time) into a unit that carries `glu =
"swiglu"`, and `bind_kernels` selects `qlinear_experts` with its facts after merges. A dense stack has no path:
`bind_kernels` fails at startup naming the unit and layout. The unit's chunk bounds memory; the kernel chunks at its
own 512 too (any T, as `qlinear`) and picks the path per chunk, as llama.cpp does per ubatch: a 513-token step runs
one mmq chunk and one mmvq chunk.

### 4.3 Kernels (`kernels.ggml_cuda`)

- `qlinear_experts`: mmvq with ids when the chunk's T ≤ `get_mmvq_mmid_max_batch(type, cc)` (fused SwiGLU when `glu =
  "swiglu"`), else mmq with ids (f32) followed, for `swiglu`, by our f32 SwiGLU kernel. Workspace
  `ggml_cuda.id[in=…]` (distinct from dense `qlinear`'s `ggml_cuda[in=…]`), sized as the pool allocates it for a
  512-token chunk: `ids_src1` and `ids_dst` (T·k int32), `expert_bounds` (E + 1), q8_1 for T·k rows plus `J_max`, the
  stream-k fixup — each padded to 256 B. M0.1's one-stream invariant (§8 there) extends to it.
- `moe_topk`: vendored `topk-moe.cu` with ids written at row stride k (`vendor.patch`: the write and the stride
  assert), so ids are an ordinary contiguous `[T, k]` tensor. The C API takes E from the logits' width and launches
  the kernel itself (no `should_use_topk_moe`, which reads E from the stride). E a power of two ≤ 512.
- `moe_combine`: vendored `moe-weighted-reduction.cu`, its store patched to bf16, writing into the caller's `out`
  (one launch); it supports `dtype = bfloat16` only.
- The f32 SwiGLU after mmq is built on the vendored `ggml_cuda_op_silu_single` with llama.cpp's flags, the function
  the fused mmvq gate uses.
- Each `qlinear_experts` instance counts its calls per path (an instance attribute, read by tests through the unit;
  no module state, no switch).
- Graph safety: no host sync; grids fixed by (T, E, k, out); per-type mmq instances and attributes are set on the
  eager warm-up run each graph key gets before capture. Padding rows get unique in-range ids from `moe_topk`, which
  both kernels require: an invalid id reads out of bounds on mmvq; a duplicate *inside a row* shifts mmq's
  `expert_bounds` (`nex_prev` counts slots, the compaction counts each token once per expert), leaving a stale
  compacted row whose leftover workspace entries can write over another token's output. Rows sharing experts with
  other rows are ordinary routing.
- Padding cost: a padding row routes to 8 more experts (≈ 16 MB of expert reads per layer at IQ3). Batch-1 decode
  (tg128) is never padded; serving batches 3, 5–7 pay it. Measured in step 5; the remedy, finer graph buckets for MoE
  models, is a core change and waits for that measurement.
- Determinism: run to run, yes; batch-invariant, no (as dense mmq).
- PDL: llama.cpp's PDL kernels wait on their inputs before reading, so they are safe beside ours. Any kernel of ours
  launched with PDL later must wait before its first global read *and* first write (llama.cpp's quantizer triggers
  its successor before it waits, and workspaces are reused).
- Launches per layer at batch 1: llama.cpp 7 (router, topk, quantize, fused gate/up, quantize, down, reduction);
  ours 8 (one more: the router's cast; the combine writes in place). §4.5 has the fallback.

### 4.4 Model plugin

`models/qwen3_moe/recipe.py` = qwen3's recipe with `ffn = Ref("std.moe@1", experts=g["expert_count"],
top_k=g["expert_used_count"], inter=g["expert_feed_forward_length"], renormalize=True)` and `GGUF_WEIGHTS` = qwen3's
with `ffn_gate_inp → ffn.router`, `ffn_{gate,up,down}_exps → ffn.{gate,up,down}.weight` in place of the dense ffn
rules; `derived_from = "models.qwen3@<commit>"`.

### 4.5 Performance

Estimated tg128 on the 5080 ≈ 0.93× llama.cpp (kernels reviewer: the router cast is the one extra launch per layer;
PDL ≈ 2 % as measured in M0.1). Prefill should beat llama.cpp (the merge saves a compaction, a quantization and a
matmul per layer). Fallbacks, in order: a fused router kernel (bf16 hidden × F32 weight), PDL launches for our
kernels (with the wait rule of §4.3), finer graph buckets for MoE (core).

### 4.6 Numerics and the judge (thresholds and predictions fixed before measuring)

- **Golden**: HF `Qwen3MoeForCausalLM` (fp32, CPU, HF config at a pinned revision) filled with gguf-py's
  dequantized tensors through the judge's own name map, built and loaded **layer by layer** (peak ≈ 130 GB);
  `experts_implementation` pinned and recorded in the meta. ≈ 5–10 min of forward, ≈ 8 min of dequantization.
- **Gates**: per tensor, NMSE(dequantized, HF bf16) ≤ 0.5 (experts per expert, the router exact); and the judge's
  **assembled** state dict (its fused `gate_up_proj`) against the bf16 checkpoint loaded by `from_pretrained`, which
  does transformers' own per-expert fusion — so a gate/up concatenation mistake shared by the judge and the engine
  fails here; NMSE applied per expert slice of the fused parameter. The judge's side is streamed per layer;
  `from_pretrained` holds the bf16 model (61 GB), which the host's 251 GB carries.
- **The floors' MoE block mirrors D3**: the judge replaces HF's MoE block (and grouped_mm) in the floors with one
  built from `std/holon_std/moe/reference.py`'s functions at the reviewed commit (as `ref_kernels` uses the std
  references) — router F32 with f32 logits, expert outputs, SwiGLU and down input in f32, combine in f32 in slot
  order, bf16 out — so the floor is no looser than the engine and D3 is defined once. HF stays the golden. The pure-HF
  bf16 floor is recorded for information only. The floor and Tier A's candidate now share those reference functions,
  so a reference bug would inflate both: floors are recorded only at a reference commit where step 1's oracle test
  and Tier A's sanity ceiling pass, carry `reference.py`'s hash in their meta (a change invalidates them), and are
  predicted to sit at or below the pure-HF floor's KL p99 (a looser one is suspect).
- **Tier A — assembly**, on the CPU in fp32: the production `gguf` source and `testing.ref_kernels` (references for
  every op, dequantizing with gguf-py), so adoption, the per-expert merge and kernel binding are covered too. Peak
  ≈ 140 GB (the references' fp32 cache ≈ 122 GB plus the blocks), run apart from the golden.
  Floor: the D3-mirroring bf16 model on A5000 GPUs with `device_map` (57 GiB over 3–4 cards). Sanity ceiling fixed in
  advance: Tier A's own KL p99 ≤ 1e-4 (fp32 throughout; any more means a real difference). Floor p99 reported with
  a sequence-bootstrap interval (`bounded`'s), and per-layer top-k flip rates vs the golden for floor and candidate.
- **Tier B — kernels**: the production engine on one A5000. Floor: the D3-mirroring bf16 model with the q8_1
  emulator on every quantized linear and expert stack — gate/up quantize each token's input (scatter), down each
  (token, slot) row (gather), per path; prefill uses each 512-token chunk's path by the plugin's predicate (a short
  last chunk is mmvq), the decode run mmvq semantics at a step size
  equal to the smallest limit any tensor of the file uses on that class (dense `ggml_cuda_should_use_mmvq` and MoE
  `use_mmvq_id`), computed from the plugin, with a counter asserting that no mmq call ran. Decode runs are eager
  (prompt-logprob steps are never graphed), so graph padding does not enter them.
- **Precision test** (CUDA unit test, not L2: Tier A is fp32 and Tier B's floor is noisier than these choices): at
  bf16, the router logits match an f64 reference within the fp32 budget (fails under TF32 or a bf16 router); our f32
  SwiGLU kernel matches an f64 reference within the fp32 budget (fails for a bf16 SwiGLU cast to f32); expert
  outputs are f32; `qlinear_experts` binds `act_dtype` bf16 with f32 output.
- **Graphs vs eager**: `holon_judge bounded` with `cuda_graphs` on vs off on MoE decode at batch 3 and 5 (padded,
  mmvq) and 12 and 20 (padded to 16 and 32: mmq). Its KL is scored eagerly, so its sensitivity to replay comes from
  greedy agreement.
- **L0** (sm_86, sm_89, sm_120a): `qlinear_experts` per type, path and `glu` vs the q8_1-emulated reference per
  element (`l0.check(bf16)`); the merged stack vs two separate stacks with `glu = "none"` (bit-exact on mmvq and on mmq
  without fixup), and the fused mmvq SwiGLU vs ours within the fp32 budget;
  routings — uniform, all tokens on the same 8 experts at T = 512 and 513 (column tiles and stream-k fixup inside
  one expert), half the experts empty, ids 0 and E−1 beside poisoned bytes; T ∈ {0, 1, limit, limit + 1, 9, 64,
  512, 513, 1025, 2049}; mmvq and mmq agree at limit and limit + 1; down with distinct per-slot inputs; mixed types in
  one layer at T = 7 on sm_86 (gate IQ3_S on mmq, down IQ4_XS on mmvq); ids straight from `moe_topk`; graph replay
  with routing unlike the capture's; every captured bucket. `moe_topk` vs the reference (near-ties compared modulo
  `__expf`, ±inf and all-NaN rows giving unique in-range ids, T not a multiple of 8). `moe_combine` within 1 ulp (FMA).
  compute-sanitizer memcheck on each class.
- **Planted bugs**, verdicts and predicted multiples of the floor's KL p99 recorded before measuring:
  - Tier A must fail: renormalisation off in every layer (gross); renormalisation off in layer 24 only (near
    threshold: predicted 1.5–3×); expert ids shifted by one in one layer; gate/up swapped in the experts; top-k 7;
    weights paired with the wrong slots (reversed) in the combine; the production merge interleaving up before gate.
    A result outside its prediction is recorded with its cause; δ never changes for it (M0.1's rule).
  - Tier B must fail: the production merge interleaving up before gate; an expert stack read as the wrong type; a
    stale q8_1 input on every 97th expert call (keyed by in and R; it models a stale per-token input, not a stale
    compacted buffer; fires counted per path), and the rare variant reported.
  - L0 must fail: down reading `x[t, 0]` (R confused); an id offset in the compaction.
  - Precision test must fail: router logits in bf16 (L2 predicted at ≤ 1.2× the floor's KL p99, recorded).
  - L0 must fail, deterministically: padding rows with all k slots on one expert, the `ggml_cuda.id` workspace
    pre-filled with a real row's index — real rows change on mmq, stay unchanged on mmvq.
  - Graphs-vs-eager, the same plant: predicted per (file, class) by the path the plugin's predicate gives at each
    padded bucket — fail where it is mmq (batches 12 and 20 everywhere; batch 5, bucket 8, for UD-IQ3_XXS on sm_86),
    pass where it is mmvq (batch 3 everywhere); a pass on mmq is recorded with its cause (the stale entries are not
    controlled there).
- **Quantization error**, reported: the file's (dequantized golden vs the bf16 checkpoint's fp32 golden, pinned
  revision) and the engine's (engine vs that golden), with top-k flip rates.

### 4.7 Memory on a 16 GB card

RTX 5080, UD-IQ3_XXS: 15.56 GiB total, 12.0 GiB weights, ≈ 0.9 GiB context and libraries → ≈ 2.6 GiB. 16K KV tokens
= 1.5 GiB; the expert path is chunked (≈ 45 MiB at 512 tokens); the worst step ≈ 0.3–0.4 GiB; graph pool ≈ 0.26 GiB
at `max_seqs` 32 (M0.1) → ≈ 0.4 GiB left. Settings: `memory_fraction 0.98`, `max_step_tokens 2048`, `max_seqs 32`;
breakdown into the evidence; ceiling ≈ 20K KV tokens (64K needs quantized KV, M0.3b). Merges at load are per layer
(≈ 170 MB transient).

### 4.8 Bench

As M0.1: `bench/llama_bench.py` vs llama-bench (`-p 512 -n 128 -d 0,256 -ngl 99 -fa on`, medians of repetitions
2–5 on both sides), and `llama-server` vs Holon over HTTP on the decode and agent traces.

## 5. Implementation plan

Same three reviewers throughout; each step ends green.

| # | Step | Deliverable / test | Days |
|---|---|---|---|
| 0 | Q4_K_M fetched; A5000 llama-bench baselines (done); per-class baselines in step 5 | baselines | done |
| 1 | `std.moe` (unit, `ExpertLinear`, contracts, references); `models/qwen3_moe`; test-only providers; tiny MoE GGUF writer | CPU, against an independent oracle: a tiny random HF `Qwen3MoeForCausalLM` (2 layers, 8 experts, top-2) written as GGUF; the engine (reference kernels) vs transformers on the dequantized weights; the Tier A plants fail here too; startup errors for unreadable stacks and for gate/up of different layouts; recipe diff ≤ 15 lines | 2.5 |
| 2 | Judge: name map, streamed golden, gates, the D3-mirroring MoE block, Tier A on the CPU with its floor on A5000s, Tier A plants | Tier A L2 for both files; plants fail at their predicted margins | 2.5 |
| 3 | `kernels.ggml_cuda`: C API, two vendored kernels and their patches, f32 SwiGLU, provenance; L0 incl. memcheck | L0 on sm_86 and sm_120a; graph capture; determinism | 4.5 |
| 4 | Emulator expert paths; Tier B floors and runs on the A5000; Tier B, L0 and precision plants; precision test; graphs vs eager; perf on the A5000 | Tier B passes; plants fail | 2.5 |
| 5 | Consumer classes: L0 on sm_89; 5080 fit ≥ 16K; bars on 5080 / 4090 / 5090 / A5000; `llama-server` traces on the 5080; Tier B on the 5080; tune + GGUF locks; snapshot ×3 | evidence; locks | 3 |
| 6 | Qwen3 GGUF locks (rtx-5080, rtx-a5000): `propagate`, approved once | locks | 0.5 |
| 7 | Docs: design §5.1 / §5.3 / §5.6, plan §5 status | — | 0.5 |

≈ 16 days; critical path 1 → 2 and 3 (parallel) → 4 → 5 → 6 → 7.

## 6. Risks

- 16K KV on the 5080: ≈ 0.4 GiB of headroom after the measured M0.1 overheads.
- Tier A on the CPU: the reference experts at fp32 over 32.8K tokens, estimated 15–30 min.
- Tier B prefill floors are loose by construction (M0.1: mmq's min-term); decode and Tier A carry the sensitivity.
- Serving batches that pad (3, 5–7) pay extra expert reads until finer buckets land.

## 7. Deferred, deliberately

Serving HF MoE checkpoints (a stacking weight rule); expert offload and the expert cache (v1, at the unit); expert
parallelism; finer graph buckets for MoE (core, on measurement); a fused router kernel; MoE models with shared experts
or sigmoid routing (DeepSeek-style; their own `moe_topk` facts).

## 8. Review findings, resolved

Revision 1.

Architecture/API: 1 dense stacks / Tier A → Tier A on the `gguf` source with reference kernels, `ExpertLinear`
quantized only (§4.2, §4.6); 2 workspace name → `ggml_cuda.id[in=…]` (§4.3); 3 lock churn → `std/holon_std/moe`
(D2, acceptance 1); 4 the cache seam → the unit (D2); 5 a tautological step-1 test → a tiny random HF MoE as oracle
(§5 step 1); 6 an aliased ids buffer → ids at stride k, contiguous (§4.3); 7 HF keys in std → the recipe reads raw
GGUF keys (D6); 8 the f32 SwiGLU provider → our kernel, combine stores bf16 (§4.3); 9 speculative recipe content →
removed, `renormalize` (§4.4); 10 lock-diff statement, step 3 sizing, UD-Q3_K_XL optional (§1, §5); 11 expert-path
memory → chunked (§4.2, §4.7).

Kernels/performance: 1 workspace collision (as above); 2 decode run on mmq → step size from the plugin's predicates,
eager runs, a counter (§4.6); 3 padding cost → measured, remedy deferred (§4.3, §7); 4 no f32 SwiGLU (as above); 5
fused gate on the merged stack by default (D4); 6 peak memory → chunking (§4.2); 7 combine ±1 ulp, topk near-ties
(§4.6 L0); 8 combine + cast in one launch, IEEE router (D3, §4.3); 9 ids stride (as above); 10 per-chunk path
(§4.2); 11 PDL wait rule (§4.3); 12 merge transient (§4.7); 13 L0 cases (§4.6); bars on medians (§1).

Numerics/judge: 1 floor looser than candidate → the D3-mirroring MoE block (§4.6); 2 precision invisible at L2 → the
precision test and its plant (§4.6); 3 decode step size → from the plugin (§4.6); 4 shared judge/engine mistake →
the assembled-state-dict gate (§4.6); 5 Tier A sensitivity → predicted margins, a near-threshold plant, a sanity
ceiling (§4.6); 6 flip tails → bootstrap intervals and flip rates reported (§4.6); 7 missing plants → added, the
clamp noted as dead for softmax (D2, §4.6); 8 stale plant definition (§4.6); 9 L0 cases (§4.6); 10 graph padding →
graphs-vs-eager bounded check (acceptance 4, §4.6); 11 quantization-error report (§4.6); 12 golden RAM → streamed
(§4.6).

Revision 2 re-review. Architecture/API: R1 the unmerged fallback had no SwiGLU to reach → dropped, a startup error
(D4); R2 `glu` bound at merge (§4.2); R3 the kernel chunks itself, any T (D2, §4.2); R4 the floors' MoE block from
the std references (§4.6); R5 per-instance path counters (§4.3); R6 Tier A's peak (§4.6); R7 the seam's selection
left to v1 (D2). Kernels/performance: 1 the combine writes in place (D2, §4.2); 2 the topk stride assert and E from
the logits (§4.3); 3 our SwiGLU on `ggml_cuda_op_silu_single` (§4.3); 4 combine bf16 only (§4.3). Numerics/judge: 1
graphs vs eager on mmq too, with a padding plant (§4.6); 2 the prefill floor's path per chunk (§4.6); 3 SwiGLU vs
f64 (§4.6); 4 bit-exactness on `glu = "none"` (§4.6); 5 the near-threshold layer fixed, the merge plant in Tier A,
the bf16-router prediction (§4.6); 6 the gate's memory and per-expert slices, Tier A's peak (§4.6).

Revision 3 re-review. Architecture/API: clean (the merge's timing worded as `finalize_weights`, §4.2).
Kernels/performance and numerics/judge, the same finding: a padding row sharing a real row's ids is harmless; the
hazard is a duplicate inside a row → reason restated (§4.3), the plant moved to L0, deterministic, and kept in
graphs-vs-eager with per-path predictions (§4.6). Numerics/judge: the floor and Tier A share the references → floors
recorded only after step 1's oracle and the ceiling pass, the reference hash in their meta, predicted ≤ the pure-HF
floor (§4.6). Kernels/performance, last: predictions by the path at the padded bucket, not the batch (§4.6).
