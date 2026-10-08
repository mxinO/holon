# M0.2 — Mixture of experts on a 16 GB card: design and implementation plan

Status: **revision 1, for review** (2026-10-08). Scope: plan §5. Builds on M0.1 (`docs/m01-gguf.md`: the GGUF source,
`kernels.ggml_cuda`, the judge's tiers) and spike S2 (`spikes/s2_gguf/README.md`, MoE sections).

## 1. Goal and acceptance

Serve Qwen3-30B-A3B (128 experts, 8 per token, ≈ 3B active) from llama.cpp GGUF files with llama.cpp's own MoE CUDA
kernels, on a 16 GB RTX 5080 and on 24–32 GB cards. Acceptance:

1. `models/qwen3_moe` is `models/qwen3` copied with `derived_from`; its recipe diff to qwen3 is ≤ ~15 lines (the
   MoE ffn `Ref`, the expert weight map). Qwen3 locks change only through `propagate`. Core diff: zero.
2. L0 for the MoE ops on sm_86, sm_89, sm_120a: every supported type on both kernel paths, skewed routings included.
3. L2 for UD-IQ3_XXS and Q4_K_M: Tier A (assembly) and Tier B (kernels, prefill and decode) pass; each planted bug
   fails at the level recorded in §4.6; the per-tensor gate passes; the quantization error vs bf16 is reported.
4. Snapshot trace run-identical over 3 fresh engines (UD-IQ3_XXS on rtx-5080, Q4_K_M on rtx-a5000).
5. RTX 5080: UD-IQ3_XXS with ≥ 16K KV tokens. RTX 4090, 5090, A5000: Q4_K_M. In-process pp512 / tg128 against
   llama-bench at the same commit on every class (bar on the 5080: ≥ 0.8× S2, pp512 ≥ 5800 t/s, tg128 ≥ 199 t/s);
   decode and agent traces against `llama-server` on the 5080.
6. GGUF locks for rtx-5080 (UD-IQ3_XXS) and rtx-a5000, rtx-4090, rtx-5090 (Q4_K_M).

Non-goals: serving the bf16 HF checkpoint (57 GiB fits no single target card; §3 D1); expert offload / expert cache
(v1; its seam is the `qlinear_experts` op); expert parallelism; Qwen3.5 (M0.3).

## 2. Starting point

- **The model.** 48 layers, all MoE (`decoder_sparse_step` 1, `mlp_only_layers` []); hidden 2048; attention 32 / 4
  heads × 128 (as Qwen3, with q/k norm); 128 experts, top-8, expert width 768, `norm_topk_prob` true; vocab 151936,
  untied head. KV: 96 KiB / token.
- **The files** (unsloth `Qwen3-30B-A3B-GGUF`): UD-IQ3_XXS 12.0 GiB (cached), Q4_K_M 17.28 GiB (fetched for M0.2),
  UD-Q3_K_XL 12.88 GiB (optional). UD-IQ3_XXS: experts IQ3_XXS / IQ3_S (gate, up), IQ3_XXS / IQ4_XS / IQ3_S (down);
  attention IQ4_XS / Q5_K / Q6_K; router `ffn_gate_inp` **F32** [128, 2048]; embedding Q4_K, head Q6_K. Gate and up
  share a type in every layer, as do q / k / v. Experts are stacked `[E, out, in]` with no permutation and HF's expert
  order (measured: NMSE vs HF 0.035 per expert, 2.0 for gate↔up, 2.3–2.6 for a neighbouring expert). The GGUF lacks
  `norm_topk_prob` and the gating function: llama.cpp hard-codes softmax + renormalisation for `qwen3moe`.
- **llama.cpp's MoE path** (`ggml_cuda_mul_mat_id`, `ggml-cuda.cu:1992-2031`) is already in our vendored tree:
  `mmvq` with ids up to a per-type limit (`get_mmvq_mmid_max_batch`: 8 on sm_89 / sm_120; on sm_86 Q3_K 5, IQ3_S 6,
  Q2_K / IQ2_S / IQ3_XXS 7, others 8), else `mmq` with ids (a device-side compaction per expert, `mmid.cu`; grid
  z = E), else a host-syncing fallback no supported type reaches. The bf16 quantizers our `vendor.patch` added
  already cover the ids paths (gather and scatter). Outputs are f32. No host sync, grids depend only on T, E, k, out:
  capture-safe. Fused routing (`topk-moe.cu`: softmax, top-k by argmax rounds, renormalisation with the sum clamped
  at 6.1e-5) and the weighted combine (`moe-weighted-reduction.cu`, f32, expert slots summed in order) are separate
  files, not yet vendored.
- **Holon.** The loader hands a `QTensor` to its owner's `adopt` whatever its rank: a 3-D expert stack arrives as
  `QTensor(layout, (E, out, in), blocks [E·out, row_bytes])` with no core change. The block calls `ffn(x)` without a
  step context, which MoE does not need: routing is data, not shape. Graphs capture pure-decode steps at fixed
  buckets after one eager warm-up run per key (`platform.Graphs`).
- **S2's measurements** rule out the SGLang-carried MoE kernels (0.07–0.45× llama.cpp at 512 tokens; 25.2 vs 2.7 ms
  per 30B layer on the A5000) and set the bars: llama-bench UD-IQ3_XXS pp512 / tg128 — rtx-5080 7247 / 248,
  rtx-4090 10133 / 257, rtx-5090 11905 / 341, rtx-a5000 3529 / 151.

## 3. Decisions

**How references are chosen** (the rule M0.1 followed, now written down). A reference has one of three jobs, and each
is picked for that job: the *oracle* is the model's own definition — HF transformers with the original checkpoint —
plus the format owner's decoding (gguf-py); never another serving engine. The *kernel source* is the best-maintained
implementation of the exact workload, chosen by measurement (S2: llama.cpp's MoE kernels over SGLang's stale fork of
them). The *bar* is the engine users run for that workload on the target hardware: llama.cpp for GGUF on consumer
cards (M0.1, M0.2), mini-sglang / nano-vllm for bf16 batching (M0), vLLM / SGLang for HF quant formats (v1).

- **D1 — GGUF only.** The recipe accepts the `gguf` source; for `hf_safetensors` it raises "no weight map". The bf16
  checkpoint fits no target card, and loading HF's 18,867 per-expert tensors into stacked parameters needs either a
  stacking weight rule (a core change) or 384 parameters per layer: both wait until a card or a multi-GPU deployment
  can hold the model. The judge builds the HF model itself (§4.6).
- **D2 — Ops.** The MoE ffn is one unit (`unit.std.ffn.moe@1`) composing three ops, so the expert matmul — the seam
  an expert cache will need — is its own op:
  - `moe_topk@1` (std.ffn): `logits [T, E] f32 → (weights [T, k] f32, ids [T, k] int32)`: softmax over E in f32,
    the k largest (ties to the lower index), renormalised by their sum (clamped at 6.1e-5, llama.cpp's guard) when
    `norm`. Facts: experts, top_k, norm. Ids are unique and in range for every row, padding rows included.
  - `qlinear_experts@1` (std.ops, beside `qlinear`): `(x [T, R, in], blocks, ids [T, k]) → y [T, k, out] f32`, row j
    of token t = expert `ids[t, j]` applied to `x[t, 0]` (R = 1: gate / up) or `x[t, j]` (R = k: down). Blocks are
    the stack `[E·out, row_bytes]`. Facts: layout, experts, in, out, top_k, rows_per_token (R), act_dtype.
  - `moe_combine@1` (std.ffn): `(y [T, k, H] f32, weights [T, k] f32) → [T, H]` in the activation dtype,
    `Σ_j weights[t, j] · y[t, j]` accumulated in f32 in slot order. Facts: hidden, top_k, dtype.
  - Between them, the existing `silu_mul@1` at `dtype=float32` on `[T·k, 2·inter]`.
- **D3 — f32 inside the expert path, as llama.cpp.** The expert matmuls write f32 (llama.cpp's MoE stores are
  f32-only; no patch), SwiGLU runs in f32 and the down projection's quantizer reads f32. The block's input and the
  combine's output are bf16. The router runs in f32 over the bf16 hidden state cast to f32, its weight F32 as stored
  (llama.cpp: f32 × F32; HF: model dtype, then an f32 softmax).
- **D4 — Gate and up merge per expert.** When gate and up experts share a layout (every layer of both files), the
  unit concatenates them per expert into one stack `[E, 2·inter, in]` (rows are independent block runs, as for dense
  merges) and runs one `qlinear_experts` instead of two; the activation is quantized once. llama.cpp runs two
  `mul_mat_id`s, fusing them with SwiGLU on its mmvq path only; we take the merge on both paths and keep SwiGLU
  separate (a fused gate on mmvq is a §4.5 fallback).
- **D5 — The router is the MoE unit's own f32 parameter**, multiplied with `F.linear` in the unit (as dense `Linear`
  does for bf16 weights): a [2048 → 128] product, not worth an op. std.ops is untouched for it.
- **D6 — Configuration from the file.** The GGUF source maps llama.cpp's MoE keys every MoE architecture shares —
  `expert_count`, `expert_used_count`, `expert_feed_forward_length` — to `num_experts`, `num_experts_per_tok`,
  `moe_intermediate_size`. What a `qwen3moe` GGUF omits (`norm_topk_prob`) the recipe defaults to llama.cpp's
  hard-coded `true`.

## 4. Design

### 4.1 Packages

| package | change |
|---|---|
| `plugins/holon_plugins/models/qwen3_moe` | new: recipe copied from qwen3 (`derived_from`), `arch.gguf.qwen3moe@1`, GGUF weight map, bundle (copied) |
| `std/holon_std/ffn` | `MoE` unit, `moe_topk@1` and `moe_combine@1` contracts and references |
| `std/holon_std/ops` | `qlinear_experts@1` contract and reference; `ExpertLinear` unit (adopts a 3-D stack, merges per expert) |
| `std/holon_std/weights/gguf` | the three shared MoE keys (D6) |
| `plugins/holon_plugins/kernels/ggml_cuda` | `qlinear_experts`, `moe_topk`, `moe_combine` kernels; `topk-moe.cu` and `moe-weighted-reduction.cu` vendored |
| `testing` | reference providers for the new ops; tiny MoE GGUF writer; `gguf_dense` dequantizes 3-D stacks |
| `judge` | MoE name map, golden, per-tensor gate, Tier A / B for experts, planted bugs |

### 4.2 The MoE unit

```python
class MoE(nn.Module):                     # unit.std.ffn.moe@1(experts, top_k, inter, norm) at site (layer, hidden)
    router: Parameter [E, H] f32
    gate_exps, up_exps, down_exps: ExpertLinear     # std.ops.experts@1 units; adopt QTensors [E, out, in]
    def forward(x [T, H]):
        w, ids = topk(F.linear(x.float(), router))                  # moe_topk@1
        h = silu_mul(gate_up(x[:, None], ids).view(T·k, 2I))       # qlinear_experts@1 (merged, D4), silu_mul@1 f32
        y = down(h.view(T, k, I), ids)                              # qlinear_experts@1, R = k
        return combine(y, w)                                        # moe_combine@1 → bf16
```

`ExpertLinear` mirrors `Linear`: `adopt` accepts a `QTensor` of logical shape `(E, out, in)`, `bind_kernels`
selects `qlinear_experts` with its facts after merges, and `merge(gate, up)` returns a unit whose stack interleaves
the two per expert (`blocks.view(E, out, rb)` concatenated on dim 1). A dense expert stack (bf16) has no provider in
M0.2 and fails at load, naming the unit and layout.

### 4.3 Kernels (`kernels.ggml_cuda`)

- `qlinear_experts`: `mmvq` with ids when T ≤ `get_mmvq_mmid_max_batch(type, cc)` (llama.cpp's own MoE limit, not the
  dense one), else `mmq` with ids in chunks of 512 tokens (4096 rows at k = 8; the compaction's shared memory holds
  ≈ 25K tokens on sm_86). Output f32 straight from the kernels. The C API gains `mmvq_id`, `mmq_id`, `use_mmvq_id`
  and `workspace_bytes_id(in, tokens, k, E)` = the q8_1 buffer for T·k rows (gather / scatter), the ids and
  `expert_bounds` arrays, and the stream-k fixup — declared per `in` to the ledger, as for `qlinear`.
- `moe_topk`: vendored `topk-moe.cu` (E a power of two ≤ 512: 128 fits). Its ids output has row stride E: the
  kernel instance owns a `[T_max, E]` int32 buffer and returns the `[T, k]` view the matmuls read with that stride.
- `moe_combine`: vendored `moe-weighted-reduction.cu` (f32 out) and one cast to the activation dtype.
- `silu_mul` at f32: FlashInfer's `silu_and_mul` if it accepts f32 (checked first in step 3), else a small Triton
  kernel in our own plugin. llama.cpp's f32 SwiGLU (`unary.cu`) is not worth vendoring for this.
- Graph safety: no host sync, grids fixed by (T, E, k, out); per-type mmq instances and `cudaFuncSetAttribute`
  happen on the eager warm-up run each graph key gets before capture. Padding rows route like any row: `moe_topk`
  gives them unique in-range ids, which both kernels require (an invalid id reads out of bounds in mmvq; a duplicate
  leaves a dst row unwritten in the compaction).
- Determinism: run to run, yes (token-ordered compaction, index-ordered ties, fixed stream-k split). Not batch
  invariant: the path switches at the per-type limit, and an expert's stream-k split depends on the other tokens
  routed to it — as for dense mmq.
- PDL: llama.cpp launches `mul_mat_vec_q_moe`, `quantize_*` and `topk_moe` with PDL on sm_90+; they wait on their
  inputs, so this is safe beside our kernels and in graphs.

### 4.4 Model plugin

`models/qwen3_moe/recipe.py` = qwen3's recipe with the ffn `Ref("std.ffn.moe@1", experts=…, top_k=…,
inter=moe_intermediate_size, norm=c.get("norm_topk_prob", True))` per layer (an `mlp_only_layers` entry keeps the
dense `swiglu`), and `GGUF_WEIGHTS` = qwen3's GGUF map with `ffn_gate_inp → ffn.router`, `ffn_{gate,up,down}_exps →
ffn.{gate,up,down}_exps.weight` in place of the dense ffn rules. `derived_from = "models.qwen3@<commit>"`.

### 4.5 Performance

Per decode step at batch 1 the expert work is three `mmvq`-with-ids launches (merged gate/up, down) plus routing and
combine; llama.cpp fuses gate + up + SwiGLU into one mmvq launch and the routing into one kernel — we take its fused
routing, merge gate/up instead of fusing them with SwiGLU, and add one f32 SwiGLU and one combine launch per layer.
Fallbacks if a bar is missed, in order: the mmvq gate fusion (`fusion{gate, SWIGLU}`, already in the vendored
kernel), a fused combine + cast, PDL for our Triton kernels (M0.1's list). Known: mmq's grid has z = E, so mid-size
decode buckets (9–64 tokens) launch many empty blocks — as in llama.cpp.

### 4.6 Numerics and the judge (thresholds fixed before measuring)

- **Golden**: HF `Qwen3MoeForCausalLM` (fp32, CPU, HF config) filled with gguf-py's dequantized tensors through the
  judge's own name map (experts stacked; `gate_up_proj` = gate then up per expert, as transformers stores it).
  Measured on this host: ≈ 35 s per 4096 tokens at 48 threads, ≈ 5–10 min for the prompt set, ≈ 8 min to dequantize,
  ≈ 122 GB RAM.
- **Per-tensor gate**: NMSE(dequantized, HF bf16) ≤ 0.5 per tensor, experts per expert; the router compared exactly.
- **Tier A — assembly** (router, top-k, renormalisation, expert order, gate/up, combine; at M0 sensitivity): the
  dense bf16 model does not fit one GPU, so Tier A runs the engine **on the CPU in fp32** — `gguf_dense` source
  (stacks dequantized), `testing.ref_kernels` (references for every op, the MoE ones included). Floor: HF bf16 of the
  dequantized weights, recorded once on A5000 GPUs with `device_map` across 3–4 cards (57 GiB). The fp32 candidate is
  closer to the golden than that floor, so a pass is cheap; planted bugs must fail against the floor's budgets.
- **Tier B — kernels**: the production engine on one A5000 (both files fit 24 GB). Floor: HF bf16 across A5000s with
  the q8_1 emulator on every quantized linear, now including experts: gate/up quantize each token's input once (the
  scatter path), down each (token, slot) row, per path (mmq for the prefill run, mmvq for the decode run at a step size
  ≤ the smallest MoE mmvq limit among the file's types on that class: 6 for UD-IQ3_XXS on sm_86, 8 on sm_89 / sm_120).
  Floors recorded once on the A5000s and reused on every class.
- **L0** (sm_86, sm_89, sm_120a): `qlinear_experts` per type and path vs the q8_1-emulated reference per element
  (`l0.check(bf16)` as M0.1); routings: uniform, every token to one expert, half the experts empty, the same expert
  for all k slots is impossible (ids unique) — checked as an invariant of `moe_topk`; T ∈ {1, limit, limit+1, 9, 64,
  512, 513, max_step_tokens}; poisoned bytes past every stack. `moe_topk` vs the reference (ties, NaN logits, k = E).
  `moe_combine` exact vs an f32 slot-ordered sum. compute-sanitizer memcheck on each class.
- **Planted bugs, verdicts recorded before measuring**: Tier A must fail: renormalisation off (`norm=False`); expert
  ids shifted by one (mod E) in one layer; gate and up swapped in the experts; top-k = 7. Tier B must fail: an expert
  stack read as the wrong type; a stale q8_1 workspace on every 97th expert call (and the rare variant reported).
  L0 must fail: an id offset in the compaction.
- **Quantization error** vs the bf16 checkpoint's golden: reported, not gated (bf16 golden recorded on the CPU once).

### 4.7 Memory on a 16 GB card

RTX 5080, UD-IQ3_XXS: 15.56 GiB total, 12.0 GiB weights, ≈ 0.9 GiB context and libraries (M0.1's measurement) →
≈ 2.6 GiB for KV, activations and the graph pool. 16K KV tokens = 1.5 GiB. The worst step's peak grows with the
expert path (`[T·k, 2·inter]` f32 = 100 MiB at 2048 step tokens) and the graph pool with `max_seqs` (0.38–0.50 GiB at
48–64 in M0.1). Start at `memory_fraction 0.98`, `max_step_tokens 2048`, `max_seqs 32`; step down on a measured
shortfall, breakdown into the evidence. The KV ceiling is ≈ 20K tokens; 64K needs quantized KV (M0.3b).

### 4.8 Bench

As M0.1 (§4.8 there): in-process `bench/llama_bench.py` vs `llama-bench -p 512 -n 128 -d 0,256 -ngl 99 -fa on`
(warm-up repetition excluded on both sides), and `llama-server` vs Holon over HTTP on the decode and agent traces.

## 5. Implementation plan

Same three reviewers throughout; each step ends green.

| # | Step | Deliverable / test | Days |
|---|---|---|---|
| 0 | Fetch Q4_K_M; llama-bench baselines at 51ce9c11 for both files on the A5000 (and per class in step 5) | baselines recorded | 0.5 |
| 1 | Ops, references, `MoE` + `ExpertLinear` units, gguf MoE keys, `models/qwen3_moe`; test-only providers; tiny MoE GGUF writer | CPU: tiny Q8_0 / Q4_0 MoE GGUF serves what its dequantized weights serve (bitwise prompt logprobs); load errors for unreadable stacks; recipe diff ≤ 15 lines | 2.5 |
| 2 | Judge: name map, golden (CPU), gate, Tier A on the CPU with its floor on A5000s, Tier A plants | Tier A L2 for both files; plants fail | 2 |
| 3 | `kernels.ggml_cuda`: C API, two vendored kernels, f32 SwiGLU; L0 incl. memcheck; emulator expert path | L0 on sm_86 and sm_120a; graph capture; determinism | 3 |
| 4 | Tier B floors and runs on the A5000; Tier B and L0 plants; perf on the A5000 vs llama-bench | Tier B (prefill + decode) passes; plants fail | 2 |
| 5 | Consumer classes: L0 on sm_89; 5080 fit ≥ 16K; bars on 5080 / 4090 / 5090 / A5000; `llama-server` traces on the 5080; Tier B on the 5080; tune + GGUF locks; snapshot ×3 | evidence; locks | 3 |
| 6 | Qwen3 locks: `propagate` on a5000 / 4090 / 5090 (bf16) and rtx-5080 / a5000 (GGUF) | locks | 0.5 |
| 7 | Docs: design §5.1 / §5.3 / §5.6, plan §5 status | — | 0.5 |

≈ 13.5 days; critical path 0 → 1 → 2 and 3 (parallel) → 4 → 5 → 6 → 7.

## 6. Risks

- 16K KV on the 5080: ≈ 1 GiB of headroom over the measured M0.1 overheads; the graph pool estimate was low last time.
- Tier A on the CPU: the reference MoE (a loop over experts) at fp32 over 32.8K tokens, estimated 15–30 min; if it
  is much slower, Tier A runs on the categories' first half and the full set only on release.
- Tier B prefill floors are loose by construction (M0.1: mmq's min-term); decode carries the sensitivity.
- `topk-moe.cu` needs `ggml_get_unary_op` from `ggml.c` (one function in the shim) and its fusion constraints
  (power-of-two E): other MoE models may need the torch-free fallback of a Triton top-k.

## 7. Deferred, deliberately

Serving HF MoE checkpoints (D1: a stacking weight rule or per-expert parameters); expert offload and the expert cache
(v1, at the `qlinear_experts` seam); expert parallelism; the fused mmvq gate + SwiGLU (§4.5 fallback); MoE models with
shared experts or sigmoid routing (DeepSeek-style; their own `moe_topk` facts).
