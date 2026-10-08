# M0.1 — GGUF quantized models: design and implementation plan

Status: **revision 2 for review** (2026-10-08). Revision 1 was challenged by three reviewers (architecture/API,
kernels/performance, numerics/judge); every finding is resolved below or listed in §8 as deliberately deferred.
Scope: plan §4. Builds on design §5.1, §5.4–5.6 and spike S2 (`spikes/s2_gguf/README.md`).

## 1. Goal and acceptance

Serve Qwen3-8B from llama.cpp GGUF files — Q4_K_M and UD-IQ3_XXS — with llama.cpp's own quantized CUDA kernels,
and take the first 16 GB consumer data point. Acceptance, made testable:

1. The `models/qwen3` diff adds only: the GGUF weight map, the `arch.gguf.qwen3` resource, and the bundle taking
   its weight source (§4.2). `canonical_hash(recipe(cfg, "hf_safetensors"))` is unchanged.
2. L0 passes for every type in the supported list (§4.7) on sm_86, sm_89 and sm_120a.
3. L2 passes both tiers of §4.7 for Q4_K_M and UD-IQ3_XXS on rtx-a5000 and rtx-5080; the planted-bug set of §4.7
   fails as recorded in advance; the quantization error vs the bf16 model is reported, and the per-tensor gate of
   §4.7 passes.
4. bf16 Qwen3-8B on rtx-a5000, rtx-4090, rtx-5090: `propagate` shows a snapshot trace byte-identical to the current
   lock and L2 passing; the only lock changes are `[units]` `src` lines (no `sem`, `[sites]`, `[engine]` or recipe
   change); approved once.
5. GGUF snapshot trace run-identical across 3 fresh engines (Q4_K_M, rtx-a5000 and rtx-5080).
6. RTX 5080: Q4_K_M and UD-IQ3_XXS with ≥ 64K KV tokens (settings in §4.8). Performance vs llama.cpp on the same
   card and file, measured as in §4.9: in-process pp512 and tg128 against the S2 llama-bench bars (≥ 0.8×: Q4_K_M
   ≥ 6380 t/s pp512 and ≥ 127 t/s tg128), and decode/agent traces vs `llama-server`.
7. GGUF locks for rtx-5080 (and rtx-a5000) — model ids `unsloth/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M@…` and
   `…/Qwen3-8B-UD-IQ3_XXS@…` (bf16 Qwen3-8B does not fit 16 GB, so rtx-5080 has GGUF locks only).

Non-goals: MoE (`mul_mat_id`, M0.2); quantized KV cache (M0.3b); HF quant formats (v1); converters other than
identity.

## 2. Starting point

- **S2** decided to port llama.cpp's current `mmvq`/`mmq`/`quantize` behind a small shim; it built and measured
  the SGLang-vendored set (correct, graph-safe) and probed that the current files compile for sm_86 against a
  stub of 20 symbols. The review found that the probe did not prove the full link (18 of 24 `mmq` cases), sm_120a,
  or torch-extension flags (§4.5).
- **API types** `Scheme`/`Layout`/`QTensor`/`SchemeImpl`/`Dialect` exist, unused and provisional.
- **Loading** builds on the meta device, materializes every parameter densely, then copies by shape — 16 GB of
  bf16 for an 8B model before a byte is read.
- **Qwen3 GGUF files** (unsloth): no Q/K permutation (llama.cpp's Qwen2/3 converters keep NEOX order); F32 norms;
  q/k share one type in every layer of both files, so do gate/up; v differs (Q6_K in half the Q4_K_M layers; Q4_K vs
  IQ2_S in UD-IQ3_XXS); the file has `qwen3.*` hyperparameters, a full tokenizer table and the chat template, but
  no base-model id; its eos field ({151645}) is narrower than the HF checkpoint's stop set ({151645, 151643}).
- **llama.cpp needs nothing but the file**: its tokenizer is built from `tokenizer.ggml.tokens/merges/token_type`
  with a pre-tokenizer regex chosen by `tokenizer.ggml.pre` from a built-in table (`src/llama-vocab.cpp:383`,
  `qwen2`), CONTROL/USER_DEFINED tokens matched atomically; its stop set is eos/eot plus CONTROL tokens whose text is
  a known end-of-turn marker (`<|im_end|>`, `<|endoftext|>`, …, `llama-vocab.cpp:3008-3037`).

## 3. Decisions

**D1 — References are oracles, never kernels** (engine-wide, not only GGUF). A unit's `reference.py` defines the
op (`sem`), serves L0 and the judge, and is never registered as a kernel. Kernel selection sees production
providers only; a missing one is a startup error naming the op, layout and device (invariant 7). The CPU
tiny-model tests get their kernels from a test-only fixture plugin (`plugins/tests/fixtures/ref_kernels`) that
wraps the references; it is installed only in the test environment. Today `std.ops` (rmsnorm, add_rmsnorm, rope),
`std.ffn` (silu_mul) and `std.mixers.gqa` (attention) register their references as priority −100 kernels; that
stops in step 0.

**D2 — The core seam is adoption at load time.** No layout pre-pass, no `layouts()`:
- `build_model` builds on meta and no longer calls `to_empty`.
- `load_weights` maps each checkpoint value as today. A dense tensor: allocate the parameter on the device
  (`empty_like` of the meta parameter) and copy (casting dtype). A `QTensor`: call
  `owner.adopt(ctx, leaf_name, qtensor)` on the module owning the parameter; the unit uploads the parts as its own
  uint8 parameters, records the layout and selects its kernel. A unit without `adopt`, or one that returns False,
  is a load error ("`<path>`: unit `<ref>` cannot read layout `<layout>`").
- A parameter still on meta after loading is "not loaded" (also covers a multi-part layout missing a part).
- Value transforms (`WeightRule.transform`) apply to dense values only; a transform matching a `QTensor` is an
  error.
- The core never iterates parts or knows block sizes; `QTensor` is a loader-time value, not a parameter type.
  Formats with several checkpoint tensors per parameter (GPTQ/AWQ, v1) are grouped by their source into one
  `QTensor`; M0.2's expert stacks are one 3-D `QTensor`.

**D3 — A GGUF file is self-contained, as in llama.cpp** (design §5.1's "HF base tokenizer" is revised). We build
the tokenizer from the GGUF table ourselves with `tokenizers` primitives — byte-level BPE from tokens + merges,
CONTROL → special added tokens, USER_DEFINED → atomic added tokens, UNUSED unreachable, the pre-tokenizer regex from
a small std table keyed by `tokenizer.ggml.pre` (unknown value → error) — and do **not** use transformers' GGUF
converter (it drops the USER_DEFINED tokens, reports the wrong eos and admits the 267 UNUSED `[PAD…]` entries,
measured by review). Stop ids = GGUF eos/eot ∪ CONTROL tokens whose text is a known end-of-turn marker (llama.cpp's
rule). The chat template is the GGUF's. An HF base is never needed at runtime; it appears only in tests, which prove
our GGUF tokenizer encodes identically to HF's (§4.2).

**D4 — No format names in core; sources are detected by their plugins.** Each weight-source plugin registers
`detect(path) -> bool` beside its factory; the core picks the single match (zero or several is an error).
`EngineConfig.source` defaults to `None` (detect); the plan records the resolved `source.kind`, so the bf16 locks'
`source = "hf_safetensors"` line is unchanged; `base` is omitted from the plan when `None`.

**D5 — The quantized-linear boundary is float32.** The ported kernels take f32 activations and write f32
(`mmq.cu:132-133`, `mmvq.cu:1437-1438`, `quantize.cuh`). The residual stream stays bf16; the kernels around a GGUF
linear absorb the conversion instead of standalone casts: rmsnorm/add_rmsnorm emit f32 where their consumer is a
GGUF linear, silu_mul reads and writes f32, the q/k norms read f32 and write bf16, add_rmsnorm accepts an f32
`x`. Only the attention output needs a cast before `o_proj`. These are small Triton kernels of our own, selected
like the existing ones, with the dtype pair as facts.

## 4. Design

### 4.1 Packages

| package | holds |
|---|---|
| `std/holon_std/quant/ggml` (`scheme.ggml.*`) | one `Scheme` per ggml type, the `gguf.native` layout, `dequant_ref` per type (gguf-py, pinned `gguf==0.19.0`, version asserted in the reference module so `sem` covers it; `numerics = "exact"`) |
| `std/holon_std/qops` (`op.qlinear@1`, `op.qembedding@1`) | quantized-op declarations and their `reference.py` — kept out of `std.ops` so its `sem` does not move |
| `std/holon_std/weights/gguf` | reader (memmap), `detect`, generic llama.cpp key mapping, base cross-check, identity |
| `std/holon_std/ops`, `ffn`, `mixers/gqa`, `blocks` | `adopt` on Linear/Embedding/LMHead; same-type merges through the kernel; f32-boundary kernel selection (D5) |
| `plugins/holon_plugins/kernels/ggml_cuda` | vendored llama.cpp subset, shim, wrapper, prebuild; `kernel.qlinear@1`, `kernel.qembedding@1` |
| `plugins/holon_plugins/kernels/triton_ops` (new) | the D5 boundary kernels (rmsnorm→f32, silu_mul f32, add_rmsnorm with f32 x, q/k norm f32→bf16) |
| `plugins/holon_plugins/models/qwen3` | `GGUF_WEIGHTS`, `arch.gguf.qwen3` resource, bundle from the source |
| `plugins/tests/fixtures/ref_kernels` | D1's test-only reference kernels |
| `judge` | GGUF golden recorder (own name map), q8_1-emulation floor, L0 for ggml types, planted bugs |
| `bench` | in-process llama-bench equivalent; `llama-server` backend |

### 4.2 Weight source `gguf`

- **Input**: a local `.gguf` path (the hub cache path included). Resolving "repo id + file" is deferred (it would
  pull every quant of the repo through `snapshot_download`).
- **`arch`** = `"gguf." + general.architecture` (`gguf.qwen3`); the model plugin's manifest provides that
  resource, so std holds no per-model table (invariant 10).
- **Config**: std maps only the llama.cpp keys every text architecture shares (`block_count`,
  `embedding_length`, `feed_forward_length`, `attention.head_count{,_kv}`, `attention.key_length`,
  `rope.freq_base`, `attention.layer_norm_rms_epsilon`, `context_length`, vocab size from the token table, tie =
  no `output.weight`). Floats are taken at their shortest float32 representation (`9.999999974752427e-07` →
  `1e-06`), so the recipe term equals the HF checkpoint's. Model-specific keys (M0.2 `expert_*`, M0.3 `ssm.*`) are
  mapped by the model plugin beside its weight map. A key the recipe needs that the file lacks is an error naming it.
- **Tokenizer** (D3), built by the source from the GGUF table: `tokenizer()` returns a `tokenizers.Tokenizer`,
  `stop_ids()` the stop set, `chat_template()` the template. Tests (not runtime): with the cached `Qwen/Qwen3-8B`,
  (1) every CONTROL/USER_DEFINED string encodes to exactly its own id; (2) `encode` equals HF's and
  `encode(decode(ids)) == ids` on the judge prompts and an adversarial set (whitespace runs, CRLF, long digit runs,
  CJK, emoji, NBSP, contractions, the special tokens inside text); (3) the stop set equals HF's
  `generation_config` (151645, 151643); (4) the GGUF and HF chat templates render the judge's agent transcripts
  identically, or the differences are listed.
- **Frontend**: `ModelPlugin.bundle(source)` replaces `bundle(path)`; for GGUF the bundle wraps the source's
  tokenizer and template; the CLI passes the source.
- **Identity**: `<hub repo>/<file stem>@<16 hex>` for hub-cache files (the blob name is the sha256, free), else
  `<file stem>@<16 hex>`. A re-upload (even template-only) is a new model id with its own golden and lock history.
- **Tensors**: dense types yield tensors; quantized types yield `QTensor(layout, (out, in), {"blocks": u8 view of
  the memmap})`. llama.cpp pads quantized rows to 512 elements (`ggml-cuda.cu:910-916`); `adopt` allocates the
  padded size and zeroes the tail (a no-op for Qwen3-8B, needed for `in` ∉ 512ℕ).

### 4.3 Units

- `Linear.adopt` / `Embedding.adopt` / `LMHead.adopt` (untied `output.weight`). A tied quantized head reuses the
  embedding's blocks: `LMHead` selects `qlinear@1` from the embedding's layout in `finalize_weights`.
- **Merges**: `Linear.merge` returns `None` whenever a layout is set, unless every unit has the same scheme and
  `in` — then rows are concatenated (ggml rows are independent block runs) and the merged unit runs through the
  kernel. q+k and gate+up merge in both target files; v stays separate where its type differs. The composites stop
  calling `F.linear` on a merged weight and call the merged unit; the unmerged SwiGLU path uses the selected
  `silu_mul` kernel, not `ref.silu_mul`.
- Kernel facts: `layout` (scheme elem + name), `in`, `out`, `act_dtype`; for D5's neighbours the in/out dtypes.

### 4.4 Core changes (generic)

1. D2 (load-time adoption) and D4 (detected sources, resolved kind in the plan).
2. **Workspaces grow to the maximum**: `ctx.workspace(name, nbytes)` returns a buffer at least `nbytes`; the
   buffer is sized to the largest request after loading and before `finalize_weights`, so sites with different
   `in` share one. `max_step_tokens` joins `ctx.options`.
3. **Memory reserve**: `EngineConfig.memory_reserve` (bytes, absolute) replaces `memory_fraction` as the safety
   margin when set — a 10 % fraction withholds ~1 GiB on a 16 GB card. Settings for a class live in its lock.
4. **KV index hoist**: `write_index // page` and `% page` move into `Executor._prepare` (computed once per step, not
   per layer — 72 small kernels per decode step today).
5. D1: kernel selection never returns a reference; a missing production kernel is a startup error.

### 4.5 Kernel plugin `kernels.ggml_cuda`

- **Vendoring**: llama.cpp `51ce9c11` files committed under the plugin directory (so `src` pins the code):
  `mmq.cu/.cuh` + all 24 `mmq-instance-*` files the type switch links, `mmvq.cu/.cuh`, `quantize.cu/.cuh`,
  `getrows.cu/.cuh`, `convert.cu/.cuh`, `common.cuh`, `vecdotq.cuh`, `mma.cuh`, `ggml-common.h` and their includes;
  `SHA256SUMS`, `vendor.sh`, a small reviewed `vendor.patch` (bypasses the `GGML_CUDA_MMQ_PREC` / `GGML_CUDA_PDL`
  environment reads — no env configuration, invariant), `LICENSE` (MIT) shipped as package data, `[provenance]` in
  `plugin.toml`; a CI check re-runs `vendor.sh` and diffs.
- **Entry points without llama.cpp's stream**: `ggml_backend_cuda_context::stream()` creates a private stream
  (`common.cuh:1519-1526`) that would escape graph capture. The wrapper calls `quantize_mmq_q8_1_cuda`,
  `mul_mat_q_case<T>` and `get_rows_cuda` directly with torch's current stream; `mmvq`, whose switch is static,
  gets the stream injected into the context per call (read every call: capture uses another stream).
- **Shim**: the ggml-base helpers (incl. `ggml_type_name`), `ggml_cuda_info` with the true `cc = 100·major +
  10·minor`, `nsm`, `smpbo`, `warp_size`; a destructor for `ggml_backend_cuda_context`; zeroed `op_params` on stack
  `ggml_tensor`s; the pool is a bump allocator over our workspace, reset per call, 256-byte aligned, **aborting on
  overflow** (never `cudaMalloc`). `get_rows` takes int32 ids (cast in the unit).
- **Dispatch**: llama.cpp's own `ggml_cuda_should_use_mmvq/mmq` per call from shapes (host-side, fixed per graph
  bucket). Workspace bound per call: mmq `rows × pad512(in) × 144/128 + J_max × 144` B plus the stream-k fixup
  `nsm × 128 × 128 × 4` B; mmvq ≤ `8 × pad512(in) × 36/32` B; rows = max(`max_step_tokens`, seqs × (1+k),
  `PROMPT_LOGPROB_CHUNK`).
- **Build**: llama.cpp's flags (`-U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__
  -U__CUDA_NO_BFLOAT16_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__ -use_fast_math -extended-lambda -O3`), `12.0`
  mapped to `12.0a` (the mxfp4 instance needs it); a static library of the vendored set + a thin wrapper, cached
  under the user cache by (vendored hash, flags, SM, nvcc/torch CUDA major). Build time ≈ 9–10 CPU-minutes per SM
  (≈ 2 min wall with ninja on 8 cores). **Prebuild is mandatory**: `python -m holon_plugins.kernels.ggml_cuda.prebuild`;
  the engine never compiles at start (restart SLO). Missing build, no nvcc, CUDA-major mismatch: the provider is
  unavailable and D1 makes it a startup error with the prebuild command in the message.
- **Props**: `graph_safe` and `deterministic` once L1 confirms them on the port: no atomics; stream-k fixup sums in
  a fixed order (`mmq.cuh:1241-1305`), so results are run-to-run identical for a given SKU (they depend on `nsm` and
  row count — not batch-invariant). The fingerprint class already fixes the SKU.

### 4.6 Performance work in M0.1

Without it the review estimates 0.70–0.75× llama.cpp at decode on the 5080 (llama.cpp tg128 Q4_K_M: 6.29 ms per
token, weights ≈ 5.2 ms): D5's fused boundary (no standalone casts), same-type merges, the selected `silu_mul`, and
the KV index hoist bring the per-layer node count near llama.cpp's (~17–22) for an estimated 0.87–0.92×. Optional
if short: fuse q/k norm + rope + KV write. Not borrowed: llama.cpp's gate+up+swiglu mmvq fusion at batch 1
(`ggml-cuda.cu:3840-3895`) — revisit if decode is short of 0.8×.

### 4.7 Numerics and the judge (all thresholds fixed before measuring)

**Supported types** (L0): those in the target files (Q4_K, Q5_K, Q6_K, Q3_K, IQ4_XS, IQ3_XXS, IQ3_S, IQ2_S, IQ2_XS)
plus Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2_K, IQ1_S, IQ1_M, IQ2_XXS, IQ4_NL. Excluded: types gguf-py 0.19 cannot
dequantize (Q1_0, Q2_0) and MXFP4/TQ (not in GGUF LLM builds we serve).

**L0** (per class: sm_86, sm_89, sm_120a):
- *Inputs*: real tensors from the cached files; random-byte blocks for every type with realistic fp16 `d`/`dmin`
  (negative, zero, subnormal, ≤ 1) — gguf-py can quantize only Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, but every byte pattern is
  a valid encoding; activation rows all-zero (amax 0) and with one large outlier; `in` ∈ {768, 2304, 4096, 12288}
  with poisoned bytes past the tensor; `out` not a multiple of the mmq tile.
- *Row counts* per (type, cc) from `ggml_cuda_should_use_mmvq`: {1, t, t+1, 8, 9, 64, 128, 256, 512,
  `max_step_tokens`} and stream-k with and without fixup.
- *Checks*: `qlinear` vs a **q8_1-emulated reference** (per 32 values `d = amax/127`, `q = round(x/d)`, D4/DS4
  storage, the Q4_K/Q5_K min-term from unquantized sums) with the per-element `l0.check(bf16, in)` budget — this
  catches one wrong row or block, which a global NMSE does not; and NMSE ≤ 5e-4 vs the unquantized reference with
  random `x`. `convert`/`get_rows` vs `dequant_ref`: bit-exact for single-product types, ≤ 1 fp32 ulp for affine
  types (FMA contraction under `-use_fast_math`), or bit-exact if those translation units build with
  `-fmad=false` — decided in step 1 by measurement, recorded in the test.

**L2 golden**: the judge builds `Qwen3ForCausalLM` (fp32, CPU) from the HF config and fills it with tensors
dequantized by calling `gguf.quants.dequantize` directly (never std's wrapper), through the judge's own name map.
The recording's meta holds the file sha256, the gguf-py version and the name-map hash. One golden per quant file.

**Per-tensor gate (judged)**: NMSE(dequantized GGUF tensor, HF bf16 tensor) ≤ 0.2 for every tensor (measured:
Q6_K ≈ 3.5e-4, Q4_K ≈ 5.3e-3, IQ3 ≈ 2–3.6e-2; a transposed `attn_q` gives 2.0, a gate/up swap 1.9). This checks
the beliefs both name maps share (no Q/K permutation, orientation). KL(dequantized golden ‖ bf16 golden) is
reported per file, with a tripwire at 10× the first measured Q4_K_M value against a broken recorder.

**L2 tiers** (budgets = M0's: KL p50/p99 ≤ 2× floor, top-1 ≥ floor − 0.25 pt, |ΔPPL| ≤ |floor| + 0.25 %):
- **Tier A — assembly**: the engine with the GGUF dequantized to dense bf16 at load (a judge-build option of the
  source, using gguf-py on the CPU; production cuBLAS path, no reference kernels) vs the dequantized golden; floor =
  HF bf16 with the dequantized weights (the M0 rule). Catches wrong config, names, wiring, rope at M0 sensitivity.
  Runs on the A5000 (16 GB of bf16).
- **Tier B — kernels**: the production GGUF engine vs the same golden; floor = HF bf16 with a judge-side q8_1
  emulation on every quantized linear's input (lm_head included), with the Q4_K/Q5_K split dequant for the
  min-term. Prefill (`mmq`) on the whole prompt set; decode (`mmvq`) via a second run with `max_step_tokens` ≤ the
  class's mmvq threshold on a 32-prompt subset. On rtx-a5000 and rtx-5080.
- **Planted bugs with expected verdicts, recorded before any measurement**: rope θ ×2, transposed `attn_output`,
  gate/up swap, Q/K permuted by mistake (Tier A must fail); the obsolete IQ3_S grid, one tensor dispatched as a
  wrong type, a stale q8_1 workspace at ~1 % of steps (Tier B must fail). δ never changes without a written cause,
  and only if the planted set still fails afterwards.

### 4.8 Memory on a 16 GB card

The RTX 5080 reports 15.47 GiB (15841 MiB). Qwen3-8B KV in bf16: 144 KiB/token; 64K tokens = 9.00 GiB.

| GiB | Q4_K_M | UD-IQ3_XXS |
|---|---|---|
| weights | 4.68 | 3.17 |
| CUDA context, FlashInfer + ggml workspaces | ≈ 0.65 | ≈ 0.65 |
| KV 64K | 9.00 | 9.00 |
| left for activations + graph pool | ≈ 1.14 | ≈ 2.65 |

Activation peak (review estimate with the f32 boundary): ≈ 0.8 GiB at 4096 step tokens, 0.4 at 2048, 0.15 at 1024;
prompt-logprob head ≈ 0.3; graph pool at the 64-row bucket ≈ 0.15–0.25. So Q4_K_M at 64K needs the absolute
reserve (§4.4.3, ~256 MiB), `max_step_tokens` 1024 and `max_seqs` 32 for the class — recorded in the `rtx-5080`
lock's `[engine]`. If measurement disagrees, the breakdown goes to the evidence and Q4_K_M at 64K is reported
partial (remedy: quantized KV, M0.3b).

### 4.9 Bench

- **Bars**: an in-process Holon equivalent of llama-bench (pp512 as one 512-token prefill; tg128 at depth 0 and 256;
  warm-up excluded; host CPU per step recorded) vs llama-bench at the same commit, flags identical
  (`-use_fast_math`, 120a), `-ngl 99 -fa on`, plus `-d 256`.
- **Serving**: `llama-server` with `-kvu` (unified KV — otherwise `-c` is split per slot), `--parallel` = max batch,
  `-b` = `max_step_tokens`, `-ub` recorded, f16 KV (noted vs our bf16, same bytes), prompts as token-id arrays,
  greedy (`temperature 0`, `top_k 1`), `ignore_eos`, `n_predict`; throughput from the server-side `timings`; prompt
  cache reported both default and `-cram 0`. Holon driven through its own HTTP frontend for this comparison.

## 5. Implementation plan

The same three reviewers for the whole milestone; each step ends green (lint, tests).

| # | Step | Deliverable / test | Days |
|---|---|---|---|
| 0 | D1: references stop registering as kernels; test fixture plugin; startup error | full suite green via the fixture; `holon serve` with a missing provider fails naming it | 0.5 |
| 1 | `quant/ggml` + `qops`; L0 dequant harness; FMA decision | dequant vs gguf-py on real + random blocks, all supported types | 1 |
| 2 | `weights/gguf`: reader, detect, generic keys, GGUF tokenizer + stop set + tests vs HF, identity; core D2/D4 | CPU: tiny synthetic Q8_0/Q4_0 GGUF model (gguf-py writer) vs dense, through the fixture kernels; bf16 path snapshot byte-identical | 2 |
| 3 | `kernels.ggml_cuda`: vendor + shim + wrapper + build/prebuild (sm_86/89/120a link test first); workspace grow-to-max | L0 (all checks of §4.7) on sm_86; graph capture; run-to-run determinism | 4 (in parallel with 1–2) |
| 4 | Qwen3 `GGUF_WEIGHTS`, `arch.gguf.qwen3`, bundle(source), CLI `.gguf` path, tools/judge resolve | Tier A L2 on A5000 for both files; per-tensor gate; planted Tier A bugs fail | 1.5 |
| 5 | D5 boundary kernels (`triton_ops`), same-type merges, KV index hoist, memory reserve | Tier B L2 (prefill + decode) on A5000; planted Tier B bugs fail; bf16 snapshot byte-identical | 2.5 |
| 6 | Consumer classes: L0 on 4090/5090 (sm_89/120a); 5080 fit at 64K; in-process bars; `llama-server` backend + traces; Tier B on 5080; `tune` + GGUF locks; GGUF snapshot ×3 | evidence; locks | 2 |
| 7 | bf16 locks re-propagated on a5000/4090/5090 (acceptance 4) | locks | 0.5 |
| 8 | Docs: design §5.1, §5.5, §5.6 (incl. "tied lm_head dequantized in chunks" → `qlinear` on the embedding's blocks), §7.3; plan §4 status | — | 0.5 |

About 14.5 days. Step 3 is the long pole and starts at once; steps 4–5 need it.

## 6. Risks

- **Port link and sm_120a**: unproven until step 3's link test (first task of the step).
- **5080 fit for Q4_K_M at 64K**: ≈ 1.1 GiB of headroom before the reserve; depends on the measured context size.
- **0.8× decode**: estimated 0.87–0.92× with §4.6; the remaining lever is llama.cpp's batch-1 gate+up+swiglu fusion.
- **Port drift**: one pinned commit; refresh = vendor.sh + link test + L0.
- **Tier B floor too tight or too loose**: guarded by the planted set; the only escape hatch is a written cause.

## 7. Answers to the revision-1 questions

1. Core seam: load-time adoption (D2), not a pre-pass or `BuildCtx` plumbing.
2. `dequant_ref` = pinned gguf-py, called directly by the judge; independence comes from the CUDA `convert` kernels
   matched against it, and from the per-tensor gate against HF bf16.
3. L2 floor: two tiers fixed in advance (§4.7); llama.cpp's logits are not a floor (they share the kernels).
4. bf16 locks: one approval with a byte-identical snapshot (acceptance 4) — unavoidable since the core `src`
   moves; new ops go to `std.qops` so `std.ops`' `sem` does not.
5. Reference kernels: never (D1).

## 8. Deferred, deliberately

"repo id + file" resolution; pre-tokenizer table entries beyond the archs we serve; llama.cpp's batch-1 gate/up/swiglu fusion; fused q/k
norm + rope + KV write; CI-built per-SM extension wheels (v0 prebuilds locally); fine-grained (per-class) `src`
pins.
