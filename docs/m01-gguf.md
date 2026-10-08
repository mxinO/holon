# M0.1 — GGUF quantized models: design and implementation plan

Status: **revision 3.1, implemented; acceptance met** (2026-10-08, plan §4.1). Revisions 1 and 2 were challenged by
three reviewers (architecture/API, kernels/performance, numerics/judge); every finding is resolved below or deferred
in §8. Scope: plan §4. Builds on design §5.1, §5.4–5.6 and spike S2 (`spikes/s2_gguf/README.md`).

## 1. Goal and acceptance

Serve Qwen3-8B from llama.cpp GGUF files — Q4_K_M and UD-IQ3_XXS (unsloth) — with llama.cpp's own quantized CUDA
kernels, and take the first 16 GB consumer data point. Acceptance:

1. The `models/qwen3` diff adds only the GGUF weight map, the `arch.gguf.qwen3` resource and the bundle taking its
   weight source. `canonical_hash(recipe(cfg, "hf_safetensors"))` is unchanged.
2. L0 passes for every supported type (§4.6) on sm_86, sm_89 and sm_120a.
3. L2 (§4.6) passes for both files — Tier A on rtx-a5000 (dense bf16 8B does not fit 16 GB), Tier B on rtx-a5000
   and rtx-5080; each planted bug fails at the level recorded in §4.6; the per-tensor gate passes; the quantization
   error vs bf16 is reported.
4. bf16 Qwen3-8B on rtx-a5000, rtx-4090, rtx-5090: `propagate` shows a snapshot trace byte-identical to the current
   lock and L2 passing; lock changes are limited to `[units]` lines (`src`, and `sem` of `std.ops`, whose
   `reference.py` gains the quantized ops; `std.ffn` and `std.mixers.gqa` `reference.py` stay byte-identical);
   `[sites]`, `[engine]` and the recipe hash unchanged; approved once.
5. GGUF snapshot trace run-identical across 3 fresh engines (Q4_K_M; rtx-a5000 and rtx-5080).
6. RTX 5080: both files with ≥ 64K KV tokens (§4.7). Performance vs llama.cpp on the same card, file and commit
   (§4.8): in-process pp512 and tg128 against the S2 llama-bench bars (≥ 0.8×: Q4_K_M ≥ 6380 t/s pp512, ≥ 127 t/s
   tg128), decode and agent traces vs `llama-server`.
7. GGUF locks for rtx-5080 and rtx-a5000 (model ids `unsloth/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M@…`,
   `…/Qwen3-8B-UD-IQ3_XXS@…`; bf16 Qwen3-8B does not fit 16 GB).

Non-goals: MoE (`mul_mat_id`, M0.2); quantized KV (M0.3b); HF quant formats (v1); converters other than identity.

## 2. Starting point

- **S2** decided to port llama.cpp's current quantized matmul kernels behind a small shim; it measured the
  SGLang-vendored set (correct, graph-safe) and compiled the current `mmq`/`mmvq` files for sm_86 against a 20-symbol
  stub. Not yet proven: the full link (all `mmq` cases), sm_120a, the torch-extension flags, PDL inside a torch graph.
- **API types** `Scheme`/`Layout`/`QTensor` exist, unused.
- **Loading** builds on meta, materializes every parameter densely, then copies by shape (16 GB of bf16 for 8B).
- **The files** (cached; also used by S2): no Q/K permutation (llama.cpp's Qwen2/3 converters keep NEOX order); F32
  norms; q/k share one type in every layer of both files, as do gate/up; v differs; the file carries `qwen3.*`
  hyperparameters, the tokenizer table and the chat template.
- **llama.cpp needs nothing but the file**: tokenizer from `tokenizer.ggml.tokens/merges/token_type` with a
  pre-tokenizer regex chosen by `tokenizer.ggml.pre` from a built-in table (`src/llama-vocab.cpp:383`, `qwen2`),
  CONTROL/USER_DEFINED tokens matched atomically; stop set = eos/eot plus CONTROL tokens whose text is a known
  end-of-turn marker (`llama-vocab.cpp:3008-3037`), plus the FIM pad/repo/file-sep ids. It does not apply Unicode
  normalization, which HF's Qwen3 `tokenizer.json` does (NFC).

## 3. Decisions

**D1 — Production code never calls a reference.** A `reference.py` defines its op (`sem`) and serves L0 and the
judge only. No reference is registered as a kernel: the `kernel.*` provides of `std.ops`, `std.ffn` and
`std.mixers.gqa` are removed, and kernel selection fails at startup on a missing production kernel, naming op, facts
and device (invariant 7). Production units stop calling `reference.py` directly (`Linear`/`LMHead` use `F.linear` in
`units.py`; a model built without state selects `attention.causal@1`). Tests get the references as CPU-only kernels
from the test-only distribution `holonlm-testing` (`testing.ref_kernels`): an entry point installed in the
development environment only, so engines in subprocesses find it too; GPU tests call its adapters directly as
oracles; an import contract keeps production code from importing it. After D1 no production CPU kernels exist, so
a CPU lock is a test artifact by construction.

**D2 — Quantized weights are adopted at load time.**
- `build_model` builds on meta without `to_empty`.
- `load_weights`: a dense value is copied into a device tensor swapped in place (`torch.utils.swap_tensors`, the
  Parameter object kept); a `QTensor` value goes to `owner.adopt(ctx, leaf_name, qtensor)` on the owning module, which uploads
  the parts as its own uint8 parameters (the ggml kernels need no padding: they support `in` % 256 = 0), and records the
  layout (the kernel is bound later, §4.3). No `adopt`, or `False`: load error naming path, unit and layout. A parameter still
  on meta afterwards: "not loaded". `WeightRule.transform` on a `QTensor`: error.
- The core never iterates parts or knows formats; `QTensor` is a loader-time value. Several checkpoint tensors per
  parameter (GPTQ/AWQ, v1) are grouped by their source; M0.2 expert stacks are one 3-D `QTensor`.

**D3 — A GGUF file is self-contained, as in llama.cpp.** The source builds the tokenizer from the file with
`tokenizers` primitives — byte-level BPE from tokens + merges; CONTROL → special added tokens; USER_DEFINED →
atomic added tokens; UNUSED unreachable; the pre-tokenizer regex from a small std table keyed by
`tokenizer.ggml.pre` (unknown → error), including the normalizer the original tokenizer uses — NFC for `qwen2`,
deliberately unlike llama.cpp, so decomposed input encodes as the model was trained. transformers' GGUF converter is
not used (it drops USER_DEFINED tokens, reports the wrong eos, admits the 267 UNUSED entries). Stop set = eos/eot ∪
CONTROL tokens whose text is on llama.cpp's end-of-turn list (not its FIM additions), kept in the existing
`WeightSource.eos_token_ids`. The chat template is the file's. No base model at runtime.

**D4 — Sources are detected from manifests.** Each weight-source plugin declares `detect` patterns in its manifest
(`["*.gguf"]`, `["config.json"]`), matched against a path's file name or the files inside a directory; the core
matches them without importing anything (exactly one match, else an error; test fixtures declare none). `EngineConfig.source = None` (detect); the plan records the resolved kind only — the bf16 locks keep
`source = "hf_safetensors"` and never pin the GGUF source.

**D5 — A quantized linear is dtype-preserving: bf16 in, bf16 out**, exactly like the dense one, so no neighbour,
residual or KV write changes; `lm_head` takes `out_dtype = float32` (its logits are f32 anyway). The ported kernels
take and return f32 (`mmq.cu:132-133`, `mmvq.cu:1437-1438`). `vendor.patch` (≈ 60–80 mechanical lines in
`quantize.cu/.cuh`, `mmq.cu`, `mmvq.cu`): the activation quantizers read bf16 (`quantize_q8_1` scalar loads;
`quantize_mmq_q8_1` 8-byte loads of 4 bf16, alignment asserted; launchers take `const void *` + type); `mmvq` writes
bf16 or f32 by a runtime flag passed as `void *dst` down its chain (no new template instantiations), branching at its
store; the callers' F32 asserts follow. So mmvq calls (decode at rows ≤ the type's threshold) launch what llama.cpp
launches. `mmq` calls (prefill, and decode buckets above the threshold) write f32 into a per-call buffer and one cast
writes bf16: ≈ 5 % of a 2048-token prefill step and ≈ 2–3 % of a b8/b64 decode step — inside the bar, stated rather
than hidden. Prefill runs `mmq` in 512-row chunks, llama.cpp's own ubatch, so that buffer is ≤ 512 × out × 4 B and
an ordinary activation (caching allocator / graph pool), never a workspace. Fusing f32 into the neighbours (revision
2's D5) is a fallback, triggered only by a measured miss (§4.5).

## 4. Design

### 4.1 Packages

| package | change |
|---|---|
| `std/holon_std/weights/gguf` | new: memmap reader, manifest `detect`, generic llama.cpp keys, tokenizer (D3), identity; both sources expose `tokenizer()` (a transformers fast tokenizer — HF via `AutoTokenizer`, imported lazily; GGUF wrapping the built `tokenizers.Tokenizer` with its template) and `chat_template()`, so bundles need no format branch |
| `std/holon_std/ops` | `adopt` on Linear/Embedding/LMHead; `reference.py` gains `qlinear` / `qembedding` over an already-dequantized weight (the ops' `sem`); same-type merges through the kernel |
| `std/holon_std/ffn`, `mixers/gqa` | composites call merged units (not `F.linear` on a merged weight); SwiGLU uses the selected `silu_mul`; KV index math moves into gqa's `Planner.prepare_device` (once per step, not per layer); gqa's state-free path becomes test-only (`units.py` only) |
| `plugins/holon_plugins/kernels/ggml_cuda` | new: vendored llama.cpp subset + patch, shim, C API, torch wrapper, prebuild; `kernel.qlinear@1`, `kernel.qembedding@1` |
| `plugins/holon_plugins/models/qwen3` | `GGUF_WEIGHTS`, `arch.gguf.qwen3`, `bundle(source)` |
| `testing/holon_testing/ref_kernels` | new, test-only distribution (D1) |
| `judge` | GGUF golden (own name map, gguf-py direct), q8_1 emulator (shared with L0), Tier A wrapper source, planted bugs |
| `bench` | in-process llama-bench equivalent; `llama-server` backend |
| `core` | D1 (no reference selection), D2, D4; `max_step_tokens` in `ctx.options`; CLI flags for `str \| None` fields |

`Scheme`/`Layout` values for ggml types are plain strings the source builds (`Layout(Scheme("ggml.q4_k"),
"gguf.native")`); no separate quant package — under D1 nothing at runtime needs a dequantization, and the judge
and L0 call gguf-py directly.

### 4.2 Weight source `gguf`

- **Input**: a local `.gguf` path (hub-cache paths included).
- **`arch`** = `"gguf." + general.architecture`; the model plugin provides it (invariant 10: no model table in std).
- **Config**: HF-style keys from the llama.cpp keys every text architecture shares (`block_count`,
  `embedding_length`, `feed_forward_length`, `attention.head_count{,_kv}`, `attention.key_length`, `rope.freq_base`,
  `attention.layer_norm_rms_epsilon`, `context_length`, vocab from the token table, tie = no `output.weight`), floats
  at their shortest float32 form (`9.999999974752427e-07` → `1e-06`), plus the raw arch-prefixed keys under `"gguf"`
  for model-specific mapping in the model plugin (M0.2/M0.3). Fields GGUF never carries need no defaults for Qwen3
  (its recipe reads `attention_bias`/`use_sliding_window` with `c.get` and never reads `hidden_act`); later models
  supply theirs in their GGUF mapping. A key the recipe needs and the file lacks: error. Test
  (tests only): the GGUF-derived config equals HF's `config.json` on every field the recipe reads.
- **Tokenizer** (D3): `tokenizer()`, `chat_template()`, and the stop set in `eos_token_ids`. Tests with the cached `Qwen/Qwen3-8B` (tests
  only): every CONTROL/USER_DEFINED string encodes to its own id (their special flag is not compared: the fim/repo
  tokens are CONTROL in GGUF but non-special in HF); `encode` equals HF's and round-trips on the judge prompts and an
  adversarial set (whitespace runs, CRLF, long digit runs, CJK, emoji, NBSP, contractions, special tokens in text,
  NFD / combining characters and Hangul jamo, the `[PAD151669]` text);
  the stop set equals HF's `generation_config` (151645, 151643); the two chat templates render the judge's agent
  transcripts identically or the differences are listed.
- **Identity**: `<hub repo>/<file stem>@<16 hex>` for hub-cache files (blob name = sha256), else
  `<file stem>@<16 hex>`. A re-upload is a new model id.
- **Tensors**: dense types as tensors; quantized as `QTensor(layout, (out, in), {"blocks": memmap view})`.

### 4.3 Units and merges

`Linear.adopt` / `Embedding.adopt` / `LMHead.adopt` upload the parts and record the layout; kernels are bound in
a core pass after every `finalize_weights` (`bind_kernels(ctx)` on each module that has it), so units merged away —
they give up their weight — leave no site in the lock; a tied quantized head takes the embedding's layout in its
`finalize_weights` and binds `qlinear` on its blocks. `Linear.merge` returns `None` when a layout is set unless all units share scheme and
`in`, in which case rows are concatenated (ggml rows are independent block runs) and the merged unit runs through
`qlinear`: q+k and gate+up merge in both files, v stays separate where its type differs. Kernel facts: `layout`,
`in`, `out`, `act_dtype`.

### 4.4 Kernel plugin `kernels.ggml_cuda`

- **Vendoring**: llama.cpp `51ce9c11` files committed under the plugin directory: `mmq.cu/.cuh`, `mmid.cu/.cuh` (the
  ids helper `mmq.cu` links), all 22 `mmq-instance-*.cu`, `mmvq.cu/.cuh`, `quantize.cu/.cuh`, `getrows.cu/.cuh`,
  `convert.cu/.cuh`, `common.cuh`, `vecdotq.cuh`, `mma.cuh`, `ggml-common.h` and their includes; `SHA256SUMS`,
  `vendor.sh`, `vendor.patch` (reviewed: D5's bf16 quantize-in and mmvq-out; the `GGML_CUDA_MMQ_PREC` and
  `GGML_CUDA_PDL` environment reads replaced by constants — **PDL on**, as in llama.cpp, for parity); `LICENSE` (MIT)
  as package data; `[provenance]` in `plugin.toml`; a CI check re-runs `vendor.sh` and diffs.
- **Structure**: a static library of the vendored code + the shim + a small `extern "C"` API (plain pointers),
  compiled with llama.cpp's flags (`-use_fast_math -extended-lambda -O3`, `-Xcompiler -fPIC`), against torch's shared
  cudart; a thin torch wrapper includes only the C API header, so torch's `-D__CUDA_NO_HALF*` never meets
  `common.cuh`. SMs 86, 89 and `120a` (the mxfp4 instance requires it).
- **Streams**: llama.cpp's context creates a private stream (`common.cuh:1519-1526`) that would escape capture; the
  shim injects torch's current stream into the context per call (read every call) for `ggml_cuda_mul_mat_q` and
  `ggml_cuda_mul_mat_vec_q` — one mechanism, llama.cpp's own argument setup kept; `get_rows_cuda` takes a stream
  directly. The stream invariant this relies on is recorded in §8.
- **Shim**: ggml-base helpers incl. `ggml_type_name`; `ggml_cuda_info` with true `cc`, `nsm`, `smpbo`, `warp_size`;
  a destructor for the context; zeroed `op_params`; the pool is a bump allocator over our workspace, reset per call,
  256-B aligned, aborting on overflow (never `cudaMalloc`); `get_rows` gets int32 ids.
- **Dispatch**: llama.cpp's `ggml_cuda_should_use_mmvq/mmq` per call from shapes (fixed per graph bucket).
- **Workspaces**: fixed-size `ctx.workspace(f"ggml_cuda[in={in}]", bytes)` per distinct `in` (today's `static()`)
  holding only what depends on `in`, the options and `nsm`: q8_1 activations for one 512-row chunk
  (`512 × pad512(in) × 144/128 + J_max × 144` B; mmvq needs less) + the stream-k fixup (`nsm × 128 × 128 × 4` B) —
  ≈ 13 MB at `in` = 12288 on the 5080. The f32 mmq output is a per-call activation (D5).
- **Build**: ≈ 9–10 CPU-minutes per SM (≈ 2 min wall with ninja on 8 cores), cached by (vendored hash, flags, SM,
  CUDA major). **Prebuild is mandatory** (`python -m holon_plugins.kernels.ggml_cuda.prebuild`); the engine never
  compiles at start. No build, no nvcc, CUDA-major mismatch → provider unavailable → D1's startup error, which prints
  the prebuild command.
- **Props**: `graph_safe` and `deterministic` once L1 confirms them on sm_86 and sm_120a (no atomics; the stream-k
  fixup sums in a fixed order, `mmq.cuh:1241-1305`; results depend on `nsm` and row count — fixed per class, not
  batch-invariant).

### 4.5 Performance

Decode at batch 1 is llama.cpp's own kernels for every quantized linear (D5); the remaining differences are ours:
non-GEMM kernels per layer (norms, rope, attention, KV write, residual adds) and per-step host work. Planned:
same-type merges, the selected `silu_mul`, the KV index hoist. Known extra launches vs llama.cpp: D5's cast after
each `mmq` call, and the copies FlashInfer's norm makes of the strided q/k views after a q+k split (as on the bf16
path today). llama.cpp launches nearly every decode kernel with
PDL on sm_90+ (`common.cuh:1696-1712`), which hides launch gaps ours (FlashInfer, Triton, torch) do not; the review
puts decode at ≈ 0.82–0.90× of llama.cpp with this plan. First task of step 3: measure llama-bench tg128 on the 5080
with `GGML_CUDA_PDL=0` vs default, to size that gap. If a bar is missed after step 4, the fallbacks in order: a
q/k norm reading the strided views, PDL launches for our Triton kernels, fused q/k norm + rope + KV write, revision
2's f32 fusion (decode and prefill), llama.cpp's batch-1 gate+up+swiglu fusion.

### 4.6 Numerics and the judge (all thresholds fixed before measuring)

**Supported types**: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
IQ3_S, IQ4_XS, IQ4_NL, IQ1_S. Excluded: IQ1_M (no `mmq`; llama.cpp falls back to dequant + cuBLAS above 8 rows),
Q1_0/Q2_0 (gguf-py 0.19 cannot dequantize them), MXFP4/NVFP4/TQ.

**q8_1 emulator** (judge code, shared by L0 and Tier B): per 32 values `d = amax/127`, `q = round(x/d)`, with each
path's own semantics — mmvq's `block_q8_1` (half `ds`, the min-term from the quantized sum) and mmq's DS4/D4/D2S6
layouts (the Q4_K/Q5_K min-term from the unquantized sum `ds8.y`, `vecdotq.cuh:521-552`).

**L0** (sm_86, sm_89, sm_120a):
- *Inputs*: real tensors; random-byte blocks for every type with realistic fp16 `d`/`dmin` (negative, zero,
  subnormal, ≤ 1); activation rows all-zero and with one large outlier; `in` ∈ {768, 2304, 4096, 12288} with
  poisoned bytes past the tensor; `out` not a multiple of the mmq tile.
- *Rows* per (type, cc) from `ggml_cuda_should_use_mmvq`: {1, t, t+1, 8, 9, 64, 128, 256, 512, 513, 1025, `max_step_tokens`} (513 and 1025 exercise the 512-row chunk remainder);
  stream-k with and without fixup.
- *Checks*: `qlinear` vs the q8_1-emulated reference of the path taken, per element with `l0.check(bf16, in)` (the
  bf16 budget absorbs the fp16 `d` and fast-math rounding; it catches a single wrong row or block); NMSE ≤ 5e-4 vs
  the unquantized reference with random `x`; `convert`/`get_rows` vs gguf-py: bit-exact for single-product types,
  ≤ 1 fp32 ulp for affine types (FMA under fast-math) unless those units build with `-fmad=false` — decided in step 3
  by measurement, recorded in the test.

**Golden**: the judge builds `Qwen3ForCausalLM` (fp32, CPU, HF config) and fills it with tensors dequantized by
`gguf.quants.dequantize` directly, through its own name map; meta = file sha256, gguf-py version, name-map hash.

**Per-tensor gate** (judged): NMSE(dequantized tensor, HF bf16 tensor) ≤ 0.5 for every tensor — "explains at least
half the energy" (measured: Q6_K ≈ 3.5e-4, Q4_K ≈ 5.3e-3, IQ3 ≈ 2–3.6e-2, IQ2_XS ≈ 0.10; transpose ≈ 2.0, gate/up
swap ≈ 1.9, sign flip ≈ 4). It checks the beliefs both name maps share. KL(dequantized golden ‖ bf16 golden) is
reported, not gated.

**Tiers** (budgets = M0's: KL p50/p99 ≤ 2× floor, top-1 ≥ floor − 0.25 pt, |ΔPPL| ≤ |floor| + 0.25 %):
- **Tier A — assembly**: a judge-owned wrapper source (test fixture `gguf_dense`, injected with an explicit
  `source=`; lock tools refuse to plan it) dequantizes the file to dense bf16 at load, so the engine runs its bf16
  path; floor = HF bf16 with the dequantized weights (the M0 rule). Catches config, names, wiring and rope at M0
  sensitivity. A5000.
- **Tier B — kernels**: the production engine; floor = HF bf16 with the q8_1 emulator on every quantized linear's
  input (lm_head included) — prefill with mmq semantics on the full prompt set; decode with mmvq semantics on the
  full prompt set at a step size ≤ the smallest mmvq threshold among the file's types on that class (≤ 5 on
  Blackwell, ≤ 8 on the A5000). Floors are recorded once on the A5000 GPU (as M0's) and reused on every class; the
  fp32 golden is recorded on the CPU. GGUF-only code paths
  (adopt, padding, merges, lm_head on the embedding's blocks, per-type dispatch) are covered by L0 and Tier B only.
- **Planted bugs, verdicts recorded before measuring**:
  - Tier A must fail: rope θ 1e6 → 9e5 (S5's calibration bug, 3.0× floor KL p99), transposed `attn_output`,
    gate/up swap, Q/K permuted by mistake.
  - Tier B must fail: one tensor dispatched as the wrong type; a stale q8_1 workspace on every 97th `qlinear` call
    (checked to fire in both the prefill and the decode run).
  - L0 must fail (Tier B reported only): the obsolete IQ3_S grid.
  - δ never changes without a written cause, and only if the planted set still fails afterwards.

### 4.7 Memory on a 16 GB card

RTX 5080: 15.47 GiB (15841 MiB). Qwen3-8B KV: 144 KiB/token; 64K tokens = 9.00 GiB.

| GiB | Q4_K_M | UD-IQ3_XXS |
|---|---|---|
| weights | 4.68 | 3.17 |
| context + FlashInfer/ggml workspaces (≈ 0.13 + 0.07) | ≈ 0.65 | ≈ 0.65 |
| KV 64K | 9.00 | 9.00 |
| left for activations + graph pool | ≈ 1.14 | ≈ 2.65 |

The worst-step peak is the largest of trunk, prompt-logprob head (≈ 0.3) and sampling — ≈ 0.4 GiB at 2048 step
tokens, with the mmq output buffer ≤ 48 MiB by the 512-row chunking — plus the 64-row graph pool (≈ 0.25) and
`memory_fraction 0.98`'s ≈ 0.2 held back. Start with `memory_fraction 0.98`, `max_step_tokens 2048`,
`max_seqs 64`; step down only on a measured shortfall (breakdown into the evidence; remedy if still short: quantized
KV, M0.3b). These are deployment settings for the class (recorded with the evidence and in the lock's `[engine]` where
planned fields apply), not code.

### 4.8 Bench

- **Bars**: an in-process Holon llama-bench equivalent (pp512 = one 512-token prefill; tg128 at depth 0 and 256;
  warm-up excluded; host CPU per step recorded; greedy argmax ≈ 10 µs noted) vs llama-bench at the same commit and
  flags (`-use_fast_math`, 120a, PDL), `-ngl 99 -fa on`, plus `-d 256`.
- **Serving**: `llama-server` with `-kvu`, `--parallel` = Holon's `max_seqs`, `-c` = Holon's measured KV tokens,
  `-b` = `max_step_tokens`, `-ub` recorded, f16 KV (noted vs bf16), token-id prompts, greedy, `ignore_eos`,
  `n_predict`; throughput from server-side `timings`; prompt cache default and `-cram 0`; optional second row with
  `--backend-sampling`. Holon through its own HTTP frontend for this comparison.

## 5. Implementation plan

Same three reviewers throughout; each step ends green.

| # | Step | Deliverable / test | Days |
|---|---|---|---|
| 0 | D1: no reference selection; test-only `holonlm-testing` plugin; units stop calling `reference.py` | full suite green via the fixture; missing provider → startup error naming it | 1 |
| 1 | Core D2/D4; `weights/gguf` (reader, detect, config, tokenizer + tests vs HF, identity); `std.ops` adopt + `qlinear`/`qembedding` references | CPU: tiny synthetic Q8_0/Q4_0 GGUF (gguf-py writer) vs dense via the fixture; tokenizer tests; bf16 snapshot byte-identical | 2.5 |
| 2 | Qwen3 `GGUF_WEIGHTS` + bundle(source); CLI `.gguf`; tools/judge resolve; golden + per-tensor gate; Tier A source | Tier A L2 on A5000 for both files; Tier A planted bugs fail | 1.5 |
| 3 | `kernels.ggml_cuda` (parallel with 1–2): PDL measurement on the 5080; link test sm_86/89/120a, which also checks that `vendor.patch` applies; vendor + patch + shim + C API + wrapper + prebuild; q8_1 emulator | L0 (all of §4.6) on sm_86 and sm_120a; graph capture (incl. PDL) on sm_120a; determinism; FMA decision | 6 |
| 4 | Merges through `qlinear`, selected `silu_mul`, KV index hoist; Tier B | Tier B (prefill + decode) on A5000; Tier B and L0 planted bugs fail; bf16 snapshot byte-identical | 3 |
| 5 | Consumer classes: L0 on sm_89; 5080 fit at 64K; bars; `llama-server` backend + traces; Tier B on 5080; tune + GGUF locks; GGUF snapshot ×3 | evidence; locks | 3.5 |
| 6 | bf16 locks re-propagated on a5000/4090/5090 (acceptance 4) | locks | 0.5 |
| 7 | Docs: design §5.1 (self-contained GGUF), §5.5, §5.6, §7.3; plan §4 status | — | 0.5 |

≈ 18 days of work; critical path 0 → 1 → 2 and 3 (parallel) → 4 → 5 → 6 → 7 ≈ 17 days.

## 6. Risks

- Port link, sm_120a, PDL under torch capture — first tasks of step 3.
- Decode parity: ≈ 0.82–0.90× estimated; fallbacks listed in §4.5.
- 5080 fit for Q4_K_M at 64K: ≈ 1.1 GiB headroom.
- `vendor.patch` (≈ 60–80 lines in upstream's most active files) raises refresh cost: keep it mechanical (no new
  templates) and limited to D5 and the environment constants; refresh = vendor.sh + patch + link test + L0.
- Tier B floors: guarded by the planted set; δ fixed.

## 7. Revision-1 questions, answered

1. Core seam: load-time adoption (D2). 2. Oracle: gguf-py (pinned), called directly by the judge and L0; nothing
at runtime dequantizes on the CPU. 3. L2 floor: Tier A (assembly, M0 floor) + Tier B (kernels, q8_1-emulated
floor), fixed in advance. 4. bf16 locks: one approval with a byte-identical snapshot (acceptance 4). 5. Reference
kernels: never (D1).

## 8. Deferred, deliberately

"Repo id + file" resolution; pre-tokenizer table entries beyond the archs served; revision 2's f32 fusion and the
other decode fallbacks of §4.5 (triggered by measurement); CI-built per-SM extension wheels; per-class `src` pins;
the engine starting from its lock (deployment settings are flags until then). The pre-tokenizer table lives in
std, so a new entry changes the `src` of every GGUF lock (acceptable for v0).

Not asserted, an invariant of `kernels.ggml_cuda`: every `qlinear` instance with the same `in` shares one q8_1/fixup
workspace, so no two such calls may be in flight on different streams. The engine runs a model step on one stream
(eager or graph replay), which satisfies it; a second stream (e.g. overlapping two models or steps on the device)
needs a workspace per stream. A per-call stream assert would not detect a violation, so there is none.
