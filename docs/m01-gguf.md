# M0.1 — GGUF quantized models: design and implementation plan

Status: **draft for review** (2026-10-08). Scope: plan §4. Builds on design §5.1 (weight sources), §5.4 (kernel
selection), §5.5 (quantization pipeline), §5.6 (borrowed kernels) and spike S2 (`spikes/s2_gguf/README.md`).

## 1. Goal and acceptance

Serve Qwen3-8B from llama.cpp GGUF files — first Q4_K_M and UD-IQ3_XXS — with llama.cpp's own quantized CUDA
kernels, and take the first 16 GB consumer data point. Acceptance (plan §4):

1. `git diff plugins/models/qwen3` is limited to adding a GGUF weight map to the recipe.
2. L0 for every supported ggml type: kernels vs the reference dequantization.
3. L2 bounded vs a golden recorded from dequantized weights in the dense HF class (CPU fp32); the quantization error
   vs the bf16 model reported separately.
4. Qwen3-8B bf16 outputs unchanged (see §7 for what this means for its locks).
5. RTX 5080: Q4_K_M and UD-IQ3_XXS with ≥ 64K KV tokens; decode tok/s vs `llama-server` on the same card and quant
   recorded. Bars from S2 (≥ 0.8× llama-bench): Q4_K_M ≥ 6380 t/s pp512 and ≥ 127 t/s tg128.
6. An `rtx-5080` lock (carried over from M0 §3.6).

Non-goals: MoE (`mul_mat_id`, M0.2); quantized KV cache (M0.3b); HF quant formats (GPTQ/AWQ/FP8, v1); converters
other than identity (GGUF kernels read the stored blocks); CPU-offloaded experts.

## 2. Starting point

- **S2 decided the kernels**: port llama.cpp's current `mmvq` / `mmq` / `quantize` (and later `mmid`) unmodified
  behind a ~200-line shim (15 ggml-base helpers, 5 ggml-cuda runtime hooks, a bump-allocator pool over our
  workspace). The probe compiled them for sm_86 with nvcc 13.0. S2 also has our own Torch wrapper
  (`csrc/ggml_ext.cu`, all ops write into caller-allocated tensors on the current stream), correctness, graph and
  perf scripts, and llama.cpp builds for sm_86/89/120. Ten types are correct and graph-safe on all classes with
  the SGLang-vendored set; the port must re-establish that.
- **API types exist but are unused**: `Scheme`, `Layout`, `QTensor`, `SchemeImpl`, `Converter`, `Dialect`
  (`holon.api.quant`, marked provisional until M0.1).
- **Weight loading today** builds the model on the meta device, materializes every parameter densely in the build
  dtype, then copies checkpoint tensors in by shape (`core/holon/core/build.py`). That cannot work for GGUF: the
  dense bf16 materialization of an 8B model alone is 16 GB.
- **Qwen3 GGUF files** (unsloth, cached): tensor names `blk.{i}.attn_q.weight` etc.; no Q/K permutation (llama.cpp's
  Qwen2/Qwen3 converters keep the NEOX order); norms are F32; per-tensor types vary within one file (Q4_K_M: Q4_K,
  attn_v / ffn_down half Q6_K, output Q6_K, token_embd Q4_K; UD-IQ3_XXS: IQ3_XXS, IQ2_S, IQ2_XS, IQ3_S, IQ4_XS,
  Q3_K, Q4_K, Q5_K). The file carries `qwen3.*` hyperparameters and a full tokenizer, but **no** base-model id.

## 3. Packages

| package | holds | why there |
|---|---|---|
| `std/holon_std/weights/gguf` (`weight_source.gguf`) | memory-mapped reader, config from metadata, tokenizer check, identity | a checkpoint format, like `weights/hf` |
| `std/holon_std/quant/ggml` (`dialect.ggml`, `scheme.ggml.*`) | ggml types as `Scheme`/`Layout`, `dequant_ref` per type, reference kernels | the number semantics every GGUF kernel and the judge agree on; format code stays out of core (invariant 1) |
| `std/holon_std/ops` (existing) | `Linear`, `Embedding`, `LMHead` learn to adopt a source layout; new ops `linear@1`, `embedding@1` | the units the recipes already use |
| `plugins/holon_plugins/kernels/ggml_cuda` | vendored llama.cpp ggml-cuda subset + shim + Torch extension; `kernel.linear@1`, `kernel.embedding@1` for `gguf.native` layouts | a borrowed kernel family (§5.6), optional at install |
| `plugins/holon_plugins/models/qwen3` | `GGUF_WEIGHTS` map | acceptance 1 |
| `judge` | GGUF golden recorder, L0 for ggml types | the oracle side stays independent of the engine's loader |
| `bench/backends/llamacpp.py` | `llama-server` HTTP backend | acceptance 5 |

## 4. Design

### 4.1 Layouts: what a ggml tensor is

One `Scheme` per ggml type and one layout name for the stored blocks:

```python
Scheme(elem="ggml.q4_k")                         # the type's meaning lives in its SchemeImpl (dequant_ref)
Layout(Scheme("ggml.q4_k"), "gguf.native")       # blocks as stored: [rows, row_bytes] uint8
QTensor(layout, logical_shape=(out, in), parts={"blocks": uint8[out, in // block_elems * block_bytes]})
```

- `Layout.tensors(logical_shape)` (already sketched in design §5.5) gives the parts and their byte shapes from the
  type's block size and type size, so a unit can allocate storage for a layout before any data is read.
- Dense GGUF tensors (F32, F16, BF16) are not layouts: they load as ordinary tensors and are cast to the build
  dtype like an HF checkpoint's.
- `dequant_ref` per type is gguf-py's numpy implementation (`gguf.quants.dequantize`, pinned 0.19.0), wrapped to
  torch. gguf-py follows llama.cpp's own reference (`ggml-quants.c`), and S2 scored every kernel against it. (The
  SGLang-vendored k-quant dequant computed in fp16 and was not bit-exact with it; current llama.cpp dequantizes in
  fp32, so the ported `convert` kernels should be — L0 checks.) Using the pinned library rather than a copy keeps
  the oracle out of our hands (design §10.3 oracle provenance); the judge pins it.

### 4.2 Weight source `gguf`

`GGUFSource(path)` where `path` is a `.gguf` file (or an HF repo id + file name, resolved through the hub cache).
`EngineConfig.source` gains `"auto"` (the new default): `.gguf` → `gguf`, a directory with `config.json` →
`hf_safetensors`.

- **Config** — an HF-style dict built from the file's arch-prefixed keys, which every llama.cpp text architecture
  shares: `block_count → num_hidden_layers`, `embedding_length → hidden_size`, `feed_forward_length →
  intermediate_size`, `attention.head_count{,_kv}`, `attention.key_length → head_dim`, `rope.freq_base →
  rope_theta`, `attention.layer_norm_rms_epsilon → rms_norm_eps`, `context_length → max_position_embeddings`;
  `vocab_size` from the token table; `tie_word_embeddings` = no `output.weight`; `architectures` from
  `general.architecture` through a small table (`qwen3 → Qwen3ForCausalLM`). If an HF base is given
  (`EngineConfig.base`, a hub id or directory), its `config.json` is loaded and **cross-checked** field by field;
  a mismatch is an error, and fields GGUF lacks come from it (design §5.1). M0.2/M0.3 extend the key table (MoE,
  GDN). Keys the model needs that neither source has are an error naming the key.
- **Tokenizer** — design §5.1 says "the HF base model's tokenizer, checked against the GGUF token table". Since
  these files name no base, the bundle needs one: the model plugin's bundle receives the source and builds the
  tokenizer from `EngineConfig.base` when given, else from the GGUF file itself (transformers' GGUF tokenizer
  loader); in both cases the hash of (tokens, token types, special ids) must match the GGUF table. The GGUF
  `tokenizer.chat_template` takes precedence over the base's (differences logged).
- **Tensors** — `tensors()` yields `(name, tensor)` where quantized tensors are `QTensor`s over a numpy memmap of
  the file (no copy until the device upload), and `layouts()` returns `{name: Layout}` for the quantized ones
  without reading data.
- **Identity** — `<basename>-<file stem>@<16 hex of the file's sha256>`; the digest cache already used for HF
  checkpoints (user cache, keyed by inode/size/mtime/ctime) makes restarts free. Every quant file gets its own
  `model_id` and lock history.

### 4.3 Core change: layouts known before materialization

The only core change, kept format-agnostic (no ggml names in core):

1. `WeightSource` gains an optional `layouts() -> {checkpoint name: Layout}` (absent = all dense).
2. Before `to_empty`, the core maps those names to parameter paths with the model's `WeightMap` (the same
   `map_name` the loader uses) and calls, on the module owning each such parameter,
   `module.adopt(param_name, layout, logical_shape)`.
3. A unit that supports the layout replaces the meta parameter with meta storage of the layout's part shapes
   (`Layout.tensors`) and selects its kernel for the layout (`ctx.select_kernel("linear@1", layout=…, …)`); it
   returns `False` otherwise, and the core fails with "`<path>`: unit `<ref>` cannot read layout `<layout>`".
4. `to_empty` then allocates exactly the stored bytes, and `load_weights` copies parts by shape as today (a
   `QTensor` source copies part by part; a dense source tensor of another dtype is cast by `copy_`).

Units that never see quantized weights need no change. `Linear.merge` returns `None` for adopted (non-dense)
weights, so q/k/v and gate/up stay separate GEMMs for GGUF (llama.cpp does the same; types differ within these
groups in mixed files anyway).

### 4.4 Ops and units

- `std.ops` declares two ops with references in `reference.py`:
  - `linear@1(x, w: QTensor) -> y`: reference = `x @ dequant_ref(w).T` in fp32, cast to `x.dtype`.
  - `embedding@1(ids, w: QTensor) -> x`: reference = dequantize the selected rows.
  These references are also registered as the lowest-priority kernels for any layout whose scheme has a
  `SchemeImpl`, so a GGUF model runs (slowly) on any device — CPU tests included — before the CUDA port exists.
- `Linear.adopt`, `Embedding.adopt`, `LMHead.adopt` (untied `output.weight`; a tied quantized head reuses the
  embedding's blocks through `linear@1`). Facts for selection: `layout` (scheme elem + name), `in`, `out`,
  `act_dtype`. Bias stays dense.

### 4.5 Kernel plugin `kernels.ggml_cuda`

- **Source**: llama.cpp ggml-cuda files at a pinned commit (S2 used `51ce9c11`) under `third_party/` with
  `vendor.sh` (copy + sha256 list) and `vendor.patch` (expected empty), MIT license and provenance next to them,
  as for the GDN vendoring (design §5.6). Files: `mmvq`, `mmq` + its per-type instances, `quantize`, `getrows`,
  `convert` (dequantize), `common.cuh`/`vecdotq.cuh`/`mma.cuh`/`ggml-common.h`. `mmid` waits for M0.2.
- **Shim** (`csrc/shim.cu`, ~200 lines): the 15 ggml-base helpers, the 5 ggml-cuda runtime hooks with the true
  device (cc, SM count — llama.cpp's kernel choices depend on them), and `new_pool_for_device` as a bump allocator
  over a workspace tensor we pass per call.
- **Wrapper** (`csrc/ops.cu`): `linear(x, blocks, type, y, workspace)` picks `mmvq` or `mmq` with llama.cpp's own
  `ggml_cuda_should_use_mmvq/mmq` for the call's row count — on the host, from shapes only, so a captured graph
  replays a fixed choice per batch bucket; `embedding(ids, blocks, type, out)` = `get_rows`. Activations bf16
  (fp16 compiled, untested); q8_1 activation quantization into the workspace inside the call.
- **Workspace**: one `ctx.workspace("ggml_cuda", bytes)` sized at build for the largest call (`max_step_tokens`
  rows × the widest `in`, plus mmq's stream-k fixup `nsm × tile` floats), shared by every linear site (calls are
  sequential on one stream).
- **Build**: `torch.utils.cpp_extension.load` with sources hashed into the build directory name, for the device's
  SM, cached under the user cache; `python -m holon_plugins.kernels.ggml_cuda.prebuild` builds ahead of time
  (≈ 65 s per SM, S2) so the first engine start does not (design §5.6 rule 7). Missing nvcc → the provider is
  unavailable and the reference kernel serves (slow, correct) with a warning.
- **KernelSpec props**: `graph_safe` (S2 checked capture); `deterministic` only after L1 confirms run-to-run
  identity of the stream-k fixup on each class — until then `mmq` sites are recorded `uncovered` in locks.

### 4.6 Numerics and the judge

- **L0 per type** (judge + plugin tests): `linear` kernel vs `x @ dequant_ref(w).T` (fp32) on real tensors from
  the cached files and on synthetic blocks for types absent from them; rows 1, 2–8, 9–64, ≥ 256 (each kernel
  path); sm_86 locally, sm_89/sm_120 on the consumer desktops. A single loose gate is not enough: S2's IQ3_S bug
  (an obsolete grid, 4.3 % of weights wrong) stayed under llama.cpp's NMSE 5e-4. So L0 has three parts: dequant
  (`convert`) and `embedding` **bit-exact** with `dequant_ref`; per-path NMSE budgets set from S2's measurements
  with 2× margin (mmvq ≈ 3e-5 → 6e-5; Q4_K/Q5_K `mmq` ≈ 1.6e-4 → 3.2e-4, whose min-term uses unquantized
  activation sums by design); and the 5e-4 ceiling.
- **L2 golden**: the judge recorder builds `Qwen3ForCausalLM` (fp32, CPU) and fills it with the GGUF tensors
  dequantized by gguf-py, through **its own** name map (judge code, not the engine's `WeightMap`), then records
  as for M0. One golden per quant file.
- **L2 floor** — the open numerics question. The kernels quantize activations to int8 (q8_1) for every matmul,
  which S2 measured at NMSE ≈ 3e-5 per op, 5× bf16 GEMM's. A floor of "HF bf16 with the same dequantized weights"
  (the M0 rule) may be too tight for a kernel family that is correct by llama.cpp's standard. Proposal: record
  that floor and run L2; if the budget fails only because of activation quantization (L0 green, KL excess spread
  evenly over positions), calibrate a `ggml` numerics class whose floor is the same HF model with activations
  rounded through q8_1 in each linear (a judge-side reference op, ~30 lines, exact q8_1 semantics from gguf-py),
  rather than relaxing δ by hand.
- **Quantization error vs bf16** — the dequantized golden vs the existing bf16 golden, reported (KL, top-1, ΔPPL)
  and not judged.

### 4.7 Memory on a 16 GB card

RTX 5080: 15.9 GiB usable. Qwen3-8B KV in bf16 is 144 KiB per token, so 64K tokens = 9.00 GiB. Weights: Q4_K_M
4.68 GiB, UD-IQ3_XXS 3.17 GiB. Left for CUDA context (~0.5 GiB), activations, workspace and graphs: ~1.7 GiB
(Q4_K_M) and ~3.2 GiB (IQ3_XXS). Measures:
- weights stay quantized everywhere (embedding through `get_rows`, `lm_head` through `linear`); no dequantized
  copies;
- the worst-case step is measured as today; for the 16 GB class the default `max_step_tokens` is lowered to 2048
  if the measured peak needs it (a deployment setting in the lock's `[engine]`, not a code path);
- if Q4_K_M still falls short of 64K, the shortfall and its breakdown go into the evidence, and quantized KV (M0.3b)
  is the remedy — acceptance 5 is then met with IQ3_XXS and recorded as partial for Q4_K_M.

### 4.8 Bench

`bench/backends/llamacpp.py`: an HTTP backend (the protocol was built for it) driving `llama-server` with the
same file, `-ngl 99 -fa on`, `--parallel` = max batch, context = the trace's need; streamed tokens timed client-side
like the other backends. Workloads: decode_b1/b8/b64 and the agent trace, ABAB, on the RTX 5080 (and A5000 for a
second point). llama.cpp is built from the same pinned commit as the vendored kernels, so differences are ours.

## 5. Implementation plan

One working branch; the same reviewers for the whole milestone; each step ends green (lint, tests).

| # | Step | Deliverable / test | Days |
|---|---|---|---|
| 1 | `quant/ggml`: Schemes, `gguf.native` layout with `tensors()`, `dequant_ref` via gguf-py for all types it supports | unit tests: part shapes, dequant vs gguf-py on real tensors | 0.5 |
| 2 | `weights/gguf`: reader, config mapping + cross-check, identity, `layouts()`, `source="auto"` | tests on the cached files: config equals HF config.json; tiny synthetic GGUF (gguf-py writer) | 1 |
| 3 | Core: `adopt` pass before materialization; `Linear`/`Embedding`/`LMHead.adopt`; `linear@1`/`embedding@1` ops with reference kernels | CPU test: a tiny synthetic Q8_0/Q4_0 GGUF model end to end vs dequantized dense; bf16 path unchanged (snapshot identical) | 1.5 |
| 4 | Qwen3 `GGUF_WEIGHTS`; bundle tokenizer from base or GGUF + hash check; judge GGUF golden recorder | Qwen3-8B Q4_K_M runs with reference kernels on GPU (slow); golden recorded for Q4_K_M and UD-IQ3_XXS | 1.5 |
| 5 | `kernels.ggml_cuda`: vendor + shim + wrapper + build/prebuild; `KernelSpec`s with workspace | L0 all types × row ranges on sm_86; graph capture; run-to-run determinism check | 2.5 |
| 6 | L2 on Q4_K_M and UD-IQ3_XXS (floor question of §4.6 settled here) | `judge/results` + evidence | 1 |
| 7 | 5080: memory fit at 64K, `llama-server` backend, decode/agent bench, `tune` + `rtx-5080` lock; 4090/5090 L0 | evidence files; lock | 1.5 |
| 8 | Docs: design §5.5 updated from this document, plan §4 status | — | 0.5 |

About 10 days. Order matters at two points: step 3 before 5 (the reference path is the oracle and the fallback),
and step 6 before 7 (no performance evidence for an unverified numeric path).

## 6. Risks

- **Port drift** — llama.cpp moves fast; we pin one commit and refresh deliberately (re-run the probe + L0).
- **16 GB fit** (§4.7) — the margin for Q4_K_M at 64K is ~1.7 GiB before activations.
- **Stream-k determinism** — if `mmq`'s fixup is order-dependent, those sites stay `uncovered` and the GGUF model's
  locks can never auto-advance (acceptable for M0.1; affects later propagation).
- **Graph capture of host-chosen kernels** — the mmvq/mmq choice is host-side per call shape; captured buckets
  must not straddle a threshold differently at replay (they cannot: the bucket's row count is fixed).
- **Tokenizer without a base** — building it from the GGUF table must give byte-identical encodings to HF's; the
  hash check guards the table, a round-trip test on the judge prompts guards the encoder.

## 7. Questions for review

1. Is the `adopt` pass the right core seam, or should layouts flow through `BuildCtx` at unit construction?
2. Should `dequant_ref` be gguf-py (pinned dependency) or our own port of `ggml-quants.c`?
3. L2 floor for int8-activation kernels (§4.6): HF bf16 vs a q8_1-activation reference — or llama.cpp's own logits?
4. Acceptance 4 ("bf16 locks unchanged"): touching `std.ops` changes the `src` pins in the bf16 locks, so propagate
   will ask for an approval even with a byte-identical snapshot. Is "snapshot byte-identical, approved" the right
   reading, or must std changes for one format avoid touching units another model's lock pins?
5. Reference kernels as lowest-priority fallbacks for quantized layouts: useful safety net, or a trap (a missing
   nvcc silently gives a 50× slower engine)?
