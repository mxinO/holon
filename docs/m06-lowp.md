# M0.6 — FP8 (block), NVFP4 and MXFP4 checkpoints of Qwen3-8B and Qwen3-30B-A3B: design and implementation plan

Status: **revision 3, reviewed clean** (2026-10-13; revisions 1–3 reviewed by the three-reviewer panel, every finding
mapped in §8; user decisions D-a…D-h and the rev 2 rulings applied). Scope: plan §2 row M0.6. Builds on design §5.5
(Scheme / Layout / QTensor), §5.6 (borrowed kernels), M0.1 (`kernels.ggml_cuda`, the judge's tiers, the q8_1 emulator),
M0.2 (`std.moe`, `linear_experts@1`, the stacking rule D7, `kernels.triton_experts` D8), M0.4 (draft heads load through
`Linear.adopt`). Research behind it: the M0.6 research note (local, not tracked), re-checked against source where cited.

## 1. Goal and acceptance

Serve low-precision checkpoints of Qwen3-8B (dense) and Qwen3-30B-A3B (MoE) — FP8 with 128×128 block scales, NVFP4,
MXFP4 — on consumer cards first (RTX 5080 / 5090 sm_120, RTX 4090 sm_89; RTX 3090 / A5000 sm_86 for FP4), with W8 /
W4 weights and **8-bit activations by default on every class** (D3). Two gates, each closed by the panel.

**Gate A — the ggml pin, the pure refactor, FP4 and FP8 loading.**

1. **Pin bump alone, first** (D7): `kernels.ggml_cuda` moves from llama.cpp `51ce9c11` to `cbe28bb3`; `vendor.sh
   --check` passes; L0 for the 18 existing types on sm_86 / sm_89 / sm_120a, memcheck clean; every GGUF lock advances
   by its own `propagate` (one approval), L2 passing; spot check Qwen3-8B Q4_K_M pp512 / tg128 on the RTX 5080 within
   ±3% of M0.1's.
2. **Pure refactor** (D2, D4, D5, D11): the core's unused `Dialect` / `Converter` / `SchemeImpl` deleted; the per-slice
   `adopt` seam (≈ 8 lines) with its toy proof in `core/tests/test_stacking.py`; `qlinear@2`, `prepare`, `act_quant` in
   std. `propagate` leaves **every existing snapshot byte-identical** (bf16, GGUF, M0.4's draft heads), L2 passing;
   `git diff plugins/holon_plugins/models` empty; denylist clean; core ≤ 6000 code lines (3643 after the M0.4 rebase, 3623 after A1).
3. **Targets load and answer the M0.1 chat set** through `models/qwen3` / `models/qwen3_moe` unchanged (D1 roles):
   `nvidia/Qwen3-8B-NVFP4`, `nvidia/Qwen3-30B-A3B-NVFP4`, `amd-quark/Qwen3-30B-A3B-MXFP4`, the GGUF
   `Qwen3-30B-A3B-Instruct-2507-MXFP4_MOE`; the fixtures `RedHatAI/Qwen3-8B-NVFP4` and `Yi30/Qwen3-8B-MXFP4`;
   `Qwen/Qwen3-8B-FP8` and `Qwen/Qwen3-30B-A3B-FP8` load through the dialect and the reference provider (CPU). A GGUF
   holding NVFP4 tensors is refused at load with a message naming the reason (D1).
4. **Lossless repack:** for every quantized tensor of the five HF FP4 files, gguf-py's dequantization of our ggml
   blocks × `scale2` equals `reference.py`'s by value in fp32; the two 8B NVFP4 files' blocks are byte-equal to the
   pin's converter output.
5. **L0** (`kernels.ggml_cuda`, sm_86 / sm_89 / sm_120a, memcheck clean): every spec × path × input dtype (bf16, f32)
   against its declared mode's emulator (§4.8); reference dequantization equals the format owners' decoders (D10).
6. **L2.** Tier A (CPU fp32) for every target and fixture, under the 1e-4 sanity ceiling; the 30B FP8 structural test.
   Tier B at the default `a8` (W4A8, both paths) against the W4A8 floor: 8B NVFP4 (A5000, 5080), 30B NVFP4 and
   `amd-quark` 30B MXFP4 (A5000, RTX PRO 6000), the GGUF MXFP4_MOE (A5000, 4090). Plants as predicted in §4.8.
   Per-tensor gates (§4.8): FP8 NMSE ≤ 1e-2, FP4 ≤ 0.1. Quantization error vs the bf16 checkpoint reported per
   format and mode.
7. **W4A4 report** (opt-in `quant.act = a4`, sm_120): 8B NVFP4 and 8B MXFP4 scored against the W4A4-dynamic floor;
   the near-threshold plant N9 passes as predicted (the documented blind spot); graphs vs eager bounded on the 5080.
   **Pre-registered:** `a4` may become a default only if, on the real 8B files, the W4A4 floor's KL p99 ≤ 1.5× the
   W4A8 floor's and its top-1 within 0.5 points — then escalated to the user. Predicted: not met (≈ 45×, §4.8).
8. **Memory** (§4.9), bf16 KV, `max_seqs` 32, `memory_fraction` 0.9: RTX 5080 8B NVFP4 ≥ 32K KV tokens; RTX 4090
   30B NVFP4 and the GGUF MXFP4_MOE ≥ 32K. Fallbacks named per row (0.98, or q8_0 KV); `amd-quark` (no lock) recorded.
9. **Speed** (D9), medians of repetitions 2–5: pp512 and tg128 ≥ 0.8× llama-bench on the same bytes and the same
   activation mode (llama.cpp forced to Q8 activations), for 8B NVFP4 (5080, 4090) and the 30B NVFP4 and GGUF
   MXFP4_MOE (4090); 5090 recorded; llama.cpp's default (W4A4 on sm_120) reported beside; serving vs `llama-server`
   on the 5080 for the 8B NVFP4.
10. **Locks:** rtx-5080 8B NVFP4 (`nvidia`); rtx-4090 8B NVFP4, 30B NVFP4, 30B MXFP4_MOE (GGUF). Snapshots
    run-identical ×3; graphs vs eager bounded on the 30B NVFP4 decode (4090).

**Gate B — FP8 kernels.**

11. **Decode gate first:** per-shape microbenchmark (the 8B's 5 projection shapes, M = 1…16) of the W8A16 split-K
    decode kernel on the 5080 and 4090, turned into an end-to-end prediction: Σ (shape time × 36 layers) + the
    measured bf16 head time + the non-GEMM decode time from the existing GGUF decode profile on that card. It must
    predict ≥ 0.85× the llama-bench Q8_0 tg128 measured in the same step, or the kernel is reworked before B2.
12. **L0** for `kernels.triton_fp8` FP8 on sm_89 and sm_120a (dense, both path modes), memcheck.
13. **L2 Tier B**, 8B FP8 on rtx-5080 and rtx-4090: prefill W8A8 vs the W8A8 floor, decode W8A16 vs the A16 floor;
    FP8 plants as predicted. Decode runs use steps of ≤ 16 tokens (the M ≤ 16 switch), so every call is W8A16.
14. **Speed:** 8B FP8 tg128 ≥ 0.8× llama-bench Q8_0 of the same checkpoint on the 5080 and 4090; pp512 recorded
    against its prediction. **Memory:** 8B FP8 on the 5080 ≥ 32K KV tokens at 0.9 (34.3K predicted; fallback 0.98).
15. **Locks:** 8B FP8 on rtx-5080 and rtx-4090, snapshots ×3.

Non-goals: FP8 on sm_86 (plan §10, v1; if the W8A16 kernel happens to run there it is noted, nothing more);
checkpoint-faithful static-scale W4A4 providers; grouped FP8 kernels (the 30B FP8 stays on the reference provider);
`quant.act = a16` (a startup error); GGUF NVFP4; ModelOpt per-tensor FP8 and `MIXED_PRECISION`; A16 FP4
kernels; gpt-oss; FP8 KV from checkpoint scales; Qwen3.5/3.8 low-precision builds.

## 2. Starting point

- **Checkpoints** (research note §1, headers read): every quantized Linear is `[out, in]`, all dimensions multiples of
  128; experts are per-expert 2-D tensors (`mlp.experts.{e}.{gate,up,down}_proj.*`); embeddings, `lm_head` and
  routers are dense (bf16, or F32 in `Qwen/Qwen3-30B-A3B-FP8`). Parts per module:
  - FP8 (`quant_method: fp8`): `weight` F8_E4M3 `[out, in]`, `weight_scale_inv` BF16 or F32 `[out/128, in/128]` (a
    dequant multiplier despite its name); activations declared dynamic.
  - NVFP4: ModelOpt `weight` U8 `[out, in/2]`, `weight_scale` F8_E4M3 `[out, in/16]`, `weight_scale_2` F32 `[]`,
    `input_scale` F32 `[]` (plus `k_scale`/`v_scale` = 1.0); compressed-tensors `weight_packed`, `weight_scale` and the
    **reciprocals** `weight_global_scale`, `input_global_scale` `[1]`.
  - MXFP4: compressed-tensors `weight_packed`, `weight_scale` U8 (e8m0) `[out, in/32]`; Quark `weight`,
    `weight_scale`, same shapes (`amd-quark` 30B: experts only).
  - ModelOpt 8B ties `weight_scale_2` across q/k/v and across gate/up; the 30B's experts carry one `weight_scale_2`
    per expert and projection (gate/up equal per expert in the files read, not guaranteed).
- **Holon today:**
  - `qlinear@1` takes one `blocks` tensor (`std/holon_std/ops/kernels.py:8`); `Linear.adopt` keeps
    `q.parts["blocks"]` (`std/holon_std/ops/units.py:90-95`); `ExpertLinear.adopt` refuses parts other than
    `{"blocks"}` (`std/holon_std/moe/units.py:42-48`); `linear_experts@1` takes `parts` by role
    (`std/holon_std/moe/kernels.py:8-14`).
  - A quantized tensor under a stacking rule is a load error (`core/holon/core/build.py:219-220`): M0.2 D7's open
    question.
  - `std.weights.hf` yields tensors one by one (`std/holon_std/weights/hf/source.py:46-50`). The core's `Dialect`,
    `Converter`, `SchemeImpl` (`core/holon/api/quant.py:55-79`) and their registries (`core/holon/api/ops.py:76-78`,
    `core/holon/core/plugins.py:66-79`) are used by nothing.
  - `kernels.ggml_cuda` (paths below relative to `plugins/holon_plugins/kernels/ggml_cuda/`) serves 18 types
    (`plugin.py:33-35`); the vendored tree has `GGML_TYPE_MXFP4 = 39`, `GGML_TYPE_NVFP4 = 40`
    (`vendor/ggml/include/ggml.h:429-430`), their blocks (`vendor/ggml/src/ggml-common.h:214-226`) and mmq / mmvq
    instances, built for sm_120a (`extension.py:25`).
- **The vendored ggml for FP4** (`vendor/ggml/src/ggml-cuda/`):
  - mmq's activation precision comes from `op_params[3]` (`mmq.cu:118-133`; env override nulled at `:96`); our shim
    zeroes `op_params` (`csrc/api.cu:58-71`) = `GGML_PREC_UNDEFINED` = **native W4A4 on Blackwell** for FP4
    (`mmq.cu:128-132`).
  - Native W4A4 quantizes from f32 only (`quantize.cuh:29`) and our patch asserts it (`vendor.patch:46` →
    `mmq.cu:184`); it allocates a per-row activation scale for NVFP4 (`mmq.cu:192-195`) that our workspace formula
    does not size (`csrc/api.cu:138-145`); its K loop steps 512 with no tail, so `in = 768` (30B down) reads past the
    last row — llama.cpp pads every tensor allocation by one row's tail to 512 elements and zeroes it
    (`ggml-cuda.cu:765, 914-916` at `cbe28bb3`). The plugin docstring's "no kernel reads past the blocks"
    (`plugin.py:8-9`) is wrong for FP4.
  - mmvq is always q8_1 (`mmvq.cu:49-50`) and applies NVFP4 per-tensor / per-expert scales in its fused path
    (`x_scale`, `gate_scale`; `mmvq.cu:669-675, 702-709, 1506-1516`).
  - Upstream fixed MMQ out-of-bounds reads (`fc9ce6b9d`) and host configs ignoring the src1 precision (`8a1a9b512`,
    the FP4-at-Q8 path on Blackwell) after our pin.
- **Triton:** `fp8e4nv` is rejected below sm_89 (design §5.6); `kernels.triton_experts` is our bf16 grouped GEMM
  (`plugins/holon_plugins/kernels/triton_experts/kernel.py:34-83`).

## 3. Decisions

**How references are chosen** (M0.2's rule): oracle = the format owner's decoding; kernel source = the best maintained
implementation of the workload; bar = the engine users run for it on the card.

- **D1 — Targets and roles** (D-g).

  | file | format, dialect | role |
  |---|---|---|
  | `nvidia/Qwen3-8B-NVFP4` (5.96 GiB) | NVFP4, ModelOpt | Tier A + B, W4A4 report, bars, locks 5080 + 4090 |
  | `nvidia/Qwen3-30B-A3B-NVFP4` (16.85) | NVFP4, ModelOpt | Tier A + B, bars, lock 4090 |
  | `Qwen3-30B-A3B-Instruct-2507-MXFP4_MOE` GGUF (experts MXFP4, rest Q8_0; e.g. `Face314/…-MXFP4_MOE`, fixed in step A0) | ggml | Tier A + B, bars (same file), lock 4090 |
  | `amd-quark/Qwen3-30B-A3B-MXFP4` (17.22) | MXFP4, Quark | Tier A + B, no lock |
  | `RedHatAI/Qwen3-8B-NVFP4` | NVFP4, compressed-tensors | fixture: Tier A only |
  | `Yi30/Qwen3-8B-MXFP4` (5.76; community test upload) | MXFP4, compressed-tensors | fixture: Tier A + W4A4 report only |
  | `Qwen/Qwen3-8B-FP8` (8.79) | FP8, `fp8` | Gate A: Tier A; Gate B: Tier B, bars, locks 5080 + 4090 |
  | `Qwen/Qwen3-30B-A3B-FP8` (30.22 on disk, ≈ 29.1 loaded) | FP8, `fp8` | structural + Tier A on the reference provider; fit computed |

  No MXFP4 Qwen3 comes from Qwen, NVIDIA or OpenAI; AMD's official MXFP4 line targets Qwen3.5/3.8 (§7). GGUF NVFP4
  files (per-tensor scales as separate `<name>.scale` tensors) are refused by `std.weights.gguf` with that reason.
- **D2 — Dialects inside the weight source, one module each** (`std/holon_std/weights/hf/quant/{fp8,compressed_tensors,
  modelopt,quark}.py`). `std.weights.hf` reads `quantization_config` (`config.json`, else `hf_quant_config.json`, which
  then joins the identity hash), picks the dialect by `quant_method` / `format` (anything else: a load error naming
  it), reads every shard header first and yields **one `QTensor` per quantized module under `<module>.weight`**, so
  the recipes' rules map it unchanged. Normalized parts:

  | scheme `elem` | parts | normalization |
  |---|---|---|
  | `fp8_e4m3` | `weight` u8 (e4m3 bits) `[out, in]`, `scale` f32 `[out/128, in/128]` | BF16 scales upcast (exact) |
  | `nvfp4` | `weight` u8 `[out, in/2]`, `scale` u8 (e4m3 bits) `[out, in/16]`, `scale2` f32 `[]` | compressed-tensors' global inverted, `1/x` in fp32 as llama.cpp's converter (`conversion/base.py:1186-1209` at `cbe28bb3`): ≤ 1 ulp from CT's division |
  | `mxfp4` | `weight` u8 `[out, in/2]`, `scale` u8 `[out, in/32]` | none |

  Layout name `hf`. Dropped at load: input scales (no provider reads them; the judge reads them from the checkpoint) and
  ModelOpt `k_scale`/`v_scale` (must be 1.0, else a load error). Load-time asserts: no NVFP4 scale byte has its sign bit
  set or is 0x7F / 0xFF; no MXFP4 e8m0 byte is 0xFF (NaN); no FP8 weight byte is 0x7F / 0xFF. **Extraction trigger**
  (recorded): the dialects move to their own plugins, with a core seam to reach them, when a second weight source
  needs them or when dialect churn pushes `std.weights.hf` approvals toward plan §9's > 3 / week.
- **D3 — Activation precision: `a8` by default everywhere; `a4` opt-in** (D-b). A fact `act_quant` ∈ {`a4`, `a8`,
  `a16`} on `qlinear@2` and `linear_experts@2` sites: the lowest activation precision the kernel may use. Default `a8`
  for every quantized layout (GGUF included), `a16` for dense; the plugin option `quant.act` (`a4` | `a8`, read as
  `kv.format` is, `std/holon_std/mixers/gqa/units.py:73-79`) sets it explicitly, lowering included; `a16` for a
  quantized layout is not offered in M0.6 (a startup error naming the option). A provider
  registers **one `KernelSpec` per mode**, and `config["act"]` maps each of its paths to the mode it computes (e.g.
  ggml default `{"mmvq": "q8_1", "mmq": "q8_1"}`, opt-in `{"mmvq": "q8_1", "mmq": "fp4"}`), recorded in the lock's
  site entry (`core/holon/core/build.py:67-68`); the judge picks floors from it. sm_120 kernels are scored at `a8`
  against the W4A8 floor; W4A4 is a report (acceptance 7).
- **D4 — Contracts.** `qlinear@1` → **`qlinear@2`**: instance `(x [T, in], parts) → y [T, out]`, `parts` by role as
  `linear_experts`' (breaker pays: `kernels.ggml_cuda`, `testing.ref_kernels` move in the same change). As built,
  `linear_experts@1` → **`@2`**: a stack's parts are per expert, `[E, …]` (ggml: `[E, out, row_bytes]`, a view of the
  stored bytes; D6). The first part is the primary one, held by the unit's `weight` (GGUF: `blocks`; A2's dialects
  yield `weight` first, and A2's tests check it). Both ops gain an optional instance method **`prepare(q: QTensor) →
  QTensor`** (D-d; `qembedding@1` has none): called once by the unit after binding; returns
  the destination layout (e.g. `Layout(Scheme("ggml.nvfp4"), "gguf.native")` with parts `{blocks, scale2}`); must not
  mutate its input (a tied head reads the embedding's parts); lossless (acceptance 4). The unit swaps `weight` to the
  prepared primary part and drops the source parts, so one tensor or stack is transient. M0.4's draft-head linears
  load through the same `Linear.adopt` and are covered.
- **D5 — Merges by shape.** `Linear.merge` / `ExpertLinear.merge` concatenate every part whose leading (per expert:
  second) dimension is `out / g` for one `g` dividing every unit's `out`. Dense: `scale2` must be bit-equal, else the
  units stay separate (`std/holon_std/ffn/units.py:23-30`, `std/holon_std/mixers/gqa/units.py:146-147`). MoE gate/up:
  `scale2` is carried as `[2, E]` (gate row, up row), so each half is contiguous as mmvq's `gate_scale` / `x_scale`
  indexing requires (`mmvq.cu:705-708, 1506-1516`); our SwiGLU takes both rows. A part of rank ≤ 1 with one element
  in every unit (`[]` or `[1]`; per expert `[E]` or `[E, 1]`) is per tensor: bit-equal for dense merges, stacked
  `[2, E]` for experts, never concatenated as rows. Per-row parts are 2-D (per expert 3-D): an FP8 128 × 128 unit's
  `[1, 1]` block scale is a row.
- **D6 — Per-expert stacking: per-slice `adopt`** (D-e, closes M0.2 D7). In `_fill_slice`, after the range and
  duplicate checks, a `QTensor` with a transform is an error, else `adopt(ctx, leaf, q, index=index)`; units hold every
  stacked part per expert, `[E, …]`, whole GGUF stacks included (a view of their rows), as `linear_experts@2` takes
  them (`kernels.ggml_cuda` reads `[E, out, row_bytes]` as llama.cpp's `[E·out, row_bytes]`); units without stacks
  refuse a slice (the "cannot read layout" error). Rejected: the source grouping
  slices — it would guess the stack from names the recipe already declares (`stack="e"`) and hold it in host memory.
- **D7 — The ggml pin moves to `cbe28bb3` as its own first step** (D-a): contains `fc9ce6b9d` and `8a1a9b512`; the
  same commit is the llama-bench baseline. Our C API follows it: `ggml_cuda_mmq_get_config` gains a fifth argument
  (the src1 precision, `mmq.cuh:230` at `cbe28bb3`), used by `holon_ggml_mmq_schedule` (`csrc/api.cu:120, 126`).
  Rejected: cherry-picking ≈ 450 upstream lines into `vendor.patch`.
- **D8 — FP4 on `kernels.ggml_cuda` through a lossless repack.** `TYPES` gains `mxfp4` / `nvfp4` for the layouts
  `prepare` produces and for GGUF MXFP4; `prepare` repacks `nvfp4/hf` / `mxfp4/hf` (llama.cpp's `_nvfp4_pack`,
  `repack_mxfp4_blocks`, our code); the `q8_1` spec's `prepare` is the identity for GGUF layouts. Only the opt-in `fp4`
  spec's `prepare` appends a zeroed tail of one row padded to 512 elements to each FP4 tensor or stack (llama.cpp's
  allocation rule; ≤ 1 row; a stack stays `[E, out, row_bytes]`, a view of a buffer one tail longer), so MXFP4_MOE
  stacks are never copied under `a8`. `scale2` is applied in f32 before SwiGLU
  on every path (§4.5). The shim sets `op_params[3]` from the spec and, **before every launch and before graph
  capture**, asks mmq which precision it will use: `vendor.patch` exports `ggml_cuda_mmq_get_prec_src1` (static at
  `mmq.cu:118`); a mismatch with the spec is a Python error. The `fp4` spec copies bf16 activations to f32 before mmq
  (exact; no quantizer patch; the vendored quantizer reads f32 only). Rejected for M0.6: a W4A16 provider (W4A8 prefill
  uses int8 MMA, ≈ 4× bf16 on GeForce, design §5.6).
- **D9 — Bars on the same bytes and mode.** FP4: GGUFs from the same HF files (NVFP4 by the pin's converter; MXFP4 —
  which the pin's converter does not take for Qwen3 — by `bench/hf_to_gguf.py` from our repack, byte-equal to
  `repack_mxfp4_blocks`), and the MXFP4_MOE GGUF itself; llama-bench at the pin with `GGML_CUDA_MMQ_PREC=q8`
  (`mmq.cu:96` upstream) so both sides run W4A8; its default (W4A4 on sm_120) reported beside. FP8: llama.cpp has no
  FP8 type; the comparator is its Q8_0 of the same checkpoint (`--outtype q8_0`).
- **D10 — Reference dequantization** (`std/holon_std/weights/hf/reference.py`, `[reference]` in the manifest): FP8
  `q · s`; NVFP4 `(e2m1(code) · e4m3(scale)) · scale2`, low nibble = even element (ModelOpt `8101d128`
  `nvfp4_tensor.py:322-326`, packing `:338`); MXFP4 `e2m1(code) · 2^(e − 127)` (OCP MX v1.0). Step A2 cross-checks
  every target tensor, fp32 outputs, compared by value (signed zeros equal): transformers' `Fp8Dequantize`
  (`integrations/finegrained_fp8.py`, the judge's 5.6.0.dev0) exact; compressed-tensors' `NVFP4PackedCompressor` ≤ 2
  ulp, `MXFP4PackedCompressor` exact; ModelOpt NVFP4 (no transformers oracle) against ModelOpt's
  `NVFP4QTensor.dequantize` ≤ 1 ulp (test-only) and against gguf-py on the pin converter's bytes × `weight_scale_2`
  exact; Quark MXFP4 against amd-quark 0.13's `Pack_fp4.unpack` (`quark/torch/utils/pack.py:603-616`, test-only
  CPU dependency; element 2i in the low nibble, the reorder unused when packing) × `2^(e − 127)` exact, and against
  gguf-py on `repack_mxfp4_blocks` exact.
- **D11 — Delete the unused core quant registries now** (D-c): `SchemeImpl`, `Converter`, `Dialect`
  (`api/quant.py:55-79`), their registry fields and imports (`api/ops.py:14, 76-78`), `_DECLARED` entries and fields
  (`core/plugins.py:66-79`), exports (`api/__init__.py:47, 69-75`). `Scale`, `Scheme`, `Layout`, `QTensor` stay.
- **D12 — FP8 on our own Triton, Gate B** (D-a, D-f). A new plugin `kernels.triton_fp8` (dense only, so nothing is shared
  with `kernels.triton_experts`, which stays as it is). One spec on sm ≥ 89, modes per path: **decode (M ≤ 16): W8A16 split-K** — u8 weights decoded to
  bf16 exactly by bit operations, bf16 activations, the block scale per 128-K step in f32; **prefill (M > 16): W8A8** —
  per-token, per-128-column e4m3 activations (`a_s = max(amax, 1e-10)/448`, `x / a_s` by IEEE division `tl.math.div_rn`,
  clamped to ±448), e4m3 `tl.dot` with f32 accumulation, `a_s · s_w` per K step. No grouped FP8 kernel: nothing in M0.6
  scores one (the 30B FP8 stays on the reference provider). The DeepSeek-V3 recipe as SGLang / vLLM
  `_w8a8_triton_block_scaled_mm` and FreeToken `fp8_block_linear.py` implement it (studied, not copied: the borrowable
  code sits in files importing half of vLLM). CUTLASS sm120 blockwise is the later prefill step.

## 4. Design

### 4.1 Packages

| gate | package | change |
|---|---|---|
| A0 | `plugins/holon_plugins/kernels/ggml_cuda` | pin `cbe28bb3`: `vendor.sh`, regenerated `vendor.patch`, `SHA256SUMS`, `[provenance]`, `THIRD_PARTY_NOTICES.md` |
| A1 | `core/holon` | D11 deletions; per-slice `adopt` (§4.2) |
| A1 | `std/holon_std/ops`, `std/holon_std/moe` | `qlinear@2`, `prepare`, `act_quant`, parts, merges by shape (D4, D5); `testing.ref_kernels` @2 |
| A2 | `std/holon_std/weights/hf` | `quant/` dialects, `reference.py`, grouped yield; `std/holon_std/weights/gguf`: NVFP4 refusal |
| A3 | `plugins/holon_plugins/kernels/ggml_cuda` | FP4 types, `prepare`, `scale2` paths, two specs, pre-launch precision check (`vendor.patch` export), f32 activation copy for `fp4` as a named workspace sized at plan time (fixed for graph capture), workspace, docstring |
| A | `testing`, `judge`, `bench` | reference providers for the `hf` layouts; tiny fixtures per dialect; HF-quant map, goldens, emulators, floors, mode check, plants; `hf_to_gguf.py` |
| B | `plugins/holon_plugins/kernels/triton_fp8` | new plugin: FP8 act quant, W8A16 split-K decode, W8A8 prefill (dense); pinned configs |

### 4.2 Core

| Seam | Today | M0.6 |
|---|---|---|
| Quantized stack slices | `_fill_slice` raises for a `QTensor` (`build.py:219-220`) | after the range and "filled twice" checks: transform → error; else `adopt(ctx, leaf, q, index=index)`, False → the existing "cannot read layout" error. The rest of the bookkeeping is unchanged (`build.py:245-281`) |
| Quant registries | `Dialect` / `Converter` / `SchemeImpl`, unused | deleted (D11) |

≈ +8 / −35 lines. The core passes an index and a `QTensor` it never opens. Toy proof in `core/tests/test_stacking.py`:
a two-part layout (int8 rows + f32 row scales) adopted slice by slice by a toy stacked unit and read back; out of
range, twice, missing, whole-then-sliced, a transform and a refusing unit are load errors. Sequencing (D-h): M0.4
(the m04-spec branch, which changes `build.py`, `load_weights` and `api/__init__.py`) merges to `main` first; Gate A's
core change rebases onto it, and its propagate runs after.

### 4.3 Units

- `_Quantized` (`std/holon_std/ops/units.py`): facts gain `act_quant` (qlinear only); after binding, `q =
  kernel.prepare(q)` when defined (qlinear only: `qembedding@1` has no `prepare`); calls pass the parts. `Linear`
  keeps the parts, `weight` holding the primary one (GGUF: `blocks`, unchanged). `Embedding` keeps its parts and
  `qembedding@1` reads `weight` as before; a tied `LMHead` reads the embedding's parts (`{"blocks": …}` for GGUF) as
  adopted and prepares its own copy.
- `ExpertLinear`: `adopt(…, index)` materializes each part as `[E, *slice_part]` on the first slice and copies slice
  `e` in place (a slice of another layout or shape is refused); a whole GGUF stack is held as the per-expert view
  `[E, out, row_bytes]` of its bytes; `merge` per D5; `bind_kernels` adds `act_quant`, calls `prepare`.
- The judge's structural test (M0.2) compares every stacked slice's dequantization after loading with the checkpoint
  slice's, by value: step A2 (it needs `reference.py` and the dialects).

### 4.4 HF weight source

Read all headers; the dialect groups each module's tensors (FP8 `weight` + `weight_scale_inv` | `weight_scale`; NVFP4
`weight` | `weight_packed` + `weight_scale` + `weight_scale_2` | `weight_global_scale`; MXFP4 `weight` |
`weight_packed` + U8 `weight_scale`) and checks the config (declared-quantized modules have their parts, ignored ones
are dense); yields dense tensors and modules in shard order, each part read once (`safe_open` handles per shard).

### 4.5 `kernels.ggml_cuda` for FP4

- **Repack** (`prepare`, on the device; ≤ 230 MB transient for a 30B gate/up stack): NVFP4 per 64 elements `{4
  e4m3 scale bytes, 32 bytes with element j low and j + 8 high per 16-element sub-block}` (`conversion/base.py:996-1018`
  at `cbe28bb3`; ggml decodes the scale ×0.5 against a doubled table, same value); MXFP4 per 32 `{e8m0, 16 bytes,
  j low, j + 16 high}` (`:917-951`). The `fp4` spec's `prepare` then adds the zeroed 512-element tail (D8); GGUF
  MXFP4 under `fp4` gets the tail only, under `q8_1` nothing.
- **Specs:** `q8_1` (every class, priority 10, every type, `act_quant` ∈ {`a4`, `a8`}); `fp4` (sm_120, priority 11,
  FP4 types, `act_quant = a4`; mmq W4A4-dynamic, mmvq q8_1).
- **`scale2`:** mmvq via the vendored fused `x_scale` / `gate_scale` (dense: 1 value; experts: rows of `[2, E]`); mmq
  dense folded into the f32 → bf16 copy; mmq experts: our f32 SwiGLU takes per-row gate and up scales gathered by
  expert id, down does `y *= scale2[ids][..., None]` (device gather, graph-safe).
- **C API** (`csrc/api.cu`, ours): mmq / mmq_id take the precision and set `dst.op_params[3]`; `mmq_prec` returns
  what the exported `ggml_cuda_mmq_get_prec_src1` will choose for those tensors, checked before launch and capture;
  mmvq / mmvq_id take the optional scales; `mmq_schedule` takes the precision; the
  workspace adds the NVFP4 per-row scale for `rows` (dense) or `tokens·k` (with ids) floats + 256 B, for every instance
  (one size per workspace name, `build.py:93-100`); a test-only entry returns the quantized activation blocks.
- Calls are counted per (path, mode) (M0.2's counters).

### 4.6 `kernels.triton_fp8` for FP8 (Gate B)

- `act_quant_fp8(x [R, in]) → (e4m3 [R, in], a_s f32 [R, in/128])`, one program per (row, group).
- Decode W8A16 split-K: one program per (N-tile, K-split) streaming weights once, f32 partials reduced in a fixed
  order (deterministic, no atomics); prefill W8A8 tiles `BM × BN × 128`. Configs pinned per (out, in) and class (no
  autotune); support `in % 128 = out % 128 = 0`.

### 4.7 Activation mode per class

| | sm_86 | sm_89 | sm_120 |
|---|---|---|---|
| FP4, default `a8` | W4A8 (mmq int8 MMA, mmvq dp4a) | W4A8 | W4A8 (mmq Ampere config) |
| FP4, `quant.act = a4` | W4A8 | W4A8 | prefill W4A4-dynamic, decode W4A8 |
| FP8, default (Gate B) | — (v1) | prefill W8A8, decode W8A16 | same |
| any quantized layout, `quant.act = a16` | startup error | startup error | startup error |

### 4.8 Numerics and the judge (thresholds and predictions fixed before measuring)

- **Golden:** HF fp32 on the CPU, weights from `reference.py` through the judge's own HF-quant name map (it reads
  the safetensors and inverts compressed-tensors' globals itself), streamed per layer for the 30B; the MXFP4_MOE GGUF
  through M0.2's GGUF path (gguf-py has MXFP4). Prompt set `v1.jsonl`; meta carries the reference hashes.
- **Gates:** per tensor (per expert) NMSE vs the HF bf16 original: FP8 ≤ 1e-2, FP4 (HF and the GGUF's MXFP4 experts)
  ≤ 0.1, other ggml types M0.1's 0.5. Predicted FP8 ≈ 1e-3, NVFP4 ≈ 1e-2, MXFP4 ≈ 2e-2.
- **Floors** (HF bf16 on the dequantized weights with M0.2's MoE mirror, the mode emulated on every quantized linear
  and expert stack), with orders of magnitude from the panel's CPU simulation on Qwen3-1.7B (KL p99 vs golden):

  | floor | emulation | sim KL p99 |
  |---|---|---|
  | A16 | none | ≈ 8e-3 |
  | W4A8 | the q8_1 emulator per path (FP4 types: no offset sums) | ≈ 2e-2 |
  | W8A8 | e4m3 per token per 128, IEEE division, ±448 | ≈ 6e-2 (≈ 3× W4A8) |
  | W4A4-dynamic | ggml's quantizer: NVFP4 row `amax/(6·448)` + UE4M3 per 16 chosen among 5 codes (`quantize.cu:199, 250-290`); MXFP4 e8m0 per 32, `log2` rounded to nearest (`:122-140`) | ≈ 0.9 (NVFP4), ≈ 3.4 (MXFP4) |
  | W4A4-static (report) | NVFP4 per-16 e4m3 scale over the checkpoint's static input scale: per projection; per expert for experts (each (token, slot) row uses its expert's); compressed-tensors `1/input_global_scale` | ≈ 1.2 |

  Predicted order on the real files: A16 < W4A8 < W8A8 ≪ W4A4. Budgets unchanged (`judge/holon_judge/compare.py:30-33`);
  Tier A sanity ceiling KL p99 ≤ 1e-4. Floors are recorded only after step A2's oracle checks pass (M0.2's
  floor-validity rule). A run is scored against the floor of the mode each path took (per-instance counters); a
  mismatch fails before scoring.
- **Tier A** (CPU fp32, production source + dialects + reference providers): every target and fixture; it covers
  dialects, normalization, merges and stacking; the repack is L0's.
- **Tier B** per acceptance 6 and 13. Decode step size: ggml from the plugin's mmvq limits (M0.1); `triton_fp8`
  from its M ≤ 16 switch.
- **L0:** ggml FP4 per spec × path × input dtype against the declared mode's emulator, `l0.check` budgets; the GEMM
  checks consume the activation codes from the test entry, so they stay tight; the codes themselves are checked
  against the emulator at a mismatch rate ≤ 1e-3, mismatches allowed only in the tie classes `-use_fast_math` can
  flip, each within 4 ulp: an element near an e2m1 rounding boundary; the whole 16-value sub-block when two of NVFP4's
  five candidate codes have near-equal squared errors; the whole 32-value block when MXFP4's `log2(amax)` is near a
  half-integer (the f32 row scale's fast-math reciprocal only nudges values by an ulp: no whole-row class). The pre-launch precision equals the
  spec's; `scale2` on every path vs f64;
  poisoned bytes after the tail. FP8 (Gate B): act quant bitwise vs torch `float8_e4m3fn` (IEEE division on both
  sides); e4m3 → bf16 decode over the 254 finite codes; distinct random block scales; T ∈ {0, 1, 7, 8, 9, 16, 17, 64,
  512, 513, 2049} (the M = 16 / 17 mode switch); graph replay; run-to-run bit equality.
- **Plants** (`judge/holon_judge/planted.py`; multiples of the floor's KL p99; outside a prediction: recorded with its
  cause, δ unchanged):

  | # | plant | where | predicted |
  |---|---|---|---|
  | N1 | compressed-tensors global not inverted | Tier A (RedHat) | fail, gross |
  | N2 | NVFP4 repack nibbles swapped | L0; Tier B | L0 fail; L2 fail, gross |
  | N3 | NVFP4 sub-scales permuted in a 64-block | L0; Tier B | L0 fail; L2 fail, gross |
  | N4 | per-expert `scale2` gathered by slot in mmq down | Tier B 30B NVFP4 | prefill fail (3–20×); decode pass |
  | N5 | `scale2` applied after SwiGLU on mmq | Tier B 30B NVFP4 | prefill fail, gross; decode pass |
  | N6 | `op_params[3]` left zero on the `q8_1` spec, 8B NVFP4, sm_120 | Tier B 5080 | a Python error from the pre-launch precision check at the first q/k/v mmq call (warm-up, before capture); sm_86 / 89 pass |
  | N6b | the same on the 30B NVFP4, `a8`, sm_120a | L0 | fail (the same Python error) |
  | N7 | the `fp4` spec scored against the W4A8 floor | verdict | mode check fails before scoring |
  | N8 | layer 18 `o_proj` `scale2` × 1.25, W4A8 | Tier B A5000 | fail, 4–8× (sim 5.8×) |
  | N8b | the same × 1.1, W4A8 | Tier B A5000 | near threshold: 1.3–2.5×, fail if > 2× |
  | N9 | the × 1.25 plant under `a4`, vs the W4A4 floor | W4A4 report, 5080 | **pass** (sim 1.01×): the W4A4 floor cannot see it |
  | M1 | MXFP4 e8m0 byte off by one in the repack | L0; Tier B | L0 fail; L2 fail, gross |
  | M2 | Quark 30B expert index off by one in per-slice adopt | structural; Tier A | fail |
  | M3 | Quark `weight_scale` decoded as e4m3, not e8m0 | gate | gate fail (NMSE ≫ 0.1) |
  | M4 | no zeroed tail in the `fp4` spec's `prepare` | L0 sm_120a, `fp4`, `in = 768` | L0 fail (NaN output); `q8_1` spec pass |
  | M5 | MXFP4 scale × 0.5 (e8m0 − 1) in the dialect | gate | fail (sim NMSE 0.271 > 0.1) |
  | T1 | dialect pairs k's `scale` with v's weight | Tier A 8B NVFP4 | fail (≥ 3×) |
  | P1 | FP8 scale read transposed | L0; Tier B | L0 fail; L2 fail, > 10× |
  | P2 | FP8 scale shifted by one K block | L0; Tier B | L0 fail; L2 fail, > 10× |
  | P3 | W8A8 activation scale per token, not per 128 | L0; Tier B 5080 | L0 fail; **L2 pass** (sim 1.3×) |
  | P4 | e4m3 subnormals decoded as normals (W8A16) | L0; Tier B | L0 fail; **L2 pass** (≤ 1.2×: moves only weights below 3.5e-5 of their block max) |
  | P5 | dense gate/up merge keeps gate's scale rows for up | Tier A | fail, > 10× |
  | P6 | 30B FP8 expert scale stack off by one (reference provider) | structural; Tier A | fail |
  | P7 | layer 18 `o_proj` FP8 block scales × 1.25, W8A8 | Tier B 5080 | near threshold: 1.5–4×, fail if > 2× |

  The GGUF MXFP4_MOE also takes M0.1's wrong-type and stale-q8_1 plants (fail).
- **Bounded:** graphs vs eager at decode batches 3, 5, 12, 20: 30B NVFP4 on the 4090 (`a8`) and 8B NVFP4 on the 5080
  (`a4`): pass. Chunked vs one-shot prefill: 8B NVFP4 (Gate A) and 8B FP8 (Gate B) on the 5080: pass.
- **Quantization error**, reported per format and mode: the file's golden vs the bf16 checkpoint's fp32 golden, the
  engine vs that golden, top-k flip rates; W4A4-dynamic vs W4A4-static for NVFP4.

### 4.9 Memory fit

Rule from the measured records: KV = `memory_fraction` × (usable − context − weights) − peak step − graph pool.
RTX 5080 (`fit_qwen3-8b-gguf_rtx-5080`): usable 15.56, context 0.91, peak 0.31, graphs 0.26 GiB at `max_seqs` 32. RTX
4090 node (`fit_qwen3-30b-a3b-gguf`): usable − context = 21.04 GiB, peak + graphs taken as the 5080's 0.57. KV per
token: 8B 144 KiB (bf16), 30B 96 KiB. Tokens at fraction 0.98 / 0.9:

| | card | weights | KV tokens 0.98 / 0.9 | acceptance (0.9, `max_seqs` 32) | fallback |
|---|---|---|---|---|---|
| 8B NVFP4 | 5080 | 5.96 GiB | 57.9K / 52.8K | ≥ 32K | — |
| 8B FP8 | 5080 | 8.79 | 37.7K / 34.3K | ≥ 32K (Gate B) | 0.98, or q8_0 KV |
| 30B NVFP4 | 4090 | 16.85 | 38.6K / 35.0K | ≥ 32K | 0.98 |
| 30B MXFP4_MOE GGUF | 4090 | ≈ 15.9–17.1 (file chosen in A0) | 48.7K–35.9K / 44.2K–32.5K | ≥ 32K | 0.98, or q8_0 KV |
| 30B MXFP4 `amd-quark` | 4090 | 17.22 | 34.7K / 31.3K | recorded (no lock) | 0.98 |
| 30B FP4 | 5080 | ≥ 15.9 | does not fit | — | GGUF UD-IQ3_XXS (M0.2) |
| 30B FP8 | 5090 | ≈ 29.1 | computed only (no grouped FP8 kernel): ≈ 9–10K if usable ≈ 31.5 | recorded estimate | q8_0 KV |

The repack keeps the byte count; the FP4 tail adds ≤ one row per tensor. 5090 numbers are unmeasured.

### 4.10 Speed measurement

As M0.1 / M0.2: `bench/llama_bench.py` in-process vs llama-bench at the pin (`-p 512 -n 128 -d 0,256 -ngl 99 -fa
on`), medians of repetitions 2–5, from the fixed bench worktree; `llama-server` vs Holon over HTTP on the decode and
agent traces. Predictions: FP4 at W4A8 0.95–1.1× (same kernels; the post-scale folds into an existing copy); FP8
tg128 0.85–1.0× Q8_0 (gate B1 predicts it end to end first), pp512 0.4–0.9× (sm_89's fp8 MMA peak is half its
int8 peak). Evidence: `evidence/<name>_<gpu-class>_<date>.json`.

## 5. Implementation plan

Hosts: **A5000 (sm_86)** — Tier A and goldens (CPU), structural tests, fixtures, sm_86 L0, Tier B W4A8 for the 8B
NVFP4, 30B NVFP4, `amd-quark` and MXFP4_MOE. **Cluster only** — sm_89 / sm_120a L0, Tier B on the 5080 / 4090 / PRO
6000, the W4A4 report, every bar, fit, lock, snapshot and lock-class propagate, all of Gate B's kernel work.

| Step | Work | Checks | Days |
|---|---|---|---|
| A0 | Pin bump (D7): re-vendor, regenerate the patch, `csrc/api.cu` for the 5-argument `ggml_cuda_mmq_get_config`, L0 18 types ×3 classes; GGUF locks propagate; spot-check bar. Files: targets, MXFP4_MOE GGUF chosen, the pin's GGUF conversions; header checks (ties, `k/v_scale`, scale bytes) | acceptance 1 | 2 |
| A1 | Core (D11, per-slice adopt + toy); `qlinear@2`, `prepare`, `act_quant`, parts, merges; ref_kernels @2. After M0.4 merges: propagate | acceptance 2 (byte-identical snapshots) | 2 |
| A2 | Dialects (each yields `weight` first: the primary part, checked), `reference.py`, owner cross-checks (D10), NVFP4 GGUF refusal; the structural test's quantized slices (§4.3); tiny fixtures per dialect (CT by compressed-tensors' compressors; FP8, ModelOpt, Quark written from their specs); engine (reference kernels) vs transformers on them | fixtures; Tier A plants on fixtures | 3 |
| A3 | ggml FP4: types, repack, `fp4` tail, `scale2`, specs, pre-launch precision check, C API, docstring | acceptance 4, 5 (A5000 first, then 4090 / 5080) | 3.5 |
| A4 | Judge: HF-quant map, gates, emulators, floors, mode check. Machine time: goldens 4 × 8B (≈ 0.5 h each) + 4 × 30B (≈ 2 h each, streamed) ≈ 10 h CPU, overnight; floors ≈ 12 × 0.5 h on A5000s; Tier A ≈ 8 runs | gates; Tier A all; floors' order | 3 + 1.5 machine days |
| A5 | Tier B (A5000, then 5080 / 4090 / PRO 6000), plants, W4A4 report, structural, bounded | acceptance 6, 7 | 2.5 |
| A6 | Consumer: fits, bars (same mode and default), serving, tune + locks, snapshots ×3; Gate A panel review | acceptance 8–10 | 3 |
| B1 | `kernels.triton_fp8`; W8A16 split-K decode kernel; per-shape microbenchmark, head and non-GEMM timings, llama-bench Q8_0 tg128 on 5080 / 4090 | acceptance 11 | 2 |
| B2 | W8A8 act quant + dense prefill GEMM; L0 sm_89 / sm_120a | acceptance 12 | 2.5 |
| B3 | W8A8 floors, Tier B, FP8 plants, bars, 5080 fit, locks, snapshots; Gate B panel review | acceptance 13–15 | 2.5 |
| C | Docs: design §4.2 (no dialect / converter slots), §5.5 as built, §5.6; plan §2 / §10 status | — | 0.5 |

≈ 26.5 developer days plus machine time; Gate A ≈ 19, Gate B ≈ 7.

## 6. Risks

- **Pin bump:** a week of upstream MMQ changes; GGUF snapshots may move (step A0's own approval, L2 decides) before
  the refactor, so acceptance 2's byte-identity is measured against the bumped locks.
- **FP8 decode bar:** gated by B1's end-to-end prediction (≥ 0.85× Q8_0); below it, the split-K kernel is reworked
  before B2.
- **8B FP8 and 30B FP4 fits** have 2–3K tokens of margin at 0.9; fallbacks named in §4.9.
- **Sequencing:** Gate A's core change and propagate wait for M0.4's merge (D-h).
- **Community fixtures** (`Yi30`): quality surprises show in the gate and the reports, not in acceptance.

## 7. Deferred, deliberately

- Static-scale W4A4 providers on sm_120 (vLLM's CUTLASS sm120 NVFP4 GEMM and grouped MoE, FlashInfer `mm_fp4` /
  `cutlass_fused_moe`), judged against the W4A4-static floor recorded here; `a4` as a default (acceptance 7's rule).
- CUTLASS sm120 blockwise FP8 (examples 87b / 87c) for prefill; grouped FP8 kernels (the 30B FP8 on GPUs); FP8 on
  sm_86 (plan §10, v1); `quant.act = a16`.
- A16 FP4 kernels; ModelOpt `MIXED_PRECISION` and per-tensor FP8; Quark NVFP4; GGUF NVFP4.
- gpt-oss (interleaved gate/up, biases, clamped SwiGLU: a new model plugin); AMD's official MXFP4 Qwen3.5/3.8 builds;
  vLLM as a secondary bar.
- Dialects as their own plugins (D2's trigger).

## 8. Review findings, resolved

**Rev 1.** User decisions: D-a two gates → acceptance split, D7, D12, §5; D-b `a8` default, `a4` opt-in, the
pre-registered rule → D3, acceptance 7, N9; D-c → D11; D-d → D4; D-e → D6, §4.2; D-f → non-goals, D12, §4.7; D-g → D1,
acceptance 3 / 10 / 15; D-h → §4.2, risks.

| # | finding | resolution |
|---|---|---|
| A-M1 | W4A4 mmq K loop steps 512 with no tail: `in = 768` reads past the end | zeroed one-row tail to 512 elements on every FP4 tensor / stack, GGUF MXFP4 included (D8, §4.5), as llama.cpp's allocation; docstring fixed; plant M4 |
| A-M2 | source parts kept after the repack | `prepare` returns a new `QTensor`, the unit drops the source parts (D4) |
| A-M3 | W4A4 codes cannot be bitwise (`-use_fast_math`) | boundary-bounded code check with a capped rate; GEMM L0 consumes the kernel's codes; floors budget-based (§4.8) |
| A-M4 | memory table not from the 0.9 rule and records | recomputed from the fit records (§4.9), fraction and `max_seqs` in acceptance 8 / 14, fallbacks named |
| A-m | owner cross-checks: fp32 outputs, CT NVFP4 ≤ 2 ulp, D2's "≤ 0.5 ulp" | D10 tolerances; D2 says ≤ 1 ulp |
| A-m | signed zeros; bad scale bytes | compare by value; load-time asserts on sign bit and 0x7F / 0xFF (D2, D10) |
| A-m | FP8 act quant division and clamp; 256-code check | `tl.math.div_rn`, ±448 (D12); 254 finite codes (§4.8) |
| A-m | N6 mispredicted | first W4A4 call is q/k/v; read-back of the chosen precision (D8, §4.5), N6 / N6b reworded |
| A-m | Tier A dialect plant with unequal shapes | k's scale with v's weight (T1) |
| A-m | merges: `input_scale` need not be equal; gate/up `scale2` may differ | input scales not loaded (D2); gate/up `scale2` carried as `[E, 2]` (D5) |
| A-m | `TYPES` admits GGUF NVFP4 | refused at the GGUF source (D1) |
| A-m | Quark has no owner oracle | gguf-py cross-check + plant M3 (D10) |
| A-m | citations | `ExpertLinear.adopt` `moe/units.py:42-48`; ModelOpt packing `nvfp4_tensor.py:338` |
| B-1 | one milestone too large | Gate A / Gate B (§1, §5) |
| B-2 | pin bump bundled with the refactor | its own first step and propagate (D7, A0) |
| B-3 | dead core registries | deleted in Gate A (D11) |
| B-4 | `prepare` signature / layout | `prepare(QTensor) → QTensor` with the destination layout, no input mutation (D4) |
| B-5 | per-slice seam size | minimal form, units keep dim 0 = E during load (D6) |
| B-6 | `a8` vs GGUF FP4 defaults | `a8` for every quantized layout, GGUF included (D3); MXFP4_MOE target (D1) |
| B-7 | sm_86 FP8 scope | stays v1 (non-goals, D12) |
| B-8 | dialects in `std.weights.hf` | one module per dialect under `weights/hf/quant/`; extraction trigger recorded (D2) |
| B-9 | rename timing | no rename: a new `kernels.triton_fp8` in Gate B (D12, B1) |
| B-10 | M0.4 sequencing, draft heads | §4.2, D4 |
| C-B1 | W4A4 default unsafe | D3 / D-b |
| C-M1 | FP8 decode bar at M = 1 | W8A16 split-K decode mode at M ≤ 16, bandwidth gate B1 (D12, acceptance 11) |
| C-M2 | fit numbers | §4.9 from the records; 5090 marked unmeasured |
| C-M3 | code checks, Triton division | §4.8 L0, D12 |
| C-M4 | sm_120a L0 coverage; zeroed `op_params` plant | spec × path × input dtype vs the declared mode's emulator; N6b |
| C-M5 | floor ordering; W8A8 near-threshold plant | §4.8 table (W8A8 ≈ 3× W4A8); P7 |
| C-m | N8 band | sim 5.8× → N8 fail 4–8×; near threshold moved to N8b (× 1.1) |
| C-m | W4A4 workspace and f32 cast | `tokens·k` + 256 B per-row scale; bf16 FP4 quantizer patch instead of a cast (superseded in rev 3: Arch-1) |
| C-m | graphs vs eager for W4A4 on sm_120 | 8B NVFP4 `a4` on the 5080 (§4.8, acceptance 7) |
| C-m | gate NMSE predictions | ≈ 1e-2 / 0.1 / 0.3; MXFP4 margin as a risk (superseded in rev 3: R1) |
| C-m | W4A4-static definition | per projection / per expert, CT reciprocal (§4.8 floors) |
| C-m | reference multiply order | `(e2m1 · e4m3) · scale2` (D10) |
| C-m | 8B FP8 Tier B on sm_89 | rtx-4090 in acceptance 13 |
| C-m | A5000 vs cluster; step-4 schedule | §5 host split; A4 machine time |

**Rev 2.** All three reviewers: nothing blocks. Rulings: the bf16 FP4 quantizer patch is dropped for rev 1's f32 copy;
grouped FP8 is dropped; `quant.act = a16` is not offered.

| # | finding | resolution |
|---|---|---|
| R1 / R2-m1 | gate NMSE predictions ≈ 10× too high | predictions FP8 ≈ 1e-3, NVFP4 ≈ 1e-2, MXFP4 ≈ 2e-2; gates FP8 ≤ 1e-2, FP4 ≤ 0.1 (§4.8, acceptance 6); plant M5; the MXFP4 margin risk removed |
| m1 | MoE gate/up `scale2` layout vs mmvq's contiguous indexing | `[2, E]`, one contiguous row per half (D5, §4.5) |
| m2 | the read-back needs `ggml_cuda_mmq_get_prec_src1`, which is static | `vendor.patch` exports it; checked before launch and before capture (D8, §4.5) |
| m3 | `ggml_cuda_mmq_get_config` gains a fifth argument at `cbe28bb3` | step A0 edits `holon_ggml_mmq_schedule` (D7, §5) |
| m4 | M4's symptom | L0 fail (NaN output), not memcheck |
| m5 | Quark has a CPU owner oracle | amd-quark 0.13 `Pack_fp4.unpack` × `2^(e − 127)`, test-only (D10) |
| m6 | MXFP4 e8m0 0xFF | load-time assert (D2) |
| R2-M1 | FP8 decode gate on bandwidth alone | end-to-end prediction from microbenchmarks + head + non-GEMM time, ≥ 0.85× the measured Q8_0 tg128 (acceptance 11, B1) |
| R2-m2 | code-check tie classes | element, sub-block candidate near-ties, MXFP4 `log2` half-integers, NVFP4 row scale (§4.8 L0) |
| R2-m3 | N6 with a pre-launch check | predicted Python error at the first q/k/v call; N6b the same (§4.8) |
| R2-m4 | Gate B decode step size | from `triton_fp8`'s M ≤ 16 switch (§4.8, acceptance 13) |
| Arch-1 | bf16 FP4 quantizer patch | dropped; `fp4` copies activations to f32 (D8) |
| Arch-2 | grouped FP8 without an L2 consumer | dropped; 30B FP8 on the reference provider, fit computed (D1, D12, §4.9) |
| Arch-3 | `a16` | not offered in M0.6, a startup error (D3, §4.7) |
| Arch-4 | tail copies under `a8` | tail only in the `fp4` spec's `prepare`; `q8_1` `prepare` is the identity for GGUF layouts (D8, §4.5) |
| nit | unused `declared_act` | removed (D2) |

**Rev 3.** All three reviewers: clean. Applied nits: the FP8 kernel is a new `kernels.triton_fp8` (no rename: dense only,
nothing shared with `triton_experts`); no whole-row tie class (the row scale is f32); the `fp4` spec's f32 activation
copy is a named plan-time workspace.

**As built (A0, A1).** Recorded deviations from rev 3; the panel reviewed A0 + A1 and its fixes are in.

| item | rev 3 | as built |
|---|---|---|
| A0 C API | 5-argument `ggml_cuda_mmq_get_config` | also `ggml_cuda_mmq_needs_fallback`; the workspace follows mmq's new allocations (q8_1 tile padding J ≤ 128 rows rounded to a load chunk; `ids_dst` J − 1 entries of read padding); the MoE padding plant reads the new layout |
| expert stacks | dim 0 = E while loading; GGUF whole stacks `[E·out, row_bytes]` unchanged; the repack to `[E·out, …]` is `prepare`'s | every part per expert, `[E, …]`, GGUF whole stacks as a view; so `linear_experts@1` → `@2` (D4, D6, D8, §4.3); `kernels.ggml_cuda` reads it as `[E·out, row_bytes]`; one merge rule for dense, GGUF and HF stacks (D5) |
| per-tensor parts | `scale2` `[]` | by rank: `[]` or `[1]` in every unit (per expert `[E]` or `[E, 1]`) is per tensor, bit-equal (dense) or `[2, E]` (experts); a per-row part is 2-D, so a `[1, 1]` block scale merges as a row (D5) |
| `qembedding@1` | "`Embedding` / `LMHead` pass `{"blocks": …}`" | unchanged contract and no `prepare`; `Embedding` keeps its parts for a tied head (§4.3) |
| primary part | unspecified | the first part of `QTensor.parts`, held by `weight`; A2's dialects yield `weight` first and A2's tests check it (D4) |
| slices to plain units | — | `Linear` / `Embedding` / `LMHead.adopt` take `index` and refuse a slice: the core's load error (D6) |
| `quant.act`, `config["act"]` | with the providers (A3) | in A1: the option (`a16` a startup error) and `kernels.ggml_cuda`'s `{"mmvq": "q8_1", "mmq": "q8_1"}`, so GGUF locks propagate once (D3) |
| structural test | A1 (§4.3) | A2: it needs `reference.py` and the dialects |
| core | registries deleted (D11) | also the dead `dialect` / `converter` slot kinds (`api/manifest.py`) |
| ggml workspace | — | the pool records its high-water mark (a test-only process-wide counter); L0 holds every call to `workspace_bytes(_id)` for its rows, and on a J = 128 fixup shape (dense, with ids) the formula's spare to ≤ 2048 + 3 · 256 bytes (the load-chunk allowance stays: `nthreads` may be any multiple of 32, so 128 · 144 bytes can round up past itself); `mmq_schedule` aborts on FP4 types until A3 passes the precision |

**As built (A2).** Recorded deviations from rev 3 in step A2.

| item | rev 3 | as built |
|---|---|---|
| declared-quantized modules | the dialect checks the config: declared-quantized modules have their parts, ignored ones are dense (§4.4) | configs name their targets by class (`Linear`), which a weight source cannot see: a module is quantized when it holds a dialect part other than a float `weight`; it must hold every part and not be ignored (`ignore`, `exclude(_modules)`, `modules_to_not_convert`: names, dotted prefixes or suffixes, `*`, `re:`), so an ignored module is dense; one the config would quantize but stores dense loads dense |
| grouped yield | dense tensors and modules in shard order, each part read once | a module at its primary part's place; dropped tensors (input scales) are never read, except `k_scale` / `v_scale`, checked to be 1.0 before the first yield; a tensor in two shards is a load error |
| group and block sizes | 16, 32, 128 × 128 | FP8 blocks from the config (`weight_block_size`, compressed-tensors' `block_structure`): both dimensions whole blocks and the scale `[out / bm, in / bn]`, else a load error; `reference.py` reads group and block sizes from the parts' shapes (a weight merged by rows dequantizes as its units); each scheme's normalization returns its logical shape (its packing) |
| compressed-tensors FP8 | `weight_scale` named in §4.4 | `float-quantized` with block weights; a fixture, no target |
| `hf_quant_config.json` | read when config.json has no `quantization_config` | its `producer.name` is the quant method (ModelOpt's file) |
| owner cross-checks (D10) | transformers exact, compressed-tensors ≤ 2 / 0 ulp, ModelOpt ≤ 1 ulp, Quark exact, gguf-py exact | `python -m holon_judge hf-quant-check` (`holon_judge/hf_quant.py`), every quantized tensor of a checkpoint. transformers' `Fp8Dequantize` is exact given the dialects' f32 scales (with BF16 scales it rounds its product to bf16); compressed-tensors' own unpacking, scale decoding and `dequantize` in f32 (its compressor's `decompress` returns bf16); gguf-py on llama.cpp's repack for every FP4 tensor, compressed-tensors' included. amd-quark 0.13 is not in the development environment (it requires huggingface-hub 2 and datasets 5, which break transformers 5.6): its check runs where it imports |
| fixture oracle | engine vs transformers on the fixtures | transformers in fp32 on each fixture's weights as the file means them (compressed-tensors' decoding; the spec writers' own for FP8, ModelOpt and Quark): transformers has no ModelOpt loader, needs amd-quark for Quark, and rounds FP8 with BF16 scales and compressed-tensors' FP4 to bf16 |
| judge HF-quant map | A4 (goldens) | in A2, for the structural test: from names and dtypes, independent of the dialects; `structure --device cpu` for stacks no GPU kernel reads yet (the reference kernels) |
| plants | §4.8 | N1, T1, M3, M5, P5, P6 in `hf_quant_plants` (layer 18 in the CLI's set), with a per-expert `scale2` stack off by one and gate/up `scale2` rows swapped in the merge (both the structural test's); M2 is `hf-stack-off-by-one`; on the fixtures each fails at ≥ 10× the passing bound |
| GGUF NVFP4 | refused at load | when the source opens the file, naming the first NVFP4 tensor |
| GGUF MXFP4 | — | an e8m0 scale of 0xFF (NaN) is a load error, as for HF |
| expert stacks in the references | — | `std.moe.reference.expert_parts(parts, experts, merged)` splits a `linear_experts@2` stack per expert and unit (rank ≥ 3 per row, ≤ 2 per tensor, the merge state given, not guessed from a [2, E] shape); std.weights.hf's reference stays per tensor |
| structural test after `prepare` (A3) | — | A3: once `kernels.ggml_cuda` prepares HF FP4 stacks, the structural test reads their ggml blocks with gguf-py × the `scale2` row and drops the 512-element tail of the `fp4` spec |

**As built (A3).** Recorded deviations from rev 3 in step A3 (`kernels.ggml_cuda` for FP4).

| item | rev 3 | as built |
|---|---|---|
| precision check | before every launch and before capture | once per instance when it is built, before any launch or capture: the precision mmq takes depends only on the type, the precision asked for and the device. The C API builds the probe's dst with the same helper as every mmq call (`with_prec`), so a zeroed `op_params[3]` (plant N6, `ggml-prec-default`) is seen by it |
| dense `scale2` on mmvq | the fused `x_scale` | llama.cpp fuses scales at one row only (`mmvq.cu`, "fusion only supported for ncols_dst=1"): one row fused (bf16 out too: `vendor.patch` lets the fused epilogue store bf16), 2–8 rows written f32 and multiplied into `out_dtype` in one step (llama.cpp's separate MUL) |
| `scale2` shapes after `prepare` | — | dense [1], one stack [E], merged gate/up [2, E] (gate's row, up's; any other shape is an error at the call); `prepare` repacks in row chunks of 4096 on the parts' device into one buffer that also holds the `fp4` spec's tail, so the transient is the output plus a chunk |
| the `fp4` spec's tail | appended by `prepare` | also checked on every native mmq call (the blocks' storage must extend one tail past them): a missing tail is an error, so plant M4 is the tail left unzeroed (0xFF, NaN e8m0 scales), as predicted a NaN output at `in` 768 for MXFP4 (L0 asserts it there). For NVFP4 an over-read is memory-safety only (a 0x7F scale byte decodes to 0 and e2m1 has no NaN), so memcheck catches it: the sm_120a scripts run NVFP4 and MXFP4 at `a4`, dense and experts, at `in` 768. L0's "poisoned bytes after the tail": `prepare` allocates exactly the tail, so a read past it is outside any allocation in memcheck (sm_120a, `act_quant` a4, `in` 768 and 2304, dense and experts) instead |
| call counters | per (path, mode) | `calls["mmvq/q8_1"]`, `calls["mmq/fp4"]`, … on qlinear and linear_experts instances; the judge's `expert_calls` sums the modes per path, so recordings and the Tier B path check are unchanged (A4's mode check reads the modes) |
| workspace | + NVFP4 per-row scale, every instance | as specified; the tightness tests' spare grows by that reserve, rows · 4 padded to 256 B |
| prepared parts' layout | — | a unit holds the layout of the parts it holds: after `prepare`, `ggml.<type>/gguf.native` (one line each in std.moe's `ExpertLinear` and std.ops' `_Quantized`; the sites' facts keep the adopted layout: `_Quantized` keeps them in `facts`, `ExpertLinear` reads them before `prepare` and asserts it binds once); the structural test reads such stacks by gguf-py × their `scale2` row, the view leaving out the tail |
| `qembedding` | every type | no NVFP4 (`getrows.cu` has none); MXFP4 tables are served |
| GGUF NVFP4 layout | refused at the source | the kernels accept `ggml.nvfp4/gguf.native` (no `scale2`): only L0 builds it (the type sweeps); the GGUF source still refuses such files |
| W4A4 emulator | A4 (emulators) | `holon_judge.w4a4`, written in A3 for L0 (A4's W4A4-dynamic floor reuses it). The code check's tie classes are the outcomes of the emulation with the fast-math results nudged by up to 4 ulp — 1/r and 0.5 / s together, max\|v\| / 6 alone; MXFP4's log2(amax) by max(4 ulp, 2^-20) — NVFP4's row scale read from the kernel. MXFP4's activation quantizer is not Blackwell-only, so its check runs on sm_86 too (0 differing codes there) |
| test-only entry | the quantized activation blocks | FP4 only (`quantize_fp4`: llama.cpp's native quantizer for a dense f32 input, as mmq calls it); q8_1 codes stay `q8_1.py`'s emulation |
| acceptance 4 | the five HF FP4 files; the two 8B NVFP4 byte-equal to the pin's converter | `python -m holon_judge hf-quant-check --repack cuda --llama-cpp …`: every FP4 module of all five files byte-compared with the converter's `_nvfp4_pack` / `repack_mxfp4_blocks` (the bytes it writes into a GGUF; no GGUF conversion is made), besides gguf-py × scale2 against the reference. Met: 252 + 252 + 252 + 18624 + 18432 modules, 0 ulp, 0 differing bytes (`evidence/repack_hf-fp4_rtx-a5000_*`) |
| provenance of the repack | "our code", following the converter (D8) | adapted from llama.cpp cbe28bb3 (`_nvfp4_pack`, `repack_mxfp4_blocks`, ggml-cuda.cu's row padding): marked in `plugin.py`, listed in `[provenance]` and THIRD_PARTY_NOTICES.md |
| sm_120a L0 at `fa5b701` | — | `test_native_w4a4_experts_against_their_own_codes` failed at `in` 2048, 513 tokens (both FP4 types): the test scored the one-token second chunk, which runs mmvq at q8_1 as the plugin dispatches per 512-token chunk, against native W4A4 codes. The reference now follows each chunk's path (the q8_1 emulator on mmvq) and the call counters must show both; the kernels were right (the 512-token chunk within 0.005 of its own codes on the RTX PRO 6000) |
| note for A4 | — | the D3 mode check and plant N7 need the dense counters too: `Linear` keeps its kernel at `m._q._kernel` (not `m._kernel`, which `judge/holon_judge/engine.py` reads), and `_by_path` folds the modes into paths |
| plants | §4.8 | in `planted.py` (`fp4_plants`, layer 18): `nvfp4-repack-nibbles-swapped` (N2), `nvfp4-repack-scales-permuted` (N3), `nvfp4-scale2-by-slot` (N4), `nvfp4-scale2-after-swiglu` (N5), `ggml-prec-default` (N6, N6b), `nvfp4-o-proj-scale2-1.25` / `-1.1` (N8, N8b: std.weights.hf's `scale2` read × the factor), `mxfp4-repack-scale-off-by-one` (M1), `fp4-tail-not-zeroed` (M4). L0 bites: N2, N3, M1 (the lossless check; M1 also the GPU structural test), N4, N5 (the MoE `scale2` checks), N6 (sm_120: the build's error; sm_86 passes), M4 (sm_120). N8 / N8b are Tier B's (A5) |

**As built (A4).** Recorded deviations from rev 3 in step A4 (the judge); results in "A4 results" below.

| item | rev 3 | as built |
|---|---|---|
| MXFP4_MOE file | fixed in A0 | fixed in A4 (A0 had not): `Face314/Qwen3-30B-A3B-Instruct-2507-MXFP4_MOE` at `68d644f`, 17.08 GB (experts MXFP4; attention, embedding, head Q8_0; norms, routers F32); its original `Qwen/Qwen3-30B-A3B-Instruct-2507` at `0d7cf23` |
| streamed goldens | the 30B's, per layer | every GGUF and quantized HF golden and floor (a dense checkpoint with `golden --stream`) runs one decoder layer at a time (`golden._streamed`): the model's own forward once per prompt with stand-in layers that keep each layer's inputs (masks, position embeddings), then each layer is loaded, `prepare`d (a floor's emulation), applied to every prompt and freed. Bit-equal to transformers' whole model (the dense and MoE GGUF goldens, a dense checkpoint streamed: judge tests). A 30B fp32 golden holds one layer (≈ 2.4 GB of experts) at a time; its bf16 floors run on one A5000 (no `device_map` over cards). `gguf.hf_model` is gone |
| HF-quant map | the judge's own | `hf_quant.state`: the judge's name/dtype map, `reference.py`, and transformers' fusion of per-expert tensors (pinned bit for bit by the streamed-golden test); a multimodal checkpoint's `model.language_model.` read as `model.` |
| gates | per tensor vs the bf16 original | `python -m holon_judge gate`. HF files: what the engine's weight source yields (its dialects) through `reference.py`, so a dialect's misreading shows (M5); a dense tensor cast to the original's dtype and held exact (bit-equal). Where a norm differs from the original's, the linears reading it (q/k/v; gate/up, experts', the router) are compared as W · diag(γ), each side with its own norm (a dense one within 1e-10, the products' f32 rounding), and the norm not alone; elsewhere each norm alone, exactly: RedHatAI's file is SmoothQuant-ed (strength 0.8 into q/k/v and gate/up), so per tensor its norms and those linears failed (NMSE up to 354) though the model is the original's. GGUF: streamed per layer against `hf_quant.state` of the original instead of `from_pretrained` (61 GB for the 30B), routers exact; Q8_0 ≤ 1e-3 (from its predicted ≈ 5e-5), MXFP4 0.1, other types M0.1's 0.5. The expert and norm name patterns are the Qwen family's |
| floors | the mode emulated on every quantized linear and stack | `holon_judge/modes.py`, `golden --q8-1 PATH --act q8_1\|fp4\|fp4-static`: each quantized linear's and expert stack's rows quantized in the mode (W4A8 `q8_1.quantize` per path; W4A4-dynamic `w4a4.values`; W4A4-static `w4a4.static_nvfp4` over the checkpoint's `input_scale`, dense linears only: the 8B report), then the f32 product with the dequantized weight (bf16-cast, as the GGUF floors); FP4 weights only (FP8's W8A8 / W8A16 floors are Gate B's: refused). Meta: `act`, `emulation` (one hash over `q8_1.py`, `modes.py`, `w4a4.py` and std's `reference.py`: a change to any invalidates the floor; GGUF floors too), `hf_reference`. MoE floors of quantized files leave the experts' input unrounded, as the GGUF floors. Streamed and whole-model floors are one: the pre-A4 `qwen3-8b-q4_k_m` mmq floor re-recorded through the streamed path is bit-identical, so older GGUF floors stay valid |
| floors recorded | W4A8, W4A4 (report) | also an A16 floor for every target and fixture (Tier A's budgets and plant multiples); W4A8 mmq for Yi30 (the a4 rule) |
| mode check | per-instance counters | `score()` records `calls` per `path/mode` from every quantized kernel, found by a duck-typed scan (any `calls` Counter held by a module or a plain object it holds), replacing `expert_calls` (a pre-M0.6 recording's `expert_calls` read as `<path>/q8_1`); every floor with a path (dense ones too, unchecked before) needs all calls on its `path/mode`, checked before the budgets (N7: `mmq/fp4` calls against the W4A8 floor fail on the mode alone). Tier B's decode step is the minimum, over every quantized tensor, of the plugin's dense (`use_mmvq`) and expert (`use_mmvq_id`) limits (`gguf.mmvq_step`): 7 for the MXFP4_MOE file on sm_86 (MXFP4 experts 7; dense and Q8_0 8) |
| Tier A | 1e-4 sanity ceiling | `l2 --ceiling 1e-4` beside the A16 floor's budgets. The reference kernels cache an expert stack's f32 decode up to 128 MiB (every test model's) and above decode only the experts each call routes to (a 30B's stacks, ≈ 122 GB in all, are never held), so a 30B Tier A fits 64 GB; its cost is a decode per layer and step: MXFP4 in torch ≈ 1 s a layer (vectorized, `ref_kernels.ggml_mxfp4`), gguf-py's k- and i-quants on one core ≈ 10–20 s a layer, so a 30B GGUF Tier A of those types takes hours |
| owners (D10) | every target tensor | the five FP4 files in A3's record (with the repack), amd-quark's `Pack_fp4` in A2's; A4 ran the two FP8 files |

**A4 results** (A5000 host; records `evidence/*_m06-lowp_*_2026-10-14.json`, `evidence/owners_m06-hf-files_cpu_2026-10-14.json`).

- **Owners per real file** (D10, 0 modules over bound): nvidia 8B / 30B NVFP4 — ModelOpt `NVFP4QTensor` (≤ 1 ulp) and
  gguf-py on llama.cpp's repack (exact); RedHatAI 8B NVFP4 — compressed-tensors (≤ 2 ulp), gguf-py; Yi30 8B MXFP4 —
  compressed-tensors (exact), gguf-py; amd-quark 30B MXFP4 — `quark.Pack_fp4` (A2, its own venv), gguf-py; Qwen 8B /
  30B FP8 — transformers' `Fp8Dequantize` (exact; 252 / 18624 modules).
- **Gates** (predicted FP8 ≈ 1e-3, NVFP4 ≈ 1e-2, MXFP4 ≈ 2e-2): worst tensor FP8 7.1e-4 (8B and 30B), NVFP4 9.1e-3
  (nvidia 8B and 30B), RedHatAI 1.2e-2 (its 72 norms differ from the original's: SmoothQuant, folded), MXFP4 1.6e-2
  (Yi30), 1.8e-2 (amd-quark), the GGUF's MXFP4 experts 1.8e-2 and its Q8_0 tensors ≤ 5.3e-5 (gate 1e-3), routers
  exact; every dense HF tensor bit-equal (every norm too, but RedHatAI's). None over its gate. Plants: M5 (scale − 1
  in the dialect, Yi30) 0.26 on every MXFP4 tensor (sim 0.271), M3 (Quark scales read as e4m3) 1.6e9: both fail.
- **Goldens** (fp32, CPU, 32 threads, `judge/goldens/<file>/v1-fp32.safetensors`): `qwen3-8b-nvfp4`,
  `qwen3-8b-nvfp4-redhatai`, `qwen3-8b-mxfp4-yi30`, `qwen3-8b-fp8` (≈ 4.5 min each), `qwen3-30b-a3b-nvfp4`,
  `qwen3-30b-a3b-mxfp4-amd-quark`, `qwen3-30b-a3b-fp8` (≈ 6 min), `qwen3-30b-a3b-instruct-2507-mxfp4_moe` (15 min,
  gguf-py's MXFP4 dequantization), and the GGUF's original `qwen3-30b-a3b-instruct-2507` (streamed); the existing bf16
  goldens of Qwen3-8B and Qwen3-30B-A3B are reused for the quantization error. HF goldens at `reference.py` b9705bf4.
- **Floors** (bf16, one A5000; KL p99 vs the file's golden; top-1):

  | file | A16 | W4A8 mmq | W4A8 mmvq | W4A4-dynamic (mmq) | W4A4-static |
  |---|---|---|---|---|---|
  | nvidia 8B NVFP4 | 6.10e-3 / 98.96% | 1.24e-2 / 98.69% | 1.42e-2 / 98.78% | 0.526 / 94.07% | 0.626 / 93.31% |
  | Yi30 8B MXFP4 | 5.89e-3 / 98.97% | 1.29e-2 / 98.79% | — | 1.87 / 88.75% | — |
  | RedHatAI 8B NVFP4 | 5.96e-3 / 98.93% | — | — | — | — |
  | Qwen 8B FP8 | 5.60e-3 / 98.96% | — (W8A8: Gate B) | — | — | — |
  | nvidia 30B NVFP4 | 3.98e-2 / 98.45% | 4.92e-2 / 98.36% | 5.23e-2 / 98.31% | — | — |
  | amd-quark 30B MXFP4 | 3.13e-2 / 98.58% | 3.61e-2 / 98.51% | 3.61e-2 / 98.50% | — | — |
  | GGUF MXFP4_MOE | 2.34e-2 / 98.40% | 2.81e-2 / 98.47% | 2.80e-2 / 98.44% | — | — |
  | Qwen 30B FP8 | 3.16e-2 / 98.54% | — | — | — | — |

  The predicted order A16 < W4A8 < W4A4 holds on every file (sim: A16 ≈ 8e-3, W4A8 ≈ 2e-2, W4A4 0.9 / 3.4 on
  Qwen3-1.7B). The pre-registered a4 rule (acceptance 7) is not met, as predicted: W4A4-dynamic's KL p99 is 42× the
  W4A8 floor's on the 8B NVFP4 (predicted ≈ 45×) and 145× on the 8B MXFP4, top-1 4.6 and 10.0 points lower.
  W4A4-static is looser than dynamic (0.626 vs 0.526), as simulated. The twelve emulated floors were re-recorded
  after the panel's minors and the rebase onto M0.5, at `emulation` 54d13262: bit-identical tensors.
- **Tier A** (CPU fp32, the production source and dialects on the reference kernels, vs the file's golden; ceiling
  KL p99 ≤ 1e-4): all eight pass, top-1 ≥ 99.99%.

  | file | KL p99 | A16 floor's | wall |
  |---|---|---|---|
  | nvidia / RedHatAI 8B NVFP4 | 6.2e-7 / 6.1e-7 | 6.10e-3 / 5.96e-3 | 8 min each |
  | Yi30 8B MXFP4, Qwen 8B FP8 | 6.3e-7 / 6.0e-7 | 5.89e-3 / 5.60e-3 | 8 min each |
  | nvidia 30B NVFP4, amd-quark 30B MXFP4 | 1.1e-6 / 8.3e-7 | 3.98e-2 / 3.13e-2 | 62 / 54 min |
  | Qwen 30B FP8 | 6.5e-6 | 3.16e-2 | 49 min |
  | GGUF MXFP4_MOE | 6.1e-6 | 2.34e-2 | 4 h 52 min (gguf-py's one-core MXFP4 decode on every call; vectorized after, b851135) |

  The 30B runs' few flipped positions are routing near-ties (KL max 4e-3 – 0.25 at one or two positions).
- **Plants** (multiples of the A16 floor's KL p99): Tier A N1 (RedHatAI, compressed-tensors' global not inverted)
  1.19e1, ≈ 2000× (predicted gross); T1 (nvidia 8B, k's scale on v, layer 18) 0.114, 18.6× (predicted ≥ 3×); P5
  (Qwen 8B FP8, gate's scale rows for up) 3.36, 600× (predicted > 10×): all fail. Gate M5, M3 fail (above).
  Structural P6 (30B FP8 scale stack off by one, layer 18) and M2 (amd-quark expert index off by one, layer 24): 384
  mismatched slices each (every expert's three), fail. N7: the mode check's unit test (judge/tests/test_compare.py);
  N2–N6, N8, N8b, M1, M4 are A3's L0 and A5's Tier B.
- **Structural test on the real files** (reference kernels, CPU): 0 mismatched slices in `Qwen/Qwen3-30B-A3B-FP8` (the 30B FP8 structural test) and `amd-quark/Qwen3-30B-A3B-MXFP4`, routers bit-equal.
- **Quantization error** (file golden vs the bf16 checkpoint's fp32 golden; top-1, KL p50, ΔPPL):
  8B NVFP4 nvidia 93.2%, 3.9e-3, +3.9%; RedHatAI 94.2%, 2.6e-3, +3.7%; Yi30 MXFP4 90.9%, 7.3e-3, +7.5%; FP8 98.0%,
  2.3e-4, +0.3%. 30B NVFP4 94.2%, 2.0e-3, +1.3%; amd-quark MXFP4 95.4%, 1.3e-3, +0.8%; FP8 97.9%, 2.5e-4, +0.04%;
  the GGUF MXFP4_MOE vs Qwen3-30B-A3B-Instruct-2507 94.4%, 3.0e-3, −0.1%. Per file in
  `judge/goldens/<file>/v1-quantization-error.json`.
- **Suite** after the rebase onto M0.5 (`pytest core std plugins judge testing tools`, two A5000s): 1134 passed, 49 skipped, 22 min 13 s (965 / 47 / 21 min 10 s before the rebase). No A4 test is among the 25 slowest (`test_gguf_qwen35`'s golden, now streamed, takes 48 s as before); `--durations=25`:

  ```
   54.11s call  plugins/tests/test_supervisor.py::test_failed_starts_roll_back_and_an_exhausted_history_gives_up
   48.55s call  judge/tests/test_gguf_qwen35.py::test_the_gguf_golden_is_the_checkpoints
   47.86s call  plugins/tests/test_dflash.py::test_the_startup_profile_runs_the_draft_block
   30.53s call  tools/tests/test_propagate.py::test_cli_on_cpu
   30.16s setup plugins/tests/test_ggml_cuda.py::test_qlinear_real_tensors
   26.88s call  plugins/tests/test_supervisor.py::test_a_killed_engine_restarts_and_streams_continue_exactly
   26.12s call  plugins/tests/test_supervisor.py::test_a_poison_request_is_quarantined_without_rollback
   25.79s call  plugins/tests/test_supervisor.py::test_a_lone_poison_request_on_a_quiet_server_is_quarantined
   20.24s call  judge/tests/test_gguf_qwen35.py::test_the_judge_undoes_the_conversion_exactly
   19.88s call  plugins/tests/test_supervisor.py::test_unexplained_crashes_roll_back_to_the_previous_config
   17.20s call  plugins/tests/test_mtp.py::test_mtp_from_gguf_drafts_as_from_hf
   16.11s call  judge/tests/test_drafter.py::test_a_gguf_mtp_chain_passes_and_its_swapped_concat_fails
   15.94s call  plugins/tests/test_attention_kernels.py::test_kernel_matches_oracle[attention.paged_prefill-kernels.flashinfer-rows0-geometry0-dtype0]
   15.08s call  plugins/tests/test_attention_kernels.py::test_kernel_matches_oracle[attention.paged_prefill-kernels.flashinfer-rows0-geometry1-dtype1]
   14.91s call  plugins/tests/test_attention_kernels.py::test_kernel_matches_oracle[attention.paged_prefill-kernels.flashinfer-rows0-geometry3-dtype0]
   14.78s call  plugins/tests/test_attention_kernels.py::test_kernel_matches_oracle[attention.paged_prefill-kernels.flashinfer-rows0-geometry3-dtype1]
   14.68s call  plugins/tests/test_ggml_cuda_moe.py::test_mixed_types_in_one_layer
   14.65s call  plugins/tests/test_supervisor.py::test_a_hung_engine_is_killed_and_its_requests_resubmitted
   14.02s setup plugins/tests/test_ggml_cuda_moe.py::test_routings_and_sizes[q4_k]
   13.44s call  plugins/tests/test_attention_kernels.py::test_kernel_matches_oracle[attention.paged_prefill-kernels.flashinfer-rows0-geometry0-dtype1]
   13.23s call  plugins/tests/test_attention_kernels.py::test_kernel_matches_oracle[attention.paged_prefill-kernels.flashinfer-rows0-geometry1-dtype0]
   13.00s call  tools/tests/test_gpu_lock.py::test_first_lock_for_qwen3_0_6b
   11.65s setup plugins/tests/test_dflash.py::test_the_startup_profile_runs_the_draft_block
   11.39s call  plugins/tests/test_engine.py::test_engine_matches_hf_greedy_qwen3_0_6b
   11.20s call  judge/tests/test_q8_1_qwen35.py::test_the_floors_offsets_follow_the_untiled_weights[blk.0.ssm_out.weight]
  ```
- **For A5:** Tier B's decode step for an HF FP4 file needs `mmvq_step` from its HF layouts (it reads GGUF types
  today); the floors here serve Tier B on every class (A5000, 5080, 4090, PRO 6000) and the W4A4 report (5080).

**As built (A5).** Recorded deviations from rev 3 in step A5 (Tier B, plants, the W4A4 report, structural, bounded).

| item | rev 3 | as built |
|---|---|---|
| Tier B's decode step for HF files | `mmvq_step` from the file's types (GGUF only) | `gguf.mmvq_step` also takes an HF checkpoint: its quantized modules from the shard headers alone (`hf_quant.elems`, `parts`' rules on meta tensors), each as the ggml type `prepare` repacks it into (one with none, FP8, is an error), stacked by the Qwen expert pattern; the GPU structural test asserts it equals the smallest mmvq limit of the instances the engine binds. Steps: 8 for the NVFP4 files (sm_86, sm_120), amd-quark 7 on sm_86 and 8 on sm_120 (its MXFP4 experts), the MXFP4_MOE GGUF 7 on sm_86 and 8 on sm_89 |
| `quant.act` in the judge | — | `l2 --act a4\|a8` and `structure --act` set the plugin option (the W4A4 report, M4, the `fp4` spec's stacks) |
| mode check (A4) | every quantized call on the floor's `path/mode` | a quantized LM head's calls are counted apart (`head_calls`, reported only): **bug found in A5**, below. The trunk's calls are checked as before |
| N6 | a Python error at the first q/k/v mmq call (warm-up, before capture) | as built in A3: at load, when `bind_kernels` builds the first quantized linear's instance (before warm-up and capture) |
| M4 | L0 (sm_120a, `fp4`, `in` = 768) | also on Tier B (RTX PRO 6000, amd-quark 30B, its down `in` = 768): `a8` bit-identical to the clean run (the `q8_1` spec has no tail), `a4` NaN. There is no W4A4 floor for a 30B, so both `a4` runs (clean beside) are scored against the W4A8 floor, which the mode check refuses too |
| M0.1's wrong-type plants on the MXFP4_MOE file | fail | not applicable: they read a type as its same-size twin (Q4_K / Q4_0, IQ3_S / Q3_K), and neither MXFP4 (17 B per 32) nor Q8_0 (34 B per 32) has one, so they bind nothing; not run. The stale-q8_1 plants (dense and experts) ran |
| RTX PRO 6000 runs | — | a shared host: one GPU, state pools fixed at 4 GiB (`--state-memory-gib 4`, 43K tokens at 96 KiB: the whole prompt set) |
| bounded | 30B NVFP4 graphs vs eager on the 4090 (§4.8; acceptance 10 lists it with the locks) | run now with the rest of the bounded checks, with M0.2's padding plant as its calibration; acceptance 10's lock-time run stays A6's |
| W4A4 noise reference | — | the two W4A4-dynamic floors recorded again on the RTX PRO 6000 (local, not goldens): a second bf16 realization of the same emulation, scored against the recorded floors |

**A5 results** (records `evidence/tier-b_m06-lowp_<class>_2026-10-14.json`, `evidence/w4a4-report_m06-lowp_rtx-5080_2026-10-14.json`).
Runs at 9dd588d or f5a37b1. The second only counts a quantized LM head apart, so the HF files' runs, whose heads are
bf16, are the same at either, and the GGUF prefill verdicts are f5a37b1's.

- **Clock screen** (60 s of bf16 matmuls before the runs): RTX 5080 SM 2677 MHz median, 2602 min; RTX 4090 2670 / 2655;
  thermal slowdown 0% of samples on both (power cap 17% on the 5080, 0% on the 4090).
- **Tier B at `a8`** (W4A8 against the floor of the path taken; KL p99, its multiple of the floor's, top-1 vs the
  floor's): **all pass.**

  | file | class | prefill (mmq) | decode (mmvq, step) |
  |---|---|---|---|
  | nvidia 8B NVFP4 | A5000 | 1.20e-2 (0.96×), 98.87 / 98.69% | 1.17e-2 (0.82×), 98.76 / 98.78%, 8 |
  | | RTX 5080 | 1.30e-2 (1.04×), 98.83 / 98.69% | 1.26e-2 (0.89×), 98.81 / 98.78%, 8 |
  | nvidia 30B NVFP4 | A5000 | 4.65e-2 (0.94×), 98.37 / 98.36% | 4.92e-2 (0.94×), 98.30 / 98.31%, 8 |
  | | RTX PRO 6000 | 4.87e-2 (0.99×), 98.32 / 98.36% | 4.46e-2 (0.85×), 98.36 / 98.31%, 8 |
  | amd-quark 30B MXFP4 | A5000 | 3.64e-2 (1.01×), 98.50 / 98.51% | 3.63e-2 (1.01×), 98.51 / 98.50%, 7 |
  | | RTX PRO 6000 | 3.30e-2 (0.91×), 98.47 / 98.51% | 3.27e-2 (0.91×), 98.53 / 98.50%, 8 |
  | GGUF MXFP4_MOE | A5000 | 2.62e-2 (0.93×), 98.41 / 98.47% | 2.60e-2 (0.93×), 98.40 / 98.44%, 7 |
  | | RTX 4090 | 2.64e-2 (0.94×), 98.44 / 98.47% | 2.34e-2 (0.83×), 98.52 / 98.44%, 8 |

  Every trunk call on the floor's `path/mode` (prefill all `mmq/q8_1`, decode all `mmvq/q8_1`).
- **Bug found — the mode check and the LM head** (judge; fixed in f5a37b1, test `test_l2_engine.py` extended, failing
  before). Both GGUF prefill runs first failed on the mode alone: "10 calls on mmvq/q8_1, the floor emulates
  mmq/q8_1", with KL p99 0.93× / 0.94×. The MXFP4_MOE file's head is Q8_0 (the HF files' heads are bf16, uncounted),
  and in a scoring run the head never sees the step's rows: it runs on the prompt-logprob slices (`executor.
  _prompt_logprobs`, 256 rows at a time) and on the sampled rows, one per prompt finishing in the step. Counted while
  scoring, with the caller (A5000): 144 slices on `mmq`, 1 slice of ≤ 8 rows (a prompt cut at a step boundary) on
  `mmvq`, and 9 sampling calls on `mmvq`, whose logits are never scored. So at most 8 of the 32783 scored positions
  had their Q8_0 head on mmvq, whose q8_1 scale is fp16 where mmq's is f32. A4's mode check, extended to dense calls,
  counted all of them. The head's calls are now `head_calls` (here `{mmq: 144, mmvq: 10}`), reported, not checked.
  M0.2's GGUF prefill runs made the same mmvq head calls unseen (only expert calls were checked then).
- **Plants** (multiples of the floor's KL p99; predicted in §4.8):

  | plant | run | result | predicted |
  |---|---|---|---|
  | N2 nibbles swapped | A5000 8B NVFP4 prefill | fail, 513× (top-1 0%) | fail, gross ✓ |
  | N3 sub-scales permuted | A5000 8B NVFP4 prefill | fail, 741× (top-1 0%) | fail, gross ✓ |
  | N4 `scale2` by slot (mmq down) | A5000 30B NVFP4 | prefill fail, **101×**; decode pass (bit-identical to clean) | prefill 3–20×; decode pass: **outside the band** (gross) |
  | N5 `scale2` after SwiGLU | A5000 30B NVFP4 | prefill fail, 170×; decode pass (bit-identical) | prefill gross; decode pass ✓ |
  | N6 `op_params[3]` zero | RTX 5080 8B NVFP4 | `RuntimeError: … mmq would quantize nvfp4/hf activations at ggml_prec 40, its spec's mode 'q8_1' is 30`, at load | Python error ✓ |
  | | A5000, RTX 4090 | pass (0.96×, 1.02×) | pass ✓ |
  | N8 o_proj `scale2` × 1.25 | A5000 8B NVFP4 prefill | fail, 4.87× (top-1 97.91 < 98.44%) | fail 4–8× ✓ |
  | N8b × 1.1 | A5000 8B NVFP4 prefill | **pass**, 1.70× (top-1 98.51%, KL p50 1.37×) | 1.3–2.5×, fail only if > 2× ✓ (in band, under 2×) |
  | N9 × 1.25 at `a4` | RTX 5080 W4A4 report | KL p99 1.13× (sim 1.01×); top-1 93.8230% vs budget 93.8231%: **fails by one position** | pass: see the W4A4 report |
  | M1 e8m0 + 1 | A5000 amd-quark prefill; GPU structural | fail, 267× (top-1 0.9%); structural 18432 mismatched slices | fail, gross ✓ |
  | M4 tail left 0xFF | RTX PRO 6000 amd-quark prefill | `a8` bit-identical to the clean run; `a4` KL NaN, top-1 0.6% (clean `a4`: 93.45%) | L0 fail (NaN), `q8_1` spec pass ✓ |
  | M0.1 stale q8_1, dense | A5000 GGUF prefill | fail, 37.8× (fired 46 on mmq while scoring) | fail ✓ |
  | M0.1 stale q8_1, experts | A5000 GGUF prefill / decode | fail, 30.8× / 166× | fail ✓ |

  N4's cause: the plant hits every layer's down stack, and the 30B's per-expert down `scale2` spans up to 8.4× within
  a layer (layer 24: 4.6e-5 … 3.8e-4; median |log2| ratio to slots 0–7's mean 0.45), more than the simulation's.
- **W4A4 report** (acceptance 7; RTX 5080, `quant.act = a4`: prefill native W4A4-dynamic, every call `mmq/fp4`, against
  the W4A4-dynamic floor):

  | file | KL p99 (× floor) | KL p50 (×) | top-1 / floor | L2 rule |
  |---|---|---|---|---|
  | nvidia 8B NVFP4 | 0.549 (1.04×) | 0.99× | 93.78 / 94.07% (−0.29) | fail: top-1 below floor − 0.25 |
  | Yi30 8B MXFP4 | 2.03 (1.08×) | 1.02× | 88.21 / 88.75% (−0.54) | fail: top-1 below floor − 0.25 |
  | N9 (8B NVFP4, o_proj × 1.25) | 0.596 (1.13×) | 1.12× | 93.82 / 94.07% (−0.25) | fail by one position |

  The KL budgets hold on all three; top-1 sits in W4A4's noise. Between the candidate and the floor, 1000–1800
  positions flip each way (3–5% of all). The same emulation recorded on a second GPU (RTX PRO 6000) differs from the
  recorded floor by −0.06 (NVFP4) and −0.36 points (MXFP4), so the MXFP4 second floor would itself fail the
  0.25-point slack. The paired SEs (prompts as clusters) are 0.13–0.18 points. The bounded / drafter bar, max(0.25,
  2.5 SE), passes the NVFP4 run and N9 and leaves Yi30 0.10 point short. N9 is the blind spot as predicted: it sits
  nearer the floor than the clean run. Graphs vs eager at `a4` passes at decode batches 3, 5, 12, 20 (below).
  The pre-registered `a4` rule stays unmet (A4: 42× / 145×).
- **Structural test on the GPU** (A5000, `a8`: prepared ggml stacks read by gguf-py × `scale2`): 0 mismatched slices
  in the 30B NVFP4 and the amd-quark 30B, routers bit-equal; M1 fails it (18432). At `a4` on the RTX PRO 6000 (the
  `fp4` spec's stacks, read without their tail): 0 in both.
- **Bounded:** graphs vs eager, the 8B NVFP4 at `a4` on the RTX 5080, decode batches 3, 5, 12, 20: pass (graphs and
  eager runs identical at batch 12, elsewhere within the baseline's margins; with more than 8 rows, batches 12 and 20
  decode through `mmq/fp4`). The 30B NVFP4 at `a8` on the RTX 4090,
  same batches: pass. M0.2's padding plant there fails at batch 12 (expert path mmq: top-1 95.15% beyond 0.79
  points of the baseline) and passes at 5 (mmvq), as predicted. Chunked (100) vs one-shot prefill, the 8B NVFP4 at
  `a8` on the RTX 5080: pass, as do prefix on/off × depth 1/0 in the same run.
- **For the lead** (decided: "Decisions during Gate A", A5's three items):
  1. The W4A4 report's top-1 at the L2 slack (floor − 0.25 points). The report's two clean runs fail it, by 0.04 and
     0.29 points past the slack, and N9 fails by one position (predicted pass). A second realization of the floor
     itself moves top-1 by up to 0.36 points. Options: (a) record the report as it stands (KL within budget, top-1
     inside W4A4's measured noise; acceptance 7 then reads "N9 indistinguishable from the clean run"), (b) judge the
     report with the bounded / drafter rule, max(0.25, 2.5 paired SE): NVFP4 and N9 pass, Yi30 stays 0.10 short.
     Either way this changes how acceptance 7 is read after seeing results, so it is yours to decide.
  2. N8b passes (1.70×). It is in its predicted band and under the 2× fail line, as §4.8 words it; the brief's "N8b
     fails at its band" does not hold.
  3. The mode-check fix changed a judge rule after a result, so it is for you to review. A quantized LM head is no
     longer checked; its calls are reported. The alternative was to keep the rule and count both GGUF prefill runs
     as failed: 9 of the 10 offending calls were unscored sampled rows, and 1 was a scored slice of ≤ 8 rows.
     Checking the head instead would need the floor to emulate the engine's logprob slicing.

**As built (A6).** Recorded deviations from rev 3 in step A6 (fits, bars, serving, locks).

| item | rev 3 | as built |
|---|---|---|
| bar files (D9) | GGUFs from the same HF files, NVFP4 by the pin's converter | `convert_hf_to_gguf.py` at `cbe28bb3` with `--outtype bf16` on the two nvidia files: 252 / 336 NVFP4 tensors, embedding and head BF16, norms and scales F32; 5.96 / 16.87 GiB of tensors, the HF files' bytes plus the per-tensor scale tensors. The MXFP4_MOE GGUF is benched as is. amd-quark has no bar (not in acceptance 9), so `bench/hf_to_gguf.py` was not needed |
| bars | llama-bench at `GGML_CUDA_MMQ_PREC=q8`, the default beside | as specified, with llama.cpp built at the pin per class (`build-rtx-5080` also serves the RTX 5090: sm_120, same container image). Holon in process at the new locks' flags (`--memory-fraction 0.9 --max-seqs 32 --max-step-tokens 2048 --max-model-len 16384`). Holon at `quant.act = a4` is reported beside on sm_120. On sm_89 `q8` is llama.cpp's default, so no second run there |
| serving | Holon vs `llama-server` on the 5080, 8B NVFP4 | `bench/run.py` takes one model for every backend, but Holon serves the HF file and llama-server the converted GGUF. So there is one run per backend (two rounds each, not interleaved): llama-server at `q8` and at its default, `LLAMA_SERVER_CTX` = Holon's measured KV tokens (51472), `--max-batch 32 --max-step-tokens 2048`, the decode_b1 / decode_b8 / agent traces |
| new locks' flags | — | one deployment for the four: acceptance 8's (`--memory-fraction 0.9 --max-seqs 32`, bf16 KV) with `--max-step-tokens 2048 --max-model-len 16384` |
| lock order | — | the 8B NVFP4 was locked on the RTX 4090 first (`--new-model`). The RTX 5080's lock then starts from it, so its snapshot differs from the 4090 lock's (another class), as for any second class |
| **bug found** — a lock's stored pools across classes | — | `holon_tools` seeded the checks' state pools with the start lock's stored snapshot `pool_bytes` even when that lock is another class's (`nearest_class`). The RTX 4090 lock's 12.78 GiB pools ran the RTX 5080 tune out of memory. Fixed in e6c9f53: they seed only on the lock's own class, and from another class the pools are measured. `test_propagate.py::test_cli_on_cpu` is extended and failed before the fix. It never showed before because every second class so far had more memory than the first (Qwen3-8B bf16: 4090, then 5090), where the carried pools merely went unmeasured |
| commits | — | measured at 74bfabb (the code of A5 + the decisions); the RTX 5080 8B NVFP4 lock at e6c9f53 (the tools fix, outside every lock's units) |

**A6 results** (records `evidence/{fit,bars}_m06-lowp_<class>_2026-10-14.json`, `evidence/bench_m06-8b-nvfp4-*_rtx-5080_2026-10-14.json`, `evidence/locks_m06-lowp_2026-10-14.json`).

- **Clock screen** (60 s of bf16 matmuls before each desktop's runs): SM median / min 2722 / 2610 MHz (RTX 5080),
  2670 / 2670 (RTX 4090), 2602 / 2400 (RTX 5090). Thermal slowdown was 0% of samples on all three. The power cap was
  active 54% (5080), 0% (4090) and 91% (5090, at its 575 W limit) of the time.
- **Fits** (acceptance 8; bf16 KV, `max_seqs` 32, `memory_fraction` 0.9; a prompt of 16128 tokens served on each):

  | file | card | weights | KV tokens | predicted (§4.9) | acceptance |
  |---|---|---|---|---|---|
  | nvidia 8B NVFP4 | RTX 5080 | 5.96 GiB | 51472 | 52.8K | ≥ 32K ✓ |
  | nvidia 30B NVFP4 | RTX 4090 | 16.87 | 35280 | 35.0K | ≥ 32K ✓ |
  | GGUF MXFP4_MOE | RTX 4090 | 15.90 | 42096 | 44.2K–32.5K | ≥ 32K ✓ |
  | amd-quark 30B MXFP4 | RTX 4090 | 17.24 | **24608** | 31.3K | recorded |

  amd-quark is 6.7K tokens short of its prediction. At the free-memory read (A5000, same flags) its allocator holds
  0.98 GiB reserved but not allocated that `empty_cache` cannot return, against 0.31 GiB for the nvidia 30B. Both
  run the same expert path, and the one difference is amd-quark's bf16 attention, so segments are probably left
  fragmented by loading and merging those dense weights. Allocated memory beyond weights and pools is 0.19 GiB for
  both. A run with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` gave identical numbers, and I did not confirm the option took effect. Not fixed in A6 (amd-quark has no lock).
- **Bars** (acceptance 9; medians of repetitions 2–5, tokens/s, Holon / llama-bench at `q8` on the same bytes):

  | file | card | pp512 | tg128 | tg128 @ d256 | ratio pp / tg |
  |---|---|---|---|---|---|
  | 8B NVFP4 | RTX 5080 | 6771 / 6445 | 144.8 / 146.3 | 144.1 / 145.2 | 1.05 / 0.99 ✓ |
  | 8B NVFP4 | RTX 4090 | 10475 / 10427 | 138.5 / 136.3 | 137.8 / 135.5 | 1.00 / 1.02 ✓ |
  | 30B NVFP4 | RTX 4090 | 9248 / 8775 | 214.0 / 213.8 | 213.2 / 212.5 | 1.05 / 1.00 ✓ |
  | GGUF MXFP4_MOE | RTX 4090 | 10904 / 10762 | 224.4 / 222.6 | 223.5 / 221.0 | 1.01 / 1.01 ✓ |
  | 8B NVFP4 | RTX 5090 (recorded) | 12718 / 11764 | 231.8 / 240.2 | 230.8 / 238.5 | 1.08 / 0.97 |
  | 30B NVFP4 | RTX 5090 | 11581 / 10213 | 307.1 / 320.6 | 306.1 / 317.5 | 1.13 / 0.96 |
  | GGUF MXFP4_MOE | RTX 5090 | 14483 / 12878 | 329.0 / 343.8 | 326.9 / 340.1 | 1.12 / 0.96 |

  Every acceptance row passes, inside the predicted 0.95–1.1× (§4.10) except the RTX 5090's prefill at 1.12–1.13×.
  Beside them on sm_120, pp512 for llama.cpp's default (native W4A4 prefill) vs Holon at `a4`: 8B 13363 vs 14394
  (5080), 21037 vs 24821 (5090); 30B NVFP4 17108 vs 19487 (5090). The MXFP4_MOE file at llama.cpp's default runs
  16257 (5090). tg128 is the same in both modes (decode is mmvq q8_1).
- **Serving** (RTX 5080, 8B NVFP4; out tok/s, TTFT p50 ms; Holon vs llama-server at `q8` / at its default):
  decode_b1 141.3, 41.9 vs 137.2, 60.3 / 138.5, 41.8; decode_b8 780.7, 298 vs 644.1, 497 / 680.9, 336; agent 77.1,
  68 vs 75.2, 208 / 75.3, 171. Holon is at or above llama-server on every trace in both of its modes.
- **Locks** (acceptance 10; L2 against the W4A8 mmq floor; tune picked `kernels.triton_attn` for decode attention
  on both classes):

  | lock | KL p99 (floor) | top-1 | verdict |
  |---|---|---|---|
  | rtx-5080 nvidia 8B NVFP4 (from the 4090's) | 1.28e-2 (1.24e-2) | 98.85% | approve, PASS |
  | rtx-4090 nvidia 8B NVFP4 (new model) | 1.26e-2 (1.24e-2) | 98.78% | approve, PASS |
  | rtx-4090 nvidia 30B NVFP4 (new model) | 4.75e-2 (4.92e-2) | 98.29% | approve, PASS |
  | rtx-4090 MXFP4_MOE GGUF (new model) | 2.64e-2 (2.81e-2) | 98.44% | approve, PASS |

  Snapshot ×3 run-identical for all four (130 steps each; the 8B NVFP4 on both classes). Graphs vs eager on the 30B
  NVFP4 at its lock's flags (RTX 4090), decode batches 3, 5, 12, 20: pass.
- **Every development-class lock propagated** to this tree (rebased on `main`): rtx-5080 30B UD-IQ3_XXS, 8B
  Q4_K_M, 8B UD-IQ3_XXS, 27B UD-IQ3_S; rtx-4090 Qwen3-8B bf16, 30B Q4_K_M, 27B UD-Q4_K_M. All seven have a
  snapshot trace byte-identical to their lock's, and L2 passes.
- **Approvals:** every verdict passes and no existing snapshot moved, so the eleven locks are approved under the
  delegation at Gate A's close: the four new ones and the seven propagated, each new `N.lock` with its snapshot.
- **Suite and CI** at e6c9f53: full GPU suite (`pytest core std plugins judge testing tools`, two A5000s) 1134
  passed, 49 skipped, 21 min 53 s (as at A4: no test added; three extended, two in A5 and one in A6); `scripts/ci/run_all.sh` green.

**Decisions during Gate A (2026-10-13).**
- **Locks are approved once, at Gate A's close.** A0 + A1 were propagated on every development class (rebased on
  `main` at the M0.4 Gate A merge) without `--approve`: all seven snapshot traces byte-identical, L2 at the M0.4
  values. A2 onward changes `std.weights.*` hashes again, so approving per step would repeat approvals for
  byte-identical results; the delegated approval (passing verdicts only) is taken at the close.
- **A0's bar:** Qwen3-8B Q4_K_M on the RTX 5080 pp512 8356 / tg128 155.05 vs M0.1's 8291 / 155.0 (+0.8%, +0.0%);
  llama-bench at `cbe28bb3`, the new baseline, 7906 / 158.3 (`evidence/bars_qwen3-8b-gguf-cbe28bb3_*`). L0 at A0 + A1:
  sm_120a 242 passed, sm_89 241 passed, memcheck 0 errors.
- **Quark's oracle:** the real `amd-quark/Qwen3-30B-A3B-MXFP4` agrees exactly with `quark.Pack_fp4` and with gguf-py on
  llama.cpp's repack over all 18432 modules (`evidence/owner_qwen3-30b-a3b-mxfp4-amd-quark_cpu_*`).
- **A5's three items (lead, 2026-10-14):**
  1. **The W4A4 report is recorded as is; no rule change.** Acceptance 7 asks that the files be scored, not that
     they pass. The KL budgets hold (1.04×, 1.08×). Top-1 is below the 0.25-point slack, at −0.29 (NVFP4) and −0.54
     (MXFP4), by amounts inside the measured floor-to-floor noise: the floors re-recorded on another GPU differ by
     −0.06 / −0.36. N9's predicted pass missed by one position of 32783, consistent with the documented blind spot
     (N9 sits closer to the floor than the clean run). `a4` stays opt-in and report-only; the pre-registered default
     rule is not met.
  2. **N8b passes at 1.70×**, inside its band and under §4.8's 2× fail line: as designed.
  3. **A judge rule changed after a result** (for the Gate A panel and the owner): the `head_calls` mode-check change
     is accepted. It narrows the scope of a check M0.6 itself added in A4 (the Q8_0 head never sees the step's row
     count, and sampled rows are never scored); no metric bar was loosened.
