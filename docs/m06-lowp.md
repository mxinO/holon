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
(`<user>/m04-spec`, which changes `build.py`, `load_weights` and `api/__init__.py`) merges to `main` first; Gate A's
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
  | M3 | Quark `weight_scale` decoded as e4m3, not e8m0 | gate; Tier A | gate fail (NMSE ≫ 0.1) |
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
