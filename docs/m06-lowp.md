# M0.6 — FP8 (block), NVFP4 and MXFP4 checkpoints of Qwen3-8B and Qwen3-30B-A3B: design and implementation plan

Status: **revision 1, draft for the challenge panel** (2026-10-13). Scope: plan §2 row M0.6. Builds on design §5.5
(Scheme / Layout / QTensor), §5.6 (borrowed kernels), M0.1 (`kernels.ggml_cuda`, the judge's tiers and the q8_1
emulator), M0.2 (`std.moe`, `linear_experts@1`, the stacking rule D7, `kernels.triton_experts` D8). Research behind
it: the M0.6 research note (checkpoints, numerics, kernels per GPU class; local, not tracked), re-checked against
source where cited below.

## 1. Goal and acceptance

Serve the low-precision HF checkpoints of Qwen3-8B (dense) and Qwen3-30B-A3B (MoE) — FP8 with 128×128 block scales,
NVFP4 and MXFP4 — on consumer cards first (RTX 5080 / 5090 sm_120, RTX 4090 sm_89, RTX 3090 / A5000 sm_86), with the
RTX PRO 6000 (sm_120) for what fits nowhere else. Acceptance:

1. **Targets load and serve** (D1): `Qwen/Qwen3-8B-FP8`, `nvidia/Qwen3-8B-NVFP4`, `RedHatAI/Qwen3-8B-NVFP4`,
   `Yi30/Qwen3-8B-MXFP4`, `Qwen/Qwen3-30B-A3B-FP8`, `nvidia/Qwen3-30B-A3B-NVFP4`, `amd-quark/Qwen3-30B-A3B-MXFP4`,
   through the existing `models/qwen3` and `models/qwen3_moe` with **no change to either model package**
   (`git diff plugins/holon_plugins/models` empty). Each answers the M0.1 chat set sensibly.
2. **Core diff:** the per-slice `adopt` of quantized stack slices only (§4.2, ≤ 20 lines), proven by a toy layout in
   `core/tests/test_stacking.py`; denylist clean; core ≤ 6000 code lines (3324 today, `scripts/ci/check_core_loc.py`).
3. **Lossless repack** (D6): for every quantized tensor of the five FP4 targets, gguf-py's dequantization of our
   ggml blocks × the global scale equals `std.weights.hf`'s reference dequantization **bitwise** in fp32; for the two
   8B NVFP4 files our blocks are **byte-equal** to llama.cpp's converter output at the vendored pin.
4. **L0** for every new kernel path on sm_86, sm_89 and sm_120a against its emulated reference (§4.8), memcheck clean;
   the reference dequantization equals the format owners' decoders (§4.8, D8).
5. **L2** for each target: Tier A (CPU fp32, production source + reference kernels) under the 1e-4 sanity ceiling;
   Tier B against the floor of the activation mode and path the kernel ran (D9), on the classes of §4.8's table;
   every plant of §4.8 fails or passes as predicted there; quantization error vs the bf16 checkpoint reported per
   format and per activation mode.
6. **Memory:** on the RTX 5080 each 8B format serves a 32K-token context with bf16 KV at `max_seqs` 32 (§4.9); the
   30B FP4 targets serve 32K on the RTX 4090; the 30B FP8 fit on the RTX 5090 is measured and recorded (≈ 10K tokens
   predicted, no bar).
7. **Speed** (D10), medians of repetitions 2–5 on both sides: FP4 pp512 and tg128 ≥ 0.8× llama-bench on the same
   bytes (8B on rtx-5080, rtx-4090, rtx-a5000; 30B on rtx-4090 and rtx-5090); FP8 tg128 ≥ 0.8× llama-bench Q8_0 of
   the same checkpoint (8B on rtx-5080 and rtx-4090), FP8 pp512 recorded against its prediction; serving vs
   `llama-server` on the 5080 for the 8B NVFP4.
8. **Locks** (rtx-5080: the three 8B targets of `Qwen`, `nvidia`, `Yi30`; rtx-4090: the same plus the 30B NVFP4 and
   MXFP4), each snapshot run-identical over 3 fresh engines; graphs vs eager bounded on the 30B NVFP4 decode.
9. **Reuse:** every existing lock (Qwen3-8B bf16 and GGUF, Qwen3-30B-A3B GGUF, Qwen3.8-27B GGUF) advances by
   `holon propagate` with one approval covering all of them (qlinear@2, the `act_quant` fact, the ggml pin, D3/D6);
   L2 passing; bf16 snapshots byte-identical.

Non-goals: checkpoint-faithful W4A4 with static input scales (CUTLASS / FlashInfer, §7); ModelOpt per-tensor FP8 and
`MIXED_PRECISION`; weight-only (A16) FP4 kernels; gpt-oss; FP8 KV from checkpoint scales; Qwen3.5/3.8 low-precision
builds.

## 2. Starting point

- **Checkpoints** (research note §1, headers read): every quantized Linear is `[out, in]` with all dimensions
  multiples of 128; experts are per-expert 2-D tensors (`mlp.experts.{e}.{gate,up,down}_proj.*`); embeddings,
  `lm_head` and routers are dense (bf16, or F32 in `Qwen/Qwen3-30B-A3B-FP8`). Parts per module:
  - FP8 (`quant_method: fp8`): `weight` F8_E4M3 `[out, in]`, `weight_scale_inv` BF16 or F32 `[out/128, in/128]` —
    a dequant *multiplier* despite its name; activations declared dynamic.
  - NVFP4: ModelOpt `weight` U8 `[out, in/2]`, `weight_scale` F8_E4M3 `[out, in/16]`, `weight_scale_2` F32 `[]`,
    `input_scale` F32 `[]` (plus `k_scale`/`v_scale` = 1.0 on attention); compressed-tensors `weight_packed`,
    `weight_scale`, and the **reciprocals** `weight_global_scale`, `input_global_scale` `[1]`.
  - MXFP4: compressed-tensors `weight_packed` U8 `[out, in/2]`, `weight_scale` U8 (e8m0) `[out, in/32]`; Quark
    `weight`, `weight_scale` with the same shapes (`amd-quark` 30B: experts only, attention bf16).
  - Global scales are tied where layers fuse: ModelOpt 8B shares `weight_scale_2` and `input_scale` across q/k/v and
    gate/up; the 30B's gate/up share `weight_scale_2` per expert (values differ across experts); the 30B's down
    `input_scale` differs per expert (research note §1.2).
- **Holon today:**
  - `qlinear@1` takes one `blocks` tensor (`std/holon_std/ops/kernels.py:8`); `Linear.adopt` stores
    `q.parts["blocks"]` only (`std/holon_std/ops/units.py:90-95`); `ExpertLinear.adopt` refuses any parts other than
    `{"blocks"}` (`std/holon_std/moe/units.py:109-115`). `linear_experts@1` already takes `parts` by role
    (`std/holon_std/moe/kernels.py:8-14`).
  - A quantized tensor under a stacking rule is a load error (`core/holon/core/build.py:219-220`), the choice M0.2
    D7 left to M0.6.
  - `std.weights.hf` yields tensors one by one in file order (`std/holon_std/weights/hf/source.py:46-50`); it has no
    notion of quantization. The core's `Dialect`, `Converter` and `SchemeImpl` (`core/holon/api/quant.py:55-79`) are
    registered nowhere.
  - `kernels.ggml_cuda` serves 18 types (`plugins/holon_plugins/kernels/ggml_cuda/plugin.py:33-35`; its paths below
    are relative to that directory). The vendored
    tree already has `GGML_TYPE_MXFP4 = 39` and `GGML_TYPE_NVFP4 = 40` (`vendor/ggml/include/ggml.h:429-430`), their
    blocks (`vendor/ggml/src/ggml-common.h:214-226`), mmq / mmvq instances and the sm_120a build
    (`extension.py:25`).
- **The vendored ggml on sm_120, as patched today:**
  - mmq takes its activation precision from `op_params[3]` (`vendor/ggml/src/ggml-cuda/mmq.cu:118-133`; the env
    override is nulled at `:96`). Our shim zeroes `op_params` (`csrc/api.cu:58-71`), i.e. `GGML_PREC_UNDEFINED`,
    which for NVFP4/MXFP4 on Blackwell means **native W4A4** (`mmq.cu:128-132`).
  - Native W4A4 quantizes activations from **f32 only** (`quantize.cuh:29`), and our own patch asserts it
    (`vendor.patch:46` → `mmq.cu:184`). Holon passes bf16 activations to gate/up and every dense `qlinear`, so a
    W4A4 call there would **abort the process**; MoE down (f32 SwiGLU input) would silently run W4A4.
  - W4A4 also allocates a per-row activation scale for NVFP4 (`mmq.cu:192-195`), which
    `holon_ggml_workspace_bytes` does not size (`csrc/api.cu:138-145`).
  - mmvq is always q8_1 (`mmvq.cu:49-50`): decode is W4A8 on every class.
  - The vendored mmvq applies NVFP4 per-tensor / per-expert scales in its fused path (`x_scale`, `gate_scale`, one
    f32 per expert; `mmvq.cu:669-675, 702-709, 1506-1516`).
  - Upstream fixed MMQ out-of-bounds reads (`fc9ce6b9d`, 2026-10-08) and host configs ignoring the src1 precision
    (`8a1a9b512`, 2026-10-09) after our pin `51ce9c11` (2026-10-06); the second is exactly the FP4-at-Q8 path on
    Blackwell.
- **Triton:** `fp8e4nv` is rejected below sm_89 (design §5.6); `kernels.triton_experts` is our bf16 grouped GEMM with
  the D8 grid (`plugins/holon_plugins/kernels/triton_experts/kernel.py:34-83`).

## 3. Decisions

**How references are chosen** (M0.2's rule): oracle = the format owner's decoding; kernel source = the best
maintained implementation of the exact workload; bar = the engine users run for it on the card.

- **D1 — Acceptance targets.** Official files where they exist; each dialect the targets need is exercised once:

  | | Qwen3-8B | Qwen3-30B-A3B |
  |---|---|---|
  | FP8 block | `Qwen/Qwen3-8B-FP8` (8.79 GiB) | `Qwen/Qwen3-30B-A3B-FP8` (30.22 GiB on disk, ≈ 29.1 loaded: F32 embed/head cast to bf16) |
  | NVFP4 | `nvidia/Qwen3-8B-NVFP4` (ModelOpt, 5.96) and `RedHatAI/Qwen3-8B-NVFP4` (compressed-tensors) | `nvidia/Qwen3-30B-A3B-NVFP4` (ModelOpt, 16.85) |
  | MXFP4 | `Yi30/Qwen3-8B-MXFP4` (compressed-tensors, W4A4, 5.76) | `amd-quark/Qwen3-30B-A3B-MXFP4` (Quark, experts only, 17.22) |

  No MXFP4 Qwen3 comes from Qwen, NVIDIA or OpenAI: `Yi30` is a community test upload (a format fixture, not a
  quality reference) and `amd-quark` is AMD's. Compressed-tensors FP8-block (`RedHatAI/Qwen3-8B-FP8-block`) is covered
  by the tiny-model fixtures only. **For the user:** keep these, or move the MXFP4 targets to AMD's official
  Qwen3.5/3.8 builds after their low-precision recipes exist (§7).
- **D2 — Dialects live in the weight source; one source layout per format.** `std.weights.hf` reads
  `quantization_config` (`config.json`, else `hf_quant_config.json`, which then joins the identity hash) and picks one
  of four internal dialects by `quant_method` / `format`: `fp8` (DeepSeek/Qwen), compressed-tensors
  (`float-quantized` block, `nvfp4-pack-quantized`, `mxfp4-pack-quantized`), `modelopt` (`quant_algo: NVFP4`), `quark`
  (fp4 group 32, e8m0). Anything else is a load error naming it. It yields **one `QTensor` per quantized module under
  `<module>.weight`**, so the model plugins' existing weight rules map it unchanged, with parts normalized:

  | scheme (`elem`) | parts (per 2-D weight) | normalization |
  |---|---|---|
  | `fp8_e4m3` (scale `fp32`, 128×128) | `weight` u8 (e4m3 bits) `[out, in]`, `scale` f32 `[out/128, in/128]` | BF16 scales upcast (exact) |
  | `nvfp4` (e4m3 per 16; f32 per tensor, level 2) | `weight` u8 `[out, in/2]`, `scale` u8 (e4m3 bits) `[out, in/16]`, `scale2` f32 `[]`, `input_scale` f32 `[]` if present | compressed-tensors' globals inverted at load, `1/x` in fp32, as llama.cpp's converter does (`conversion/base.py:1186-1209` at `cbe28bb3`); ≤ 0.5 ulp from CT's own division |
  | `mxfp4` (e8m0 per 32) | `weight` u8 `[out, in/2]`, `scale` u8 `[out, in/32]` | none |

  Layout name `hf`; the layout's `act` param records the declared activation scheme (D3). ModelOpt's
  `k_scale`/`v_scale` must be 1.0 and are dropped (any other value: a load error; the KV format is the deployment's,
  M0.3 D6). The source reads every shard's header first and yields a module's parts together (they may sit in
  different shards). Why not registered dialect plugins: a weight source is built from a path alone
  (`core/holon/core/assembly.py:26`), so reaching a registry would be a core seam with no second consumer. The core's
  unused `Dialect` / `Converter` / `SchemeImpl` stay untouched (§7).
- **D3 — Contracts.** `std.ops` replaces `qlinear@1` by **`qlinear@2`**: instance `(x [T, in], parts) → y [T, out]`,
  `parts` the unit's stored parts by role as in `linear_experts@1` (breaker pays: `kernels.ggml_cuda` and
  `testing.ref_kernels` move to @2 in the same change). Both ops gain, additively:
  - an optional instance method **`prepare(parts) → parts`**, called once by the unit after binding; later calls
    receive its result. It is the design §5.5 converter, owned by the plugin that owns the destination layout, and must
    be lossless (L0, acceptance 3);
  - a fact **`act_quant` ∈ {`a4`, `a8`, `a16`}**: the lowest activation precision the kernel may use. The unit sets it
    to the layout's declared value — W4A4 checkpoints `a4`, W8A8 `a8`, ggml layouts `a8`, dense `a16`; weight-only FP4
    files `a8`, as llama.cpp's precision policy runs them (`src/llama-model.cpp:1270` at `cbe28bb3`) — raised, never
    lowered, by the deployment's plugin option `quant.act` (`a8` | `a16`), as `kv.format` is read
    (`std/holon_std/mixers/gqa/units.py:73-79`). `qembedding@1` is unchanged.

  A provider registers **one `KernelSpec` per activation mode**; `config["act"]` names the mode (`bf16`, `fp8_g128`,
  `q8_1`, `fp4`) and is recorded in the lock's site entry (`core/holon/core/build.py:67-68`), so the judge reads which
  floor applies. Rejected: one spec choosing its mode from the SM inside the instance — the lock would not show the
  numerics.
- **D4 — Merges by shape.** `Linear.merge` and `ExpertLinear.merge` concatenate every part whose leading (per expert:
  second) dimension is `out / g` for one `g` dividing every unit's `out` (weights, block scales); per-tensor and
  per-expert scalars (`scale2`, `input_scale`) must be bit-equal across the merged units, else the merge is refused —
  dense units stay separate (`std/holon_std/ffn/units.py:23-30`, `gqa/units.py:146-147`), MoE gate/up is a startup
  error naming the layer (M0.2 D4). Every target satisfies the ties (checked from headers and values in step 0).
- **D5 — Per-expert stacking: per-slice `adopt`** (closes M0.2 D7's question). A quantized slice under a stacking rule
  goes to the owning unit's `adopt(ctx, name, q, index=e)`; the core keeps the bookkeeping it has. Rejected: the
  source grouping slices into a 3-D `QTensor` — it would have to guess the stack from names (`experts.{e}`) that the
  recipe already declares with `stack="e"`, and it would hold a whole stack in host memory.
- **D6 — FP4 runs on `kernels.ggml_cuda` through a lossless repack** (zero new kernel family; the same code as the
  llama.cpp bar):
  - `TYPES` gains `"mxfp4": 39, "nvfp4": 40`; the provider's `prepare` repacks `nvfp4/hf` and `mxfp4/hf` into ggml
    blocks (`blocks` `[out, row_bytes]`, stacks `[E·out, row_bytes]`) and keeps `scale2` (§4.5).
  - The NVFP4 global scale is applied **in f32 before SwiGLU** on every path: mmvq through the vendored fused
    `x_scale` / `gate_scale`; mmq by our post-multiply (dense: folded into the existing f32 → bf16 copy; experts: our
    f32 SwiGLU kernel takes per-row gate/up scales gathered by expert id, and down multiplies its f32 output by
    `scale2[ids]`).
  - Activation modes are explicit: the shim sets `op_params[3]` from the spec (`GGML_PREC_Q8` = 30 or `GGML_PREC_Q4`
    = 40) and never relies on the zero default; the `fp4` spec (sm_120 only, FP4 layouts with `act_quant = a4`)
    casts bf16 activations to f32 before mmq (exact; the vendored quantizer reads f32 only) and its workspace adds the
    per-row scale. mmvq stays q8_1 in both specs.
  - The vendored pin moves from `51ce9c11` to `cbe28bb3` (2026-10-13), which contains `fc9ce6b9d` and `8a1a9b512`;
    `vendor.patch` regenerated, `vendor.sh --check`. The same commit is the llama-bench baseline of D10. Rejected:
    cherry-picking ≈ 450 lines of upstream into our patch file.
  - Rejected for M0.6: a W4A16 (Marlin-style) provider. On GeForce, int8 MMA runs ≈ 4× bf16 (design §5.6), so W4A8
    prefill should beat W4A16 at equal bytes; decode is bandwidth-bound either way; W4A8's activation noise is judged
    exactly as for GGUF.
- **D7 — FP8 runs on our own Triton**, in `kernels.triton_experts` renamed **`kernels.triton_gemm`** (no lock names it;
  one module holds the bf16 grouped GEMM, an FP8 grouped path on the same D8 grid and an FP8 dense GEMM, sharing one
  FP8 K-loop and one activation-quantization kernel). Two specs per op:
  - `fp8_g128` (sm ≥ 89): per-token, per-128-column e4m3 activations (`a_s = max(amax, 1e-10)/448`), `tl.dot` of e4m3
    tiles with f32 accumulation, `a_s · s_w` applied per 128-K step — the DeepSeek-V3 recipe, as SGLang's / vLLM's
    `_w8a8_triton_block_scaled_mm` and FreeToken's `fp8_block_linear.py` (studied, not copied);
  - `bf16` (sm ≥ 80): weights read as u8 and decoded to bf16 exactly by bit operations (e4m3 ⊂ bf16, subnormals
    included), bf16 × bf16 `tl.dot`, the block scale applied per 128-K step in f32.
  On sm_86 only `bf16` exists; on sm_89/120 `fp8_g128` wins by priority unless `quant.act = a16`. **Scope change
  against plan §10** (FP8 W8A16 on sm_86 was v1): it is the same kernel with a different operand path, and the A5000
  is the L2 host — for the user's approval. Not vendored: the borrowable kernels are ≈ 130 lines inside files that
  import half of vLLM, and none has a W8A16 path. CUTLASS sm120 blockwise (87b / 87c) is the later performance step.
- **D8 — Reference dequantization** (`std/holon_std/weights/hf/reference.py`, `[reference]` in the manifest so its
  hash pins the oracle): FP8 `q · s`; NVFP4 `e2m1(code) · e4m3(scale) · scale2`, low nibble = even element (ModelOpt
  `8101d128` `modelopt/torch/quantization/qtensor/nvfp4_tensor.py:322-326, 356-357`); MXFP4 `e2m1(code) · 2^(e − 127)`
  (OCP MX v1.0). Cross-checked in step 1, fp32, per tensor of every target: transformers' `Fp8Dequantize`
  (`integrations/finegrained_fp8.py`, the judge's 5.6.0.dev0) bitwise; compressed-tensors' `NVFP4PackedCompressor` /
  `MXFP4PackedCompressor` (0.17.0) within 1 ulp; **ModelOpt NVFP4 has no transformers oracle**, so its files are
  checked against ModelOpt's own `NVFP4QTensor.dequantize` (test-only, within 1 ulp: it forms `scale·scale2` first)
  and, independently, against gguf-py's NVFP4 dequantization of llama.cpp's converter output × `weight_scale_2`
  (bitwise); Quark MXFP4 against gguf-py on llama.cpp's `repack_mxfp4_blocks` (bitwise).
- **D9 — One floor per activation mode** (research note §4 item 4), all HF bf16 on the dequantized weights with the
  M0.2 MoE mirror, the mode emulated on every quantized linear and expert stack: **A16** (none); **W8A8** (e4m3 per
  token per 128); **W4A8** (the existing q8_1 emulator, per path); **W4A4-dynamic** (ggml's Blackwell quantizer:
  NVFP4 per-row `amax/(6·448)` and a UE4M3 per 16 chosen among 5 codes by squared error, `quantize.cu:199, 250-290`;
  MXFP4 e8m0 per 32 with round-to-nearest `log2`, `:122-140`); **W4A4-static** (ModelOpt's `input_scale`, NVFP4 only;
  information: no provider). A run is judged against the floor of the mode and path every quantized call took,
  counted per instance as M0.2's paths; a mismatch fails the verdict before scoring.
- **D10 — Bars are llama.cpp's, on the same bytes where a format exists.** FP4: GGUFs from the same HF files — NVFP4
  by the pin's `convert_hf_to_gguf.py`; MXFP4 (which the pin's converter does not take for Qwen3) written from our
  repack by a bench tool, its blocks byte-equal to the pin's `repack_mxfp4_blocks` — run by llama-bench at the pin.
  llama.cpp defaults to W4A4 mmq on Blackwell for these types, as our `a4` targets do. FP8: llama.cpp has no FP8
  type; the comparator is its Q8_0 of the same checkpoint (`--outtype q8_0`), the nearest bytes (8.5 vs ≈ 8.0 bits);
  the 30B FP8 has none (Q8_0 does not fit 32 GB with any KV).

## 4. Design

### 4.1 Packages

| package | change |
|---|---|
| `core/holon` | per-slice `adopt` of quantized stack slices (§4.2) |
| `std/holon_std/weights/hf` | `quant.py`: the four dialects (D2); `reference.py`: reference dequantization (D8); grouped yield, identity incl. `hf_quant_config.json` |
| `std/holon_std/ops` | `qlinear@2`; `Linear` keeps parts, merges by shape (D4), `act_quant`, calls `prepare` |
| `std/holon_std/moe` | `ExpertLinear`: parts, per-slice adopt (D5), merges by shape, `act_quant`, `prepare`; contract text |
| `plugins/holon_plugins/kernels/ggml_cuda` | FP4 types, `prepare` repack, `scale2` on every path, two activation specs, workspace, pin `cbe28bb3` (`vendor.sh`, `vendor.patch`, `SHA256SUMS`, `[provenance]`, `THIRD_PARTY_NOTICES.md`) |
| `plugins/holon_plugins/kernels/triton_gemm` | renamed from `triton_experts`; FP8 dense + grouped, act quant, pinned configs per class (D7) |
| `testing` | reference providers for `fp8_e4m3/hf`, `nvfp4/hf`, `mxfp4/hf` (via `reference.py`); tiny Qwen3 / Qwen3-MoE fixtures written in each dialect |
| `judge` | HF-quant name map and golden, gates, activation emulators (D9), floors, mode check, plants |
| `bench` | `hf_to_gguf.py` (MXFP4 GGUF from our repack, D10) |

### 4.2 Core seam

| Seam | Today | M0.6 |
|---|---|---|
| Quantized stack slices | `_fill_slice` raises for a `QTensor` (`build.py:219-220`) | a `QTensor` under a stacking rule goes to the owning unit's **`adopt(ctx, leaf, q, index=e)`** (False → the existing "cannot read layout" error). The core reads the stack length from the meta parameter at the first slice (the unit may then store a different shape, e.g. ggml's `[E·out, row_bytes]`), and keeps "each index once", "in range", "not both whole and sliced" and "all slices present" (`build.py:245-281`). A unit storing slices must leave its parameter non-meta (the existing "parameters not loaded" check) |

Generality: the core passes an index and a `QTensor` it never opens; no format, model or MoE word. Proof in
`core/tests/test_stacking.py`: a toy two-part layout (int8 rows + f32 row scales) adopted slice by slice into a toy
stacked unit and read back; out of range, twice, missing, whole-then-sliced and a refusing unit are load errors.
Existing units' `adopt` is called without `index` as today. M0.4 also changes the core: M0.6's change rebases onto it
(parallel-tracks rule).

### 4.3 The HF weight source

`HFSafetensors.tensors()` (`source.py:46-50`) becomes: read all shard headers; if `quantization_config` is present,
the dialect (D2) groups each quantized module's tensors (FP8: `weight` F8_E4M3 + `weight_scale_inv` | `weight_scale`;
NVFP4: `weight`|`weight_packed` + `weight_scale` + `weight_scale_2`|`weight_global_scale` [+ input scale]; MXFP4:
`weight`|`weight_packed` + U8 `weight_scale`) and cross-checks the config (every module it declares quantized has its
parts; ignored ones are dense); yield order: dense tensors and modules in shard order. Each part is read once
(`safe_open` handles kept per shard). ≈ 200 lines with the dialects.

### 4.4 Units

- `_Quantized` (`std/holon_std/ops/units.py:51-65`) gains `act_quant` (qlinear only) and, after binding, `parts =
  kernel.prepare(parts)` when the instance defines it; calls pass `parts`. `Linear` keeps `self.parts` with `weight`
  holding the primary part (so the meta check and the GGUF path are unchanged: ggml layouts have `blocks` only).
  `Embedding` / `LMHead` (dense in every target) pass `{"blocks": …}` for GGUF.
- `ExpertLinear`: `adopt(…, index)` materializes each part as `[E, *slice_part]` on the first slice and copies slice
  `e` in place; a 2-D whole-stack adopt (GGUF) is unchanged. `merge` views parts as `[E, rows, …]` (D4); `bind_kernels`
  adds `act_quant` and calls `prepare`.
- The judge's structural test (M0.2) extends to parts: every stacked slice's dequantization equals the checkpoint
  slice's, bitwise, after loading (so after `prepare`).

### 4.5 `kernels.ggml_cuda` for FP4

- **Repack** (`prepare`, on the device, one tensor or one merged stack at a time — ≤ 230 MB transient for a 30B
  gate/up stack): NVFP4: per 64 elements `{4 × e4m3 scale bytes (sign bit cleared), 32 bytes: element j in the low and
  j + 8 in the high nibble of each 16-element sub-block}` — llama.cpp's `_nvfp4_pack` (`conversion/base.py:996-1018`
  at `cbe28bb3`); ggml decodes the scale ×0.5 against a doubled value table, so the value is unchanged. MXFP4: per 32
  `{e8m0 byte, 16 bytes: element j low, j + 16 high}` — `repack_mxfp4_blocks` (`:917-951`). Our code, following
  both; `in % 256 = 0` holds for every target shape (`plugin.py:162-176` support checks unchanged).
- **Specs:** `q8_1` (every class, priority 10; every supported type with `act_quant` ∈ {`a4`, `a8`}) and `fp4`
  (`sms = {120}`, priority 11; NVFP4 / MXFP4 with `act_quant = a4`). `config["act"]` as D3.
- **C API** (`csrc/api.cu`, ours): `mmq` / `mmq_id` take the precision and set `dst.op_params[3]`;
  `mmvq` / `mmvq_id` take optional `x_scale` / `gate_scale` (f32, 1 or E values) into the vendored fusion args;
  `mmq_schedule` takes the precision (the host config differs for FP4 at Q4 vs Q8); the workspace adds `rows` f32 for
  the NVFP4 activation scale for every instance (one size per workspace name, `build.py:93-100`); our f32 SwiGLU
  takes optional per-row gate/up scales.
- **NVFP4 down** on mmq: `y *= scale2[ids][..., None]` (device gather, graph-safe); gate/up: inside our SwiGLU.
- Each instance counts its calls per (path, mode) for the judge (M0.2's counters).

### 4.6 `kernels.triton_gemm` for FP8

- `act_quant_fp8(x [R, in]) → (x_q e4m3 [R, in], a_s f32 [R, in/128])`, one program per (row, group): the W8A8
  operand; for experts it runs once per input row (R = T for gate/up, T·k for down — the f32 SwiGLU output quantized
  directly; the floor does the same).
- Dense GEMM: tiles `BM × BN × 128` (BK = the scale block), scales loaded once per K step, output `out_dtype`;
  grouped: D8's routing and grid unchanged, the same K-loop over gathered rows, f32 out.
- `bf16` mode: `u8 → bf16` decode in-register (sign, exponent + 120, mantissa << 4; exponent 0 → `m · 2⁻⁹`), the
  dense contract's operand rounding (M0.2 D2) for the activation, scale per K step.
- Configurations pinned per (out, in) and class in the plugin (no autotune, design §5.6 rule 7); determinism and
  capture as D8; support checks: `in % 128 = 0`, `out % 128 = 0` (every target).

### 4.7 Activation mode per class (defaults; `quant.act` raises them)

| format (targets) | sm_86 (3090, A5000) | sm_89 (4090) | sm_120 (5080, 5090, PRO 6000) |
|---|---|---|---|
| FP8 block (`a8`) | W8A16, Triton bit-decode | W8A8 Triton | W8A8 Triton |
| NVFP4 (`a4`) | W4A8: mmq int8 MMA, mmvq dp4a | W4A8 | prefill (mmq) W4A4-dynamic, native `mxf4nvf4` MMA; decode (mmvq) W4A8 |
| MXFP4 (`a4`) | W4A8 | W4A8 | prefill W4A4-dynamic (`mxf4`); decode W4A8 |
| any, `quant.act = a8` | as above | as above | FP4: W4A8 on both paths |
| any, `quant.act = a16` | FP8: W8A16; FP4: no provider (startup error) | same | same |

### 4.8 Numerics and the judge (thresholds and predictions fixed before measuring)

- **Golden:** HF fp32 on the CPU, weights from `reference.py` through the judge's own HF-quant name map (never the
  engine's dialect: it reads the safetensors itself and inverts compressed-tensors' globals itself), streamed per
  layer for the 30B (M0.2: ≈ 122 GB); meta carries `reference.py`'s hash. Prompt set `v1.jsonl`.
- **Gates:** per tensor (per expert), NMSE(reference dequantization, HF bf16 original) ≤ 0.5 (M0.1); predicted
  FP8 ≈ 1e-3, NVFP4 ≈ 1e-2, MXFP4 ≈ 2–3e-2 (reported). Routers, embeddings and heads exact (bf16) or the F32 → bf16
  cast of the 30B FP8.
- **Budgets:** unchanged (`judge/holon_judge/compare.py:30-33`): KL p50/p99 ≤ 2× floor, top-1 ≥ floor − 0.25 points,
  paired ΔPPL guard. Tier A sanity ceiling KL p99 ≤ 1e-4. Floors recorded only after step 1's oracle checks pass,
  carrying `reference.py`'s and `moe/reference.py`'s hashes (M0.2 floor-validity rule); predicted order: A16 ≤ W8A8,
  W4A8 ≤ W4A4-dynamic; W4A4-dynamic vs W4A4-static is the measurement the research note left open.
- **Tier A** (CPU fp32, production `hf` source + dialects + reference providers reading the source layouts, as
  M0.2): every target. It covers dialects, normalization, merges and stacking; the repack is L0's (acceptance 3).
- **Tier B** (production kernels; floor of the mode and path, D9):

  | target | rtx-a5000 | rtx-5080 | RTX PRO 6000 |
  |---|---|---|---|
  | 8B FP8 | W8A16 | W8A8 | — |
  | 8B NVFP4 (both files), 8B MXFP4 | W4A8 prefill + decode | W4A4-dyn prefill, W4A8 decode | — |
  | 30B NVFP4, 30B MXFP4 | W4A8 prefill + decode | — | W4A4-dyn prefill, W4A8 decode |
  | 30B FP8 | — (does not fit) | — | W8A8, and W8A16 (`quant.act = a16`) |

  Decode runs use M0.1's step size from the plugin's mmvq limits (NVFP4 8, MXFP4 7 with ids on sm_86,
  `mmvq.cu:182-183`; 8 on sm_89/120, `:288-289`).
- **L0** (sm_86, sm_89, sm_120a; memcheck each): lossless repack bitwise (acceptance 3); qlinear@2 and
  linear_experts per type, mode and path vs the emulated reference with `l0.check`'s budgets; mmvq `x_scale` /
  `gate_scale` dense and with ids, mmq post-scale and the scaled SwiGLU vs f64; W4A4 activation codes vs the
  emulator bitwise (a test-only quantize entry in the C API); FP8 act quant bitwise vs torch `float8_e4m3fn`; the
  e4m3 → bf16 decode over all 256 codes; distinct random block scales (so a transposed or shifted scale index fails);
  T ∈ {0, 1, 7, 8, 9, 64, 512, 513, 2049}; M0.2's routings for experts; graph replay; run-to-run bit equality;
  poisoned bytes after weights and workspaces.
- **Quantization error**, reported per format and mode: the file's golden vs the bf16 checkpoint's fp32 golden (M0 /
  M0.2), and the engine vs that golden, with top-k flip rates; the W4A4-dynamic vs W4A4-static floor gap for NVFP4.
- **Plants** (`judge/holon_judge/planted.py`; predictions are multiples of the floor's KL p99; outside a prediction:
  recorded with its cause, δ unchanged — M0.1's rule):

  | # | plant | where it bites | predicted |
  |---|---|---|---|
  | P1 | FP8 scale read transposed (`s[k/128, n/128]`) | L0; Tier B | L0 fail; L2 fail, > 10× |
  | P2 | FP8 scale shifted by one K block | L0; Tier B | L0 fail; L2 fail, > 10× |
  | P3 | W8A8 activation scale per token, not per 128 group | L0 (codes); Tier B 5080 | L0 fail; **L2 pass** (≤ 1.5×: with either scale every value above 3.5e-5 of the token's max stays an e4m3 normal, relative error 2⁻⁴) |
  | P4 | e4m3 subnormals decoded as normals (W8A16) | L0 (256 codes); Tier B A5000 | L0 fail; **L2 pass** (≤ 1.2×: it moves only weights below 3.5e-5 of their block's max, by at most that much) |
  | P5 | dense gate/up merge keeps gate's scale rows for up | Tier A; Tier B | fail, > 10× |
  | P6 | 30B FP8 expert scale stack off by one expert | structural; Tier B | structural fail; L2 fail |
  | N1 | compressed-tensors global not inverted | Tier A; Tier B (RedHat 8B) | fail, gross |
  | N2 | NVFP4 repack with the nibbles swapped | L0 (lossless); Tier B | L0 fail; L2 fail, gross |
  | N3 | NVFP4 sub-scales permuted within a 64-block | L0; Tier B | L0 fail; L2 fail, gross |
  | N4 | per-expert `scale2` gathered by slot, not expert id, in mmq down | Tier B 30B NVFP4 | prefill fail (3–20×); decode pass (the vendored fusion indexes by id) |
  | N5 | `scale2` applied after SwiGLU on mmq | Tier B 30B NVFP4 | prefill fail, gross; decode pass |
  | N6 | `op_params[3]` left zero on the `q8_1` spec (today's shim) | Tier B 8B NVFP4 | sm_120: aborts at the first gate/up mmq call (`mmq.cu:184`); sm_86/89: pass |
  | N7 | the `fp4` spec run against the W4A8 floor | verdict | fails the mode check before scoring |
  | N8 | one layer's (layer 18) `o_proj` `scale2` × 1.25 | Tier B 8B NVFP4, A5000 | near threshold: 1.5–4×, fail if > 2× |
  | M1 | MXFP4 e8m0 byte off by one in the repack | L0; Tier B | L0 fail; L2 fail, gross |
  | M2 | Quark 30B expert index off by one in per-slice adopt | structural; Tier B | structural fail; L2 fail |

  Tier A must also fail P5, N1 and a dialect plant (q's `scale` paired with k's weight).
- **Bounded:** graphs vs eager at decode batches 3, 5, 12, 20 on the 30B NVFP4 (rtx-4090): pass (M0.2's padding
  rule holds: unique ids per row); chunked vs one-shot prefill on the 8B FP8 and NVFP4 (rtx-5080): pass.

### 4.9 Memory fit

Plan §2's formula: weights + KV + ≈ 1.5 GiB. Usable totals: 15.56 GiB on the 5080 (measured in M0.2), ≈ 23.6 and
≈ 31.5 GiB estimated for the 4090 and 5090. KV per token: 8B 144 KiB bf16, 76.5 KiB q8_0; 30B 96 and 51 KiB. Free
memory for state, and the KV tokens it holds:

| weights loaded | RTX 5080 16 GB | RTX 4090 24 GB | RTX 5090 32 GB |
|---|---|---|---|
| 8B FP8, 8.79 GiB | 5.3 GiB: ≈ 38K tokens bf16 KV (72K q8_0) | ≈ 97K | ≈ 154K |
| 8B NVFP4, 5.96 GiB | 8.1 GiB: ≈ 59K (111K) | ≈ 118K | ≈ 175K |
| 8B MXFP4, 5.76 GiB | 8.3 GiB: ≈ 60K (114K) | ≈ 119K | ≈ 177K |
| 30B NVFP4, 16.85 GiB | does not fit | 5.3 GiB: ≈ 57K (108K) | ≈ 144K |
| 30B MXFP4 (Quark), 17.22 GiB | does not fit | 4.9 GiB: ≈ 53K (100K) | ≈ 140K |
| 30B FP8, ≈ 29.1 GiB | no | no | 0.9 GiB: ≈ 10K (19K) |

The repack keeps the byte count (4.5 / 4.25 bits per weight on both sides); `input_scale` scalars are the only dropped
bytes. The 16 GB MoE answer stays GGUF UD-IQ3_XXS (M0.2). `prepare` transients: ≤ 230 MB (§4.5). Measured fits go to
`evidence/fit_*`; the 5080's deployment `memory_fraction` 0.9 (M0.2) applies.

### 4.10 Speed measurement

As M0.1 / M0.2: `bench/llama_bench.py` in-process vs llama-bench at the pin (`-p 512 -n 128 -d 0,256 -ngl 99 -fa
on`), medians of repetitions 2–5, from the fixed bench worktree; `llama-server` vs Holon over HTTP on the decode and
agent traces (8B NVFP4, 5080). Predictions: FP4 0.95–1.1× on both bars (same kernels; the W4A4 f32 cast costs ≤ 3% of
a 512-token prefill on sm_120); FP8 tg128 0.85–1.0× Q8_0 (bytes per token ≈ equal: FP8 linears + bf16 head vs Q8_0
linears + Q8_0 head); FP8 pp512 0.4–0.9× Q8_0 (sm_89: fp8 MMA peak is half int8's; Triton vs tuned mmq) — below 0.5×
on sm_120, the CUTLASS sm120 blockwise follow-up is escalated. Evidence: `evidence/<name>_<gpu-class>_<date>.json`.

## 5. Implementation plan

Same panel throughout; each step ends green.

| Step | Work | Checks | Days |
|---|---|---|---|
| 0 | Files (7 targets, llama.cpp GGUFs: NVFP4 by the pin's converter, Q8_0 of the 8B FP8); headers: ties of D4, `k/v_scale` = 1.0, shapes | ties hold for every target; baselines llama-bench at `cbe28bb3` on 5080 / 4090 / A5000 | 0.5 |
| 1 | `std.weights.hf` dialects + `reference.py`; owner cross-checks (D8); `qlinear@2`, units (parts, D4 merges, `act_quant`, `prepare`); core per-slice adopt + toy test; reference providers; tiny Qwen3 / Qwen3-MoE fixtures in each dialect (FP8, CT NVFP4 / MXFP4 by compressed-tensors' own compressors; ModelOpt and Quark written from their specs) | engine (reference kernels) vs transformers on the fixtures' dequantized weights; Tier A plants fail on the fixtures; denylist, core LOC | 3.5 |
| 2 | ggml pin bump to `cbe28bb3` (patch regenerated, existing L0 re-run); FP4 types, repack, `scale2` paths, two specs, C API, workspace | acceptance 3; L0 sm_86 / sm_120a (A5000, 5080), memcheck; existing GGUF L0 unchanged | 3.5 |
| 3 | `kernels.triton_gemm`: rename, FP8 act quant, dense and grouped FP8 in both modes, pinned configs | L0 sm_86 / sm_89 / sm_120a, capture, determinism, memcheck | 3.5 |
| 4 | Judge: HF-quant map, goldens (8B ×4, 30B ×3), gates, emulators (D9), floors, mode check, Tier A on the CPU | gates; Tier A for every target; floors' order | 3 |
| 5 | Tier B per §4.8's table; plants; quantization-error reports; structural test; bounded checks | acceptance 5; plants as predicted | 2.5 |
| 6 | Existing locks: `propagate` (qlinear@2, `act_quant`, pin), one approval | acceptance 9 | 0.5 |
| 7 | Consumer classes: fits, bars, serving, tune + locks, snapshots ×3; sm_89 L0 | acceptance 6–8 | 3 |
| 8 | Docs: design §5.5 (as built: dialects in the source, `prepare` as converter, references), §5.6; plan §2 / §10 status | — | 0.5 |

≈ 20.5 days; critical path 0 → 1 → 2 → 4 → 5 → 7 → 8, step 3 parallel to 2.

## 6. Risks

- **W4A4-dynamic quality** on sm_120 may cost more than the checkpoints' calibrated static scales; it is judged
  against its own floor, so it passes while degrading quality. The report (W4A4-dyn vs W4A4-static vs W4A8 floors, and
  the engine vs bf16) decides whether `a8` becomes the sm_120 default — escalated if so (a default change).
- **Pin bump** (`51ce9c11` → `cbe28bb3`): a week of upstream MMQ changes; the existing GGUF locks may not stay
  snapshot-identical (one approval, L2 decides) and the GGUF bars get a spot check (Qwen3-8B Q4_K_M on the 5080 within
  ±3% of M0.1's).
- **FP8 decode at M = 1:** a tile GEMM wastes 15/16 of its rows; if tg128 < 0.8× Q8_0, a split-K GEMV variant
  (FreeToken's decode GEMV is prior art) is added in step 7.
- **30B FP8 on the 5090:** ≈ 0.9 GiB for state; may not reach a useful context.
- **Community MXFP4 files** (`Yi30`) may hold quality surprises unrelated to Holon; the gate and the quantization-error
  report make that visible, not a failure.
- **Parallel core change:** M0.4 changes the core; M0.6 rebases its one seam after M0.4 merges.

## 7. Deferred, deliberately

- Checkpoint-faithful W4A4 (static input scales) on sm_120: vLLM's CUTLASS sm120 NVFP4 GEMM / grouped MoE, or
  FlashInfer `mm_fp4` / `cutlass_fused_moe`, judged against the W4A4-static floor recorded here.
- CUTLASS sm120 blockwise FP8 (examples 87b / 87c) as the faster FP8 prefill.
- A W4A16 FP4 provider (true A16 for weight-only checkpoints); ModelOpt `MIXED_PRECISION` (needed by NVIDIA's newer
  Qwen3.6/3.8 builds) and per-tensor FP8; Quark NVFP4.
- gpt-oss (canonical MXFP4; a new model plugin with interleaved gate/up, biases and its clamped SwiGLU).
- AMD's official MXFP4 Qwen3.5/3.8 builds; vLLM as a secondary FP8 / FP4 bar.
- Removing the core's unused `Dialect`, `Converter` and `SchemeImpl` registries (M0.6 does not need them: dialects
  live in the source, conversion is a kernel's `prepare`, references are std `reference.py` modules) — a separate
  cleanup with its own design §4.2 / §5.5 edit.
- A bf16-input W4A4 quantizer in `vendor.patch` (removes the f32 cast) if the cast measures above 3%.

## 8. Review findings, resolved

(none yet)
