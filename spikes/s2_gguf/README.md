# Spike S2 — GGUF kernels

## Question

Can we take the ggml-derived GGUF CUDA kernels that SGLang carries (dequant, `mmvq`, `mmq`, MoE), build them as our
own standalone Torch extension with Ledger-style buffers, and reach ≥ 0.8× llama.cpp on the same GPU? We measure
per type (IQ3_XXS, IQ3_S, IQ4_XS, Q3_K, Q4_K, Q8_0) and per path (decode with 1/8/32 rows, prefill with 512 rows,
dense and MoE). Where the kernels fall short, what does it cost to port llama.cpp's current kernels instead, and
how does that compare with writing our own Triton kernels (design §5.6, plan §3.1 S2, M0.1, M0.2)?

**Answer, in short.** The vendored set builds standalone, is graph-safe and byte-deterministic. After one bug fix
(IQ3_S) it is correct for all 10 types in the target files. It only meets the bar for **single-row decode**. Two
things are far behind:
- Decode with 8–32 rows: its `mmvq` re-reads the weights once per row.
- All batched work: its `mmq` and MoE kernels are the llama.cpp b2899 (May 2024) generation. The i-quants have
  no batched kernel at all.

Dequant+cuBLAS reaches parity for dense prefill on sm_86 but only 0.71–0.76× on sm_120. The per-token MoE path is
0.07–0.12× at prefill.

**Decision: port llama.cpp's current quantized-matmul kernels** (`mmvq`, `mmq`, `mmid`, `quantize`) for all
types and paths, about 5 developer-days. A probe shows they compile unmodified outside ggml with a small shim. No
Triton for these types. The SGLang set is superseded, except as a short-lived stopgap for single-row decode.

## Setup

| | |
|---|---|
| sm_86 | this host, RTX A5000 24 GB, `CUDA_VISIBLE_DEVICES=1` (llama.cpp) / `2` (ours), driver <version> |
| sm_120 | the Blackwell workstation, RTX PRO 6000 Blackwell Max-Q, `CUDA_VISIBLE_DEVICES=6` (shared, ~20 GB held by others, idle during runs), driver <version> |
| toolchain | CUDA 13.0 (nvcc V13.0.48), torch 2.11.0+cu130, gguf-py 0.19.0, Python 3.12, cmake 3.22, ninja |
| kernel source | sgl-project/sglang `968726f3cea0f6d6f44bea84797e5e5914658f4e`, `python/sglang/kernels/aot/csrc/quantization/gguf` (Apache-2.0; from vLLM `4492e3a5`/`755ed7b0`, from llama.cpp ggml-cuda b2899, MIT) |
| llama.cpp | ggml-org/llama.cpp `51ce9c11a6f2dfa895696c0048c4333e8953b728` (2026-10-06, MIT), CUDA build, Release, `CMAKE_CUDA_ARCHITECTURES=86` / `120` |
| models | `unsloth/Qwen3-8B-GGUF` {Q8_0, Q4_K_M, IQ4_XS, UD-IQ3_XXS, UD-Q3_K_XL}, `unsloth/Qwen3-30B-A3B-GGUF` UD-IQ3_XXS (default HF cache) |

Third-party code lives in `third_party/` (gitignored). Its provenance is pinned by the scripts:
- `third_party/sglang_gguf/` holds 7 headers copied verbatim (`SHA256SUMS.upstream`, plus SGLang's `LICENSE`).
  `patches/0001-iq3_s-grid.patch` is applied on top of them.
- `third_party/llama.cpp/` is the clone at the commit above. `patches/llama.cpp-s2-perf-cases.patch` adds our layer
  shapes to `test-backend-ops perf` behind `S2_SHAPES=1`.

Files:
- `csrc/ggml_ext.cu`, `csrc/compat.h` are our own Torch wrapper (SGLang's `gguf_kernel.cu` is not used).
- `ggml_ops.py` builds and loads the extension (`out/ext-sm{86,120}`) and holds the workspace sizing,
  weight-tiling and `moe_align` helpers.
- `models.py` maps the S2 files.

Commands (each step writes `results/`):
```
scripts/download_models.sh                       # HF cache
scripts/vendor_kernels.sh                        # SGLang headers -> third_party/sglang_gguf (+ patches)
scripts/build_llamacpp.sh 86                     # clone/pin/patch llama.cpp, build -> out/llama-sm86
CUDA_VISIBLE_DEVICES=1 scripts/run_all_sm.sh 86  # build ext, correctness, graph check, our op perf, llama.cpp op perf, llama-bench
python scripts/inventory.py                      # results/inventory.json
python scripts/report.py 86 > out/report_sm86.md # joins -> results/compare_sm86.json + all tables
scripts/probe_llamacpp_port.sh 86                # port-feasibility probe (below)
# sm_120: same on the Blackwell workstation with CUDA_VISIBLE_DEVICES=6 and SM=120
```

### Extension API (all ops write into caller-allocated tensors, current stream, no allocation)

```
quantize_q8_1(x, xq)                       dequantize(w_u8, type, out)            # out: f32/f16/bf16
mul_mat_vec(w, x, type, y, xq)             mul_mat(w, x, type, y, xq)             # mmvq / tiled mmq
moe_vec(x, w, topk_ids, type, y, xq)       moe(x, w, sorted_ids, expert_ids, n_post_pad, type, top_k, y, xq)
moe_block_size(type)                       # xq: int32 workspace of ggml_ops.q8_1_words(rows, k)
```
Activations are bf16 or fp16. Weights are the raw GGUF block bytes as `uint8`, shape `[rows, row_bytes]`
(`[E, rows, row_bytes]` for experts) — the `gguf.native` layout, no repack. Build time per SM is about 65 s.
Batched `mul_mat`/`moe` exist only for Q4_0…Q8_0 and Q2_K…Q6_K. The `moe` path needs `moe_align` routing (here in
torch with a host sync). A graph-safe device-side align kernel would still have to be written.

## Type inventory (`results/inventory.json`)

| file | ggml types (tensor count) | notable per-role mixing |
|---|---|---|
| 8B-Q8_0 | Q8_0 254, F32 145 | — |
| 8B-Q4_K_M | Q4_K 217, Q6_K 37, F32 145 | attn_v / ffn_down: half the layers Q6_K; output Q6_K |
| 8B-IQ4_XS | IQ4_XS 216, Q5_K 36, Q6_K 1, Q4_K 1, F32 145 | attn_v Q5_K; output Q6_K; token_embd Q4_K |
| 8B-UD-IQ3_XXS (3.18 GiB) | IQ3_XXS 113, IQ2_S 52, IQ3_S 36, Q4_K 36, IQ2_XS 10, IQ4_XS 5, Q5_K 1, Q3_K 1, F32 145 | attn_q/k mostly **IQ2_S/IQ2_XS**; attn_output IQ3_S; attn_v Q4_K; ffn_down 5 layers IQ4_XS |
| 8B-UD-Q3_K_XL | Q3_K 104, Q4_K 68, Q5_K 40, IQ4_XS 20, IQ3_XXS 10, IQ3_S 10, Q6_K 2, F32 145 | ffn_gate/up 5 layers IQ3_S; attn_q/k 5 IQ3_XXS + 5 IQ4_XS |
| 30B-A3B-UD-IQ3_XXS | IQ4_XS 177, IQ3_XXS 89, IQ3_S 42, Q5_K 18, Q6_K 11, Q4_K 1, F32 241 | experts: gate/up IQ3_XXS (32) / IQ3_S (16), down IQ3_XXS 25 / IQ4_XS 13 / IQ3_S 10; attention IQ4_XS/Q5_K/Q6_K; router F32 |

Quantized types to support: **Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS, IQ3_XXS, IQ3_S, IQ2_S, IQ2_XS** (+ F32
norms/router). The "UD" files mix up to 9 types per model, so the kernel plugin must dispatch per tensor.

## Correctness (`results/correctness_sm{86,120}.json`, `scripts/correctness.py`)

Method: real tensors, the first 2048 rows of the first tensor of each type. Our dequant is compared with
`gguf.quants.dequantize`. The matmul paths are compared with `x.float() @ dequant_ref(W).T` and scored by NMSE.
The budget is 5e-4, which is llama.cpp's `test-backend-ops` MUL_MAT threshold. MoE splits the tensor's rows into 8
experts, with 64 tokens and top-4.

| type | dequant f32 bit-exact | dequant max err / max abs w | mmvq n=1/8/32 | mmq n=32/512 | MoE vec / mmq | dequant→bf16 GEMM | pass (86 / 120) |
|---|---|---|---|---|---|---|---|
| Q8_0 | 17 % | 1.6e-4 | 2.9–3.1e-5 | 3.1e-5 | 3.1e-5 / 3.1e-5 | 5–8e-6 | ✓ / ✓ |
| Q3_K | 35 % | 2.5e-4 | 3.2–3.3e-5 | 3.1–3.2e-5 | 3.1e-5 / 3.1e-5 | 6–7e-6 | ✓ / ✓ |
| Q4_K | 1.5 % | 4.9e-4 | 3.1–3.2e-5 | **1.5–1.6e-4** | 3.1e-5 / **1.6e-4** | 6–8e-6 | ✓ / ✓ |
| Q5_K | 1.2 % | 7.9e-4 | 2.9–3.2e-5 | **1.5–1.6e-4** | 3.2e-5 / **1.8e-4** | 6–8e-6 | ✓ / ✓ |
| Q6_K | 14 % | 1.9e-4 | 3.0–3.2e-5 | 3.1–3.2e-5 | 3.1e-5 / 3.1e-5 | 6–9e-6 | ✓ / ✓ |
| IQ4_XS | 100 % | 0 | 3.1–3.4e-5 | — | 3.1e-5 / — | 6–7e-6 | ✓ / ✓ |
| IQ3_XXS | 100 % | 0 | 2.8–3.2e-5 | — | 3.1e-5 / — | 6–7e-6 | ✓ / ✓ |
| IQ3_S (patched) | 100 % | 0 | 3.1–3.4e-5 | — | 3.1e-5 / — | 6–9e-6 | ✓ / ✓ |
| IQ2_S | 100 % | 0 | 2.7–3.1e-5 | — | 3.2e-5 / — | 5–7e-6 | ✓ / ✓ |
| IQ2_XS | 100 % | 0 | 2.7–3.2e-5 | — | 3.2e-5 / — | 5–7e-6 | ✓ / ✓ |

Findings:
1. **Bug in the vendored IQ3_S (fixed by `patches/0001-iq3_s-grid.patch`).** SGLang's IQ3_S dequant and
   `vec_dot_iq3_s_q8_1` (used by `mmvq` and `moe_vec`) decode with the obsolete `iq3xs_grid` and a
   `(0.5+s)*0.5` scale. llama.cpp and gguf-py use `iq3s_grid` and `(1+2s)`. The two grids differ only in the
   largest magnitude, which decodes as 15.5 instead of 15. Before the patch, 4.3 % of the weights were wrong (max
   err 0.023 on |w| ≤ 0.68), and mmvq NMSE was 2.6e-4 instead of 3e-5. That stays under the 5e-4 budget, so a
   loose NMSE gate would not have caught it. The patch copies llama.cpp's `iq3s_grid` (MIT). Report upstream to
   SGLang. vLLM no longer carries GGUF kernels in-tree.
2. **The legacy and k-quant dequant compute in fp16** (`typedef half dfloat`, `__hmul`). It is not bit-exact vs
   gguf-py: the error is ≤ 8e-4·max|w|, and 75–95 % of bf16 outputs match. On the dequant+GEMM fallback this
   costs nothing measurable (NMSE 5.6e-6 vs 5.4e-6 with exact weights). It still violates "L0 bit-exact vs
   `dequant_ref`" if we want that. The i-quant dequant is bit-exact. Current llama.cpp dequantizes in fp32.
3. q8_1 activation quantization sets the error floor of every int8-dot path: NMSE about 3e-5, versus about 6e-6
   for bf16 GEMM on dequantized weights. This is the same as llama.cpp. The judge budgets for GGUF classes must
   allow for it.
4. Q4_K/Q5_K `mmq` (and the MoE `mmq`) have 5× the NMSE of `mmvq`. The cause is the algorithm, not a bug: the
   min-term uses the fp16 sum of the *unquantized* activations (`ds.y`) while the scale term uses the quantized
   ones, so the q8_1 rounding no longer cancels. Current llama.cpp `mmq` has the same structure. It is 3× under
   budget.

## Performance

Methodology for the per-kernel comparison:
- **Same as llama.cpp `test-backend-ops perf`.** One op repeated back to back on the same weights. Ours: R
  repeats inside one CUDA graph, best of 5 replays timed with events. llama.cpp: its own graph, run for ≥ 1 s.
  Small weights can be L2-resident in both harnesses: 6 MB L2 on the A5000, 128 MB on the RTX PRO 6000. The
  absolute numbers on sm_120 are therefore L2-flattered; the ratios are like for like.
- Timed ops include the q8_1 activation quantization. Ours use bf16 activations, llama.cpp f32. MoE routing
  (`moe_align`) is not timed.
- MoE shapes are Qwen3-30B-A3B experts: 128 experts, top-8. gate/up are 768×2048 with one input row per token
  (llama.cpp `b=true`); down is 2048×768 with one input row per (token, slot).
- The ratio is llama.cpp time / our time, so ≥ 1 means we are faster. The bar is ≥ 0.8.

### End-to-end llama.cpp baselines (llama-bench, `-ngl 99 -fa on -r 5`, `results/llama_bench_*`)

| model | sm_86 pp512 t/s | sm_86 tg128 t/s | sm_120 pp512 t/s | sm_120 tg128 t/s |
|---|---|---|---|---|
| Qwen3-8B Q4_K_M | 3856 ± 84 | 107.6 ± 0.2 | 9871 ± 822 | 198.1 ± 2.5 |
| Qwen3-8B UD-IQ3_XXS | 3862 ± 79 | 111.0 ± 0.2 | 10065 ± 847 | 234.9 ± 2.2 |
| Qwen3-30B-A3B UD-IQ3_XXS | 3529 ± 87 | 151.4 ± 0.2 | 8444 ± 971 | 261.5 ± 5.6 |

These are the numbers M0.1/M0.2 must reach (≥ 0.8×). The ±8–11 % pp512 spread on sm_120 comes from a shared,
power-capped (Max-Q) card. The binding numbers are on the consumer cards.

### Per-layer linear time (best of our paths per op; llama.cpp µs / ours µs = ratio)

Qwen3-8B layer = q, o (4096×4096), k, v (1024×4096), gate, up (12288×4096), down (4096×12288). Qwen3-30B-A3B layer
= q (4096×2048), k, v (512×2048), o (2048×4096), plus the expert ops gate, up and down. Every op uses the row's
type.

sm_86 (A5000):

| Qwen3-8B | n=1 | n=8 | n=32 | n=512 |
|---|---|---|---|---|
| IQ3_XXS | 214 / 219 = **0.97** | 426 / 1421 = **0.30** | 467 / 1452 = **0.32** | 3014 / 2999 = **1.00** |
| IQ3_S | 235 / 228 = **1.03** | 454 / 1491 = **0.30** | 493 / 1562 = **0.32** | 3098 / 3096 = **1.00** |
| IQ4_XS | 185 / 240 = **0.77** | 380 / 1418 = **0.27** | 405 / 1515 = **0.27** | 2794 / 2932 = **0.95** |
| Q3_K | 284 / 368 = **0.77** | 635 / 1051 = **0.60** | 454 / 1706 = **0.27** | 3528 / 3269 = **1.08** |
| Q4_K | 201 / 217 = **0.93** | 523 / 689 = **0.76** | 396 / 1506 = **0.26** | 3102 / 3036 = **1.02** |
| Q8_0 | 316 / 316 = **1.00** | 522 / 755 = **0.69** | 454 / 1598 = **0.28** | 3009 / 3328 = **0.90** |

| Qwen3-30B-A3B | n=1 | n=8 | n=32 | n=512 |
|---|---|---|---|---|
| IQ3_XXS | 99 / 85 = **1.18** | 436 / 536 = **0.81** | 1125 / 1757 = **0.64** | 2727 / 25179 = **0.11** |
| IQ3_S | 103 / 91 = **1.14** | 484 / 568 = **0.85** | 1208 / 1874 = **0.64** | 2794 / 26614 = **0.10** |
| IQ4_XS | 77 / 103 = **0.75** | 305 / 630 = **0.48** | 967 / 2079 = **0.47** | 2534 / 30565 = **0.08** |
| Q3_K | 113 / 114 = **0.99** | 419 / 636 = **0.66** | 1094 / 1633 = **0.67** | 3140 / 12673 = **0.25** |
| Q4_K | 84 / 86 = **0.98** | 335 / 431 = **0.78** | 1063 / 1048 = **1.01** | 2811 / 7935 = **0.35** |
| Q8_0 | 119 / 117 = **1.02** | 546 / 558 = **0.98** | 1217 / 1247 = **0.98** | 2721 / 8330 = **0.33** |

sm_120 (RTX PRO 6000 Max-Q):

| Qwen3-8B | n=1 | n=8 | n=32 | n=512 |
|---|---|---|---|---|
| IQ3_XXS | 78 / 79 = **0.98** | 154 / 439 = **0.35** | 188 / 648 = **0.29** | 1098 / 1441 = **0.76** |
| IQ3_S | 84 / 86 = **0.98** | 164 / 473 = **0.35** | 200 / 854 = **0.23** | 1223 / 1673 = **0.73** |
| IQ4_XS | 74 / 87 = **0.85** | 138 / 515 = **0.27** | 169 / 652 = **0.26** | 1048 / 1485 = **0.71** |
| Q3_K | 86 / 134 = **0.64** | 118 / 621 = **0.19** | 188 / 828 = **0.23** | 1363 / 1785 = **0.76** |
| Q4_K | 70 / 78 = **0.90** | 115 / 278 = **0.42** | 169 / 645 = **0.26** | 1122 / 1474 = **0.76** |
| Q8_0 | 76 / 81 = **0.94** | 181 / 303 = **0.60** | 181 / 573 = **0.32** | 1069 / 1409 = **0.76** |

| Qwen3-30B-A3B | n=1 | n=8 | n=32 | n=512 |
|---|---|---|---|---|
| IQ3_XXS | 51 / 50 = **1.01** | 124 / 212 = **0.58** | 418 / 683 = **0.61** | 1053 / 9507 = **0.11** |
| IQ3_S | 51 / 52 = **0.98** | 140 / 216 = **0.65** | 472 / 716 = **0.66** | 1127 / 9746 = **0.12** |
| IQ4_XS | 36 / 52 = **0.69** | 97 / 222 = **0.44** | 372 / 706 = **0.53** | 1076 / 9913 = **0.11** |
| Q3_K | 47 / 62 = **0.76** | 155 / 227 = **0.68** | 433 / 546 = **0.79** | 1376 / 4248 = **0.32** |
| Q4_K | 40 / 50 = **0.80** | 123 / 154 = **0.79** | 376 / 365 = **1.03** | 1191 / 2630 = **0.45** |
| Q8_0 | 41 / 50 = **0.83** | 120 / 163 = **0.74** | 599 / 555 = **1.08** | 1186 / 3229 = **0.37** |

### Per path (ratio llama.cpp / ours; sm_86 · sm_120)

Dense (Qwen3-8B layer):

| type | mmvq n=1 | mmvq n=8 | mmvq n=32 | vendored mmq n=8 / 32 / 512 | dequant+cuBLAS n=32 | dequant+cuBLAS n=512 |
|---|---|---|---|---|---|---|
| IQ3_XXS | 0.97 · 0.98 | 0.30 · 0.35 | 0.08 · 0.11 | none | 0.32 · 0.29 | **1.00 · 0.76** |
| IQ3_S | 1.03 · 0.98 | 0.29 · 0.35 | 0.08 · 0.11 | none | 0.32 · 0.23 | **1.00 · 0.73** |
| IQ4_XS | 0.77 · 0.85 | 0.23 · 0.27 | 0.06 · 0.08 | none | 0.27 · 0.26 | **0.95 · 0.71** |
| Q3_K | 0.77 · 0.64 | 0.24 · 0.16 | 0.04 · 0.07 | 0.60/0.12/0.06 · 0.17/0.11/0.05 | 0.27 · 0.23 | **1.08 · 0.76** |
| Q4_K | 0.93 · 0.90 | 0.35 · 0.29 | 0.07 · 0.11 | 0.76/0.15/0.08 · 0.40/0.21/0.10 | 0.26 · 0.26 | **1.02 · 0.76** |
| Q8_0 | 1.00 · 0.94 | 0.23 · 0.41 | 0.05 · 0.11 | 0.69/0.16/0.07 · 0.55/0.21/0.08 | 0.28 · 0.32 | **0.90 · 0.76** |

MoE (Qwen3-30B-A3B expert ops gate+up+down):

| type | moe_vec n=1 / 8 / 32 / 512 | vendored moe mmq n=1 / 8 / 32 / 512 |
|---|---|---|
| IQ3_XXS | 1.27/0.98/0.66/**0.09** · 1.06/0.62/0.59/**0.09** | none |
| IQ3_S | 1.19/1.02/0.66/**0.09** · 1.08/0.70/0.66/**0.10** | none |
| IQ4_XS | 0.71/0.54/0.47/**0.07** · 0.64/0.45/0.50/**0.09** | none |
| Q3_K | 1.02/0.59/0.44/0.07 · 0.85/0.64/0.59/0.12 | 0.63/0.52/0.70/0.22 · 0.37/0.47/0.78/0.28 |
| Q4_K | 0.91/0.69/0.61/0.09 · 0.77/0.47/0.53/0.11 | 0.64/0.66/1.12/0.32 · 0.46/0.56/1.13/0.41 |
| Q8_0 | 1.02/0.96/0.56/0.07 · 0.75/0.57/0.78/0.09 | 0.71/0.95/1.04/0.29 · 0.40/0.67/1.08/0.32 |

Raw per-op numbers: `results/compare_sm*.json` (joined), `results/ours_ops_sm*.json`, and
`results/llamacpp_ops_sm*.sql`. `python scripts/report.py <sm>` regenerates every table, including the
per-op detail.

### Known gap, quantified (task 5)

- **Dense, i-quants, batched.** There is no i-quant `mmq`. The fallback is dequant to bf16 (a workspace the size
  of the weight) followed by cuBLAS.
  - n=512: 0.95–1.00× on sm_86 (cuBLAS bf16 is strong there) but only 0.71–0.76× on sm_120, where llama.cpp's
    int8-MMA `mmq` is relatively faster.
  - n=8–32: 0.27–0.32×. The fallback pays the full dequant (352 µs for a 12288×4096 IQ3_XXS weight on sm_86)
    whatever the batch size, and `mmvq` re-reads the weights per row.
  - Even for k-quants the vendored `mmq` is not an alternative: 0.05–0.16× at n=32/512 (old b2899 tiling).
  - The fallback also needs a workspace the size of the largest weight. The 151936×4096 `lm_head` needs 1.2 GB in
    bf16 and must be chunked.
- **MoE, i-quants, batched.** The only kernel is per-token `moe_vec`: 0.98–1.27× at n=1 but 0.47–0.66× at n=32
  and **0.07–0.12× at n=512**. Prefill of the 30B model would be about 10× slower than llama.cpp. A dequant
  fallback does not help. Dequantizing the three 128-expert tensors of one layer (604 M weights; at n=512 nearly
  all experts are active) would take about 4.2 ms on sm_86 and 1.9 ms on sm_120 at the measured dequant rate. That
  is ≥ 1.8× llama.cpp's *entire* expert time (2.36 ms and 0.87 ms) before any GEMM, and torch has no grouped GEMM
  on sm_86.
- **MoE, k-quants, batched.** The vendored `moe` mmq runs at 0.22–0.45× at n=512. At n=1 it reaches only
  0.37–0.71× (0.47–0.95× at n=8), because the graph-safe grid is sized for the worst-case padded routing (n·k + E·(B−1) slots), so most
  blocks exit immediately.

## CUDA graphs and determinism (`results/graph_sm{86,120}.json`, `scripts/graph_check.py`)

All 10 types were captured on sm_86 and sm_120. The paths cover `mmvq` at n=1/8/32, `mmq` at n=32 (k-quants),
dequant+cuBLAS at n=1/8/32, `moe_vec` at n=1/8/32, and `moe` mmq (k-quants, routing precomputed). Results:
- Capture succeeds: no host syncs on the op path; the wrappers only do host-side shape checks.
- Graph replay equals eager execution bytewise.
- 20 replays are byte-identical.
- Feeding a new input through the static buffer reproduces its eager result bytewise.
- `all_ok = true` on both SMs.

None of these kernels uses atomics or split-K, so they are run-to-run deterministic. They are **not**
batch-invariant: `mmvq`, `mmq` and the fallback give different bits for the same row at different n. That is
input for S4.

## Decision per type and path

The criterion is ≥ 0.8× llama.cpp on both SMs.

| path | IQ3_XXS | IQ3_S | IQ4_XS | Q3_K | Q4_K | Q8_0 |
|---|---|---|---|---|---|---|
| dense decode n=1 | adopt vendored mmvq (stopgap) → port | adopt (stopgap) → port | **port** (0.77/0.85) | **port** (0.77/0.64) | adopt (stopgap) → port | adopt (stopgap) → port |
| dense decode n=2–8 | **port** llama.cpp multi-column `mmvq` | port | port | port | port | port |
| dense n=9–~64 | **port** llama.cpp `mmq` (i-quant) | port | port | **port** (vendored mmq 0.12–0.21) | port | port |
| dense prefill n≥256 | sm_86: dequant+cuBLAS (≥0.95) OK; sm_120: **port** `mmq` | same | same | same | same | same |
| MoE decode n=1 | adopt `moe_vec` (stopgap) → port | adopt (stopgap) → port | **port** (0.71/0.64) | adopt (stopgap, 1.02/0.85) → port | **port** (0.91/0.77) | **port** (1.02/0.75) |
| MoE n≥8 / prefill | **port** llama.cpp `mul_mat_id` (`mmid` + `mmq` with ids, `mmvq` ids) | port | port | port (vendored 0.2–0.7) | port | port |
| dequant (oracle, fallback, fp32/bf16 out) | adopt (bit-exact) | adopt (patched, bit-exact) | adopt (bit-exact) | port llama.cpp `convert.cu` (vendored is fp16) | same | same |
| own Triton | — | — | — | — | — | — |

**Recommendation.** Port the current llama.cpp quantized matmul family once, for every type, and use it on all
paths. It is the baseline we are measured against, so parity comes by construction, and it covers
k-quants and i-quants, dense and MoE, and every batch size with llama.cpp's own per-SM kernel choice
(`ggml_cuda_should_use_mmvq/mmq`). The vendored SGLang set can serve as the M0.1 bring-up kernel for single-row
decode, where it is at 0.9–1.0× for IQ3_XXS/IQ3_S/Q4_K/Q8_0. It should then be deleted rather than maintained
next to the port. Writing our own Triton is not justified for these types.

## Effort estimate

**Option A — port llama.cpp's current kernels (recommended), about 5 developer-days for M0.1 + M0.2.**
The probe (`scripts/probe_llamacpp_port.sh`) compiled `mmvq.cu`, `mmq.cu`, `quantize.cu`, `mmid.cu` and the
mmq instances for IQ3_XXS/IQ3_S/IQ4_XS/Q3_K/Q4_K/Q8_0 **unmodified** with nvcc 13.0 for sm_86 against llama.cpp's
headers. Without `ggml-cuda.cu` and `ggml-base`, only these symbols are left unresolved:
- 15 ggml-base helpers: `ggml_nbytes`, `ggml_type_size`, `ggml_blck_size`, `ggml_is_contiguous*`, `ggml_abort`,
  `ggml_log_internal`, …
- 5 ggml-cuda runtime hooks: `ggml_cuda_info`, `ggml_cuda_get_device`, `ggml_cuda_set_device`, `ggml_cuda_error`,
  `ggml_backend_cuda_context::new_pool_for_device`.

The entry points take `ggml_tensor`, a plain struct we can fill on the stack over torch tensors.

| step | days |
|---|---|
| Shim (~200 lines): device info from `cudaGetDeviceProperties`; pool = bump allocator over a declared workspace (q8_1/mmq activation buffer, stream-k fixup `nsm·I·J` floats, MoE `ids_src1/ids_dst/expert_bounds`), with a sizing query per (type, max rows); ggml-base helpers; build instances for sm_86/89/120 | 1.5 |
| Wrappers + `KernelSpec` + workspace declaration; dense mmvq/mmq selection copied from llama.cpp's thresholds | 1 |
| MoE `mul_mat_id` (device-side `mm_ids_helper` routing, so no `moe_align` host sync) | 1 |
| Reuse `correctness.py` / `graph_check.py` / `bench_ops.py` for parity checks on sm_86 + sm_120 (+ sm_89 on consumer desktops) | 1 |
| Follow llama.cpp updates (pinned commit; refresh = re-run the probe + tests) | ongoing |

Risks:
- Kernel heuristics depend on `cc`/`nsm`, so the shim must report the true device.
- Stream-k `mmq` uses a fixup pass. It should be deterministic (fixed order), but S4 must confirm it.
- llama.cpp's MoE path uses its own q8_1 "mmq" activation layout. That is internal workspace, not a
  weight layout, so the weights stay in `gguf.native`.

**Option B — our own Triton kernels: 2–4 weeks, high risk.**
- Each i-quant needs lookup-table gathers inside the K-loop: IQ3_XXS 256×u32, IQ3_S 512×u32, IQ4_XS/NL 16 values,
  plus sign masks and 6-bit sub-scales. They feed bf16 `tl.dot`.
- Prefill: llama.cpp's `mmq` uses int8 MMA (2× bf16 throughput on sm_86). Our dequant+cuBLAS bf16 path already
  shows 0.71–0.76× on sm_120, so a bf16 Triton kernel would likely land at or below that.
- Decode: i-quant decode is ALU-heavy. llama.cpp's 8B UD-IQ3_XXS (3.18 GiB) decodes only 3 % faster than Q4_K_M
  (4.68 GiB) on sm_86, and 19 % faster on sm_120.
- It would need 3 types × {multi-column GEMV, GEMM, grouped MoE} × per-SM configs. Not recommended. Keep Triton
  for formats llama.cpp does not cover (v1 HF formats).

## Open issues

- A device-side routing/align kernel is needed if any vendored `moe` mmq survives. It is moot with Option A, whose
  `mm_ids_helper` already runs on the device.
- The dequant+cuBLAS fallback needs a chunked dequant for `lm_head` (1.2 GB of bf16), or a workspace cap.
- Per-kernel numbers on the RTX PRO 6000 are L2-flattered (128 MB L2 holds most single weights). Consumer
  evidence (5080/5090/4090) must be measured on consumer desktops, including the llama-bench baselines (the plan's M0.1
  bar is on the RTX 5080).
- Batch invariance: every path changes bits with n. That goes to S4, and so does the stream-k fixup determinism
  under Option A.
- Judge budgets: int8-dot paths sit at NMSE ≈ 3e-5 vs fp32 (q8_1 activations). The GGUF numerics class must be
  calibrated for that, not for bf16 GEMM (≈ 6e-6).
- The IQ3_S fix should be reported upstream to SGLang.
- Still untested: IQ1_S/IQ1_M/IQ2_XXS/IQ4_NL/Q2_K/Q5_0/1 (not present in the S2 files), sm_89, and fp16
  activations (compiled, not exercised).
