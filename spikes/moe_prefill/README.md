# moe_prefill — where Holon's all-resident MoE prefill loses to FreeToken

Record: `evidence/moe_prefill_rtx-a5000_2026-10-18.json`. The step-0 diagnosis of the MoE prefill speed item (plan
§10; `docs/v1-offload.md` §13, "Owner decision on the gate"); the design it feeds is `docs/v1-moe-prefill.md`. No code
in core, std or plugins changed: the probes patch module constants in the worker process only.

## Method

- **Vehicle:** Qwen3-30B-A3B Q4_K_M (`unsloth/Qwen3-30B-A3B-GGUF`): 48 layers, 128 experts, top-8, inter 768,
  hidden 2048. Gate/up experts are Q4_K; down experts are Q6_K in 24 layers and Q4_K in 24. All resident on one
  RTX A5000 (sm_86, 24 GB), batch 1.
- **Engines:**
  - Holon at `b5b1e7f` (main), default flags: `std.moe@1`, resident store, `max_step_tokens` 4,096, prefix cache
    off.
  - FreeToken `21d9578` (`bench/setup_baselines.sh offload`) with `--moe-strategy fused` (every expert resident). It
    runs on sm_86 as built.
  - Both allocate a 33,792-token context.
- **Driver `prefill.py`:**
  - Each arm runs in a fresh process: an untimed 512-token warm-up, then 3 repetitions of one-token requests at
    1,024 / 4,096 / 32,768 prompt tokens. The time is to the first token.
  - Prompts are natural text (`bench/offload_ctx.py`'s corpus). Each (repetition, length) starts at its own offset,
    so FreeToken's radix cache never hits.
  - 2 rounds, interleaved: 6 times per cell (3 prompts × 2 rounds). The spread within a cell is mostly the prompts':
    each repetition is a different prompt, and the third 1K prompt is 5–10% faster in every arm. The same prompt
    agrees across rounds within 3% (≤ 0.5% for FreeToken and the probe arms).
- **Probes (Holon worker only):**
  - `--pass-tokens 4096`: the resident store's expert pass (`Resident.pass_tokens`; 512 as shipped).
  - `--expert-rows 4096`: kernels.ggml_cuda's mmq rows per `linear_experts` call (512 as shipped for a resident
    stack; offload arenas already take 4,096).
  - `--dense-rows 4096`: the same for `qlinear`.
  - The router and top-k stay per 512-token chunk in every arm.
- **nsys** (`-t cuda,nvtx --capture-range=cudaProfilerApi`): one profiled request per length after the timed ones.
  `breakdown.py` gives each kernel the innermost NVTX region open at its launch: `experts`, `dense`, `router` or
  `combine`, pushed around each engine's Python calls in this process only. Inside a region kernels are split by
  name; outside, x4_gap's rules apply. Profiled walls are within 1–3% of the timed ones.
- **Per-call bench `gemm.py`:** one layer's expert GEMMs, CUDA-event timed, median of 20.
  - Routing: top-8 distinct per token, popularity ∝ rank^-0.5 (the top 10% of experts take ≈ 27% of routes; M6's
    range is 23–36%).
  - `mmq_id`: Holon's call, which includes its q8_1 quantize and id helper.
  - `dense_mmq`: the same MACs as one expert's dense mmq at T·k rows, the perfect-grouping bound.
  - `bf16_mm`: cuBLAS bf16 at T·k rows on one dequantized expert.
  - `dequant`: the whole stack to bf16.
  - `--freetoken`: FreeToken's `mul_mat_id` on the same stacks and routing.
  - `--j-check`: whether mmq's column-tile width J changes a row's bits.

## 1. Prefill time to first token (A5000, median of 6)

| Arm | 1K s (tok/s) | 4K s (tok/s) | 32K s (tok/s) | vs FreeToken 1K / 4K / 32K |
|---|---|---|---|---|
| Holon as shipped | 0.279 (3,666) | 1.180 (3,472) | 14.03 (2,335) | 0.63 / 0.49 / 0.66 |
| + expert pass 4,096 (mmq still 512 rows per call) | 0.281 | 1.181 | 14.13 | 0.63 / 0.49 / 0.65 |
| + mmq with ids 4,096 rows per call | 0.215 (4,757) | 0.703 (5,827) | 10.20 (3,213) | 0.82 / 0.83 / 0.90 |
| + dense mmq 4,096 rows per call | 0.212 (4,841) | 0.683 (5,997) | 10.03 (3,266) | 0.83 / 0.85 / 0.92 |
| FreeToken fused (resident) | **0.176 (5,818)** | **0.582 (7,035)** | **9.23 (3,551)** | — |

- The ratios match the RTX 5090's: FreeToken's resident prefill there was 1.6× / 2.0× Holon's (x4_gap Q1).
- A larger pass alone buys nothing; the gain is in mmq calls of 4,096 rows.

## 2. Where the time goes (nsys, ms of GPU time per prompt)

| Component | 1K: shipped / probe / FT | 4K: shipped / probe / FT | 32K: shipped / probe / FT |
|---|---|---|---|
| **expert GEMMs (mmq with ids)** | 211.5 / 143.7 / 113.5 | **832.1 / 383.4 / 298.8** | **6,809 / 3,084 / 2,151** |
| dense GEMMs (mmq) | 33.5 / 29.9 / 28.7 | 132.5 / 108.0 / 108.5 | 1,063 / 875 / 881 |
| attention (incl. q/k norm, RoPE, KV write) | 10.5 / 10.4 / 11.8 | 97.2 / 94.8 / 107.5 | 5,318 / 5,252 / **5,605** |
| router (Holon: f32 cast, SGEMM, split-K reduce, top-k; FT: bf16 GEMM, Triton top-k) | 5.5 / 5.4 / 0.8 | 22.2 / 21.9 / 2.3 | 179 / 177 / 16.1 |
| MoE combine (Holon f32 y; FT bf16 y) | 5.8 / 5.6 / 2.7 | 23.4 / 21.5 / 10.6 | 187 / 172 / 84 |
| SwiGLU (Holon f32; FT bf16) | 5.2 / 5.3 / 2.6 | 20.8 / 21.0 / 10.5 | 166 / 168 / 85 |
| expert q8_1 quantize | 4.2 / 4.2 / 3.4 | 16.6 / 16.5 / 13.5 | 133 / 133 / 109 |
| id helper and sort | 3.0 / 3.6 / 3.6 | 12.2 / 2.8 / 13.3 | 98 / 22 / 106 |
| dense q8_1, norms, glue | 4.6 / 4.3 / 5.0 | 18.5 / 16.3 / 18.6 | 147 / 131 / 147 |
| **busy / wall** | 284 / 290 · 212 / 217 · 172 / 226 | 1,176 / 1,185 · 686 / 692 · 584 / 588 | 14,101 / 14,165 · 10,013 / 10,050 · 9,184 / 9,199 |

The probe column is the all-4,096 arm. FreeToken runs 8,192-token prefill chunks (`max_extend_tokens`), so its 32K
prompt makes one expert call per layer per 8K. At 1K its wall exceeds its busy time by 54 ms of host gaps.

## 3. Per call: Holon's grouped mmq against FreeToken's and against the bound

Layer 0 (Q4_K gate/up, Q6_K down); layer 6 (Q4_K down) agrees within 10%. Times in ms per call:

| Stack | Tokens | Holon mmq_id | FreeToken mul_mat_id | dense mmq bound | Holon / FT | FT / bound | Column tiles used / launched (Holon) |
|---|---|---|---|---|---|---|---|
| gate/up (out 1,536) | 512 | 1.52 | 0.95 | 0.44 | 1.60 | 2.18 | 130 / 544 |
| | 1,024 | 1.72 | 1.37 | 0.85 | 1.25 | 1.62 | 137 / 1,088 |
| | 2,048 | 2.62 | 2.32 | 1.72 | 1.13 | 1.35 | 177 / 2,176 |
| | 4,096 | 4.95 | 4.11 | 3.44 | 1.20 | 1.20 | 331 / 4,352 |
| | 8,192 | — (`ID_PAIRS`) | 7.71 | — | — | — | — |
| down (out 2,048, f32 in) | 512 | 1.04 | 0.61 | 0.27 | 1.70 | 2.27 | 129 / 544 |
| | 1,024 | 1.23 | 0.98 | 0.51 | 1.25 | 1.95 | 136 / 1,088 |
| | 2,048 | 1.73 | 1.51 | 0.99 | 1.15 | 1.53 | 174 / 2,176 |
| | 4,096 | 3.57 | 2.81 | 1.99 | 1.27 | 1.41 | 332 / 4,352 |
| | 8,192 | — | 5.03 | — | — | — | — |

- **Holon's grouped launch.** Holon sizes mmq's column tile J against all of a call's tokens, so J is 128 from 128
  tokens on. Its grid has ceil(T / J) column tiles for every one of up to `experts + top_k` slots
  (`vendor/…/mmq.cu` J_best loop, `mmq.cuh` `block_nums_xy_tiling`).
  - At 512 tokens an expert holds ≈ 32 tokens, so ≈ 75% of each J = 128 tile's MMA work is padding. Of the column
    tiles launched, 76–92% hold no token and exit.
  - FreeToken sizes J against the mean tokens per expert and launches only the (expert, column tile) pairs that hold
    tokens. It also writes bf16, not f32 (`freetoken.patch` on its vendored ggml, per the source study below).
- **Size of that cost.** At 4,096 tokens per call FreeToken is 1.20–1.41× the bound, and Holon 1.44–1.86×.
- **Per token, FreeToken gains ≈ 6–12% more at 8,192 tokens per call.** That is its 32K edge in §2 (65.6 against
  73 µs of expert GEMM per token at 4K).
- **Holon and FreeToken run the same arithmetic class:** ggml mmq, q8_1 activations, int8 MMA, f32 accumulation.
  FreeToken vendors llama.cpp `8c2134a8` plus its own patch; Holon vendors `cbe28bb3`. Neither dequantizes to bf16
  for quantized experts. llama.cpp itself uses mmq for k-quants on every Turing+ GPU at every batch size
  (`ggml_cuda_should_use_mmq`: `turing_mma_available` → true); it never falls back to dequant + cuBLAS.
- **J does not change a row's bits** (`--j-check`). The first 16 / 32 / 64 rows of a 512- or 4,096-token call
  (J = 128) are bitwise the rows of a call of those tokens alone (J = 16 / 32 / 64), for Q4_K and Q6_K, both stacks,
  layers 0 and 6. L0 already pins this for every type
  (`plugins/tests/test_ggml_cuda_moe.py::test_expert_rows_do_not_depend_on_the_slots_or_the_tile_width`). A grid that
  launches only the occupied tiles, with J chosen against the mean, is therefore a bitwise change.
- **Dequantizing to bf16 with tensor-core GEMMs does not pay.**
  - On this card cuBLAS bf16 runs at 80–90 TFLOPS against dense mmq's 48–61 TOPS. But dequantizing a layer's stacks
    costs 2.5 ms per pass (1.63 gate/up + 0.87 down), more than the 1.7 ms that bf16 would save at the bound at
    4,096 tokens. The scratch is also 1.2 GB per layer.
  - The A5000 is a professional card (bf16 with f32 accumulation at full rate). On the GeForce classes Holon's bars
    use (RTX 3090 / 4090 / 5080 / 5090), bf16 with f32 accumulation runs at a quarter of the int8 rate (NVIDIA's
    published figures; to be confirmed on a 5080 by `gemm.py` in the design's step 0), so mmq stays ahead.

## The gap, ranked (A5000, all-resident)

GPU busy ms per prompt, Holon as shipped against FreeToken; Δ is what the fix would recover, measured or bounded by
FreeToken's own number.

| # | Component | 4K Δ | 32K Δ | 1K Δ | Fix (exactness) |
|---|---|---|---|---|---|
| 1 | Expert mmq at 512 rows per call (the resident store's pass) | **−449** (measured) | **−3,725** | −68 | 4,096-row calls on resident stacks too (**bitwise**) |
| 2 | Expert mmq grouping: J against all tokens, empty column tiles launched | −85 (to FT) | −694 (to FT's 4K rate per token) | −30 | a tile map and J from the mean tokens per slot (**bitwise**: J-invariance) |
| 3 | Call size 4,096 against FreeToken's 8,192 | — | −239 (FT's 32K rate per token) | — | 8,192-token passes (bitwise; 2× activation memory) |
| 4 | Dense mmq at 512 rows per call | −25 (measured) | −188 (measured) | −4 | larger qlinear calls (**bounded**: stream-k split points move) |
| 5 | Router: f32 cast + SGEMM + split-K + top-k per 512 chunk | ≈ −13 | ≈ −107 | ≈ −3 | `moe_router@1` serving prompt chunks, an f32 kernel reading bf16 x (**bounded**: its f32 order); target ≈ 15 TFLOPS f32, not FreeToken's bf16 2.3 ms |
| 6 | f32 SwiGLU pass | −21 | −166 | −5 | SwiGLU in gate/up's mmq epilogue (**bitwise**) |
| 7 | f32 combine and the f32 down input's q8_1 | −14 | −112 | −4 | bf16 expert outputs as FreeToken (**bounded**, against D3's f32 expert path) |
| — | Holon ahead: attention, id helper | +24 | +437 | +1 | — |

**Predictions.** Start from the all-4,096 probe, which measures rows 1 and 4, and subtract rows 2, 5 and 6:
- 4K: ≈ 567 ms busy against FreeToken's 584 (≈ 1.03×);
- 32K: ≈ 9.05 s against 9.18 s (≈ 1.015×; ≈ 8.81 s, 1.04×, with row 3);
- 1K: ≈ 174 ms against 172 (≈ 0.99×).

With the bitwise rows alone (1, 2 and 6, dense calls left at 512 rows): ≈ 605 ms (0.97×) at 4K and ≈ 9.34 s (0.98×)
at 32K; without row 6, ≈ 0.93× / 0.97×. These are sums of measured or FreeToken-bounded parts; the design's step 0
re-states them per class before any code.

## FreeToken's and llama.cpp's prefill paths (source study, file:line)

FreeToken `21d9578`, paths under `python/freetoken/`:
- `layers/quantization/moe/gguf.py:92-127` `GgmlMoEKernel.apply`: tokens > `mmvq_max_tokens` (8) go to
  `ggml.mul_mat_id` (gate/up, then `gated_act_and_mul` in bf16), then `mul_mat_id` (down), then `moe_weighted_sum`
  in f32. All intermediates are bf16.
- `kernel/csrc/ggml/VERSION:1`: llama.cpp `8c2134a8` plus `freetoken.patch`. The patch:
  - sizes J on `(T·k + E − 1) / E`;
  - builds a flat tile map of occupied (expert, column tile) pairs, `ceil(T·k / J) + min(E, T·k)` blocks;
  - runs no stream-k for ids;
  - stores bf16 directly.
- `scheduler/config.py:16`: `max_extend_tokens = 8192`.
- Dense GGUF linears call `ggml.mul_mat` (mmvq or mmq; there is no cuBLAS path), `layers/quantization/linear/gguf.py:22-40`.
- Attention is FlashInfer's FA2 paged prefill (`attention/fi.py:117-120`).

llama.cpp (`ggml/src/ggml-cuda/`):
- `mmq.cu:355-487` `ggml_cuda_should_use_mmq`: true on every Turing+ NVIDIA GPU for k-quants, at every batch size.
- `ggml-cuda.cu:2042-2196` `ggml_cuda_mul_mat_id`: mmvq with ids up to 8 tokens; otherwise one mmq call with ids
  (device-sorted). The per-expert dequant + cuBLAS fallback is unreachable for k-quants unless
  `GGML_CUDA_FORCE_CUBLAS` is set.
- Upstream picks J against the mean tokens per expert only on RDNA3/4 (`mmq.cu`'s J_best loop). ik_llama.cpp batches
  `mul_mat_q_id` only up to 32 × experts tokens and above that loops dense mmq per expert
  (`ggml-cuda.cu:2998-3000, 3081-3094`).

## Caveats

- **One card, all-resident.**
  - The A5000 is not a lock class. The 5080's shortfall rows are offload rows at the natural fit, where Holon's
    arena already takes 4,096-row calls (D8), so row 1 does not apply there. Rows 2–7 do.
  - The design's step 0 repeats these runs on a 5080 and a 5090 desktop before re-stating the predictions.
- **Synthetic routing in the per-call bench.** In the engine, the expert GEMMs on real routing (383 ms at 4K with
  4,096-row calls) agree with the bench's sum (8.5 ms per layer → 408 ms).
- **FreeToken's router is bf16.** Its logits come from bf16 tensor cores; Holon's contract sums f32 from x's exact
  values. Row 5's target is therefore an f32 kernel, not FreeToken's 2.3 ms.
- **Holon's 4,096-row resident calls cost memory:** ≈ 0.59 GB more at 4,096 tokens per pass, transient f32
  activations plus the id workspaces (the design's §6). The engine's startup profile sizes the KV pool after it, so
  the timed arms show no peak difference.

## Reproduce

```bash
python spikes/moe_prefill/prefill.py run --arms spikes/moe_prefill/arms.json --model <gguf> --out <dir>/timed \
    --rounds 2 --holon-python <holon venv python> --freetoken-python <freetoken venv python>
python spikes/moe_prefill/prefill.py run --arms spikes/moe_prefill/arms_nsys.json --model <gguf> --out <dir>/nsys \
    --rounds 1 --holon-python <holon venv python> --freetoken-python <freetoken venv python>
nsys export --type sqlite -o <dir>/nsys/<arm>.sqlite <dir>/nsys/<arm>.nsys-rep
python spikes/moe_prefill/breakdown.py <dir>/nsys/<arm>.sqlite > <dir>/nsys/<arm>.json
python spikes/moe_prefill/gemm.py --model <gguf> --layer 0 --tokens 512 1024 2048 4096 --json <dir>/gemm_holon_l0.json
python spikes/moe_prefill/gemm.py --model <gguf> --layer 0 --tokens 512 1024 2048 4096 8192 --freetoken \
    --json <dir>/gemm_ft_l0.json                     # in FreeToken's venv, with gguf-py on PYTHONPATH
python spikes/moe_prefill/gemm.py --model <gguf> --layer 0 --tokens 512 4096 --j-check --json <dir>/jcheck_l0.json
python spikes/moe_prefill/record.py --timed <dir>/timed --nsys <dir>/nsys --gemm <dir>/gemm_*.json <dir>/jcheck_*.json \
    --gpu "RTX A5000" --date 2026-10-18 --commit b5b1e7f > evidence/moe_prefill_rtx-a5000_2026-10-18.json
```

Every command ran pinned to one GPU (`CUDA_VISIBLE_DEVICES`) under a memory cap.
