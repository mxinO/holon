# x3c_kernels — step 3c: the MoE decode glue kernels

Records: `evidence/x3c_kernels_rtx-5090_2026-10-17.json`, `evidence/x3c_kernels_rtx-a5000_2026-10-17.json`.
Design: `docs/v1-offload.md` §14 "Step 3c as built". The gap these close: `spikes/x4_gap/README.md` rows 5–6 and the
RTX 5090 all-resident breakdown (glue, router, q8_1 quantization).

## What changed (one commit each, then the review's fixes)

| Change | Commit | Class | What |
|---|---|---|---|
| residual | `8e606ec` | bitwise | The layer's residual add runs in the MoE combine's kernel: `moe_combine@1`'s optional `add(y, weights, residual, out)` writes `residual + bf16(Σ)`, added in f32 (`add.rn`, no fast-math flush) and rounded again — what `h + ffn(x)` gave. `std.moe`'s `add(x, residual)`; `std.blocks.prenorm` calls an ffn's `add` where it has one. |
| router | `5473c4a` | bounded | `moe_router@1` (std.moe contract; oracle `moe_topk` of the f32 linear; CPU provider): for decode and verify calls (≤ 16 rows) one launch reads the bf16 activations, sums the f32 router GEMV (four warps per expert, quarters of hidden, xor trees, added in order) and runs `topk_moe_cuda`'s softmax selection in the last block of each 4-row tile (`holon_topk_select`, the same arithmetic, register-only). The f32 cast, cuBLAS's GEMV (dot + reduce on sm_120) and the separate top-k leave decode steps. Prompt chunks (> 16 rows) compose as before, so prefill is unchanged. |
| prepare (q8_1) | `3015a3c` | bitwise vs router | The router's launch also writes the gate/up input's q8_1 activations (`holon_quantize_q8_1_block`, `quantize_q8_1`'s arithmetic): `moe_router@1`'s optional `route_prepare(x, router, experts)` returns a `Prepared` handle (the buffer and generations `qlinear`'s `prepare_input` shares per `in`, M0.7 D6) for its provider's `linear_experts` instance on the mmvq path; mmvq with ids reads a pre-quantized src1 (`vendor.patch`). |
| review fixes | `7ea4b34` | — | The judge's expert stale-q8_1 plants take a prepared input; one routing path (`MoE._route` → weights, ids, gate/up input) so the routing plants reach it; f32 input 16-byte aligned, `top_k ≤ 32` served; contract wording; tests (exact ties, the q8_1 loop's rounds, a handle's buffer shown read). Snapshot-identical to `3015a3c`. |

Core is unchanged. The vendored changes are in `vendor.patch` (`vendor.sh --check` passes): `moe-weighted-reduction.cu`
(residual), `topk-moe.cu/.cuh` (the router kernel, appended; `topk_moe_cuda` untouched), `mmvq.cu` (Q8_1 src1 with
ids).

## Method

- **Vehicle:** Qwen3-30B-A3B Q4_K_M, batch 1, graphed decode.
- **`ab.py`:** M0.7's keep-or-revert method (`bench/fusion_ab.py`). One engine per deployment. The changes are
  switched off in process by restoring the code they replaced (the composed router, `h + ffn(x)`, no prepared input),
  and each arm's graphs are captured again. A row is the per-step wall over 256 graphed decode steps after a prompt
  of 1,024 / 4,096 random tokens; prefill is one eager prompt of 512 / 4,096 tokens (median of 3). One warm-up round,
  then 8 rounds in ABBA order. A change's saving is against the previous arm, and its sd is that of the per-round
  differences. **Kept if ≥ max(0.5%, 3 sd) on some row.**
- **End to end:** `bench/holon_tg.py` (tg128 at depth 1K / 4K, 3 repetitions), base `c79f968` against `3015a3c`, one
  process per run, 4 rounds per arm in ABBA order.
- **Hardware:** RTX 5090 desktop (sm_120a; clock screen steady, SM p50 2857 MHz, 0% slowdown, 2.3% at the power
  cap); RTX A5000 (sm_86). The offload rows are `moe.store=std.moe.offload@1 moe.cache=*=0.47`, the 5080's natural
  fit, under F11 (`lfu`).
- **nsys:** `profile.py` (`--arm`), 128 decode steps after a 1K prompt; `spikes/x4_gap/nsys_breakdown.py`.

## Results

**Per change, decode ms per token saved (% of the step ± sd), 1K / 4K:**

| Change | RTX 5090 resident | RTX 5090 offload 47% | A5000 resident | A5000 offload 47% | Verdict |
|---|---|---|---|---|---|
| residual (bitwise) | 0.030 / 0.034 (1.16 ± 0.07 / 1.21 ± 0.03%) | 0.031 / 0.033 (1.07 / 1.04%) | 0.036 / 0.041 (0.65 ± 0.43 / 0.69 ± 0.27%) | 0.017 / 0.045 (0.28 / 0.71%, sd 0.4) | **kept** (pays on the 5090; under 3 sd on the A5000) |
| router (bounded) | 0.137 / 0.141 (5.29 ± 0.05 / 5.04 ± 0.03%) | 0.159 / 0.160 (5.46 / 5.14%) | 0.133 / 0.127 (2.45 ± 0.15 / 2.15 ± 0.12%) | 0.150 / 0.125 (2.58 / 2.00%) | **kept**, every row |
| prepare (bitwise) | 0.039 / 0.038 (1.57 ± 0.05 / 1.45 ± 0.04%) | 0.024 / 0.024 (0.86 / 0.80%) | 0.045 / 0.054 (0.85 ± 0.29 / 0.94 ± 0.22%) | 0.027 / 0.042 (0.47 / 0.68%, sd 0.4) | **kept** (5090 rows; A5000 resident 4K) |
| **sum** | **0.206 / 0.213** (2.624 → 2.419 ms; 2.824 → 2.612) | **0.214 / 0.215** | **0.214 / 0.222** | 0.194 / 0.212 | |

Prefill (512 / 4,096 tokens) is unchanged by every change, within ±0.4% (sd 0.1–1.5%).

**End to end (`holon_tg`, tok/s, median of 4 rounds; ms per token saved ± sd):**

| | 1K | 4K |
|---|---|---|
| RTX 5090 resident | 383.4 → **413.3** (×1.078; −0.189 ± 0.001 ms) | 357.1 → **383.5** (×1.074; −0.192 ± 0.002 ms) |
| A5000 resident | 181.7 → 190.2 (×1.047; −0.226 ± 0.034 ms) | 169.3 → 176.2 (×1.041; −0.224 ± 0.019 ms) |
| A5000 offload 47% | 119.6 → 122.0 (−0.21 ± 0.15 ms) | 140.4 → 145.1 (−0.29 ± 0.14 ms) |

On the 5090 `holon_tg` sees 0.19 ms of `ab.py`'s 0.21: its steps include sampling and the host loop.

**nsys, µs per decode token at 1K (profiled; launches per token):**

| Category | 5090 before | 5090 after | A5000 before | A5000 after |
|---|---|---|---|---|
| router (GEMV + top-k) | 289 (144) | 260 (48) | 370 (96) | 358 (48) |
| activation quant (q8_1) | 280 (193) | 165 (145) | 297 (193) | 231 (145) |
| embedding / glue | 183 (127) | 44 (31) | 280 (127) | 72 (31) |
| MoE combine (+ residual) | 67 | 79 | 128 | 144 |
| MoE expert GEMV | 913 | 842 | 2,166 | 2,189 |
| window | 2,668 | 2,449 | 5,628 | 5,383 |

The 5090's expert GEMV row falls by 71 µs without any change to it: under programmatic dependent launch a kernel's
measured duration includes its wait on the previous one (the gate/up mmvq no longer waits on a quantize). That is
also where x4_gap's "stacked gate/up" (+0.07 ms) and part of its "q8_1" row came from: no layout cost was found (the
A5000, without PDL, shows the resident and arena GEMVs within 2% of each other).

## Judge

- **L0:** sm_86 — `test_ggml_cuda_moe.py`, `test_moe_engine.py`, `test_quantized_units.py`, `judge/tests` at
  `7ea4b34`: 228 passed. sm_120a (the 5090) — `test_ggml_cuda_moe.py` at `3015a3c`: 107 passed, 1 skipped.
  - Bitwise checks: the fused add equals `residual + out` (cancelling and subnormal sums too; a flushing add fails it).
    The router's weights and ids equal `moe_topk` on its own logits (ties, NaN rows; a mutated renormalising-sum order
    fails it). Its q8_1 bytes equal `quantize_q8_1`'s. The gate/up output from a handle equals that from x.
  - Bounded checks: the router's logits are within the f32 budget of the f64 product, bf16 and f32 inputs. A row's
    logits are the same at 1 to 16 rows.
- **L2 (router, bounded):** 1-token steps against the vehicle's mmvq floor, A5000. After: top-1 98.569% (floor
  98.536%, budget ≥ 98.286%), KL p99 0.0374 (floor 0.0398), ΔPPL vs floor +0.138% ± 0.108% (budget ≤ 0.52%):
  **PASS**. Before (`c79f968`, cuBLAS): 98.594%, 0.0381, −0.037% ± 0.118%.
- **Snapshots (A5000, resident and `*=0.47`):**
  - residual: identical to `c79f968`;
  - router: differs (bounded, as predicted);
  - prepare and the review fixes: identical to the router commit;
  - offload identical to resident at every commit.
  The existing locks are carried at milestone X's close: decode routing is no longer cuBLAS's bits, and MoE sites now
  name `moe_router@1`.

## Not done, recorded

- **About 30 per-step index and copy kernels in the core executor** (slot mapping, positions, logits indices;
  `core/holon/core/executor.py`): ≈ 40 µs per token on the 5090 and ≈ 70 µs on the A5000, the remaining "glue" row
  above. This is a core change, so it is out of this step's scope.
- **The offload store's per-step edit application** (`std.tiered` residency / resolver: ≈ 18 small kernels per step,
  ≈ 30 µs on the A5000): about 0.5% of an offload step, under the keep rule; not built.
- **The router kernel does not trigger PDL early launch** (`ggml_cuda_pdl_lc`): a possible small gain on sm_90+, not
  measured.
- **The selection tail** (≈ 2.8 µs of the A5000 router's 6.3): a `redux.sync`-based argmax could take ≈ 0.5 µs off,
  under the keep rule alone.

## Reproduce

```bash
python spikes/x3c_kernels/ab.py --model <gguf> --rounds 8 --prefill 512 4096 --json <ab.json>     # + --engine-args for offload
python bench/holon_tg.py --model <gguf> -n 128 -r 3 --depths 1024 4096 --engine-args "--max-seqs 1"  # per commit, ABBA
nsys profile -t cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop -o <rep> \
    python spikes/x3c_kernels/profile.py --model <gguf> --arm base|prepare
python spikes/x3c_kernels/record.py --gpu <class> --date 2026-10-17 --commit 3015a3c --out <record.json> --ab ... --tg ... --nsys ...
```
