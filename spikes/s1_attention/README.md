# Spike S1 — attention kernels

Status: done 2026-10-06 · plan.md §3.1 (S1), design.md §5.6, §7.1–§7.4, §7.10 · throwaway code, evidence in `results/`.

## Question

Which paged-attention kernel does plug-serve use per fingerprint class for (1) prefill, (2) non-speculative
decode with host `plan()` in `plan_bounds`, and (3) speculative Seqs at overlap depth 1, where the host does not
know the exact KV length (it lives in the device SeqTable)? Which KV layouts (bf16, fp8-e4m3, q8_0) does each
kernel support, are the kernels run-to-run deterministic, and what does the FlashInfer JIT cost?

Geometries (HF configs in `~/.cache/huggingface/hub`):

| name | model | q heads | kv heads | head_dim | group |
|---|---|---|---|---|---|
| `qwen3_8b` | Qwen3-8B | 32 | 8 | 128 | 4 |
| `qwen3_5_9b` | Qwen3.5-9B full-attention layers | 16 | 4 | 256 | 4 |
| `qwen3_8_27b` | Qwen3.8-27B full-attention layers | 24 | 4 | 256 | **6** |

## Setup

| | a5000 | rtx-pro-6000 |
|---|---|---|
| GPU | RTX A5000 24 GB, sm_86, 64 SMs, 6 MB L2 | RTX PRO 6000 Blackwell Max-Q 96 GB, sm_120, 188 SMs, 128 MB L2 |
| host / GPU index | the A5000 server, `CUDA_VISIBLE_DEVICES=0` | the Blackwell workstation, `CUDA_VISIBLE_DEVICES=6` (shared with other users; < 20 GB used) |
| driver | <version> | <version> |

Software (both): torch 2.11.0+cu130, CUDA 13.0 (nvcc V13.0.48), Triton 3.6.0, flashinfer-python 0.6.12 +
flashinfer-cubin 0.6.12, Python 3.12, the repo `.venv`. (design.md §5.6 mentions FlashInfer 0.7; the env has 0.6.12.)

KV layout everywhere: HND paged (`[pages, kv_heads, page_size, head_dim]`), random page permutation, page sizes
16 and 64. Query dtype bf16, output bf16.

**Timing method.** Each kernel call is planned on the host (where it has a plan), then captured into a CUDA graph
together with enough KV *replicas* (one per "layer", all sharing the plan, like a real model) to exceed 3× L2;
µs = median graph time / replicas over 10 replays. Run-to-run noise (two full bf16 runs per class, 684/900 cells):
median 0.2 % / 0.4 %, p90 2.4 % / 1.4 %, max 13 % / 10 % (`results/bench_bf16_rep2_*.json`).

**Correctness.** fp32 torch reference (`common.reference`) on ragged lengths `[1, 17, 1000, 4097, 513, 2·page,
8191, 64]` incl. lengths not multiple of the page; query row *i* of a sequence sits at position `len − nq + i`
(bottom-right causal, the verify layout). Worst max-abs error vs fp32 for every working kernel: ≤ 0.0100
(bf16 output rounding alone is ≈ 0.002); no NaN anywhere. Per-kernel numbers in `results/tables_*.md`.

Commands (logs in `out/`, JSON in `results/<name>_<class>.json`):

```sh
spikes/s1_attention/run_all.sh 0 a5000                         # A5000 host
ssh <blackwell-host> 'S1_WAIT_FOR_IDLE=s2_gguf .../spikes/s1_attention/run_all.sh 6 sm120'
python summarize.py results/bench_bf16_a5000.json ...         # -> results/tables_<class>.md
```

| script | answers |
|---|---|
| `fi_kernels.py` | FlashInfer paths: CUDA-core decode, tensor-core (FA2) decode, paged/ragged prefill, XQA; `write_csr_on_device` (Triton, 1 launch) |
| `triton_attn.py` | our prototype: paged decode + verify rows, GQA, hd 128/256, page table + lengths read on the device, deterministic split-KV, optional q8_0 KV |
| `bench.py` | Q1/Q3/Q4: correctness + latency matrix (geom × page × B ∈ {1,8,64} × ctx ∈ {1K,8K,32K} × nq ∈ {1,2,3,5}), prefill |
| `graph_test.py` | Q2/Q3: graph replay with changing lengths; host `plan()` cost; FlashInfer padded-CTA hazard |
| `determinism.py` | Q5 · `jit_time.py`, `prebuild.py` Q6 · `sweep_triton.py` Triton config pinning · `bench_q8.py` q8_0 KV · `prep_cost.py` device CSR cost |

Kernel names in the tables:

| name | what | host work per step |
|---|---|---|
| `fi_decode_cudacore` | `BatchDecodeWithPagedKVCacheWrapper(use_cuda_graph=True)` | `plan()` with exact lengths |
| `fi_decode_tc` | same, `use_tensor_cores=True` (FA2 prefill kernel) | `plan()` with exact lengths |
| `fi_paged_prefill` | `BatchPrefillWithPagedKVCacheWrapper(use_cuda_graph=True)`, nq causal rows per seq | `plan()` with exact lengths |
| `*_devmeta` | same FA2 kernel, **non-graph** wrapper planned once with `disable_split_kv=True`; CSR (`indptr/indices/last_page_len`) rewritten in-graph from device page table + lengths | none |
| `triton` | `triton_attn.py`; splits chosen from the graph bucket (B, length upper bound), never from lengths | none |
| `xqa` | `flashinfer.xqa` (TRT-LLM XQA), device `seq_lens` + page table, spec mask for nq > 1 | none (SM90+ only) |

## Results

### Q1 — decode, bf16 KV, page 16, µs per layer call

Page 64 is within −18 %/+18 % of page 16 (mostly ≤ 5 %; full tables in `results/tables_*.md`). `—` = unsupported.

| geom | B | ctx | A5000 fi_decode_cudacore | A5000 fi_decode_tc | A5000 fi_decode_tc_devmeta | A5000 triton | PRO6000 fi_decode_cudacore | PRO6000 fi_decode_tc | PRO6000 fi_decode_tc_devmeta | PRO6000 triton | PRO6000 xqa |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3_8b | 1 | 1K | 33.6 | 16.8 | 25.0 | 11.9 | 11.7 | 12.2 | 21.4 | 7.7 | 12.0 |
| qwen3_8b | 1 | 8K | 61.4 | 62.5 | 167 | 57.3 | 30.7 | 32.6 | 140 | 28.5 | 31.6 |
| qwen3_8b | 1 | 32K | 208 | 205 | 643 | 199 | 105 | 101 | 566 | 97.5 | 97.9 |
| qwen3_8b | 8 | 1K | 61.4 | 63.5 | 58.4 | 57.3 | 29.9 | 32.6 | 27.3 | 27.8 | 28.4 |
| qwen3_8b | 8 | 8K | 397 | 391 | 389 | 386 | 199 | 190 | 187 | 191 | 183 |
| qwen3_8b | 8 | 32K | 1561 | 1537 | 1539 | 1518 | 769 | 715 | 701 | 732 | 703 |
| qwen3_8b | 64 | 1K | 429 | 406 | 395 | 386 | 200 | 189 | 184 | 184 | 178 |
| qwen3_8b | 64 | 8K | 3123 | 3065 | 3060 | 3048 | 1529 | 1401 | 1395 | 1415 | 1357 |
| qwen3_8b | 64 | 32K | 12524 | 12142 | 12131 | 12154 | 6244 | 5606 | 5646 | 5821 | 6002 |
| qwen3_5_9b | 1 | 1K | 57.5 | 17.0 | 32.8 | 12.3 | 31.1 | 12.8 | 28.8 | 8.9 | 17.4 |
| qwen3_5_9b | 1 | 8K | 66.6 | 68.6 | 214 | 59.4 | 36.2 | 36.6 | 191 | 30.4 | 36.0 |
| qwen3_5_9b | 1 | 32K | 214 | 209 | 830 | 203 | 117 | 107 | 753 | 99.1 | 106 |
| qwen3_5_9b | 8 | 1K | 59.4 | 66.6 | 59.4 | 58.4 | 34.2 | 34.2 | 32.9 | 29.0 | 29.6 |
| qwen3_5_9b | 8 | 8K | 401 | 393 | 390 | 389 | 202 | 191 | 219 | 192 | 186 |
| qwen3_5_9b | 8 | 32K | 1567 | 1534 | 1533 | 1524 | 785 | 714 | 841 | 784 | 703 |
| qwen3_5_9b | 64 | 1K | 409 | 402 | 396 | 390 | 203 | 190 | 186 | 188 | 179 |
| qwen3_5_9b | 64 | 8K | 3155 | 3067 | 3058 | 3070 | 1533 | 1404 | 1402 | 1410 | 1368 |
| qwen3_5_9b | 64 | 32K | 12631 | 12178 | 12161 | 12209 | 6447 | 5578 | 5597 | 5640 | 6122 |
| qwen3_8_27b | 1 | 1K | — | 17.0 | 32.4 | 12.3 | — | 12.8 | 28.8 | 9.1 | 17.6 |
| qwen3_8_27b | 1 | 8K | — | 70.7 | 218 | 60.4 | — | 36.3 | 191 | 31.7 | 36.1 |
| qwen3_8_27b | 1 | 32K | — | 208 | 830 | 202 | — | 107 | 752 | 99.3 | 106 |
| qwen3_8_27b | 8 | 1K | — | 68.6 | 60.4 | 59.4 | — | 34.7 | 33.1 | 29.7 | 29.8 |
| qwen3_8_27b | 8 | 8K | — | 395 | 393 | 392 | — | 192 | 219 | 194 | 186 |
| qwen3_8_27b | 8 | 32K | — | 1529 | 1529 | 1521 | — | 712 | 838 | 782 | 704 |
| qwen3_8_27b | 64 | 1K | — | 403 | 397 | 390 | — | 191 | 186 | 189 | 180 |
| qwen3_8_27b | 64 | 8K | — | 3074 | 3070 | 3028 | — | 1406 | 1400 | 1411 | 1374 |
| qwen3_8_27b | 64 | 32K | — | 12190 | 12151 | 12098 | — | 5549 | 5542 | 5612 | 6105 |

Large-batch decode is at the memory roofline: 64 × 32K reads 8.6 GB per layer → 0.71 TB/s on A5000 (peak 0.77)
and 1.55 TB/s on the PRO 6000 Max-Q (peak 1.79).

### Q1 — verify (k + 1 = nq query rows per sequence), bf16 KV, page 16, µs

nq = 3 (k = 2):

| geom | B | ctx | A5000 fi_paged_prefill | A5000 fi_paged_prefill_devmeta | A5000 triton | PRO6000 fi_paged_prefill | PRO6000 fi_paged_prefill_devmeta | PRO6000 triton | PRO6000 xqa |
|---|---|---|---|---|---|---|---|---|---|
| qwen3_8b | 1 | 1K | 16.8 | 24.8 | 11.9 | 12.4 | 22.0 | 8.0 | 15.0 |
| qwen3_8b | 1 | 32K | 205 | 629 | 200 | 101 | 593 | 98.1 | 103 |
| qwen3_8b | 8 | 8K | 412 | 389 | 388 | 215 | 191 | 193 | 190 |
| qwen3_8b | 64 | 8K | 3173 | 3075 | 3058 | 1926 | 1397 | 1422 | 1391 |
| qwen3_8b | 64 | 32K | 14635 | 12172 | 12165 | 8116 | 5647 | 5925 | 6271 |
| qwen3_5_9b | 1 | 1K | 17.2 | 32.6 | 12.7 | 13.2 | 29.0 | 9.4 | 24.5 |
| qwen3_5_9b | 1 | 32K | 212 | 809 | 204 | 107 | 745 | 99.7 | 116 |
| qwen3_5_9b | 8 | 8K | 394 | 391 | 392 | 198 | 219 | 195 | 196 |
| qwen3_5_9b | 64 | 8K | 3043 | 3070 | 3082 | 1497 | 1403 | 1416 | 1404 |
| qwen3_5_9b | 64 | 32K | 12125 | 12176 | 12222 | 6463 | 5756 | 5664 | 6388 |
| qwen3_8_27b | 1 | 1K | 21.7 | 64.1 | 14.3 | 18.1 | 64.5 | 10.5 | 24.7 |
| qwen3_8_27b | 1 | 32K | 207 | 1944 | 208 | 106 | 1962 | 100 | 117 |
| qwen3_8_27b | 8 | 8K | 395 | 476 | 396 | 201 | 501 | 202 | 197 |
| qwen3_8_27b | 64 | 8K | 3047 | 3037 | 3057 | 1493 | 1498 | 1419 | 1408 |
| qwen3_8_27b | 64 | 32K | 12127 | 12106 | 12196 | 6446 | 6880 | 5974 | 6425 |

nq = 5 (k = 4):

| geom | B | ctx | A5000 fi_paged_prefill | A5000 fi_paged_prefill_devmeta | A5000 triton | PRO6000 fi_paged_prefill | PRO6000 fi_paged_prefill_devmeta | PRO6000 triton | PRO6000 xqa |
|---|---|---|---|---|---|---|---|---|---|
| qwen3_8b | 1 | 1K | 16.2 | 33.2 | 13.9 | 12.6 | 31.0 | 8.7 | 15.2 |
| qwen3_8b | 1 | 32K | 206 | 925 | 203 | 103 | 903 | 100 | 104 |
| qwen3_8b | 8 | 8K | 417 | 388 | 392 | 216 | 248 | 199 | 190 |
| qwen3_8b | 64 | 8K | 3500 | 3127 | 3112 | 1924 | 1489 | 1430 | 1399 |
| qwen3_8b | 64 | 32K | 14778 | 12382 | 12410 | 8151 | 6582 | 6235 | 6311 |
| qwen3_5_9b | 1 | 1K | 21.7 | 64.7 | 14.3 | 18.3 | 64.5 | 10.5 | 24.7 |
| qwen3_5_9b | 1 | 32K | 208 | 1946 | 209 | 106 | 1961 | 102 | 117 |
| qwen3_5_9b | 8 | 8K | 395 | 478 | 397 | 201 | 500 | 202 | 196 |
| qwen3_5_9b | 64 | 8K | 3049 | 3040 | 3053 | 1496 | 1501 | 1432 | 1413 |
| qwen3_5_9b | 64 | 32K | 12130 | 12115 | 12171 | 6460 | 6868 | 5959 | 6425 |
| qwen3_8_27b | 1 | 1K | 22.3 | 64.7 | 15.6 | 18.9 | 64.5 | 11.2 | 25.1 |
| qwen3_8_27b | 1 | 32K | 209 | 1949 | 213 | 108 | 1961 | 102 | 117 |
| qwen3_8_27b | 8 | 8K | 399 | 479 | 402 | 202 | 504 | 204 | 197 |
| qwen3_8_27b | 64 | 8K | 3063 | 3048 | 3064 | 1502 | 1505 | 1432 | 1421 |
| qwen3_8_27b | 64 | 32K | 12149 | 12122 | 12216 | 6495 | 6907 | 6056 | 6496 |

Aggregate over all 216 decode + verify cells per class (time / best host-plan FlashInfer path of the same cell):

| kernel | a5000 median [min, max] | rtx-pro-6000 median [min, max] |
|---|---|---|
| `triton` | 0.98 [0.66, 1.09] | 0.95 [0.57, 1.10] |
| `xqa` | — (SM90+ only) | 0.97 [0.70, 1.91] (slow at B = 1, short ctx) |
| `fi_*_devmeta` (no split-KV) | 1.00 [0.81, 9.4] | 1.01–1.09 [0.68, 18.5] (slow at small B, long ctx) |

### Q1 — prefill (first chunk, causal, q = kv = ctx), bf16, µs and attention TFLOPS

Ragged (K/V from the projection) and paged (K/V read back from page-16 cache) are within ±11 %.

| geom | B | ctx | A5000 ragged µs | A5000 paged µs | A5000 TFLOPS (paged) | PRO6000 ragged µs | PRO6000 paged µs | PRO6000 TFLOPS (paged) |
|---|---|---|---|---|---|---|---|---|
| qwen3_8b | 1 | 1K | 144 | 145 | 59 | 90.6 | 94.0 | 91 |
| qwen3_8b | 1 | 8K | 6581 | 6661 | 83 | 2437 | 2491 | 221 |
| qwen3_8b | 1 | 32K | 110086 | 110581 | 80 | 36434 | 38259 | 230 |
| qwen3_8b | 8 | 8K | 52866 | 53309 | 83 | 19074 | 19246 | 229 |
| qwen3_8b | 64 | 1K | 7220 | 7077 | 78 | 3355 | 3007 | 183 |
| qwen3_5_9b | 1 | 8K | 7820 | 8077 | 68 | 2866 | 2955 | 186 |
| qwen3_5_9b | 1 | 32K | 144457 | 144527 | 61 | 44760 | 46930 | 187 |
| qwen3_8_27b | 1 | 1K | 225 | 226 | 57 | 113 | 118 | 110 |
| qwen3_8_27b | 1 | 8K | 12226 | 12081 | 68 | 4056 | 4162 | 198 |
| qwen3_8_27b | 1 | 32K | 217665 | 217426 | 61 | 68662 | 70249 | 188 |
| qwen3_8_27b | 64 | 1K | 13036 | 13254 | 62 | 5833 | 5331 | 155 |

### Q2 — CUDA graph replay with host `plan()` between replays

`graph_test.py`: B = 8, page table sized for 8192 tokens, graph captured once, then 7 steps whose lengths grow by
1..nq, jump to random lengths up to the bound, shrink to the minimum and go to the bound; `plan()` on the host
before each replay; compared to the fp32 reference. **PASS on both classes** for `fi_decode_cudacore` (group 4),
`fi_decode_tc` and `fi_paged_prefill` (nq = 3), qwen3_8b and qwen3_8_27b (worst max-abs over all replay tests
incl. Q3: 0.011, tolerance 0.02). In CUDA-graph
mode FlashInfer's split decision depends only on the batch size (padded grid fixed at plan time), which is why
replays stay valid as lengths change.

Host cost of one `plan()` from pinned host CSR arrays (ctx 8K), µs — paid per step per attention geometry:

| B | tc decode | verify (paged prefill, nq 3) | CUDA-core decode |
|---|---|---|---|
| 1 | 82–99 | 107–254 | 81–83 |
| 8 | 100–104 | 142–146 | 83–85 |
| 64 | 226–228 | 359–365 | 112 |

### Q3 — device-side lengths (spec Seqs at depth 1)

Same replay test, but nothing exact on the host between replays (only the device page table / lengths change):

| path | how | result (both classes) |
|---|---|---|
| `triton` (nq 1, 3, 5) | reads `page_table[B, max_pages]` and `lens[B]` directly | PASS |
| `fi_*_devmeta` (nq 1, 3) | FA2, non-graph wrapper, `disable_split_kv=True`, planned **once**; the plan is then independent of KV lengths; `write_csr_on_device` rewrites the aliased CSR buffers in-graph (1 Triton launch, **1.7–3.6 µs per step**, `prep_cost.py`) | PASS |
| `xqa` (nq 1, 3; sm_120 only) | native device `seq_lens` | PASS |
| FlashInfer **upper-bound plan** (`fi_decode_cudacore`, `fi_decode_tc`, `fi_paged_prefill` nq 3, 5) | CUDA-graph wrapper with split-KV; host `plan()` with an *upper bound* per Seq (actual + up to nq−1, what the host mirror knows) and the exact CSR written in-graph from the device lengths | PASS — FA2/decode kernels clamp each KV chunk to the actual `kv_len` and empty chunks merge correctly. Keeps split-KV, but still a host `plan()` per step and relies on undocumented kernel behaviour |

**Hazard found:** the CUDA-graph FlashInfer wrapper with `disable_split_kv=True` launches
`padded_batch_size = max(2·SMs/kv_heads, tiles)` CTAs (16 on A5000 / 47 on PRO 6000 for B = 1, of which 1 is real)
and, because `block_valid_mask` is only set when splitting, the padded CTAs read request/tile indices that the plan
never wrote (stale pinned staging memory). Zeroed staging → duplicate (benign) work; poisoned staging (0x7F) →
illegal memory access (`graph_test.py` `hazard`); an earlier smoke run of this configuration hung the GPU. Hence `devmeta` uses
the non-graph wrapper (captured into our graph anyway).

**Gap Triton vs FlashInfer:** see the aggregate table above — the Triton prototype is at parity (median 0.98× on
A5000, 0.95× on PRO 6000), up to 1.5–1.8× faster at B = 1 / short context (fewer launches, split-KV tuned to the
bucket) and at worst 1.09× / 1.10× slower (PRO 6000: 8 × 32K, hd 256, where FlashInfer's tensor-core decode wins
by 10 %). The no-split FlashInfer device path is 3–18× slower at B = 1 and long context.

Pinned Triton configs (`sweep_triton.py`, 36 configs × 8 shapes per head_dim; chosen = worst case within 3 % (sm_86) /
8–13 % (sm_120) of the per-shape best): sm_86 hd128 `BLOCK_N 64, warps 4, stages 1`, hd256 `32/4/3`; sm_120 hd128
`64/4/1`, hd256 `64/4/2`. Splits = min(⌈2·SMs / (B·kv_heads)⌉, ⌈len_bound / 64⌉).

### Q4 — KV layouts

fp8-e4m3 KV (scale 1.0), µs, page 16 (full tables incl. nq 2/5 in `results/tables_*.md`):

| geom | B | ctx | A5000 fi_decode_cudacore | A5000 fi_decode_tc | A5000 fi_decode_tc_devmeta | PRO6000 fi_decode_cudacore | PRO6000 fi_decode_tc | PRO6000 fi_decode_tc_devmeta | PRO6000 xqa |
|---|---|---|---|---|---|---|---|---|---|
| qwen3_8b | 1 | 1K | 139 | 15.4 | 22.9 | 11.4 | 11.5 | 18.7 | 11.2 |
| qwen3_8b | 1 | 32K | 916 | 115 | 571 | 71.1 | 60.5 | 468 | 59.8 |
| qwen3_8b | 8 | 8K | 1614 | 208 | 228 | 122 | 109 | 138 | 106 |
| qwen3_8b | 64 | 8K | 15346 | 1571 | 1575 | 967 | 783 | 781 | 787 |
| qwen3_8b | 64 | 32K | 62386 | 6196 | 6183 | 3974 | 3027 | 3146 | 3382 |
| qwen3_8_27b | 1 | 1K | — | 14.1 | 36.0 | — | 12.1 | 30.8 | 14.9 |
| qwen3_8_27b | 1 | 32K | — | 128 | 955 | — | 65.5 | 823 | 64.7 |
| qwen3_8_27b | 8 | 8K | — | 234 | 256 | — | 108 | 223 | 102 |
| qwen3_8_27b | 64 | 8K | — | 1694 | 1687 | — | 801 | 797 | 768 |
| qwen3_8_27b | 64 | 32K | — | 6610 | 6604 | — | 3070 | 3068 | 3189 |

fp8 verify, nq = 3:

| geom | B | ctx | A5000 fi_paged_prefill | A5000 fi_paged_prefill_devmeta | PRO6000 fi_paged_prefill | PRO6000 fi_paged_prefill_devmeta | PRO6000 xqa |
|---|---|---|---|---|---|---|---|
| qwen3_8b | 1 | 32K | 116 | 573 | 60.8 | 461 | 65.7 |
| qwen3_8b | 8 | 8K | 519 | 231 | 228 | 137 | 109 |
| qwen3_8b | 64 | 32K | 18493 | 6181 | 7632 | 3475 | 3480 |
| qwen3_8_27b | 1 | 32K | 202 | 3153 | 84.0 | 2551 | 76.5 |
| qwen3_8_27b | 8 | 8K | 399 | 830 | 160 | 652 | 109 |
| qwen3_8_27b | 64 | 32K | 14174 | 14154 | 5745 | 6611 | 3700 |

q8_0 KV in the Triton prototype (`bench_q8.py`; int8 + one fp16 scale per 32 elements as GGUF q8_0, dequantized
in registers; configs not re-tuned for q8_0; FI fp8 column = `fi_decode_tc` / `fi_paged_prefill` with fp8 KV;
error = kernel vs fp32 reference on the dequantized cache):

| class | geom | B | ctx | nq | Triton bf16 µs | Triton q8_0 µs | FI fp8 µs | q8_0 max-abs |
|---|---|---|---|---|---|---|---|---|
| a5000 | qwen3_8b | 1 | 32K | 1 | 213 | 170 | 113 | 0.0001 |
| a5000 | qwen3_8b | 64 | 8K | 1 | 3306 | 1684 | 1578 | — |
| a5000 | qwen3_8b | 64 | 8K | 3 | 3056 | 1949 | 4449 | — |
| a5000 | qwen3_8_27b | 1 | 32K | 1 | 218 | 164 | 122 | 0.0001 |
| a5000 | qwen3_8_27b | 8 | 8K | 3 | 433 | 332 | 407 | 0.0003 |
| a5000 | qwen3_8_27b | 64 | 8K | 1 | 3303 | 2394 | 1564 | — |
| a5000 | qwen3_8_27b | 64 | 8K | 3 | 3048 | 2544 | 3494 | — |
| rtx-pro-6000 | qwen3_8b | 64 | 8K | 1 | 1413 | 828 | 779 | — |
| rtx-pro-6000 | qwen3_8_27b | 1 | 32K | 1 | 98.7 | 85.4 | 65.6 | 0.0001 |
| rtx-pro-6000 | qwen3_8_27b | 64 | 8K | 1 | 1410 | 1047 | 793 | — |
| rtx-pro-6000 | qwen3_8_27b | 64 | 8K | 3 | 1416 | 1152 | 1318 | — |

Support matrix (✓ measured correct; speed relative to bf16 of the same kernel):

| kernel | bf16 | fp8-e4m3 sm_86 | fp8-e4m3 sm_120 | q8_0 / q4_0 | GQA group 6 |
|---|---|---|---|---|---|
| FI CUDA-core decode | ✓ | ✓ but 4–5× **slower** than bf16 (software fp8 convert) | ✓ ≈ 0.65× | ✗ | **✗** ("Unsupported group_size: 6") |
| FI FA2 tc decode / paged prefill | ✓ | ✓ decode ≈ 0.5×; verify: split path slower than bf16, no-split ≈ 0.5× (group 4), ≈ 1.15× (group 6) | ✓ decode ≈ 0.55×; verify split path ≈ 0.9×, no-split ≈ 0.6× (group 4), ≈ 1.0× (group 6) | ✗ | ✓ |
| XQA | — | — | ✓ ≈ 0.55× (best fp8 verify) | int8 / NVFP4 in API, not tested | ✓ |
| Triton prototype | ✓ | not implemented (Triton 3.6 rejects `fp8e4nv` on sm_86 → uint8 bit-decode needed) | not implemented (native fp8 load possible) | q8_0 ✓ ≈ 0.5–0.85×; q4_0 = same pattern with 4-bit unpack | ✓ |

### Q5 — determinism

`determinism.py`, both classes: for every path (FI CUDA-core/tc decode, paged prefill verify, both `devmeta`
variants, Triton, XQA, ragged prefill), at 1 × 32K (split-KV active) and a ragged batch of 8 up to 32K, nq 1 and 3:
**5 eager re-plan+runs and 20 graph replays are byte-identical** (`torch.equal`), and eager == graph for the same
plan. No path showed a run-to-run difference, including the split-KV merges (FlashInfer `merge_states`, XQA
multi-block, our Triton merge kernel, which reduces splits in a fixed order without atomics).

What changes bytes (bounded comparisons, design §7.10): the split decision. FlashInfer's split depends on the batch
(and, outside graph mode, on lengths); Triton's on the graph bucket (B, length bound); hence a sequence's output
alone vs inside a batch differs for every split-KV path (`batch_invariant_seq0 = False`) — only the
`disable_split_kv` FlashInfer path is batch-invariant. `fixed_split_size` (tensor-core path only) would make
FlashInfer batch-invariant but is documented as not CUDA-graph safe.

### Q6 — JIT

Cold compile + load per module in a fresh cache (`jit_time.py`, 96-core hosts, sequential), warm load:

| module | a5000 cold | rtx-pro-6000 cold | warm load | cache size |
|---|---|---|---|---|
| CUDA-core decode bf16 hd128 / hd256 | 10.0 / 9.6 s | 7.5 / 7.3 s | 20–30 ms | 1.2 MB |
| CUDA-core decode fp8 hd128 / hd256 | 36.4 / 24.1 s | 8.1 / 7.9 s | 20–30 ms | 1.3–1.4 MB |
| FA2 prefill (= tc decode, verify, ragged) bf16 hd128 / hd256 | 13.0 / 12.5 s | 12.6 / 13.7 s | 25 ms | 5.3–5.9 MB |
| FA2 prefill fp8 hd128 / hd256 | 16.3 / 25.5 s | 14.4 / 17.0 s | 25 ms | 6.7–8.0 MB |
| XQA (per page size × head_dim × group × nq × kv dtype) | — | 5.6–5.7 s | **200–250 ms** | 0.7 MB |

Pre-build: `flashinfer.jit.build_jit_specs([gen_batch_prefill_module(...), gen_batch_decode_module(...),
gen_xqa_module(...)])` builds all specs in one parallel ninja run (`prebuild.py`; `MAX_JOBS` caps jobs,
`FLASHINFER_CUDA_ARCH_LIST="8.6 12.0"` cross-builds): the 8 FA2/decode modules in 28 s (a5000), all 56
modules incl. 48 XQA variants in 28.6 s (sm_120). Cache location:
`$FLASHINFER_WORKSPACE_BASE/.cache/flashinfer/<version>/<arch>/cached_ops/<module>/` (default base `$HOME`; arch
dir `86`, `120f`); ninja must be on `PATH`. A read-only mount of that directory is enough at run time (warm load
only `dlopen`s). Triton kernels cache under `TRITON_CACHE_DIR`.

## Decision

Per fingerprint class (consumer sm_89/sm_120 cards not measured — see open issues):

| | **a5000 (sm_86)** | **rtx-pro-6000 (sm_120)** |
|---|---|---|
| prefill (eager, v0) | FlashInfer FA2 `BatchPrefillWithPagedKVCacheWrapper` (paged for every chunk; ragged gives nothing) | same |
| non-spec decode (depth ≥ 1, host `plan()` in `plan_bounds`) | FlashInfer FA2 **tensor-core** decode (`use_tensor_cores=True`, CUDA-graph wrapper). Not the CUDA-core decode: no group 6, up to 3.4× slower at B = 1, fp8 4–5× slower than bf16 on sm_86 | same (XQA is an equal-speed alternative except at B = 1, kept as candidate) |
| spec at depth 1 (verify rows, lengths on device) | **our Triton paged attention** (`triton_attn.py` → kernel plugin): no host plan, deterministic, parity with FlashInfer (0.66–1.09×), same kernel for decode and verify | **our Triton paged attention** (0.57–1.10×). XQA is the borrowed alternative (device lengths, fp8), slower at B = 1 |
| KV layouts | bf16: FA2 + Triton. fp8: FA2 tc decode (≈ 0.5×) for non-spec decode only — fp8 verify has no good FlashInfer path on sm_86 (split path slower than bf16; no-split ≈ 0.5× only for group 4 at B ≥ 8, never faster for group 6). Quantized KV for spec → Triton: q8_0 measured, fp8 needs a uint8 decode, q4_0 same pattern | bf16: FA2 + Triton (+ XQA). fp8: FA2 tc decode, XQA for verify (best). q8_0/q4_0: Triton |
| fallback for spec at depth 1 | FlashInfer upper-bound plan + device CSR (passes; keeps split-KV; bf16) | XQA (bf16 and fp8) |

Rationale: the Triton kernel is the only path that works on both classes, all three geometries, needs no host
plan, is deterministic and can carry GGUF-style quantized KV — and costs nothing in speed. M0 may therefore also
consider Triton for non-spec decode (it would remove the 80–365 µs host `plan()` per step and the FlashInfer
staging-buffer hazards); this spike keeps FlashInfer there as the plan prescribes, since fp8 KV and prefill need it
anyway.

## Open issues

1. **Consumer classes not measured** (4090 sm_89, 5080/5090 sm_120 with 64–96 MB less L2 and far fewer SMs);
   the RTX PRO 6000 Max-Q is not a consumer proxy. Re-run `run_all.sh` on consumer desktops; re-sweep Triton configs and
   the split heuristic there.
2. FlashInfer `plan()` stages metadata in one pinned buffer per wrapper and copies it with `cudaMemcpyAsync`; at
   depth ≥ 1 a second `plan()` can overwrite the staging buffer before the previous copy ran. Not tested here —
   M0 should ping-pong two wrappers per geometry (or record an event) and test it.
3. The upper-bound plan path depends on undocumented FlashInfer behaviour (chunk clamping, empty-chunk merge);
   if adopted it needs an L0 test pinned to the FlashInfer version.
4. Do not combine `use_cuda_graph=True` with `disable_split_kv=True` in FlashInfer 0.6.12 (padded CTAs read
   unwritten indices) — worth an upstream issue.
5. Triton prototype gaps: fp8 KV, prefill/chunked-prefill (FlashInfer does it today), q8_0 tuning (currently
   0.5–0.85× of bf16, FI fp8 reaches ≈ 0.5×), q4_0, sliding window/soft-cap (not needed for Qwen); split count
   heuristic is per bucket, so outputs are batch-variant (fine for run-identical, bounded otherwise).
6. FA2 fp8 verify with split-KV is slower than bf16 on both classes (e.g. qwen3_8b 64 × 32K: 18.5 ms vs 14.6 ms
   bf16 on A5000); with group 6 on sm_86 even the no-split path is not faster than bf16.
7. a7 GPU 6 is shared; the sm_120 run waited for the S2 spike to be idle, noise measured above.
