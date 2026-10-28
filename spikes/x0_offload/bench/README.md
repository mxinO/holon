# Milestone X, step 0: offload microbenchmarks (M1, M2, M3, M5, M7, prefill)

## Question

`docs/dev/v1-offload.md` §8 fixes what step 0 measures before the offload design is frozen. This directory answers, on
the two bar cards (RTX 5080 and RTX 5090 desktops):

| # | Measure | Decides (§8) |
|---|---|---|
| M1 | H2D: pinned `cudaMemcpyAsync` 1–512 MiB; `cudaMemcpyBatchAsync` of N = 1–64 expert blobs; the zero-copy SM gather (2–128 CTAs) of the same blobs | the fetch path's rate and CTA count; whether the in-graph gather reaches ≥ 0.9× DMA |
| M2 | CPU expert compute on 1–6 threads concurrently with DMA or the SM gather; the per-miss-count split measured | D4's `plan[n]`; whether F5b can gain ≥ 5% |
| M3 | ggml-cpu `vec_dot` experts (Q4_K, Q6_K down, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S), 1–8 tokens per job, 1–7 threads, AVX-512 and AVX2 builds; D7's bitwise contract | F5b's worth; multi-token kernels; worker counts |
| M5 | S6's spin handshake in a FULL graph with a real payload: k ≤ 10 CPU expert jobs per layer, rows 1–8, 48 layers; device- vs host-planned job lists | D5's plan location; `cpu_task@1` |
| M7 | `O_DIRECT` reads of expert-sized ranges at queue depth 1–32 with 1–8 I/O workers from the node-local NVMe | F9's reader |
| prefill | one MoE layer's routed-expert GEMMs (`kernels.ggml_cuda`, mmq with ids) at 64–4096-token chunks | D8's stream-or-stage rule |

## Setup

| | |
|---|---|
| GPUs | RTX 5080 and RTX 5090 (sm_120), PCIe 5.0 x16, headless; one consumer desktop per class (8-core desktop CPU) |
| CPU ISA tier | `x86-avx512` (ggml-cpu built `-march=native`); an AVX2-only build of the same sources for M3's comparison |
| software | CUDA 13.0 container (nvcc 13.0.48, g++ 13.3), torch 2.11.0+cu130, Python 3.12.12; ggml-cpu (static, no OpenMP) from the vendored llama.cpp checkout `spikes/s2_gguf/third_party/llama.cpp` at 51ce9c1 (that clone does not carry `kernels.ggml_cuda`'s pin cbe28bb3); `kernels.ggml_cuda` prebuilt for sm_120 |
| cores | the first physical core keeps the OS, the Python thread (every runtime thread is pinned there) and the GPU feeder; compute threads take the other seven physical cores, one each, SMT siblings idle |
| expert bytes | real GGUF bytes of Qwen3-30B-A3B: Q4_K_M layer 0 (Q4_K gate/up, Q6_K down: 3.06 MB) and layer 6 (Q4_K down: 2.65 MB); UD-IQ3_XXS layer 0 (IQ3_XXS) and layer 25 (IQ3_S). IQ2_XXS / IQ2_XS / IQ2_S and the Maya-S mix (IQ2_XXS gate/up, IQ2_S down) are 16 real Q4_K experts requantized with ggml's quantizer (uniform importance: the bytes and the compute are the type's; accuracy is not measured) |
| host arenas | exact `cudaHostAlloc(Mapped)` (no rounding, D10); each class's experts are tiled over 384 slots (0.46–1.17 GB) so CPU sweeps read DRAM: the desktops' L3 is large enough that a 128-expert sweep (0.39 GB) still hit it in a first run, which was discarded |
| SSD | the desktop's own NVMe (ext4, node-local): one 16 GiB file of random data written with `O_DIRECT`, read, deleted; `test_file_removed: true` in each record |

```bash
cd spikes/x0_offload/bench
CUDA_VISIBLE_DEVICES=0 python bench.py --tag rtx-5080 --out ../../../evidence/x0_bench_rtx-5080_<date>.json \
    --llama <llama.cpp checkout> --build <build dir> --nvme-dir <node-local dir>      # ~20 min
#   --only m1,m3,m2,m2iq,m5,m7,prefill   --merge (update a record)   --quick
python report.py ../../../evidence/x0_bench_rtx-50{8,9}0_<date>.json    # the tables below
```

Files: `bench.py` (driver, one entry point), `csrc/cpu_experts.h` (the CPU expert batch and pool), `csrc/x0.cpp`
(host arena, copies, contention, split, the handshake worker, `O_DIRECT` reader, ggml helpers; pybind),
`csrc/kernels.cu` (gather, flag pair, M5's producer and consumer kernels), `build_ggml.sh`, `report.py`. Evidence:
`evidence/x0_bench_rtx-5080_2026-10-17.json`, `evidence/x0_bench_rtx-5090_2026-10-17.json` (every run's command and
commit in `runs`; M2 and M2-iq on the 5080 were rerun on a second desktop of the class).

## Method

- **M1.** Every rate is GPU time between events around back-to-back repeats behind a 3 ms busy kernel (the host
  runs ahead). Blobs sit at random distinct slots of a 1 GiB mapped arena and land in distinct device staging slots.
  `memcpy ×N` = N `cudaMemcpyAsync`; `BatchAsync` = one `cudaMemcpyBatchAsync` (source access in stream order; the
  legacy default stream is refused, so everything runs on a created stream). The gather kernel walks the blob list
  with 16-byte loads, U loads in flight per thread before the stores (`CTAs × 256 threads, unroll U`), swept over
  CTAs {2…128} × U {2, 4, 8}; "CTAs ≥ 95%" is the fewest CTAs within 5% of the best. The best gather at N = 8 is also
  timed captured in a CUDA graph (decode's form). Staging slots are checked equal to their sources.
- **M3.** One job = one expert: x quantized to the gate/up `vec_dot_type` (Q8_K here), gate and up rows, f32 SwiGLU,
  requantize, down rows, all with ggml-cpu's own `from_float` and `vec_dot` on the GGUF blocks in place, one
  `vec_dot` per (row, token). A pool of persistent spinning threads splits rows (never K): each thread takes 16-row
  chunks from the front of its own share and then steals from the back of the others'. Batches cycle through the 384
  slots; p50 of 384 batches. "8 per batch" = a layer-like batch of 8 experts of 1 token. **Bitwise (D7):** per class
  and build, 3 experts × 8 tokens: each token alone (1 thread) vs inside groups of 2–8 tokens at 1 and 7 threads, and
  one job vs a batch of 4 jobs — 45 comparisons; the outputs' SHA-256 is compared between the AVX-512 and AVX2 builds.
- **M2.** Steady state: a feeder thread keeps a stream busy with 8-blob rounds (DMA or gather; blobs from the upper
  half of the arena) while the compute pool runs one-expert jobs from the lower half; phase 1 measures the CPU under
  GPU traffic, phase 2 the GPU under CPU load (300 ms each); each cell is the median of three interleaved rounds
  (one window can catch a slowed core, below). Split: for n = 1–10 misses and every k, the gather of n − k blobs is
  launched and the CPU computes k experts at once; the path time is max(CPU, gather), median of 35 repeats; `plan[n]`
  is the best k. Measured for the Q4_K_M expert (3.06 MB) and for an IQ2_XS expert (1.36 MB, Flash-Next-like).
- **M5.** S6's worker extended with a real payload: per layer slot, a job list, activations (`rows × 2048` f32) and
  one output block per job in mapped memory; the coordinator is pool thread 0 (4 compute threads) and runs M3's batch.
  Device-planned: a kernel writes the job list (k distinct experts, each with all rows) and x, then signal → wait →
  a kernel reads the k × rows CPU rows back over PCIe and reduces them. Host-planned: a kernel writes the routes
  (rows × top-8), the coordinator builds the job list and answers PLANNED before computing (one more flag round trip:
  the GPU waits for the plan before its own dependent work), then DONE. Spin-kernel waits (S6's default), 48 layers,
  FULL graph, 60 replays. Overhead per layer = (step − the same graph without hand-offs − CPU compute) / 48.
  Overlap: a busy kernel as long as the layer's CPU time on a branch between signal and wait; "exposed" = step minus
  the GPU-only graph with the same busy kernel, per layer. After each run the CPU rows of slot 0 are checked bitwise
  against a fresh 1-thread ggml-cpu run of the same jobs.
- **M7.** Each read is an expert-sized range from a random 16-byte-aligned offset, read as its 4 KiB-aligned
  superset (`O_DIRECT`). `aio`: each of w worker threads keeps qd / w reads in flight with Linux AIO; `pread`: each
  worker issues blocking reads (qd = w). 1.5 s per point; GB/s counts the expert bytes.
- **prefill.** `kernels.ggml_cuda`'s llama.cpp kernels as `LinearExperts` runs them: per 512-token chunk, mmq with
  ids into f32 for the merged gate/up stack, f32 SwiGLU, mmq with ids for down; bf16 activations, uniform random
  top-8 of 128; the layer's real Q4_K_M experts in VRAM; median of 10. The streaming time is the layer's non-resident
  experts' bytes over M1's gather rate.

**A slowed core.** On every desktop, one core at a time (it moved between runs) intermittently ran a compute-bound
expert 1.6–1.7× slower for seconds (IRQ or sibling load; the OS is outside our control). A static row split turns
that into a 1.6× slower batch for every thread count that includes the core; a single shared claim counter absorbed
it but cost 10–20% when no core was slowed; per-thread shares with stealing cost ≤ 3% clean and held the slowed
case to +7% (63 → 69 µs for IQ3_S at 7 threads). The 1-thread column cannot steal: its outliers (5090 native IQ2_XXS
449 µs, IQ2_XS 381 µs, against 270 µs in the AVX2 build) are this effect.

## Results and what they decide

### M1 — the fetch path

Both cards behave alike: pinned copies run at 46.6–48.4 GB/s from 8 MiB up; expert-sized blobs at N ≥ 8 move at
44–48 GB/s with N `cudaMemcpyAsync`, 46.7–48.4 with `cudaMemcpyBatchAsync`, and 46.6–48.3 with the SM gather, which
**saturates PCIe 5.0 with 2–8 CTAs** of 256 threads (`8 × 256, unroll 8` is within 3% of the best at N ≥ 8, 5% at
any N) and runs at 46–48 GB/s captured in a graph. **Gather / best DMA = 0.97–1.02 at N ≥ 8 (0.94–1.06 for a single
blob): the in-graph gather reaches ≥ 0.9× DMA.** F5a: an 8-CTA gather; one 2.86 MB expert lands every ≈ 60 µs, one layer's
non-resident experts (68–96 of 128) in 3.7–6.1 ms.

### M3 — CPU experts (ggml-cpu)

| per expert, 1 token, µs (GB/s) | 1 thread | best (threads) | 8 per batch | 8 tokens, µs per token |
|---|---|---|---|---|
| Q4_K_M, Q6_K down (3.06 MB) | 143–148 | 68–69 (4 thr; 45 GB/s) | 62–63 (49 GB/s) | 19 |
| Q4_K_M, Q4_K down (2.65 MB) | 128–130 | 59–60 (4–6 thr; 45 GB/s) | 54 | 19 |
| IQ3_XXS (1.81 MB) / IQ3_S (2.03 MB) | 358–362 / 405–417 | 60–61 / 67–70 (7 thr; 29–30 GB/s) | 54–56 / 62–64 | 50–56 / 62–64 |
| IQ2_XXS / IQ2_XS / IQ2_S (1.22–1.51 MB) | 254–285 | 44–49 (7 thr; 25–34 GB/s) | 39–43 | 39–42 |
| Maya-S mix (1.31 MB) | 270–276 | 45 (7 thr; 29 GB/s) | 40 | 39–42 |

- Q4_K_M experts are **DRAM-bound at 45 GB/s from 2–4 threads** (49–50 GB/s batched); S6's bf16 GEMV reached
  66–73 GB/s, so one ggml `vec_dot` per (row, token) leaves ≈ 30% of DRAM on the table. Several tokens per job pay
  off (8 tokens: 19 µs per token, 3.6× cheaper).
- **i-quants are compute-bound**: linear in threads up to 7 (25–34 GB/s) and **no cheaper per token with more
  tokens** (ggml's i-quant `vec_dot` decodes the block per token): Strata's and Maya's multi-token i-quant rows
  (admitted only if bitwise equal, D7) are where i-quant CPU speed would come from.
- **AVX-512 vs AVX2 builds: equal speed within noise, and bitwise-equal outputs for all eight classes** (this ggml
  commit has no AVX-512 path that changes these types' reductions).
- **D7 holds: 0 mismatches in 45 checks per class and build** (token alone vs in groups of 2–8, 1 vs 7 threads, one
  job vs a batch of four); M5 adds 40 more exact checks of in-graph CPU rows against a fresh ggml-cpu run.

### M2 — contention and `plan[n]`

With 6 compute threads, the CPU expert takes 63–65 µs alone, **112–114 µs while the SM gather runs (1.8×) and
148–149 µs under DMA (2.3×)**; the gather drops from 47.4–47.7 to 27.3–27.6 GB/s and DMA from 46.5–46.7 to 34.1–34.2. Both read host
DRAM: CPU (27 GB/s) + gather (27.5 GB/s) ≈ 55 GB/s, only 15% above the gather alone. The measured split agrees:

| misses n (Q4_K_M) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| fetch-only µs (both cards) | 67 | 131 | 194 | 258 | 322 | 385 | 450 | 513 | 578 | 640 |
| `plan[n]`, 6 threads, 5080 | 1 | 1 | 2 | 2 | 3 | 3 | 4 | 5 | 5 | 6 |
| `plan[n]`, 6 threads, 5090 | 1 | 1 | 2 | 2 | 3 | 3 | 4 | 4 | 5 | 6 |
| gain over fetch-only, 5080 / 5090 | 9 / 8% | 12 / 14% | 12 / 13% | 14 / 15% | 15 / 14% | 15 / 14% | 14 / 14% | 14 / 14% | 14 / 13% | 14 / 14% |

`plan[n]` ≈ n / 2 on both cards, and the optimum is flat (k ± 1 within 1–3%); 4 threads give gains within 3 points
of 6 threads, 2 threads 9–16% (n ≥ 2). For the IQ2_XS expert the CPU is slower than PCIe per byte but barely
contends (55 vs 49–51 µs with the gather): `plan[n]` ≈ 0.4 n, gain 0–6% at n ≤ 2, 8–15% at n = 3–4, 17–23% at
n ≥ 5 (6 threads).

**F5b's worth (Qwen3-30B-A3B Q4_K_M on the 5080).** A decode token pays 48 layers × fetch(n) on top of the resident
GPU work (≈ 4–5 ms all-resident; estimated from the 5090's 2.6 ms and the UD-IQ3_XXS 5080 lock's 3.8 ms). With the
measured split (6 threads) and 2.2 µs of handshake per layer: n = 1 miss per layer → ≈ 2%; n = 2 → ≈ 5–6%;
n = 4 → ≈ 8%; n = 6 → ≈ 10%. **F5b can clear D4's 5% bar only from ≈ 2 misses per layer (hit rate ≤ ≈ 75%), and
then by 5–10%**: likely at the capped 25% fit, marginal at the natural 47% fit (it depends on M6's hit rate under
the profile).

### M5 — the handshake with a real payload

| | 5080 | 5090 |
|---|---|---|
| overhead per layer, rows 1–2 (k 1–10) | 2.1–2.7 µs | 2.1–2.9 µs |
| rows 4 / rows 8 (k 1–10) | 2.3–3.6 / 2.8–6.5 µs | 2.1–3.8 / 2.6–6.5 µs |
| host-planned − device-planned overhead | −0.38 … +0.02 µs | −0.33 … +0.02 µs |
| exposed with matched GPU work, device / host (worst) | 9.5 / 10.8 µs | 8.0 / 10.1 µs |
| CPU time hidden (worst config) | ≥ 96.6% | ≥ 97.8% |
| stranded slots at stop; CPU rows bitwise | 0; 20 of 20 configs exact | 0; 20 of 20 exact |

The per-layer cost grows with rows × k because the GPU reads k × rows × 2048 f32 rows back over PCIe (655 KB at rows 8,
k 10). **The host-planned variant buys nothing**: its extra round trip is hidden when nothing on the GPU depends on
the plan and costs 0.4–0.9 µs per layer on average when something does. D5's device-planned job list stands.

### M7 — the SSD reader

The desktops' node-local NVMe gives 1.7–2.4 GB/s at queue depth 1 (0.74–0.80 / 1.28–1.33 / 2.9–3.0 ms for a 1.46 /
2.8 / 7.1 MB read) and **saturates at 2.5–2.7 GB/s from queue depth 2 (or 2 blocking `pread` workers)**; deeper queues and
more workers only add latency. F9: 2 I/O workers with blocking `pread` of the aligned superset (or AIO at depth 2–4).
A disk miss costs ≈ 20× a RAM miss (1.3 ms vs 60 µs for 2.8 MB), so the RAM tier comes first.

### Prefill — D8

| expert GEMMs per layer, ms (Q6_K-down layer) | 64 | 256 | 1024 | 4096 tokens |
|---|---|---|---|---|
| RTX 5080 | 0.78 | 1.27 | 2.72 | 10.97 |
| RTX 5090 | 0.42 | 0.71 | 1.50 | 6.00 |

Streaming a layer's non-resident experts at 47.5–48 GB/s takes 4.3–4.4 ms at 47% resident and 6.1–6.2 ms at 25%
(3.7–3.8 / 5.3–5.4 for a Q4_K-down layer). From 64 tokens on, 127–128 of the 128 experts are routed, so routed-only staging moves the same
bytes as streaming and D8's time rule always picks streaming (it overlaps). **Streaming hides under the expert GEMMs
only from ≈ 2K-token chunks on the 5080 and ≈ 3–4K on the 5090** (the rest of the layer's compute, not counted here,
lowers that); below, prefill is PCIe-bound at ≈ 0.2 s per chunk (48 × 4.3 ms) whatever the chunk size, so prefill
chunks should be as large as VRAM allows.

### Implications for the design

- **D4 `plan[n]`:** ≈ round(n / 2) misses to the CPU for Q4_K_M on both cards (table above), ≈ 0.4 n for IQ2_XS;
  flat optimum, so tune's table is robust to ± 1.
- **D6 worker counts:** Q4_K_M 1-token experts saturate DRAM at 2–4 compute threads, and the split gains no more at
  6 than at 4; i-quants and multi-token jobs scale to all 7 non-loop cores. Default 4 compute threads for Q4_K_M
  decode, tune per (model, class) up to 7; the first core stays the loop core (as S6 found); F9's 2 I/O workers
  are separate. The pool must steal rows (bitwise-safe) or one slowed core stalls it by 1.6×.
- **F5b:** ≤ 15% of miss time on these desktops because CPU and PCIe share host DRAM; worth ≥ 5% of decode only
  from ≈ 2 misses per layer; FN/GLM's i-quants contend less and gain up to ≈ 20% of miss time at n ≥ 5.
- **F9:** 2 workers, depth 2; ≈ 2.6 GB/s on these drives.

### Against the design and the research note

1. R §2's "desktop CPU ≈ 68 GB/s, about as fast as Gen5; using both is the only way past either limit": ggml-cpu runs a
   Q4_K_M expert at 45–50 GB/s, and CPU + PCIe together cap near 55 GB/s, so the hybrid split adds ≈ 15% of miss
   throughput, not a second link's worth.
2. D7 / S5: "ggml-cpu's reductions differ by ISA": the AVX-512 and AVX2 builds gave bitwise-equal outputs for all
   eight classes here. Keep `cpu_isa` in the class of F5b locks, but test equality before splitting a lock by tier.
3. D6 (from S6) "≤ 4 compute threads saturate DRAM": true for Q4_K 1-token jobs only; i-quants and multi-token
   jobs are compute-bound and want every core.
4. S2's copy-stream MemOps (F3 placement copies, by DMA) slow concurrent CPU experts 2.3× against the gather's 1.8×:
   with F5b on, placement copies should not overlap the CPU lane, or should use the gather.
5. R §2 assumed NVMe at 7–14 GB/s; these desktops' SSDs read 2.6 GB/s (re-measure F9 on a Gen4/Gen5 SSD).
6. D7 names `kernels.ggml_cuda`'s pin for ggml-cpu; these numbers come from the vendored checkout at 51ce9c1.
7. Not measured here: Strata's and Maya's multi-token CPU kernels (§8 M3 "Strata's kernels beside them"); M6.

## Full tables (`report.py`)

### rtx-5080 (NVIDIA GeForce RTX 5080, CPU x86-avx512)


#### m1

pinned cudaMemcpyAsync, GB/s: 1 MiB 49.2, 2 MiB 49.4, 4 MiB 49.2, 8 MiB 47.9, 16 MiB 48.2, 32 MiB 48.3, 64 MiB 48.4, 128 MiB 48.4, 256 MiB 48.3, 512 MiB 48.4
| blob | N | memcpy ×N | BatchAsync | gather best | config (CTAs×threads, unroll) | CTAs ≥ 95% | gather / best DMA |
|---|---|---|---|---|---|---|---|
| 1.46MB | 1 | 51.5 | 51.4 | 48.6 | 8x256u4 | 2 | 0.95 |
| 1.46MB | 8 | 44.2 | 46.7 | 46.6 | 8x256u8 | 4 | 1.00 |
| 1.46MB | 64 | 45.1 | 48.1 | 46.6 | 8x256u8 | 2 | 0.97 |
| 1.46MB | 8, in a graph |  |  | 45.9 | 8x256u8 |  |  |
| 2.8MB | 1 | 50.4 | 50.5 | 48.8 | 8x256u4 | 2 | 0.96 |
| 2.8MB | 8 | 47.6 | 48.4 | 47.5 | 8x256u8 | 2 | 0.98 |
| 2.8MB | 64 | 46.8 | 47.9 | 48.3 | 8x256u8 | 4 | 1.01 |
| 2.8MB | 8, in a graph |  |  | 48.1 | 8x256u8 |  |  |
| 7.1MB | 1 | 46.6 | 46.6 | 47.8 | 8x256u8 | 2 | 1.03 |
| 7.1MB | 8 | 47.5 | 48.1 | 47.8 | 8x256u8 | 2 | 0.99 |
| 7.1MB | 64 | 47.5 | 47.8 | 48.0 | 8x256u8 | 4 | 1.00 |
| 7.1MB | 8, in a graph |  |  | 48.1 | 8x256u8 |  |  |

#### m3_native

| expert (gate·up + down) | MB | 1 thr | 2 thr | 4 thr | 6 thr | 7 thr | 8 per batch | GB/s best | µs/token at 2/4/8 tokens, 7 thr | bitwise mismatches |
|---|---|---|---|---|---|---|---|---|---|---|
| q4_k+q6_k (Q4_K_M, layer 0) | 3.06 | 143.0 | 72.9 | 68.3 | 68.9 | 69.0 | 61.6 | 45 | 35.0 / 20.2 / 19.3 | 0 of 45 |
| q4_k+q4_k (Q4_K_M, layer 6) | 2.65 | 128.0 | 69.5 | 59.2 | 58.7 | 59.2 | 54.1 | 45 | 29.8 / 19.4 / 19.3 | 0 of 45 |
| iq3_xxs+iq3_xxs (UD-IQ3_XXS, layer 0) | 1.81 | 361.9 | 183.1 | 94.8 | 72.9 | 59.6 | 54.3 | 30 | 52.1 / 49.6 / 49.5 | 0 of 45 |
| iq3_s+iq3_s (UD-IQ3_XXS, layer 25) | 2.03 | 417.0 | 203.1 | 104.8 | 72.2 | 67.4 | 61.8 | 30 | 59.2 / 57.7 / 62.3 | 0 of 45 |
| iq2_xxs | 1.22 | 284.5 | 143.2 | 72.2 | 50.8 | 49.0 | 42.5 | 25 | 43.5 / 41.2 / 41.0 | 0 of 45 |
| iq2_xs | 1.36 | 253.9 | 126.8 | 67.0 | 50.7 | 44.1 | 38.7 | 31 | 40.2 / 38.9 / 38.8 | 0 of 45 |
| iq2_s | 1.51 | 277.0 | 139.3 | 74.3 | 55.9 | 44.9 | 41.0 | 34 | 41.8 / 41.2 / 40.7 | 0 of 45 |
| iq2_xxs+iq2_s (Maya-S mix) | 1.31 | 270.5 | 144.3 | 73.0 | 55.1 | 44.6 | 40.0 | 30 | 40.8 / 39.9 / 42.0 | 0 of 45 |

#### m3_avx2

| expert (gate·up + down) | MB | 1 thr | 2 thr | 4 thr | 6 thr | 7 thr | 8 per batch | GB/s best | µs/token at 2/4/8 tokens, 7 thr | bitwise mismatches |
|---|---|---|---|---|---|---|---|---|---|---|
| q4_k+q6_k (Q4_K_M, layer 0) | 3.06 | 160.6 | 74.5 | 68.4 | 68.3 | 68.6 | 64.7 | 45 | 34.8 / 20.9 / 20.2 | 0 of 45 |
| q4_k+q4_k (Q4_K_M, layer 6) | 2.65 | 127.5 | 69.3 | 60.0 | 59.5 | 60.3 | 54.9 | 45 | 29.9 / 19.7 / 19.5 | 0 of 45 |
| iq3_xxs+iq3_xxs (UD-IQ3_XXS, layer 0) | 1.81 | 359.0 | 182.6 | 94.1 | 71.1 | 60.6 | 55.5 | 30 | 56.4 / 54.8 / 55.2 | 0 of 45 |
| iq3_s+iq3_s (UD-IQ3_XXS, layer 25) | 2.03 | 422.6 | 214.4 | 111.1 | 82.6 | 70.7 | 65.4 | 29 | 67.1 / 65.5 / 65.8 | 0 of 45 |
| iq2_xxs | 1.22 | 269.3 | 141.9 | 71.2 | 54.0 | 45.2 | 40.3 | 27 | 41.9 / 39.6 / 39.6 | 0 of 45 |
| iq2_xs | 1.36 | 264.7 | 136.5 | 71.2 | 54.8 | 45.5 | 39.4 | 30 | 41.2 / 39.8 / 39.8 | 0 of 45 |
| iq2_s | 1.51 | 276.9 | 136.8 | 73.3 | 55.4 | 46.3 | 41.3 | 33 | 42.0 / 40.5 / 39.8 | 0 of 45 |
| iq2_xxs+iq2_s (Maya-S mix) | 1.31 | 268.2 | 138.0 | 72.4 | 49.9 | 46.6 | 41.0 | 28 | 42.2 / 40.4 / 40.2 | 0 of 45 |

#### m3_isa_bitwise_equal

AVX-512 build vs AVX2 build, the same expert and tokens bitwise equal: q4_k+q6_k yes, q4_k+q4_k yes, iq3_xxs+iq3_xxs yes, iq3_s+iq3_s yes, iq2_xxs yes, iq2_xs yes, iq2_s yes, iq2_xxs+iq2_s yes

#### m2

expert 3.06 MB, gather 8x256u8
| CPU threads | GPU traffic | CPU µs/expert | GPU GB/s |
|---|---|---|---|
| 0 | dma | — | 46.7 |
| 0 | gather | — | 47.4 |
| 1 | none | 130.1 | — |
| 1 | dma | 317.7 | 42.0 |
| 1 | gather | 184.8 | 36.3 |
| 2 | none | 70.8 | — |
| 2 | dma | 197.7 | 37.7 |
| 2 | gather | 124.7 | 29.7 |
| 4 | none | 62.3 | — |
| 4 | dma | 174.2 | 36.5 |
| 4 | gather | 124.0 | 29.7 |
| 6 | none | 64.9 | — |
| 6 | dma | 148.6 | 34.2 |
| 6 | gather | 113.5 | 27.6 |

per-miss-count split: plan[n] = misses the CPU takes, path = max(CPU, gather) at the best k; gain vs fetch-only
| misses n | fetch-only µs | 2 threads: plan[n] / µs / gain | 4 threads: plan[n] / µs / gain | 6 threads: plan[n] / µs / gain |
|---|---|---|---|---|
| 1 | 67.4 | 0 / 67.4 / 0% | 1 / 62.1 / 8% | 1 / 61.8 / 9% |
| 2 | 131.0 | 1 / 115.1 / 14% | 1 / 117.8 / 11% | 1 / 116.4 / 12% |
| 3 | 194.2 | 1 / 178.3 / 9% | 2 / 173.8 / 12% | 2 / 172.9 / 12% |
| 4 | 259.7 | 2 / 224.8 / 16% | 3 / 228.7 / 13% | 2 / 226.8 / 14% |
| 5 | 323.6 | 2 / 286.7 / 13% | 3 / 284.0 / 13% | 3 / 280.8 / 15% |
| 6 | 384.7 | 3 / 339.3 / 13% | 3 / 338.5 / 14% | 3 / 336.2 / 15% |
| 7 | 450.8 | 4 / 395.4 / 14% | 4 / 395.7 / 14% | 4 / 394.0 / 14% |
| 8 | 513.9 | 4 / 451.7 / 14% | 4 / 451.4 / 14% | 5 / 450.8 / 14% |
| 9 | 577.5 | 5 / 513.9 / 12% | 8 / 508.1 / 14% | 5 / 506.4 / 14% |
| 10 | 640.7 | 5 / 566.0 / 13% | 6 / 561.5 / 14% | 6 / 559.0 / 14% |

#### m2_iq2_xs

expert 1.36 MB, gather 8x256u8
| CPU threads | GPU traffic | CPU µs/expert | GPU GB/s |
|---|---|---|---|
| 0 | dma | — | 44.2 |
| 0 | gather | — | 46.8 |
| 4 | none | 68.9 | — |
| 4 | dma | 112.2 | 40.1 |
| 4 | gather | 74.8 | 37.3 |
| 6 | none | 48.7 | — |
| 6 | dma | 74.5 | 37.0 |
| 6 | gather | 55.4 | 33.4 |

per-miss-count split: plan[n] = misses the CPU takes, path = max(CPU, gather) at the best k; gain vs fetch-only
| misses n | fetch-only µs | 4 threads: plan[n] / µs / gain | 6 threads: plan[n] / µs / gain |
|---|---|---|---|
| 1 | 34.6 | 0 / 34.6 / 0% | 0 / 32.9 / 0% |
| 2 | 61.0 | 0 / 61.0 / 0% | 1 / 59.3 / 3% |
| 3 | 89.5 | 1 / 80.6 / 11% | 1 / 81.2 / 9% |
| 4 | 116.5 | 1 / 105.0 / 11% | 1 / 104.4 / 12% |
| 5 | 145.5 | 1 / 132.9 / 10% | 2 / 120.3 / 21% |
| 6 | 174.1 | 2 / 151.9 / 15% | 2 / 148.0 / 17% |
| 7 | 202.4 | 2 / 177.1 / 14% | 3 / 168.1 / 20% |
| 8 | 229.5 | 2 / 205.9 / 11% | 3 / 192.4 / 20% |
| 9 | 258.3 | 3 / 223.2 / 16% | 4 / 217.0 / 20% |
| 10 | 288.4 | 3 / 251.3 / 15% | 4 / 241.2 / 19% |

#### m5

4 compute threads; stranded at stop: 0
| rows | jobs k | CPU µs/layer | overhead device-planned | overhead host-planned | exposed with GPU work, device | exposed, host | bitwise vs ggml-cpu |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 66.9 | 2.1 | 2.1 | 2.2 | 2.6 | yes |
| 1 | 2 | 126.0 | 2.1 | 2.0 | 2.6 | 3.8 | yes |
| 1 | 4 | 246.4 | 2.3 | 2.3 | 4.1 | 4.7 | yes |
| 1 | 8 | 493.5 | 2.3 | 2.2 | 5.6 | 6.2 | yes |
| 1 | 10 | 621.5 | 2.2 | 2.1 | 5.8 | 2.4 | yes |
| 2 | 1 | 72.1 | 2.1 | 2.1 | 1.5 | 2.4 | yes |
| 2 | 2 | 136.0 | 2.3 | 2.3 | 2.3 | 2.6 | yes |
| 2 | 4 | 266.1 | 2.6 | 2.6 | 3.6 | 3.2 | yes |
| 2 | 8 | 531.5 | 2.7 | 2.6 | 5.0 | 5.8 | yes |
| 2 | 10 | 681.3 | 2.5 | 2.3 | 9.5 | 6.4 | yes |
| 4 | 1 | 122.8 | 2.3 | 2.3 | 2.8 | 2.8 | yes |
| 4 | 2 | 242.8 | 2.8 | 2.8 | 1.9 | 1.7 | yes |
| 4 | 4 | 477.8 | 3.4 | 3.3 | 4.4 | 4.1 | yes |
| 4 | 8 | 986.7 | 3.5 | 3.5 | 0.8 | 2.4 | yes |
| 4 | 10 | 1188.1 | 3.6 | 3.7 | 9.0 | 10.7 | yes |
| 8 | 1 | 252.5 | 2.8 | 2.7 | 3.4 | 2.9 | yes |
| 8 | 2 | 492.7 | 3.7 | 3.7 | 3.4 | 3.9 | yes |
| 8 | 4 | 985.2 | 4.6 | 4.6 | 2.2 | 5.2 | yes |
| 8 | 8 | 1975.6 | 5.8 | 5.8 | 8.7 | 10.8 | yes |
| 8 | 10 | 2461.8 | 6.5 | 6.1 | 7.7 | 10.3 | yes |

#### m7

test file 16 GiB, written at 1.8 GB/s, removed: True
| read | QD 1 GB/s | QD 1 p50 µs | best GB/s | at | pread 1/2/4/8 workers GB/s |
|---|---|---|---|---|---|
| 1.46MB | 1.74 | 736 | 2.59 | aio w=4 qd=32 | 1.95 / 2.53 / 2.53 / 2.53 |
| 2.8MB | 2.19 | 1276 | 2.59 | aio w=4 qd=16 | 2.15 / 2.59 / 2.58 / 2.59 |
| 7.1MB | 2.39 | 2915 | 2.66 | pread w=8 qd=8 | 2.46 / 2.64 / 2.64 / 2.66 |

#### prefill

streaming time = non-resident experts' bytes / 47.5 GB/s (M1 gather)
| layer | chunk tokens | expert GEMMs ms | stream at 47% resident ms | stream at 25% resident ms |
|---|---|---|---|---|
| q4_k gate/up, q6_k down | 64 | 0.779 | 4.37 | 6.18 |
| q4_k gate/up, q6_k down | 256 | 1.274 | 4.37 | 6.18 |
| q4_k gate/up, q6_k down | 1024 | 2.720 | 4.37 | 6.18 |
| q4_k gate/up, q6_k down | 4096 | 10.970 | 4.37 | 6.18 |
| q4_k gate/up, q4_k down | 64 | 0.708 | 3.79 | 5.36 |
| q4_k gate/up, q4_k down | 256 | 1.180 | 3.79 | 5.36 |
| q4_k gate/up, q4_k down | 1024 | 2.532 | 3.79 | 5.36 |
| q4_k gate/up, q4_k down | 4096 | 10.179 | 3.79 | 5.36 |

### rtx-5090 (NVIDIA GeForce RTX 5090, CPU x86-avx512)


#### m1

pinned cudaMemcpyAsync, GB/s: 1 MiB 52.0, 2 MiB 53.1, 4 MiB 53.6, 8 MiB 46.6, 16 MiB 46.6, 32 MiB 47.0, 64 MiB 47.5, 128 MiB 47.5, 256 MiB 47.4, 512 MiB 47.3
| blob | N | memcpy ×N | BatchAsync | gather best | config (CTAs×threads, unroll) | CTAs ≥ 95% | gather / best DMA |
|---|---|---|---|---|---|---|---|
| 1.46MB | 1 | 52.3 | 52.5 | 49.6 | 8x256u4 | 2 | 0.94 |
| 1.46MB | 8 | 44.7 | 47.4 | 47.3 | 8x256u8 | 2 | 1.00 |
| 1.46MB | 64 | 44.4 | 47.3 | 47.0 | 8x256u8 | 2 | 0.99 |
| 1.46MB | 8, in a graph |  |  | 46.9 | 8x256u8 |  |  |
| 2.8MB | 1 | 51.6 | 51.7 | 50.9 | 8x256u4 | 4 | 0.98 |
| 2.8MB | 8 | 46.0 | 47.0 | 48.1 | 32x256u4 | 4 | 1.02 |
| 2.8MB | 64 | 46.2 | 47.6 | 48.2 | 8x256u8 | 4 | 1.01 |
| 2.8MB | 8, in a graph |  |  | 48.1 | 32x256u4 |  |  |
| 7.1MB | 1 | 46.1 | 46.1 | 48.7 | 64x256u2 | 8 | 1.06 |
| 7.1MB | 8 | 46.8 | 47.6 | 48.0 | 24x256u4 | 4 | 1.01 |
| 7.1MB | 64 | 46.6 | 47.4 | 48.0 | 24x256u4 | 4 | 1.01 |
| 7.1MB | 8, in a graph |  |  | 48.2 | 24x256u4 |  |  |

#### m3_native

| expert (gate·up + down) | MB | 1 thr | 2 thr | 4 thr | 6 thr | 7 thr | 8 per batch | GB/s best | µs/token at 2/4/8 tokens, 7 thr | bitwise mismatches |
|---|---|---|---|---|---|---|---|---|---|---|
| q4_k+q6_k (Q4_K_M, layer 0) | 3.06 | 147.6 | 75.0 | 69.1 | 68.7 | 69.0 | 63.3 | 45 | 34.5 / 20.4 / 18.9 | 0 of 45 |
| q4_k+q4_k (Q4_K_M, layer 6) | 2.65 | 129.8 | 69.5 | 60.0 | 59.6 | 60.2 | 54.2 | 45 | 29.9 / 18.9 / 18.6 | 0 of 45 |
| iq3_xxs+iq3_xxs (UD-IQ3_XXS, layer 0) | 1.81 | 358.1 | 231.2 | 93.3 | 66.2 | 61.3 | 56.2 | 29 | 57.7 / 56.4 / 56.0 | 0 of 45 |
| iq3_s+iq3_s (UD-IQ3_XXS, layer 25) | 2.03 | 405.2 | 256.3 | 121.0 | 79.8 | 70.3 | 64.1 | 29 | 65.6 / 63.9 / 64.1 | 0 of 45 |
| iq2_xxs | 1.22 | 448.7 | 172.8 | 80.3 | 53.7 | 47.5 | 42.3 | 26 | 44.3 / 42.6 / 42.3 | 0 of 45 |
| iq2_xs | 1.36 | 381.4 | 161.1 | 75.6 | 51.4 | 45.3 | 39.7 | 30 | 41.3 / 39.8 / 39.7 | 0 of 45 |
| iq2_s | 1.51 | 282.8 | 180.7 | 73.9 | 55.1 | 47.4 | 42.8 | 32 | 45.0 / 42.7 / 42.1 | 0 of 45 |
| iq2_xxs+iq2_s (Maya-S mix) | 1.31 | 276.1 | 139.9 | 71.5 | 50.5 | 44.7 | 40.2 | 29 | 42.1 / 40.0 / 39.3 | 0 of 45 |

#### m3_avx2

| expert (gate·up + down) | MB | 1 thr | 2 thr | 4 thr | 6 thr | 7 thr | 8 per batch | GB/s best | µs/token at 2/4/8 tokens, 7 thr | bitwise mismatches |
|---|---|---|---|---|---|---|---|---|---|---|
| q4_k+q6_k (Q4_K_M, layer 0) | 3.06 | 155.3 | 76.1 | 69.3 | 68.9 | 68.7 | 65.8 | 45 | 34.8 / 20.4 / 19.7 | 0 of 45 |
| q4_k+q4_k (Q4_K_M, layer 6) | 2.65 | 131.9 | 71.2 | 60.3 | 59.7 | 59.4 | 55.0 | 45 | 29.8 / 19.0 / 19.1 | 0 of 45 |
| iq3_xxs+iq3_xxs (UD-IQ3_XXS, layer 0) | 1.81 | 358.2 | 182.3 | 95.4 | 66.9 | 58.8 | 53.6 | 31 | 54.0 / 52.0 / 52.4 | 0 of 45 |
| iq3_s+iq3_s (UD-IQ3_XXS, layer 25) | 2.03 | 426.9 | 215.1 | 112.5 | 78.2 | 68.8 | 63.4 | 29 | 64.5 / 63.3 / 63.4 | 0 of 45 |
| iq2_xxs | 1.22 | 271.0 | 137.5 | 70.7 | 49.4 | 44.0 | 38.9 | 28 | 39.7 / 38.3 / 38.4 | 0 of 45 |
| iq2_xs | 1.36 | 270.1 | 137.3 | 72.8 | 50.8 | 45.5 | 41.3 | 30 | 44.4 / 43.2 / 42.9 | 0 of 45 |
| iq2_s | 1.51 | 281.7 | 135.3 | 72.7 | 50.4 | 44.4 | 40.4 | 34 | 40.4 / 38.5 / 38.2 | 0 of 45 |
| iq2_xxs+iq2_s (Maya-S mix) | 1.31 | 269.5 | 141.2 | 71.0 | 50.0 | 44.6 | 39.2 | 29 | 40.5 / 38.7 / 38.4 | 0 of 45 |

#### m3_isa_bitwise_equal

AVX-512 build vs AVX2 build, the same expert and tokens bitwise equal: q4_k+q6_k yes, q4_k+q4_k yes, iq3_xxs+iq3_xxs yes, iq3_s+iq3_s yes, iq2_xxs yes, iq2_xs yes, iq2_s yes, iq2_xxs+iq2_s yes

#### m2

expert 3.06 MB, gather 32x256u4
| CPU threads | GPU traffic | CPU µs/expert | GPU GB/s |
|---|---|---|---|
| 0 | dma | — | 46.5 |
| 0 | gather | — | 47.7 |
| 1 | none | 159.4 | — |
| 1 | dma | 366.7 | 42.2 |
| 1 | gather | 189.4 | 36.1 |
| 2 | none | 73.0 | — |
| 2 | dma | 253.1 | 40.1 |
| 2 | gather | 129.9 | 30.4 |
| 4 | none | 63.1 | — |
| 4 | dma | 196.2 | 38.1 |
| 4 | gather | 123.7 | 29.6 |
| 6 | none | 63.0 | — |
| 6 | dma | 148.3 | 34.1 |
| 6 | gather | 112.2 | 27.3 |

per-miss-count split: plan[n] = misses the CPU takes, path = max(CPU, gather) at the best k; gain vs fetch-only
| misses n | fetch-only µs | 2 threads: plan[n] / µs / gain | 4 threads: plan[n] / µs / gain | 6 threads: plan[n] / µs / gain |
|---|---|---|---|---|
| 1 | 67.4 | 0 / 67.4 / 0% | 1 / 63.6 / 6% | 1 / 62.5 / 8% |
| 2 | 130.5 | 1 / 118.8 / 10% | 1 / 118.0 / 11% | 1 / 114.8 / 14% |
| 3 | 193.9 | 1 / 176.2 / 10% | 2 / 172.6 / 12% | 2 / 171.4 / 13% |
| 4 | 257.6 | 2 / 234.9 / 10% | 2 / 230.3 / 12% | 2 / 223.5 / 15% |
| 5 | 321.0 | 2 / 294.3 / 9% | 3 / 284.5 / 12% | 3 / 282.0 / 14% |
| 6 | 384.7 | 3 / 347.0 / 11% | 3 / 341.1 / 13% | 3 / 336.1 / 14% |
| 7 | 449.9 | 3 / 403.2 / 12% | 4 / 396.4 / 13% | 4 / 394.1 / 14% |
| 8 | 513.3 | 4 / 453.5 / 13% | 4 / 452.5 / 13% | 4 / 450.0 / 14% |
| 9 | 577.1 | 4 / 516.5 / 12% | 5 / 510.1 / 13% | 5 / 510.1 / 13% |
| 10 | 640.3 | 5 / 569.5 / 12% | 5 / 567.6 / 13% | 6 / 564.0 / 14% |

#### m2_iq2_xs

expert 1.36 MB, gather 32x256u4
| CPU threads | GPU traffic | CPU µs/expert | GPU GB/s |
|---|---|---|---|
| 0 | dma | — | 45.1 |
| 0 | gather | — | 47.7 |
| 4 | none | 77.7 | — |
| 4 | dma | 127.7 | 41.2 |
| 4 | gather | 79.5 | 38.2 |
| 6 | none | 51.0 | — |
| 6 | dma | 82.2 | 38.3 |
| 6 | gather | 54.7 | 33.3 |

per-miss-count split: plan[n] = misses the CPU takes, path = max(CPU, gather) at the best k; gain vs fetch-only
| misses n | fetch-only µs | 4 threads: plan[n] / µs / gain | 6 threads: plan[n] / µs / gain |
|---|---|---|---|
| 1 | 32.5 | 0 / 32.5 / 0% | 0 / 32.6 / 0% |
| 2 | 59.6 | 0 / 59.6 / 0% | 1 / 56.0 / 6% |
| 3 | 87.9 | 1 / 81.7 / 8% | 1 / 81.5 / 8% |
| 4 | 116.2 | 1 / 102.2 / 14% | 2 / 101.2 / 15% |
| 5 | 144.7 | 1 / 130.8 / 11% | 2 / 118.6 / 22% |
| 6 | 172.1 | 2 / 156.4 / 10% | 2 / 147.6 / 18% |
| 7 | 200.6 | 2 / 175.7 / 14% | 3 / 165.7 / 21% |
| 8 | 229.0 | 2 / 206.4 / 11% | 3 / 194.0 / 18% |
| 9 | 258.1 | 3 / 231.5 / 12% | 4 / 211.0 / 23% |
| 10 | 286.4 | 3 / 249.2 / 15% | 4 / 241.0 / 19% |

#### m5

4 compute threads; stranded at stop: 0
| rows | jobs k | CPU µs/layer | overhead device-planned | overhead host-planned | exposed with GPU work, device | exposed, host | bitwise vs ggml-cpu |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 67.4 | 2.1 | 2.0 | 1.0 | 1.5 | yes |
| 1 | 2 | 127.5 | 2.1 | 2.0 | 0.4 | 1.4 | yes |
| 1 | 4 | 246.6 | 2.2 | 2.3 | 1.9 | 2.6 | yes |
| 1 | 8 | 498.9 | 2.3 | 2.2 | 4.3 | 5.0 | yes |
| 1 | 10 | 627.6 | 2.2 | 2.2 | 4.5 | 2.5 | yes |
| 2 | 1 | 72.4 | 2.1 | 2.0 | -0.6 | 0.6 | yes |
| 2 | 2 | 136.8 | 2.3 | 2.3 | 0.6 | 1.9 | yes |
| 2 | 4 | 273.5 | 2.6 | 2.6 | 1.6 | 3.1 | yes |
| 2 | 8 | 536.5 | 2.9 | 2.7 | 3.0 | 4.5 | yes |
| 2 | 10 | 674.2 | 2.6 | 2.5 | 2.1 | 3.4 | yes |
| 4 | 1 | 123.3 | 2.1 | 2.1 | 1.7 | 2.3 | yes |
| 4 | 2 | 243.4 | 2.8 | 2.7 | 5.4 | 5.1 | yes |
| 4 | 4 | 482.1 | 3.3 | 3.3 | 2.6 | 3.4 | yes |
| 4 | 8 | 966.3 | 3.5 | 3.4 | 3.9 | 5.5 | yes |
| 4 | 10 | 1202.8 | 3.8 | 3.7 | 3.4 | 4.5 | yes |
| 8 | 1 | 250.8 | 2.6 | 2.6 | 1.0 | 1.4 | yes |
| 8 | 2 | 498.7 | 3.7 | 3.7 | 1.7 | 2.9 | yes |
| 8 | 4 | 1001.9 | 4.7 | 4.7 | 2.7 | 3.8 | yes |
| 8 | 8 | 1991.8 | 5.4 | 5.4 | 7.4 | 10.1 | yes |
| 8 | 10 | 2472.1 | 6.5 | 6.2 | 8.0 | 9.7 | yes |

#### m7

test file 16 GiB, written at 1.7 GB/s, removed: True
| read | QD 1 GB/s | QD 1 p50 µs | best GB/s | at | pread 1/2/4/8 workers GB/s |
|---|---|---|---|---|---|
| 1.46MB | 1.77 | 802 | 2.57 | aio w=4 qd=32 | 1.30 / 2.45 / 2.54 / 2.52 |
| 2.8MB | 2.10 | 1331 | 2.60 | pread w=4 qd=4 | 2.12 / 2.59 / 2.60 / 2.59 |
| 7.1MB | 2.35 | 2974 | 2.65 | pread w=8 qd=8 | 2.45 / 2.62 / 2.63 / 2.65 |

#### prefill

streaming time = non-resident experts' bytes / 48.1 GB/s (M1 gather)
| layer | chunk tokens | expert GEMMs ms | stream at 47% resident ms | stream at 25% resident ms |
|---|---|---|---|---|
| q4_k gate/up, q6_k down | 64 | 0.415 | 4.32 | 6.11 |
| q4_k gate/up, q6_k down | 256 | 0.709 | 4.32 | 6.11 |
| q4_k gate/up, q6_k down | 1024 | 1.502 | 4.32 | 6.11 |
| q4_k gate/up, q6_k down | 4096 | 5.997 | 4.32 | 6.11 |
| q4_k gate/up, q4_k down | 64 | 0.374 | 3.74 | 5.30 |
| q4_k gate/up, q4_k down | 256 | 0.656 | 3.74 | 5.30 |
| q4_k gate/up, q4_k down | 1024 | 1.393 | 3.74 | 5.30 |
| q4_k gate/up, q4_k down | 4096 | 5.548 | 3.74 | 5.30 |
