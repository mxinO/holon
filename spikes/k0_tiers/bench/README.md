# Milestone K, step 0: copy paths (K-M2, K-M3 folded in) and NVMe (K-M4)

## Question

`docs/v1-kv-tiers.md` §8 fixes what step 0 measures before the tier design is frozen. This directory answers, on the
two bar cards (RTX 5080 and RTX 5090 desktops), with the RTX A5000 development host as a reference:

| # | Measure | Decides (§8) |
|---|---|---|
| K-M2 | H2D and D2H between a layer-major device pool and an S2 host pool (layer-major or unit-major), at the 27B's q8_0 KV chunk (34,816 B), Qwen3-8B's bf16 KV chunk (65,536 B) and the 27B's GDN checkpoint segments (61,440 B conv, 3 MiB SSM); fragmented units and runs of 4 / 16; loads of 1K / 16K / 32K tokens of the 27B; three paths: today's per-run loop, one `cudaMemcpyBatchAsync`, the SM copy kernel | `state_copy@1`'s path; `h2d_rate`; `stall_bytes` |
| K-M3 | the 27B decoding 4 sequences at its lock's settings while 1.29 GB H2D loads or 154 MB D2H checkpoint stores run on another stream from / into S2 host memory | 3b's prediction; whether stores need a rate cap |
| K-M4 | `os.preadv` / `os.pwritev` from Python threads of 557,056 B units and of 27B checkpoints as per-segment iovecs; O_DIRECT and buffered + `DONTNEED`; 1–4 threads; burst and sustained writes; mixed reads and writes on one drive; the engine loop's host time beside the I/O threads (GIL) | `io_threads`; the lane's read and write rates and the mixed-rate check; the close-flush estimate; D9's SSD rate |

## Setup

| | |
|---|---|
| GPUs | RTX 5080 and RTX 5090 (sm_120), PCIe 5.0 x16, one consumer desktop per class; RTX A5000 (sm_86, PCIe 4.0) as the development reference |
| software | CUDA 13.0 (container on the desktops), torch 2.11.0+cu130, Python 3.12; this directory's extension built per GPU class under `$TORCH_EXTENSIONS_DIR`. On the desktops, a uv-synced environment of the K worktree (the cluster venv was stale). The RTX 5080 runs imported a `core/` with step 1's then-uncommitted edits, tiers off (`platform.host_memory` and `memory.copy_units` unchanged from b5b1e7f); the RTX 5090 runs, the K worktree at 0accff8 (step 1 committed). The A5000 runs imported the main checkout's core at b5b1e7f |
| host pools | the core's own `holon.core.memory.Pool` (so `platform.host_memory`: an anonymous mmap + `cudaHostRegister`, portable, mapped so the SM path can address it), exactly as S2 allocates them |
| device pools | `Pool` too: layer-major `[segments, capacity, chunk]` |
| K-M3 models | RTX 5080: Qwen3.8-27B UD-IQ3_S with its rtx-5080 lock (q8_0 KV, `max_seqs` 4); RTX 5090: UD-Q4_K_M with its rtx-5090 lock; RTX A5000: UD-Q4_K_M with the rtx-3090 lock's settings (the 24 GB class; bf16 KV) |
| storage | the desktops' node-local NVMe; on the development host a network filesystem (a reference only) |

```bash
cd spikes/k0_tiers/bench
CUDA_VISIBLE_DEVICES=0 python bench.py --tag rtx-5080 --model <Qwen3.8-27B-UD-IQ3_S.gguf> \
    --lock <repo>/plans/unsloth/Qwen3.8-27B-GGUF/<file>/rtx-5080/cu130-torch2.11/10.lock \
    --nvme-dir <node-local dir> --storage "local NVMe" --clock-screen <clock_check.py json>
#   --only m2,m3,m4   --merge (update the records)   --quick
python report.py ../../../evidence/k0_m2_copy_<tag>_<date>.json ../../../evidence/k0_m4_nvme_<tag>_<date>.json
```

Files: `bench.py` (driver), `csrc/k0.cpp` (one `cudaMemcpyBatchAsync` over address lists, the K-M3 copy feeder, the
`preadv` / `pwritev` C reference; pybind), `csrc/kernels.cu` (the SM chunk-list copy, a busy kernel), `report.py`.

## Method

- **Chunk lists.** A copy of n units moves `segments` chunks per unit. The device pool is layer-major; the host pool
  is layer-major (the same shape: a run of units consecutive in both pools is one chunk per segment) or unit-major
  (one segment per unit, the SSD level's layout: no runs on the device side). The lists are built vectorised in
  numpy from the unit ids; only addresses differ between the layouts. Units are picked at random distinct positions
  in both pools (every unit its own run) or in runs of 4 / 16 units consecutive in both pools.
- **Loads.** 1K tokens of the 27B (64 KV pages, 35.7 MB), 16K (1,024 pages, 0.57 GB) and a 32K resume (2,048 pages
  plus one checkpoint, 1.29 GB), expressed in units of each chunk shape (one checkpoint SSM unit is 151 MB, so its
  "1K" cell is one unit).
- **Paths.** (1) `memory.copy_units` itself (torch `copy_` per segment per run); (2) one `cudaMemcpyBatchAsync`
  per load (source in stream order, `PreferOverlapWithCompute`, as `kernels.expert_io`'s `copy_batch`); (3) the SM
  copy kernel over (src, dst, 16-byte words) work items, chunks above 64 KiB split into 64 KiB tiles, CTA b taking
  items b, b + grid, …, 16-byte loads U deep before the stores (CTAs 2–128 × unroll 2 / 4 / 8 on the fragmented
  cells; the runs cells reuse the fragmented cell's best). H2D reads mapped host memory; D2H writes it.
- **Timing.** GPU time between events around back-to-back repeats behind a busy kernel long enough that the host runs
  ahead (median of 3); the host time of one call (submission; for the batch path also the numpy list build; for the
  SM path the build and the work-item upload) and the end-to-end rate from an idle GPU are kept beside it. Every
  cell's destination is zeroed, copied once per path and compared with its source on the device.
- **Blob rate.** One 256 MiB / 1 GiB `cudaMemcpyAsync` from S2 host memory (and torch's pinned allocator beside it),
  each direction. The page-locking time of S2 host memory per GiB is recorded (§7).
- **K-M3.** The engine is built from the lock's `[engine]` table (`bench/run.py`'s `lock_flags`), 4 random prompts
  of 4,096 tokens decode (`max_tokens` large, `ignore_eos`). Per-step wall time between `engine.step()` returns. The
  H2D load is a fragmented 32K resume: 2,048 unit-major KV pages (32,768 chunks of 34,816 B) plus one checkpoint's 96
  segments, 1.29 GB, into a 64 MiB device ring (the bytes over PCIe are what matter); copies run back to back from a
  C++ thread on its own stream (no GIL; its waits sleep). Arms: `control`; `h2d_dma` (one `cudaMemcpyBatchAsync`
  per load); `h2d_paced512x1` / `h2d_paced2048x1` (the same list in batches of 512 / 2,048 chunks, 18 / 71 MB, the
  next issued once the previous has landed); `h2d_each` (one `cudaMemcpyAsync` per chunk: the per-run loop's device
  behaviour); `h2d_sm` (the SM copy, 8 CTAs × 256, unroll 4); `d2h_answer` (one 154 MB checkpoint store of 96
  segments every 16 steps from the engine's loop: 4 rows each crossing a 64-token boundary every 64 steps, the
  `answer` rate); `d2h_answer_paced` (the same, 8 segments at a time from a thread); `d2h_stress` /
  `d2h_stress_paced` (stores back to back). 64 measured steps per segment after 8 warm-up steps; per round the arms
  run forward then backward (ABBA), 4 rounds. Clocks sampled with `bench/clock_check.py`'s sampler; the desktop was
  clock-screened with `bench/clock_check.py` first. Before the engine starts, a **small-copy probe** times a 4 KiB /
  64 KiB / 1 MiB copy on another stream while each arm's copies run (what an engine step's own inputs and
  read-backs wait).
- **K-M4.** One test directory on the drive, removed at the end (`test_file_removed`). A preallocated read file
  (16 GiB, filled with O_DIRECT writes of non-repeating data) and a write file (8 GiB); random slots. Shapes: the
  557,056 B unit-major KV unit (one iovec), a checkpoint's SSM unit as its 48 per-segment 3 MiB iovecs (laid out as an
  S2 layer-major pool's segments, apart) and as one iovec, the conv unit as 48 × 61,440 B iovecs. Python threads
  (`os.preadv` / `os.pwritev`, one fd each), 2 s windows; a C++ reference at 1 / 2 threads. Mixed: 2 reader + 2
  writer threads, each kind held at its byte share (50/50, 75/25; the rule's cells), plus free-running 2 + 2 and
  held 1 + 1, each against 1 / (f / read + (1 − f) / write) from pure rates at the same thread counts taken just
  before. GIL: a fixed CPU-bound Python body of ≈ 3 ms, back to back and as a 15 ms step (3 ms of work, then a wait
  that releases the GIL with the core busy, as a CUDA sync does), each row against the loop alone just before it,
  beside 1–4 Python I/O threads (and 2 C threads). Sustained writes last:
  one pass over a fresh slot file sized to leave ≥ 20% of the filesystem free (≤ 400 GiB, ≤ 300 s), 2 s windows.

## Predictions (fixed before measuring; also in every evidence record)

| | Prediction |
|---|---|
| blob rate | H2D ≈ 47.5 GB/s on the RTX 5080 / 5090 (X's M1: 46.6–48.4); D2H ≈ 45–53 GB/s; A5000 (PCIe 4.0) ≈ 22–26 GB/s |
| per-run loop, fragmented 34,816 B | ≤ 0.5× the blob rate and host-bound; ≈ 5–10 µs of host time per chunk (≈ 0.15–0.3 s for a fragmented 32K load of 32,768 chunks; the design says ≈ 0.3 s) |
| `cudaMemcpyBatchAsync`, fragmented 34,816 B | 0.7–0.95× blob (uncertain); at 65,536 B and ≥ 3 MiB ≥ 0.9×; host ≈ 0.1–0.3 µs per chunk |
| SM copy, fragmented 34,816 B | gather H2D ≥ 0.9× blob with 8–32 CTAs; scatter D2H into mapped host ≥ 0.8× |
| K-M3 | decode step median slowdown ≤ 5% with H2D loads in flight, p90 ≤ 15% (3b's bar 1.1× / 1.2×); 154 MB stores at the `answer` rate cost < 0.5% of decode |
| K-M4 (bar desktops) | O_DIRECT reads of 557,056 B units 1.6–2.2 GB/s at 1 thread, 2.4–2.8 at 2, no gain past 2–3; burst writes 1.5–3 GB/s; sustained past SLC 0.5–1.5 GB/s; buffered reads ≤ O_DIRECT with `DONTNEED`; mixed within 15% of the modelled shared lane; per-segment checkpoint iovecs within 10% of the unit rate; engine-loop slowdown beside 2 I/O threads < 3% |

**Decision rules** (STOP and report, on a bar desktop): the 2-thread O_DIRECT read rate of 557,056 B units
< 1.75 GB/s; the measured mixed rate off the modelled shared lane by > 15%; neither `cudaMemcpyBatchAsync` nor the SM
gather at ≥ 0.8× the blob rate on fragmented 34,816 B chunks; 154 MB stores costing > 2% of decode at the `answer`
rate.

## Results

**One decision rule fired on both bar cards: mixed reads and writes on one drive run at 0.79–0.84× (50/50) and 0.76×
(75/25) of the modelled shared lane (bar: within 15%).** The RTX 5080 run stopped there and reported. The lead ruled
it a tier-side performance issue that does not block, and the RTX 5090 was then measured with the same protocol. The
other three rules pass on both cards. Every copy in every cell was verified byte-equal.

Evidence:
- `evidence/k0_m2_copy_rtx-5080_2026-10-18.json` and `evidence/k0_m2_copy_rtx-5090_2026-10-18.json` (K-M2, K-M3);
- `evidence/k0_m4_nvme_rtx-5080_2026-10-18.json` and `evidence/k0_m4_nvme_rtx-5090_2026-10-18.json`;
- the RTX A5000 references `evidence/k0_m2_copy_rtx-a5000_2026-10-18.json` and
  `evidence/k0_m4_nvme_rtx-a5000_2026-10-18.json`. Its link gives 13.2 GB/s each way and its storage is a network
  filesystem, so its absolute rates are a dev reference only.

| Decision rule | RTX 5080 | RTX 5090 | Verdict |
|---|---|---|---|
| 2-thread O_DIRECT reads of 557,056 B units ≥ 1.75 GB/s | 2.17 GB/s | 2.17 GB/s | pass |
| mixed within 15% of the modelled shared lane | 0.843× / 0.760× | 0.786× / 0.764× | **fires** (ruled non-blocking) |
| BatchAsync or SM gather ≥ 0.8× blob on fragmented 34,816 B | 0.99× (0.987× at worst) | 0.99× (0.987× at worst) | pass |
| 154 MB stores ≤ 2% of decode at the `answer` rate | −0.7% | −0.1% | pass (≈ 0) |

### K-M2 — copy paths (RTX 5080 / RTX 5090; A5000 in brackets)

| | Prediction | Measured |
|---|---|---|
| blob H2D / D2H | ≈ 47.5 / 45–53 GB/s [22–26] | **47.6 / 49.1 and 47.4 / 47.9 GB/s** [13.2 / 13.2: the dev host's link] |
| per-run loop, fragmented 34,816 B | ≤ 0.5× blob, host-bound, 5–10 µs per chunk | **0.13× (6.0–6.6 GB/s); 5.6–5.9 µs per chunk, 209–218 ms for a 32K load (37,184 chunks)** [0.26–0.30×, 9.5–12 µs]. The GPU itself is slow on separate 34 KB copies (≈ 5.8 µs each even with the host ahead), so the loop is device- as well as host-bound |
| BatchAsync, fragmented 34,816 B | 0.7–0.95× blob; host 0.1–0.3 µs per chunk | **0.99–1.00× H2D at 16K / 32K (0.93× at 1K); D2H 0.82–0.94×.** Host per chunk: **0.31–0.35 µs at 16K** (5.0–5.4 ms submit + 0.1–0.2 ms numpy build); **0.56–0.67 µs at 32K** (21–25 ms submit + 0.1–10 ms build). The submit grows faster than the list |
| BatchAsync, 65,536 B / 61,440 B / 3 MiB | ≥ 0.9× | 0.96–0.98× / 1.01–1.02× / 0.96–1.00× H2D; 0.92× / 0.98–1.03× / 0.93–1.03× D2H (0.84–0.87× at the 1K loads) |
| SM gather H2D, fragmented 34,816 B | ≥ 0.9× with 8–32 CTAs | **0.99× (46.8–47.3 GB/s); within 95% of the best from 8 CTAs; best 16–32 × 256, unroll 2–4** |
| SM scatter D2H into mapped host | ≥ 0.8× | **0.84–0.88×** for KV chunks, 0.89–0.95× for the other shapes [A5000: 0.45–0.5×] |
| layouts and runs | only addresses differ | unit-major and layer-major host pools move alike (± 4%); runs of 4 / 16 change nothing for BatchAsync and the SM copy, and lift the per-run loop to 24 / 41 GB/s |

S2 host memory page-locks at 0.25 s per GiB on both desktops (0.34 on the A5000).

### K-M3 — decode beside copies

- RTX 5080: 27B UD-IQ3_S with the rtx-5080 lock.
- RTX 5090: UD-Q4_K_M with the rtx-5090 lock.
- Both: 4 rows growing from 4,096 to 9,912 tokens.

Both clock screens were steady (SM p10 / p50 2,677 / 2,685 and 2,595 / 2,640 MHz, no slowdown samples; the 5090 at
its power cap). Copies run back to back through whole segments, so every measured step overlaps them. That is the
worst case for 3b, whose 32K load spans ≈ 1.5–2 steps.

| arm | 5080 step p50 ms | median × | p90 × | 5090 step p50 ms | median × | p90 × | copies GB/s (5080 / 5090) |
|---|---|---|---|---|---|---|---|
| control | 22.30 | 1 | 1 | 14.72 | 1 | 1 | – |
| `h2d_dma` (one batch per 1.29 GB load) | 31.69 | **1.42** | **1.55** | 30.53 | **2.07** | **2.85** | 40.3 / 39.9 |
| `h2d_paced512x1` (18 MB pieces, one in flight) | 22.81 | 1.023 | 1.037 | 14.95 | 1.015 | 1.031 | 32.7 / 32.2 |
| `h2d_paced2048x1` (71 MB pieces) | 23.06 | 1.034 | 1.076 | 15.40 | 1.046 | 1.087 | 33.7 / 33.4 |
| `h2d_each` (one copy per chunk) | 61.29 | 2.75 | 2.68 | 64.46 | 4.38 | 4.25 | 21.1 / 20.1 |
| `h2d_sm` (8 CTAs) | 23.86 | 1.070 | 1.077 | 15.33 | 1.042 | 1.047 | 43.3 / 41.5 |
| `d2h_answer` (one 154 MB store per 16 steps) | 22.12 | 0.992 | 0.993 | 14.67 | 0.997 | 0.997 | – |
| `d2h_answer_paced` | 22.16 | 0.994 | 0.999 | 14.73 | 1.000 | 0.994 | – |
| `d2h_stress` (stores back to back) | 25.13 | 1.127 | 1.103 | 18.11 | 1.231 | 1.197 | 42.9 / 42.3 |
| `d2h_stress_paced` (8 segments at a time) | 22.86 | 1.025 | 1.021 | 15.13 | 1.028 | 1.030 | 39.3 / 38.8 |

**Why unpaced DMA hurts: head-of-line blocking on the copy engine.**
- A 64 KiB or 1 MiB copy issued on another stream while a batched load is in flight waits for the load to finish:
  p90 25.6 / 27.1 ms on the 5080 / 5090, 99 ms on the A5000, against 0.03–0.08 ms idle.
- A 4 KiB copy does not wait (it goes inline in the command stream).
- The engine's per-step H2D copies are evidently above that size, so a decode step waits for the whole load. The
  shorter the step, the larger the ratio: the 5090's 14.7 ms step against the 5080's 22.3.
- Pacing helps only with **one** piece in flight; two pieces in flight block like the whole load (25 ms p90). It
  costs rate: 32.2–33.9 GB/s alone, 0.68–0.71× blob, from the host round trip between pieces.
- The `PreferOverlapWithCompute` flag changes nothing.
- The SM copy uses no copy engine (probes 0.05–0.35 ms), but takes SMs and DRAM bandwidth: +4–7% per step while it
  runs.
- D2H stores block the engine's D2H read-backs the same way: 0.39 / 0.53 ms p50 under stress, 8.8 ms on the A5000.
  Pacing halves that (0.21 / 0.33 ms).

### K-M4 — NVMe (node-local NVMe of each desktop; the two drives behave alike)

| | Prediction | RTX 5080 desktop | RTX 5090 desktop |
|---|---|---|---|
| O_DIRECT reads of 557,056 B units, 1 / 2 / 3 / 4 threads | 1.6–2.2 / 2.4–2.8 / no gain past 2–3 | **1.38 / 2.17 / 2.55 / 2.58 GB/s** | **1.37 / 2.17 / 2.54 / 2.58 GB/s** |
| the same from C threads, 1 / 2 | – | 1.39 / 2.17 | 1.38 / 2.17 |
| checkpoint SSM unit as 48 × 3 MiB iovecs, 1–2 threads | within 10% of the unit rate | 2.68–2.69 GB/s | 2.64–2.66 GB/s |
| ratios for that row | – | 1.00× one 151 MB iovec, 1.23× the unit's 2-thread rate | 1.00× / 1.21× |
| conv unit as 48 × 61,440 B iovecs, 1 / 2 threads | – | 2.20 / 2.58 | 2.22 / 2.59 |
| buffered + `DONTNEED` reads, 1–4 threads | ≤ O_DIRECT | 0.86 / 1.59 / 2.11 / 2.47 | 0.86 / 1.58 / 2.10 / 2.47 |
| burst writes, 1–4 threads | 1.5–3 GB/s | **2.19–2.21** (flat) | **2.09–2.13** (flat) |
| burst writes, other cells | – | 48-iovec checkpoints 2.22; buffered 2.20 | 2.17; 2.16 |
| sustained writes past SLC | 0.5–1.5 GB/s | **no cliff:** 1,063 GB in 600 s, 1.77 GB/s mean | **no cliff:** 1,078 GB in 600 s, 1.80 GB/s mean |
| mixed vs model, 2 + 2 threads at 50/50 · 75/25 | within 15% | **0.843× · 0.760×** | **0.786× · 0.764×** |
| mixed vs model, 1 + 1 threads at 50/50 | – | 0.840× | 0.786× |
| mixed, free-running 2 + 2 | – | writes take 78% of the bytes; reads 0.43 GB/s; 0.98× the model at that share | writes 79%; reads 0.42 GB/s; 0.91× |
| engine loop beside 2 I/O threads | < 3% slower | no measurable slowdown (all rows −7% … +2%, the noise of paired baselines) | same |

Both drives had ≥ 2.5 TB free, so their caches, if any, reach beyond 1 TB.

**The GIL starves the I/O threads instead.**
- Beside a loop that holds the GIL (3 ms Python bodies back to back), 2 Python `preadv` threads read 0.76–0.80
  GB/s. C threads beside the same loop read 2.17.
- With 3 ms of Python per 15 ms step: 1.81–1.83 GB/s at 2 threads, 2.13–2.16 at 3.
- Each `preadv` releases the GIL. But the thread must take it back between calls, and a busy loop hands it over only
  at the 5 ms switch interval.

**O_DIRECT alignment.**
- Every GDN segment (61,440 B, 3 MiB) and the unit-major KV unit (557,056 B) is a multiple of 4,096, so none of them
  needs a bounce buffer.
- Only the 34,816 B layer-major KV chunk is not. It is a multiple of 512, and both filesystems accepted 512-aligned
  lengths and offsets with O_DIRECT.

### What the numbers suggest for K (the lead's decisions are in `docs/v1-kv-tiers.md` §13, rev 8.1)

- **`state_copy@1`.**
  - On raw rate, the two paths tie on fragmented 34,816 B H2D (0.99×). The SM gather is slightly ahead on small
    loads (0.99× against 0.93× at 1K).
  - Beside decode, an unpaced batch fails 3b's bar by itself: 1.42× / 2.07× median, 1.55× / 2.85× p90 while in
    flight on the 5080 / 5090.
  - The SM gather costs +4–7% per step at 41–43 GB/s; paced DMA costs +1.5–2.3% at 32–33 GB/s.
  - For loads, the SM gather (8–32 CTAs) is the faster path and needs no pacing thread.
  - For D2H stores, the batch is the faster path: 0.89–0.94× against 0.84–0.88× for KV chunks, up to 1.03× for
    checkpoints. Pace it in ≈ 25 MB pieces if stores ever run back to back.
  - Today's per-run loop is out (0.13×).
- **`h2d_rate`.** ≈ 47 GB/s idle; ≈ 41–43 GB/s for the SM gather beside 4 decoding rows; 32–33 GB/s for paced DMA.
- **`stall_bytes`.** A load due now stalls a step for bytes / rate. At 5% of a 4-row step that is ≈ 48 MB on the 5080
  (22 ms) and ≈ 30 MB on the 5090 (14.7 ms), at 41–43 GB/s. A 1K-token load (35.7 MB) is about that size; a 154 MB
  payload (3.6 ms) is well above it.
- **`io_threads`.** 3: reads reach 2.54–2.55 GB/s against 2.17 at 2, and writes are flat from 1 thread. The I/O should
  also not need the GIL between calls (native threads), or a busy engine loop throttles it.
- **Lane rates.**
  - Read: ≈ 2.2–2.6 GB/s for units at 2–3 threads; 2.65–2.7 GB/s for 151 MB checkpoints.
  - Write: 2.1–2.2 GB/s burst; 1.77–1.80 GB/s sustained over 1 TB.
  - The serial shared-lane model overestimates read-heavy mixes by 16–24% on both drives.
- **Close flush.** ≈ 0.56 s per GB at 1.8 GB/s: ≈ 9 s for 16 GB, ≈ 56 s for 100 GB.
- **D9's SSD rate.** 2.2–2.7 GB/s here; the design assumes 2.6.
