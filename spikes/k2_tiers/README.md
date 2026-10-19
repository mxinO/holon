# Milestone K, step 2: `state_copy@1`'s CUDA providers and three bar-desktop measurements

## Question

Step 2 builds std.state_tier's copier on `state_copy@1` (`docs/v1-kv-tiers.md` D10). This directory answers:
- whether the `state_copy@1` providers meet the contract (L0);
- three numbers the rev 8.2 design leaves to step 2, on the RTX 5080 desktop:

| # | Measure | Feeds |
|---|---|---|
| a | The SM gather (8 CTAs) loading 0.57 / 1.29 GB beside compute-bound prefill of the 27B (UD-IQ3_S, rtx-5080 lock, 512-token steps): step time median / p90 against no copies | 3b's and 3c's predictions beside prefill (K-M3 measured decode only) |
| b | The host cost per load of a fragmented 16K / 32K load of the 27B: the row-list build in numpy as the tier does it, plus the provider's submit | the engine-loop cost of a resume |
| c | The mixed NVMe factor under rev 8.2's dispatch ("reads go first", D11): one writer, two or three readers, 557,056 B units, O_DIRECT, 50/50 and 75/25 read shares by bytes | D4's mixed charge |

## The providers (`plugins/holon_plugins/kernels/state_copy`)

Measured while they lived in `kernels.expert_io`; they moved unchanged to `kernels.state_copy` afterwards
(docs/v1-kv-tiers.md rev 8.3). The commands below name the new plugin.

The provider is `state_copy@1` with facts `direction`, `chunk` and an optional `copy`. It is supported when
`chunk % 16 == 0`.

- **`copy = "sm"` (the default for h2d).**
  - The kernel is `k_rows`: 8 CTAs × 256 threads, 16-byte loads two deep. That configuration measured best at
    8 CTAs on fragmented 34,816 B rows in K-M2: 46.0 GB/s against 41–43 at unroll 8.
  - Each CTA takes work items b, b + 8, …. A row longer than 64 KiB is split into 64 KiB tiles inside the kernel,
    so a few 3 MiB checkpoint segments still spread over all 8 CTAs.
  - Host work per call is one C++ pass over the two lists. It bounds-checks them and writes them into one block of
    torch's pinned caching allocator. One H2D copy then puts that block on the device; the allocator keeps the block
    until that copy lands.
  - It uses no copy engine, so an engine step's own copies never queue behind a load (K-M3's head-of-line
    blocking).
- **`copy = "dma"` (the default for d2h).**
  - One `cudaMemcpyBatchAsync` per call, with runs of rows consecutive in both views merged. K-M2 measured it at
    0.89–1.03× of the blob rate for stores.
  - On the legacy default stream, which the batch refuses, it falls back to one `cudaMemcpyAsync` per run.

## Predictions (fixed before measuring)

| # | Prediction |
|---|---|
| L0 | Bitwise equal to the oracle (`std.state_tier.reference`) in both directions and both paths at 34,816 / 65,536 / 61,440 / 3,145,728 B, and for n = 0. `compute-sanitizer` memcheck clean |
| a, one 0.57 / 1.29 GB load per prefill step | The step (≈ 0.22 s for 512 tokens) overlaps the load for 12 / 27 ms. Median +0.5–2%, p90 ≤ +3% |
| a, 1.29 GB loads back to back | The gather holds 8 of 84 SMs and ≈ 5% of DRAM bandwidth for the whole step. Median +3–8%, p90 ≤ +10% (K-M3: +7% beside decode). Gather rate beside prefill 40–46 GB/s |
| b, numpy row lists (src and dst, 16 segments) | 16K (16,384 chunks): 30–80 µs; 32K (32,768 chunks): 60–160 µs |
| b, provider submit, sm (bounds check, pinned copy, one H2D, launch) | 16K: 40–100 µs; 32K: 80–200 µs. Together ≈ 4–10 ns per chunk, against K-M2's per-run loop at 5.9 µs per chunk and the batch DMA's 0.3–0.7 µs |
| b, provider submit, dma (d2h) | ≈ K-M2's batch: 5 / 21–25 ms at 16K / 32K (the driver's per-copy cost dominates) |
| c, measured / modelled (1 writer + 2 or 3 readers) | At 50/50: 0.80–0.90. At 75/25: 0.75–0.85. Read priority bounds write concurrency, but the drive's own interference stays (K-M4: 2 + 2 threads 0.84 / 0.76; 1 + 1 at 50/50 0.84) |

## Setup and method

- **Desktop.** The RTX 5080 desktop: one allocation and one CUDA 13.0 container, with a uv-synced venv of the K
  worktree. `kernels.expert_io` and `kernels.ggml_cuda` were prebuilt for sm_120 in the fresh container. The clock
  screen (`bench/clock_check.py`, rtx-5080 lock) was steady: SM p10 / p50 2,640 / 2,647 MHz, no slowdown samples.
- **L0** (now `plugins/tests/test_state_copy.py::test_state_copy_moves_the_listed_rows_exactly`).
  - Copies go between a device pool and a mapped host pool's device view, through the engine's plugin selection, on
    a stream of their own.
  - Each cell copies half of 128 rows (16 for 3 MiB chunks), scattered plus a run of three rows consecutive in both
    pools, then an empty list.
  - The destination is compared bitwise with `std.state_tier.reference`, and must differ from before.
  - 16 cells: {sm, dma} × {h2d, d2h} × the four chunk sizes.
  - `compute-sanitizer --tool memcheck` ran on three cells covering `k_rows` both ways and the DMA path.
- **a.**
  - Pools: a layer-major device KV pool and a mapped host KV pool of 2,400 units each (1.34 GB), allocated before the
    engine.
  - Engine: the 27B from its rtx-5080 lock with `max_step_tokens` 512 and `chunk_with_decode` 0, so steps are
    prefill-only 512-token chunks of one 8,192-token prompt at a time. `max_model_len` is 16K: the 1.34 GB device pool
    leaves less KV than the lock's 49K.
  - The provider is called from the engine's loop on a copy stream of its own, as the tier will. The loads are
    fragmented 27B KV loads: 16,384 rows (0.57 GB) and 37,184 rows (1.29 GB).
  - Arms:
    - no copies;
    - one 0.57 GB load enqueued before each step;
    - one 1.29 GB load before each step;
    - 1.29 GB loads back to back (about 12 queued ahead).
  - Timing: per-step wall time between `engine.step()` returns. 10 steps per segment after 2 warm-up steps, arms
    forward then backward, 4 rounds.
- **b.** Per load, two host times:
  - the row lists of both sides in numpy, as the tier builds them: `rows = s · capacity + unit` over 16 segments;
  - the provider's submit, on a copy stream that is not the legacy default stream.

  Each is the median of 50 loads with the GPU idle between them, plus 12 loads queued back to back.
- **c.** Rev 8.2's dispatch, "reads go first": while a read is queued, at most one write job is in flight.
  - Setup: one writer thread and two or three reader threads of 557,056 B units with O_DIRECT, on the node-local
    NVMe. Each kind is held at its byte share (50/50, 75/25), plus a free-running cell.
  - Model: D4's serial lane 1 / (f / read_N + (1 − f) / write_1), from pure rates at the same thread counts measured
    just before each cell. Reads never overtake the write that frees their own units, which is vacuous here (the
    slots are disjoint).
  - The test files were removed (`test_file_removed: true`).

```bash
python -m holon_plugins.kernels.state_copy.prebuild --sm 120
python -m pytest plugins/tests/test_state_copy.py
cd spikes/k2_tiers
CUDA_VISIBLE_DEVICES=0 python bench.py --tag rtx-5080 --only b,a,c --model <Qwen3.8-27B-UD-IQ3_S.gguf> \
    --lock <repo>/plans/unsloth/Qwen3.8-27B-GGUF/<file>/rtx-5080/cu130-torch2.11/10.lock \
    --nvme-dir <node-local dir> --l0 <l0 summary json> --clock-screen <clock_check json>
```

## Results

Evidence: `evidence/k2_state_copy_rtx-5080_2026-10-18.json` (L0 for the RTX 5080 and the RTX A5000, a, b) and
`evidence/k2_mixed_nvme_rtx-5080_2026-10-18.json` (c).

### L0

| | RTX A5000 (sm_86) | RTX 5080 (sm_120) |
|---|---|---|
| 16 cells against the oracle, bitwise | 16 passed | 16 passed |
| `compute-sanitizer` memcheck (3 cells) | 0 errors | 0 errors |
| `pytest core/tests std plugins/tests -k "tier or state_copy or expert_io"` | 146 passed, 1 skipped | – |

On the A5000, memcheck with API errors reported shows exactly one report: torch's expandable-segment `cuMemCreate`
being refused under the sanitizer. It is not ours.

### a — the gather beside compute-bound prefill (RTX 5080, 27B UD-IQ3_S, 512-token steps)

| arm | step p50 ms | p90 ms | median × | p90 × | mean × | gather GB/s |
|---|---|---|---|---|---|---|
| no copies | 218.6 | 230.2 | 1 | 1 | 1 | – |
| one 0.57 GB load per step | 214.8 | 224.6 | 0.983 | 0.976 | 0.986 | 43.3 |
| one 1.29 GB load per step | 216.3 | 226.3 | 0.990 | 0.983 | 0.993 | 43.8 |
| 1.29 GB loads back to back | 256.8 | 269.9 | **1.175** | **1.173** | **1.176** | 44.8 |

- **One load per step: no measurable cost** (prediction: +0.5–2%). The per-step ratios below 1 are the arms' position
  in the 16-step prompt cycle; segments of one arm vary by ±3%.
- **Back to back: +17.5%** (prediction: +3–8%, p90 ≤ +10%). The gather holds 8 of the 84 SMs through every kernel of
  the step. The cost is nearly twice those SMs' 9.5% share. A likely cause, not profiled here: wave quantization, a
  compute-bound kernel whose grid fills the card running on 76 SMs or taking an extra partial wave. The RTX A5000 shows the same at +31% (8 of 64 SMs, 13 GB/s, a dev run of one
  round).
- The cost scales with copy time, not bytes. A 1.29 GB load overlaps ≈ 29 ms of a ≈ 219 ms step: 29 × 0.175 ≈ 5 ms,
  ≈ 2% of that step, hidden in the per-step arms' noise.
- The gather keeps 43–45 GB/s beside prefill.

### b — host cost per load (RTX 5080 desktop)

| load | rows | numpy row lists | submit, `sm` (h2d) | together | per chunk | submit, `dma` (d2h) |
|---|---|---|---|---|---|---|
| 16K | 16,384 | 28 µs | 39 µs (queued: 38) | **67 µs** | **4.1 ns** | 4.6 ms (queued: 12.6 ms) |
| 32K | 32,768 | 53 µs | 62 µs (queued: 54) | **115 µs** | **3.5 ns** | 19.9 ms (queued: 24.3 ms) |

- Predictions: lists 30–80 / 60–160 µs; `sm` submit 40–100 / 80–200 µs; 4–10 ns per chunk; `dma` submit 5 / 21–25 ms.
- The `sm` path is within its predictions: about 1,400× cheaper per chunk than the per-run loop (5.9 µs per chunk)
  and ≈ 70–190× cheaper than the batch DMA's submit (K-M2: 0.3–0.7 µs per chunk).
- The `dma` submit is the driver's: as in K-M2, it grows faster than the list (0.28 → 0.61 µs per chunk) and blocks
  longer when copies are already queued. A large store by DMA therefore costs the engine loop ms per call.

### c — mixed NVMe under "reads go first" (RTX 5080 desktop, node-local NVMe)

| readers + writer | read share held | reads GB/s | writes GB/s | measured GB/s | modelled GB/s | mixed factor | pure read / write GB/s |
|---|---|---|---|---|---|---|---|
| 2 + 1 | 0.50 | 0.85 | 0.85 | 1.70 | 2.16 | **0.788** | 2.17 / 2.16 |
| 2 + 1 | 0.75 | 1.24 | 0.41 | 1.65 | 2.14 | **0.769** | 2.13 / 2.20 |
| 2 + 1 | free (0.24) | 0.47 | 1.48 | 1.95 | 2.13 | 0.913 | 2.07 / 2.15 |
| 3 + 1 | 0.50 | 0.93 | 0.93 | 1.86 | 2.34 | **0.796** | 2.54 / 2.17 |
| 3 + 1 | 0.75 | 1.46 | 0.49 | 1.94 | 2.44 | **0.794** | 2.54 / 2.18 |
| 3 + 1 | free (0.35) | 0.69 | 1.28 | 1.97 | 2.30 | 0.854 | 2.55 / 2.19 |

Predictions: 0.80–0.90 at 50/50, 0.75–0.85 at 75/25. Measured: 0.79–0.80 at 50/50 (just below), 0.77–0.79 at 75/25.

- One writer under read priority changes nothing against K-M4's 2 + 2 threads (0.84 / 0.76) or 1 + 1 (0.84). The
  factor is the drive's own read/write interference, not thread counts.
- A third reader raises both throughput and the model; the factor stays ≈ 0.79.
- Free-running, a single writer still takes most of the bytes: reads fall to 0.47–0.69 GB/s.

### What this suggests (for the owner)

- **Loads beside prefill.** The gather costs in proportion to its copy time, ≈ 17.5% of the overlapped time on the
  RTX 5080. One 32K resume per 512-token step is free in practice. Continuous streaming beside prefill (a burst of
  resumes) costs up to 1.18×.
- **D4's mixed factor.** ≈ 0.79 for this drive under rev 8.2's dispatch, for 2–3 readers and both read shares. A
  single factor of 0.78–0.80 per host fits every held cell within ±1.5%.
- **Store path.** `dma` stays the default for stores, being faster on the wire, but its submit costs the engine loop
  4.6 / 20 ms at 16K / 32K-sized stores. Write-through publishes are small, so this matters only for large eviction
  write-backs: tune can pick the `sm` scatter (`copy = "sm"`) there, at 0.84–0.88× of the wire rate and ≈ 0.1 ms of
  host time.
