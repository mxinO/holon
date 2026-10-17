# x4_refs — reference engines for milestone X's development gate (RTX 5080)

Acceptance 3 of `docs/v1-offload.md` (D14's method, §13's roofline): Qwen3-30B-A3B Q4_K_M
(`unsloth/Qwen3-30B-A3B-GGUF`), batch 1, greedy, contexts ≤ 32K, at the natural fit (≈ 47% of the expert bytes in
VRAM) and the capped fit (≈ 25%). Record: `evidence/x4_refs_rtx-5080_2026-10-17.json`.

## Method

- **Engines** (`LLAMA_ARCHS=120a bash bench/setup_baselines.sh offload`): llama.cpp `f74e2fa6a31c` (2026-10-16,
  includes `--moe-cache-mib` from PR #29887), `llama-server`; FreeToken `21d9578bde22` (2026-10-16), torch
  2.14.1+cu130 with `[accel]`, its offline `LLM` on the GGUF.
- **Driver** `bench/offload_ctx.py`: every arm in a fresh process; an untimed warm-up request (512 + 64 tokens), then
  one request per context of 1,024 / 4,096 / 32,768 prompt tokens generating 256 tokens, `ignore_eos`. Prompts are
  spans of natural text (the judge prompt set's token ids, distinct offsets), the same ids for both engines, so the
  expert caches see real routing. Prefill tok/s = prompt / prompt time; decode tok/s = 255 / (last − first token).
  llama.cpp is timed by its own `timings`, FreeToken in process as each token reaches its frontend. Every engine
  allocates a 33,792-token context. 13 arms × 3 rounds, interleaved (reversed on odd rounds), the GPU checked idle
  before each; medians with min–max. Arms: `arms.json`.
- **Fits.** The experts are 16,740 MiB (6,144 × 2.86 MB). A `--moe-cache-mib` of 7,900 holds 47.1% of them and 4,200
  holds 25.0% (1,448 / 769 slots per layout group). FreeToken uses `--moe-cache-rate 0.47 / 0.25` (2 × 1,444 slots at
  0.47). The static split gets 47.9% / 25.0% at `--n-cpu-moe 25 / 36`. The capped `--fit` uses `-fitt 7150` (a
  7,150 MiB margin), which leaves 24.7% of the experts on the GPU. Its own natural choices, recorded beside these:
  - `--fit` at its default 1 GiB margin keeps 61.3% on the GPU.
  - The tuned `--n-cpu-moe` sweep (4K, one round; 16–40) is monotonic: fewer CPU layers is faster. Its best that
    fits at 32K is `--n-cpu-moe 16` (66.4%); 15 runs out of memory.
- **No-approximation settings** equal the defaults for both engines: neither has an approximate default.
  - llama.cpp's default `--load-mode` is `mmap`, which keeps host experts in pageable memory (it warns and suggests
    `--load-mode none`). Each cache arm therefore runs twice: at the default and pinned (`--load-mode none`).
  - FreeToken's `auto` resolved to `offload` on this GGUF in every run (log: "Auto-selected MoE strategy: offload";
    no `ft bench bw` profile, and its CPU path has no k-quant kernels). So `auto` and `offload` ran the same
    configuration, and their numbers agree within 0.3%.
  - FreeToken with no sizing flag picks 72.7% of the experts with an 8,206-token KV. That cannot hold the 32K rows.
    One informational run at 1K / 4K gave 226 / 207 tok/s decode.

## Results (tok/s, median [min–max] of 3 rounds)

**Decode** (256 tokens after the prompt)

| Arm | Experts in VRAM | 1K | 4K | 32K |
|---|---|---|---|---|
| llama.cpp `--fit` (default) | 61.3% | 78.2 [77.9–80.4] | 76.3 [73.8–76.3] | 61.5 [61.2–61.5] |
| llama.cpp `--n-cpu-moe 16` (tuned, best that fits) | 66.4% | 88.4 [88.0–88.8] | 83.1 [82.3–83.2] | 67.2 [66.6–67.7] |
| llama.cpp `--n-cpu-moe 25` | 47.9% | 64.2 [64.2–64.4] | 61.4 [58.8–61.8] | 51.3 [50.7–51.8] |
| llama.cpp `-cmoe --moe-cache-mib 7900` (mmap, default) | 47.1% | 159.1 [158.9–159.8] | 109.2 [108.8–109.8] | 81.7 [81.5–82.2] |
| llama.cpp the same, `--load-mode none` (pinned) | 47.1% | 188.8 [188.8–188.9] | 147.2 [147.2–147.3] | 102.0 [101.9–102.0] |
| FreeToken `auto` = `offload`, rate 0.47 | 47.0% | 213.8 [213.8–213.9] | 175.3 [175.2–175.4] | 111.4 [111.4–111.5] |
| llama.cpp `--fit -fitt 7150` | 24.7% | 46.6 [45.8–46.6] | 45.7 [45.6–45.8] | 39.9 [39.8–40.3] |
| llama.cpp `--n-cpu-moe 36` | 25.0% | 47.4 [47.2–47.5] | 45.5 [44.7–46.0] | 40.6 [40.5–40.7] |
| llama.cpp `-cmoe --moe-cache-mib 4200` (mmap) | 25.0% | 54.6 [54.0–54.8] | 46.9 [46.4–47.0] | 42.5 [42.2–42.7] |
| llama.cpp the same, pinned | 25.0% | 100.3 [100.1–100.3] | 87.0 [86.9–87.0] | 70.5 [70.4–70.5] |
| FreeToken `auto` = `offload`, rate 0.25 | 25.0% | 115.6 [115.6–115.7] | 109.5 [109.5–109.6] | 76.6 [76.5–76.6] |

**Prefill**

| Arm | 1K | 4K | 32K |
|---|---|---|---|
| natural: `--fit` / `--n-cpu-moe 16` / `--n-cpu-moe 25` | 1314 / 1340 / 963 | 1194 / 1245 / 892 | 1158 / 1184 / 879 |
| natural: cache mmap / pinned | 1085 / 1833 | 933 / 1645 | 942 / 1522 |
| natural: FreeToken | 2707 | 10657 | 5381 |
| capped: `--fit` / `--n-cpu-moe 36` | 735 / 710 | 670 / 658 | 681 / 662 |
| capped: cache mmap / pinned | 753 / 1525 | 678 / 1403 | 681 / 1308 |
| capped: FreeToken | 2704 | 10653 | 5387 |

Prefill spread is ≤ 1.5% for every arm.

**llama.cpp's cache counters.** These come from the exit log and cover the warm-up plus the three requests:
- At 47.1%, the hit rate is 95.2%: 15,390 misses, of which ≈ 2,900 are the first fill. That leaves ≈ 15 misses
  (43 MB) per decode token.
- At 25.0%, the hit rate is 79.6%.

Peak VRAM:

| Configuration | Peak VRAM |
|---|---|
| Cache arms | 12.5 GB |
| `--n-cpu-moe 16` | 15.7 GB |
| `--fit` | 14.9 GB |
| FreeToken (its KV takes the rest) | 15.4 GB |

## llama.cpp's per-layer host sync (nsys)

`nsys_moe_gap.py` profiles the natural-fit cache arm at 1K context and reads the last 200 decode steps. For each MoE
layer's ids copy, it measures the GPU's idle gap from the last kernel before the copy to the first kernel after it:
the stream sync, the id read, the host LRU plan, the miss uploads, the slot-map upload and the next split's launch.
The upload time inside each gap is subtracted from it.

Profiles with `--cuda-graph-trace=graph` run close to the unprofiled rate (175.7 against 188.8 tok/s). With `=node`,
the profiler inflates the gaps (≈ 29 µs per layer).

| Pinned (`--load-mode none`), graph trace | Value |
|---|---|
| MoE layers with a host round trip per token | 48 |
| Host gap per layer, no miss (89.7% of layers) | median 17.8 µs (p10–p90 17.3–20.8) |
| Host gap per layer, with a miss (10.3%), excluding the upload | median 26.7 µs; upload 71 µs for 3.1 MB (≈ 43 GB/s) |
| Host gap per layer, all layers | mean **19.5 µs**, median 17.8 |
| **Per token** (step 5.36 ms median, 5.54 mean) | GPU work 3.93 ms, host gaps **0.94 ms** (17%), uploads 0.27 ms median / 0.41 mean (11–17 MB) |

With the default `mmap`, a layer without a miss costs the same 18.3 µs. A layer with a miss pays a pageable,
synchronous copy: 113 µs of upload plus 114 µs of host time. A token then costs 1.47 ms of host gaps (mean) and
0.66 ms of uploads.

## What this implies for §13's bands

**The natural-fit `--moe-cache-mib` row, recomputed.** The roofline band's width was llama.cpp's unknown per-layer
sync, put at 10–50 µs. The measured value is 19.5 µs per layer, which is 0.94 ms per token.

The model behind the recomputed band (`record.py`):
- llama.cpp time per token = G + H + B / 43 GB/s, where G (its GPU work) is solved from the pinned arm's measured
  rate at each context.
- Holon F5a time per token = G / k + O + s · B / 47.5 GB/s.

The parameters:

| Parameter | Range | Source |
|---|---|---|
| k (Holon all-resident speed / llama.cpp's) | 0.93 to parity | 0.93 from the release record |
| O (Holon overhead) | 0.45–0.55 ms | step 2 + F11 |
| s (lfu's miss bytes / lru's) | 0.77–1.0 | M6 |
| B (miss bytes per token) | 17–43 MB | nsys at 1K to the run average |

| Holon F5a over | 1K | 4K | 32K |
|---|---|---|---|
| llama.cpp cache, pinned (bar 1.1×), k = 0.93 | 1.02–1.11 (central 1.06) | 1.00–1.07 (1.03) | 0.98–1.02 (1.00) |
| the same, k up to parity | 1.02–1.17 | 1.00–1.13 | 0.98–1.09 |
| llama.cpp cache, default mmap (bar 1.1×) | 1.22–1.39 | 1.35–1.53 | 1.22–1.36 |
| FreeToken (bar 1.0×) | 0.90–1.04 (0.94) | 0.84–0.95 (0.86) | 0.90–1.00 (0.91) |

- **Against the pinned cache, the upper edge is below 1.1×.** It is 1.02–1.07× at 4K and 32K, and reaches 1.11×
  only at 1K. The roofline's ≈ 0.92–1.47× (central 1.18×) becomes ≈ 0.98–1.11× (central ≈ 1.0–1.06×) under §13's
  k = 0.93. Even at kernel parity, the 32K row stays below 1.1×. By §8's rule this natural-fit row predicts a miss,
  so it goes to the owner before step 4.
- **Against llama.cpp's default (`mmap`),** the bar holds (≥ 1.2×), so the verdict depends on which llama.cpp
  configuration acceptance 3 names.
- **The reference is faster than the roofline assumed, in three ways:**
  - llama.cpp's GPU work per token is 3.93 ms at 1K. The roofline assumed ≈ 4.5 ms for all-resident work.
  - Its sync costs ≈ 19.5 µs per layer, not the band's upper 50 µs.
  - This prompt set hits more than M6's traces: 95% LRU hits at 47%, against M6's ≈ 90%.
- **FreeToken is faster than llama.cpp's best cache config:** +13% / +19% / +9% decode at the natural fit and
  +15% / +26% / +9% at the capped fit. §13's thin ≈ 1.0–1.1× FreeToken band becomes ≈ 0.84–1.04×: Holon F5a is
  predicted to miss the FreeToken decode bar at 4K and 32K unless its kernels reach parity with llama.cpp's.
- **Prefill against FreeToken is a risk too.** FreeToken streams every expert once per ≤ 8K chunk, ≈ 0.38 s,
  so it reaches 10.7K tok/s at 4K and 5.4K at 32K. Holon's mmq expert GEMMs at 4096 tokens (10.97 ms per layer, M1)
  bound a compute-only prefill at ≈ 7.8K tok/s. Its 1K row (2.7K tok/s, stream-bound) is within the staging path's
  reach.
- **The static-split rows keep their margin.** At the gate's own residency, the best static split decodes at
  64 / 61 / 51 tok/s at 47.9% and 47 / 46 / 41 at 25%. llama.cpp's natural `--n-cpu-moe 16` (66% of the experts)
  reaches 88 / 83 / 67. Against it, the 1.5× bar needs ≥ 133 / 125 / 101 tok/s. F5a's predicted
  193–222 / 148–167 / 100–111 (the same model) meets it at 1K and 4K. At 32K it straddles the bar.

## Reproduce

```bash
LLAMA_ARCHS=120a bash bench/setup_baselines.sh offload               # where nvcc (CUDA 13) is on PATH
python bench/offload_ctx.py run --name x4_refs --gpu 0 --model <Qwen3-30B-A3B-Q4_K_M.gguf> \
    --arms spikes/x4_refs/arms.json --rounds 3
python bench/offload_ctx.py worker --engine llama --server bench/.envs/llama.cpp-offload/bin/llama-server \
    --model <gguf> --flags "-cmoe --moe-cache-mib 7900 --load-mode none -lv 4" --contexts 1024 \
    --wrap "nsys profile -t cuda --cuda-graph-trace=graph --sample=none --cpuctxsw=none -o <report>"
nsys export --type sqlite -o <report>.sqlite <report>.nsys-rep      # the nsys that recorded the report
python spikes/x4_refs/nsys_moe_gap.py <report>.sqlite > bench/.out/x4_nsys/graph_pinned.json
python spikes/x4_refs/record.py --out evidence/x4_refs_rtx-5080_<date>.json
```
