# Milestone X, step 2: the offload store on the development vehicle

## Question

`docs/v1-offload.md` §9 step 2: is the offload store (`std.moe.offload@1` over `std.tiered`, F1 static placement,
F5a: every miss gathered from pinned RAM) bitwise `std.moe@1` all-resident on the vehicle, at a full, a ≈ 47% and
a ≈ 25% fit? And the two measurements §13 asks of step 2: Holon's own offload-path overhead (decode with every
expert cached against `std.moe@1`), and the gather's rate from the store's host allocation (`platform.host_memory`:
an anonymous map registered with `cudaHostRegister`) against `cudaHostAlloc`, with and without huge pages.

## Setup

| | |
|---|---|
| GPU | one RTX A5000 (sm_86, 24 GB, PCIe 4.0 x16 — another link than the bar desktops': ratios, not absolutes) |
| model | Qwen3-30B-A3B Q4_K_M (unsloth GGUF): 48 layers × 128 experts top-8; three classes — gate/up Q4_K (6,144 pieces of 1.77 MB), down Q4_K and down Q6_K (3,072 pieces each, 0.88 and 1.29 MB) |
| engine | bf16, graphs on, overlap on, `max_seqs` 64, `max_step_tokens` 4096, `max_model_len` 8192, `paged_min_tokens` 16384 (every run keeps the KV the trace needs, so the schedule is the trace's), `host_memory` 20 GiB |
| software | torch 2.11.0+cu130; `kernels.ggml_cuda` at its pin **without** the forced-tiling / compact-launch patches (on their own branch, not merged yet) |
| commit | `c7d59dd` (the gather runs at `0775b35`: its kernel is unchanged since; evidence: `evidence/x2_store_rtx-a5000_2026-10-17.json`) |

```bash
python spikes/x2_store/exact.py <gguf> <runs> "resident@16384;*=1.0@16384;@16384;*=0.47@16384"   # ~6 min
python spikes/x2_store/exact.py <gguf> <runs> "*=0.25@16384" 3                                    # run-identical
python bench/holon_tg.py --model <gguf> -n 128 -r 5 --engine-args "--max-seqs 1"                  # 3 rounds each,
python bench/holon_tg.py --model <gguf> -n 128 -r 5 --engine-args "--max-seqs 1 --host-memory 21474836480 \
    --plugin-options moe.store=std.moe.offload@1 moe.cache=*=1.0"                          # interleaved
python spikes/x2_store/gather_alloc.py --out <runs>/gather_alloc_<i>.json                         # 4 processes
python spikes/x2_store/streamk.py                                                                  # placement probe
python spikes/x2_store/record.py <runs> evidence/x2_store_rtx-a5000_2026-10-17.json c7d59dd <commands…>
```

## Results

**Exactness** — the judge's snapshot trace (64 requests: a 2,048-token prompt and its prefix-cache extension, 62
short ones through every decode bucket, seeded sampling, logprobs, prompt logprobs), each offload run against
`std.moe@1` all-resident with `holon_judge snapshot-diff` (cached experts: the static seed over each run's arena pool,
recomputed from its recorded size):

| Fit (`moe.cache`) | Cached experts (of 6,144) | Peak distinct misses per layer pass (staging 128) | Snapshot vs all-resident |
|---|---|---|---|
| `*=1.0` (every expert) | 6,144 | 0 | identical |
| fill (free VRAM above the KV floor) | 6,119 | 1 | identical |
| `*=0.47` | 2,888 | 67 | identical |
| `*=0.25` (×3 fresh engines) | 1,536 | 96 | identical, and run-identical |

The unpatched expert-id mmq is placement-invariant at the vehicle's arena sizes: `streamk.py` (Q4_K gate/up 2048 →
2×768 and Q6_K down 768 → 2048, 9–512 tokens, arenas of 128–6,280 slots with the experts at random slots) ran no
stream-k fixup kernel and gave bitwise-equal expert rows at every arena size — stream-k does not engage at ≥ 136
slots on any card. It does at small arenas: the GPU engine test's model (8 experts of width 1024, arenas of 10–26
slots) takes stream-k on sm_86 and differs from the 8-slot stack without the forced-tiling patch (D4), and passes
with it. For the vehicle the D4 patches matter for the class arenas' launch cost, not for exactness: unpatched, the
mmq grid grows with the arena's slots (7.9 ms against 2.5 ms per launch at S ≈ 3,000, T = 512, as measured on the
patch branch), so no prefill timing from the unpatched runs feeds the roofline.

**With the D4 patches merged** (`e074e9f`, `evidence/x2_store_patched_rtx-a5000_2026-10-17.json`): the `*=1.0`,
`*=0.47` and `*=0.25` snapshots are again identical to all-resident (patched and unpatched all-resident agree too),
and the full-fit trace takes 4.1 s against all-resident's 4.0 s (11.8 s unpatched; 18.0 s at 47%, 24.0 s at 25%,
miss-bound).

**Offload-path overhead** (decode tg128 at batch 1, depth 0; three interleaved rounds of five repetitions each):
`std.moe@1` 5.160 ms per token, the offload store with every expert cached 5.357 ms: **+0.20 ms per token
(+3.8%)** — per MoE layer one resolve and two gather launches that find nothing to copy, and the step's row kinds
and counts. Inside §13's 0.25–0.5 ms band's low end (that band also holds F11's pick and decay, step 3).

**Gather rate by host allocation** (8 CTAs × 256 threads, unroll 8, 64 blobs per call, 20 calls per graph,
from a 2 GiB arena): `cudaHostAlloc` 26.5–26.6 GB/s on this link. The registered anonymous map, ratio to it (median
of four processes, range):

| Blob | shared map | private map | private + MADV_HUGEPAGE |
|---|---|---|---|
| 2.65 MB | 0.998 (0.97–1.00) | 0.986 (0.96–1.00) | 0.997 (0.94–1.00) |
| 3.06 MB | 1.003 (1.00–1.01) | 1.002 (0.98–1.00) | 1.001 (0.97–1.00) |

No allocation is consistently slower: the spread is between processes (this host has two CPU sockets; which one
holds the pages moves a run by a few percent), and huge pages took (2 GB of `AnonHugePages`) without changing the
rate. `platform.host_memory` needs no change for the gather.

## What it decides

- Step 2's check passes on the A5000, unpatched and patched: every fit is bitwise all-resident, run-identical at the
  capped fit.
- The roofline's overhead term is ≈ 0.2 ms per token at F5a with static placement.
- The store keeps `mmap` + `cudaHostRegister` (exact bytes, D10); huge pages are neither needed nor harmful.
