# Milestone X, steps 6 and 5: F5b (the CPU's share of the misses) and F9 (the SSD tier, exclusive RAM)

## Question

`docs/v1-offload.md` §9, steps 6 and 5 (run in that order): does the CPU's share of a layer's misses (F5b: the
`cpu_task@1` handshake, `cpu_experts@1` on ggml-cpu, `moe_resolve@1`'s `plan[n]`) pay on this card, stay
run-identical with adaptive placement on, and stay within L2's bounds against a path-aware floor at a frozen placement?
Do the F5a configurations stay byte-identical to all-resident? With RAM holding fewer experts than the store
(exclusive tiers, F9's SSD tier, F11 with deferred admission), is the vehicle still exact, and what does it decode at?

## Setup

| | |
|---|---|
| GPU | RTX A5000 (sm_86, 24 GB, PCIe 4.0) on a shared host; the process and its CPU workers on the GPU's NUMA node |
| CPU tier | `x86-avx2` (kernels.ggml_cpu's AVX2 module; the AVX-512 one builds here but cannot run) |
| model | Qwen3-30B-A3B Q4_K_M (unsloth GGUF): 48 layers × 128 experts top-8; classes gate/up Q4_K, down Q4_K, down Q6_K |
| "SSD" | the checkpoint's file where it lives here: a network filesystem, not a local NVMe (M7 measured those) |

```bash
M=<Qwen3-30B-A3B-Q4_K_M.gguf>
python -m holon_tools tune $M --cpu-split --no-lock --batch 1 --tokens 128 --ops rmsnorm --max-seqs 1 \
    --max-model-len 4096 --host-memory 21474836480 --loop-core 2 \
    --plugin-options moe.store=std.moe.offload@1 'moe.cache=*=0.25' cpu_task.cores=4-9 --json tune.json
python spikes/x6_cpu/decode.py $M out.json "<arm>|<plugin options k=v;…>" …   # decode A/B, natural prompts
python spikes/x6_cpu/run.py $M <dir> "<name>|<runs>|<plant>|<options>" …     # snapshot traces
python spikes/x6_cpu/l2.py $M cand.safetensors split.json "<options>" [plant]  # F5b's L2 candidate + its split
python -m holon_judge golden --model $M --config Qwen/Qwen3-30B-A3B --q8-1 mmvq --cpu-split split.json \
    --dtype bfloat16 --device cuda --prompts judge/prompts/v1.jsonl --out floor.safetensors
python -m holon_judge compare --golden judge/goldens/qwen3-30b-a3b-q4_k_m/v1-fp32.safetensors \
    --candidate cand.safetensors --floor floor.safetensors
python spikes/x6_cpu/record.py <runs> evidence/x6_cpu_rtx-a5000_<date>.json <commit>
```

Snapshot runs use the step-3 engine settings (`max_seqs` 64, graphs and overlap on); the L2 candidate runs one token
per step (Tier B's decode path; F5b's rule per token, as the floor applies it). Wrap heavy commands in a
memory-capped scope on a shared host.

## Results (step 6)

**F5b's gain** (`tune --cpu-split`, batch 1, 25% cached, F11, the prompt set's first prompt, 128 tokens):

| arm | tok/s |
|---|---|
| F5a | 68.1 |
| half the misses, 2 / 4 / 6 threads | 83.7 / 93.5 / 93.6 |
| 6 threads, share 0.25 / 0.4 / 0.5 / 0.6 / 0.75 / 1.0 | 76.8 / 79.6 / 93.6 / 94.9 / **105.3** / 52.4 |

**+55% at share 0.75** (a first run without the 1.0 arm: 67.8 → 103.9, +53%, the same pick) (`moe.cpu_plan=1,2,2,3,4,5,5,6`, 6 threads): F5b enters a lock here by D4's 5% rule. A
separate A/B (`decode.py`, share 0.5, 4 threads): 78.2 → 94.8 tok/s (+21%). The A5000's PCIe 4.0 gathers at about half
the RTX 5080's rate, so its CPU share is larger than M2's n / 2; the 5080 decides at step 4 with the same command.

**Snapshots** (the judge's snapshot trace, 64 requests):

| configuration | vs all-resident | run-identical | notes |
|---|---|---|---|
| F11, 25% cached | identical | — | 2,252 admissions; parity with the replay at every step |
| F11, 47% | identical | — | 3,255 admissions; parity |
| F1, 25% | identical | — | |
| F11 25% + F5b (tuned split) | differs (bounded) | ×3 | 2,185 admissions, 92,134 CPU experts, 6,048 handshakes per run, flag 0; parity (the CPU's misses excluded) |
| + plant `cpu-wait-skipped` | differs | **fails** (runs differ from step 3) | fired 36 eager + 30 captured patches per run |

**F5b's bounded L2** (F1 frozen at 25%, the tuned split, 32,783 positions at one token per step, 1.58 M handshakes):

| | top-1 | KL p50 | KL p99 | ΔPPL vs floor |
|---|---|---|---|---|
| plain mmvq floor vs golden | 98.54% | 1.18e-4 | 3.98e-2 | |
| path-aware floor vs golden (computed before the GPU run) | 98.39% | 1.25e-4 | 4.30e-2 | |
| F5b candidate | 98.54% | 1.26e-4 | 3.92e-2 | −0.08% ± 0.09% |
| budget | ≥ 98.14% | ≤ 2.49e-4 | ≤ 8.60e-2 | ≤ 0.48% |

**PASS.** The floor's KL is 1.06× / 1.08× the plain floor's: Q8_K's larger error reaches only the CPU's share of the
experts.

**Plant `cpu-gate-up-order`** (the CPU experts read the merged gate / up halves swapped; fired in all 48 layers'
instances): L2 **fails** — top-1 62.7%, KL p50 0.47, p99 12.8, ΔPPL +473% against the same floor. F5a configurations
build no CPU expert instance, so it cannot fire there. Both plants behave as §6 predicted.

## Results (step 5)

**Exactness, RAM-capped** (`moe.tiers=std.tiered.pinned@1,std.tiered.gguf_direct@1`, `moe.ram=*=0.5`,
`host_memory` 12 GiB — below the experts' 17.6 GB —, 25% cached): the snapshot trace is **byte-identical** to
all-resident under F11 (2,115 admissions; F11's residency its replay's at every step, the device's map the mirror's)
and under F1.

**Decode** (batch 1, 512 tokens, two interleaved passes over the arms, 3 rounds each; the checkpoint on a network
filesystem):

| arm | host memory | tok/s |
|---|---|---|
| RAM holds every expert (F11, the default) | 20 GiB | 70.9 / 71.2 |
| RAM holds half the pieces | 12 GiB | 18.6 / 17.5 |
| RAM holds 37.5% | 10 GiB | 11.3 / 11.6 |

RAM-short is a fit mode (D10: 1.9–8.3× slower on the desktops' NVMe; here 3.9× and 6.2×, bound by the file's reads
over the network). A first pass at 128 tokens timed `generate(N + 1) − generate(1)` with the prompt re-read from the
SSD each time, which the reads' variance swamped (one round at 143.8 tok/s): the 512-token pass replaces it.
