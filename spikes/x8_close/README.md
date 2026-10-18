# x8_close — milestone X's close: F5b on the RTX 5080, the capped-fit and F1 rows, the offload locks, D15

`docs/v1-offload.md` §1 acceptance 2, 3 and 7, D4 (F5b's ≥ 5% rule), D15, §6 (F5b's bounded L2), §13 (F5b's prediction
for the RTX 5080: +5–8% at the capped fit). Qwen3-30B-A3B Q4_K_M (`unsloth/Qwen3-30B-A3B-GGUF`) on one RTX 5080
desktop (CPU tier `x86-avx512`), batch 1, greedy, no drafting, contexts ≤ 32K. Record:
`evidence/x8_close_rtx-5080_2026-10-18.json` (every round, the tune record, the lock texts and verdict logs, the
propagation). Holon at `40811a0` (the branch after step 8 part 1, the docs and the `kernels.ggml_cpu` prebuild fix
below); llama.cpp `f74e2fa6a31c` and FreeToken `21d9578bde22`, x4_refs' builds.

## Method

- **Rows:** `spikes/x4_gap/ctx.py`, as at step 4 (`spikes/x4_gate/README.md`): every arm in a fresh process, an
  untimed 512 + 64 warm-up, one request per context (1,024 / 4,096 / 32,768 prompt tokens of natural text, the same
  ids for every engine) generating 256 tokens; every engine allocates a 33,792-token context; arms interleaved and
  reversed on odd rounds (ABBA), 3 rounds, one desktop, one day; medians. Spread over the median: Holon ≤ 0.6%
  decode (F1 with F5b ≤ 1.2%) and ≤ 0.5% prefill; FreeToken and llama.cpp's caches ≤ 1.6%; llama.cpp's static splits
  up to 4.5% decode.
- **Arms** (`arms.json`, `arms_natural.json`): Holon under the natural lock's settings and kernel pins (`lfu`,
  `ring_copy` `dma`, the held-out profile) with `moe.cache=*=0.25` (`holon_25`), the same with `--loop-core 0`
  (`holon_25_lc0`, F5a's control for F5b's arm, which pins the loop to core 0 beside the CPU workers) and with
  tune's split (`holon_25_f5b`); F1 is `std.tiered.static@1` with the same profile seed, F5a and F5b, at 25% and at
  its own natural fit; the references are x4_refs' and step 4's arms (capped: `-fitt 7150`, `--n-cpu-moe 36`,
  `--moe-cache-mib 4200` mmap and pinned, FreeToken `--moe-cache-rate 0.25`; natural: `--fit` at its default margin,
  `--n-cpu-moe 16`, the caches at the lock's share 52.7%: 8,817 MiB and rate 0.527). `arms_natural.json`'s
  `STATIC_PIN` is F1's own fill (`fit.py --opts moe.placement=std.tiered.static@1 …`, with F5b's options for
  `f1_nat_f5b`), substituted at run time; `arms_natural_f5b.json` is F5b at the natural fit.
- **F5b:** `holon_tools tune --cpu-split --no-lock` at the capped fit under F11 (the lock's settings, `--loop-core 0`,
  128 tokens of the prompt set's first prompt): F5a, then 2 / 4 / 6 CPU threads at half the misses, then the share
  at the best count; D4's rule takes the best at ≥ 5%.
- **F5b's exactness class** (§6, `bounded`): `spikes/x6_cpu/l2.py` (F1 frozen at 25%, one token per step, tune's
  split) on the RTX 5080, its split's path-aware floor (`holon_judge golden --q8-1 mmvq --cpu-split`) on an RTX A5000,
  `holon_judge compare`; run-identity: the judge's snapshot trace ×3 with F11's adaptive placement and the split on.

## F5b's tune at the capped fit (D4, §13)

| Arm | tok/s |
|---|---|
| F5a | 100.1 |
| half the misses, 2 / 4 / 6 threads | 104.6 / **111.9** / 111.6 |
| 4 threads, share 0.25 / 0.4 / 0.5 / 0.6 / 0.75 / 1.0 | 101.3 / 104.1 / **111.9** / 111.8 / 109.7 / 66.1 |

**+11.7% at share 0.5, 4 threads: F5b is taken** (`moe.cpu_plan=1,1,2,2,3,3,4,4`, `moe.cpu_isa=x86-avx512`,
`cpu_task.threads=4`). §13 predicted +5–8%; the interleaved rows measure +15.6% / +9.0% / +7.4% at 1K / 4K / 32K
(F5b over `holon_25_lc0`, each beyond 2 sd). The share is M2's n / 2, as predicted for Q4_K_M on this card.
`--loop-core 0` alone changes nothing (≤ 0.01%). Prefill is unchanged by F5b (prompt passes are larger than the share's
pass size).

At the natural fit F5b does not fit beside the lock's pin (the CPU share's buffers: the arena falls one unit short,
refused at the first launch, D4); with F5b on, the fill is 3,221 / 1,692 / 1,529. There tune measures **+5.4%** (F5a
203.3, share 0.5 at 6 threads 214.3 tok/s; `tune_split_nat.json`), but the interleaved rows (`arms_natural_f5b.json`, 3
rounds) contradict it:

| Natural fit, decode tok/s | 1K | 4K | 32K |
|---|---|---|---|
| the lock (F5a, 3,236 / 1,618 / 1,618) | 234.7 | 190.8 | 117.5 |
| F5a at F5b's fit (3,221 / 1,692 / 1,529), `--loop-core 0` | 235.6 | 191.5 | 118.0 |
| F5b at that fit (share 0.5, 6 threads) | 219.7 | 180.3 | 110.9 |
| F5b over F5a at the same fit | **−6.7%** | **−5.9%** | **−6.0%** |

**F5b is not taken at the natural fit** (§13: "marginal at the natural fit"; ≈ 0.3 misses per layer, so the handshakes
cost more than the CPU's share saves). Tune's benchmark repeats one short prompt's generation (prefix hits, the same
routing every repeat), which misstates the split at a high hit rate; at the capped fit tune and the rows agree.

## Capped fit (≈ 25%): the rows (acceptance 3)

Ratio of Holon to the reference (median of 3 rounds), F5a (`holon_25`) | F5b at tune's split (`holon_25_f5b`):

**Decode**

| Row (bar) | 1K | 4K | 32K |
|---|---|---|---|
| FreeToken (1.0×) | 1.03 \| **1.20** | **0.95** \| **1.04** | **0.99** \| **1.06** |
| llama.cpp `-fitt 7150` (1.5×) | 2.55 \| 2.95 | 2.29 \| 2.50 | 1.89 \| 2.03 |
| llama.cpp `--n-cpu-moe 36` (1.5×) | 2.51 \| 2.90 | 2.24 \| 2.44 | 1.85 \| 1.99 |
| llama.cpp cache, mmap (1.1×) | 2.17 \| 2.51 | 2.20 \| 2.40 | 1.75 \| 1.88 |
| llama.cpp cache, pinned (1.1×) | 1.18 \| 1.37 | 1.19 \| 1.30 | **1.06** \| **1.14** |

**Prefill** (F5b leaves it unchanged; F5a | F5b within 0.6%)

| Row (bar) | 1K | 4K | 32K |
|---|---|---|---|
| FreeToken (1.0×) | 1.20 | **0.92 miss** | **0.87 miss** |
| llama.cpp `-fitt 7150` / `--n-cpu-moe 36` (1.0×) | 4.29 / 4.47 | 14.32 / 14.61 | 6.84 / 7.03 |
| llama.cpp cache, mmap / pinned (1.0×) | 4.23 / 2.11 | 14.25 / 6.94 | 6.88 / 3.61 |

**Absolute** (tok/s, decode / prefill):

| Arm | 1K | 4K | 32K |
|---|---|---|---|
| Holon F5a (`holon_25`) | 119.2 / 3,198 | 104.0 / 9,672 | 75.4 / 4,700 |
| Holon F5b (`holon_25_f5b`) | 137.7 / 3,189 | 113.3 / 9,676 | 80.9 / 4,700 |
| FreeToken | 115.2 / 2,669 | 109.2 / 10,510 | 76.2 / 5,416 |
| llama.cpp cache, pinned | 100.7 / 1,513 | 87.4 / 1,394 | 70.9 / 1,301 |
| llama.cpp cache, mmap | 55.0 / 753 | 47.3 / 679 | 43.0 / 683 |
| llama.cpp `-fitt 7150` | 46.7 / 743 | 45.4 / 676 | 39.9 / 688 |
| llama.cpp `--n-cpu-moe 36` | 47.4 / 713 | 46.3 / 662 | 40.7 / 668 |

**Verdicts:** with the lock tune picks (F5b), **every capped decode row passes**; F5a alone misses FreeToken at 4K /
32K (0.95× / 0.99×) and the pinned cache at 32K (1.06×), as §13's re-band warned (FreeToken ≈ 0.94× at 4K capped).
Prefill passes against every llama.cpp configuration and **misses FreeToken at 4K and 32K** (0.92× / 0.87×): the
shortfall the owner accepted at step 4 (the expert GEMMs, plan §10's MoE prefill item), the same at both fits.

## The F1 row (Holon static vs llama.cpp's static split, decode ≥ 0.95×)

F1's own fits: a host policy keeps a prefill pass's staging (it never lends, §14 step 3b), so the lock's 3,236 /
1,618 / 1,618 do not fit under `static` (refused at the first launch, D4); its fill is 3,093 / 1,614 / 1,479 (50.3%),
3,089 / 1,613 / 1,476 with F5b's buffers. The static splits run at their own choice (natural) and at the cap.

| F1 against (bar 0.95×) | 1K | 4K | 32K |
|---|---|---|---|
| natural, F5a: `--fit` / `--n-cpu-moe 16` | 2.30 / 2.13 | 1.40 / 1.27 | **0.88 / 0.82** |
| natural, F5b: `--fit` / `--n-cpu-moe 16` | 2.59 / 2.40 | 1.46 / 1.33 | 1.04 / 0.97 |
| capped, F5a: `-fitt 7150` / `--n-cpu-moe 36` | 1.79 / 1.76 | 1.22 / 1.20 | 0.97 / **0.949** |
| capped, F5b: `-fitt 7150` / `--n-cpu-moe 36` | 2.20 / 2.17 | 1.58 / 1.55 | 1.17 / 1.15 |

With tune's split **the F1 row passes at every context and both fits**; F5a alone misses at 32K (a static map at
≈ 50% misses ≈ half the routed experts per token, all over PCIe; llama.cpp computes its CPU layers in RAM).
§13 predicted 0.95–1.0× with F5a at the natural fit; it holds at 1K / 4K only.

## The natural fit at the closing code (a re-check of step 4's verdicts)

Holon at the natural lock (3,236 / 1,618 / 1,618, 52.7%), the caches at the same share:

| Row (bar) | decode 1K | 4K | 32K | prefill 1K | 4K | 32K |
|---|---|---|---|---|---|---|
| FreeToken (1.0×) | 1.06 | 1.04 | 1.02 | 1.72 | **0.93** | **0.88** |
| llama.cpp `--fit` (1.5×) | 2.98 | 2.58 | 1.93 | 3.52 | 8.24 | 4.07 |
| llama.cpp `--n-cpu-moe 16` (1.5×) | 2.75 | 2.35 | 1.79 | 3.46 | 7.91 | 3.98 |
| llama.cpp cache, mmap (1.1×) | 1.33 | 1.56 | 1.32 | 3.99 | 9.64 | 4.55 |
| llama.cpp cache, pinned (1.1×) | 1.19 | 1.23 | **1.10** | 2.50 | 5.82 | 3.02 |

Every decode row still passes; the pinned cache at 32K sits on its bar (1.104×). Holon decodes 234.7 / 190.9 / 117.5
tok/s, 1.3–1.5% below step 4's 238.3 / 193.7 / 119.0 (at 3,220 slots), and 2.6–3.0% below at the capped fit (119.2
/ 104.0 / 75.4 against 122.9 / 106.8 / 77.0), while FreeToken and llama.cpp reproduce step 4 within 0.5%: about
0.07–0.28 ms more per decode token, growing with the misses. Bisected on an RTX A5000 afterwards (below): step 6's
CPU share phase in `moe_resolve@1`, run on every call under F5a too; fixed in `b45e86f`.

## The decode regression after step 4, bisected (RTX A5000)

Run after the close, on one RTX A5000 of the shared dev host. The question was which commit after step 4 made Holon's
offload decode slower, and whether the A5000 shows it.

**Method.**
- Arm: x8_close's `holon_25`, the capped arm (F11 `lfu`, F5a, `ring_copy` `dma`, the held-out profile,
  `moe.cache=*=0.25`, batch 1, graphs on, a 33,792-token context).
- Driver: `spikes/x4_gap/ctx.py`, one fresh process per arm, ABBA, 4 rounds (6 for the last table), contexts 1K and
  4K, 256 tokens. Figures are ms per decode token, means.
- Each commit runs from its own source tree (`git archive`), put first on `PYTHONPATH`. A `.dist-info` overlay carries
  that commit's own entry points; the venv's would name step 7's plugins, which older trees do not have. Each tree's
  kernels are prebuilt into `~/.cache/holon`.
- Kernel times: an nsys diff of the 4K request (graph nodes traced), then `resolve_mb.py` for variants of the kernel.

**Where.** Step 4 measured `63c6e4b`, before steps 5 and 6 merged (`1588c0e`), not `478aa33`. Deltas against
`63c6e4b`, from the same interleaved rounds:

| Commit | 1K | 4K |
|---|---|---|
| `478aa33` (step 4's merge; steps 5–6 in it) | +0.051 ms (+0.45%) | +0.062 ms (+0.45%) |
| `2d5ec61` (the close) | +0.054 ms (+0.48%) | +0.065 ms (+0.46%) |
| `1588c0e` (steps 5–6 merged into step 4) | +0.050 ms (+0.45%) | +0.059 ms (+0.42%) |
| `8c5a826` (step 4's follow-ups) | +0.063 ms (+0.56%) | +0.065 ms (+0.46%) |

- `2d5ec61` against `478aa33`: +0.003 / +0.002 ms (sd 0.012). Step 7 and close part 1 are clear: route_dump's copy is
  judge-only, and the padding mask runs only with a route transform.
- Per round, sd is 0.01–0.03 ms.
- The whole delta comes with steps 5 and 6.
- Placement is the same in every arm: 53.9 / 74.6 misses per token at 1K / 4K.

**Mechanism.**
- The nsys diff moves one kernel, `k_resolve<lfu>`: 7.44 → 8.97 µs per call. There are 48 calls per token, so
  that is +74 µs. `k_fetch` and every other kernel stay within 0.3%.
- Step 6 (`7596aae`) runs the CPU share's phase on every call, under F5a too. The phase is:
  - a scan of every expert's map entries (global reads);
  - thread 0's dependent read of `plan[n]`;
  - five barriers.
- A zero plan cannot use any of it: c = min(0, n) = 0.
- On this card the cost hardly grows with the misses: +0.065 ms at 54 misses per token, +0.068 at 75.

**Fix (`b45e86f`).**
- The kernel loads the plan at its start and reduces it with `__syncthreads_or` after the entries.
- It runs the share's phase only for a nonzero plan; with one, the path is as before.
- The contract (`std.moe.kernels`) is unchanged: a zero plan takes nothing, and `jobs[0, 0]` = 0 is still written.

Variants in `resolve_mb.py` (48 resolves per graph, min of 7, in µs per graph):

| Variant | µs per graph |
|---|---|
| `63c6e4b` | 214 |
| `2d5ec61` | 256 |
| plan read before the entries' barrier | 237 |
| the fix | 231 |
| no plan read at all | 224 |
| that, without the share's `best` bookkeeping | 223 |

- Skipping the phase without reading the plan saves 0.14 µs per call more. It would need a contract change (a
  table of no rows meaning no share), so it is not taken.
- The last ≈ 0.2 µs per call is the step-6 kernel's layout (larger shared tables, more arguments), not any one phase.

**Recovery.** nsys: `k_resolve<lfu>` at 7.85 µs. ABBA ×6, against `63c6e4b`:

| | 1K | 4K |
|---|---|---|
| `2d5ec61` | +0.065 ms (+0.58%) | +0.068 ms (+0.49%) |
| the fix | +0.016 ms (+0.14%) | +0.015 ms (+0.11%) |

Against `2d5ec61` the fix is −0.049 / −0.053 ms (sd 0.011), about 75% of the regression.

**Exactness.** The judge's snapshot trace, `spikes/x7_heldout/run.py`, at the fix and 25%:
- F11 at the defaults, F11 at the lock's settings, `lru` and F1 are each byte-identical to all-resident.
- F5b (`moe.cpu_plan=1,2,2,3,4,5,5,6`, 6 threads) is byte-identical to the same run at `2d5ec61`, so the share's path
  did not change.

**Tests.**
- `test_expert_io.py` and `test_placement_promote.py`: 169 passed.
- `test_moe_engine.py`, `test_tier_pagecache.py`, `test_transform_tail_skip.py` and `test_tune.py`: 49 passed.
- No new test. The regression costs time only: outputs were identical before the fix and are after it. The oracle
  tests already cover a zero plan and a share.

**Open.**
- The A5000 shows about a quarter of the RTX 5080's +0.25 ms at the capped 4K row. The fix is not measured on the
  5080, and no cluster desktop was used.
- Any of the 5080's gap beyond this kernel is not explained on this card.

## The offload locks

- **Natural fit, F5a (committed, `plans/…/Qwen3-30B-A3B-Q4_K_M@9f1a24700a339b09/rtx-5080/cu130-torch2.11/1.lock`):**
  `holon_tools tune` from the RTX 5090's lock (no lock of the class yet), the gate's settings (batch 1, 33,792
  context, `lfu`, `ring_copy` `dma`, `moe.placement.profile=profile_x4_gate.json`, the fill's 3,236 / 1,618 / 1,618).
  Decode op tuned: `attention.paged_decode` triton_attn 220.6 tok/s against FlashInfer 60.4. Verdict `approve`: L2
  against the mmq floor top-1 98.264% (≥ 97.969%), KL p99 5.38e-2 (≤ 0.118), ΔPPL vs floor −0.042% ± 0.091%
  (≤ 0.479%) — step 4's scratch lock's figures —; the snapshot trace ×3 run-identical (2,163 steps). Approved
  (delegated). `--host-memory` is not planned (a deployment gives it), and the profile is a local file beside the
  checkpoint (§5: profiles in model bundles stay open); under F5a outputs do not depend on it, only speed.
- **Capped fit, F5b: verdicts passing (below), the lock not written — escalated.** (1) No tool writes an F5b lock:
  `propagate`'s L2 scores the lock's own settings and the judge refuses F5b's floor at more than one token per step
  (`ValueError: F5b's floor needs one token per step, not max_step_tokens 4096`; `f5b-verdict.log`), so `tune
  --cpu-split` without `--no-lock` fails the same way; the bounded L2 runs by hand (below). (2) S5 ranks a lock
  carrying a CPU tier first on a host of that tier, so on an AVX-512 RTX 5080 this capped benchmark deployment (25% of
  the experts) would be served by default over the natural lock.

## Exactness of F5b on the RTX 5080 (§6)

- **L0, `kernels.ggml_cpu` on this desktop's CPU (`x86-avx512`):** `plugins/tests/test_ggml_cpu.py` 5 passed —
  synthetic Q8_0 / Q4_0 and the vehicle's Q4_K / Q6_K, Q4_K / Q4_K and UD-IQ3_XXS classes within the oracle's budget,
  a token bitwise alone or grouped at 1, 7 and 40 threads (step 6's "AVX-512 L0 on an AVX-512 host", closed).
- **Bounded L2** (F1 frozen at 25%, one token per step, tune's split; 32,783 positions, 1,575,360 handshakes, flag 0):
  **PASS** against the path-aware floor — top-1 98.649% (≥ 98.411%), KL p50 1.08e-4 (≤ 2.33e-4), p99 3.90e-2
  (≤ 8.05e-2), ΔPPL vs floor +0.152% ± 0.088% (≤ 0.469%). The floor against the golden: top-1 98.661%, KL p50 1.17e-4,
  p99 4.03e-2 (the plain mmvq floor 98.536%, 1.18e-4, 3.98e-2).
- **Run-identity** (the judge's snapshot trace ×3 at the natural lock's settings with `moe.cache=*=0.25`, F11's adaptive
  placement and the split on; the CPU share's pools present): **run-identical**, 2,163 steps.

## Found on the way

- **`40811a0` kernels.ggml_cpu: the AVX-512 module was never loadable after `holon prebuild`.** Prebuild built both
  tiers in one process, and torch names a module linked a second time with other arguments `<name>_v1`
  (`holon_ggml_cpu_v1.so`), which `extension.load` never finds: on this AVX-512 desktop F5b's first engine failed with
  `NotBuilt`. The A5000 (AVX2) never loads that tier, so step 6 did not see it. Prebuild now links each tier in its own
  process; a test that fails without the fix (`test_prebuild_links_each_tier_in_its_own_process`).
- **An RTX 3090 desktop lost its GPU in the middle of the 27B's verdict** (`unspecified launch failure`, then no device
  on the node); the class was re-run from the start on another desktop, all four verdicts there.

## Reproduce

```bash
M=<Qwen3-30B-A3B-Q4_K_M.gguf>; PIN=gate_up.q4_k.1536x2048=3236,down.q6_k.2048x768=1618,down.q4_k.2048x768=1618
python spikes/x4_gate/fit.py $M [--opts moe.placement=std.tiered.static@1 moe.placement.profile=profile_x4_gate.json]
python -m holon_tools tune $M --cpu-split --no-lock --batch 1 --tokens 128 --ops rmsnorm --max-seqs 1 \
    --max-model-len 33792 --host-memory 21474836480 --loop-core 0 --slots logits_processor=grammar.xgrammar \
    --plugin-options moe.store=std.moe.offload@1 moe.placement=placement.lfu@1 \
    moe.placement.profile=profile_x4_gate.json 'moe.cache=*=0.25' moe.ring_copy=dma --json tune_split.json
python spikes/x4_gap/ctx.py run --name close --arms spikes/x8_close/arms.json --rounds 3 --model $M \
    --contexts 1024 4096 32768 --ctx 33792 --out-dir <runs> --python <freetoken python> --server <llama-server>
python spikes/x6_cpu/l2.py $M cand.safetensors split.json "moe.store=std.moe.offload@1;moe.placement=std.tiered.static@1;\
moe.placement.profile=profile_x4_gate.json;moe.cache=*=0.25;moe.cpu_plan=1,1,2,2,3,3,4,4;moe.cpu_isa=x86-avx512;cpu_task.threads=4"
python -m holon_judge golden --model $M --config Qwen/Qwen3-30B-A3B --q8-1 mmvq --cpu-split split.json \
    --dtype bfloat16 --device cuda --prompts judge/prompts/v1.jsonl --out floor.safetensors
python -m holon_judge compare --golden judge/goldens/qwen3-30b-a3b-q4_k_m/v1-fp32.safetensors \
    --candidate cand.safetensors --floor floor.safetensors
python -m holon_tools tune $M --batch 1 --golden judge/goldens/qwen3-30b-a3b-q4_k_m/v1-fp32.safetensors \
    --floor judge/goldens/qwen3-30b-a3b-q4_k_m/v1-bfloat16-mmq-floor.safetensors --max-seqs 1 --max-model-len 33792 \
    --host-memory 21474836480 --slots logits_processor=grammar.xgrammar --plugin-options moe.store=std.moe.offload@1 \
    moe.placement=placement.lfu@1 moe.placement.profile=profile_x4_gate.json moe.cache=$PIN moe.ring_copy=dma --approve
python -m holon_tools propagate <model> --golden … --floor … --prompts … <the lock's [engine] as flags>  # each lock
python spikes/x4_gap/ctx.py run --name bisect --arms <arms: holon_25 per commit, wrap 'env PYTHONPATH=<tree>'> \
    --rounds 4 --model $M --contexts 1024 4096 --ctx 33792 --out-dir <runs>      # the bisect (A5000)
PYTHONPATH=<tree> python spikes/x8_close/resolve_mb.py <label>                    # the resolve alone, per tree
python spikes/x8_close/record.py --runs <dir> --date 2026-10-18 --commit 40811a0 --out evidence/x8_close_rtx-5080_2026-10-18.json
```
