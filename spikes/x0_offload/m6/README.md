# x0 / M6: routing traces of Qwen3-30B-A3B Q4_K_M and an offline policy replay

Step 0 of milestone X (`docs/dev/v1-offload.md` §8, M6). Record:
[`evidence/x0_m6_routing_qwen3-30b-a3b_rtx-a5000_2026-10-17.json`](../../../evidence/x0_m6_routing_qwen3-30b-a3b_rtx-a5000_2026-10-17.json).

## Question

Which placement policy should be the default, and with which knobs, by §8's rule: the least modelled time per
token on a held-out trace? The run also gives D8's prefill inputs and checks D11's prefetch trigger (next-layer
recall ≥ 85%).

## Method

**Capture** (`capture.py`). Holon's engine served the development vehicle on one RTX A5000, eagerly (no CUDA
graphs, so Python wrappers see every MoE call). The wrappers exist only inside this process: `MoE._topk` and
`MoE.forward` per instance, and `Engine._launch` for the step's rows. No core, std or plugin code changed. For every
launched step and every token row, the capture records the row's sequence, whether it is a decode row, and per MoE
layer its top-8 ids (in routing order) and weights. On every 4th step with decode rows, it also applies layer l+1's
f32 router to layer l's MoE input and keeps that prediction's top 12 ids. Traces are stored as `.npz` with uint8 ids
under `local/tmp/x0-m6/` and are not committed.

| Trace | Workload | Engine | Sampling | Decode rows / steps | Prefill rows |
|---|---|---|---|---|---|
| agent | `bench/traces/claude_code.jsonl`: 16 sessions, 225 requests, 16K shared system prompt, two closed-loop clients | max_seqs 4, chunk_with_decode 0, prefix cache on | temperature 1.0 | 23,721 / 13,861 (batch 1–2) | 102,169 |
| mixed | `bench/traces/agent.jsonl` + `long_prompt.jsonl` | max_seqs 4, chunk_with_decode 512 | greedy | 16,447 / 6,534 (batch 1–4) | 78,517 |
| chat | the 6 chat prompts of `judge/prompts/v1.jsonl` × 2 seeds, ≤ 1024 tokens each | max_seqs 1 | T 0.6, top-p 0.95, top-k 20 | 11,128 | 390 |
| code | the 6 code prompts × 2 seeds | max_seqs 1 | same | 10,683 | 1,236 |

In total 244,291 routed rows, 61,979 of them decode. The agent and mixed prompts are bench's `--corpus` spans of
`judge/prompts/v1.jsonl`.

**Replay** (`replay.py`). The replay runs on the CPU with numpy. It covers decode steps only:
- **Unit of demand.** A step needs the distinct (layer, expert) pairs its decode rows route to, so a miss is paid
  once per step (D12).
- **Prefill rows never change residency.** D8 stages or streams them; their coverage is reported separately below.
- **Same starting point for every policy.** Each gets the same per-layer slot quotas and the same warm start from a
  profile learned on a *training* trace: 4 slots per layer first, then the profile's global (layer, expert) ranking
  (Strata X1, Maya G6).

The policies follow their references' code (citations from the research note §3.1 and §3.1b, read in the local
clones):

| Policy | Rule |
|---|---|
| `static` | the profile's residency, frozen (F1) |
| `profile_decay` | Strata X2 (`generate.cpp:8829-8987`, `expert_source.cpp:3717-3719`). Each decode token adds +1 per routed entry. After every 4th decode step a round runs: per layer, non-resident candidates with a count ≥ 2.0 (in descending order) pair with residents (in ascending order) while candidate ≥ victim + 1.5; the global top 96 by gain are swapped; then every count is multiplied by 0.7. A round is skipped while edits are pending. Edits land under D4: the victim is unmapped at c+2 and the new expert mapped at c+4. The swap copy counts as admission bytes. |
| `lru` | F4 (FreeToken, llama.cpp's cache). Each miss is fetched into its layer's least recently used resident that the step does not need. Timestamps are logical steps. |
| `promote_on_fetch` | Maya G1/G3 (`glm_fast.cu:3475-3483`, `glm_fast_path.cu:4628-4652`). Misses, in route order, land in the layer's 3 spare slots and become resident; any further misses go to scratch. Each step boundary refills the spares by evicting the layer's least recently used residents. |
| `lfu` | Decayed counts: each count is multiplied by 0.7^¼ ≈ 0.915 per step. A miss replaces the coldest resident the step does not need, if the miss's count is higher; otherwise it is only fetched. |
| `belady` | The offline optimum per layer, with bypass: after each step a layer keeps its slots' worth of the experts needed soonest. It is an upper bound for demand policies. |

Fetch-on-miss policies (`lru`, `promote_on_fetch`, `lfu`, `belady`) admit the fetched blob in place, so their
admission bytes are 0.

**Expert bytes** come from the file's tensor table:
- gate and up are Q4_K in every layer;
- down is Q6_K in 24 layers (0–5, every third from 8 to 41, 42–47) and Q4_K in the other 24;
- one expert is therefore 2,654,208 or 3,059,712 bytes (mean 2.857 MB, 17.55 GB for all 6,144);
- a decode step also reads 0.823 GB of dense weights (every non-expert tensor except the token embedding).

**Modelled time per token.** For each step:

    (dense + hit + fetch) / 900 GB/s + (fetch + admission) / PCIe

The per-step times are summed and divided by the number of decode tokens, with PCIe at 25 or 50 GB/s.
- **The rates are placeholders until M1/M3 land.**
- CPU bytes are 0, because the miss path is F5a only.
- SSD bytes are 0, because RAM holds every expert. The hook is there: `--ram-fraction < 1` would send the misses
  outside a profile-ranked RAM tier to an SSD rate.

**Held-out split.** Profiles and knobs are learned on chat + code. The scores come from agent and mixed (forward).
A second run learns the profile on agent and scores chat, code and mixed (reverse). Knobs were swept one at a time,
on chat's profile scored on code and the reverse, at 25% and 47% of the experts cached.

## Results (time in ms per token at the placeholder rates)

### Held-out agent, profile from chat + code: hit % / ms at PCIe 25 / ms at PCIe 50

| Cached | static | profile_decay | lru | promote_on_fetch | lfu | belady |
|---|---|---|---|---|---|---|
| 10% | 12.7 / 38.40 / 20.05 | 36.9 / 29.47 / 15.59 | 43.4 / **25.67** / **13.68** | 33.4 / 29.88 / 15.79 | 43.0 / 25.82 / 13.76 | 60.9 / 18.29 / 10.00 |
| 15% | 19.7 / 35.45 / 18.57 | 47.1 / 25.43 / 13.57 | 55.2 / **20.71** / **11.20** | 46.5 / 24.37 / 13.04 | 54.2 / 21.10 / 11.40 | 72.0 / 13.60 / 7.65 |
| 20% | 26.4 / 32.63 / 17.16 | 55.5 / 21.94 / 11.82 | 63.3 / **17.26** / **9.48** | 56.5 / 20.13 / 10.91 | 63.1 / 17.34 / 9.52 | 79.0 / 10.61 / 6.15 |
| 25% | 32.5 / 30.03 / 15.86 | 63.0 / 18.77 / 10.23 | 70.2 / 14.33 / 8.02 | 64.9 / 16.54 / 9.12 | 70.6 / **14.14** / **7.92** | 84.4 / 8.33 / 5.02 |
| 35% | 43.7 / 25.30 / 13.50 | 75.0 / 13.53 / 7.61 | 80.8 / 9.81 / 5.75 | 77.7 / 11.11 / 6.41 | 81.9 / **9.36** / **5.53** | 91.1 / 5.45 / 3.58 |
| 47% | 55.5 / 20.33 / 11.02 | 85.4 / 8.81 / 5.26 | 89.6 / 6.08 / 3.89 | 88.0 / 6.78 / 4.24 | 90.3 / **5.78** / **3.74** | 95.6 / 3.54 / 2.62 |
| 60% | 67.8 / 15.14 / 8.42 | 92.5 / 5.38 / 3.54 | 95.4 / 3.65 / 2.68 | 94.5 / 4.01 / 2.86 | 95.5 / **3.58** / **2.64** | 98.1 / 2.49 / 2.10 |

### Every held-out trace, ms per token at PCIe 25 (best non-Belady in bold)

| Test | Profile | Cached | static | profile_decay | lru | promote_on_fetch | lfu | belady |
|---|---|---|---|---|---|---|---|---|
| agent | chat+code | 10% | 38.40 | 29.47 | **25.67** | 29.88 | 25.82 | 18.29 |
| agent | chat+code | 25% | 30.03 | 18.77 | 14.33 | 16.54 | **14.14** | 8.33 |
| agent | chat+code | 47% | 20.33 | 8.81 | 6.08 | 6.78 | **5.78** | 3.54 |
| mixed | chat+code | 10% | 34.31 | 30.33 | 28.40 | 31.58 | **27.64** | 21.27 |
| mixed | chat+code | 25% | 26.05 | 20.50 | 16.99 | 20.15 | **16.82** | 10.15 |
| mixed | chat+code | 47% | 16.46 | 10.59 | 8.08 | 9.35 | **7.60** | 4.32 |
| chat | agent | 25% | 30.31 | 13.89 | 10.76 | 12.04 | **10.07** | 6.38 |
| chat | agent | 47% | 19.30 | 5.73 | 4.19 | 4.53 | **4.14** | 3.03 |
| code | agent | 25% | 32.97 | 15.93 | 11.77 | 13.12 | **11.20** | 7.05 |
| code | agent | 47% | 21.44 | 7.15 | 5.02 | 5.43 | **4.93** | 3.39 |
| mixed | agent | 25% | 26.60 | 20.50 | 17.00 | 20.15 | **16.83** | 10.16 |
| mixed | agent | 47% | 16.09 | 10.63 | 8.11 | 9.38 | **7.63** | 4.34 |

Over the 15 held-out cells (10/25/47% × 5 test/profile pairs) at both rates, `lfu` has the least time in 26 of 30
and `lru` in 4: 10% cached on agent and code, by ≤ 0.6%. `lru` is also best at 15–20% on agent, by ≤ 1.9%.
`lfu` and `lru` stay within 7% of each other in every cell. Against `lfu` at 25–47%:
- `profile_decay` is 22–52% slower;
- `promote_on_fetch` is 9–23% slower;
- `static` is 1.5–4.7× slower;
- Belady is 1.4–1.8× faster, which is the headroom left.

### Bytes per decode token by path (MB), held-out agent

| Cached | Policy | Hit | Fetch | Admission | CPU | SSD | Dense | ms @25, admission hidden |
|---|---|---|---|---|---|---|---|---|
| 25% | static | 341.5 | 708.2 | 0 | 0 | 0 | 480.6 | 30.03 |
| 25% | profile_decay | 658.7 | 391.0 | 35.7 | 0 | 0 | 480.6 | 17.34 |
| 25% | lru | 733.9 | 315.8 | 0 | 0 | 0 | 480.6 | 14.33 |
| 25% | promote_on_fetch | 678.5 | 371.1 | 0 | 0 | 0 | 480.6 | 16.54 |
| 25% | lfu | 738.7 | 311.0 | 0 | 0 | 0 | 480.6 | 14.14 |
| 25% | belady | 883.9 | 165.8 | 0 | 0 | 0 | 480.6 | 8.33 |
| 47% | profile_decay | 896.0 | 153.7 | 24.1 | 0 | 0 | 480.6 | 7.85 |
| 47% | lru | 940.2 | 109.4 | 0 | 0 | 0 | 480.6 | 6.08 |
| 47% | lfu | 947.8 | 101.9 | 0 | 0 | 0 | 480.6 | 5.78 |

Dense is 480.6 MB per token rather than 823 MB because the agent trace runs batch 2 on 71% of its decode steps.

**Why `profile_decay` loses: admission, not counts.** Checked on agent with chat + code's profile (ms @25):

| Variant | 25% | 47% |
|---|---|---|
| Defaults (lag c+2 / c+4, every 4, 96 swaps, gain 1.5, u ≥ 2) | 18.77 | 8.81 |
| No D4 lag | 17.71 | 8.18 |
| No lag, a round every step | 17.42 | 8.13 |
| No lag, every step, unlimited swaps, no gain or threshold | 26.07 (13.95 with admission hidden) | 10.33 (6.02 hidden) |

- **The lag costs ≈ 6%.** The rest of the gap is that a host swap copies the expert a second time.
- **Fetch-on-miss admits the blob the miss already fetched.** Strata could afford separate copies because its
  misses mostly go to the CPU, which leaves PCIe idle. On Holon's F5a-only default path, the copies compete with the
  fetches on the link.

### Knob sweep

Each knob was changed alone. The tuning pairs are chat's profile scored on code and the reverse, at 25% and 47%.
The table gives the mean ms per token @25.

| Policy | Knob | ms | vs defaults |
|---|---|---|---|
| profile_decay | defaults (0.7 / 4 / 96 / 1.5 / 2.0) | 10.62 | — |
| profile_decay | decay 0.5 / **0.9** | 10.73 / 10.24 | +1.0% / −3.6% |
| profile_decay | every 1 / 16 | 11.58 / 10.90 | +9.0% / +2.6% |
| profile_decay | swaps 32 / 384 | 10.74 / 10.59 | +1.1% / −0.3% |
| profile_decay | gain 0.5 / 4.0 | 10.75 / 11.08 | +1.2% / +4.3% |
| profile_decay | u_min 1 / 4 | 10.69 / 11.06 | +0.7% / +4.2% |
| promote_on_fetch | spares 1 / 8 | 8.57 / 10.24 | −1.6% / +17.4% |
| lfu | decay 0.915 (defaults) / 0.9 / 0.99 per step | 7.52 / 7.52 / 8.79 | — / −0.0% / +16.9% |

On the held-out traces, the tuned knobs move each policy by ≤ 4%:
- `profile_decay` at decay 0.9: −1.6 to −3%;
- `lfu` at 0.9: within ±1%;
- `promote_on_fetch` with 1 spare: **+3 to +7% worse** (it does not transfer, so 3 spares stay).

Retuning changes no ranking.

### Profile transfer

The table gives `static`'s hit % at 25 / 47% cached, by the trace the profile came from.

| Trace | Own | chat | code | agent | mixed | chat+code |
|---|---|---|---|---|---|---|
| agent | 50.4 / 75.8 | 33.4 / 58.1 | 27.6 / 51.9 | — | 41.2 / 68.0 | 32.5 / 55.5 |
| chat | 55.6 / 80.3 | — | 35.7 / 60.8 | 35.8 / 60.8 | 37.4 / 61.7 | — |
| code | 61.6 / 85.1 | 35.6 / 57.8 | — | 29.6 / 55.9 | 50.5 / 75.9 | — |
| mixed | 44.1 / 70.2 | 31.8 / 55.5 | 39.1 / 63.7 | 37.5 / 63.6 | — | 38.9 / 62.7 |

- **Profiles transfer poorly.** At 25% cached, a foreign profile loses 5–32 points against the trace's own.
- **Even the trace's own profile is far from adaptive.** Its 50.4% on agent at 25% stays below the adaptive
  policies' 63–71% with a foreign profile.
- **The warm start matters little.** Its trace-length effect is small, and in a long-running server it only decides
  the first few hundred steps.

### Distinct experts, skew

| Trace | Per decode step (rows: distinct per layer) | Window w = 1 / 2 / 3 / 4 tokens of one sequence |
|---|---|---|
| agent | 1: 8.0, 2: 15.2 | 8.0 / 12.5 / 16.2 / 19.3 |
| chat | 1: 8.0 | 8.0 / 12.5 / 16.1 / 19.2 |
| code | 1: 8.0 | 8.0 / 12.6 / 16.3 / 19.5 |
| mixed | 1: 8.0, 2: 15.0, 3: 22.0, 4: 28.1 | 8.0 / 12.6 / 16.4 / 19.7 |

- **Consecutive tokens of one sequence share ≈ 3.5 of their 8 experts per layer.** A 4-token verify window
  touches 19.3–19.7 distinct experts, not 32. GLM shares ≈ 1 of 8 (Maya EG2).
- **Rows of different sequences in one step share almost nothing:** 2 rows touch 15.0–15.2 experts and 4 rows
  touch 28.1.

The skew of decode routing per layer (mean [min, max] over the layers):

| Trace | Top-10% experts' share | Normalized entropy |
|---|---|---|
| agent | 0.276 [0.172, 0.331] | 0.938 [0.910, 0.978] |
| chat | 0.294 [0.172, 0.365] | 0.928 [0.891, 0.984] |
| code | 0.360 [0.199, 0.502] | 0.892 [0.804, 0.969] |
| mixed | 0.229 [0.162, 0.278] | 0.958 [0.930, 0.984] |

Routing is flat by frequency (the top 13 of 128 experts take 23–36% of the routes) and local in time. That is why
recency or decayed counts beat any frozen ranking.

### Next-layer recall (D11)

The predictor is layer l+1's router applied to layer l's MoE input, on decode rows.

| Trace | Rows | Recall top-8 | Per layer min / max | Recall top-12 | Precision top-12 | 25%: misses covered by top-12 | 25%: prefetched non-resident unused |
|---|---|---|---|---|---|---|---|
| agent | 5,928 | 85.4% | 54.5 / 90.5% | 95.3% | 63.5% | 94.7% | 39.6% |
| chat | 2,782 | 87.2% | 56.2 / 92.2% | 96.3% | 64.2% | 95.2% | 43.0% |
| code | 2,671 | 84.9% | 52.2 / 90.7% | 94.7% | 63.2% | 92.1% | 46.4% |
| mixed | 4,113 | 84.3% | 51.8 / 89.5% | 94.5% | 63.0% | 93.4% | 42.0% |

The minimum is always the 0 → 1 pair (52–56%). From layer 6 on, top-8 recall is 77–92% per pair, and 30–37 of
those 41 pairs reach 85%.

### Prefill coverage and D8

Each chunk is C consecutive prompt tokens of one request. All MB figures are per layer. The A5000's layer compute
(step / 48, measured on prompts of each size) stands in for the RTX 5080's.

| Chunk | Coverage (agent / mixed) | A5000 layer ms | Cached | All non-resident MB | Routed non-resident MB (agent) | D8 @25 GB/s | D8 @50 GB/s |
|---|---|---|---|---|---|---|---|
| 64 | 51% / 52% | 1.08 | 25% / 47% | 274 / 193 | 130 / 82 | staged | staged |
| 256 | 68% / 71% | 2.59 | 25% / 47% | 274 / 193 | 181 / 120 | staged | stream |
| 1024 | 83% / 85% | 6.22 | 25% / 47% | 274 / 193 | 224 / 154 | stream | stream |
| 4096 | 97% / 94% | 26.1 | 25% / 47% | 274 / 193 | 263 / 185 | stream (hidden under compute) | stream (hidden) |

The MoE share of the A5000 layer is 0.91 / 2.20 / 5.14 / 20.8 ms at 64 / 256 / 1024 / 4096 tokens.

## What this decides (§8's rules)

1. **Default placement policy: `lfu`**, the policy with the least modelled time on the held-out traces.
   - It wins 26 of 30 cells. `lru` wins the other 4 (10% cached, by ≤ 0.6%) and stays within 7% everywhere.
   - **Knobs:**
     - decay ×0.915 per decode step (0.7 per 4 steps); 0.9 is equal and 0.99 is 17% worse;
     - admit a miss in place when its count beats the coldest resident the step does not need;
     - warm start from a profile, with 4 slots per layer and then the global ranking;
     - counts: +1 per routed entry of decode rows.
   - **The default must admit on the fetch path** (a device-owned victim, F4's machinery). A host swap that copies
     a second time loses 22–52% on an F5a path.
   - **Other policies keep a role:**
     - `profile_decay` stays as a strategy, at decay 0.9;
     - `static` is the no-profile fallback only;
     - `promote_on_fetch` keeps 3 spares.
   - L4 on the development gate confirms the choice (§8).
2. **D8 / F7.**
   - **Routed-only staging below ≈ 256 tokens, streaming from ≈ 1K tokens**, at both the natural (47%) and the
     capped (25%) fit.
   - At 256 tokens the choice flips with the link: staging at 25 GB/s, stream at 50 GB/s.
   - From 1K tokens on, staging would move 80–96% of the stream's bytes anyway.
   - The crossover moves to larger chunks on cards that compute faster than the A5000. Re-derive it with M1's rate
     and the 5080's layer time.
3. **D11.**
   - **Recall trigger:** met at top-12 (94.5–96.3% everywhere) and borderline at top-8 (84.3–87.2%; the agent trace
     has 85.4%). D11 therefore waits on M1's idle-PCIe measurement.
   - **The replay's own model argues against a prefetch on the F5a path.** On agent with `lfu`, the link is busy
     88% / 79% of a token at 25% cached (25 / 50 GB/s) and 71% / 55% at 47%. With fetch fully overlapped with
     compute (idealized):
     - the prefetched non-resident ids include 40–46% that are never used;
     - at 25% cached: 20.2 vs 14.1 ms (25 GB/s) and 10.1 vs 7.9 ms (50 GB/s): a loss;
     - at 47% cached: 6.8 vs 5.8 ms (25 GB/s), a loss; 3.4 vs 3.7 ms (50 GB/s), a ≈ 9% gain.
   - **Recommendation: keep D11 (no prefetch in X).** If M1 shows ≥ 30% idle PCIe, re-check only the
     Gen5-at-natural-fit case.
4. **F8** (prefill CPU share): not decided here; it needs M3.
5. **Drafting (D14):** a 2–4-token verify window costs 12.5 / 16.2 / 19.3 distinct experts per layer (1.6 / 2.0 /
   2.4 tokens' worth). That is much better than GLM's ≈ 1 of 8 shared.

## Deviations

- **The agent trace.** The pi sessions of the RTX 5090 test were not kept locally (only their shapes are in the
  record). `agent` is therefore bench's Claude-Code-shaped trace on corpus text, the closest pi-style agent workload
  in the repo.
- **The mixed trace.** `mixed` is m07's mixed-step arm on `agent.jsonl` and `long_prompt.jsonl`
  (chunk_with_decode 512).
- **The engine ran eagerly with `cuda_graphs=False`.** Routing does not depend on graphs.
- **Greedy repeats.** The mixed trace's greedy streams repeat: 41% of long_prompt's output tokens and 9% of
  agent.jsonl's end a repeated 16-gram. That inflates mixed's hit rates for every policy alike.
- **Corpus overlap.** The training traces' prompts are judge prompts. The agent and mixed prompts are corpus spans
  of the same file, so the prefill text overlaps but decode does not. Profiles transfer poorly anyway, so the
  overlap does not favour `static`.
- **The time model is serial.** It does not overlap fetch with compute and does not cost in-graph LRU/LFU
  bookkeeping. LFU needs a decayed count per (layer, expert) on the device, which `lru` does not.
- **The capture GPU is an A5000.** The rates are placeholders, and the A5000 prefill times stand in for the RTX
  5080's.

## Contradictions with the design (for the lead)

- **D1 / acceptance 1.**
  - D1 makes `profile_decay` (F2–F3) the default and `lru` (F4) a comparison arm. §8's rule instead picks a
    device-side fetch-on-miss policy, and among those `lfu`.
  - Acceptance 1 planned LFU as the held-out "unnamed" plugin.
  - Two ways out:
    - ship `lfu` in X as the default and hold out another strategy (e.g. ARC);
    - default to `lru`, which is already in X and within 7% (it wins at 10% cached), and keep LFU held out.
- **D5 / F5a staging.** Misses land in staging slots and host edits copy experts in later. The replay shows the
  second copy is the cost: the default path should gather a miss into its victim slot (a device-owned region under
  D4's one-writer rule), as F4 and F10 already do.
- **D11's trigger is close to firing.** Recall is ≥ 85% at top-12 everywhere and at top-8 on two of four traces.
  Strata and Maya measured losses, and the replay's model agrees except at Gen5 with natural fit.
- **The design's ≈ 2.8 MB per expert is the mean.** Experts are really two sizes: 2.65 MB with Q4_K down and
  3.06 MB with Q6_K down, 24 layers each. Class arenas (D5) will see both.

## Reproduce

```bash
export TMPDIR=local/tmp/x0-m6
for w in chat code ptime mixed agent; do       # ~25 min on one A5000, one at a time
  CUDA_VISIBLE_DEVICES=<gpu> python spikes/x0_offload/m6/capture.py --model <Qwen3-30B-A3B-Q4_K_M.gguf> \
      --workload $w --out local/tmp/x0-m6/$w.npz
done
python spikes/x0_offload/m6/replay.py --selfcheck
python spikes/x0_offload/m6/replay.py --traces local/tmp/x0-m6 --gguf <gguf> --ptime local/tmp/x0-m6/ptime.json \
    --out local/tmp/x0-m6/replay.json                                          # ~9 min, CPU
python spikes/x0_offload/m6/replay.py --traces local/tmp/x0-m6 --gguf <gguf> --train agent \
    --test chat code mixed --out local/tmp/x0-m6/replay_rev.json
python spikes/x0_offload/m6/replay.py --report local/tmp/x0-m6/replay.json local/tmp/x0-m6/replay_rev.json
```

Wrap heavy commands in a memory-capped scope on shared hosts.
