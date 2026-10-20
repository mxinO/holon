# K-M5: CPU replay of the core index with a simulated tier (milestone K step 0, `docs/v1-kv-tiers.md` §8)

**Question.** With the step-1 core seams (KS1–KS5) driving a tier, what do the gate traces gain on the RTX 5080's
27B deployment, before any GPU run? The replay measures hit tokens per level, load bytes, demotions and drops, the
checkpoint staging peak, and TTFT and output throughput in simulated time. It also asks three sizing and cost
questions:
- What should the default `state_tier.host.bytes` be, and how large the staging pool?
- What does the core's own host time per step cost, with and without a tier?
- What does `evict_payloads`' scan cost without one (panel C13)?

Evidence: `evidence/k0_m5_replay_cpu_2026-10-18.json`. It holds the command, the base commit plus "uncommitted
step-1 seams", the predictions, the cost model, the rates, and the raw numbers per configuration.

## Predictions (fixed before the replay was written or run)

From the design's §8 and §7 (RTX 5080, Qwen3.8-27B UD-IQ3_S):

| Row | Prediction | Bar |
|---|---|---|
| 3c, VRAM-only | the 4-unit checkpoint pool keeps no returning session's payloads, so every hit trims to 0 and each turn recomputes its context: ≈ 9–10K tokens at resume, ≈ 4 s | — |
| 3c, tiered | ≈ 300 MB loaded per resume (≈ 6 ms) plus the append: ≈ 400 tokens recomputed with `answer`, ≈ 700 without → TTFT p50 ≈ 0.2–0.35 s; TTFT p50 ratio ≈ 0.05–0.1× | ≤ 0.25× |
| 3c, throughput | output throughput 1.3–2× from removed prefill | ≥ 1.2× |
| 3c, demotion (C11) | ≥ 90% of a returning session's KV resident in VRAM through turn 5, 35–50% after; ≈ 140K tokens parked at the end | — |
| 3e | `claude_code`, 2 turns per session, tiers on in both arms: sessions 1–2 recompute 16K (≈ 7 s); sessions 3–16 hit 16,384 with `branch` → first-turn TTFT ≈ 0.1–0.2 s; p50 ratio ≈ 0.03× | ≤ 0.2× |
| Staging | 5–6 units (`max_seqs + p`) with `dropped_publishes` 0 | 0 drops |
| Host time | the resident-only LRU keeps the device scan at today's size: the core's per-step host time with a tier ≈ without | — |
| `host.bytes` (from §7) | `agent_park` parks ≈ 4.9 GB of KV + ≈ 5.5 GB of payloads: 16 GiB holds all of it (the recommended default); 8 GiB drops some payloads and lower KV, so its 3c ratio is worse than 16 GiB's but still within the bar | — |

## Method

`replay.py` drives the **real** `holon.core` `PrefixIndex`, `StateManager` and `Scheduler`, unmodified: the
worktree's uncommitted KS1–KS5 over base 8c37948. The index is subclassed only to count demotions and drops and to
time `evict_payloads`. Each step follows `Engine.step` / `_launch` / `_commit` at **overlap depth 0**:
1. `poll`, then `schedule`, `step_ops` and `launching`;
2. the tier's `before_launch`, which advances the clock by the step's cost;
3. `launched` and `unpin`;
4. advance `num_computed`, then `finished_prefill`;
5. the commit: append each sampled token, `publish`, `finish`, then `register(…, done)`.

**Deployment** (the RTX 5080 27B lock, `10.lock`):
- `max_seqs` 4, `chunk_with_decode` 512, `max_step_tokens` 2048, page 16, `max_model_len` 49152, 64-token blocks.
- Paged q8_0 KV at 34,816 B per token.
- Two recurrent specs shaped like the GDN's conv (48 × 61,440 B) and ssm (48 × 3 MiB) states: one checkpoint is
  147 MiB (154 MB). Each spec has `in_step` 64 and its own checkpoint pool, as in std.
- VRAM KV is 57,376 tokens at 4 checkpoint units (the lock's `kv_tokens` in the bench evidence). Each further
  staging unit costs 4,421 tokens. Tiered runs use 5 units, so 52,944 tokens.
- Pools are CPU pools of a few bytes per unit: units are what count, and every byte figure comes from the unit
  sizes above.

**Checkpoint policies.** A toy kind mirrors `std.state.recurrent` (`hit_length`, `adopt`, in-step publishes, `cut`,
best-effort `publish`) and adds D8's policies:
- `prompt_end`;
- `branch`, at `SeqView.branch_at`, taken in-step during prefill;
- `answer:2`: a tentative payload at each 64-aligned position a decode row reaches, with `keep_tentative = 2`.

Policy names in the tables: `pe`, `pe+br`, `pe+br+a2`.

**Simulated tier** (`SimTier`, implementing `holon.api.StateTier` without bytes):
- A host level of `host.bytes` (one byte budget across unit shapes) and an optional SSD level.
- Page stores are write-through at insert (admission `publish`); payload stores at registration are mandatory.
- Each level evicts leaf-first by LRU. Pages are ordered by store, hit and load; payloads by store and adopt, in one
  recency order with pages.
- On SSD, copies the host level also holds are dropped first.
- Below the `host.reserve` watermark (one `max_model_len` load, 1.87 GB), host leaves are written back to SSD in the
  step hook. The tier's own drops are reported through `dropped()`.
- Every decision reads logical byte counts only.

**Clock and lanes.** The clock is advanced at each launch by `a + b·tokens + c·rows`. Loads are due over virtual
lanes: H2D at 47.5 GB/s, D2H at 45 GB/s (K-M2's blob rate), and one shared NVMe lane. NVMe uses K-M4's measured
rates: 2.17 GB/s for a 2-thread O_DIRECT read of 557,056 B units, and 1.77 GB/s for sustained writes past the SLC
cache. Waits:
- A launch that waits on a load starts once the load lands, so the clock jumps to its due time. That is a stall.
- Store fences never stall: at the locks' depth 1 the free rule waits on them one full step later. Their lateness
  at depth 0 is recorded.
- With no tier, a plain clock runs the same cost model. Idle time jumps to the next release.

**Traces and release rules.** `bench/traces/` holds `agent_park` (4 clients × 3 sessions), `claude_code`, the same
truncated to 2 turns per session (3e), and `agent`. Requests are released by `bench/driver.py`'s rules in simulated
time: arrival, `gap` after the turn a request continues, `after`, and `client` round-robin. Generated tokens are
deterministic per (salt, session, turn) and resent verbatim in the next prompt.

**Configurations.**
- VRAM-only (no tier) × 3 policies; `vram/pe` is today.
- Tier with `host.bytes` ∈ {8, 16, 32} GiB × 3 policies.
- Host + SSD: 4 or 8 GiB of host over a 100 GiB SSD (4 GiB is below acceptance 4's in-flight floor of ≈ 7.6 GB),
  and 4 GiB over an 8 GiB SSD, so the SSD fills and the tier drops.
- Staging units 4 / 5 / 6 / 8 at 16 GiB on `agent_park`.

**Metrics.**
- TTFT is the first token's time minus the release time.
- Throughput is output tokens / (last token − first release).
- Turn numbers are 1-based: "turns 1–5" are the first five requests of a session.
- Host time is the wall clock of the core calls per step, with the sim tier's own time subtracted.
- Every configuration also ran with `--check`: after every step, the D6 invariants (as `test_state_liveness.py`
  checks them) and the core/tier agreement.

### Cost model (derived in `cost_model()` from the lock's evidence)

| Point | Source | Seconds |
|---|---|---|
| a + b + c (batch-1 decode step) | tg128 63.75 tok/s (`bars_release-…-iq3_s_rtx-5080_2026-10-16`) | 15.69 ms |
| a + 4b + 4c (batch-4 decode step) | decode_b8 at batch 4, step wall p50 (`bench_release-…-iq3_s_rtx-5080_2026-10-16`) | 20.55 ms |
| a + 512b + c (a 512-token chunk) | pp512 @ d8192, 2,153.9 tok/s (same bars file) | 237.7 ms |

This gives a = 14.07 ms, b = 0.4345 ms per token and c = 1.187 ms per row. A 2048-token step costs 0.91 s
(2.26K tok/s), and a mixed step of 4 decode rows plus a 512-token chunk costs 245 ms. The depth-8K point is used
because the agent traces resume at 4–19K. At depth 0 the model overstates a chunk by 18% (pp512 is 201 ms).

## Results

### Against the predictions

| Row | Predicted | Measured | Verdict |
|---|---|---|---|
| 3c VRAM-only resume | ≈ 9–10K recomputed, ≈ 4 s | 9,623 tokens recomputed (p50); resume TTFT p50 5.76 s (4.2 s of prefill plus queueing behind the other clients) | as predicted (TTFT longer by queueing) |
| 3c tiered per resume | ≈ 300 MB + ≈ 400 tokens (`answer`), ≈ 700 (without) | 294 MB loaded p50; 437 tokens recomputed with `answer:2`, 818 without | as predicted |
| 3c TTFT p50 ratio (bar ≤ 0.25×) | 0.05–0.1×; TTFT 0.2–0.35 s | **0.046×** (0.255 s vs 5.52 s) with `pe+br+a2` at 16 GiB; 0.075× with `pe+br`; 0.085× with `pe` | bar met; slightly better than predicted |
| 3c throughput (bar ≥ 1.2×) | 1.3–2× | **2.41×** (135.3 vs 56.2 tok/s); 2.28× with `pe+br`; 2.20× with `pe` | bar met; above the predicted range |
| 3c turns 1–5 / 6+ | — | TTFT ratio 0.076× / 0.035× (0.26 s / 0.23 s vs 3.44 s / 6.71 s) | — |
| C11 residency | ≥ 90% through turn 5, 35–50% after | share of each resume's hit served from VRAM: p50 100% through turn 5, 39–40% after | as predicted |
| Parked at the end | ≈ 140K tokens | 136,832 tokens (2,138 blocks) when nothing is lost (`pe` at 16 GiB, any policy at 32 GiB) | as predicted |
| 3e first-turn ratio (bar ≤ 0.2×) | ≈ 0.03×; sessions 1–2 recompute, 3–16 hit | **0.015×** (0.121 s vs 8.13 s), at every host size. Sessions 1–3 recompute 16.5K (7.3–10 s); session 4 hits 16,384 but waits 6.7 s behind session 3's prefill (FCFS); sessions 5–16 take 0.07–0.17 s | bar met. One more session recomputes than predicted: session 2 arrives while session 1 is still prefilling the shared prompt, so its match has no children yet (`branch_at` = 0) |
| Staging | 5–6 units, 0 drops | peak 2–3 units at depth 0; estimated depth-1 demand (`pubs(s) + pubs(s+1)`) ≤ 3 on every trace; `dropped_publishes` 0 in every tiered run (4–8 units) | 0 drops; demand below the formula |
| Host time | with a tier ≈ without | core p50 25–31 µs per step either way; p99 0.40 vs 0.48 ms (`agent_park`), 1.95 vs 2.09 ms (`claude_code`) | as predicted (details below) |
| `host.bytes` | 16 GiB holds `agent_park`; 8 GiB worse but within the bar | 16 GiB = 32 GiB on every trace and policy; 8 GiB = 16 GiB without `answer`; with `answer:2`, 8 GiB keeps the 3c bar (0.056×) but p90 is 3.78 s (vs 0.45 s) | as predicted |

### `agent_park` (VRAM-only `vram/pe` is today)

| config | TTFT p50 / p90 (s) | turns 1–5 p50 | turns 6+ p50 | first turn p50 | resume recompute p50 (tok) | load per resume p50 (MB) | out tok/s | hit tokens VRAM / RAM / SSD (K) | VRAM demoted / dropped | tier drops pages / payloads | SSD writes GB (payloads) | load stalls (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vram/pe | 5.52 / 11.19 | 3.44 | 6.71 | 2.18 | 9623 | 0 | 56.2 | 92 / 0 / 0 | 0 / 9123 | – | – | – |
| ram8/pe | 0.47 / 1.21 | 0.50 | 0.44 | 2.18 | 818 | 270 | 120.8 | 821 / 537 / 0 | 10420 / 0 | 563 / 252 | – | 0.083 |
| ram16/pe | 0.47 / 1.05 | 0.50 | 0.44 | 2.18 | 818 | 279 | 123.5 | 830 / 552 / 0 | 10422 / 0 | 0 / 158 | – | 0.079 |
| ram32/pe | 0.47 / 1.05 | 0.50 | 0.44 | 2.18 | 818 | 279 | 123.5 | 830 / 552 / 0 | 10422 / 0 | 0 / 0 | – | 0.079 |
| vram/pe+br | 3.07 / 7.05 | 1.14 | 4.53 | 0.24 | 5807 | 0 | 75.7 | 690 / 0 / 0 | 0 / 9507 | – | – | – |
| ram8/pe+br | 0.41 / 0.79 | 0.38 | 0.42 | 0.26 | 818 | 250 | 125.3 | 882 / 517 / 0 | 10096 / 0 | 563 / 254 | – | 0.063 |
| ram16/pe+br | 0.41 / 0.76 | 0.38 | 0.42 | 0.26 | 818 | 270 | 128.3 | 889 / 533 / 0 | 10121 / 0 | 0 / 160 | – | 0.052 |
| ram32/pe+br | 0.41 / 0.76 | 0.38 | 0.42 | 0.26 | 818 | 270 | 128.3 | 889 / 533 / 0 | 10121 / 0 | 0 / 0 | – | 0.052 |
| vram/pe+br+a2 | 5.03 / 8.75 | 3.37 | 6.21 | 2.18 | 8797 | 0 | 58.3 | 174 / 0 / 0 | 0 / 9459 | – | – | – |
| ram8/pe+br+a2 | 0.31 / 3.78 | 0.27 | 0.33 | 0.26 | 503 | 154 | 113.4 | 881 / 411 / 0 | 10013 / 0 | 3186 / 814 | – | 0.057 |
| **ram16/pe+br+a2** | **0.26 / 0.45** | 0.26 | 0.23 | 0.26 | 437 | 294 | **135.3** | 907 / 570 / 0 | 10211 / 0 | 563 / 766 | – | 0.026 |
| ram32/pe+br+a2 | 0.26 / 0.45 | 0.26 | 0.23 | 0.26 | 437 | 294 | 135.3 | 907 / 570 / 0 | 10211 / 0 | 0 / 566 | – | 0.026 |
| ram4+ssd/pe+br+a2 | 0.40 / 0.64 | 0.30 | 0.43 | 0.26 | 437 | 305 | 134.3 | 899 / 0 / 577 | 10335 / 0 | 0 / 0 | 87.9 (83.4) | 0.328 |
| ram8+ssd/pe+br+a2 | 0.35 / 0.62 | 0.27 | 0.37 | 0.26 | 437 | 305 | 134.6 | 900 / 186 / 390 | 10319 / 0 | 0 / 0 | 72.7 (68.5) | 0.187 |
| ram4+ssd8/pe+br+a2 | 0.39 / 0.86 | 0.29 | 0.46 | 0.26 | 460 | 163 | 126.5 | 898 / 0 / 523 | 10153 / 0 | 1260 / 832 | 127.2 (103.5) | 1.537 |
| ram16/pe+br+a2/s4 | 0.26 / 0.45 | 0.26 | 0.23 | 0.26 | 437 | 221 | 135.3 | 941 / 536 / 0 | 9611 / 0 | 563 / 766 | – | 0.026 |
| ram16/pe+br+a2/s6 | 0.26 / 0.45 | 0.26 | 0.23 | 0.26 | 437 | 305 | 135.3 | 873 / 604 / 0 | 10811 / 0 | 563 / 766 | – | 0.029 |
| ram16/pe+br+a2/s8 | 0.26 / 0.48 | 0.26 | 0.24 | 0.26 | 458 | 337 | 135.3 | 765 / 710 / 0 | 12647 / 0 | 563 / 766 | – | 0.050 |

What the hit and load columns mean:
- **Hit tokens per level** count each request's first admission.
- **RAM loads** total 42.9–44.0 GB per run at 16 GiB.
- **D2H stores** total 29 GB with `pe` and 169 GB with `answer:2`: the checkpoints dominate (1,069 per spec, 2,138
  payload stores), ≈ 0.39 GB/s on average, under 1% of the link.
- **Host-level peaks** at 16 GiB with `answer:2`: pages ≤ 4.2 GB, payloads 16.0 GB. The level fills at every size,
  so its LRU always runs.
- **At 16 GiB vs 32 GiB:** 16 GiB parks 100,800 tokens at the end against 136,832 at 32 GiB. Finished sessions'
  pages go for payloads, with no effect on any request.
- **s8 preempts.** At 8 staging units VRAM KV falls to 39,680 tokens and 4 preemptions appear.

### Other traces (TTFT p50 / p90 s; first-turn p50 s; output tok/s)

| config | `claude_code` | `claude_code` 2 turns (3e) | `agent` |
|---|---|---|---|
| vram/pe | 0.33 / 8.67; 7.75; 30.1 | 7.63 / 13.30; **8.13**; 21.4 | 1.31 / 5.88; 1.42; 68.0 |
| vram/pe+br | 0.31 / 8.53; 7.77; 38.2 | 0.21 / 6.66; 0.11; 47.6 | 0.47 / 1.79; 0.47; 71.0 |
| vram/pe+br+a2 | 0.21 / 8.61; 7.75; 36.9 | 6.59 / 7.99; 7.46; 23.6 | 1.67 / 5.58; 0.96; 64.3 |
| ram8/pe | 0.24 / 0.84; 7.76; 55.6 | 7.34 / 13.30; 8.13; 21.3 | 0.33 / 1.05; 0.74; 72.5 |
| ram16/pe | 0.24 / 0.84; 7.76; 55.6 | 7.34 / 13.30; **8.13**; 21.3 | 0.33 / 1.05; 0.74; 72.5 |
| ram16/pe+br | 0.23 / 0.50; 0.11; 70.6 | 0.20 / 6.67; **0.121**; 47.6 | 0.26 / 0.66; 0.16; 72.5 |
| ram8/pe+br+a2 | 0.19 / 0.53; **7.76**; 63.2 | 0.16 / 6.69; 0.12; 48.1 | 0.20 / 0.59; 0.13; 72.8 |
| ram16/pe+br+a2 | 0.17 / 0.44; 0.12; 72.5 | 0.16 / 6.69; 0.12; 48.1 | 0.20 / 0.59; 0.13; 72.8 |
| ram32/pe+br+a2 | 0.17 / 0.44; 0.12; 72.5 | 0.16 / 6.69; 0.12; 48.1 | 0.20 / 0.59; 0.13; 72.8 |
| ram8+ssd/pe+br+a2 | 0.18 / 0.44; 0.17; 72.6 | 0.16 / 6.69; 0.12; 48.1 | 0.20 / 0.59; 0.13; 72.8 |

On `claude_code` and `agent` every tier hit is a VRAM hit: their KV fits in 53K tokens. The tier's whole gain there
is **payload retention**: the 4-unit pool loses the returning sessions' checkpoints, and the tier keeps them.
- **`claude_code`:** throughput is 2.41× today's (72.5 vs 30.1 tok/s) and TTFT p90 is 0.44 s vs 8.67 s.
- **`agent`:** TTFT p50 is 0.15× today's; throughput is arrival-bound (1.07×).
- **8 GiB with `answer:2` on `claude_code`:** the shared prompt's `branch` payload is evicted between session
  starts, and first-turn TTFT goes back to 7.76 s.

### Host time of the core (the wall clock of the scheduler and state calls per step; the sim tier's time subtracted)

| trace | today (vram/pe): p50 / p99 per step | tier (ram16/pe+br+a2): p50 / p99 | decode-only steps p99: today / tier | `evict_payloads` without a tier: calls, p50 / p99 per call, entries, share of core time |
|---|---|---|---|---|
| agent_park | 28 / 483 µs | 30 / 401 µs | 120 / 275 µs | 278; 92 / 140 µs; 886; 1.6% (with `answer:2`: 2,170 calls, 160 / 211 µs, 17.5%) |
| claude_code | 23 / 2,089 µs | 26 / 1,949 µs | 393 / 1,744 µs | 442; 125 / 185 µs; 875; 3.2% (with `answer:2`: 1,194 calls, 11.1%) |
| claude_code 2 turns | 23 / 1,213 µs | 25 / 1,698 µs | 356 / 378 µs | 56; 27 / 44 µs; 348 |
| agent | 29 / 529 µs | 26 / 365 µs | 246 / 299 µs | 244; 90 / 370 µs; 490; 7.7% (with `answer:2`: 18.4%) |

- **The resident LRU stays at today's size.** There are 826–895 resident entries, while lower entries (up to 1,312)
  are never scanned. A VRAM eviction checks the same number of candidates with and without a tier: p50 63 to 157
  per `evict` call, ≈ 20 µs per victim.
- **Per-step p50 does not move.** p99 is equal or lower with a tier, because the tier skips `evict_payloads` for
  staging pools.
- **Decode-step p99 rises with a tier**, and the cause is promotion at admission. A gated request is admitted in a
  step that carries only decode rows (without a tier its admission lands in its own prefill step). That admission
  costs 2–4 ms of host time for a 13–16K resume: a pure eviction of ≈ 150–200 victims plus `_take` for each entry.
  It happens once per resume and is below one decode step (20.5 ms), so overlap depth 1 hides it.
- **`claude_code`'s ≈ 2 ms p99 is today's code.** Admitting a 16K hit takes the 256 resident entries one `_take`
  (one `np.add.at` incref) at a time.
- **C13.** Without a tier, `evict_payloads` costs ≈ 0.1–0.16 ms once per publish with a full pool: 2–8% of core host
  time with `prompt_end`, 11–18% with `answer:2`. Outliers reach 10 ms (GC-like). With a tier it never runs.
- **The sim tier's own time** is p50 0.6 µs and p99 50–110 µs. It is a dict-and-LRU model, not std's tier.

## Recommendations

- **`state_tier.host.bytes` = 16 GiB** for the RTX 5080 27B agent locks with `prompt_end,branch,answer:2`.
  - 16 GiB = 32 GiB on all four traces.
  - 8 GiB suffices without `answer` but loses with it: `agent_park` p90 3.8 s, and `claude_code` loses its `branch`
    payload.
  - 16 GiB is above acceptance 4's in-flight floor (≈ 7.6 GB).
- **Staging (`recurrent.checkpoints` with a tier) = 5.**
  - The measured demand is ≤ 3 units, with an estimated depth-1 peak of 3 and 0 drops at 4–8 units, so 5 keeps
    the formula's `max_seqs + 1` with headroom.
  - TTFT is identical from 4 to 8 units. Each further unit costs 4,421 VRAM KV tokens, and 8 units cause
    preemptions.
  - Tune's 5 vs 6 can drop to 4 vs 5.
- **Enable `branch` in today's locks, tier or not.** Without a tier it already gives `agent_park` 0.56× TTFT and
  1.35× throughput, and 3e 0.013×; the bounded check is still owed.
- **Make `answer:<k>` require a tier.** With today's 4-unit pool it floods the pool: `claude_code` 2-turn first-turn
  TTFT is 7.46 s vs 0.11 s with `pe+br`, and `agent` TTFT p50 1.67 vs 0.47 s.
- **SSD write-back of payloads needs a policy.** With `answer:2`, an SSD below 4–8 GiB of host takes 73–88 GB per
  `agent_park` run (≈ 170–200 MB/s on average), 94% of it checkpoints, while only 24–42 GB are read back. Options:
  write back only payloads that were adopted, or skip superseded tentatives (see finding 2). The SSD arms' TTFT is
  0.35–0.40 s p50 vs 0.26 s from RAM.

## Findings in the core seams

1. **A tier's drops made outside `store` are acted on stale (bug; reproduced).**
   - **The gap.** The core polls `dropped()` only before `schedule` and after each `store` (`engine.py:482`; `prefix.py:189, 220`). An SSD
     level makes the tier drop copies elsewhere, whenever the SSD is full:
     - in its step hook: the write-back watermark;
     - inside `load`: an SSD read needs host room.

     The core then loads a copy the tier no longer holds.
   - **Three paths, reproduced:**
     - **`before_launch` → commit.** The watermark drops copies during the launch. The commit that follows runs
       `publish` → promote into place (`state.py:444-445` → `prefix.py:166`) and loads a dropped block.
     - **Promote into place, block by block.** `publish` issues one `load` per block (`state.py:445`). One block's
       SSD read evicts a later block of the same chain, and the next iteration loads it.
     - **Admission.** `_promote`'s page load (`state.py:265`) drops the hit's own payload, which the fixed point
       counted and `_adopt` then loads (`state.py:241`).
   - **Reproduce:** `python spikes/k0_tiers/m5/replay.py --only agent_park --tier 2 2` (also `--tier 3 1`). It
     raises `RuntimeError: the core loads (key, None), a copy the tier dropped (reported by dropped(), not yet
     polled)`.
   - **When it does not fire.** It never fires at 4–8 GiB of host over a 100 GiB SSD, nor on `claude_code` and
     `agent` with tiny levels. It needs a full SSD level, so it is latent at the planned sizes. Nothing in the
     contract excludes it.
   - **Fix options** (the lead's call):
     - poll `dropped()` after every `load` and at the commit's start;
     - in admission, take the payloads in the same `load` as the pages, so the contract's "never dropped before
       `loaded`" covers them, and re-run the fixed point if the hit shrank;
     - promote a commit's lower blocks with one `load`;
     - or restrict a tier to drops inside `store`, with `load` and hooks only demoting. A full SSD makes that
       impossible without refusing.
2. **`Payload.tentative` stays True after release (protocol gap, minor).**
   - At release, the request's `_tentative` list goes, but its kept payloads keep `tentative=True` on their entries.
     The tier saw `tentative=True` at store, and no call tells it the payload became ordinary.
   - Harmless to the core. It does mean a tier cannot use the flag for admission, for example to skip SSD write-back
     of tentatives (the recommendation above).
3. **Branch fires only when the shared chain has children at admission.** Sessions that start while the first one is
   still prefilling the shared prompt never see a branch (3e: 3 sessions recompute, not 2). This is benign, and
   expected from D8's definition.
4. **Store fences at depth 0.** The free rule makes the next launch wait on payload D2H stores: up to 4–10 ms late at
   depth 0 in this replay. At the locks' depth 1 this falls a full step later and costs nothing. Depth-0 steps (a
   `sync` step for prompt logprobs) would pay it.
5. **No waiterless or abandoned loads on these traces.** Only 8 occur, at staging 8, each with a zero catch-up. So
   the contract's "`loaded` advances the clock to its due" for unread loads never moved the clock here.

All 66 configurations pass `--check` after every step: D6's invariants and the core/tier agreement.

## Caveats (what the replay cannot see)

- **Overlap depth 0**, not the locks' depth 1. Staging demand at depth 1 is estimated from the depth-0 schedule.
  Store fences are not charged, because at depth 1 they land within the next step.
- **The cost model is linear**, with no attention-depth term. b comes from a chunk at depth 8K: it understates
  prefill at 16–19K (VRAM-only resumes, so today's TTFT may be worse) and overstates it near depth 0. Load copies do
  not slow compute (3b's question), and the copy path is assumed at link rate (K-M2: 0.99× on fragmented chunks).
- **SSD timing is optimistic under mixed I/O.** K-M4's mixed-I/O rule is STOP: measured 0.76–0.84× of the
  shared-lane model the replay uses. The SSD arms' load times can be up to ≈ 24% better than real when reads and
  write-backs overlap.
- **One byte budget across shapes.** std's host level is per-shape S2 pools, which need a page/payload split. The
  peaks reported (pages ≤ 4.8 GB, payloads the rest) inform it.
- **The host level is never told of pure-free VRAM demotions,** so it cannot drop "copies redundant with VRAM". The
  replay's tier does not try to.
- **No bytes.** Tier invariance and stream ordering are the judge's and the plants' job. The checks here are
  structural.
- **Verbatim resend** of generated tokens (`agent_park`'s assumption; K-M1 measures real agents). Closed-loop
  clients with fixed think times. No llama-server (3d), no expert-fetch contention (3i).
- **Host times** come from a server CPU running Python, not the bar desktops. Read them as comparisons between arms
  only. The sim tier is not std's tier, and its host time (p99 50–110 µs) says nothing about std's.

## Re-run on the step-1 seams as committed (rev 8.1)

The step-1 review's fixes changed what the core does around the tier: it polls `dropped()` at each commit and after
each load, re-runs the hit's fixed point after its loads, and settles a released request's tentatives
(`StateTier.settled`). The replay now polls at its commit as the engine does, and its simulated tier implements
`settled` and the rev 8.1 policy (a tentative payload is dropped at host eviction, never written back). Finding 1's
repros (`--tier 2 2`, `--tier 3 1`) run clean, and `--check` passes on all 63 configurations. Every number above is
unchanged except the SSD arms' write-back, and the full-SSD arm (ram4+ssd8: TTFT p50 0.389 → 0.396 s, load stalls
1.54 → 0.81 s):

| `agent_park` arm | SSD writes GB (payloads) before | after | settled at release |
|---|---|---|---|
| ram4+ssd | 87.9 (83.4) | 73.1 (68.7) | 590 |
| ram8+ssd | 72.7 (68.5) | 72.7 (68.5) | 620 |
| ram4+ssd8 | 127.2 (103.5) | 113.4 (89.6) | 596 |

The policy cuts payload write-back by only ≈ 17%: most kept tentatives become ordinary when their request is released,
before the host level evicts them, and are written back then. Cutting the rest needs another rule (write back only
payloads that were adopted, or none that a later payload of the same chain supersedes); that is the lead's call.

## Reproduce

```
CUDA_VISIBLE_DEVICES="" python spikes/k0_tiers/m5/replay.py --jobs 1 --out evidence/k0_m5_replay_cpu_2026-10-18.json
CUDA_VISIBLE_DEVICES="" python spikes/k0_tiers/m5/replay.py --check --jobs 16      # invariants after every step
CUDA_VISIBLE_DEVICES="" python spikes/k0_tiers/m5/replay.py --only agent_park --tier 2 2    # finding 1
```

The whole matrix replays in ≈ 70 s of CPU time. Results are deterministic; host times vary run to run.
