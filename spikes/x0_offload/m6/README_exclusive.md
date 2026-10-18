# x5 / D10: M6's replay with an exclusive RAM tier — which placement is the RAM-short default

This is the input to step 5 of milestone X (`docs/v1-offload.md` D10 and §9). The record is
[`evidence/x5_exclusive_tier_replay_2026-10-17.json`](../../../evidence/x5_exclusive_tier_replay_2026-10-17.json).
The code is `exclusive.py`. It reuses `replay.py`'s trace loader, quotas, warm start and F11 pairing. `replay.py` is
unchanged, so M6's results reproduce as before.

## Question

Suppose RAM cannot hold every expert, so an expert lives in exactly one of VRAM, RAM or SSD. Which placement should
be the default, judged by least modelled time per token on M6's held-out traces? D10 asks for three variants:

- **(a) F11 with SSD-only victims.**
- **(b) F11 with deferred admission.**
- **(c) F10 with spares.**

**(d)**, `lfu` with RAM holding every expert, is the reference.

## Method

**Traces and splits.** The traces are M6's decode steps, with the same held-out splits and the same warm start.

| Split | Profile and warm start learned on | Scored on |
|---|---|---|
| Forward | chat + code | agent (13,861 steps, 23,721 tokens) and mixed (6,534 steps, 16,447 tokens) |
| Reverse | agent | chat (11,128), code (10,683) and mixed |

That gives 5 held-out (test, profile) pairs. The VRAM quotas and warm start are M6's: 4 slots per layer, then the
profile's global ranking.

**Grid.**
- **VRAM** holds 10%, 25% or 47% of the 6,144 experts.
- **RAM** holds 25%, 50% or 75% of the experts not in VRAM, seeded by the profile's next ranks.
- **SSD** holds the rest. The GGUF file is always valid, so leaving RAM for SSD costs no I/O.
- Each of the 9 cells is replayed on all 5 pairs.

**Variants** (`exclusive.py`'s docstring has the exact rules):

| Variant | Rule |
|---|---|
| (a) `f11_ssd` | M6's `lfu` admission in the graph. The victim becomes SSD-only. The admitted expert's RAM slot is released, and the host can reuse it 2 steps later (D4). A host policy promotes hot SSD-only experts back into RAM: free slots first, then over RAM's coldest expert when strictly hotter. It decides after commit c, and the promotion lands at c+4. Promotion reads are background bytes on the same 2-worker reader as demand reads (budget, stalls and waits are under "Time" below). |
| (b) `f11_deferred` | The same pairing, but the miss is only staged this step. Its victim is unmapped and demoted D2H into a free RAM slot. With no free slot, the victim takes the place of RAM's coldest expert if it is hotter, and becomes SSD-only otherwise. The staged copy goes D2D into the victim's slot, and the map flips at c+2. In that window the admitted miss is still read from its old tier; its RAM slot is released at the flip. |
| (c) `f10_spares` | M6's `promote_on_fetch` (Maya). There are 3 spares per layer. Misses land in the spares in route order. At each step boundary the layer's LRU residents are evicted, and each victim is demoted D2H if it is hotter than RAM's victim (a free slot counts as coldest), else it goes to disk only. |
| (d) `lfu_allram` | M6's `lfu`. It reproduces M6's hit rates and fetch bytes exactly: 70.6% and 311.0 MB per token on agent at 25%. |

**RAM heat.** For every variant, RAM's heat is F11's decayed count: ×0.915 per step, +1 per routed decode entry.

**When a demotion becomes readable.** A demotion is readable from RAM at the next step: D10 orders it before the
next H2D on the same copy stream. Until any copy lands, the expert's tier code says SSD.

**Sensitivities:**

| Suffix | Change |
|---|---|
| `_nopromo`, `_freepromo` | (a) without promotion, and with free promotion (a bound). |
| `_flip1` | (b)'s map flips at c+1, so there is no window. |
| `_dlag2` | Demotions land at c+2, as if the host had picked their slots. |
| `_rlag2` | (c)'s refill is a host edit applied before launch c+2, as D4 specifies for F10. |
| `_keep` | A demand SSD read that stays out of VRAM is kept in RAM. The read has already landed in host memory, so this costs no I/O; Maya sends it back to disk. |
| `--ram-decay` | RAM heat decays by 0.99 or 1.0 per step instead of 0.915. |

**Time.** Time per token is a serial model, computed per step and divided by the decode tokens:

| Term | Model |
|---|---|
| GPU | (dense + every routed expert's bytes) / 426.6 GB/s. The rate makes a batch-1 token cost the RTX 5080's all-resident 4.5 ms (§13), whatever the placement. |
| H2D | RAM and SSD misses at 47.5 GB/s (M1's gather). |
| D2H | Demotions at 47.5 GB/s, added serially. A table below also shows them overlapped, since PCIe is full duplex. |
| SSD | Demand reads at 2.6 GB/s, which is **the throughput model**: M7's 2 workers saturate, and each expert is 3 file extents, so both workers are fed. The alternative **depth-1 floor** gives each layer with an SSD read at least one expert's depth-1 read time: 1.24 / 1.40 ms for 2.65 / 3.06 MB, interpolated from M7's 0.77 ms per 1.46 MB and 1.30 ms per 2.8 MB. The floor changes no ranking. |
| Promotions, (a) only | The reader drains them in the step's non-SSD time. The host issues them only up to what it can drain before landing. A demand read of a promoted expert completes its promotion. Bytes still unread at landing stall that step. A step with backlog pays 0.19 ms per layer with demand SSD reads: the residual of an in-flight extent on the first of 2 busy workers. |

**Expert sizes** come from the vehicle: 2,654,208 bytes (Q4_K down) and 3,059,712 bytes (Q6_K down in 24 layers).
A decode step also reads 822.5 MB of dense weights.

**Checks.**
- `--selfcheck` runs every variant on synthetic traces with per-step asserts: exclusivity; RAM slot conservation
  (resident + in flight + free + released = capacity); mapped + pending + free spares = quota per layer. It also
  checks that (d) equals M6's `lfu`.
- The same asserts passed on agent at 10/50, 25/25 and 47/75 for all 15 variants.
- One reviewer checked the logic. Its fixes are in, and the remaining minor points are listed under Limits.

## Results

### Mean over the 5 held-out pairs, ms per token

Bold is the least of (a)–(c). The floor column uses the depth-1 SSD floor.

| VRAM | RAM of rest | (a) f11_ssd | (b) f11_deferred | (c) f10_spares | (d) lfu_allram | floor (a) / (b) / (c) | (b) at c+1 |
|---|---|---|---|---|---|---|---|
| 10% | 25% | 158.48 | **132.78** | 141.98 | 16.05 | 159.74 / 134.25 / 143.53 | 127.06 |
| 10% | 50% | 111.77 | **69.11** | 77.74 | 16.05 | 113.34 / 70.63 / 79.08 | 65.84 |
| 10% | 75% | 89.43 | **35.44** | 43.01 | 16.05 | 91.32 / 36.64 / 43.98 | 33.60 |
| 25% | 25% | 101.44 | 82.80 | **82.08** | 10.10 | 103.13 / 84.28 / 83.61 | 74.62 |
| 25% | 50% | 92.88 | **44.70** | 46.08 | 10.10 | 94.62 / 45.90 / 47.21 | 40.24 |
| 25% | 75% | 91.28 | **24.66** | 26.10 | 10.10 | 93.04 / 25.51 / 26.86 | 22.27 |
| 47% | 25% | 41.16 | 34.81 | **34.08** | 5.99 | 42.56 / 35.95 / 35.25 | 29.96 |
| 47% | 50% | 40.90 | 21.08 | **20.53** | 5.99 | 42.30 / 21.95 / 21.39 | 18.25 |
| 47% | 75% | 40.83 | 13.02 | **12.68** | 5.99 | 42.22 / 13.55 / 13.16 | 11.47 |

### Wins per cell, out of the 5 pairs

The ratio column is the slowest variant's time over the fastest's, as a range over the 5 pairs.

| VRAM / RAM | (a) | (b) | (c) | slowest / fastest | (b) at c+1 vs (c) | (c)'s time / (b at c+1)'s |
|---|---|---|---|---|---|---|
| 10% / 25% | 0 | 4 | 1 | 1.08–1.52 | 5 – 0 | 1.03–1.14 |
| 10% / 50% | 0 | 4 | 1 | 1.20–3.00 | 5 – 0 | 1.05–1.25 |
| 10% / 75% | 0 | 5 | 0 | 1.34–4.79 | 5 – 0 | 1.16–1.36 |
| 25% / 25% | 0 | 2 | 3 | 1.06–1.76 | 5 – 0 | 1.08–1.11 |
| 25% / 50% | 0 | 2 | 3 | 1.58–3.28 | 5 – 0 | 1.08–1.19 |
| 25% / 75% | 0 | 4 | 1 | 2.93–4.73 | 5 – 0 | 1.10–1.23 |
| 47% / 25% | 0 | 0 | 5 | 1.11–1.38 | 5 – 0 | 1.12–1.15 |
| 47% / 50% | 0 | 1 | 4 | 1.86–2.27 | 5 – 0 | 1.09–1.13 |
| 47% / 75% | 0 | 2 | 3 | 2.59–3.49 | 5 – 0 | 1.07–1.12 |

The depth-1 floor moves one pair-cell, at 25% / 75%, from (b) to (c).

### Sensitivities: mean over the 5 pairs, ms per token

**(a)'s promotion.** Even free promotion loses to (b) in every cell.

| VRAM / RAM | f11_ssd | no promotion | free promotion | keep |
|---|---|---|---|---|
| 10% / 50% | 111.77 | 238.23 | 74.61 | 72.19 |
| 25% / 50% | 92.88 | 125.42 | 55.33 | 70.29 |
| 47% / 50% | 40.90 | 46.52 | 24.49 | 35.42 |

**Timing of (b) and (c).** Here `rlag2` is F10 as D4 would apply it.

| VRAM / RAM | (b) | (b) c+1 | (b) dlag2 | (b) c+1 dlag2 | (c) | (c) dlag2 | (c) rlag2 |
|---|---|---|---|---|---|---|---|
| 10% / 25% | 132.78 | **127.06** | 139.11 | 134.07 | 141.98 | 162.34 | 138.69 |
| 10% / 50% | 69.11 | **65.84** | 75.44 | 72.85 | 77.74 | 98.11 | 75.40 |
| 10% / 75% | 35.44 | **33.60** | 41.79 | 40.62 | 43.01 | 63.24 | 39.46 |
| 25% / 25% | 82.80 | **74.62** | 89.11 | 81.35 | 82.08 | 90.85 | 83.11 |
| 25% / 50% | 44.70 | **40.24** | 51.00 | 46.97 | 46.08 | 54.99 | 46.14 |
| 25% / 75% | 24.66 | **22.27** | 30.97 | 29.00 | 26.10 | 34.91 | 24.85 |
| 47% / 25% | 34.81 | **29.96** | 37.39 | 32.62 | 34.08 | 37.21 | 34.79 |
| 47% / 50% | 21.08 | **18.25** | 23.67 | 20.91 | 20.53 | 23.81 | 20.52 |
| 47% / 75% | 13.02 | **11.47** | 15.62 | 14.14 | 12.68 | 15.93 | 12.18 |

**Keeping demand SSD reads in RAM, and overlapping D2H.**

| VRAM / RAM | (b) keep | (b) c+1 keep | (c) keep | (b) D2H overlapped | (c) D2H overlapped | (b) c+1 D2H overlapped |
|---|---|---|---|---|---|---|
| 10% / 25% | 112.63 | **109.34** | 123.54 | 130.71 | 136.70 | 124.84 |
| 10% / 50% | 49.58 | **47.92** | 58.85 | 67.04 | 72.32 | 63.61 |
| 10% / 75% | 25.55 | **24.55** | 32.06 | 33.37 | 37.59 | 31.37 |
| 25% / 25% | 77.20 | **70.57** | 73.26 | 79.73 | 78.45 | 71.45 |
| 25% / 50% | 38.53 | **35.35** | 37.78 | 41.62 | 42.26 | 37.07 |
| 25% / 75% | 20.73 | **19.03** | 21.36 | 21.58 | 22.28 | 19.10 |
| 47% / 25% | 33.92 | **29.34** | 31.94 | 33.07 | 32.30 | 28.20 |
| 47% / 50% | 20.10 | **17.63** | 18.77 | 19.32 | 18.59 | 16.48 |
| 47% / 75% | 12.22 | **10.90** | 11.74 | 11.27 | 10.75 | 9.70 |

**RAM heat decay.** With 0.99 or 1.0 per step instead of F11's 0.915:
- (b) and (c) change by −3% to +21%. F11's short memory is best or within 3.5% everywhere.
- (b) at c+1 stays the least in every cell at both decays.
- (b) against (c) at c+2 stays a near tie at 25–47% VRAM.

### Bytes per decode token by path and time parts: held-out agent (profile chat+code)

Bytes are MB per token. Times are ms per token.

| VRAM / RAM | Variant | Hit % | VRAM hit | RAM fetch | SSD read | Demotion D2H | Promotion SSD | Admits per token | H2D | D2H | SSD | Promo stall + wait | Total ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10% / 50% | (a) | 43.0 | 446.5 | 414.1 | 189.0 | 0 | 72.0 | 33.7 | 12.70 | 0 | 72.68 | 4.57 | 93.53 |
| 10% / 50% | (b) | 39.4 | 408.7 | 501.8 | 139.1 | 87.5 | 0 | 30.7 | 13.49 | 1.84 | 53.51 | 0 | 72.44 |
| 10% / 50% | (c) | 33.4 | 345.2 | 555.3 | 149.1 | 233.0 | 0 | 81.5 | 14.83 | 4.90 | 57.35 | 0 | 80.67 |
| 10% / 50% | (d) | 43.0 | 446.5 | 603.1 | 0 | 0 | 0 | 33.7 | 12.70 | 0 | 0 | 0 | 16.28 |
| 25% / 25% | (a) | 70.6 | 738.7 | 75.4 | 235.5 | 0 | 40.3 | 54.6 | 6.55 | 0 | 90.59 | 4.85 | 105.58 |
| 25% / 25% | (b) | 66.2 | 691.8 | 172.4 | 185.5 | 152.1 | 0 | 53.0 | 7.53 | 3.20 | 71.33 | 0 | 85.65 |
| 25% / 25% | (c) | 64.9 | 678.5 | 197.6 | 173.5 | 187.4 | 0 | 68.9 | 7.81 | 3.94 | 66.74 | 0 | 82.09 |
| 25% / 50% | (a) | 70.6 | 738.7 | 76.1 | 234.9 | 0 | 40.3 | 54.6 | 6.55 | 0 | 90.34 | 4.85 | 105.32 |
| 25% / 50% | (a) no promotion | 70.6 | 738.7 | 0.9 | 310.1 | 0 | 0 | 54.6 | 6.55 | 0 | 119.3 | 0 | 129.41 |
| 25% / 50% | (b) | 66.2 | 691.8 | 276.6 | 81.3 | 152.2 | 0 | 53.0 | 7.53 | 3.20 | 31.26 | 0 | 45.58 |
| 25% / 50% | (b) c+1 | 70.6 | 738.7 | 241.0 | 70.0 | 156.7 | 0 | 54.6 | 6.55 | 3.30 | 26.93 | 0 | 40.36 |
| 25% / 50% | (c) | 64.9 | 678.5 | 296.9 | 74.3 | 197.3 | 0 | 68.9 | 7.81 | 4.15 | 28.56 | 0 | 44.11 |
| 25% / 50% | (d) | 70.6 | 738.7 | 311.0 | 0 | 0 | 0 | 54.6 | 6.55 | 0 | 0 | 0 | 10.13 |
| 25% / 75% | (b) | 66.2 | 691.8 | 329.4 | 28.4 | 152.2 | 0 | 53.0 | 7.53 | 3.20 | 10.94 | 0 | 25.26 |
| 25% / 75% | (c) | 64.9 | 678.5 | 347.2 | 23.9 | 197.4 | 0 | 68.9 | 7.81 | 4.15 | 9.18 | 0 | 24.74 |
| 47% / 50% | (a) | 90.3 | 947.8 | 16.1 | 85.8 | 0 | 19.4 | 34.8 | 2.15 | 0 | 32.99 | 3.25 | 41.97 |
| 47% / 50% | (b) | 88.1 | 923.9 | 95.8 | 29.9 | 99.6 | 0 | 34.7 | 2.65 | 2.10 | 11.51 | 0 | 19.84 |
| 47% / 50% | (c) | 88.0 | 922.6 | 100.2 | 26.8 | 105.6 | 0 | 36.8 | 2.68 | 2.22 | 10.32 | 0 | 18.81 |

GPU time is 3.59 ms per token in every row: batch 2 on 71% of agent's steps. Dense is 480.6 MB per token.

**Why (a) loses.** About 55 admissions per token free RAM slots. The 2.6 GB/s reader refills only about 40 MB per
token of them in its idle time, so RAM drains: (a)'s RAM fetch is about 76 MB per token at every RAM fraction. The
demoted victims are re-read from SSD.

**Why (b) and (c) trade places.**
- (c) admits more (69 against 53 per token at 25%) and so demotes more (187–197 against 152 MB per token).
- At c+2, (b)'s window costs it 4.4 points of hit rate (66.2 against 70.6%). The admitted expert is re-read from
  RAM or SSD for one step.
- At c+1, (b) keeps F11's hit rate and the fewer demotions, and wins everywhere.

## Recommendation

**The RAM-short default is F11 with deferred admission, variant (b).** It flips the map at the first step at which
the demotion D2H and the D2D copy are stream-ordered before the launch. That is c+1 when both run on the compute
stream or are event-joined before launch c+1, and c+2 otherwise.

**Where each variant wins, at c+2 as D10 specified:**
- **(b)** has the least mean time in 5 of 9 cells, winning 24 of 45 pair-cells:
  - at 10% VRAM, for every RAM fraction, by 7–21%;
  - at 25% VRAM with RAM ≥ 50% of the rest, by 3–6%.
- **(c)** has the least in the other 4 cells, by 0.9–2.7%: 25% / 25%, and 47% VRAM at every RAM fraction. It wins
  15 of the 20 pair-cells there.
- **(a)** wins nowhere. It is 1.2–3.7× slower than the best on the mean, and the free-promotion bound still loses
  to (b) in every cell.

**Where (b) wins at c+1:** every one of the 45 pair-cells, with (c) 3–36% slower; under the depth-1 floor it also
wins all 45. It has the least mean time in every cell:
- with D2H overlapped;
- with demotions landing at c+2 (`dlag2`, compared with (c) `dlag2`);
- against F10 refilled as D4 specifies (`rlag2`);
- with RAM heat at 0.99 or 1.0.

With the `keep` rule added to both, it wins all 45 pair-cells, though at 25% / 25% its lead over (c) shrinks to
0–13%.

**One policy.** (b) keeps F11's counts, pairing, tie-breaks and oracle. Only the landing changes: staged now,
mapped at the fixed flip, with the victim demoted. So the RAM-full default (F11, immediate) and the RAM-short
default are one policy plugin with two landings, and F10 does not need to ship as a second default.

**Add the `keep` rule.** A demand SSD read that stays out of VRAM should be kept in RAM, in a free slot or over
RAM's coldest expert when hotter. It costs no I/O. On (b) at c+1 it cuts 2–27% of the time, the most at 10% VRAM
(10% / 50%: 65.8 → 47.9 ms).

**The cost of being RAM-short.** On these desktops' SSDs, the RAM-short default costs 1.9–7.9× the time of RAM
holding every expert at c+1, and 2.2–8.3× at c+2. For example, at the capped fit with RAM for half the rest it is
40.2 against 10.1 ms. SSD reads are about half to three quarters of the time in the agent rows above. RAM-short is therefore a fit-anything mode, not a performance mode, as M7 said for
F9. Step 5's RAM-capped run confirms it (D10).

## Contradictions with the design (for the lead)

- **D10's churn estimate is low.**
  - D10 says (a) re-reads ≈ 157 MB per token from SSD at 25% cached.
  - The replay finds 235 MB per token on agent with background promotion, at every RAM fraction, because RAM
    drains. Without promotion it is 310 MB, and with free promotion still 120 MB.
  - D10's conclusion stands, and more strongly: F11 with SSD-only victims is 2.1–3.7× slower than (b) at 25% VRAM
    once RAM holds half the rest or more.
- **D10's (b) wording, "its victim demoted D2H into the freed RAM slot."**
  - Under D10's own rule, the admitted expert's RAM slot is freed only after its map flips, plus the delta-log lag.
    The victim needs its slot at step c.
  - So the tier needs a pool of free slots, which the flips and releases refill. Otherwise the victim must displace
    RAM's coldest expert, or go SSD-only.
  - The model does the former and falls back to the latter. Step 5's design should say which.
- **"The map flips at a fixed later step, e.g. +2" costs 5–14%.**
  - The admission is in the graph with no host in the loop, so D4's c+2 for host edits need not apply.
  - The flip can be at c+1 if the demotion and the staged copy are ordered on the stream before the next launch.
- **Maya's "else disk-only."** Sending a demand SSD read straight back to disk wastes a read that is already in
  host memory. Keeping it (`keep`) saves (b) 3–28% at no cost, and neither D10 nor F10 has it.
- **The F10 that D4 implies is not the F10 that M6 and (c) replayed.**
  - D4 makes F10's spare refill a host edit applied at c+2. Maya and the replays refill at every boundary.
  - The D4 form (`rlag2`) is −8% to +2% of (c). It is better in 6 of 9 cells, because victims stay mapped for one
    more step and fewer free spares mean fewer promotions and demotions.
  - Against (b) at c+2, the D4 form is the least in 25 of 45 pair-cells but in only 3 of 9 cell means (47% VRAM,
    by ≤ 7%). (b) at c+1 beats it in every cell. The conclusion is unchanged.

## Limits and deviations

**The time model.**
- The time model is serial, like M6's. Nothing overlaps compute, and D2H is added rather than overlapped. A table
  above overlaps D2H, and the rankings are the same.
- KV reads and Holon's offload overhead are not modelled. The 4.5 ms GPU time is §13's estimate.

**Decode steps only.** As in M6, prefill rows never change residency.

**(a)'s promotion model is approximate.** These approximations all work against (a):
- RAM's coldest expert is evicted when a promotion is decided, not when it lands.
- The backlog wait applies to a whole step even if the backlog drains early in it.
- A promotion whose expert was admitted to VRAM in the meantime still stalls at landing.
- A promoted expert that demand reads in the promotion's window is read from SSD again on later window steps.
- They total ≤ 6 ms per token, and even the free-promotion bound loses to (b).

**(c)'s warm start.** The 144 experts emptied out of VRAM for (c)'s spares start SSD-only. This only affects the
opening steps.

**RAM placement.** A demotion with no free slot, and the `keep` rule, displace RAM's coldest expert only when
strictly hotter. A free slot always takes the expert. The RAM tier never refills empty slots except through (a)'s
promotions.

**SSD rate.** 2.6 GB/s is M7's desktop NVMe. On a Gen4 or Gen5 SSD, all three variants gain, and (a)'s reader bound
eases. Re-run with F9's measured rate.

## Reproduce

```bash
export TMPDIR=local/tmp/x5-replay
python spikes/x0_offload/m6/exclusive.py --selfcheck
for d in "" 0.99 1.0; do                                       # ~2.5 min each with 24 processes, CPU only
  python spikes/x0_offload/m6/exclusive.py --traces local/tmp/x0-m6 --jobs 24 ${d:+--ram-decay $d} \
      --out local/tmp/x5-replay/exclusive${d:+_rd$d}.json
done
python spikes/x0_offload/m6/exclusive.py --report local/tmp/x5-replay/exclusive.json \
    local/tmp/x5-replay/exclusive_rd0.99.json local/tmp/x5-replay/exclusive_rd1.0.json \
    --evidence local/tmp/x5-replay/evidence.json                 # prints the tables above
```

The traces are M6's captures (`local/tmp/x0-m6/*.npz`, not committed; see `README.md` to capture them). Wrap heavy
commands in a memory-capped scope on shared hosts.
