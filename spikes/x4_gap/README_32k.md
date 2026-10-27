# x4_gap at 32K: acceptance 3's decode rows predicted at 1K / 4K / 32K

Record: `evidence/x4_32k_rtx-5080_2026-10-17.json`. This answers the open item in `docs/dev/v1-offload.md` §13 ("32K
is replayed before the owner decides"). It applies x4_gap's 1K / 4K method (`README.md`, "What a dynamic policy
would fetch") to 32K. No code in core, std or plugins changed.

## Method

**Runs.** RTX 5080. Holon at `e866c10`, which has the same code as x4_gap's `b492595`: step 2's offload store,
static placement, F5a.
- `ctx.py` ran x4_gap's three Holon arms at 1,024 / 4,096 / 32,768 prompt tokens, 2 interleaved rounds:
  - static with the id-order seed at 47%;
  - static with the profile seed at 47%;
  - the profile seed at 25%.
- One FreeToken round per fit (`moe_collect_stats`) gave its own decode miss bytes.

**Reproduction.** The reruns' 1K / 4K rows match x4_gap: the same misses and replays, and decode
within 0.2%. The intercepts are 3.86 / 4.30 ms against x4_gap's 3.89 / 4.30. FreeToken's 32K decode
(111.1 / 76.3 tok/s) is within 0.4% of x4_refs'.

**Holon's decode time per token** = intercept + F11 + slope × the replayed `lfu` miss MB.
- The line comes from two measured placements of the 47% cache, as x4_gap fitted it.
  - At 32K it is **7.472 ms + 0.02464 ms per MB**, a 40.6 GB/s effective gather.
  - The intercept is everything except the misses, including the offload store's own ≈ 0.2 ms.
  - The 25% arm sits on the same line within 0.1% at every context (18.19 ms measured against 18.18 on the line
    at 32K), so the 47% line also serves the capped rows.
- F11's measured cost with candidates: 0.08 ms per token at 47%, 0.11 ms at 25% (§13, after step 3).
- The `lfu` miss bytes come from `replay.py` on the 32K routing trace, starting from the seed's actual residency.
  - At 47%: 60.8 MB per token (94.6% hits).
  - At 25%: 214.7 MB per token (80.7% hits).

**Holon's attention at 32K is measured, not derived.** The intercept rises 4.30 → 7.47 ms from 4K to 32K
(+3.18 ms). FreeToken's step rises +3.23 ms with its misses taken out (its miss bytes are nearly flat: 51.9 →
53.5 MB). So Holon's ≈ 0.3 ms attention lead at 4K holds at 32K but does not grow.

**References.** Ratios are against `x4_refs`' medians:

| Reference | Natural | Capped |
|---|---|---|
| FreeToken | 111.4 | 76.6 |
| llama.cpp cache, pinned | 102.0 | 70.5 |
| llama.cpp cache, default (mmap) | 81.7 | 42.5 |
| llama.cpp static split | `--fit` 61.5, tuned `--n-cpu-moe 16` 67.2 | `-fitt 7150` 39.9, `--n-cpu-moe 36` 40.6 |

**The band, and verdicts.**
- The band varies F11's cost by ± 0.04 ms and the miss bytes by ± 10% (replay against the engine).
- **pass**: the band's low edge is at or above the bar. **miss**: its high edge is below the bar. **straddles**:
  anything between.
- The 1K / 4K rows are x4_gap's record as is. They reproduce §13's re-band.

## Predicted ratios (Holon F5a + `lfu`, step 3 as built, before step 3c)

| Row (bar) | Fit | 1K | 4K | 32K | 32K verdict |
|---|---|---|---|---|---|
| vs FreeToken (1.0×) | natural | 0.97 [0.95–1.00] | 0.95 [0.92–0.98] | **0.99** [0.97–1.01] | straddles (central miss) |
| | capped | 1.05 [0.99–1.11] | 0.94 [0.89–1.00] | **1.02** [0.97–1.06] | straddles (central pass) |
| vs llama.cpp cache, pinned (1.1×) | natural | 1.10 [1.07–1.13] | 1.13 [1.09–1.17] | **1.08** [1.06–1.11] | straddles (central miss) |
| | capped | 1.21 [1.15–1.28] | 1.19 [1.12–1.26] | **1.10** [1.06–1.15] | straddles (on the bar) |
| vs llama.cpp cache, default mmap (1.1×) | natural | 1.30 | 1.52 | **1.35** [1.32–1.38] | pass |
| | capped | 2.22 | 2.20 | **1.83** [1.75–1.91] | pass |
| vs llama.cpp static `--fit` (1.5×) | natural | 2.65 | 2.18 | **1.80** [1.76–1.84] | pass |
| | capped | 2.61 | 2.26 | **1.95** [1.86–2.04] | pass |
| vs llama.cpp static, tuned `--n-cpu-moe` (1.5×) | natural | 2.35 | 2.00 | **1.64** [1.61–1.68] | pass |
| | capped | 2.56 | 2.27 | **1.91** [1.83–2.00] | pass |

Predicted Holon decode at 32K is 110.5 tok/s natural and 77.7 capped.

**At FreeToken's own miss bytes** (the same model with its counted bytes, i.e. if prompt-time admission closed the
warm-start gap), at 1K / 4K / 32K:

| Fit | Miss MB per token | vs FreeToken | vs llama.cpp pinned |
|---|---|---|---|
| natural | 23.7 / 51.9 / 53.5 | 1.03 / 1.01 / 1.01 | 1.17 / 1.20 / 1.11 |
| capped | 194.8 / 200.5 / 231.3 | 0.99 / 0.98 / 0.98 | 1.14 / 1.24 / 1.07 |

## What 32K changes

- **The FreeToken row is tightest at 4K, not 32K.**
  - At 32K natural, `lfu` replays 60.8 MB per token against FreeToken's 53.5: the warm-start gap shrinks to
    7 MB (15 MB at 4K). The fixed differences are also a smaller share of a 9 ms step. So the row rises to
    0.99×, on the bar.
  - At 32K capped, `lfu` fetches less than FreeToken (214.7 against 231.3 MB). The row is 1.02×.
- **The pinned-cache row is weakest at 32K.**
  - 1.08× natural and 1.10× capped. llama.cpp's fixed 0.94 ms of host sync is a smaller share of its 9.8 ms step.
  - x4_refs' pre-step-3 band was 0.98–1.02×. The 1.15–1.23× server-loop lead x4_gap found lifts it to 1.08×, still
    under 1.1×.
- **Step 3c's ≈ 0.25 ms** (glue, router GEMV, q8_1; x4_gap rows 5–6) moves the 32K rows by +2.8% natural and
  +1.9% capped: FreeToken 1.02× natural and 1.04× capped, pinned cache 1.11× and 1.12×. With 3c, both 32K cache
  rows are predicted on or just over their bars, with no margin. Without it, both natural-fit cache rows sit just
  under their bars at 32K.
- **The static-split and default-cache rows pass at 32K** with margin: ≥ 1.64× against a 1.5× bar, ≥ 1.35× against
  1.1×. x4_refs' "straddles at 32K" for the static rows used the pre-x4_gap kernel factor (k = 0.93), and no longer
  holds.

## Caveats

- **Replay, not run.** The `lfu` bytes come from the replay. The engine's own `lfu` (step 3) was not run at 32K
  here; step 4 measures it.
- **One prompt per context.** Each context is one prompt of 256 decode steps; miss bytes vary with the text (the 1K
  and 4K prompts differ 2×).
- **Short FreeToken counts.** FreeToken's miss bytes are one round, and use the mean expert size (2.857 MB).
- **Medians of 2 rounds.** Holon's spread is ≤ 0.1%.

## Reproduce

```bash
python spikes/x4_gap/ctx.py run --name x4_32k --arms <x4_gap's holon_static, holon_prof, holon_prof25> --rounds 2 \
    --model <gguf> --contexts 1024 4096 32768 --ctx 33792 --holon-python <holon venv python> --out-dir <out>
python spikes/x4_gap/ctx.py run --name x4_32k_ft --arms <ft_nat, ft_cap> --rounds 1 --model <gguf> \
    --contexts 1024 4096 32768 --ctx 33792 --python <freetoken venv python> --out-dir <out>
python spikes/x4_gap/replay.py <out>/x4_32k/r0_holon_{static,prof,prof25}.npz > <out>/x4_32k/replay.json
python spikes/x4_gap/predict_32k.py --runs <out> --date 2026-10-17 --commit e866c10
```
