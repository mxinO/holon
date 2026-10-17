# x4_gate — milestone X step 4: the development gate's natural-fit rows (RTX 5080)

`docs/v1-offload.md` §1 acceptance 3, D4, D9, D14, §13. Qwen3-30B-A3B Q4_K_M (`unsloth/Qwen3-30B-A3B-GGUF`) on one
RTX 5080 desktop, batch 1, greedy, no drafting, contexts ≤ 32K. Record: `evidence/x4_gate_rtx-5080_2026-10-17.json`
(every round, the lock's text and verdict, the tables below). Holon at `63c6e4b` (the v1-offload branch after step 3d,
plus the two fixes below); llama.cpp `f74e2fa6a31c` and FreeToken `21d9578bde22`, x4_refs' builds.

## Method

- **Driver:** `spikes/x4_gap/ctx.py` (`bench/offload_ctx.py`'s method with a Holon worker).
  - Every arm runs in a fresh process: an untimed 512 + 64 warm-up, then one request per context (1,024 / 4,096 /
    32,768 prompt tokens of natural text, the same ids for every engine), each generating 256 tokens.
  - Every engine allocates a 33,792-token context.
  - Arms are interleaved and reversed on odd rounds (ABBA); every arm ran on the same node on the same day.
  - The tables give medians: 3 rounds per gate arm, 4 per tuning arm. Spread (max − min over the median): Holon
    ≤ 0.1% decode, ≤ 3.1% prefill (the 1K prompt); the cache engines ≤ 0.6%; llama.cpp's static splits up to 4%
    decode (the record has min, max and sd).
- **Ratios:** the bar is acceptance 3's.
  - **pass** means the median ratio is ≥ the bar.
  - Predicted bands are §13's (`spikes/x4_gap/README_32k.md`), written at 47% before steps 3c and 3d. In brackets
    they are converted to the code measured here: 0.25 ms less per decode token (3c 0.21 ms, 3d 0.04 ms) at each
    row's predicted Holon rate (`record.py`, `BANDS`).
  - A single value means §13 gave a point, not a band.
  - §13 predicted prefill only as "≥ 1.0× llama.cpp". Against FreeToken it gave a risk, not a band.
- **Arms** (`arms.py` writes `arms_natural.json` and `arms_capped.json`):
  - Holon runs under the offload lock's settings and kernel pins.
  - The reference arms are x4_refs' arms, at the same expert share where they have a cache:
    - `--moe-cache-mib` 8,773 at 52.4% (7,900 at 47.1%, 4,200 at 25.1%);
    - FreeToken `--moe-cache-rate` 0.524 (its cache: 3,220 experts, the same count as Holon's), 0.47 and 0.25.
  - The static splits run at their own natural choice: `--fit` at its default margin, and `--n-cpu-moe 16`, x4_refs'
    tuned best that fits 32K. The capped static arms are `-fitt 7150` and `--n-cpu-moe 36`.

## The natural fit and the offload lock

**Natural fit** (`fit.py`). The fill was run at a 33,792-token paged floor (the default, `max_model_len`), batch 1,
`memory_fraction` 0.9.
- The store filled 3,236 / 1,618 / 1,618 cache slots (gate/up Q4_K, down Q6_K, down Q4_K): **52.7%** of the experts,
  above the ≈ 47% §1 estimated. Step 3b's smaller staging returned ≈ 0.47 GB to the cache.
- Pinning those counts fails at startup: `state.kv` gets 2,101 of the 2,113 units it needs.
  - The pinned layout's startup profile peaks ≈ 18 MB higher than the fill's: the fill profiles on minimum pools,
    where the cache cannot lend, so its streaming step stages instead.
  - Pinned with 4, 6 or 8 fewer experts, the engine starts.
- **The lock pins 8 experts less: 3,220 / 1,610 / 1,610, 52.4%** (9.20 of 17.55 GB, `state.kv` 2,142 pages).

**Lock** (`holon_tools tune <gguf> --plans <scratch copy of plans/> --batch 1 --golden … --floor
v1-bfloat16-mmq-floor …`):
- **Engine settings:**
  - `max_seqs` 1, `max_model_len` 33,792, `--slots logits_processor=grammar.xgrammar` (as the model's other locks);
  - `plugin_options`: `moe.store=std.moe.offload@1`, `moe.placement=placement.lfu@1`,
    `moe.placement.profile=profile_x4_gate.json`, `moe.cache=gate_up.q4_k.1536x2048=3220,down.q6_k.2048x768=1610,
    down.q4_k.2048x768=1610`, `moe.ring_copy=dma`;
  - `--host-memory` 20 GiB, which is not planned.
- **Start:** the model has locks on other classes, so `tune` refuses `--new-model`. It started from the RTX 5090's lock.
- **Decode op tuned:** only `attention.paged_decode` has two providers here: triton_attn 222.4 tok/s against
  FlashInfer 63.8.
- **Verdict `approve`:**
  - L2 against the mmq floor: top-1 98.264% (floor 98.219%, budget ≥ 97.969%); KL p99 5.38e-2 (≤ 0.118); ΔPPL vs
    floor −0.042% ± 0.091% (≤ 0.479%).
  - Snapshot trace run-identical.
- **Where it went:** written with `--approve` into a scratch copy of `plans/`, not into the repo. D15 commits the
  locks once, at X's close; its text and verdict log are in the record.

## Tuning: ring copy, warm start, lfu vs lru

Holon arms at the natural pin, 4 interleaved rounds each (`arms_tune.json`; tok/s, median).

| | 1K | 4K | 32K |
|---|---|---|---|
| prefill, `ring_copy=sm` | 3,298 | 8,909 | 4,592 |
| prefill, `ring_copy=dma` | **4,534 (+37.5%)** | **9,803 (+10.0%)** | **4,707 (+2.5%)** |
| decode, sm / dma | 228.9 / 229.0 | 188.1 / 188.1 | 116.5 / 116.5 |
| decode, no profile / held-out profile | 228.9 / **238.2 (+4.1%)** | 188.1 / **193.7 (+3.0%)** | 116.5 / **119.0 (+2.2%)** |
| decode miss MB per token, no profile / profile | 13.6 / 8.3 | 62.7 / 49.6 | 59.6 / 53.1 |
| decode, `lfu` / `lru` | 228.9 / 227.6 (−0.5%) | 188.1 / 188.2 (+0.04%, sd 0.10) | 116.5 / 116.2 (−0.3%) |

- **`moe.ring_copy = dma`.** It is faster at every prefill context by far more than 2 sd, and decode is unchanged. The
  lock takes it, and both rates are recorded.
- **The profile is kept** (F2's warm start):
  - It is `holon_tools profile` over a prompt set held out from the gate's corpus. `heldout.py` builds 32 chat
    prompts on excerpts of `docs/design.md`, `plan.md`, `v1-offload.md` and `release.md` at `4b6de6c`, tokenized
    with Qwen3's tokenizer (sha256 `a2d626ed…`; 128 greedy tokens each).
  - The output is `profile_x4_gate.json` (sha256 `9f2edfe2…`) beside the checkpoint.
  - It cuts decode misses by 11–39% and gains 2–4% decode.
- **`lfu` stays the default.** `lru` does not beat it by more than 2 sd at 4K (+0.04%, within the noise; §13's rule),
  and loses at 1K and 32K. `lru` fetches 5% more (65.8 against 62.7 MB per token at 4K) yet runs as fast there:
  its resolve is cheaper than F11's pairing.

## The gate: natural fit (52.4% — the lock)

Ratio of Holon to the reference (median). After the verdict, in brackets: §13's band converted to steps 3c + 3d.

**Decode**

| Row (bar) | 1K | 4K | 32K |
|---|---|---|---|
| FreeToken `auto` (1.0×) | 1.08 pass [1.00–1.05] | 1.06 pass [0.96–1.02] | 1.03 pass [1.00–1.04] |
| llama.cpp static split, `--fit` (1.5×) | 2.98 pass [2.79] | 2.53 pass [2.27] | 1.97 pass [1.81–1.89] |
| llama.cpp static split, tuned `--n-cpu-moe 16` (1.5×) | 2.71 pass [2.48] | 2.35 pass [2.09] | 1.79 pass [1.66–1.73] |
| llama.cpp `--moe-cache-mib`, default mmap (1.1×) | 1.36 pass [1.37] | 1.60 pass [1.59] | 1.34 pass [1.36–1.42] |
| llama.cpp `--moe-cache-mib`, pinned (1.1×) | 1.21 pass [1.13–1.19] | 1.25 pass [1.14–1.22] | 1.12 pass [1.09–1.14] |

**Prefill**

| Row (bar) | 1K | 4K | 32K |
|---|---|---|---|
| FreeToken `auto` (1.0×) | 1.71 pass | **0.92 miss** | **0.88 miss** |
| llama.cpp `--fit` (1.0×) | 3.52 pass | 8.25 pass | 4.07 pass |
| llama.cpp `--n-cpu-moe 16` (1.0×) | 3.48 pass | 7.92 pass | 3.99 pass |
| llama.cpp cache, mmap (1.0×) | 3.99 pass | 9.66 pass | 4.57 pass |
| llama.cpp cache, pinned (1.0×) | 2.44 pass | 5.71 pass | 2.96 pass |

**Absolute rates** (tok/s, decode / prefill):

| Engine | 1K | 4K | 32K |
|---|---|---|---|
| Holon | 238.3 / 4,638 | 193.7 / 9,846 | 119.0 / 4,716 |
| FreeToken | 220.6 / 2,712 | 183.0 / 10,670 | 115.1 / 5,392 |
| llama.cpp cache, pinned | 196.8 / 1,903 | 154.9 / 1,724 | 106.0 / 1,594 |

Holon's decode hit rate is 99.3% / 95.6% / 95.3%, missing 8.3 / 49.6 / 53.1 MB per token.

**At x4_refs' 47.1%** (the share §13's bands were written for). The references reproduce x4_refs within 0.6%
(FreeToken 212.8 / 174.6 / 111.1, pinned cache 188.6 / 147.0 / 101.9). Holon decodes 224.2 / 179.7 / 113.4 and
prefills 4,137 / 9,830 / 4,714.

| Row (bar) | decode 1K | 4K | 32K | prefill 1K | 4K | 32K |
|---|---|---|---|---|---|---|
| FreeToken (1.0×) | 1.05 [1.00–1.05] | 1.03 [0.96–1.02] | 1.02 [1.00–1.04] | 1.53 | **0.92** | **0.87** |
| `--fit` (1.5×) | 2.81 [2.79] | 2.35 [2.27] | 1.87 [1.81–1.89] | 3.14 | 8.23 | 4.07 |
| `--n-cpu-moe 16` (1.5×) | 2.55 [2.48] | 2.18 [2.09] | 1.70 [1.66–1.73] | 3.10 | 7.91 | 3.99 |
| cache, mmap (1.1×) | 1.42 [1.37] | 1.66 [1.59] | 1.39 [1.36–1.42] | 3.83 | 10.57 | 5.02 |
| cache, pinned (1.1×) | 1.19 [1.13–1.19] | 1.22 [1.14–1.22] | 1.11 [1.09–1.14] | 2.26 | 5.98 | 3.10 |

**Verdicts:**
- **Decode passes every natural-fit row at every context.**
  - The tightest rows are FreeToken at 32K (1.03×) and the pinned cache at 32K (1.12×).
  - The decode ratios land inside or above §13's converted bands. They are above them at 1K and 4K (the profile's
    warm start, which the bands did not model, is +3–4%), and inside at 32K. The exceptions are the default-mmap
    cache at 1K and 32K at 52.4% (1.36× and 1.34× against 1.37 and [1.36–1.42]): the bands were written for 47%, and
    llama.cpp's mmap cache gains more from the larger share than Holon does. At 47% both rows sit inside or above
    their bands (1.42×, 1.39×), and every row clears its 1.1× bar by a wide margin.
- **Prefill passes against every llama.cpp configuration** (2.4–10.6×).
- **Prefill against FreeToken misses at 4K (0.92×) and 32K (0.87–0.88×)**, at both shares. FreeToken streams every
  expert once per chunk of up to 8K tokens and overlaps it with compute.
  - A probe with 8,192-token steps (`max_step_tokens` 8192, pinned again at 3,184 / 1,590 / 1,594;
    `arms_step8k.json`, 3 rounds) lifts the 32K prefill by 2.3%, to 4,827 tok/s (0.90×). It costs 0.6% of decode.
    It neither closes the row nor enters the lock.
  - The 4K prompt is a single step either way.
  - **By §8's rule, a natural-fit row measured below its bar goes to the owner.**

## Capped fit (25%: recorded, judged after step 6 with F5b)

Holon is `holon_25` (`moe.cache=*=0.25`, the same lock settings otherwise).

| Row (bar) | decode 1K | 4K | 32K | prefill 1K | 4K | 32K |
|---|---|---|---|---|---|---|
| FreeToken (1.0×) | 1.07 [1.02–1.15] | **0.98** [0.91–1.03] | 1.01 [0.99–1.08] | 1.19 | **0.90** | **0.87** |
| `-fitt 7150` (1.5×) | 2.67 [2.69] | 2.35 [2.32] | 1.91 [1.90–2.08] | 4.38 | 14.46 | 6.90 |
| `--n-cpu-moe 36` (1.5×) | 2.63 [2.64] | 2.31 [2.33] | 1.91 [1.87–2.04] | 4.54 | 14.74 | 7.09 |
| cache, mmap (1.1×) | 2.26 [2.29] | 2.29 [2.26] | 1.82 [1.78–1.95] | 4.30 | 14.32 | 6.92 |
| cache, pinned (1.1×) | 1.22 [1.19–1.32] | 1.22 [1.15–1.29] | **1.09** [1.08–1.17] | 2.10 | 6.87 | 3.58 |

Holon decodes 122.9 / 106.8 / 77.0 tok/s. It misses 174 / 338 / 340 MB per token, well above the replay's
≈ 200–230 MB at 4K / 32K (§13). That gap is where F5b, or a look at the 25% seed, would act at step 6.

## Exactness on the RTX 5080 (D4, §6)

- **The lock's own settings:** `holon_judge snapshot --runs 3` (batch 1, 33,792 context, the profile, `dma`) gives
  2,163 steps, **run-identical ×3**.
- **Cross-placement identity** (`spikes/x3_policies/run.py`, the judge's snapshot trace at `max_seqs` 64, `ring_copy`
  `dma`, 3 fresh engines each). All 12 snapshots are **byte-identical to each other**:

| Configuration | Cached experts | Adaptivity per run | F11 parity |
|---|---|---|---|
| natural, F11 (`lfu`) | 3,220 | 3,664 in-graph admissions, 1 ring step, 3 lends | equal at every step; device map = mirror |
| capped `*=0.25`, F11 | 1,536 | 2,228 admissions, 1 ring step | equal at every step; device map = mirror |
| natural, `profile_decay` (F2–F3) | 3,220 | 16 two-phase swap rounds | — |
| natural, `lru` (F4) | 3,220 | 17,423 admissions | — |

  Every run of each configuration is identical too: run-identical ×3 in each.

## Found on the way (fixed, each with a test that failed before)

- **`9b49cee` std.tiered: a fill arena's minimum.** At batch 1, a graphed step's staging (k × `max_seqs` = 8) is
  below a prefill pass's (128). The startup profile's minimum pools cache nothing, so the layout falls back to
  staging a pass, which the minimum did not hold. The 5080's fill run at `max_seqs` 1 crashed at startup with
  `shape [136, 1769472]` for a 16-slot arena. A fill class's minimum is now its null slots plus a pass's staging.
- **`63c6e4b` profiles: hub-cache checkpoints.**
  - The store read a profile beside the file as the weight source resolves it, which is the hub cache's blob
    directory. `holon_tools profile` wrote beside the snapshot link, so the first profiled arm failed with
    `FileNotFoundError`.
  - Both now use `TieredStore.profile_dir`. Design §14 records it.
- **Not fixed (recorded):** a fill's slot counts do not start when pinned (≈ 18 MB, above). A lock pins a few
  experts below the fill until the fill and pinned startup profiles agree.

## Reproduce

```bash
python spikes/x4_gate/fit.py <gguf> [--cache <pin>] [--step 8192]          # the natural fit, then the pin
python spikes/x4_gate/heldout.py <heldout.jsonl>
python -m holon_tools profile <gguf> --prompts <heldout.jsonl> --max-tokens 128 --name profile_x4_gate.json \
    --max-seqs 8 --max-model-len 4096 --host-memory 21474836480 --plugin-options moe.cache=<pin>
python spikes/x4_gap/ctx.py run --name tune --arms spikes/x4_gate/arms_tune.json --rounds 4 --model <gguf> \
    --contexts 1024 4096 32768 --ctx 33792 --out-dir <runs> --python <freetoken venv python> --server <llama-server>
python -m holon_tools tune <gguf> --plans <copy of plans> --batch 1 --golden judge/goldens/qwen3-30b-a3b-q4_k_m/v1-fp32.safetensors \
    --floor judge/goldens/qwen3-30b-a3b-q4_k_m/v1-bfloat16-mmq-floor.safetensors --max-seqs 1 --max-model-len 33792 \
    --host-memory 21474836480 --slots logits_processor=grammar.xgrammar --plugin-options moe.store=std.moe.offload@1 \
    moe.placement=placement.lfu@1 moe.placement.profile=profile_x4_gate.json moe.cache=<pin> moe.ring_copy=dma --approve
python spikes/x4_gate/arms.py --pin <pin> --share 0.5241 --kernels <the lock's pins as op=provider>
python spikes/x4_gap/ctx.py run --name gate_nat --arms spikes/x4_gate/arms_natural.json --rounds 3 ...   # and gate_cap, step8k
python -m holon_judge snapshot --model <gguf> --prompts judge/prompts/v1.jsonl --runs 3 --engine-args '<the lock's flags>'
python spikes/x3_policies/run.py <gguf> <exact>/x3 3 "nat|<pin>|||||copy=dma" "cap|*=0.25|||||copy=dma" \
    "adapt|<pin>|placement.profile_decay@1||||copy=dma" "lru|<pin>|placement.lru@1||||copy=dma"
python spikes/x4_gate/record.py --runs <runs> --lock <lock> --date 2026-10-17 --commit 63c6e4b \
    --out evidence/x4_gate_rtx-5080_2026-10-17.json
```
