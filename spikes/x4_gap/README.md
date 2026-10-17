# x4_gap — where FreeToken is faster than Holon on the development vehicle

Records: `evidence/x4_gap_rtx-5090_2026-10-17.json` (question 1), `evidence/x4_gap_rtx-5080_2026-10-17.json`
(questions 2 and 3). A diagnosis for the owner's choice of the development gate's bars (`docs/v1-offload.md` §1
acceptance 3, §13): no code in core, std or plugins changed.

## Questions

1. **All-resident decode** (RTX 5090, everything fits): Holon `std.moe@1` vs llama.cpp vs FreeToken at batch 1, 1K
   and 4K context. Where does Holon's extra time go?
2. **Offload decode at the natural fit** (RTX 5080, ≈ 47% of the experts in VRAM): Holon's offload store as built at
   step 2 (static placement F1, every miss gathered F5a) vs FreeToken. Separate the policy (bytes fetched) from the
   per-byte and per-kernel efficiency.
3. **Prefill at 4K** on the 5080's natural fit: FreeToken's 10.7K tok/s vs Holon's routed-only staging in 512-token
   passes. What would large-chunk streaming (D8) buy?

## Method

- **Vehicle:** Qwen3-30B-A3B Q4_K_M (`unsloth/Qwen3-30B-A3B-GGUF`), batch 1, greedy, `ignore_eos`.
- **Engines:**
  - Holon on the v1-offload branch at `b492595`: step 2's `std.moe.offload@1` with `moe.cache=*=0.47` (or `*=0.25`, `*=1.0`)
    and `--host-memory 20 GiB`; the static policy without a profile (its default: experts in id order) and with
    `moe.placement.profile` (see below).
  - llama.cpp `f74e2fa` and FreeToken `21d9578`: the `x4_refs` builds (`spikes/x4_refs/README.md`).
- **Driver** `ctx.py`: `bench/offload_ctx.py`'s method. Each arm runs in a fresh process: an untimed 512 + 64
  warm-up, then one request per context (1,024 and 4,096 prompt tokens of natural text, the same ids for every
  engine) generating 256 tokens. Every engine allocates a 33,792-token context. Rounds are interleaved; the tables
  give medians (3 rounds for the main arms, spread ≤ 0.6%; 1–2 rounds for the probes).
  - Holon's worker steps the engine in process and stamps each token as it reaches the host, as FreeToken's worker
    does.
  - With the offload store, the worker records every decode step's routed (layer, expert) pairs from the store's own
    telemetry (`after_commit`'s counts). Misses and bytes are therefore counted, not modelled.
  - FreeToken runs with `moe_collect_stats` (its in-graph decode miss counters). Its miss bytes are its misses ×
    2.857 MB, the mean expert size.
- **nsys** (2025.3, `--cuda-graph-trace=node`): one extra run per arm, with the measured request bracketed by
  `cudaProfilerStart`/`Stop` and its decode marked by an NVTX range.
  - `nsys_breakdown.py` sorts every kernel and copy into a category and reports µs per token. `idle` is the window
    less the union of busy intervals; `overlap` is time where streams ran concurrently.
  - llama-server is not marked, so its window is its last 200 decode steps, delimited by the per-step logits copies.
  - Profiling slows the runs by 3% (Holon), 4% (FreeToken) and 20% (llama.cpp, which runs many small, concurrent
    kernels). llama.cpp's profiled idle (0.90 ms) is mostly inflation: its unprofiled step is 2.94 ms against
    2.76 ms busy.
- **Routing profile** (`routing_profile.py`): decode routing counts from M6's agent and mixed traces. These share no
  prompt with the judge-corpus spans decoded here, as in M6's held-out scoring. The file lies beside the checkpoint,
  as `moe.placement.profile` requires.
- **Replay** (`replay.py`): M6's simulator on the routing Holon recorded, starting from the residency Holon's seed
  actually had. Its `static` row reproduces the measured misses exactly (e.g. 40.18% hits, 655.44 MB per token at
  25%, 1K).
- **Prefill probes** (Q3): two arm keys set, in Holon's worker process only, std.moe's expert-path pass (`CHUNK`, 512)
  and kernels.ggml_cuda's mmq rows per call (`CHUNK`, 512; at most 2,048, the id helper's 16,384 pairs ÷ top-8).
  Neither is a shipped setting. Larger passes change the greedy tokens slightly (mmq rounding), so those runs' decode
  routing differs a little.

## 1. All-resident decode (RTX 5090): Holon is the fastest; no kernel gap to close

Decode tok/s (3 rounds, median):

| Arm | 1K | 4K | Prefill 1K / 4K |
|---|---|---|---|
| Holon `std.moe@1` | **389.3** | **361.5** | 13.0K / 11.9K |
| llama-server `-ngl 99 -fa on` | 339.9 | 294.4 | 11.4K / 10.3K |
| FreeToken `--moe-strategy fused` (resident) | 340.6 | 322.0 | **20.3K / 24.2K** |
| Holon offload store, `*=1.0` (all cached) | 358.9 | 334.9 | 13.1K / 11.9K |
| FreeToken offload, rate 1.0 (1.3% misses) | 288.4 | 269.7 | 2.7K / 10.8K |

- **Ratios.** Holon runs 1.15× / 1.23× llama-server and 1.14× / 1.12× FreeToken's resident path.
- **Why this differs from `docs/release.md`.** The release's 0.93× was llama-bench tg128, which has no server loop or
  sampling, at Holon's older ggml pin. The gate measures llama-server, as here.
- **The offload store costs +0.22 ms per token on this card with every expert cached.** That is
  `holon_off100 − holon_res`, against the A5000's +0.20. FreeToken's offload path costs +0.53 / +0.60 ms over its
  resident path.
- **Prefill is the exception.** FreeToken's resident prefill runs 1.6× / 2.0× Holon's (see Q3: mmq rows per call).

Per decode token at 1K (nsys, µs; launches in the record):

| Category | Holon | llama-server | FreeToken fused | Holon off `*=1.0` | FreeToken off 1.0 |
|---|---|---|---|---|---|
| MoE expert GEMV (mmvq with ids) | 900 | 908 | 832 | 836 | 823 |
| dense linear GEMV | 521 | 924 | 520 | 517 | 520 |
| attention (incl. q/k norm, RoPE, KV write) | 327 | 851 | 620 | 327 | 622 |
| activation quant (q8_1) | 275 | 618 | 200 | 211 | 200 |
| router (f32 gate GEMV + top-k) | 281 | 269 | 198 | 275 | 192 |
| embedding / glue (casts, adds, index kernels) | 177 | 63 | 14 | 190 | 16 |
| lm_head | 153 | 153 | 152 | 153 | 152 |
| norms | 143 | 176 | 137 | 144 | 140 |
| MoE combine | 64 | 69 | 43 | 63 | 42 |
| expert fetch / resolve | — | — | — | 69 / 104 | 369 / 93 |
| busy / idle (profiled) | 2,565 / 84 | 2,758 / 900 | 2,706 / 350 | 2,762 / 110 | 3,161 / 428 |

- **Where Holon wins:**
  - attention: one fused q/k-norm + RoPE + KV-write kernel and Triton paged decode, −0.29 ms against FreeToken and
    −0.52 ms against llama.cpp;
  - merged QKV, so fewer GEMVs and q8_1 quantizations, −0.40 ms and −0.34 ms against llama.cpp;
  - near-zero idle.
- **Where Holon loses to FreeToken (0.41 ms per token, all of it kernel work):**

| Source | Cost per token | Detail |
|---|---|---|
| Glue | +0.16 ms | `MoE.forward`'s `xc.float()` cast for the f32 router (48 copies), the unfused residual add (48), small index kernels |
| Router | +0.08 ms | cuBLAS runs the f32 128 × 2048 GEMV as dot + reduce, two kernels per layer, beside ggml's top-k |
| Resident gate/up mmvq | +0.07 ms | the stacked layout; the arena runs at FreeToken's speed (836 against 900) |
| q8_1 activation quantization | +0.07 ms | resident path only (275 against 211 in the arena) |
| Combine | +0.02 ms | |

## 2. Offload decode at the natural fit (RTX 5080): the gap is the placement policy

Decode tok/s (median of 3 rounds; 25% rows 1 round), routed experts that missed, and miss bytes per token:

| Arm (47%: 2,888 of 6,144 experts) | 1K tok/s | 4K tok/s | Hit rate 1K / 4K | Miss MB per token 1K / 4K |
|---|---|---|---|---|
| Holon static, id-order seed (step 2 as built) | 52.6 | 53.4 | 43.6% / 45.9% | 617 / 592 |
| Holon static, profile seed | 79.9 | 86.2 | 68.0% / 72.8% | 352 / 299 |
| FreeToken offload, rate 0.47 (its LRU) | **212.7** | **174.5** | 97.8% / 95.3% | 24 / 52 |
| llama.cpp `--moe-cache-mib 7900`, pinned | 188.9 | 147.2 | (95.2%, x4_refs) | (≈ 43) |
| Holon static, profile seed, 25% | 50.1 | 51.7 | 40.2% / 43.9% | 655 / 616 |
| FreeToken, rate 0.25 | 115.1 | 109.1 | 82.3% / 81.7% | 195 / 201 |

The FreeToken and llama.cpp rows reproduce `x4_refs` within 0.5%.

**Per-byte efficiency is equal.** The fetch cost per MB, at 4K:

| Engine | µs per MB | GB/s |
|---|---|---|
| Holon (static), in-graph `k_fetch` gather | 25.1 | 40 |
| FreeToken, `fast_index_copy_multi` | 23.0 | 43 |

Holon's two placements of one cache size also give a line through its unprofiled step times:

| Context | ms per token |
|---|---|
| 1K | 3.885 + 0.0245 · MB |
| 4K | 4.302 + 0.0244 · MB |

The slope is a 41 GB/s effective gather. The intercept is everything but the misses. Both engines move bytes alike;
Holon fetches 11–26× more of them.

**Gap at 4K, Holon as built minus FreeToken** (nsys, µs per decode token; the window is 18,925 against 5,830):

| # | Component | Holon | FreeToken | Δ | Closed by |
|---|---|---|---|---|---|
| 1 | expert fetch: 592 against 52 MB per token | 14,817 | 1,192 | **+13,625** | **policy**: per byte the two are close (≈ +0.1 ms at FreeToken's bytes) |
| 2 | glue (casts, adds, index kernels) | 201 | 15 | +186 | kernel work: fold the f32 cast into the router, fuse the residual add |
| 3 | router (cuBLAS dot + reduce, top-k) | 264 | 208 | +57 | kernel work: one router GEMV (+ top-k) kernel |
| 4 | dense GEMV | 829 | 775 | +54 | small (the same mmvq kernels; QKV is already merged) |
| 5 | resolve / cache bookkeeping | 150 | 112 | +39 | offload path: `k_resolve` 3.1 µs per layer against FreeToken's `_lru_ensure` 2.3 |
| 6 | combine, q8_1, norms, lm_head | — | — | +44 | kernel work (small) |
| — | attention | 745 | 1,042 | −298 | Holon ahead |
| — | MoE expert GEMV (arena) | 1,197 | 1,369 | −172 | Holon ahead |
| — | idle | 117 | 444 | −328 | Holon ahead |
| — | overlap (concurrent streams) | −135 | −23 | −111 | |

Everything but row 1 sums to −0.53 ms: **outside the fetch, Holon is faster than FreeToken.**

**What a dynamic policy would fetch.** The replay starts from Holon's profile seed. Its time comes from the line
above, and the ratios are against this run's medians:

| Placement at 47% | MB per token 1K / 4K | Holon tok/s 1K / 4K | vs FreeToken 1K / 4K | vs llama.cpp pinned cache 1K / 4K |
|---|---|---|---|---|
| static, id-order seed (measured) | 617 / 592 | 52.6 / 53.4 | 0.25 / 0.31 | 0.28 / 0.36 |
| static, profile seed (measured) | 352 / 299 | 79.9 / 86.2 | 0.38 / 0.49 | 0.42 / 0.59 |
| `lru` (replay) | 36.3 / 68.0 | 209 / 168 | 0.98 / 0.96 | 1.11 / 1.14 |
| `lfu`, step 3's default (replay) | 35.0 / 66.9 | 211 / 169 | 0.99 / 0.97 | 1.12 / 1.15 |
| FreeToken's own miss bytes | 23.7 / 51.9 | 224 / 180 | **1.05 / 1.03** | 1.19 / 1.22 |
| `belady` (replay, the bound) | 19.2 / 35.2 | 230 / 194 | 1.08 / 1.11 | 1.22 / 1.32 |

The same replays at 25%:

| Placement at 25% | MB per token 1K / 4K | vs FreeToken (rate 0.25) 1K / 4K |
|---|---|---|
| `lru` | 201 / 228 | 0.99 / 0.93 |
| `lfu` | 173 / 216 | 1.07 / 0.96 |
| FreeToken's bytes | 195 / 201 | 1.00 / 1.00 |

**Why the replays fetch more than FreeToken.** The replayed `lru` / `lfu` fetch 12–16 MB more per token than
FreeToken does. The likely cause is the warm start: Holon's trace holds decode routing only (prefill rows never count
or admit, D4), so a replayed policy starts cold for the prompt's experts. FreeToken's cache has just streamed the
prompt.

## 3. Prefill at 4K (RTX 5080, natural fit)

| Arm | tok/s | Prefill time (nsys) | Fetch / stream | mmq (expert + dense) | Attention | Rest |
|---|---|---|---|---|---|---|
| FreeToken (streams all 17.55 GB once per chunk, copy engine, overlapped) | **10,688** | 386 ms | 368 ms H2D (overlapped) | 200 ms | 77 ms | ≈ 40 ms |
| Holon static, profile seed (as built: 512-token passes, 512-row mmq) | 2,041 | 2,015 ms | 1,379 ms `k_fetch` | 493 ms | 75 ms | ≈ 68 ms |
| + expert pass 4,096 tokens (probe) | 4,865 | 848 ms | 211 ms | 494 ms | 76 ms | ≈ 67 ms |
| + mmq 2,048 rows per call (probe) | 6,637 | 622 ms | 211 ms | 266 ms | 75 ms | ≈ 70 ms |

1K prefill: FreeToken 2,711 tok/s; Holon as built 2,304; with 4,096-token passes 3,118 (1.15×).

**What the as-built path loses:**
- **Refetching per 512-token pass: 1.17 s.** Each pass gathers its routed misses again: 8 passes × 48 layers,
  ≈ 55 GB at 4K. One pass per chunk gathers the ≈ 8.9 GB once. This is offload-path design (D8's chunk size).
- **512-row mmq calls: 0.29 s.** The expert GEMMs take 2.5× FreeToken's: every call re-reads every expert's weights
  for ≈ 32 tokens. 2,048-row calls take 266 ms; FreeToken's 4,096-row calls take 200 ms. Reaching 4,096 rows is
  kernel work: the id helper's 16,384-pair bound (`ID_PAIRS`) must grow. The same cause gives FreeToken's 1.6–2×
  resident prefill on the 5090.
- **No overlap: 0.21 s serial.** The gather runs between the layers' compute. FreeToken hides its 368 ms of
  copy-engine streaming behind compute.

**What D8 would buy.** Hide the 211 ms behind each layer's compute (≈ 410 ms at 2,048-row calls). Use a copy engine
on a side stream (M1's DMA rate); the SM gather would compete with the GEMMs.

| Variant | Prefill time | tok/s | vs FreeToken |
|---|---|---|---|
| Serial, 2,048-row mmq (measured) | 0.62 s | 6.6K | 0.62× |
| Overlapped, 2,048-row mmq (modelled) | ≈ 0.41 s | ≈ 10.0K | ≈ 0.93× |
| Overlapped, 4,096-row mmq at FreeToken's GEMM time (modelled) | ≈ 0.35 s | ≈ 11.9K | ≈ 1.1× |

Holon streams only the non-resident 53%, so it is compute-bound where FreeToken is transfer-bound.

## The gap, ranked (ms per token, RTX 5080 natural fit, 4K)

| Rank | Component | Decode Δ (ms per token) | Prefill Δ (s per 4K prompt) | Kind of fix |
|---|---|---|---|---|
| 1 | **Placement policy**: 592 against 52 MB of misses per token. A profile seed alone cuts the bytes to 299 MB; replayed `lfu` / `lru` reach 67–68 MB. | **+13.6** (profile seed: +6.4; `lfu`: ≈ +0.4) | — | policy: step 3's `lfu` (default) / `lru`; the remaining ≈ 0.4 ms is prompt-time admission (FreeToken's cache has seen the prompt; D4 never admits prefill rows) |
| 2 | **Prefill refetch per 512-token pass** | — | **+1.17** | offload-path design: D8, one expert pass per large chunk |
| 3 | **Prefill mmq at 512 rows per call** | — | +0.29 (+0.07 left at 2,048 rows) | kernel work: larger mmq calls (`ID_PAIRS` beyond 16,384 for 4,096 rows) |
| 4 | **Prefill transfer not overlapped** | — | +0.21 | offload-path design: D8's ring on a copy engine |
| 5 | glue kernels (MoE input cast, residual add, index) | +0.19 | ≈ 0 | kernel work |
| 6 | router (cuBLAS dot + reduce) | +0.06 | ≈ 0 | kernel work |
| 7 | resolve / bookkeeping | +0.04 | ≈ 0 | offload path (small) |
| 8 | per-byte gather efficiency (25.1 against 23.0 µs per MB) | ≈ +0.1 at FreeToken's bytes | — | small; larger gathers per launch |
| — | Holon ahead: attention −0.30, expert GEMV in the arena −0.17, idle −0.33 | −0.80 | ≈ 0 | — |

## What this implies for the development gate's bars (the owner's choice)

- **The FreeToken decode bar (≥ 1.0×) is a policy bar, not a kernel bar.**
  - F1 static cannot reach it: 0.25–0.31× as built, 0.38–0.49× with a profile. F1's own row (≥ 0.95× llama.cpp's
    static split) is unaffected.
  - With step 3's `lfu`, the replay predicts **0.97–0.99× at 47%** and 0.96–1.07× at 25%. FreeToken's own miss bytes
    would give 1.03–1.05× / 1.00×. So 1.0× sits on the warm-start detail: whether prefill-routed experts enter the
    cache.
  - Kernel work on rows 5–6 (≈ 0.25 ms, 4–5% at 4K) moves every one of these predictions up by about as much.
  - The bar is predicted to pass if both land: prompt-time admission, which is a D4 change, and the glue and router
    kernels. With either alone it is marginal. A 0.95× bar would pass on `lfu` alone.
- **The llama.cpp `--moe-cache-mib` bar (≥ 1.1×) now looks reachable.**
  - The prediction is 1.12–1.15× with `lfu` (replay) and 1.19–1.22× at FreeToken's bytes.
  - §13's re-band (0.98–1.11×) assumed Holon's all-resident work at 0.93× llama.cpp's. Against llama-server it is
    1.15–1.23× faster. Holon's 1K base (everything but the misses) is 3.89 ms, against llama.cpp's 3.93 ms of GPU
    work plus 0.94 ms of host sync (x4_refs, 1K).
- **The FreeToken prefill bar (≥ 1.0×) needs all of D8.**
  - Required: one expert pass per large chunk, overlap on a copy engine, and mmq calls of ≥ 2,048 (better 4,096)
    rows.
  - Modelled ≈ 0.93× at 2,048 rows and ≈ 1.1× at 4,096; serial large passes reach only 0.62×.
  - With staging only (no D8), Holon prefills at 0.19× at 4K and 0.85× at 1K.
  - The llama.cpp prefill bar (≥ 1.0× its pinned cache's 1.8K / 1.6K tok/s at 1K / 4K) passes even as built
    (2.3K / 2.0K, 1.26× / 1.24×), and by 1.7× / 3.0× with 4,096-token passes alone.
- **None of this needs core changes or new kernels for decode beyond rows 5–6.** The decode gap is closed by step 3's
  policy plus prompt-time admission. The prefill gap is D8, plus larger mmq calls in `kernels.ggml_cuda`.

## Caveats

- **Contexts.** Only 1K and 4K were measured: the 32K rows, and the RTX 5080's 32K natural fit, are not re-measured
  here. The fits are those of `x4_refs`.
- **Replays.** The policy replays use Holon's own decode routing, 255 steps per context. They model the miss bytes;
  their tok/s comes from Holon's measured bytes-to-time line. They were not run on the engine: step 3 builds `lfu`.
- **FreeToken's miss bytes** use the mean expert size. Its Q6_K-down layers are 7% larger than the mean, Q4_K's 7%
  smaller.
- **Prefill probes.** These patch two module constants in the worker process. The 4,096-row mmq and overlapped rows
  are models from the measured kernel times, not runs.

## Reproduce

```bash
python spikes/x4_gap/routing_profile.py <checkpoint dir>/profile_agent_mixed.json <M6 agent.npz> <M6 mixed.npz>
# per question, inside the bar desktop's container (x4_refs' llama.cpp and FreeToken builds; Holon at this commit):
python spikes/x4_gap/ctx.py run --name q1 --arms <arms.json> --rounds 3 --model <gguf> --contexts 1024 4096 \
    --ctx 33792 --holon-python <holon venv python> --python <freetoken venv python> --server <llama-server>
#   arms: {"name", "engine": holon|llama|freetoken, "flags"} (+ "rounds", "contexts", "chunk", "kernel_chunk",
#   "nsys_context" and "wrap": "nsys profile -t cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi
#   --capture-range-end=stop -o <report>", or "server_wrap" for llama-server); the arms as run are in the records
nsys export --type sqlite -o <report>.sqlite <report>.nsys-rep
python spikes/x4_gap/nsys_breakdown.py <report>.sqlite [--window logits:200] [--prompt 4096]
python spikes/x4_gap/replay.py <holon trace .npz> ...      # > replay.json beside the runs
python spikes/x4_gap/record.py --runs <runs dir> --date 2026-10-17 --commit b492595
```
