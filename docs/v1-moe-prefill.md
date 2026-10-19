# v1 — MoE prefill speed

Status: **design rev 6 (2026-10-18); building: steps 0–3, 3b, 3c and 3d done** (§2.1–§2.2, §7.1). Step 0 tripped no
decision rule; C-M6's fallback is taken (J against T up to 16 tokens; 64 on sm_89, B-dec). **Step 3: D3 fails M0.7's
keep rule on both lock classes** and is reverted (§10 L3-1); steps 3b / 3c's 0.1–0.3% prefill cost is gone (L3-2: the
id workspace at `pass_rows` again). **Step 3d:** D3′ (the router per pass) is kept: it passes M0.7's rule on the 5090's
32K row and the 5080's 4K row (not on every row). The 5090 4K row stays under its bar with every item (O2, with the
owner); the sm_89 record is a gate before step 5 (§10 L3-4). **Step 4:** every prefill bar is met in one session
(5090 4K 1.008×, 32K 1.016× gated; 5080 4K 1.141×, 32K 1.006× gated); the 5080 offload lock's 4K decode row misses
acceptance 5's tolerance by 0.05 pp (§7.1).
- Rev 2 answered the three-lens panel's first round (A correctness, B architecture, C performance: 3 blockers, 19
  majors, 23 minors).
- Rev 3 answered its second round and took in FN's requests.
- Rev 4 answered its third round: D11 goes through the policy contract, and D10's contract is written out.
- Rev 5 answered the fourth round (A's and C's shared major: the tail step's order and counting; C's minor on the
  fallback buffer).
- Rev 6 answers the fifth round: A's minor on `lend_tail`'s call number and C's minor on who owns the pass's
  fallback buffer. B is clean.

All are applied with the lead's decisions (§10). Rev 6 awaits the panel's re-review.

This is the follow-up that plan §10 opened when milestone X closed (`docs/v1-offload.md` §13, "Owner decision on the
gate").

**The shortfall.** Holon's prefill trails FreeToken 0.92× / 0.88× at 4K / 32K on the RTX 5080's natural-fit offload
rows. FreeToken's all-resident prefill runs 1.6–2.0× Holon's on the RTX 5090.

**Step 0** is `spikes/moe_prefill/README.md` (`evidence/moe_prefill_rtx-a5000_2026-10-18.json`). It ran Qwen3-30B-A3B
Q4_K_M all-resident on one RTX A5000, with nsys per component, a per-call GEMM bench, and a source study of FreeToken's
and llama.cpp's prefill GEMMs. Its bar-card half (§7 step 0) is still to run.

## 1. Goal and acceptance

Close the MoE prefill gap to FreeToken without leaving Holon's exactness classes and without changing the core.

**The cause** is the expert GEMMs' launch shape, not their arithmetic:
- both engines run ggml's mmq with q8_1 activations and int8 MMA;
- Holon calls it at 512 rows on resident stacks;
- its grouped grid launches every column tile of every slot.

The bars are **proposed for the owner to confirm** (§9). The predictions per item set are in §1.1 and are fixed
before any code. Every bar is measured by §5.2's method.

1. **The recorded shortfall rows, against FreeToken's gated arm.** RTX 5080, Qwen3-30B-A3B Q4_K_M at the natural fit
   (the offload lock, re-filled and re-pinned at the close, §5.3):
   - prefill ≥ 1.0× FreeToken **gated** (acceptance 3) at 4K and 32K.
   - The 0.92× / 0.88× shortfall was recorded against FreeToken's default 8,192-token chunks. At 4K the two arms are
     the same run, since a 4K prompt is one chunk either way.
   - At 32K the gated reference is a different, slower FreeToken. Step 0 measures it at 4,096 and 8,192 chunks; until
     then it is modelled ≈ 4–7% slower than the reference that recorded 0.88× (C-N-M1: it streams every expert once
     per chunk, 8 times instead of 4). The record states the measured distance beside the bar.
   - the 1K row stays ≥ 1.5×;
   - every llama.cpp prefill row stays ≥ 2.1×.
2. **All-resident.** RTX 5090, same model (the lock's settings with `max_model_len` 33,792 and `max_seqs` 1, stated in
   the record):
   - prefill ≥ 1.0× FreeToken at 4K and 32K, and ≥ 0.95× at 1K;
   - prefill ≥ 1.0× llama-server at every length (1.14–1.16× today).
   - The A5000 rows are recorded as the development measurement; they are not a bar.
3. **FreeToken's arms** (lead decision; release.md's rule that every backend gets the lock's step budget):
   - the **gated** arm pins FreeToken's `--max-extend-length` to the lock's `max_step_tokens`;
   - FreeToken's default (8,192-token chunks) is reported beside it, **not gated**.
   - **Next to every bar:** Holon keeps the f32 expert path and the f32 router (m02 D3; the `moe_router@1` contract).
     FreeToken's bf16 intermediates and bf16 router are worth about 3.5% to it at 4K on the A5000 (row 7 ≈ 2.4%, its
     router's remainder ≈ 1.1%). That is Holon's numerics handicap, not a gap D1–D6 can close.
4. **Exactness** (§5):
   - **D1, D2, D4 bitwise:**
     - every row's prefill logits (the `prompt_logprobs` hash) are identical to main's at 1,027 and 4,096 tokens, on
       the resident 30B, the natural-fit offload and the capped fit (F5b on: identical to main's capped run,
       run-identical ×3);
     - the snapshot traces are identical to main's;
     - L0 holds every type × SM pair the locks use (derived from `plans/`), plus FN's types and shape, and FN's
       requests: the passed buffers and the tail lending (D10, D11).
   - **D3, D5 bounded:** L2 passing against the Tier B prefill floor, plus the bounded chunked check.
     - Far plants must fail.
     - A near-bound plant (`router_prefill_bf16_logits`, `qlinear_partials_bf16`) that passes L2 records L2's
       blindness at that scale, and D3 / D5 are kept: they are bounded by design and the plants sit at the bound
       (lead decision).
5. **Decode keeps its bits and its speed.**
   - **Bits:**
     - steps of ≤ `mmvq_tokens` (8) rows are untouched;
     - D3 applies only to router calls of ≥ `CHUNK` rows, so batched decode and verify keep main's router bits while
       a step's router chunk stays under 512 rows: `max_seqs · (1 + max_draft_tokens)` < 512, true for every lock today;
     - batched decode above 8 rows runs mmq with ids inside graphs, so D2 changes its launch but not its bits
       (graphs-vs-eager bitwise).
   - **Speed:** a same-session ABBA against main's build, `holon_tg` at batch 1 (1K / 4K) and batch 16 / 32 / 64 on
     the 5090, and the gate's offload decode rows — measured on the re-filled lock against main's lock (C). Each must
     stay within ≤ max(0.5%, 2 sd).
6. **Memory** (§6):
   - every lock's startup KV pool stays ≥ its `max_model_len` and its recorded capability (each pass size is capped
     per class to keep this);
   - startup KV tokens are recorded per lock, main against the branch;
   - the 5080 offload lock's expert-cache share falls after its re-fill by no more than the growth of its persistent
     buffers (D5's ≈ 25 MB ≈ 9 experts, ≈ 0.3% of the share), and its decode rows meet acceptance 5's tolerance.
7. **No core diff** and no ABI bump. The std minors are `pass_rows` (§4), the passed buffers (D10) and tail lending
   (D11). Suite time does not grow (lean tests, §5.1).

### 1.1 Predictions per item set (A5000 step 0; restated from the bar cards in §2.2)

The base is the measured all-4,096 probe (D1 and D5 in, A5000 GPU-busy ms):

| Length | Probe | FreeToken default | FreeToken gated at 4,096 |
|---|---|---|---|
| 1K | 212 | 172 | 172 |
| 4K | 686 | 584 | 584 |
| 32K | 10,013 | 9,184 | ≈ 9,423, **modelled** (its 4K-call GEMM rate per token); step 0 measures it |

Item savings at 1K / 4K / 32K (ms):

| Item | 1K | 4K | 32K | Note |
|---|---|---|---|---|
| D2 | −26…−30 | −70…−85 | −577…−694 | the lower end removes FreeToken's bf16-store advantage (C-M1; step 0 splits it with a probe) |
| D3 | −3 | −13 | −107 | |
| D4 | −5 | −21 | −166 | upper bound: the fused write-back assumed free |
| D6 | — | — | −239 | |

**Offload rows.** They scale the A5000 fractions from today's measured 5080 ratios (0.92× / 0.88×). D1 does not apply
there (the arena already makes 4,096-row calls), so the A5000 base is the 4,096-expert-row arm (703 ms at 4K) with
dense calls at 512 rows; D5 saves 20 ms there (C2-m3).

**Gated cells are modelled until step 0 measures FreeToken at 4,096 and 8,192 chunks** (lead decision; C-N-M1):
- resident: FreeToken +2.6% at 32K (its 4K-call GEMM rate);
- offload: +2.6% (C's first model) up to +4–7%. Gating doubles FreeToken's full-expert streams at 32K, so the band
  runs from the first to the second.
- Step 0's measurements replace every gated cell.

**Rows at or below their bar are bold** (the "≤ bar" column). Rows within 3% of the bar take ≥ 5 rounds (§5.2).

| Row (bar) | D1–D3 + D5 | + D4 (rev 2's set) | + D6 | ≤ bar |
|---|---|---|---|---|
| 5090 resident 1K (≥ 0.95×) | **0.94–0.96×** | 0.96–0.99× | — | D1–D3 + D5 |
| 5090 resident 4K (≥ 1.0×) | **0.97–0.99×** | **1.00**–1.03× | — | both: at the bar at the low end |
| 5090 resident 32K, gated (≥ 1.0×), modelled | 1.01–1.02× | 1.03–1.04× | 1.03–1.04× (FreeToken also at 8,192) | within 3% |
| 5090 resident 32K, FreeToken default (reported) | **0.98–1.00×** | **1.00**–1.02× | 1.03–1.04× | — |
| 5080 offload 4K (≥ 1.0×) | 1.08–1.11× | 1.12–1.15× | — | — (before a transfer-bound haircut; step 0's nsys decides) |
| 5080 offload 32K, gated (≥ 1.0×), modelled | **0.99**–1.04× | **1.00**–1.06× | not built on this lock (§9 O1) | within 3% at the low end |
| 5080 offload 32K, FreeToken default (reported; the 0.88× reference) | **0.96–0.98×** | **0.98–0.99×** | 1.00–1.02× at ≈ −2.4% decode at 32K | §9 O1 |
| 5080 offload 1K (≥ 1.5×) | ≈ 1.7× | ≈ 1.7× | — | — |

- **D4 is part of the set** (lead decision, C-B1). Without it the 5090 4K and 1K rows sit under their bars.
- **D6 is nearly neutral in the gated comparison.** With the lock's budget at 8,192, FreeToken's gated chunk is 8,192
  too. It pays only against FreeToken's ungated default, so it is built only where the lead / owner decide (§9 O1),
  never by default on offload locks.
- **Risks to these numbers:**
  - the 5080 4K row's FreeToken is transfer-bound, which caps the gain;
  - Holon's 32K attention lead (+437 ms on the A5000) is unmeasured on sm_120. If it halves, the 32K rows lose ≈ 2–3%.
  - Step 0 on the bar cards restates every cell from their nsys breakdowns (C-M3).

## 2. Where the time goes (step 0, A5000, all-resident; the README's §1–§3)

Time to the first token in s (median of 6), and GPU-busy ms by component (nsys):

| | 1K | 4K | 32K |
|---|---|---|---|
| Holon as shipped | 0.279 | 1.180 | 14.03 |
| Holon, mmq with ids at 4,096 rows per call (probe) | 0.215 | 0.703 | 10.20 |
| Holon, plus dense mmq at 4,096 rows (probe) | 0.212 | 0.683 | 10.03 |
| FreeToken fused | **0.176** | **0.582** | **9.23** |
| expert GEMMs: shipped / probe / FreeToken (ms) | 212 / 144 / 114 | 832 / 383 / 299 | 6,809 / 3,084 / 2,151 |
| attention: Holon probe / FreeToken (ms) | 10 / 12 | 95 / 108 | 5,252 / 5,605 |

The gap, ranked:

| # | Component | Δ at 1K / 4K / 32K (ms) | Fix | Class |
|---|---|---|---|---|
| 1 | Resident expert mmq at 512 rows per call (each expert gets ≈ 32 tokens; J = 128 tiles are 75% padding) | −68 / **−449** / **−3,725** (measured) | D1 | bitwise |
| 2 | Grouped launch: J sized on all tokens; ceil(T/J) column tiles for every slot, 76–92% empty | −26…−30 / −70…−85 / −577…−694 (to FreeToken's per-token rate; the low end at equal I/O dtype, C-M1) | D2 | bitwise |
| 3 | Calls of 4,096 against FreeToken's 8,192 rows | — / — / −239 (against FreeToken's ungated 8,192 default) | D6 | bitwise per row; a lock setting |
| 4 | Dense mmq at 512 rows per call | −4 / −25 / −188 (measured) | D5 | bounded |
| 5 | Router on prompt chunks: f32 cast, SGEMM, split-K, top-k per 512 rows (22 ms at 4K against FreeToken's bf16 2.3) | ≈ −3 / −13 / −107 | D3 | bounded |
| 6 | The f32 SwiGLU pass over gate/up's f32 output | −5 / −21 / −166 (upper bound) | D4 | bitwise |
| 7 | f32 combine and the f32 down input's q8_1 (FreeToken keeps bf16) | −4 / −14 / −112 | not taken (D7) | bounded |
| — | Holon ahead (attention, the id helper) | +1 / +24 / +437 | — | — |

**Holon and FreeToken share the GEMM.** FreeToken's prefill GEMM is ggml's mmq with ids as well:
- `layers/quantization/moe/gguf.py:92-127` → `ggml.mul_mat_id`, its vendored llama.cpp `8c2134a8` plus a patch;
- that patch adds J against the mean tokens per expert, a flat map of occupied tiles, no stream-k, and bf16 out.

**llama.cpp never falls back to dequant + cuBLAS** for k-quants on Turing+ GPUs: `ggml_cuda_should_use_mmq` →
`turing_mma_available`, and `ggml_cuda_mul_mat_id` runs one mmq call with ids above 8 tokens. Per call at 4,096 tokens,
Holon's grouped mmq is 1.11–1.30× FreeToken's on the same stacks and routing (Holon f32 out and f32 down input, FreeToken bf16: not equal I/O, C-M1); FreeToken's is 1.20–1.43× the
dense-mmq bound.

### 2.1 Step 0 on the bar cards: predictions (written before the runs, 2026-10-18)

The runs: RTX 5090 (resident, the 5090 lock's flags at `max_model_len` 33,792 and `max_seqs` 1), RTX 5080 (the
natural-fit offload lock), and the A5000 again for the gated arm and the memory checks; one persistent container per
card. Every engine runs the **fixed prompt set**: `prefill.py`'s prompts r = 0…4 at 1K and 4K and r = 0…2 at 32K
(`--reps 5 5 3`; distinct corpus offsets, so no prompt is a prefix of another). The offload rows use x4_gap's
`ctx.py` (the method that recorded 0.92× / 0.88×). FreeToken's gated arm adds `--max-extend-length 4096`; its default
is 8,192.

| # | Measurement | Card | Prediction | Feeds |
|---|---|---|---|---|
| P1 | FreeToken gated (4,096-token chunks) against its default (8,192), time to the first token | 5090 resident, 5080 offload, A5000 resident | 1K and 4K: the same within noise (one chunk); 32K: gated slower by ≈ 2.6% resident (5090, A5000: 9.47 s against 9.23 s) and by 2.6–7% offload (5080) | every gated cell of §1.1; a gated cell below its bar even with the full set stops the track |
| P2 | Holon as shipped / FreeToken default | 5090 | 0.60–0.66 / 0.47–0.52 / 0.63–0.69 at 1K / 4K / 32K (the A5000's 0.63 / 0.49 / 0.66; x4_gap's 1.6× / 2.0×) | §1.1's base |
| P3 | Holon all-4,096 probe / FreeToken default | 5090 | 0.80–0.86 / 0.82–0.88 / 0.89–0.95 | §1.1's base |
| P4 | Holon natural-fit offload / FreeToken default | 5080 | 1K ≥ 1.5×; 4K 0.89–0.95×; 32K 0.85–0.91× (recorded 0.92× / 0.88×) | the offload rows' base |
| P5 | Achieved rates at T·k = 32K / 64K: dense mmq TOPS, cuBLAS bf16 TFLOPS | 5090 / 5080 | mmq 185–226 / 99–122 TOPS (22–27% of the int8 peaks 838 / 450); bf16 147–168 / 79–90 TFLOPS (70–80% of the f32-accumulate peaks 209.5 / 112.6, a quarter of int8's); mmq / bf16 = 1.1–1.5× | D7 |
| P6 | Dequant of one layer's stacks to bf16 | 5090 / 5080 | ≈ 1.1 / ≈ 2.0 ms | D7 |
| P7 | Per layer at 4,096 tokens: shipped mmq with ids against the bf16 bound + dequant | 5090, 5080 | mmq wins on both (bf16 + dequant ≥ 1.2× mmq with ids); the break-even against shipped mmq at ≥ 8,192 tokens or never, against the dense-mmq bound never | D7: bf16 winning at the lock's 4,096 on a bar card stops the track |
| P8 | Row 2's I/O split: FreeToken per call with f32 activations (f32 store) against bf16 | A5000, 5090 | f32 costs it 3–10%; Holon / FreeToken-f32 at 4,096 tokens 1.08–1.20 (gate/up) and 1.15–1.30 (down) | row 2's band (its low end, −70 ms at 4K) |
| P9 | Batched-decode J sweep at T = 16–64 (C-M6): Holon's call (J against T) against FreeToken-f32's (J against the mean, occupied tiles only) | 5090 | FreeToken-f32 ≤ Holon at every T, so D2 keeps J against the mean at every T (no ≤ 64-row fallback) | D2's J rule |
| P10 | nsys, resident 1K / 4K / 32K | 5090 | expert GEMMs 65–75% of busy at 4K as shipped; attention at 32K: Holon ≤ FreeToken (a 0–8% lead) | the 32K rows' risk (a halved lead costs 2–3%) |
| P11 | nsys, natural-fit offload 1K / 4K / 32K | 5080 | FreeToken's 4K transfer-bound (streaming ≥ 85% of its wall); Holon's 4K expert GEMMs ≤ 40% of its busy time | the 5080 4K row's regime |
| P12 | Startup KV tokens, the 16 GB lock (5080 UD-IQ3_XXS, its flags) with the `--pass-tokens` / `--expert-rows` probes | 5080 | main ≈ 18.6K; 2,048 ≈ 16.8K (above 16,384 by ≈ 0.4K), so the class can take 2,048 | O3, acceptance 6 |
| P13 | Startup KV tokens, 24 GB locks | 4090 (its lock); 3090 (its lock at the recorded 30,208-token context, 64 seats) | 4090: main ≈ 26K, 4,096 −4.2K, 2,048 −1.8K, every recorded row kept. 3090: main ≥ 30.2K; 4,096 breaks the 30,208-token row, and 2,048 breaks it unless main ≥ 32.0K | O3 |
| P14 | The `peak_alloc` rise (22.20 → 22.37 GiB at 4,096 passes) | A5000 | a pass-size transient the startup profile does not see: the run's allocation peak over the profiled peak grows by ≈ 0.17 GiB from 512 to 4,096 passes; a memory snapshot at the run's peak names it | §6 |
| P15 | Reserved bytes after a 32K prefill, over what was allocated at start, against the profiled peak (C4-m) | A5000, 5090 | over the profile already as shipped (the tool's A5000 smoke test read 0.48 GiB reserved against a 0.34 GiB profile after a 4K prompt), by more at 4,096 passes; the 10% `memory_fraction` margin absorbs both | §6, D10's fallback |

### 2.2 Step 0 on the bar cards: measured (2026-10-18)

Records: `evidence/mp0_prefill_rtx-5090_2026-10-18.json`, `mp0_prefill_rtx-5080_…`, `mp0_prefill_rtx-a5000_…`,
`mp0_kv_rtx-4090_…`, `mp0_kv_rtx-3090_…` (code at main's `b5b1e7f`; the commit adds docs and spikes only). Both bar
desktops held their clocks (5090 p10 / p50 2,587–2,602 / 2,632–2,647 MHz; 5080 2,760 / 2,760). Time to the first
token, medians of the fixed prompt set over 2 rounds (10 / 10 / 6 per cell); "×" is FreeToken's time over Holon's.

| # | Measured | Against the prediction |
|---|---|---|
| P1 | FreeToken gated / default at 32K: **5090 3.240 / 3.022 s (+7.2%)**, **5080 offload 6.270 / 5.979 s (+4.9%)**, A5000 9.338 / 9.210 s (+1.4%); 1K and 4K equal within 0.6% on every card | 5090 off (+7.2% against +2.6%); 5080 in its band; A5000 under |
| P2 | 5090 as shipped: 0.632 / 0.503 / 0.666× | held |
| P3 | 5090 all-4,096 probe: 0.843 / 0.852 / 0.880× (0.944× against the gated arm at 32K) | 32K under the band (attention, P10) |
| P4 | 5080 natural fit: 1.795 / 0.936 / 0.873× (0.916× against the gated arm); the recorded 0.92× / 0.88× reproduce | held |
| P5 | Dense mmq: 5090 191–240 TOPS, 5080 101–129; cuBLAS bf16 at T·k = 32K / 64K: **5090 205–226 TFLOPS, 5080 106–114** (at their f32-accumulate peaks); mmq / bf16 0.93–1.18 | mmq held; bf16 far above the band (78%-of-peak assumed), so mmq's edge is ≈ 1.0–1.2×, not 1.1–1.5× |
| P6 | Dequant per layer: 5090 1.12–1.15 ms, 5080 2.05–2.09 ms | held |
| P7 | Per layer at 4,096 tokens, shipped mmq with ids against the bf16 bound + dequant: **mmq wins, 5090 by 1.17–1.23×, 5080 by 1.10–1.18×**. Break-even pass: against shipped mmq ≈ 6.1K (5090) and ≈ 5.3K (5080) tokens; against FreeToken's f32 rate (D2's target) ≈ 7.9K / 9.0K; against the dense-mmq bound never | **D7's rule holds** (mmq wins at the locks' 4,096) but with less margin than predicted; an 8,192 pass (D6) sits at the break-even against D2's rate on the 5090 |
| P8 | FreeToken f32 / bf16 per call: 5090 1.01–1.06 (1.03–1.06 at 4,096), 5080 1.00–1.06, A5000 0.93–1.06 (noise). Holon / FreeToken-f32 at 4,096: 5090 1.04 (gate/up), 1.16 (down); 5080 1.12, 1.26–1.30; A5000 1.14–1.15, 1.26–1.35 | the store costs FreeToken 0–6%; the 5090's gate/up gap is under the band |
| P9 | J sweep, Holon / FreeToken-f32 per call at T = 16 / 24 / 32 / 40–64: 5090 0.86–0.95 / 0.95–0.99 / 1.0–1.13 / 1.1–1.4; 5080 0.92–1.00 / 0.96–1.06 / 1.07–1.18 / 1.15–1.49; A5000 0.90–0.99 / 0.97–1.19 / 1.04–1.32 / 1.25–1.95 | **off at T ≤ 24**: J against the mean (FreeToken's proxy) is up to 14% slower at 16 rows. D2's own launch is measured against main at T = 16–64 at step 2; if it is slower at any of them, C-M6's fallback (J against T up to 64 rows) is taken |
| P10 | 5090 nsys at 4K as shipped: expert GEMMs 211 of 332 ms busy (64%). **Attention at 32K: Holon 2,172–2,178 ms, FreeToken 2,038 (default) / 2,220 (gated)** | the attention lead is gone against the default arm (−6.6%); it holds (+1.9%) against the gated arm only |
| P11 | 5080 nsys at 4K: FreeToken streams 372 ms of its 387 ms wall (transfer-bound); Holon 412 ms busy, compute-bound (streaming 200 ms, overlapped), expert GEMMs 200 ms (48%). The Holon profile ran with a filled cache: under nsys the lock's pinned arena does not fit (C-M7) | FreeToken held; Holon's expert share over the band |
| P12 | 16 GB lock (5080 UD-IQ3_XXS, 32 seats, 16,384): KV tokens 18,752 as shipped; 18,560 at 2,048 passes (−192); 18,176 with dense 2,048 too (−576); a 16,000-token prompt served in each | better than predicted (−0.2K, not −1.8K): the class takes 2,048 (its step budget) |
| P13 | 4090 lock: 26,736 → 25,776 at 4,096 (−960), 26,352 at 2,048. 3090 lock: **42,800** → 42,224 at 4,096 (−576), 42,416 at 2,048 | off: the 3090's pool keeps its 30,208-token row at 4,096 (≥ 30,224 needed); every 24 GB class takes 4,096 |
| P14 | A5000: the run's allocation peak over its start is 0.125 GiB as shipped and 0.391 at 4,096 passes; the profiled peaks are 0.339 and 0.425 GiB (KV −928 tokens) | off: the run never exceeds the profile; the 0.17 GiB `peak_alloc` rise is the profile's slack (0.21 GiB as shipped, set by its sampling and logprob phases) taken up by the larger pass, so §6's live-peak sum over-states D1's KV cost on every class |
| P15 | Reserved over start after the run, against the profiled peak (GiB): A5000 0.477 / 0.339 shipped, 0.757 / 0.425 at 4,096; 5090 0.453 / 0.339, 0.786 / 0.425; 5080 offload 0.755 / 0.425; 16 GB lock 0.661–0.776 / 0.563; 4090 0.937–1.245 / 0.974; 3090 0.943–1.261 / 0.974 | held, and the margin is thin on the 16 GB lock: its excess at 2,048 passes (0.21 GiB) against the 10% margin (≈ 0.25 GiB). D10's fallback buffer must not add to it |

**§1.1 restated from the bar cards** (the base is each card's measured probe: 5090 the all-4,096 arm, 5080 the natural
fit, which has D1 already; items from their nsys: D2 to FreeToken's per-token GEMM rate at the same call size, the low
end at equal I/O, P8; D3 59% of the router, the A5000's ratio; D4 the SwiGLU pass, an upper bound; D5 on the 5080 the
5090's and the A5000's measured dense ratios, 0.74–0.82; D6 FreeToken's own 8,192-chunk GEMM gain):

| Row (bar) | D1–D3 + D5 | + D4 (the set) | + D6 | ≤ bar |
|---|---|---|---|---|
| 5090 resident 1K (≥ 0.95×) | 0.983–0.996× | 1.010–1.025× | — | — |
| 5090 resident 4K (≥ 1.0×) | **0.966–0.983×** | 1.008–1.027× | — | without D4; within 3% with it |
| 5090 resident 32K, gated (≥ 1.0×) | **0.999**–1.007× | 1.018–1.026× | 0.964–0.971× (the gated arm then runs 8,192 too: the default's) | within 3% |
| 5090 resident 32K, default (reported) | 0.931–0.939× | 0.949–0.957× | 0.964–0.971× | — |
| 5080 offload 4K (≥ 1.0×) | 1.101–1.135× | 1.152–1.189× | — | — |
| 5080 offload 32K, gated (≥ 1.0×) | **0.989–1.002×** | 1.008–1.021× | not built on this lock (O1) | within 3% |
| 5080 offload 32K, default (reported; the 0.88× reference, measured 0.873×) | 0.943–0.955× | 0.961–0.974× | 0.977–0.990× at ≈ −2.4% decode (O1) | — |
| 5080 offload 1K (≥ 1.5×) | 1.944–1.966× | 1.976–1.999× | — | — |

- **No decision rule tripped.** D7's rejection stands at the locks' pass size on both bar cards (P7), and no gated
  cell falls below its bar with the full set. Three cells sit within 3% of their bars (5090 4K and 32K gated, 5080
  32K gated): ≥ 5 rounds at step 4. Without D4 the 5090 4K row is under its bar and both 32K gated rows reach it only
  at their high ends; D4's saving is an upper bound, so these go to the owner (O2) with the measured gated arm.
- **The gated reference's distance:** FreeToken's gated arm is 4.9% slower than the arm that recorded 0.88× on the
  5080 (7.2% on the 5090); Holon's natural fit stands at 0.916× against it today.
- **Changed risks.** The 32K default rows lost the attention lead (P10); the 5080 4K row is compute-bound for Holon,
  so D2–D5 pay there in full (P11); D1's KV cost is a fraction of §6's (P12–P14): no class needs `ggml_cuda.pass_rows`
  below its step budget (O3: 2,048 on the 2,048-token locks, 4,096 elsewhere).

**Restated from the step-2 base** (the panel's C1, written before step 3's runs). The base is step 2's measured time
to the first token (§7.1); the remaining items' savings are the nsys terms above: D3 59% of the router, D4 the SwiGLU
pass (an upper bound), D5 the dense GEMM and q8_1 terms at 512 against 4,096 rows (5090: the shipped arm's against the
probe's, 1.3–1.4 / 9.5–10.3 / 76.5–83.6 ms; 5080: as above).

| Row (bar) | Step 2, measured | + D3, D4, D5 (predicted) | ≤ bar |
|---|---|---|---|
| 5090 resident 1K (≥ 0.95×) | 0.923× | 0.992–0.994× | — |
| 5090 resident 4K (≥ 1.0×) | 0.880× | **0.988–0.993×** | **under the bar with the full set (O2)** |
| 5090 resident 32K, gated (≥ 1.0×) | 0.955× | 1.006–1.009× | within 3% |
| 5090 resident 32K, default (reported) | 0.891× | 0.939–0.941× | — |
| 5080 offload 4K (≥ 1.0×) | 1.036× | 1.135–1.152× | — |
| 5080 offload 32K, gated (≥ 1.0×) | 0.963× | 1.002–1.008× | within 3% |
| 5080 offload 32K, default (reported) | 0.918× | 0.955–0.961× | — |
| 5080 offload 1K (≥ 1.5×) | 1.816× | 1.872–1.876× | — |

- D2 realised 80–90% of its band (§7.1, step 2), so the 5090 4K row now falls short even with D4's upper bound: it is
  reported to the owner (O2) and does not block step 3.

## 3. Decisions

- **D0 — One source for the pass size: the bound kernel's `pass_rows`** (lead decision; B-M1, B-M3, C-B3).
  - **What it is.** An optional attribute of a bound `linear_experts@2` / `linear_experts_arena@1` instance: the most
    rows per call the instance gives at the bits of its `CHUNK`-row calls (§4 states the promise). The default is
    `CHUNK`.
  - **Who sets it.** The provider sets it per type, arch and path, from a pinned **proven table** in `plugin.py`
    (`PASS_PROVEN`: (SM, type, path) → proven, like `CROSSOVER`; B2-N3).
  - **The cap.** `ggml_cuda.pass_rows` (with `max_step_tokens`) caps every large call kernels.ggml_cuda makes: mmq
    with ids (this item), dense `qlinear` mmq (D5's `call_rows`) and the router GEMM's pass (D3′).
    - The shared `holon_testing` helper iterates that table, so an entry exists only where its L0 runs. A pair not in
      the table keeps `CHUNK`.
    - For a proven mmq-with-ids pair it is `max(CHUNK, min(PASS_ROWS, ID_PAIRS ÷ top_k, its cap) ÷ CHUNK × CHUNK)`.
    - FN's IQ1_M fallback, if it is ever needed, sets its own (D9).
  - **The reference provider** (`testing`'s CPU `LinearExperts`) loops per `CHUNK` inside a call, which satisfies
    §4's promise exactly, so it declares `pass_rows` too (B2-N4). CPU tests keep exercising large and multi-chunk
    passes; offload on the CPU equals resident on the CPU.
  - **Its cap** is `max_step_tokens` from the build context (missing: no cap), and below that the provider's plugin
    option `ggml_cuda.pass_rows`. That option is declared in kernels.ggml_cuda's manifest `[options]`, with choices
    512 / 2,048 / 4,096 / 8,192 (B2-N2: `check_locks.py` refuses an undeclared option).
    - That option is a lock setting, written by hand through `propagate` (a checked settings change, as
      `engine:max_step_tokens` diffs are reported today; `tools/tests/test_propagate.py:197`). tune does not choose it
      (B-M4: tune benchmarks decode only).
    - Per class it is capped so the startup KV pool stays ≥ `max_model_len` and the lock's recorded capability:
      2,048 where 4,096 would break it, `CHUNK` where 2,048 would (§6, step 0).
  - **Sizes follow it.** The provider sizes its workspaces for `pass_rows`, so a 2,048-token class sizes for 2,048.
- **D1 — Both stores take passes of the bound kernels' `pass_rows`.**
  - **Resident.** `Resident.pass_tokens` gives min(gate/up's, down's `pass_rows`) on steps of ≥ `CHUNK` tokens, else
    `CHUNK`.
  - **Offload.** The offload store takes the same min in place of its `PASS` constant (B-m2: one rule), under its
    unchanged `large_passes` condition (`tiered/store.py:522-528`). On its one lock this stays 4,096.
  - **Why it is bitwise.**
    - `pass_rows` is a multiple of `CHUNK`, so router chunks never move. That holds on verify steps longer than
      `max_step_tokens` too (`executor.py:139`), and a step's last partial chunk stays a pass of its own
      (`units.py:271-272`).
    - mmq with ids always runs one block per tile (`mmq.cuh:1482`: K is never split), and its per-element arithmetic
      is the same expression at every J and tile position (A: `mmq-vec-dot.cuh:313-440`; ampere and blackwell configs
      share `I` = 128, `nthreads` 256 and `MMQ_ITER_K`).
    - The q8_1 / W4A4 activation quantization is row-local (`quantize.cu:164-207`).
    - All three id helpers order rows slot-ascending, then by token (`mmid.cu:9-20`). D1 moves resident calls from
      the per-slot helper to the compact and device-wide ones.
  - **The oracle** (`std.moe.reference.linear_experts`) multiplies per expert batch, so it is not a provider of large
    passes. `testing`'s CPU provider loops per `CHUNK` and declares `pass_rows` (D0). CPU tests keep their bits.
  - *Why:* this is the gap's largest row (−449 of 1,176 ms at 4K).
- **D2 — A map-free compact grouped launch** (Holon's own `vendor.patch` code; FreeToken's `mmq_flat_tile` shape was
  studied, not copied; A-3, lead decision).
  - **J against the mean.** J is chosen against the mean rows per launch entry, T·k ÷ n_z, with `n_z` from
    `ggml_cuda_mm_ids_nz`. It is not the upstream RDNA formula's ÷ ne02, which for an arena is the slot count
    (6,160 at the gate) and would give J = 8 on the offload rows.
    - If step 0's batched-decode sweep (`gemm.py` at T = 16–64 on the 5090) shows hot experts re-reading their
      weights across extra column tiles (C-M6), J stays against T for calls of ≤ 64 rows. The bits are the same
      either way.
  - **Grid:** nty × (n_z + ⌊T·k ÷ J⌋ + 1). The bound is fixed per T, so a captured bucket replays any routing.
  - **Block decode.** Block b finds its entry z by binary search over `start(z) = z + ⌊expert_bounds[z] ÷ J⌋` (n_z + 1
    bounds already in the workspace); its tile is jt = b − start(z). It exits when jt·J ≥ the entry's rows. Because
    ⌈c ÷ J⌉ ≤ 1 + ⌊b₁ ÷ J⌋ − ⌊b₀ ÷ J⌋, every tile is covered.
  - **Fixup.** `fixup_needed` is gated to `!ids_dst` and asserted, so the ids path never adds `tmp_fixup` (A-3: a
    grid ≠ ntiles would otherwise trigger it).
  - **Where it lives.** One Holon `__device__` function replaces the three copies of the tile decode in `mul_mat_q`
    (`mmq.cuh` ≈ :1024, :1114, :1205), plus the `J_best` line, the grid line and the fixup gate. **No helper changes
    and no new memory** (A-3).
  - **On sm_120** (PDL launches, `common.cuh:1720-1740`) the `expert_bounds` reads stay after `ggml_cuda_pdl_sync`;
    memcheck and racecheck run there.
  - **`ERR_COLUMNS` keeps its meaning** (an id repeated in a row). Such rows may now be computed, but they stay
    flagged and their outputs stay unspecified.
  - *Why:* this is row 2: −70…−85 ms at 4K, −577…−694 at 32K. The bits do not move: the arithmetic is J-independent
    (D1's argument), and the launch only changes which block computes a tile.
  - It changes batched decode above 8 rows (J 64 → 8 at 64 rows). The bits are kept and the speed is measured
    (acceptance 5).
- **D3 — `moe_router@1` serves prompt chunks of ≥ `CHUNK` rows** (kernels.ggml_cuda, bounded; A-1). **Reverted at step
  3** by M0.7's keep rule (§7.1, §10): the router stays bitwise main's.
  - **Path.** Calls of ≥ 512 rows run one launch: a tiled f32 GEMM reading bf16 x directly (no f32 copy), with f32
    fma chains from x's exact values in an order the kernel fixes, independent of the row count, then `moe_topk@1`'s
    selection. Calls of 17–511 rows keep today's cast + f32 linear + `moe_topk`; calls of ≤ 16 rows keep the fused
    decode path.
  - **Decode keeps its bits.** Batched decode (17–64 rows) and verify keep main's router bits. A mixed step's rows
    already take their call's path (`chunk_with_decode`), as today.
  - **What it replaces:** cast + cuBLAS SGEMM + split-K reduce + top-k, 4 launches and 22 ms at 4K on the A5000.
  - **Kept** by M0.7's rule: ≥ max(0.5%, 3 sd) on some prefill row of a lock class.
- **D3′ — the router per pass, not per chunk** (step 3d, the lead's decision after D3's revert; bounded as D3).
  - D3's router GEMM is row-invariant, so one launch per expert pass gives every row its chunk's bits. kernels.ggml_cuda's
    `moe_router@1` instance declares `pass_rows` (D0's cap; `CHUNK` where the GEMM does not serve the stack's
    experts), and std.moe routes a pass of whole `CHUNK`s in one call where the router declares it (a contract term
    of `moe_router@1`, as `linear_experts@2`'s `pass_rows`); the last partial chunk is a pass of its own.
  - Keep rule: M0.7's, again, on both lock classes (§7.1, step 3d). D4's fused q8_1 quantize of down's input is
    optional, taken only if D3′ leaves the 5090 4K row under 1.0×, and bitwise.

- **D4 — SwiGLU in gate/up's mmq store** (bitwise; **in the set**, lead decision C-B1).
  - **No repack.** The offload store refuses kernels that repack (`offload.py:164-168`). Instead a tile of I/2 output
    rows loads gate rows i… and up rows inter + i… from two base offsets of the stored layout.
  - **Fused write-back.** For one output the g and u values sit 64 rows apart, in different warps' fragments, so the
    write-back exchanges them through shared memory. It applies each half's per-expert `scale2` (NVFP4, gate's then
    up's) with `swiglu_f32`'s plain fast-math `*` (`csrc/api.cu:114-128`, flush-to-zero), not `mmq_store`'s
    `ggml_cuda_mul_rn`. It writes `silu(g)·u` as [T, k, inter].
  - **Untouched:** arenas, F5b's CPU reader and mmvq's fused gate.
  - **Applies where `inter % 64 == 0`** (the half-tile split: 768 and FN's 640 both pass). Other shapes keep the
    unfused store and `swiglu_f32` (A2).
  - **Saving:** an upper bound of −21 ms at 4K and −166 at 32K, assuming the fused write-back is free. It does not
    lower the live memory peak (§6).
- **D5 — Dense `qlinear` mmq calls of `pass_rows` rows** (bounded).
  - **Bits move.** On the A5000, 512-row calls of the 30B's dense shapes tile at ≈ 83% efficiency, so they run
    stream-k; 4,096-row calls tile at 100%, so they don't. Bits move, so D5 is bounded unless L0 shows a class where
    both sizes tile.
  - **Measured:** −25 ms at 4K and −188 at 32K. Kept by M0.7's rule.
  - **Workspaces** grow with the call: ≈ +25 MB on the 30B and ≈ +66 MB on the 8B locks at 4,096 (§6).
- **D6 — 8,192-token passes** (not in the default set; §1.1 and §9 O1).
  - **Mechanism.** `ID_PAIRS` and the vendored `GGML_CUDA_MM_IDS_MAX_PAIRS` are raised statically to 65,536 (the
    device-wide helper path, its L0 extended). A class gets 8,192 only through its lock's `max_step_tokens` and
    `ggml_cuda.pass_rows`, set by hand through `propagate`.
  - **Cost:** +0.47 GB of live peak (§6). On the 5080 offload lock that is ≈ 166 cached experts (share 52.7% →
    ≈ 50.0%). x4_gate's measured share-to-decode slope predicts ≈ −3.0% / −3.7% / −2.4% decode at 1K / 4K / 32K (C2-m1):
    the FreeToken decode rows go 1.06× → ≈ 1.02× (4K) and 1.03× → ≈ 1.005× (32K).
  - **Taking D6 on the 5080 offload lock waives acceptance 5's decode tolerance there.** That is the trade O1 asks
    the owner to make.
  - **Batched locks:** an 8K prompt step also stalls decode twice as long (inter-token gap p99), since
    `chunk_with_decode` is 0.
  - **Default:** no D6 on offload locks; decode wins (lead decision).
- **D7 — Not taken (recorded):**
  - **Dequant to bf16 + a tensor-core grouped GEMM** (cuBLAS / Triton / CUTLASS). Rejected on achieved rates
    (C-M2), not peaks. On the A5000:
    - mmq reaches ≈ 22–27% of the ≈ 222 TOPS int8 peak (48–61 TOPS), and cuBLAS bf16 ≈ 78% (80–90 TFLOPS);
    - dequant costs 2.5 ms per layer pass and 1.2 GB of scratch;
    - so at 4,096 tokens dequant + bf16 loses, but at 8,192-token passes it would win (7.3 + 2.5 against 10.9 ms).
  - **On the GeForce lock classes** bf16 with f32 accumulation runs at 1/4 of the int8 peak (GA102 / AD102; GB20x
    to be confirmed). Re-scaled at the achieved rates, mmq's GEMM advantage there is ≈ 1.4×, and dequant takes
    ≈ 1.1 ms per layer on the 5090 and ≈ 2.0 ms on the 5080.
  - **Step 0 decides it per card** (5090 and 5080): mmq TOPS against cuBLAS bf16 TFLOPS at T·k = 32K and 64K for
    both stacks, dequant ms per layer, and the break-even pass size. The rejection stands only where mmq wins at the
    lock's pass size.
  - **Marlin-style W4A16:** the same rate argument.
  - **bf16 expert intermediates** (FreeToken's; row 7): bounded, and against m02 D3's f32 expert path. Recorded as
    Holon's numerics handicap (acceptance 3), §9 O4.
  - **Cross-reference, FN's IQ1_M:** FN's D1′ takes a native mmq tile loader for IQ1_M, which this track builds
    (D9). A dequant path is only D9's fallback, behind D0's seam with its own `pass_rows`, so this rejection is
    unaffected.
- **D8 — Ownership** (B-M5; FN design rev 2, `32c99f2`, its D1′).
  - This track owns `LinearExperts`' prefill dispatch and `vendor.patch`. Everything FN needs lands **before step
    5's single propagation round**, so FN's step 4 finds it on main:
    - D0's seam;
    - D2;
    - D9 with Q2_0's `TYPES` entry;
    - D10;
    - D11.
  - Every semantic hunk carries a "Holon:" marker; signature plumbing (parameters threaded through calls and
    signatures without changing behaviour) is named in each patched file's header note instead (the step-3 panel's
    B). `test_link`'s marker check (`test_ggml_cuda_moe.py:234-235`) adds `mmq.cu` and `mmq.cuh` (B-m7).

- **D9 — IQ1_M's native mmq tile loader** (a `vendor.patch` hunk for FN; bitwise in q8_1's class).
  - **Why it fits int8.** IQ1_M weights scaled by 8 are 8·g ± 1 ∈ {±1, ±7, ±9} (the grid nibble is in {0, 1, 2}, and
    delta is −1 ± 0.125, `dequantize.cuh:397-403`), so they fit int8. Their per-16 scales are d·(2s + 1) ÷ 8, the same
    layout family as IQ2_XS and IQ2_S, whose mmq loaders it follows.
  - **Exact in q8_1's class** (A2):
    - the scale is exact in f32 (fp16 d, at most 4 more bits, a power-of-two divide);
    - a 16-value int32 sum is ≤ 16·9·127, so it cannot overflow;
    - accumulation follows the J- and position-independent `q8_0_16` MMA pattern;
    - there is no stored-sum term (IQ1_M stays out of `_MMQ_SUMS` / `_MMVQ_SUMS`, as its mmvq uses the quantized
      `sumy`, `vecdotq.cuh:1311-1315`).
  - **The hunk**, all marked "Holon:" (FN rev 2's review; A2):
    - IQ1_M → `MMQ_Q8_1_DS_LAYOUT_D4` in `mmq_get_q8_1_ds_layout` (its default aborts);
    - the MMA vec-dot instance and the dp4a one (`mmq.cuh` ≈ :653, :786);
    - IQ1_M in `ggml_cuda_should_use_mmq` and in the ampere config table, which blackwell falls through to;
    - the template instance file;
    - **the condition: the per-16 scale is stored as an f32 `x_df`** (the IQ2_XS / IQ2_S layout with D4
      activations). A half2 `x_dm`, upstream IQ1_S's DS4 style, would need up to 15 significant bits and has 11, so
      it would round.
  - **The vendored ggml lacks it today:** it has IQ1_M's mmvq but no mmq instance (FN `v1-flash-next.md:73`).
  - **Fallback**, if the loader fails its L0 or its speed: a grouped dequant of 16 experts per launch into bf16 tiles,
    behind D0's seam with that path's own `pass_rows`. It is bounded, judged against its own floor.
  - **Ordering** (lead decision; A2-major = B2-N1). D9 and Q2_0's `TYPES` entry land at step 3c, **before step 5**.
    A lock's `src` hashes every file of the plugin (`plans.py:41-52`), so landing them after step 5 would leave all
    18 classes pinning other code and serving defaults (`cli.py:167`). Before step 5, the single round covers them;
    they move no snapshot (no lock uses IQ1_M or Q2_0).

- **D10 — Passed buffers for `linear_experts@2` / `linear_experts_arena@1`** (std minor; FN's request).
  - **Today** the scratch is `torch.empty` inside kernels.ggml_cuda (`plugin.py:464`, `:481`). Large passes on FN's
    shape need dense [rows, N] buffers of 0.9–1.8 GB, which std.moe must be able to place in lent memory (D11).
  - **Who accepts them** (B3-2). Only an instance that defines `scratch_bytes(tokens)` accepts the optional keywords
    `out` (the [T, k, out_y] f32 result) and `scratch` (a byte buffer of at least `scratch_bytes(tokens)`). std calls
    every other instance without them, as it does for `add` and `route_prepare` (`units.py:235`, `:262`). So
    `kernels.triton_experts` and third-party providers are unaffected.
  - **What `scratch_bytes` covers, per path** (B3-4):
    - mmq with ids: gate/up's f32 intermediate (none where D4's fused store runs). The call's q8_1 rows and id maps
      stay in the persistent named id workspace at `pass_rows` (§10 L3-2b: built first as decode-size, which cost the
      5080 lock 0.2% of its 4K prefill);
    - mmvq calls (≤ `mmvq_tokens` rows): 0. They skip D10's checks and carving (an eager small step is host-bound).
    - An mmq call without `scratch` allocates its own transient, as today, so nothing is refused and today's callers
      keep working.
  - **Contract terms** (B3-3, A3):
    - the provider keeps no reference to `out` or `scratch` past the call;
    - both are read and written on the compute stream, ordered with the work enqueued before the call;
    - one `scratch` may be reused by the successive calls of a pass (gate/up, then down);
    - neither `out` nor `scratch` may alias `x`, nor each other, and `scratch` must be 256-byte aligned (the
      device-wide helper asserts it, `mmid.cu:319`). Where the call uses the buffer, a violation is refused before
      launch (the mmvq-only path uses none and skips the checks).
    - Under `ring_copy = dma`, the refill of returned slots is ordered against these buffers exactly as X's lent ring
      slots are: after the step's compute stream.
    - **Lent memory is eager-only.** Graphed steps never borrow (`store.py:532`), so a buffer passed inside a captured
      graph is a static allocation whose address the graph keeps.
  - **Results are bitwise the same** with or without passed buffers, wherever those buffers live.
  - **When there is no lent buffer** (A3, A4; C4-m):
    - X's `Lent` does not always stage when lending falls short. If the ring falls short but every class's lent slots
      reach `prefill`, the step stages into them. If some class's do not, it sets `SHORT` and `after_commit` raises
      (`ring.py:44-56`, `store.py:482-488`).
    - D11's bind check keeps tail mode away from that raise.
    - **Where no tail is lent** (`coldest`, or the resident store), `std.moe` allocates the pass's buffer itself (C5).
      Gate/up and down are separate instances, so neither can own a per-pass buffer:
      - the buffer is sized to the larger of the two instances' `scratch_bytes(pass_rows)`;
      - it is passed as `scratch` to both calls, gate/up then down;
      - an instance allocates its own transients only for callers that pass nothing.
    - The startup profile's worst-case step runs that fallback, so the KV pool is sized for its peak. Step 0 checks
      that the reserved memory after a 32K prefill is ≤ the profiled peak.
  - Today's callers pass nothing, so no snapshot moves.

- **D11 — Contiguous-tail lending, through the policy contract** (a std minor; FN's request; FreeToken / Strata
  style; lead decision on A3-1, A3-3 and B3-1).
  - **The contract.** `moe_resolve@1`'s variant contract gains an optional `lend_tail(class, n)` term beside `lend`.
    The policy, not the store, picks: its instance returns the experts holding any of class c's last n slots
    (`kernels.py:88-95` keeps `lend`'s terms otherwise).
  - **What the store does with the pick:**
    - **Whole experts** (A3-1). An expert holding any tail slot is evicted in every role. Its slots in other classes
      join the step's lent set there, which is not contiguous.
    - **Empty slots in the range** sitting in a layer's empty stack are taken out as the whole stack entry, all
      roles, since `empty [G, R, E]` shares one count per layer (`store.py:212-213`).
    - **Spares in the range** are released by the policy first: promote's spares hold an evicted owner's bytes
      (`promote/reference.py:21-27`). A spare that is lent stops being one, and its owner's next promotion fetches.
    - **The return.** At the step's end every slot returns to its layer as a whole empty entry (D4's R0), as X's
      lending does. The records are `(g, −1, item)` per evicted expert, as `lend`'s.
  - **Which class arena** (FN rev 3 panel A, F3). The tail is lent from the largest class arena, i.e. the gate/up
    class of the layers' most common gate/up type; FN's files mix gate/up types per layer. Ties go to the lower class
    index. The rule is the store's, stated in the contract text, so every policy lends from the same arena.
  - **One owner per step for the D10 bytes** (lead decision; A3-2).
    - The tail range handed to D10 as `out` / `scratch` is excluded, for the whole step, from `Lent`'s `pool`, from
      the `stage_tab`, and from the empty stacks (`ring.py:60-63`, `:80-91`).
    - The ring and the staging take their slots from the rest of the lent set: the other classes' slots of the
      evicted experts, then the empties outside the range, then what step 2 of the order below lends.
    - A CPU and L0 assertion checks per step that no pool, stage or ring slot, and no mapped id, falls in the D10
      range.
  - **The order of a tail step** (lead decision on A4-major = C4-major; `ring.py:42-56`, `:101-102`):
    1. `lend_tail(class, n)` first, which evicts the range's holders whole.
    2. Then X's `lend(need)` rounds, unchanged, for every class's remaining ring and stage need, the tail's class
       included. After step 1 no resident holds a range slot, so the policy's coldest picks fall outside the range.
    - **What it fixes.** Without step 2, the tail's class would get no ring or stage slots except the empties outside
      the range. A short step would then set `SHORT` and raise instead of staging.
    - **Counting.** `held` and `want` for the tail's class count only slots outside the range. Counting the range
      would let `ring_ok` pass while `pool` holds −1 entries, and the ring would read them.
    - lfu's replay models both calls in this order.
    - **Bookkeeping** (A5). `lend_tail` takes its own call number. Otherwise `_give_back`, which sorts the lent slots
      by `lent_call · E + id` against a `ROUNDS · E` sentinel (`ring.py:147-152`), would interleave the tail's slots
      with the first round's and corrupt the empty stacks.
      - `lend_tail` is call 0, and the coldest rounds become calls 1…`ROUNDS` (in `coldest` mode, call 0 lends
        nothing).
      - `lend_needs` is sized `ROUNDS + 1` (`store.py:227`), and the sentinel becomes `(ROUNDS + 1) · E`.
      - The `lends` telemetry record (`store.py:240`, `:501`) gains row 0: the tail's need, n at its class and 0
        elsewhere. lfu's replay (`lfu/reference.py:168-170`) and `placement.parity` (`placement.py:85`) read that
        shape, replaying row 0 as `lend_tail` when it is non-zero.
  - **The bind check** (the fallback, lead pick).
    - The rule: at bind, and at each materialize where X decides the layout's lending, `moe.ring_lend = tail` is
      refused when the tail class's lendable cache slots outside the range (its arena less the null, staging and
      range slots) cannot reach `prefill`.
    - The range is sized from D10's `scratch_bytes(pass_rows)` and `out` at the bound configuration.
    - So a tail step can always stage, and `SHORT` is unreachable in tail mode.
  - **Who may lend a tail.** `placement.lfu` implements `lend_tail` (FN's lock). `moe.ring_lend = tail` is refused at
    bind under a policy that does not declare it, so lru, promote and profile_decay refuse until they implement it,
    with their own reference and replay.
  - **Mode.** `moe.ring_lend` = `coldest` (default, X's behaviour) | `tail`, declared in std.moe's `[options]`.
  - **Exactness.** Every placement is exact (X's acceptance 2), so tail lending changes speed, not outputs.
  - **Checks:**
    - a CPU check of a tail step on a cache with **no spare empties**, at FN's E = 512:
      - `lend_tail` as call 0, then the `lend` rounds;
      - `held` excludes the range;
      - the pool holds no −1 entries in use, and there is no `SHORT`;
      - after the give-back, the empty stacks and `n_empty` equal the replay's (A5);
    - the bind-time refusal on a configuration whose tail class cannot reach `prefill` outside the range;
    - **policy parity** with `tail` for every policy that accepts it: lfu's replay models `lend_tail`, the whole-expert
      eviction, the spare release (none in lfu) and the whole-entry return;
    - a **tail-path plant**, `tail_unmap_undone` (the tail's unmap undone, so an evicted expert stays mapped to a slot
      D10 writes), with its fired count asserted, predicted to fail parity and the `prompt_logprobs` identity.
      `lent-slot-still-mapped` patches `TieredStore.lend`, which the tail path never calls (`planted.py:1051-1066`),
      so it cannot stand in;
    - the `prompt_logprobs` identity (acceptance 4) runs with `tail` on FN's mode and the 30B offload lock's
      settings.
  - No current lock sets `tail`, so no snapshot moves.

## 4. Contracts and plugins (no core)

| Where | Today | Change | Class |
|---|---|---|---|
| `std.moe.kernels` `linear_experts@2` / `linear_experts_arena@1` text | silent on rows per call | **std minor** (additive optional term, `docs/design.md:338`; B-m1): an instance may expose `pass_rows` (int, a multiple of `CHUNK`, default `CHUNK`) | — |
| `std.moe` `Resident.pass_tokens` (`units.py:187`), `std.moe.offload` `pass_tokens` (`offload.py:119-121`) | `CHUNK`; offload `PASS` | min over the step's stacks' `pass_rows` (`PASS` retired) | bitwise (D1) |
| kernels.ggml_cuda `LinearExperts` (`plugin.py:405-480`) | chunk 4,096 for arenas, 512 otherwise | `pass_rows` per type / arch / path, capped by `max_step_tokens` and `ggml_cuda.pass_rows`; workspaces sized for it | bitwise (D0, D1) |
| kernels.ggml_cuda `vendor.patch` (`mmq.cu` J_best, `mmq.cuh` launch, the tile decode and the fixup gate) | J against T; grid nty × ⌈T ÷ J⌉ × n_z | J against T·k ÷ n_z; grid nty × (n_z + ⌊T·k ÷ J⌋ + 1); binary-search decode; no ids fixup | bitwise (D2) |
| the same, gate/up's store | f32 [T, k, 2·inter], then `swiglu_f32` | fused `silu(g)·u` [T, k, inter] | bitwise (D4) |
| the same, a new type's tile loader (IQ1_M: DS_LAYOUT_D4, the MMA and dp4a instances, `should_use_mmq`, the ampere config, an f32 `x_df`); `TYPES` gains iq1_m and q2_0 | IQ1_M: mmvq only | an mmq tile loader (×8 int8 values, per-16 f32 scales) | bitwise in q8_1's class (D9) |
| kernels.ggml_cuda manifest `[options]`, `plugin.py` `PASS_PROVEN` | none | `ggml_cuda.pass_rows` (512 / 2,048 / 4,096 / 8,192); the proven (SM, type, path) table | — (D0) |
| `linear_experts@2` / `linear_experts_arena@1` text; kernels.ggml_cuda's and `testing`'s instances | the instance allocates | **std minor:** an instance defining `scratch_bytes(tokens)` accepts `out` / `scratch`, with the lifetime, aliasing, alignment and stream terms of D10; the persistent id workspace shrinks to decode size | bitwise (D10) |
| `moe_resolve@1` variant contract; `std.tiered` lending and `Lent`; `placement.lfu`; std.moe manifest `[options]` | `lend(n)`: coldest whole experts | **std minor:** an optional `lend_tail(class, n)` term (whole experts in every role, spares released, whole-entry return, the largest-class rule); `Lent` keeps the D10 range out of pool, stage and empty stacks; lfu implements it; `moe.ring_lend` = `coldest` (default) / `tail`, refused under a policy without `lend_tail` | exact, as every placement (D11) |
| `testing`'s CPU `LinearExperts` | per expert batch | loops per `CHUNK` inside a call; declares `pass_rows` | bitwise against its `CHUNK` calls (D0) |
| kernels.ggml_cuda `MoeRouter` | > 16 rows: cast + f32 linear + `moe_topk` | ≥ `CHUNK` rows: its own f32 router GEMM + selection (reverted at step 3: as today) | bounded (D3); bitwise after the revert |
| kernels.ggml_cuda `QLinear` | mmq per 512 rows | per `pass_rows` | bounded (D5) |

**The `pass_rows` promise, exactly** (B-M2, lead decision):
- For every n ≥ 1 with n·`CHUNK` ≤ `pass_rows`, a call of n·`CHUNK` rows gives each row the bits that the `CHUNK`-row
  call of that row's own chunk gives it.
- Nothing is promised for a call that is not a whole number of `CHUNK`s (a ≤ `mmvq_tokens` call takes mmvq), nor
  across placements.
- Placement invariance stays the arena's separate term (plan §10, "Bitwise placement invariance…").
- Every declaring provider's L0 calls one shared helper in `holon_testing` (design §L1: declared props verified).

**Generality.** Nothing names a model: pass and call sizes are generic, the launch serves every ggml type with ids,
and the router kernel serves `moe_router@1`'s facts. `kernels.triton_experts` declares nothing (no lock uses it).

## 5. Numerics, judge and measurement

| Change | Class | Check | Predicted |
|---|---|---|---|
| D0 / D1 | bitwise | **end-to-end:** the `prompt_logprobs` hash of every prompt row at 1,027 and 4,096 tokens, main against the branch, on the resident 30B, natural-fit offload and capped (F5b) configurations, modelled on `judge/holon_judge/kernel_identity.py` (A-2). **Snapshots:** compared with main's without writing locks (B-m3). **CPU unit test:** records `LinearExperts` call sizes per pass at 600 / 1,027 / 4,100 tokens and fails when a tail merges (the `pass_eats_tail` mutation, A-2, B-m5). **L0:** see below | identical; the mutation fails |
| D2 | bitwise | L0 below. The output is pre-filled with NaN and passed to `lib.mmq_id` (as `gemm.py` does; `LinearExperts.__call__` allocates its own, A2). The L0 asserts with `launched()` that `mul_mat_q_stream_k_fixup` never launches on the ids path, replacing the fixup mutation, which may be a no-op (A2). The mutation "the decode skips each entry's last partial tile" leaves NaN rows. Also: graph replay of a captured bucket on new routing; graphs-vs-eager at batch 64; memcheck everywhere, racecheck on sm_120 | identical; the mutation fails; no fixup launch |
| D9 | bitwise in q8_1's class | L0 against `q8_1.linear_experts` for IQ1_M, every invariance test of §5.1 with iq1_m, memcheck on sm_120 | within q8_1's emulation; row-invariant |
| D10 | bitwise | L0: NaN-poisoned passed `out` / `scratch` against the allocating call; a short, misaligned or aliasing buffer refused before launch; an instance without `scratch_bytes` called without the keywords (std); graph replay with static buffers | identical |
| D11 | exact (placement) | CPU / L0: the tail range is contiguous and of the requested bytes; evicted experts are unmapped in every role; empties leave the stacks as whole entries and return as whole entries; per step no pool, stage or ring slot and no mapped id falls in the D10 range; `tail` refused at bind under lru / promote / profile_decay. Judge: policy parity with `tail` for every accepting policy (lfu today); the `prompt_logprobs` identity with `tail`; the plant `tail_unmap_undone` with its fired count asserted | identical; the plant fails parity and the identity |
| D4 | bitwise | L0: the fused output is bitwise `swiglu_f32` of the unfused output, for every glu type, with subnormal and NVFP4-scaled inputs | identical |
| D3 | bounded | L0 at 512–4,096 rows: logits within the f32 budget of the f64 product; selection bitwise `moe_topk`; row-invariant across ≥ 512-row calls. L2: Tier B prefill against the mmq floor. **Plants:** `router_prefill_bf16_logits` (logits rounded to bf16: near the bound) and `router_prefill_short_k` (hidden less its last 256: far) | passes; KL p99 at or under the floor (the decode router's precedent: 0.037 against 0.040). The near plant's verdict is predicted at step 3 from a CPU estimate before the GPU run; the far plant fails |
| D5 | bounded | L2 (Tier B prefill); `holon_judge bounded --chunked`. **Plants:** `qlinear_partials_bf16` (stream-k partials rounded to bf16 before the fixup: near) and `qlinear_large_no_fixup` (far). The fixup launches are counted so the record shows each plant fired on that card and shape (A-5) | passes; the far plant fails where it fires; the near plant's verdict is predicted as D3's |

### 5.1 L0 coverage and lean tests (lead decision; A-4, B-m4)

**Types and SMs** are derived from `plans/` at step 1 and recorded:
- sm_86 (3090): q4_k, q6_k;
- sm_89 (4090): q4_k, q6_k, mxfp4, nvfp4;
- sm_120 (5080 / 5090): q4_k, q5_k, q6_k, iq3_s, iq3_xxs, iq4_xs.

- **FN's requirements** (FN rev 2, D1′):
  - its types are IQ2_S, IQ2_XXS, IQ1_M (D9), Q2_0, IQ3_XXS, IQ3_S, IQ4_NL and IQ4_XS; IQ1_M and Q2_0 join `TYPES` in
    this track's step 3c;
  - its MoE shape is E = 512, top-10, H = 2,560, I = 640, which runs the device-wide helper. At top-10 `pass_rows` is
    3,072: ⌊32,768 ÷ 10⌋ rounded down to whole `CHUNK`s.
  - Both go into the same tests below, with no new suites.
- No lock sets `quant.act`, so W4A4 is not parametrized.

**Where the checks go, without new suites:**
- `test_linear_experts_every_type_at_prefill_chunks` (`test_ggml_cuda_moe.py:254-263`) gains a 4,608-row against
  512-row comparison (every type, small stack). It also gains one 128-expert stack (q4_k, q6_k) at T = 512 / 2,048 /
  4,096, which crosses all three helpers (A-minor).
- `test_an_arena_call_of_many_chunks…` stays at two types.
- `PLACED` (:417) is parametrized **per SM** from the derived list plus FN's types, not from their union (B2-N5).
  - On the sm_86 development card that is q4_k and q6_k plus FN's types.
  - The sm_89 / sm_120 nodes add their own lock types.
  - FN's shape is one more stack in the every-type test.
- D10's and D11's checks extend the existing arena and store tests (`test_ggml_cuda_moe.py`'s arena tests,
  `plugins/tests/test_moe_engine.py`'s lending tests). They are not new suites.
- **Step 1's named test edits** (B2-N4): `test_moe_engine.py:19` (imports `PASS`), `:357` and `:393` (assert
  `pass_tokens(10) == PASS`) assert against the bound kernels' `pass_rows` instead. The CPU call-size test uses the
  reference provider's `pass_rows` (4,096 by default), so it exercises D1's min and cap (A2).
- The device-wide helper is not added to `HELPERS` (it is asserted at :516).
- D3's router is tested at 512–4,096 rows.
- `pytest --durations` before and after is recorded in the step's record.

### 5.2 Measurement (lead decision; C-M5, C-M6, release.md)

- **Arms.** A same-session ABBA against main's build, with both engines timed at the same boundary: the time from
  submission to the first token, at the engine API.
- **Prompts.** A fixed set chosen before code: ≥ 5 prompts at 1K and 4K and ≥ 3 at 32K, the same for every engine.
- **Rounds.** ≥ 3 rounds, rising to ≥ 5 for rows predicted within 3% of their bar (§1.1); on the 5090, 6 balanced
  ABBA rounds for those rows (the step-3 panel's C).
- **Short kernels** (under ≈ 50 µs): timed inside a captured graph, or over ≥ 100 back-to-back launches, and checked
  against nsys device time (C).
- **Allocator work** (C): `prefill.py` records each timed prompt's `num_alloc_retries`, `num_device_alloc` and
  `num_device_free` deltas; steady state is zero on the final tree, the 16 GB lock included.
- **Statistic.** The median over rounds of each round's geomean of the paired ratios. A difference under 2 sd of the
  rounds is reported as "≈" and escalated.
- **Warm-up and outliers** (the panel's C2, from step 3 on; `prefill.py --warm-set`, `abrun.sh` / `abstat.py`):
  - each engine process first runs the prompt set's shapes once, untimed, on prompts of their own (FreeToken's radix
    cache would otherwise serve the timed prompts: step 4's first 5090 round read its 32K prompts back in 14 ms);
  - a repetition over 1.5× its cell's median (engine, length, all rounds) is flagged, and its round is re-run once;
    a cell flagged again is reported with the flag;
  - the median of all paired ratios is reported beside the statistic;
  - every record keeps each row's round index and the round's arm order.
- **Decode tolerance:** ≤ max(0.5%, 2 sd).
- **Comparisons against main** pin pool sizes, or record that no preemption or eviction occurred (A-minor: D1's KV
  shrink could change admission).

### 5.3 Locks (lead decision; A-6, B-M6, C-M7)

- **Lands first.** This track needs no core change. It closes with its own propagation round covering D1–D5, its L2
  verdicts recorded as its own evidence, before K or FN rebase onto it.
- **Scope:** the 18 classes that pin kernels.ggml_cuda or std.moe. Five do not (Qwen3-8B bf16 ×3, FP8 ×2).
  - D1 / D2 / D4 move no snapshot.
  - D5 moves every class whose `qlinear` runs mmq on prompt chunks (D3, which would have moved the 7 MoE classes, is
    reverted).
  - `ggml_cuda.pass_rows` and any `max_step_tokens` change are written by hand through `propagate` in the same round.
- **No interim writes.** Nothing propagates mid-track; identity is checked against main without writing locks.
- **The 5080 offload lock is re-filled and re-pinned** at the close (`8c5a826` pins its cache at the fill's counts
  with no slack, so D5's ≈ 25 MB alone would make its pinned arena refuse). The gate runs on the new lock.
- **Approval** is delegated for passing verdicts.

## 6. Memory (C-m1: the live peak, not a sum)

`LinearExperts.__call__`'s gate/up output is local to its call, so it is freed before down's output exists. The
expert path's live peak at T tokens per pass (k = 8, inter 768, hidden 2,048, f32) is:

max(gate/up out + SwiGLU out, SwiGLU out + down out) + the id workspaces (q8_1 of T·k rows at `in` 2,048 and 768,
and the id maps).

| T | gate/up + SwiGLU | SwiGLU + down | workspaces | **live peak** | over 512 |
|---|---|---|---|---|---|
| 512 (today) | 38 MB | 47 MB | 13 MB | **60 MB** | — |
| 2,048 | 151 MB | 184 MB | 52 MB | **236 MB** | +176 MB ≈ 1.8K KV tokens |
| 4,096 | 302 MB | 369 MB | 104 MB | **473 MB** | +413 MB ≈ 4.2K KV tokens |
| 8,192 (D6) | 604 MB | 738 MB | 208 MB | **946 MB** | +886 MB |

- **D4** removes the gate/up output, but the peak is the SwiGLU + down term, so the peak is unchanged.
- **D10** moves gate/up's f32 intermediate (where D4's store does not apply) to passed `scratch`; the id workspaces
  stay persistent at `pass_rows` (§10 L3-2b). The peak is unchanged.
- **D5** adds ≈ 25 MB (30B) or ≈ 66 MB (8B) of `qlinear` workspace at 4,096.
- **D3′**'s router transient grows with the pass: its partials [ks, T, E] f32 plus the logits [T, E], ≈ 34 MB on the
  30B (E = 128) and ≈ 42 MB at FN's E = 512, at 4,096 tokens (the step-3 panel's A). The 16 GB soak watches it
  (§7 step 4).
- **Per class** (96 KiB of KV per token on the 30B):
  - **16 GB RTX 5080, resident UD-IQ3_XXS** (`max_step_tokens` 2,048, `max_model_len` 16,384): M0.2 recorded 18.6K KV
    tokens, so −1.8K leaves ≈ 16.8K, barely above 16,384. Step 0 measures it; if it lands under 16,384 or under the
    recorded 16K-prompt capability, the class keeps `ggml_cuda.pass_rows` at 512.
  - **The 4090 MXFP4 / NVFP4 30B locks** (2,048): ≈ −1.8K each.
  - **24 GB 4,096-token locks** (30B Q4_K_M on 3090 / 4090 / 5090): −4.2K. The 4090's pool (≈ 26K tokens; release.md
    `OutOfUnits` 1,622 < 1,889) and the 3090's recorded 30,208-token long-prompt row make 4,096 doubtful there. Step 0
    measures one 24 GB lock, and the class takes 2,048 if 4,096 breaks a recorded row.
  - **5080 offload lock:** unchanged by D1 (it already passes 4,096). D5's workspace is covered by the re-fill. D6
    would take ≈ 166 cached experts.
- **Unexplained:** `peak_alloc_gib` rose 22.20 → 22.36–22.38 GiB in every 4,096 arm. That rise is not fully absorbed
  by the startup profile (C-m2). Step 0 explains it from startup KV tokens and the allocator's peak before any 16 GB
  class relies on §6.
- If K lands first and changes KV pools, acceptance 6's figures are re-measured after K (B-m8).

## 7. Steps

| Step | Work | Checks | Days |
|---|---|---|---|
| 0 | **A5000: done.** **Bar cards** (5090, 5080; one persistent container each): **FreeToken at 4,096 and 8,192 chunks, resident on the 5090 and offload on the 5080, at 4K and 32K** (the gated arm, measured: C2-N-M1); achieved mmq TOPS against cuBLAS bf16 TFLOPS at T·k = 32K / 64K, dequant ms per layer, the break-even pass size (D7); nsys breakdowns of resident (5090) and natural-fit offload (5080) at 1K / 4K / 32K, attention included; row 2's I/O split (`gemm.py --freetoken` with f32 activations, so its store is f32 too); the batched-decode J sweep at T = 16–64 (C-M6); startup KV tokens on the 16 GB lock and one 24 GB lock with the `--pass-tokens` / `--expert-rows` probes; the `peak_alloc` rise explained, and the reserved memory after a 32K prefill ≤ the profiled peak (C4-m); the fixed prompt set | every §1.1 cell restated per class from these, the gated cells from the measured FreeToken; the gated reference's distance from the 0.88× reference stated; rows at or below their bar to the owner before step 1 | 1.5–2 |
| 1 | D0 + D1 (`pass_rows`, the std minor, both stores, the shared `holon_testing` helper); `ggml_cuda.pass_rows` per class from step 0 | §5.1's L0; the CPU call-size test; the logits-hash identity on the A5000; snapshots against main (no lock writes); KV tokens per class | 1–1.5 |
| 2 | D2 (`vendor.patch`, `vendor.sh --check`, "Holon:" markers, `test_link`) | L0 + NaN poison + both mutations on sm_86 / 89 / 120, memcheck, racecheck on sm_120; per-call `gemm.py` at equal I/O dtype, interleaved repeats (C-m4), ≤ FreeToken's at 512–4,096 tokens; batched decode B16–64 ABBA | 2–3 |
| 3 | D4, then D3 and D5, each measured alone | D4's L0 and identity; D3 / D5 keep-or-revert by M0.7's rule, L2 with near and far plants (a passing near plant records L2's blindness at that scale) | 3–4 |
| 3b | FN's requests: D10 (passed buffers and their contract terms; the decode-size persistent workspace; the one-buffer allocating fallback) and D11 (`lend_tail` in the `moe_resolve` variant contract; the tail-then-coldest order with `held` / `want` excluding the range; lfu's implementation with its reference and replay of both calls; `Lent`'s D10 exclusion; the bind check; `moe.ring_lend`) | D10 / D11 L0 and CPU checks, the E = 512 no-spare-empties tail step included; policy parity with `tail`; the tail plant's fired count; the `prompt_logprobs` identity with `tail`; **perf with `tail` on the 30B offload lock at 4K and 32K** (C4): `ring_ok` true and no `SHORT`, prefill against `coldest`, and miss bytes per token over the first decode tokens against `coldest` | 3.5–4.5 |
| 3c | D9: IQ1_M's mmq hunk (or its fallback), with IQ1_M and Q2_0 in `TYPES` | D9's L0 on sm_86 / sm_120, memcheck; per-call speed against IQ2_S's loader | 2–3 |
| 3d | D3′ (the router per pass) and the step-3 panel's fixes | D3′'s keep rule on both lock classes; identity (D3′'s bits are per-chunk D3's); sm_120 L0 and identity at the final tree; allocator counters; the 5090's A/B against `9e23a71` | 1 |
| 4 | The bars on the bar cards (§5.2's method) | acceptance 1, 2, 3, 5, 6; on the 16 GB lock a soak (16K prompts, then 32-seat decode, repeated): no OOM and `num_alloc_retries` = 0, free memory at its worst point recorded; `ggml_cuda.pass_rows` drops to 512 there only if retries appear (C6) | 1–2 |
| 5 | Close: **one** propagation round over the 18 classes, covering D1–D5, D9 / Q2_0, D10, D11 and `pass_rows`; the 5080 offload lock re-filled and re-pinned; `pytest` on a GPU, `scripts/ci/run_all.sh`, panel code review, plan §10 status. FN's step 4 then finds D0, D2, D9, D10 and D11 on main | acceptance 4, 7; its L2 verdicts as its own evidence | 1.5–2 |

**Schedule:** about 15.5–22 working days, in the order 0 → 1 → 2 → 3 → 3b → 3c → 4 → 5. Steps 3b and 3c do not depend on
3 and may run beside it on another GPU. Nothing merges to main before step 5, so no lock is ever left pinning other
code.

The panel reviews this revision before step 1, the code at step 4, and again at the close.

### 7.1 Step records

**Step 3 predictions (written before each item's GPU run).**
- **D5, L2 on the A5000** (Tier B prefill: the mmq floor; `--max-step-tokens 4096 --max-seqs 4`, so dense calls of
  1–8 chunks):
  - D5 passes, within one SE of the pre-D5 tree's ΔPPL (`6b256ed`: KL p99 6.03e-2, ΔPPL +0.225%; budget KL p99 ≤ 1.18e-1,
    ΔPPL ≤ 0.573%).
  - On sm_86 only the 30B's `attn_k` / `attn_v` (out 512) take the stream-k fixup past one chunk, at 1,024 and 3,072
    rows (`mmq_schedule`); `attn_q`, `attn_output` and the head tile at every size. So the plants fire there only.
  - `qlinear-partials-bf16` (near): **passes**. A partial rounded to bf16 errs by ≤ 2⁻⁹ relative, about what the
    output's own bf16 rounding adds, on a fraction of the k/v rows: L2 is blind at that scale (recorded as such).
  - `qlinear-large-no-fixup` (far): **fails**. Every fixed-up tile loses the partials of the blocks before its end, so
    k / v of those rows are wrong.
- **D3, L2 on the A5000** (the same settings; the tree with D5):
  - D3 passes: the logits move by the f32 order only (≈ 1e-6 relative), which the bf16 floor cannot see.
  - `router-prefill-bf16-logits` (near), an analytic estimate in place of a CPU run: bf16 keeps 8 bits, so a logit
    moves by ≤ 2⁻⁹ relative (≈ 0.01 at the 30B's logit scale of a few units). The softmax weights then move by ≤ 1%,
    and the selection flips only where the 8th and 9th logits sit within that, swapping a low-weight expert. That is the
    decode router's precedent (its bf16-against-f32 KL p99 0.037 against the floor's 0.040): **passes**, L2 blind at
    that scale.
  - `router-prefill-short-k` (far): a logit loses an eighth of its terms, so most routes change: **fails**.

**Step 1 (D0, D1).**
- **What landed.**
  - `std.moe.kernels` states the `pass_rows` term and owns `CHUNK`.
  - `Resident` and `Offload` take the min of their stacks' kernels' `pass_rows` (`units.pass_rows`); `PASS` is
    retired.
  - kernels.ggml_cuda sets `pass_rows` from `PASS_PROVEN`, capped by `max_step_tokens` and `ggml_cuda.pass_rows`
    (declared in `[options]`), and sizes its workspaces for it.
  - testing's CPU `LinearExperts` runs per `CHUNK` and declares `pass_rows` (4,096 within the step budget).
  - `holon_testing.pass_rows.check_pass_rows` is the shared L0 helper.
- **`PASS_PROVEN`** is derived from `plans/` (the GGUF expert types of every lock that binds `linear_experts`):
  - sm_86: q4_k, q6_k;
  - sm_89: q4_k, q6_k, mxfp4, nvfp4;
  - sm_120: q4_k, q6_k, iq3_xxs, iq3_s, iq4_xs, plus §5.1's q5_k;
  - FN's six mmq types on every SM.
  - `PLACED` is that table's types on the running SM.
- **Decided while building.** A call's last partial chunk runs as a call of its own inside the instance too, so a
  row's kernel path is its chunk's whatever the caller cuts. Without this, a direct caller's 2,049-row call would put
  its last row on mmq, not mmvq.
- **Checks.**
  - L0 on sm_86: `test_ggml_cuda_moe.py` 124 passed in 177 s (main: 106 in 172 s; `PLACED` grew from 5 to 8 types).
  - Memcheck on sm_86 (now with 2,048 and 4,608 rows where `pass_rows` > 512): 0 errors.
  - The CPU call-size test (`test_a_prefill_step_passes_whole_chunks_then_its_tail`) fails on main's 512-token pass
    and on the `pass_eats_tail` mutation.
  - CPU suite `test_moe_engine.py`: 22 passed.
  - Static CI (`run_all.sh` against `b5b1e7f`) is clean.
  - Identity (`spikes/moe_prefill/identity.py`) on the A5000, the resident 30B and the natural-fit offload flags:
    main's and the branch's prompt-logprob hashes are identical at 1,027 and 4,096 tokens, each run-identical ×2.
    The branch's resident pass is 4,096 tokens, main's 512.
  - On the cluster (records `evidence/mp2_checks_<class>_2026-10-18.json`):
    - L0: sm_89 (RTX 4090) 135 passed; sm_120 (RTX 5090) 131 passed, and (RTX 5080) 131 passed.
    - Memcheck: 0 errors on both SMs.
    - Identity, main against the branch: resident on the 4090 and the 5090; the natural fit and the capped fit with
      F5b on the 5080. All identical. F5b's CPU share serves decode passes only, so the capped prompt rows are the
      GPU's, and they equal the natural fit's (placement-exact).
  - Startup KV tokens with the branch: 16 GB lock 18,560 (main 18,752); 4090 lock 25,776 (main 26,736); A5000 at
    33,792 / 1 seat 48,192 (main 50,272). The persistent id workspace sized for 4,096 rows is most of the A5000's
    difference; D10 moves it to transients (step 3b).

**Step 2 (D2).**
- **What landed** (a `vendor.patch` hunk, "Holon:" markers, `vendor.sh --check` clean):
  - `mmq_tile`, one `__device__` function, replaces the three tile decodes of `mul_mat_q`;
  - the stream-k work split counts the flat range;
  - `ggml_cuda_mul_mat_q` computes `n_z` first and picks J against T·k ÷ n_z;
  - the launch makes the flat grid nty × (n_z + ⌊T·k ÷ J⌋ + 1);
  - `fixup_needed` is gated to `!ids_dst` and asserted;
  - `test_link` checks the markers in `mmq.cu` / `mmq.cuh`.
- **C-M6 decided by the sweep.** J against the mean made the down stack up to 8% slower at 16 rows on the 5090 and the
  5080 (A/B against main's build). So, by the rule, **J stays against T for calls of ≤ 64 tokens**.
- **Decided while building.** With J against T a call's tokens fit one column tile, and there the flat range only adds
  blocks and a binary search: 2–7% slower per call than main on the A5000. **The flat range is used only where a call
  spans several column tiles** (⌈T ÷ J⌉ > 1, so ntx > 1, which the kernel reads); otherwise the launch is upstream's,
  one tile per entry. The bits are the same either way.
- **Checks.**
  - L0 (`test_ggml_cuda_moe.py` + `test_ggml_cuda.py`): sm_86 266 passed; sm_89 277; sm_120 279 (5090 and 5080).
    Suite time on sm_86: `test_ggml_cuda_moe.py` 126 passed in 252 s on a loaded host (step 1: 177 s; main: 172 s);
    the new test adds ≈ 15 s.
    The new `test_every_tile_of_every_entry_is_written` fills the output with NaN and calls `lib.mmq_id` at 9–4,096
    tokens over E = 128 and all three helpers, with uniform, same-8-experts and skewed routings. Every row is written
    and equals the 4,096-token call's row, and no fixup kernel is launched. The tile-width test's sweep follows D2's
    rule (J 16–128).
  - Mutations, on every SM:
    - "the decode skips each entry's last partial tile" fails 50 (sm_86), 62 (sm_89) and 56 (sm_120) tests;
    - a forced fixup on the ids path (`fixup_needed` true for every call with ids, its assert dropped) aborts the test
      process (SIGABRT) in the test's first call with ids (A4).
  - Memcheck: 0 errors on sm_86, sm_89 and sm_120. Racecheck on sm_120 (5090, 5080): 0 hazards.
  - Identity, main against the branch:
    - the A5000: resident and natural;
    - the 4090 and 5090: resident;
    - the 5080: natural and capped F5b.
    - All identical; graph replay (`test_graph_replay_follows_the_routing`, buckets to 256) passes.
  - Per call, D2 against main, interleaved ×3, layers 0 and 6, mean of the stacks' ratios:
    - ≤ 64 tokens: 0.99–1.02 on the A5000, 5090 and 5080;
    - 512 tokens: 0.65 / 0.67 / 0.67;
    - 4,096 tokens: 0.84 / 0.83 / 0.81.
  - Batched decode ABBA on the 5090 (`holon_tg` at depth 1,024, ×2 rounds of 3): B16 0.996×, B32 0.996×, B64 1.006×
    main's tokens/s, inside the run-to-run spread (±7%).
  - Per-step prefill timings: time to the first token, ms, medians of the fixed prompt set over 3 interleaved rounds
    (main, step 1, step 2). The ratios are each against main, and against FreeToken's default / gated arms of §2.2.

    | Card, configuration | main | step 1 (D0, D1) | step 2 (+ D2) | step 2 / FreeToken default (gated) |
    |---|---|---|---|---|
    | 5090 resident 1K | 80.5 | 61.9 (1.30×) | 54.8 (1.47×) | 0.92× |
    | 5090 resident 4K | 342.1 | 212.1 (1.61×) | 194.7 (1.76×) | 0.88× |
    | 5090 resident 32K | 4,548 | 3,534 (1.29×) | 3,392 (1.34×) | 0.89× (0.96×) |
    | 5080 natural fit 1K | 210.8 | 210.8 (1.00×) | 206.8 (1.02×) | 1.83× |
    | 5080 natural fit 4K | 408.5 | 408.5 (1.00×) | 367.2 (1.11×) | 1.04× |
    | 5080 natural fit 32K | 6,847 | 6,847 (1.00×) | 6,513 (1.05×) | 0.92× (0.96×) |
    | A5000 resident 1K / 4K / 32K | 281 / 1,207 / 14,032 | 217 / 704 / 10,168 | 193 / 641 / 9,654 | 0.92 / 0.91 / 0.95× |

    - D1 does not move the offload rows: the arena already took 4,096-row calls.
    - D2 realised 80–90% of §2.2's band at its low end: 5090 −17 / −142 ms at 4K / 32K against −19…−22 /
      −155…−179; 5080 −41 / −335 against −46…−52 / −368…−411.
    - The 5080 4K row already meets its bar.

**Step 3 (D5, D3, D4; 2026-10-19).**
- **What landed.**
  - D5 (`aded4cd`): `qlinear`'s mmq calls take whole `CHUNK`s up to `call_rows` (the step budget), its workspace sized
    for them; the judge's fixup plants (`qlinear-partials-bf16` near, `qlinear-large-no-fixup` far) through an mmq plant
    hook that counts the fixed-up tiles.
  - D3 (`0488063`): `moe_router@1` serves calls of ≥ `CHUNK` rows with its own router GEMM (f32 fma chains over fixed
    parts of hidden from x's bf16 values, the parts added in order: a row's logits do not depend on the call), then
    `moe_topk`'s selection; plants `router-prefill-bf16-logits` (near) and `router-prefill-short-k` (far).
  - D4 (`996eaf6`): SwiGLU in mmq's store for merged gate/up stacks where `mmq_glu` holds (the stack's rows in whole
    128-row tiles, MMA layouts): a tile loads gate rows in its upper half and the matching up rows (`split` rows on) in
    its lower half, the halves meet in shared memory, and the store applies `swiglu_f32`'s expression; an mmq
    instance of its own; every loader takes `split`. N5 plants the unfused path.
  - `e9048de`: B-dec's threshold (16; 64 on sm_89). The merges: step 3c (`2dc93fc`: IQ1_M's loader takes `split`),
    step 3b (`26fb532`: D10's `scratch` has no gate/up output where D4's store applies).
- **L2 on the A5000** (Tier B prefill floor, `--max-step-tokens 4096 --max-seqs 4`; the predictions above held):
  - D5 passes (ΔPPL +0.019…+0.371% against a ≤ 0.539% budget; KL p99 ≤ 9.75e-2). `qlinear-partials-bf16` passes
    (24 fixed-up calls planted): L2 is blind at that scale. `qlinear-large-no-fixup` fails (NaN logits in the chat
    class, top-1 87.0%).
  - D3 passes (ΔPPL +0.004…+0.370% against ≤ 0.538%). `router-prefill-bf16-logits` passes (3,408 chunks planted): L2
    is blind at that scale. `router-prefill-short-k` fails (top-1 93.4%, KL p99 0.957, |ΔPPL| 1.80%).
  - Bounded `--chunked 512`: the D5 tree and the integrated tree (`af3322d`) pass (top-1 98.46% / 98.35%, KL p99
    4.26e-2 / 4.58e-2).
- **Bitwise** (D4, and steps 3b / 3c on top): the `prompt_logprobs` hashes of the D3 tree (D4's parent), the D4 tree
  and the integrated tree are identical at 1,027 / 4,096 tokens, each run-identical ×2: sm_86 resident and natural;
  sm_120: the 5090 resident, the 5080 natural fit (at the A/B pin, below) and capped F5b. Against main they differ, as D3 / D5 are bounded.
- **L0 and sanitizers at the integrated tree** (`af3322d`):
  - sm_86: 295 passed, 8 skipped; memcheck 0 errors; the skip-tail mutation fails 62 tests, the forced fixup aborts.
  - sm_120: 5090 and 5080: 308 passed, 1 skipped (IQ1_M and Q2_0 included: B-dev5's sm_120 half holds); memcheck 0 errors; racecheck 0 hazards; the skip-tail mutation fails 68 tests, the forced fixup aborts.
- **Per call.**
  - D4, gate/up at layers 0 / 6, fused store over mmq + SwiGLU (5090): 0.999 at 64 tokens, 0.97 at 512, 0.93 at 1,024,
    0.88 at 2,048, 0.84–0.85 at 4,096 (the fused call is no slower than mmq alone); the A5000 0.95 at 4,096.
  - D3, the router at 512 rows (5090): E = 128, hidden 2,048: 28.4 → 17.3 µs; E = 512, hidden 2,560: 85.3 → 66.3 µs.
- **Time to the first token, each item alone** (§5.2's method with C2's warm-up; 3 interleaved rounds, arm order
  reversed on odd rounds; the fixed prompt set; no repetition flagged; each cell the median over rounds of the round
  geomeans, speed over the previous tree; sd over rounds):

  | Card, configuration | D5 (over step 2) | D3 (over D5) | D4 (over D3) | 3b + 3c (over D4) | all (over step 2) |
  |---|---|---|---|---|---|
  | 5090 resident 1K | 1.026 (0.001) | 1.003 (0.004) | 1.022 (0.003) | 0.999 (0.004) | 1.054 |
  | 5090 resident 4K | 1.054 (0.001) | 1.005 (0.004) | 1.052 (0.003) | 0.997 (0.003) | 1.114 |
  | 5090 resident 32K | 1.023 (0.003) | 1.002 (0.003) | 1.023 (0.004) | 0.998 (0.003) | 1.046 |
  | 5080 natural fit 1K | 1.000 (0.008) | 1.000 (0.002) | 1.003 (0.001) | 0.999 (0.000) | 1.002 |
  | 5080 natural fit 4K | 1.029 (0.001) | 1.004 (0.000) | 1.054 (0.001) | 0.998 (0.001) | 1.087 |
  | 5080 natural fit 32K | 1.013 (0.000) | 1.002 (0.000) | 1.023 (0.000) | 0.999 (0.000) | 1.037 |

  - Medians (ms), step 2 → the integrated tree: 5090 55.0 → 52.1, 196.0 → 176.2, 3,404 → 3,249; 5080 206.3 → 205.9,
    365.8 → 336.4, 6,502 → 6,269. The 5080 arms pin the re-fill's counts (below), the same for every tree.
  - **D5 is kept** (≥ max(0.5%, 3 sd) on every 4K / 32K row). **D3 fails M0.7's rule on every row of both lock
    classes** (best: +0.51% at 5090 4K with sd 0.42%; 5080 4K +0.42% with sd 0.04%, under 0.5%): by the rule it is
    reverted (`8f434fd`, §10 L3-1).
  - 3b + 3c cost 0.1–0.3% (consistent on the 5080, sd ≤ 0.1%). The cause was the host time of D10's checks on the
    eager small steps' mmvq calls, not the per-pass buffer (§10 L3-2).
  - Against step 0's FreeToken medians (a different session; step 4 measures both in one): 5090 1K 0.971×, 4K
    0.972×, 32K 0.997× gated (0.930× default); 5080 1K 1.82×, 4K 1.13×, 32K 1.000× gated (0.954× default). The 5090 4K
    row is under its bar with every item (O2, as C1 predicted), and both 32K gated rows sit at their bars.
- **The 5080 re-fill (C4; scratch, not committed).** The natural fill at the A/B's flags (`max_model_len` 33,792, one
  seat) gives gate/up / down Q6_K / down Q4_K cache slots: step 2 3,223 / 1,694 / 1,529; D5 3,217 / 1,688 / 1,529; D3
  and D4 3,229 / 1,700 / 1,529; the integrated tree 3,235 / 1,706 / 1,529 (52.65% of the experts, against the lock's
  3,236 / 1,618 / 1,618, 52.7%). D5's workspace costs 6 gate/up and 6 down slots; D3 returns 12 + 12, D10 6 + 6.
  The A/B pins the minimum, 3,217 / 1,688 / 1,529, which starts and steps on every tree.
- **Step 3b's perf check** (RTX 5080, the integrated tree at that pin; `spikes/moe_prefill/tail.py`, `coldest` and
  `tail` alternating ×2; 3 prompts after a warm-up, then 64 decode tokens): the tail is gate/up's last 252 slots.
  - Every borrowing step is a ring step with `ring_ok` (4K: 3 of 3; 32K: 24 of 24) and no `SHORT`, in both modes.
  - Prefill: 4K 336.4 ms both; 32K 6,268 ms (`tail`) against 6,276 (`coldest`).
  - Miss bytes per decode token: 4K 120.9 against 121.3 MB; 32K 57.5 against 58.6 MB.
- **Suite time** (acceptance 7; `test_ggml_cuda_moe.py` + `test_ggml_cuda.py` + `test_moe_engine.py`, sm_86, two
  interleaved runs each on the shared host): main 387.5 / 331.6 s (278 tests), the integrated tree 383.9 / 371.5 s
  (331 tests: the two new types, D2–D4's and 3b's tests). ≈ +5% at the medians, inside the host's run-to-run spread
  (56 s on main).

**Step 4 (the bars, 2026-10-19).** Records: `evidence/mp4_bars_rtx-5090_2026-10-19.json`,
`evidence/mp4_bars_rtx-5080_2026-10-19.json`. The step-4 tree is `22717b3` (the code of `fed13f1`: D0–D2, D3′, D4, D5,
D9–D11 and the panels' fixes); main is `b5b1e7f`.
- **Method** (§5.2): one engine process per arm and round; an untimed warm-up over the set's shapes on prompts of its
  own; the fixed prompt set (5 / 5 / 3 at 1K / 4K / 32K); 6 balanced rounds (ABCD, DCBA, …) of main, the step-4 tree,
  FreeToken default and FreeToken gated, all in one session; 60 s idle between arms; each arm's SM clock sampled while
  busy; real round indices. Step 0's desktops: the 5090 session held p50 2,587–2,647 MHz (step 0: 2,632–2,647; arms
  2,775–2,820 under load), the 5080 2,760 (step 0: 2,760). No gated or Holon arm drifted more than 0.74% across rounds.
  One repetition was flagged (FreeToken default, 5090, 1K, round 0, prompt 1: 83 ms); its round was re-run and the
  repetition flagged again (208 ms): reported with the flag, on a reported row. FreeToken default's 5080 4K arm drifted
  1.05% across rounds (a reported row; the gated arms ≤ 0.29%).
- **The bars** (time to the first token; the step-4 tree's speed over the reference: median of 6 round geomeans, sd
  over rounds; "median paired" is the median of all paired ratios):

  | Row (bar) | Step-4 tree, ms | Reference, ms | Ratio (sd) | Median paired | Bar |
  |---|---|---|---|---|---|
  | 5090 resident 1K vs FreeToken gated (≥ 0.95×) | 50.7 | 50.6 | 1.029 (0.002) | 1.000 | **met** |
  | 5090 resident 4K vs FreeToken gated (≥ 1.0×) | 170.4 | 171.7 | **1.008** (0.001) | 1.008 | **met** (by > 2 sd) |
  | 5090 resident 32K vs FreeToken gated (≥ 1.0×) | 3,178 | 3,227 | **1.016** (0.001) | 1.017 | **met** |
  | 5090 resident 32K vs FreeToken default (reported) | 3,178 | 3,019 | 0.950 (0.003) | 0.950 | — |
  | 5080 natural fit 1K vs FreeToken gated (≥ 1.5×) | 205.1 | 376.2 | 1.863 (0.010) | 1.836 | **met** |
  | 5080 natural fit 4K vs FreeToken gated (≥ 1.0×) | 333.2 | 380.2 | **1.141** (0.001) | 1.141 | **met** |
  | 5080 natural fit 32K vs FreeToken gated (≥ 1.0×) | 6,241 | 6,277 | **1.006** (0.000) | 1.006 | **met** |
  | 5080 natural fit 32K vs FreeToken default (reported; the 0.88× reference) | 6,241 | 5,988 | 0.960 (0.000) | 0.960 | — |

  - Against main: 5090 1.599 / 2.011 / 1.424× at 1K / 4K / 32K; 5080 (main's lock against the re-fill) 1.025 /
    1.226 / 1.097×.
  - The 5080 arms ran at their pins: main at its lock's 3,236 / 1,618 / 1,618, the step-4 tree at the re-fill
    3,229 / 1,700 / 1,529 (its natural fill at the lock's settings, §10 step 3d). FreeToken at its natural share,
    0.527.
- **llama.cpp rows** (`spikes/x4_gap/ctx.py`, the x4 gate's method, 3 rounds; prefill tokens/s ratios):
  - 5090 resident against llama-server resident: 1.78 / 2.36 / 1.45× (≥ 1.0×: **met**).
  - 5080 natural fit against llama.cpp's four offload configurations (`--fit`, `--n-cpu-moe 16`, `--moe-cache-mib`
    mmap and pinned): 2.51–3.64× at 1K, 7.07–12.0× at 4K, 3.27–5.07× at 32K (≥ 2.1×: **met**; the least is the
    pinned cache at 1K).
- **Decode** (acceptance 5, ≤ max(0.5%, 2 sd)):
  - 5090, `holon_tg` (4 alternating rounds, tokens/s over main's): batch 1 at depth 1,024 1.0019×, at 4,096 0.9976×
    (sd 0.0003): within; batch 16 / 32 / 64 at 1,024: 1.020 / 1.022 / 1.054×.
  - 5080 offload, the re-filled lock against main's lock (`ctx.py`, 256 decode tokens, 3 rounds; repeated in a
    second session): 1.0065 / **0.9942** / 1.0028× at 1K / 4K / 32K (second session 1.0062 / 0.9945 / 1.0029). **The
    4K row misses the tolerance by 0.05 pp** (sd < 0.001). The re-fill's down split is not the cause: at main's split
    (3,229 / 1,618 / 1,618) the row is 0.9891×. The 5090's resident batch-1 row at 4K (0.9976×) suggests a small
    decode cost at that depth in the code, not the lock: for the lead.
- **Memory** (acceptance 6):
  - startup KV, main → the step-4 tree: 5090 resident (33,792, one seat) 122,528 → 121,184 tokens; 5080 offload lock
    33,792 → 33,808; the 16 GB lock 18,752 (step 0) → 18,416 (≥ its 16,384).
  - The 16 GB soak (UD-IQ3_XXS at its lock's flags; 6 rounds of a 16,000-token prompt, then 32 seats of 512-token
    prompts decoding 128 tokens): no OOM; `num_alloc_retries` 0 every round; device allocations 4 in the first round,
    0 after; the least free device memory 192 MiB; allocation peak 14.13 GiB. `ggml_cuda.pass_rows` stays at the
    class's 2,048 (no retries).
  - Allocator counters zero in steady state on every timed prompt of every Holon arm, both cards.
- **sm_89:** still blocked (every 4090 we can reach is held by other jobs or in maintenance);
  the gate before step 5 (§10 L3-4) is unrun.

**Step 3b (D10, D11).**
- **What landed.**
  - `std.moe.kernels` (std minors, additive): `linear_experts@2`'s passed-buffer term (D10: `scratch_bytes`, `out`,
    `scratch`, no reference past the call, the current stream, one `scratch` for a pass's calls, no overlap of `out`
    with `x` or `scratch`, 256-byte alignment, refusal before launch, the same bits); `moe_resolve@1`'s variant term
    `lend_tail(cls, n, residency, classes, limits, **state)` with the largest-class rule (D11).
  - kernels.ggml_cuda `LinearExperts` (no native change):
    - `scratch_bytes(tokens)` is the call's largest mmq launch: `workspace_bytes_id` past the persistent workspace's
      rows, plus gate/up's f32 output; 0 for mmvq-only calls. Merged onto step 3, a stack whose SwiGLU runs in mmq's
      store (D4, `glu_store`) has no f32 output, so its `scratch_bytes` is the id maps only.
    - `out` / `scratch` are checked before any launch: `out` a contiguous f32 of the call's shape; `scratch`
      contiguous uint8, 256-byte aligned and at least `scratch_bytes`; `out` overlapping neither `x` nor `scratch`.
    - (Undone after step 3, §10 L3-2b: the workspace is at `pass_rows` again.) The persistent `ggml_cuda.id[…]` workspace
      is sized for `resident_rows` = min(`pass_rows`, max(`mmvq_tokens`,
      `max_seqs` · (1 + `max_draft_tokens`))). A larger mmq call without `scratch` allocates one buffer of
      `scratch_bytes`.
  - testing's CPU `LinearExperts` takes the buffers: each chunk's f32 result goes through `scratch`, then `out`. Its
    arena decode now re-decodes only the slots whose bytes changed (the tail's buffers change some on every call).
  - std.moe:
    - `MoE._forward` passes each pass the store's lent buffers (`pass_buffers`, the tail's), else the fallback
      (`pass_scratch`: one 256-byte-aligned buffer of the larger `scratch_bytes`, for both calls).
    - `ExpertLinear` / `ArenaExperts` pass the keywords only to instances defining `scratch_bytes`.
    - `Offload` lays the range out as `scratch`, gate/up's `out`, then down's `out`, at 256-byte boundaries, and gives
      the store the bytes of its largest pass.
    - `moe.ring_lend` (`coldest` | `tail`) is declared in `[options]`.
  - std.tiered:
    - `ring_lend`; the tail (class, n) at each materialize, with the bind refusals.
    - `lend_tail` as call 0, sharing `_unmap` with `lend`; the rounds are calls 1…`ROUNDS`; `lend_needs` is
      [`ROUNDS` + 1, C]; the give-back sentinel is (`ROUNDS` + 1)·E.
    - `held` is recounted from the tables after every call, outside the range; the pool takes slots outside the range
      only (`lend_ends`); `tail_view()` gives the range.
  - placement.lfu: `lend_tail` (oracle and CPU provider). Its replay tracks slots from the trace's start (slot map,
    empty stacks, class sizes), models call 0, and returns each step's stacks. kernels.expert_io's F11 resolve
    (`MoeResolveLfu`) has `lend_tail`; lru's does not.
  - judge: `route_dump` records the slots, the stacks and the class sizes, and `parity` also compares the empty slots'
    stacks. The plant `tail-unmap-undone` (`tail_unmap_undone`) is registered in `PLANTS`.
  - `spikes/moe_prefill/identity.py` records the offload store's borrowing steps and tail.
- **Decided while building** (B-dev6…B-dev10 in §10 are the deviations from rev 6):
  - Declaring `lend_tail` means the bound resolve instance has it. lru, promote, profile_decay and static are refused
    at the first resolver, with the error naming the policy.
  - An empty stack entry with a slot in the range leaves the stack whole (every empty slot does, for the step) and
    returns unchanged; its other roles' slots serve the step's pool, as the evicted experts' other-class slots do.
  - The range's class: the first role's class that most layers have, ties to the lower index (the contract text).
  - `TieredStore.lend`'s default call number is 1, the first round.
- **Checks** (sm_86, RTX A5000).
  - **CPU, FN's shape:** `test_a_tail_step_lends_its_passes_a_contiguous_range`, at E = 512, top-10, two layers,
    `*=0.75`, steps of 10 tokens, `chunk_with_decode` 4. The range is 15 gate/up slots. `_tail_checked` asserts on
    every tail step:
    - `lends` row 0 is n at the tail's class, then the rounds;
    - the range's holders are evicted in every role, with call 0, and no round picks one;
    - the range is contiguous, the class's last n slots, and at least the clients' bytes;
    - no pool, stage or ring slot falls in the range, and no mapped piece does, at the start and after every layer;
    - `held` equals the pool's count per class; a held ring has a slot for every missing piece; no `SHORT`;
    - after the give-back, the found entries are unchanged below and the lent ones lie whole above, by call then id.

    The first tail step finds no empty slot. The outputs are the resident store's, mixed steps included. Parity
    (residency, empty counts and stacks per step) holds, and the device's final stacks equal the replay's. `tail` is
    refused at bind under lru, promote and profile_decay, and under `*=0.5` (512 − 15 slots < `prefill` 512).
  - **CPU, std:** the call-size test records the fallback, one `scratch` per pass of the larger `scratch_bytes`, the
    same buffer for both calls, no `out`. Without `scratch_bytes` the keywords are not passed.
  - **Mutations** of the tail test, each failing it:
    - the range counted as held (only `_held`): "held counts other slots than the pool's";
    - `Lent.end` = the arena's end: "the pool holds the tail";
    - the tail numbered as round 1 (A5);
    - empty slots in the range left in the pool;
    - `lend_tail` picking nothing.
  - **L0 (`test_ggml_cuda_moe.py`):** the arena test runs `passed_buffers` for glu none and swiglu:
    - NaN-poisoned `out` and `scratch` give the allocating call's bits at 4,608 (4,096 + 512) and 4,097 (4,096 + an
      mmvq row) tokens;
    - a `max_seqs` 64 binding keeps 64-row calls' id maps in its workspace: `scratch_bytes(64)` is gate/up's output,
      less than `scratch_bytes(65)`, and both calls are bitwise;
    - short, misaligned and overlapping (with `scratch`, with `x`) buffers and a wrong `out` shape are refused with
      nothing launched.

    The graph-replay test also captures every decode bucket with static NaN-poisoned `out`s and one `scratch` for
    gate/up then down, replayed on new routing: bitwise the eager allocating call. Mutation: gate/up's f32 output
    carved over the id workspace aborts the test process.
  - **L0, the store on the GPU:** `test_the_offload_store_on_the_gpu_is_bitwise_the_resident_store[-*=0.75 tail]`:
    - ring steps lend the tail;
    - outputs are bitwise the resident store's, with parity and `_tail_checked`;
    - no host sync in `Lent`.
  - **Memcheck** (sm_86, passed buffers of exactly their sizes added): 0 errors, at both commits.
  - **Identity** (`identity.py run --config natural`, 30B Q4_K_M, A5000; 1,027 / 4,096 tokens;
    `evidence/mp3b_identity_rtx-a5000_2026-10-18.json`):
    - main (`b5b1e7f`), the branch, and the branch with `moe.ring_lend=tail` (tail 366 gate/up slots, 4 ring steps)
      all give `3a7308bf…` / `e7670f73…`, each run-identical ×2, at both commits of the step.
    - The branch with `tail` under the plant gives `ed55a19a…` / `1e8e5ecf…`: the plant fired 695 times in 2 ring
      steps.
  - **The plant on the CPU:** it fired 15 times and parity first differs at step 1. The outputs stay the resident
    store's there: the staging step's rounds lend the planted experts again, which unmaps them, so only the rounds'
    picks differ from the replay's.
  - **Static CI** (`run_all.sh b5b1e7f`): clean.
  - **Suite time** (`--durations`, the shared host; `test_moe_engine.py` / `test_ggml_cuda_moe.py`):
    - before (`7cd24de`): 68.5 / 177.0 s, 159 passed in 253.7 s;
    - after: 53.9 / 199.0 s, 161 passed in 257.8 s (an earlier run of the step's first commit: 57.4 / 158.5 s).
    - The added tests cost ≈ 5 s (the E = 512 test 4.0 s plus its 1.0 s fixture), ≈ 1 s (the GPU tail case) and
      ≈ 0.5 s (graph replay). The CPU tests got faster: the reference provider re-decodes only changed slots.
  - **Review:** one correctness reviewer over the step's diff. First round: no blockers; one major (the test did not
    check that `held` excludes the range), one minor and four nits. All were fixed in the second commit, and the
    re-review found none remaining.
  - Not run here: step 3b's perf check on the cluster (the lead's).

**Step 3c (D9, IQ1_M and Q2_0).**
- **What landed** (`d4e11f4`; a `vendor.patch` hunk, every part marked "Holon:", `vendor.sh --check` clean):
  - `ggml_cuda_mmq_load_tiles_iq1_m` (`mmq-load-tiles.cuh`), after IQ2_S's: per 32-value sub-block, each weight
    `8·g − 7` or `8·g − 9` (the grid nibble g ∈ {0, 1, 2}, delta's sign bit) as an int8 in {±1, ±7, ±9} through
    `__vsub4`, and the per-16 scale `d·(2s + 1)/8` as an f32 `x_df` in IQ2_XS / IQ2_S's layout (the SRAM layout
    `Q3_K`);
  - IQ1_M → `MMQ_Q8_1_DS_LAYOUT_D4`; the MMA and dp4a entries in `ggml_cuda_mmq_get_util_funcs` (IQ2_S's
    `q8_0_16` vec dots); `MMQ_DP4A_TXS_Q8_0_16`; IQ2_S's 16 ampere config rows for IQ1_M (blackwell falls through);
    IQ1_M in `ggml_cuda_should_use_mmq` and in `mmq.cu`'s type switch; `extern DECL_MMQ_CASE(GGML_TYPE_IQ1_M)`;
  - `template-instances/mmq-instance-iq1_m.cu`, created by the patch: upstream has none (neither the pinned
    `cbe28bb3` nor a 2026-10-16 checkout), so it is Holon's own and not in `vendor.sh`'s FILES or `SHA256SUMS`;
  - `TYPES` gains `iq1_m` (29) and `q2_0` (42) (the vendored `ggml.h`); Q2_0 already had upstream mmq, mmvq,
    convert and get_rows;
  - `PASS_PROVEN`: FN's types now include iq1_m and q2_0, on all three SMs (sm_86, sm_89, sm_120), as the other FN
    types are. Only sm_86's L0 has run: the lead's sm_89 / sm_120 L0 (`PLACED` per SM) covers the other two before
    step 5, and an entry whose L0 fails there is removed (B-dev5);
  - `test_link` checks the markers in `mmq-load-tiles.cuh`, `mmq-config-ampere.cuh` and the new instance too.
- **The q8_1 emulator needs no change.** IQ1_M's mmq is D4 with no stored-sum term (the offset is inside the int8
  weights), and its mmvq uses the quantized sums (`vecdotq.cuh:1311-1315`); Q2_0 is D4 on mmq and sum-free on mmvq.
  `blocking`'s defaults give both; the docstring now says so.
- **Decided while building.**
  - `vendor.sh --check` reverses the patch with `patch -E`, so the file the patch creates is removed again and the
    file list equals `SHA256SUMS` (without it `patch -R` leaves an empty file and the check fails). A refresh applies
    the patch forward and creates it (B-dev3).
  - gguf-py 0.19 has no Q2_0 (`GGMLQuantizationType` stops at Q1_0), so the tests read its blocks themselves
    (`test_ggml_cuda.dequantize`, `BLOCK_SIZES`): fp16 d, 2-bit codes, `(c − 1)·d`, as `ggml-quants.c`'s
    `dequantize_row_q2_0` and a later gguf-py's `Q2_0` define it. Convert and get_rows match it bit for bit (B-dev4).
  - IQ1_M has no fp16 d of its own: `random_blocks` writes the realistic d into the top nibbles of its four uint16
    scales (random bytes there would make about one block in 32 an inf or NaN scale).
- **Checks** (sm_86, RTX A5000; `evidence/mp3c_checks_rtx-a5000_2026-10-18.json`; run on the tree committed as
  `d4e11f4`, which differs from it in docstrings only):
  - L0, `test_ggml_cuda.py` + `test_ggml_cuda_moe.py`: **294 passed, 8 skipped in 336 s** (before, at `7cd24de`:
    266 passed, 8 skipped in 292 s). The 28 new cases are the two types in every `sorted(TYPES)` test (dense
    `qlinear` at `in` 768–12,288 on both paths, the mmq schedules, convert and get_rows; `linear_experts` every type
    and at prefill chunks, D0's 4,608-row pass check included) and in sm_86's `PLACED` tests (placement and tile-width
    invariance, compacted launches, both stages and helpers). No new suite.
  - Mutation: the loader's two delta offsets swapped fails 7 of the 14 IQ1_M tests (every one that runs its mmq:
    dense at every `in`, the schedules, both every-type MoE tests); the invariance tests pass under it, as they compare
    the kernel with itself.
  - Memcheck: MoE (26 layouts and modes) and dense (22 types): 0 errors.
  - Static CI (`run_all.sh b5b1e7f`): clean, `vendor.sh --check` included.
  - Per call, IQ1_M's loader against IQ2_S's, FN's gate/up stack (E = 512, top-10, [E, 1,280, 2,560], bf16 in,
    random blocks, skewed routing), interleaved ×5, median ms:

    | Tokens per call | IQ1_M | IQ2_S | IQ1_M / IQ2_S |
    |---|---|---|---|
    | 512 | 2.81 | 3.22 | 0.87 |
    | 1,024 | 3.72 | 4.18 | 0.89 |
    | 2,048 | 5.94 | 5.77 | 1.03 |
    | 3,072 | 6.87 | 7.26 | 0.95 |

    - `python spikes/moe_prefill/loader_speed.py --tokens 512 1024 2048 3072 --rounds 5`.
    - Down is not measured: FN's I = 640 is not a multiple of 256, so neither type can hold it.
    - 3,072 is the largest call at top-10 (`ID_PAIRS` ÷ 10 = 3,276 tokens: FN's `pass_rows`), so 4,096 is not run.
    - The loader holds IQ2_S's speed, so the grouped-dequant fallback is not needed.
- **Not in this step** (found while building): §5.1's "FN's shape is one more stack in the every-type test" is not in
  `test_ggml_cuda_moe.py` (no E = 512, top-10 stack). Suite time grew by 44 s (+15%) from the two types; acceptance
  7 asks that it not grow.

## 8. Risks

- **Thin margins at 32K.** Holon's attention lead is unmeasured on sm_120 (C-M3). The 5080 32K row reaches its bar
  only against the gated FreeToken arm, and only with D4. Beyond D6 lies an attention item (FlashInfer at
  4,096-token steps), which this plan does not take.
- **The 5080 4K row's regime.** FreeToken's 4K there is transfer-bound (368 ms of streaming in 386 ms), so the scaled
  band may be optimistic. Step 0's nsys decides.
- **J-invariance off sm_86.** It holds structurally on sm_86 / 89 / 120 (A), but each J compiles separately, so L0
  runs per SM. A type or arch whose bits move keeps `pass_rows` = `CHUNK` and J against T. That is D0's per-type
  switch, recorded.
- **Batched decode under D2.** A smaller J could re-read hot experts' weights (C-M6). The fallback (J against T at
  ≤ 64 rows) is bitwise and is chosen by step 0's sweep.
- **FreeToken is a moving target:** the bars are pinned to `21d9578`.
- **The vendor patch's growth.** D2 is kept to one Holon function plus three lines (A's maintainability note). D4's
  fused store is the larger hunk.

## 9. Owner items

- **O1 (escalated by the lead; not blocking): D6 on the 5080 offload lock trades 32K prefill against decode.**
  - Default: no D6 on offload locks, so decode wins.
  - Then the 32K row is predicted at ≈ 1.00–1.06× against the gated FreeToken arm (modelled until step 0 measures
    it), and ≈ 0.98–0.99× against FreeToken's ungated 8,192-token default (reported, not gated).
  - With D6: ≈ 1.00–1.02× against the default, at ≈ −3.0% / −3.7% / −2.4% decode at 1K / 4K / 32K. The FreeToken
    decode rows go 1.06× → ≈ 1.02× and 1.03× → ≈ 1.005×: still at or above their bar, but with no margin. Taking D6
    there **waives acceptance 5's decode tolerance** (≤ max(0.5%, 2 sd)) on that lock.
- **O2: the bars and arms.** **Closed at step 4** (the lead): the 5090 resident 4K row met its bar, 1.008× FreeToken
  gated by more than 2 sd, in one session with every other prefill bar (§7.1, step 4).
  - Confirm acceptance 1–2, with FreeToken's gated arm at the lock's step budget and its default reported only.
  - Rows at or under their bar in §1.1:
    - 5090 1K and 4K, without D4;
    - 5090 4K and the 5080 32K-gated row at the bar's low end, even with D4.
  - These are re-sent with the measured gated arm after the bar cards' step 0.
  - **From the step-2 base (C1; escalated by the lead):** the 5090 resident 4K row is predicted at 0.988–0.993× with
    every remaining item, under its bar; both 32K gated rows at ≈ 1.002–1.009×, within 3% (§2.2). Step 3's
    measurements replace the prediction.
  - Acceptance 1's 32K row is now against the gated FreeToken, which step 0 measures as slower than the reference
    that recorded 0.88× (≈ 4–7% modelled). The owner confirms the relabel.
- **O3: pass size per class.** `ggml_cuda.pass_rows` lowers the step-size gain on classes whose KV capacity cannot
  absorb 4,096 (likely the 24 GB 30B classes at 2,048 and the 16 GB class at 512 or 2,048). The values are set by
  hand at the close from step 0's measurements.
- **O4: bf16 expert intermediates** (FreeToken's, ≈ 2.4% at 4K) stay out (m02 D3). The handicap is stated next to the
  bars.
- **For the lead (from B-M6, outside this track):** K and FN both claim ABI 1.4 (`v1-kv-tiers.md:450`,
  `v1-flash-next.md:376`).

## 10. Review log

### Building: the step-3 panel's code review (2026-10-19; A, B, C) and step 3d

Records: `evidence/mp3d_rtx-5080_2026-10-19.json`, `evidence/mp3d_rtx-5090_2026-10-19.json`,
`evidence/mp3_bdec_2026-10-19.json`.

| # | Finding | Disposition |
|---|---|---|
| B1 | "Holon:" markers on every hunk is not what the patch does | D8 restated: every semantic hunk carries a marker; signature plumbing is named in each patched file's header note (`mmq.cu`, `mmq.cuh`, `mmq-load-tiles.cuh`, `mmvq.cu`, `quantize.cu`, `moe-weighted-reduction.cu`); the semantic hunks that lacked one (the compact launch's slot and bounds, glu's output rows, the bf16 stores) are marked (`a1e284a`) |
| B2 | D10's refusal is not what the mmvq-only path does | the contract (`std.moe.kernels`) and D10: refused where the call uses the buffer |
| B3 | `ggml_cuda.pass_rows` also caps dense `qlinear` | the manifest help and D0 name every call it caps: mmq with ids, D5's dense calls, D3′'s router pass |
| B4 | The batched-decode threshold lives in `mmid.cuh` | `plugin.py`'s `IDS_J_TOKENS` per SM beside `CROSSOVER` (16 on sm_86 / sm_120; 64 elsewhere, sm_89 provisional), passed through `mmq_id` / `mmq_id_glu` to `ggml_cuda_mul_mat_q` (`ids_j_tokens`; upstream's rule without it); the L0 tile-width test reads the table |
| B5 | iq1_m / q2_0 in `PASS_PROVEN` on sm_89 and sm_120 before their L0 | sm_120: the 5080's L0 at the final tree runs them (`PLACED` per SM), passed. sm_89: a hard gate before step 5, with A1 (L3-4) |
| B6 | std.moe's expert-store members are undocumented | listed in `Resident`'s docstring; `pass_buffers` None means the pass allocates |
| C | Step 4 on the 5090: 6 balanced ABBA rounds for rows within 3% of their bar; short kernels timed in a graph or over ≥ 100 launches, checked against nsys; acceptance 5's offload decode rows on the re-filled lock against main's lock | §5.2 and acceptance 5 |
| C | Commit B-dec's evidence | `evidence/mp3_bdec_2026-10-19.json` (per call on the 5090, 5080 and A5000; batched decode on the 5090 and 5080) |
| C | Re-run the sm_120 L0 and identity at the final tree | the 5080 at `fed13f1`: L0 308 passed, memcheck 0 errors; identity below (step 3d) |
| C-M2 | Confirm L3-2b's mechanism with the allocator's counters | `prefill.py` records each timed prompt's `num_alloc_retries`, `num_device_alloc`, `num_device_free` deltas. **All zero in steady state** on the final tree (5080 natural fit, 1K / 4K / 32K, 5 rounds; the 16 GB lock, UD-IQ3_XXS at its flags, 1K / 4K / 16K) — and on the tree with the cost (`a9a0333`) too. So the mechanism is not allocator traffic; it stays unexplained (GPU kernel time unchanged, idle time between kernels grown, cured by the id workspace's placement). **Closed** (the panel's C): the allocator hypothesis is ruled out; the residual is recorded as an unexplained host-path effect of where a call's id maps and activations live |
| A1 | sm_89 has no step-3 record | hard gate before step 5 (L3-4) |
| A2 | The fused store's coverage | the every-type and routings tests assert `glu_store` for every type at the 30B's shape: every type takes it on sm_86 and sm_120 (MMA layouts), so the fused set is all of `TYPES` there |
| A3 | `_check_buffers` misses scratch over `x` | refused (and in the CPU reference); an L0 case |
| A4 | D4 on arenas | recorded: the offload store's arena stacks bind `linear_experts_arena@1` to the same `LinearExperts`, so D4's store runs there too; bits unaffected (identity, natural fit and capped F5b) |

**Step 3d (D3′, the router per pass).**
- **What landed** (`a1e284a`): D3's router GEMM back (the revert of `8f434fd`); kernels.ggml_cuda's router declares
  `pass_rows`, and std.moe routes a pass of whole `CHUNK`s in one call (a `moe_router@1` contract term); the
  call-size test checks the router's call sizes (4,096 then 4 at 4,100 tokens).
- **Bits:** the final tree's `prompt_logprobs` hashes equal the per-chunk D3 + D4 tree's (`e9048de`): sm_86 resident
  and natural; sm_120 5080 natural and capped F5b, 5090 resident. D3's L2 verdicts (§7.1) therefore stand for D3′.
- **Keep rule** (each tree against `9e23a71`, D3 reverted; 5 interleaved rounds, §5.2's method; it passes on the 5080's 4K
  row and the 5090's 32K row):
  - 5080 natural fit: 1.0001 / **1.0116** / 1.0050× at 1K / 4K / 32K (sd 0.0043 / 0.0002 / 0.0003): **kept**
    (4K ≥ max(0.5%, 3 sd)). Per chunk, D3 had given +0.42% there: the per-chunk launches were the gap (L3-1).
  - 5090 resident: 1.0149 / 1.0167 / **1.0101**× (sd 0.0073 / 0.0075 / 0.0028): **kept** (32K ≥ max(0.5%, 3 sd)).
    This 5090 desktop held lower clocks than step 3's (p10 / p50 2,467 / 2,520 MHz), hence the wider sd.
  - The 5090's medians, `9e23a71` → the final tree: 53.1 → 52.1, 179.3 → 174.7, 3,300 → 3,260 ms. Against step 0's
    FreeToken medians (another session and desktop; step 4 measures both in one): 0.971× / 0.981× / 0.994× gated
    (0.927× default): the 4K and 32K gated rows stay under their bars (O2).
- **D4's fused q8_1 quantize** (optional, only if the 5090 4K row stays under 1.0×): not built. Step 0's nsys puts all
  of the experts' q8_1 quantize at 4.9 ms of the 5090's 4K step (both stacks' inputs, ≈ 2.8%; down's alone less),
  short of the row's gap, and a bitwise q8_1 store in mmq's tile is a kernel of its own: for the lead.
- **The re-fill:** the final tree's natural fill on the 5080 is 3,229 / 1,700 / 1,529 (D3 returns 12 + 12 slots);
  the A/B pinned both trees at 3,217 / 1,688 / 1,529. The 16 GB lock's startup KV: 18,416 tokens (main 18,752).

### Building: the lead's decisions after step 3 (2026-10-19)

Records: `evidence/mp3_followup_rtx-5080_2026-10-19.json` (every A/B, the bisects, the fits, the final tree's checks).

| # | Decision | Done | Result |
|---|---|---|---|
| L3-1 | Revert D3 by its keep rule | `8f434fd` (`git revert` of `0488063`: the router GEMM, its bindings, its L0 checks and the two router plants) | the router is main's again. Identity: the integrated tree without D3 gives the D5 tree's `prompt_logprobs` hashes, run-identical ×2 (sm_86 resident and natural; sm_120 5080 natural and capped F5b), so D4, 3b and 3c stay bitwise. **The gap:** per call the router GEMM saved 11 µs per 512-row chunk (5090, E = 128: 28.4 → 17.3 µs), ≈ 4.3 ms at 4K and ≈ 34 ms at 32K over 48 layers; end to end it saved 1.2 / 10 ms (5090) and 1.3 / 12 ms (5080), 25–35% of that. Not isolated further: the per-call bench times the router alone, warm, and nsys was not run on the D3 tree |
| L3-2a | D10's fallback a persistent buffer: one workspace of `scratch_bytes(pass_rows)`, reused by every pass and step | `28ad86e` (`MoE.bind_kernels` binds its store and stacks, then sizes it; the call-size test asserts one buffer for every call) | **no gain**: on the 5080, the tree with it against D4 without D3: 0.9995 / 0.9975 / 0.9990× at 1K / 4K / 32K (sd ≤ 0.002), the old cost unchanged (the old trees, same session: 1.0004 / 0.9980 / 0.9990×). Memory: the 5080 fill returns to the D5 tree's 3,217 / 1,688 / 1,529 (D10's 6 + 6 slots are taken back); A5000 resident KV 48,544 → 48,576 tokens. **Reverted** after L3-2b (the lead's decision): it measured no gain, and simpler is better. With the id workspace at `pass_rows` again, ggml's `scratch_bytes` is gate/up's f32 output alone, none under D4's store, so on the 30B locks the fallback allocates nothing either way. After the revert the A5000's `prompt_logprobs` hashes (natural fit and resident, 1,027 / 4,096 tokens, run-identical ×2) equal the D5 tree's and `1f535e8`'s |
| L3-2b | Find the cost: bisected on the 5080 (4K prompts, 10 per arm, 5 interleaved rounds; each arm against D4 without D3) | 3c alone 0.9999×; step 3 with 3b's kernels.ggml_cuda `plugin.py` swapped for 3c's 0.9998×, with 3b's std / tiered / policy files swapped instead 0.9971×: the cost is in kernels.ggml_cuda's D10. Host time was not it: mmvq calls skipping D10's work (`e076f00`, an L0 check fails without it), a ring step counting held slots as the rounds pick them (`a815981`) and host caches for D10's helpers (2 → 0.4 ms of host time per 4K prompt on the A5000) left 0.9979×. The persistent id workspace back at `pass_rows` (3b had sized it for decode, so a 4,096-row call took its id maps and q8_1 activations from the caller's scratch, shared by gate/up and down, or a fresh buffer) gave 1.0000× | `c0fe9c2`: one persistent id workspace per `in` at `pass_rows` again; D10's `scratch` holds gate/up's f32 output only (none under D4's store); `1f535e8` drops the host caches. **Against D4 without D3, 5 rounds, the fixed prompt set: 0.9999 / 0.9997 / 1.0000× at 1K / 4K / 32K (sd ≤ 0.0008), within ±0.05%**; the final tree at 4K: 1.0000× (sd 0.0002). The mechanism (why a shared or fresh buffer costs 0.2%) is not isolated; GPU kernel time was unchanged on the A5000's nsys, the idle time between kernels grew |
| L3-2c | The final tree's checks (`1f535e8`; the L3-2a revert after it changes no bits and, on these stacks, no memory) | sm_86: 618 passed (the kernel, MoE, tiered, offload and judge suites); sm_120 (5080): L0 308 passed, memcheck 0 errors | `prompt_logprobs` hashes equal to the D5 tree's (sm_86 resident and natural; 5080 natural and capped F5b), so D4, 3b, 3c and the fixes stay bitwise. The 5080's natural fill: 3,217 / 1,688 / 1,529 cache slots (52.36%, the D5 tree's and 6 + 6 slots under the step-3 tree's; the lock's 3,236 / 1,618 / 1,618 hold 52.7%). Acceptance 6: against the step-2 tree's fill in the same session (3,223 / 1,694) the cost is D5's 6 + 6 slots, inside its ≈ 9-expert budget; against the lock's counts, filled by `fit.py`'s default configuration, it is 19 gate/up slots: step 5's re-fill settles it. Step 3b's tail check (one round each): every borrowing step a ring step with `ring_ok`, no `SHORT`; the tail is gate/up's last 209 slots (the passes' `out`s only now); prefill 337.4 / 6,278 ms (`tail`) against 336.3 / 6,280 (`coldest`); miss bytes per decode token 120.6 / 57.4 against 121.3 / 58.6 MB. The 5090 (queued) did not run: its resident rows had shown no cost beyond 1 sd |
| L3-3 | O2 to the owner; the bar stays; step 4 measures | — | — |
| L3-4 | Retry the 4090 for B-dev5's sm_89 L0 (IQ1_M, Q2_0) and C5's per-call A/B; with the panel's A1, the sm_89 record | queued again twice (2026-10-19); never started (Priority) | **Hard gate before step 5** (not step 4), on a 4090 NVFP4 or MXFP4 lock at the final tree: the full L0 (`test_ggml_cuda_moe.py` + `test_ggml_cuda.py`), memcheck and the `prompt_logprobs` identity — covering D4's NVFP4 scale2 epilogue, D5, B-dec's 64-token branch, D10 and the iq1_m / q2_0 `PASS_PROVEN` entries; any sm_89 pair that fails is dropped. Plus C5's per-call sweep (B-dec's sm_89 threshold) |

### Building: the panel's code review of steps 0–2 (2026-10-18; dispositions)

| # | Finding | Disposition |
|---|---|---|
| A1 | The step-2 checks ran on a tree with uncommitted edits | re-run at `1c82ae1` on a clean tree (sm_86, RTX A5000): L0 266 passed, 8 skipped; memcheck 0 errors; skip-tail mutation 50 failed; forced-fixup mutation aborts the process; identity, resident and natural, identical to main's at 1,027 / 4,096 tokens (the branch's pass 4,096 on both stores) |
| A2 | The call-size test covers the resident store only | parametrized over both stores: offload runs a streaming step, the ring on, at the 4,100-token budget ([1,024, 3], [4,096, 4] among the cases) (`3d705ff`) |
| A3 | `pass_rows` is trusted to be whole chunks | `units.pass_rows` asserts ≥ `CHUNK` and a multiple of it (`3d705ff`) |
| A4 | The ids-fixup mutation's description | described as a forced fixup on the ids path, its assert dropped (§7.1, step 2) |
| B1 | The arena contract does not say an arena instance may expose `pass_rows` | `linear_experts_arena@1` states the term, at the bits of its `CHUNK`-row calls over the same slots (`3d705ff`) |
| B2 | The 8,192 choice of `ggml_cuda.pass_rows` exceeds `PASS_ROWS` | dropped: choices 512 / 2,048 / 4,096 (`3d705ff`) |
| B3 | A stale comment at `plugin.py:80-81` | reworded: the instance cuts a call's last partial chunk itself, whatever its caller passes (`3d705ff`) |
| B4 | Lean tests | the every-type pass check runs only where `pass_rows` > `CHUNK`; the E = 128 checks are folded into the NaN-poisoned tile test (`3d705ff`); suite times before and after: §7.1, step 3 (acceptance 7) |
| B5 | The flat launch is chosen inside the kernel | the flat range is a template instance of its own, chosen on the host (ntx > 1); the non-flat instance decodes tiles as upstream (`3d705ff`) |
| C1 | §2.2's cells from the step-2 base | restated in §2.2 before step 3's runs: the 5090 4K row 0.988–0.993×, under its bar with every item; to the owner as O2 (escalated by the lead), not blocking |
| C2 | §5.2's method | from step 3 on: an untimed pass over the whole prompt set per engine process, the 1.5×-median repetition flag and re-run, the median of paired ratios beside the statistic, round index and arm order recorded (§5.2) |
| C3 | B16 ≈ −0.5% and a 1–2.5% cost at T = 16 on the down stack | found: the flat range's binary search, compiled into every instance, raised Q6_K J = 32 from 128 to 146 registers (occupancy halved). Fixed by B5's separate instance (`3d705ff`). Measured: §7.1, step 3 |
| C4 | The 5080 offload lock does not fit D5 | re-filled (scratch, not committed) at the start of step 3's runs: §7.1, step 3 |
| C5 | Batched decode: J against T only up to 16 tokens | taken as deviation B-dec (decided before step 4): 16 on sm_86 and sm_120, 64 on sm_89 until the 4090's per-call A/B runs. Measured: B-dec |
| C6 | "Reserved ≤ profiled peak" is not the right memory gate | dropped; step 4 runs a soak on the 16 GB lock instead (§7: no OOM, `num_alloc_retries` = 0, free memory at the worst point; `pass_rows` 512 there only if retries appear) |

### Building: deviations from rev 6 (steps 1–2, 3b and 3c, 2026-10-18; recorded for the panel)

| # | Deviation | Why | Effect |
|---|---|---|---|
| B-dev1 | `LinearExperts` cuts a call itself, into whole-`CHUNK` calls of up to `pass_rows` rows, then the last partial chunk as a call of its own (D0 left the cut to the caller) | a direct caller's 2,049-row call would otherwise put its last row on mmq, not mmvq: its bits would depend on the caller's cut | every row's kernel path is its chunk's for any caller; std.moe's passes unchanged (they cut the same way) |
| B-dev2 | D2's flat range is launched only where a call's tokens span several column tiles (⌈T ÷ J⌉ > 1, read in the kernel as ntx > 1); otherwise upstream's launch, one tile per entry (D2 said the flat grid always) | with C-M6's fallback (J against T up to 64 tokens) every entry fits one tile, and the flat range's extra blocks and binary search cost 2–7% per call against main on the A5000 | ≤ 64-token calls run main's launch exactly (A/B 0.99–1.02 on three cards); bits the same either way |
| B-dev3 | `vendor.sh --check` reverses `vendor.patch` with `patch -E` (D9 named the template instance file but not how the check treats a file upstream lacks) | the IQ1_M instance is Holon's own, created by the patch; `patch -R` alone leaves it empty, and the check's file list then differs from `SHA256SUMS` | the check passes with the patch-created file; a refresh still creates it by applying the patch |
| B-dev4 | Q2_0's L0 oracle reads its blocks in the test helper, not through gguf-py (§5 assumes gguf-py's dequantization for every type) | gguf-py 0.19 (`uv.lock`) has no Q2_0 | the reader is ggml-quants.c's `dequantize_row_q2_0` definition; convert and get_rows equal it bit for bit; `testing`'s reference provider still cannot serve a Q2_0 file (FN's concern) |
| B-dev5 | iq1_m and q2_0 are in `PASS_PROVEN` on sm_89 and sm_120 before their L0 runs there (D0: an entry exists only where its L0 runs) | the lead's instruction: `PASS_PROVEN` lists FN's types on every SM, and the cluster L0 (`PLACED` per SM) runs them | pending the lead's sm_89 / sm_120 L0 before step 5; a failing pair is removed; with the panel's A1 the whole sm_89 record is one gate before step 5 (L3-4) |
| B-dec | C-M6's fallback narrowed (the panel's C5, decided before step 4; D2 said J against the mean at every T): J against T for calls of ≤ 16 tokens on sm_86 and sm_120, ≤ 64 on sm_89 until the 4090's per-call A/B runs | D2's own A/B: the mean's 8-row tiles made the down stack up to 8% slower at 16 tokens; above 16 the mean wins per call (5090 / 5080, every lock and FN type: 0.52–0.99× main's time at T = 17–64, 0.998–1.013× at 16) | batched decode, paired ×4 rounds, tokens/s over main's: 5090 Q4_K_M B16 / 32 / 64 1.001 / 1.018 / 1.054× (sd ≤ 0.003); 5080 UD-IQ3_XXS B16 / 32 1.000 / 1.029× (sd ≤ 0.001). Types outside the sweep (no lock uses them) are accepted only behind an end-to-end decode check: IQ1_M and Q2_0's is FN's |
| B-dev6 | The tail's range is sized for the bound configuration's largest pass, min(`pass_rows`, `max_step_tokens`) rows (D11: `scratch_bytes(pass_rows)`) | no step carries more than `max_step_tokens` tokens (verify steps beyond it are graphed and never borrow), and `pass_rows` ≥ `CHUNK` would over-size short budgets | the 30B lock: unchanged (4,096, 366 slots); the CPU tests' 10-token budget lends 15 slots, not ≈ 770 (which the bind check would refuse) |
| B-dev7 | The range holds one `scratch` plus gate/up's `out` and down's `out` (D10 named `out` and `scratch`) | down reads gate/up's output while writing its own (`out` must not overlap `x`), and the combine reads down's afterwards | three buffers at 256-byte boundaries; a pass larger than the range would fall back to allocating (never at the bound configuration) |
| B-dev8 | The tail is lent on streaming (ring) steps only, and only where steps stream at all (`ring_tokens` after X's forcing); a staging step lends as X | the passes that need the range are the large ones (D8), which only streaming steps take; a store that never streams cannot use a tail | staging steps and non-streaming stores run D10's allocating fallback; no refusal where the tail would never be lent |
| B-dev9 | The bind check's capacity half runs at each materialize's start, against the class's cap (pinned, or every piece), and again on a layout that lends; its policy half at the first resolver (D11: "at bind") | the range's bytes come from the stacks' kernels, bound after the store's resolvers | refused at engine construction, before any step, as at bind |
| B-dev10 | D10's fallback buffer is sized per pass, from the pass's own tokens (none where no kernel needs one), not from `pass_rows`; ggml's `scratch_bytes` leaves out the id maps of launches the decode-size workspace holds | a decode pass would otherwise allocate a prefill pass's buffer, and graphed buckets would take id maps into the graph pool (P15: the fallback must not add to the reserved excess) | the startup profile's worst step allocates the largest pass's buffer; decode keeps main's graph-pool use (gate/up's f32 output only; none where D4 fuses SwiGLU into the store) Superseded after step 3 (§10 L3-2): kernels.ggml_cuda's persistent id workspace holds every call's id maps again (`pass_rows` rows), so `scratch_bytes` is gate/up's f32 output alone (none under D4's store); std.moe's fallback is per pass again (L3-2a reverted), allocating nothing where `scratch_bytes` is 0 |

### Rev 5 → rev 6 (the panel's fifth round, 2026-10-18; B clean)

| # | Finding | Verified | Disposition |
|---|---|---|---|
| A5-m | `lend_tail` needs its own call number, or `_give_back` (sorted by `lent_call · E + id` against a `ROUNDS · E` sentinel) corrupts the empty stacks | yes (`ring.py:147-152`; `store.py:227, :240, :501`; `lfu/reference.py:168-170`) | `lend_tail` is call 0, the coldest rounds 1…`ROUNDS`; `lend_needs` sized `ROUNDS + 1`; the sentinel `(ROUNDS + 1) · E`; the `lends` record gains row 0, read by lfu's replay and `placement.parity`; the E = 512 CPU tail step checks the empty stacks and `n_empty` after the give-back against the replay's |
| C5-m | Gate/up and down are separate instances, so no instance can own a per-pass buffer | yes (`MoE._forward` calls two `ExpertLinear`s) | `std.moe` allocates one buffer per pass, sized to the larger `scratch_bytes(pass_rows)`, passed as `scratch` to both; an instance allocates only for callers that pass nothing |
| C5-nit | The step-day total | yes | 15.5–22 days |

### Rev 4 → rev 5 (the panel's fourth round, 2026-10-18; B clean)

Both findings were verified at the cited lines.

| # | Finding | Verified | Disposition |
|---|---|---|---|
| A4-M = C4-M | In tail mode the tail's class gets no ring or stage slots but the empties outside the range. A short step sets `SHORT` and `after_commit` raises instead of staging; the counts can admit −1 pool entries | yes (`ring.py:42-56`, `:101-102`; `store.py:482-488`) | `lend_tail` first, then X's coldest `lend(need)` rounds for every class's remaining need; `held` / `want` for the tail's class exclude the range; lfu's replay models both calls; a CPU tail step at E = 512 with no spare empties; a bind check refusing `tail` when the tail class's lendable slots outside the range cannot reach `prefill`; D10's "the step stages" corrected to X's stage-or-`SHORT`→raise; step 3b's perf check with `tail` on the 30B offload lock (C) |
| C4-m | The allocating fallback's size and its reserved memory | — | one `scratch_bytes(pass_rows)` buffer per pass, reused for gate/up and down; step 0 checks that the reserved memory after a 32K prefill is ≤ the profiled peak |

Step 3b becomes 3.5–4.5 days; the schedule is 15.5–21.5 days.

### Rev 3 → rev 4 (the panel's third round, 2026-10-18; FN's arena rule; the lead's decisions)

Every finding was verified at the cited lines.

| # | Finding | Verified | Disposition |
|---|---|---|---|
| A3-1 | Tail lending must evict whole experts; empties share one count per layer | yes (`store.py:212-213`, `ring.py:142-155`, `lfu/reference.py:19-21`) | D11: whole experts in every role, empties out and back as whole entries, `(g, −1, item)` records |
| A3-2 | D10's bytes and the ring / stage pool can overlap | yes (`ring.py:60-63`, `:80-91`) | one owner per step: the D10 range is out of pool, `stage_tab` and the empty stacks; per-step CPU / L0 assertion; `tail` in the `prompt_logprobs` identity (lead decision 2) |
| A3-3 = B3-1 | The tail path bypasses the policy: it breaks promote's spares, parity covers lfu only, and the plant cannot fire | yes (`store.py:549-556`, `kernels.py:88-95`, `promote/reference.py:21-27`, `placement.py:120-124`, `planted.py:1051-1066`) | `lend_tail(class, n)` in the `moe_resolve` variant contract (a std minor); lfu implements it; `tail` refused at bind elsewhere; the plant `tail_unmap_undone` with an asserted fired count; parity for every accepting policy (lead decision 1) |
| A3-m | D10 alignment | yes (`mmid.cu:319`) | 256-byte alignment refused before launch |
| A3-m | D10 aliasing; graph addresses | yes (`store.py:532`) | `out` must not alias `scratch` / `x`; lent memory is eager-only |
| A3-m | D10's fallback memory | — | the startup profile runs the allocating fallback's peak |
| A3-m | A stale D1 sentence on CPU providers | yes | restricted to the oracle |
| B3-2 | std must gate the keywords | yes (`units.py:235`, `:262`) | only instances defining `scratch_bytes` accept them |
| B3-3 | Lifetime and aliasing unstated; the dma refill | — | no retention past the call; compute-stream ordering; the dma refill ordered as X's lent ring slots; scratch reusable within a pass |
| B3-4 | Unclear memory saving | yes | `scratch_bytes` per path stated; the persistent workspace shrinks to decode size, and larger calls allocate or take `scratch` |
| C3-m (via FN) | D6 waives acceptance 5 on the 5080 offload lock | — | stated in D6 and O1 |
| FN rev 3 F3 | The arena-choice rule | — | the largest class arena (the gate/up class of the most common type), ties to the lower index, in the contract text |

Step 3b grows to 3–4 days (`lend_tail`, lfu's reference and replay, `Lent`'s exclusion, the plant). The schedule
becomes 15–21 days.

### Rev 2 → rev 3 (the panel's second round, 2026-10-18; FN's requests; the lead's decisions)

Every finding was verified at the cited lines.

| # | Finding | Verified | Disposition |
|---|---|---|---|
| A2-M = B2-N1 | D9 / Q2_0 after step 5 would outdate the 18 locks (the plugin `src` hash) | yes (`plans.py:41-52`; `cli.py:167`) | D9 and Q2_0 land at step 3c, before step 5's single round (lead decision 1) |
| A2-m | The D2 fixup mutation may be a no-op; NaN poison needs a passed output | yes | `launched()` asserts no `mul_mat_q_stream_k_fixup` on the ids path; a NaN-filled output is passed to `lib.mmq_id` |
| A2-m | The CPU call-size test needs `pass_rows` above `CHUNK` | yes | the reference provider declares `pass_rows` (B2-N4), 4,096 by default |
| A2-m | "Verify keeps the router bits" is conditional | yes | the condition `max_seqs·(1 + k) < 512` stated (acceptance 5) |
| A2-m | D9's exactness condition; the missing hunk parts | yes (`dequantize.cuh:397-403`, `vecdotq.cuh:1311-1315`) | f32 `x_df`; DS_LAYOUT_D4, `should_use_mmq`, the ampere config, the MMA and dp4a instances (D9) |
| A2-m | D4 needs `inter % 64 == 0` | yes | stated, with the unfused fallback |
| A2-m | Near-plant verdicts | — | a passing near plant records L2's blindness; D3 / D5 kept (lead decision 4) |
| B2-N2 | `ggml_cuda.pass_rows` must be declared | yes (`check_locks.py:9, :42-44`; no `[options]` in kernels.ggml_cuda's manifest) | `[options]` with choices 512 / 2,048 / 4,096 / 8,192 |
| B2-N3 | "After its L0 holds" needs a table | — | `PASS_PROVEN` (SM, type, path), iterated by the shared helper |
| B2-N4 | Offload CPU tests become trivial | yes (`test_moe_engine.py:19, :357, :393`) | the reference loops per `CHUNK` and declares `pass_rows`; the test edits named in step 1 |
| B2-N5 | `PLACED` grows to 13 types | — | parametrized per SM |
| C2-N-M1 | The gated FreeToken arm is modelled; offload streaming not modelled | — | step 0 measures FreeToken at 4,096 and 8,192 chunks (5090 resident, 5080 offload, 4K and 32K); gated cells labelled modelled, the offload band widened to +2.6…7%; acceptance 1 relabelled gated, with the distance from the 0.88× reference stated (lead decision 3) |
| C2-m1 | D6's decode cost is stale | yes (x4_gate's slope at +0.47 GB) | −3.0% / −3.7% / −2.4% at 1K / 4K / 32K; O1 "at the bar" |
| C2-m2 | 5080 32K gated with D4 reads 1.00–1.02 | yes | the cell restated within the widened band (1.00–1.06×) |
| C2-m3 | 5080 4K low ends do not reproduce | yes | 1.08–1.11× / 1.12–1.15×, before a transfer-bound haircut |
| C2-m4 | Acceptance 6 against acceptance 5 | yes | acceptance 6 bounded by the persistent-buffer growth (≈ 0.3% of the share), referencing acceptance 5 |
| C2-m5 | D7 wording | yes | "≈ 22–27% of ≈ 222 TOPS" |

**FN's requests** (FN rev 2, its panel C's second round), taken into this track before step 5 (lead decision 2):

| Request | Disposition |
|---|---|
| A workspace / output argument on `linear_experts@2` / `linear_experts_arena@1` (`plugin.py:464, :481` allocate today) | D10, a std minor; its L0 in §5; step 3b |
| Contiguous-tail lending from one class arena, evicted slots returning empty | D11 (`moe.ring_lend` = `tail`); its CPU and parity checks in §5; step 3b |
| IQ1_M: the DS_LAYOUT_D4 case, the MMA and dp4a instances (`mmq.cuh:653, :786`) | D9's hunk list; step 3c |
| D9 and Q2_0 before step 5 | step 3c |

### Rev 1.2 → rev 2 (the three-lens panel, 2026-10-18; the lead's decisions; FN rev 2's requirements)

Every finding was checked against the code or the plans before its disposition. "Verified" means it was confirmed at
the cited lines.

**A — correctness** (6 major, 8 minor):

| # | Finding | Verified | Disposition |
|---|---|---|---|
| A-1 | D3 reaches batched decode (> 16 rows, `HOLON_ROUTER_MAX_ROWS`) | yes (`plugin.py` `MoeRouter.__call__`; `topk-moe.cuh:40`) | D3 only at ≥ `CHUNK` rows; decode keeps main's bits (acceptance 5) |
| A-2 | The snapshot cannot see D1 / D2; `pass_eats_tail` would not fail | yes (`snapshot.py:55-58`: tokens, plus logprobs only where asked) | a `prompt_logprobs` hash at 1,027 / 4K, main against the branch, three configurations; the plant becomes a CPU call-size test |
| A-3 | The tile map in three helpers is invasive; a grid ≠ ntiles triggers the ids fixup | yes (`mmq.cuh:1487`) | D2 is map-free: binary search over `expert_bounds`, `fixup_needed` gated to `!ids_dst`, no helper changes |
| A-4 | L0 misses types the locks use | yes (iq3_s, iq4_xs, q5_k, mxfp4, nvfp4 in `plans/`) | type × SM derived from `plans/` (§5.1) |
| A-5 | The D5 plant may never fire; plants are far from the bound | yes (4,096-row dense calls tile at 100% on the A5000) | fixup launches counted; near-bound plants for D3 and D5 |
| A-6 | Propagating D3 / D5 later conflicts with the plugin hash pin | yes | one close-round propagation, nothing written mid-track (§5.3) |
| A-m | J against the mean must use n_z, not ne02 | yes (`mmq.cu:204`) | D2 uses T·k ÷ n_z |
| A-m | NaN-poison the outputs in the tile mutation | — | §5 D2 |
| A-m | PDL ordering on sm_120 | yes (`common.cuh:1720-1740`) | D2: reads after `ggml_cuda_pdl_sync`; racecheck on sm_120 |
| A-m | `row_invariant` is static per spec | yes | superseded by `pass_rows` per instance (D0) |
| A-m | D4: scale2's multiply; g / u in different warps | yes (`api.cu:114-128`) | D4 uses fast-math `*`, a shared-memory exchange, subnormal L0 |
| A-m | The per-slot rule also needs n_pairs ≤ 4,096; "D1 does not touch arenas"; "products are exact" | yes (`mmid.cuh:21-23`) | text corrected (D1 states both stores; D3 says fma chains) |
| A-m | KV shrink against main's snapshot | — | comparisons pin pools or record no preemption (§5.2) |
| A-m | Resident L0 at 128 experts crossing all three helpers | — | one 128-expert stack at T = 512 / 2,048 / 4,096 (§5.1) |
| A-m | `ERR_COLUMNS` rows become computable | yes | meaning restated in D2 |

**B — architecture** (6 major, 8 minor):

| # | Finding | Verified | Disposition |
|---|---|---|---|
| B-M1 | The invariance flag is per spec, but the truth is per type / arch / path | yes (`plugin.py:617-625`; `units.py:149-154`) | `pass_rows` on the bound instance, set by the provider (D0) |
| B-M2 | The contract text is false (≤ 8-row calls take mmvq) | yes | the exact promise in §4; placement invariance kept out |
| B-M3 | Two formulas; `PASS` in `offload.py` (import cycle); no default | yes (`offload.py:29` imports `units`) | one source (D0); `PASS` retired; a missing `max_step_tokens` means no cap |
| B-M4 | tune cannot choose step sizes | yes (`tune.py:1-11`) | pass sizes and 8,192 are lock settings set by hand through `propagate` |
| B-M5 | FN and this plan both own `LinearExperts`' prefill path; dequant contradiction | yes (FN `v1-flash-next.md:157`) | D8: this track owns it and lands steps 1–2 and D9 before FN's step 4; D7 cross-references FN's IQ1_M |
| B-M6 | Batched propagation leaves main serving defaults; K and FN both claim ABI 1.4 | yes (`cli.py:167`) | this track lands first with its own round (§5.3); ABI 1.4 passed to the lead (§9) |
| B-m1 | The change is a std minor | yes (`design.md:338`) | called a std minor |
| B-m2 | Only the resident store checks | yes | both stores take the same min (D1) |
| B-m3 | 18 classes, not 23; the step-1 propagation is wasted | yes (`plans/`) | 18; no interim writes |
| B-m4 | Lean tests | — | adopted as proposed (§5.1) |
| B-m5 | The plant would edit the snapshot trace | — | a CPU unit test |
| B-m6 | §6 omits the 2,048 MXFP4 / NVFP4 locks and FP4's f32 workspace | yes | added; W4A4 unused by locks |
| B-m7 | "Holon:" markers and the `test_link` check | yes (`test_ggml_cuda_moe.py:234-235`) | D8 |
| B-m8 | K first changes the pools | — | re-measure after K (§6) |

**C — performance** (3 blockers, 7 major, 7 minor):

| # | Finding | Verified | Disposition |
|---|---|---|---|
| C-B1 | The predictions count D4, which was conditional | yes (README: probe minus rows 2, 5, 6) | D4 in the set; predictions per item set (§1.1), at-or-below-bar rows flagged |
| C-B2 | D6 on the 5080 offload lock trades 32K prefill against decode | yes (x4_gate: 52.4% against 47.1% share) | default no D6 on offload locks; owner item O1 |
| C-B3 | D1 can push locks below their context or capability | yes (`plans/`; plan M0.2's 18.6K; release.md `OutOfUnits`) | per-class cap through `ggml_cuda.pass_rows`; step 0 measures the 16 GB and one 24 GB lock |
| C-M1 | Row 2 includes FreeToken's bf16-store advantage | yes (`gemm.py --freetoken` used bf16 I/O) | D2 banded at −70…−85 ms; step 0's equal-I/O probe; step 2's check at equal I/O |
| C-M2 | D7 used peak rates | yes | rejection by achieved rates; per-card step 0 decides; the 8,192-pass A5000 case recorded |
| C-M3 | The same ratios do not mean the same parts (attention on sm_120, the 5080 transfer regime) | — | every cell restated from the bar cards' nsys at step 0 |
| C-M4 | FreeToken's chunk and numerics | yes (`--max-extend-length`; release.md:368, :967) | gated arm at the lock's step budget, default reported; the handicap stated next to the bars |
| C-M5 | The method cannot resolve 1–3% | — | §5.2 (≥ 5 / 3 prompts, ≥ 3–5 rounds, geomean-of-pairs median, "≈" under 2 sd) |
| C-M6 | Cross-session decode; D2's small J at batched decode | — | same-session ABBA, ≤ max(0.5%, 2 sd); step 0's J sweep with a bitwise fallback |
| C-M7 | The 5080 offload lock is pinned without slack | yes (`8c5a826`) | re-filled and re-pinned at the close (§5.3) |
| C-m1 | §6 summed buffers never alive together | yes (`LinearExperts.__call__`) | §6 rebuilt on the live peak (473 MB at 4K; D6 +0.47 GB) |
| C-m2 | `peak_alloc` rose 0.17 GiB | yes (evidence) | step 0 explains it |
| C-m3 | Profiled against timed walls (FreeToken 1K +28%) | yes | the README corrected |
| C-m4 | Per-call bench noise | yes (layers 0 / 6) | interleaved repeats at step 2 |
| C-m5 | Row 6 is an upper bound | — | labelled |
| C-m6 | "Products are exact" assumes ≤ 16-bit router mantissas | yes (`ffn_gate_inp` is F32) | D3 restated as fma chains in f32 |
| C-m7 | The 5080 4K band had no derivation | — | derived in §1.1 (1.06–1.11× / 1.08–1.15×) |

**FN rev 2 (`32c99f2`, its D1′):**
- FN's eight types and its shape (E = 512, top-10, H 2,560, I 640) join L0 (§5.1);
- IQ1_M's native mmq tile loader becomes D9, step 6, before FN's step 4 and outside the propagation round (superseded
  in rev 3: step 3c, inside the single round);
- the fallback is a grouped dequant behind `pass_rows`.

**Rev 1.1 → rev 1.2** (the same reviewer; 2 major, 5 minor). The fixes:
- D1's pass is rounded to whole `CHUNK`s: an unrounded `max_step_tokens` (e.g. 1,000) would move the router chunks of
  a longer verify step.
- D2 names the third id helper (the compact block one, 4,097–16,384 pairs), and L0 covers all three.
- Step 1 re-propagates the MoE locks for the recorded `row_invariant` config.
- `pass_eats_tail` pins a 1,027-token prompt.
- Acceptance 4 says "≤ `mmvq_tokens` rows".
- The J-sweep claim is limited to five types.
- The plan's revision is updated.
- Final round (3 minor, applied): "all three helpers" throughout; the helper dispatch stated as `mmid.cuh`'s rule;
  re-propagation covers every class binding kernels.ggml_cuda.

**Rev 1 → rev 1.1** (one reviewer, 2026-10-18; 0 blockers, 5 major, 10 minor). The reviewer confirmed from the vendored
code that D1 and D2 are bitwise on the GPU: mmq with ids always runs one block per tile, and a row's K order does not
depend on J.

| Finding | Answer |
|---|---|
| D4's `prepare` repack is refused by the offload store (`offload.py:164-168`) | D4 keeps the stored layout: gate and up rows from two base offsets, no repack |
| A contract-wide row-invariance term contradicts the CPU providers (per-batch oracle) | a provider-declared `config["row_invariant"]`; std passes 4,096 tokens only when both stacks declare it |
| §6 wrong for 16 GB cards (a resident 5080 IQ3_XXS lock exists); id workspaces under-counted; "D2–D5 add no buffers" false | §6 rebuilt with the id and `qlinear` workspaces and the 2,048-token lock; chunk sized by `max_step_tokens`; acceptance 5 per lock |
| Batched decode above 8 rows runs mmq with ids in graphs, so D2 changes it | acceptance 4 adds B16–64 and graphs-vs-eager; D2's map is written inside the helpers' launches |
| L0 sweeps J for 5 types only; the 4,096-row arena test for Q4_K / Q6_K only | both parametrized over every lock-class type, on sm_120 too |
| D5's "tile at ≥ 90%" variant is already the vendored rule | dropped |
| D1's chunk formula drops whole-`CHUNK` rounding; which pass rule | the formula restored; the rule stated (≥ `CHUNK` tokens and `row_invariant`) |
| D2 omits nty, the stream-k tiling branch, the per-slot helper | stated, with an L0 mutation for each |
| Lock counts | 7 MoE classes, 23 in all |
| README: the id helper cell; the prediction's label; plan's bitwise-only band | fixed: "the all-4,096 probe minus rows 2, 5, 6"; D1 + D2 alone ≈ 0.93× / 0.97× |
| Acceptance 3 wrong for the capped fit (F5b) | each configuration against main's own snapshot |
| Plants missing; D6's plumbing unspecified | plants for D1, D2 (L0), D3, D5; D6 rides the build context's `max_step_tokens` |
