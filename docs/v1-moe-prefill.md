# v1 — MoE prefill speed

Status: **design rev 2 (2026-10-18).** Step 0 is measured on the A5000. Rev 2 answers the three-lens panel (A
correctness, B architecture, C performance: 3 blockers, 19 majors, 23 minors; §10) with the lead's decisions. It awaits
the panel's re-review.

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

1. **The recorded shortfall rows.** RTX 5080, Qwen3-30B-A3B Q4_K_M at the natural fit (the offload lock, re-filled
   and re-pinned at the close, §5.3):
   - prefill ≥ 1.0× FreeToken at 4K and 32K;
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
     - L0 holds every type × SM pair the locks use (derived from `plans/`), plus FN's types and shape.
   - **D3, D5 bounded:** L2 passing against the Tier B prefill floor, plus the bounded chunked check; plants near the
     bound must fail.
5. **Decode keeps its bits and its speed.**
   - **Bits:**
     - steps of ≤ `mmvq_tokens` (8) rows are untouched;
     - D3 applies only to router calls of ≥ `CHUNK` rows, so batched decode and verify keep main's router bits;
     - batched decode above 8 rows runs mmq with ids inside graphs, so D2 changes its launch but not its bits
       (graphs-vs-eager bitwise).
   - **Speed:** a same-session ABBA against main's build, `holon_tg` at batch 1 (1K / 4K) and batch 16 / 32 / 64 on
     the 5090, and the gate's offload decode rows. Each must stay within ≤ max(0.5%, 2 sd).
6. **Memory** (§6):
   - every lock's startup KV pool stays ≥ its `max_model_len` and its recorded capability (each pass size is capped
     per class to keep this);
   - startup KV tokens are recorded per lock, main against the branch;
   - the 5080 offload lock's expert-cache share stays within 1% after its re-fill.
7. **No core diff** and no ABI bump. One std minor: `pass_rows` (§4). Suite time does not grow (lean tests, §5.1).

### 1.1 Predictions per item set (A5000 step 0; the bar cards restate them at step 0)

The base is the measured all-4,096 probe (D1 and D5 in, A5000 GPU-busy ms):

| Length | Probe | FreeToken default | FreeToken gated at 4,096 |
|---|---|---|---|
| 1K | 212 | 172 | 172 |
| 4K | 686 | 584 | 584 |
| 32K | 10,013 | 9,184 | ≈ 9,423 (its 4K-call GEMM rate per token, step 0's per-call bench) |

Item savings at 1K / 4K / 32K (ms):

| Item | 1K | 4K | 32K | Note |
|---|---|---|---|---|
| D2 | −26…−30 | −70…−85 | −577…−694 | the lower end removes FreeToken's bf16-store advantage (C-M1; step 0 splits it with a probe) |
| D3 | −3 | −13 | −107 | |
| D4 | −5 | −21 | −166 | upper bound: the fused write-back assumed free |
| D6 | — | — | −239 | |

The offload rows scale the A5000 fractions from today's measured 5080 ratios (0.92× / 0.88×). There D1 does not apply
(the arena already makes 4,096-row calls), and the A5000 base is the 4,096-expert-row arm with dense at 512 rows.

**Rows at or below their bar are bold** (the "≤ bar" column). Rows within 3% of the bar take ≥ 5 rounds (§5.2).

| Row (bar) | D1–D3 + D5 | + D4 (rev 2's set) | + D6 | ≤ bar |
|---|---|---|---|---|
| 5090 resident 1K (≥ 0.95×) | **0.94–0.96×** | 0.96–0.99× | — | D1–D3 + D5 |
| 5090 resident 4K (≥ 1.0×) | **0.97–0.99×** | **1.00**–1.03× | — | both: at the bar at the low end |
| 5090 resident 32K, gated (≥ 1.0×) | 1.01–1.02× | 1.03–1.04× | 1.03–1.04× (FreeToken also at 8,192) | within 3% |
| 5090 resident 32K, FreeToken default (reported) | **0.98–1.00×** | **1.00**–1.02× | 1.03–1.04× | — |
| 5080 offload 4K (≥ 1.0×) | 1.06–1.11× | 1.08–1.15× | — | — |
| 5080 offload 32K, gated (≥ 1.0×) | **0.99–1.00×** | 1.01–1.02× | not built on this lock (§9 O1) | within 3% |
| 5080 offload 32K, FreeToken default (reported) | **0.96–0.98×** | **0.98–0.99×** | 1.00–1.02× at ≈ −3.4% decode at 32K | §9 O1 |
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

## 3. Decisions

- **D0 — One source for the pass size: the bound kernel's `pass_rows`** (lead decision; B-M1, B-M3, C-B3).
  - **What it is.** An optional attribute of a bound `linear_experts@2` / `linear_experts_arena@1` instance: the most
    rows per call the instance gives at the bits of its `CHUNK`-row calls (§4 states the promise). The default is
    `CHUNK`.
  - **Who sets it.** The provider sets it per type, arch and path, after its L0 holds there. For mmq with ids in
    kernels.ggml_cuda it is `max(CHUNK, min(PASS_ROWS, ID_PAIRS ÷ top_k, its cap) ÷ CHUNK × CHUNK)`. A path that is
    not proven keeps `CHUNK`. FN's IQ1_M fallback, if it is ever needed, sets its own (D9).
  - **Its cap** is `max_step_tokens` from the build context (missing: no cap), and below that the provider's plugin
    option `ggml_cuda.pass_rows`.
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
  - **CPU providers** (the oracle and `testing`'s) multiply per expert batch and keep the default `CHUNK`, so CPU
    tests keep their bits.
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
- **D3 — `moe_router@1` serves prompt chunks of ≥ `CHUNK` rows** (kernels.ggml_cuda, bounded; A-1).
  - **Path.** Calls of ≥ 512 rows run one launch: a tiled f32 GEMM reading bf16 x directly (no f32 copy), with f32
    fma chains from x's exact values in an order the kernel fixes, independent of the row count, then `moe_topk@1`'s
    selection. Calls of 17–511 rows keep today's cast + f32 linear + `moe_topk`; calls of ≤ 16 rows keep the fused
    decode path.
  - **Decode keeps its bits.** Batched decode (17–64 rows) and verify keep main's router bits. A mixed step's rows
    already take their call's path (`chunk_with_decode`), as today.
  - **What it replaces:** cast + cuBLAS SGEMM + split-K reduce + top-k, 4 launches and 22 ms at 4K on the A5000.
  - **Kept** by M0.7's rule: ≥ max(0.5%, 3 sd) on some prefill row of a lock class.
- **D4 — SwiGLU in gate/up's mmq store** (bitwise; **in the set**, lead decision C-B1).
  - **No repack.** The offload store refuses kernels that repack (`offload.py:164-168`). Instead a tile of I/2 output
    rows loads gate rows i… and up rows inter + i… from two base offsets of the stored layout.
  - **Fused write-back.** For one output the g and u values sit 64 rows apart, in different warps' fragments, so the
    write-back exchanges them through shared memory. It applies each half's per-expert `scale2` (NVFP4, gate's then
    up's) with `swiglu_f32`'s plain fast-math `*` (`csrc/api.cu:114-128`, flush-to-zero), not `mmq_store`'s
    `ggml_cuda_mul_rn`. It writes `silu(g)·u` as [T, k, inter].
  - **Untouched:** arenas, F5b's CPU reader and mmvq's fused gate.
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
  - **Cost:** +0.47 GB of live peak (§6). On the 5080 offload lock that is ≈ 166 cached experts (5%), predicted
    ≈ −3.4% decode at 32K (C-B2, from x4_gate's measured share-to-decode slope).
  - **Batched locks:** an 8K prompt step also stalls decode twice as long (inter-token gap p99), since
    `chunk_with_decode` is 0.
  - **Default:** no D6 on offload locks; decode wins (lead decision).
- **D7 — Not taken (recorded):**
  - **Dequant to bf16 + a tensor-core grouped GEMM** (cuBLAS / Triton / CUTLASS). Rejected on achieved rates
    (C-M2), not peaks. On the A5000:
    - mmq reaches ≈ 27% of its int8 peak (48–61 TOPS) and cuBLAS bf16 ≈ 78% (80–90 TFLOPS);
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
  - This track owns `LinearExperts`' prefill dispatch and `vendor.patch`. Before FN's step 4 it lands steps 1–2
    (D0's seam and D2) and step 6 (D9). FN depends on all three.
  - Every hunk carries a "Holon:" marker, and `test_link`'s marker check (`test_ggml_cuda_moe.py:234-235`) adds
    `mmq.cu` and `mmq.cuh` (B-m7).

- **D9 — IQ1_M's native mmq tile loader** (a `vendor.patch` hunk for FN; bitwise in q8_1's class).
  - **Why it fits int8.** IQ1_M weights scaled by 8 are 8·g ± 1, which fits int8. Their per-16 scales are
    d·(2s + 1) ÷ 8, the same layout family as IQ2_XS and IQ2_S, whose mmq loaders it follows. So IQ1_M gets q8_1
    activations, int8 MMA, and every row-invariance and placement L0 of this plan.
  - **The vendored ggml lacks it today:** it has IQ1_M's mmvq but no mmq instance (FN `v1-flash-next.md:73`).
  - **Fallback**, if the loader fails its L0 or its speed: a grouped dequant of 16 experts per launch into bf16 tiles,
    behind D0's seam with that path's own `pass_rows`. It is bounded, judged against its own floor.
  - **Ordering.** D9 is a later step of this track (step 6). It lands before FN's step 4, but outside step 5's
    18-class propagation round: no current lock uses IQ1_M, so no lock moves.

## 4. Contracts and plugins (no core)

| Where | Today | Change | Class |
|---|---|---|---|
| `std.moe.kernels` `linear_experts@2` / `linear_experts_arena@1` text | silent on rows per call | **std minor** (additive optional term, `docs/design.md:338`; B-m1): an instance may expose `pass_rows` (int, a multiple of `CHUNK`, default `CHUNK`) | — |
| `std.moe` `Resident.pass_tokens` (`units.py:187`), `std.moe.offload` `pass_tokens` (`offload.py:119-121`) | `CHUNK`; offload `PASS` | min over the step's stacks' `pass_rows` (`PASS` retired) | bitwise (D1) |
| kernels.ggml_cuda `LinearExperts` (`plugin.py:405-480`) | chunk 4,096 for arenas, 512 otherwise | `pass_rows` per type / arch / path, capped by `max_step_tokens` and `ggml_cuda.pass_rows`; workspaces sized for it | bitwise (D0, D1) |
| kernels.ggml_cuda `vendor.patch` (`mmq.cu` J_best, `mmq.cuh` launch, the tile decode and the fixup gate) | J against T; grid nty × ⌈T ÷ J⌉ × n_z | J against T·k ÷ n_z; grid nty × (n_z + ⌊T·k ÷ J⌋ + 1); binary-search decode; no ids fixup | bitwise (D2) |
| the same, gate/up's store | f32 [T, k, 2·inter], then `swiglu_f32` | fused `silu(g)·u` [T, k, inter] | bitwise (D4) |
| the same, a new type's tile loader (IQ1_M; `TYPES` and the mmq type switch) | IQ1_M: mmvq only | mmq tile loader (×8 int8 values, per-16 scales) | bitwise in q8_1's class (D9) |
| kernels.ggml_cuda `MoeRouter` | > 16 rows: cast + f32 linear + `moe_topk` | ≥ `CHUNK` rows: its own f32 router GEMM + selection | bounded (D3) |
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
| D2 | bitwise | L0 below, with the output **NaN-poisoned** before each call (A-minor) and two mutations: the decode skips each entry's last partial tile (rows stay NaN), and the ids path keeps `fixup_needed` (A-3); graph replay of a captured bucket on new routing; graphs-vs-eager at batch 64; memcheck everywhere, racecheck on sm_120 | identical; both mutations fail |
| D4 | bitwise | L0: the fused output is bitwise `swiglu_f32` of the unfused output, for every glu type, with subnormal and NVFP4-scaled inputs | identical |
| D3 | bounded | L0 at 512–4,096 rows: logits within the f32 budget of the f64 product; selection bitwise `moe_topk`; row-invariant across ≥ 512-row calls. L2: Tier B prefill against the mmq floor. **Plants:** `router_prefill_bf16_logits` (logits rounded to bf16: near the bound) and `router_prefill_short_k` (hidden less its last 256: far) | passes; KL p99 at or under the floor (the decode router's precedent: 0.037 against 0.040). The near plant's verdict is predicted at step 3 from a CPU estimate before the GPU run; the far plant fails |
| D5 | bounded | L2 (Tier B prefill); `holon_judge bounded --chunked`. **Plants:** `qlinear_partials_bf16` (stream-k partials rounded to bf16 before the fixup: near) and `qlinear_large_no_fixup` (far). The fixup launches are counted so the record shows each plant fired on that card and shape (A-5) | passes; the far plant fails where it fires; the near plant's verdict is predicted as D3's |

### 5.1 L0 coverage and lean tests (lead decision; A-4, B-m4)

**Types and SMs** are derived from `plans/` at step 1 and recorded:
- sm_86 (3090): q4_k, q6_k;
- sm_89 (4090): q4_k, q6_k, mxfp4, nvfp4;
- sm_120 (5080 / 5090): q4_k, q5_k, q6_k, iq3_s, iq3_xxs, iq4_xs.

- **FN's requirements** (FN rev 2, D1′):
  - its types are IQ2_S, IQ2_XXS, IQ1_M (D9, step 6), Q2_0, IQ3_XXS, IQ3_S, IQ4_NL and IQ4_XS; Q2_0 joins `TYPES` with
    FN or with this track's step 6, whichever lands first;
  - its MoE shape is E = 512, top-10, H = 2,560, I = 640, which runs the device-wide helper. At top-10 `pass_rows` is
    3,072: ⌊32,768 ÷ 10⌋ rounded down to whole `CHUNK`s.
  - Both go into the same tests below, with no new suites.
- No lock sets `quant.act`, so W4A4 is not parametrized.

**Where the checks go, without new suites:**
- `test_linear_experts_every_type_at_prefill_chunks` (`test_ggml_cuda_moe.py:254-263`) gains a 4,608-row against
  512-row comparison (every type, small stack). It also gains one 128-expert stack (q4_k, q6_k) at T = 512 / 2,048 /
  4,096, which crosses all three helpers (A-minor).
- `test_an_arena_call_of_many_chunks…` stays at two types.
- `PLACED` (:417) gains only the lock types it lacks (iq3_s, iq4_xs, q5_k, mxfp4, nvfp4), plus FN's (iq1_m, q2_0,
  iq4_nl). FN's shape is one more stack in the every-type test.
- The device-wide helper is not added to `HELPERS` (it is asserted at :516).
- D3's router is tested at 512–4,096 rows.
- `pytest --durations` before and after is recorded in the step's record.

### 5.2 Measurement (lead decision; C-M5, C-M6, release.md)

- **Arms.** A same-session ABBA against main's build, with both engines timed at the same boundary: the time from
  submission to the first token, at the engine API.
- **Prompts.** A fixed set chosen before code: ≥ 5 prompts at 1K and 4K and ≥ 3 at 32K, the same for every engine.
- **Rounds.** ≥ 3 rounds, rising to ≥ 5 for rows predicted within 3% of their bar (§1.1).
- **Statistic.** The median over rounds of each round's geomean of the paired ratios. A difference under 2 sd of the
  rounds is reported as "≈" and escalated.
- **Decode tolerance:** ≤ max(0.5%, 2 sd).
- **Comparisons against main** pin pool sizes, or record that no preemption or eviction occurred (A-minor: D1's KV
  shrink could change admission).

### 5.3 Locks (lead decision; A-6, B-M6, C-M7)

- **Lands first.** This track needs no core change. It closes with its own propagation round covering D1–D5, its L2
  verdicts recorded as its own evidence, before K or FN rebase onto it.
- **Scope:** the 18 classes that pin kernels.ggml_cuda or std.moe. Five do not (Qwen3-8B bf16 ×3, FP8 ×2).
  - D1 / D2 / D4 move no snapshot.
  - D3 moves the 7 MoE classes; D5 moves every class whose `qlinear` runs mmq on prompt chunks.
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
- **D5** adds ≈ 25 MB (30B) or ≈ 66 MB (8B) of `qlinear` workspace at 4,096.
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
| 0 | **A5000: done.** **Bar cards** (5090, 5080; one persistent container each): achieved mmq TOPS against cuBLAS bf16 TFLOPS at T·k = 32K / 64K, dequant ms per layer, the break-even pass size (D7); nsys breakdowns of resident (5090) and natural-fit offload (5080) at 1K / 4K / 32K, attention included; row 2's I/O split (`gemm.py --freetoken` with f32 activations, so its store is f32 too); the batched-decode J sweep at T = 16–64 (C-M6); startup KV tokens on the 16 GB lock and one 24 GB lock with the `--pass-tokens` / `--expert-rows` probes; the `peak_alloc` rise explained; the fixed prompt set | every §1.1 cell restated per class from these; rows at or below their bar to the owner before step 1 | 1–2 |
| 1 | D0 + D1 (`pass_rows`, the std minor, both stores, the shared `holon_testing` helper); `ggml_cuda.pass_rows` per class from step 0 | §5.1's L0; the CPU call-size test; the logits-hash identity on the A5000; snapshots against main (no lock writes); KV tokens per class | 1–1.5 |
| 2 | D2 (`vendor.patch`, `vendor.sh --check`, "Holon:" markers, `test_link`) | L0 + NaN poison + both mutations on sm_86 / 89 / 120, memcheck, racecheck on sm_120; per-call `gemm.py` at equal I/O dtype, interleaved repeats (C-m4), ≤ FreeToken's at 512–4,096 tokens; batched decode B16–64 ABBA | 2–3 |
| 3 | D4, then D3 and D5, each measured alone | D4's L0 and identity; D3 / D5 keep-or-revert by M0.7's rule, L2 with near and far plants | 3–4 |
| 4 | The bars on the bar cards (§5.2's method) | acceptance 1, 2, 3, 5, 6 | 1–2 |
| 5 | Close: one propagation round over the 18 classes (D1–D5, `pass_rows`), the 5080 offload lock re-filled and re-pinned, `pytest` on a GPU, `scripts/ci/run_all.sh`, panel code review, plan §10 status | acceptance 4, 7; its L2 verdicts as its own evidence | 1–2 |
| 6 | D9: IQ1_M's mmq tile loader (or its fallback); before FN's step 4, outside step 5's propagation (no lock uses IQ1_M) | L0 against `q8_1.linear_experts` for IQ1_M; the §5.1 invariance tests with iq1_m and FN's shape; memcheck on sm_120; per-call speed against IQ2_S's loader | 2–3 |

The panel reviews this revision before step 1, the code at step 4, and again at the close.

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
  - Then the 32K row is predicted at ≈ 1.01–1.02× against the gated FreeToken arm, and ≈ 0.98–0.99× against
    FreeToken's ungated 8,192-token default (reported, not gated).
  - With D6: ≈ 1.00–1.02× against the default, at ≈ −3.4% decode at 32K (the FreeToken decode row 1.03× → ≈ 0.995×).
- **O2: the bars and arms.**
  - Confirm acceptance 1–2, with FreeToken's gated arm at the lock's step budget and its default reported only.
  - Rows at or under their bar in §1.1: 5090 1K and 4K without D4; 5090 4K and the 5080 32K-gated row at the bar's
    low end even with D4. They are re-sent after the bar cards' step 0.
- **O3: pass size per class.** `ggml_cuda.pass_rows` lowers the step-size gain on classes whose KV capacity cannot
  absorb 4,096 (likely the 24 GB 30B classes at 2,048 and the 16 GB class at 512 or 2,048). The values are set by
  hand at the close from step 0's measurements.
- **O4: bf16 expert intermediates** (FreeToken's, ≈ 2.4% at 4K) stay out (m02 D3). The handicap is stated next to the
  bars.
- **For the lead (from B-M6, outside this track):** K and FN both claim ABI 1.4 (`v1-kv-tiers.md:450`,
  `v1-flash-next.md:376`).

## 10. Review log

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
- IQ1_M's native mmq tile loader becomes D9, step 6, before FN's step 4 and outside the propagation round;
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
