# v1 — MoE prefill speed

Status: **design rev 1.2 (2026-10-18), step 0 measured on the A5000; one reviewer's two rounds applied (§10), not yet
challenged by the three-lens panel.** This is the
follow-up plan §10 opened when milestone X closed (`docs/v1-offload.md` §13, "Owner decision on the gate").

**The shortfall.** Holon's prefill trails FreeToken 0.92× / 0.88× at 4K / 32K on the RTX 5080's natural-fit offload
rows. FreeToken's all-resident prefill runs 1.6–2.0× Holon's on the RTX 5090.

**Step 0** is `spikes/moe_prefill/README.md` (`evidence/moe_prefill_rtx-a5000_2026-10-18.json`): Qwen3-30B-A3B Q4_K_M,
all-resident, on one RTX A5000, nsys per component, a per-call GEMM bench, and the source study of FreeToken's and
llama.cpp's prefill GEMMs.

## 1. Goal and acceptance

Close the MoE prefill gap to FreeToken without leaving Holon's exactness classes, and change no core.

**The cause** is the expert GEMMs' launch shape, not their arithmetic:
- Both engines run ggml's mmq with q8_1 activations and int8 MMA.
- Holon calls it at 512 rows on resident stacks.
- Its grouped grid launches every column tile of every slot.

The acceptance bars are **proposed, for the owner to confirm** (§9 Q1). Predictions are fixed here, before any code.

1. **The recorded shortfall rows.** RTX 5080, Qwen3-30B-A3B Q4_K_M at the natural fit (offload store, its lock),
   `spikes/x4_gate`'s method (3 interleaved rounds):
   - prefill ≥ 1.0× FreeToken at 4K and 32K;
   - the 1K row stays ≥ 1.5×;
   - every llama.cpp prefill row stays ≥ 2.1×.
   - Predicted: 4K ≈ 1.1–1.2×; 32K ≈ 0.97–1.02× after steps 1–3, ≥ 1.0× with D6. D1 does not apply there (the
     arena already takes 4,096-row calls); D2, D3 and D5 do.
2. **All-resident.** RTX 5090, same model:
   - prefill ≥ 1.0× FreeToken (`--moe-strategy fused`) at 4K and 32K, ≥ 0.95× at 1K;
   - ≥ 1.0× llama-server at every length (today 1.14–1.16×).
   - Predicted with D1–D3 and D5 (D4 if built): ≈ 1.03× / ≈ 1.015× (1.04× with D6) / ≈ 0.99× at 4K / 32K / 1K.
     With the bitwise items alone: D1 + D2 ≈ 0.93× / 0.97×, plus D4 ≈ 0.97× / 0.98× (§9 Q2).
   - Shipped Holon sits at the same ratios on the A5000 and the 5090 (0.63 / 0.49 at 1K / 4K), so the A5000's sums
     carry over.
   - The A5000 rows are recorded as the development measurement, not a bar.
3. **Exactness.**
   - D1, D2 (and D4 if built) are **bitwise**:
     - every configuration's snapshot is byte-identical to main's own: the 30B resident, the 30B offload store at the
       natural fit (still identical to resident, X's acceptance 2) and at the capped fit (F5b: run-identical ×3
       and identical to main's capped snapshot), and an 8B dense lock;
     - L0 holds row invariance across rows per call, tile width and tile map for every type a lock class uses (Q4_K,
       Q6_K, Q5_K, Q8_0, IQ2/IQ3, MXFP4, NVFP4; the W4A4 path on sm_120), on sm_86, sm_89 and sm_120.
   - D3 and D5 are **bounded**: L2 passing against the Tier B prefill floor, the bounded chunked check, and the locks
     they move propagated once at the close.
4. **Decode keeps its bits and its speed:**
   - steps of at most `mmvq_tokens` (8) rows are untouched (mmvq; `moe_router@1`'s ≤ 16-row path). A verify step at
     batch b has b·(1 + d) rows, so a larger one counts with the batched rows below;
   - batched decode above 8 rows runs mmq with ids inside graphs (the 30B locks run `max_seqs` 32–64), so D2 changes
     its launch but not its bits: the graphs-vs-eager bounded check is bitwise, and `holon_tg` at batch 16 / 32 / 64
     on the 5090 does not regress beyond noise;
   - `holon_tg` at batch 1, 1K / 4K, on the 5090 and the gate's offload decode rows stay within noise of X's close.
5. **Memory** (§6):
   - each lock's KV pool shrinks by at most §6's transient at its `max_step_tokens`, measured at startup and
     recorded per lock (the resident 5080 IQ3_XXS 30B and the 8B locks included);
   - the 5080 natural-fit offload lock keeps its expert-cache share within 1%;
   - D6's extra transient is recorded and left to tune.
6. **No core diff** (`check_core_diff.sh`), no ABI bump; a std convention at most (D1's `row_invariant` config key);
   suite time does not grow (lean tests: parametrize `test_ggml_cuda_moe.py`'s existing invariance tests).

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
| 2 | Grouped launch: J sized on all tokens; ceil(T/J) column tiles for every slot, 76–92% empty | −30 / −85 / −694 (to FreeToken's per-token rate) | D2 | bitwise |
| 3 | Calls of 4,096 against FreeToken's 8,192 rows | — / — / −239 | D6 | bitwise per row; the step size is a lock setting |
| 4 | Dense mmq at 512 rows per call | −4 / −25 / −188 (measured) | D5 | bounded |
| 5 | Router on prompt chunks: f32 cast, SGEMM, split-K, top-k per 512 rows (22 ms at 4K against FreeToken's bf16 2.3) | ≈ −3 / −13 / −107 | D3 | bounded |
| 6 | The f32 SwiGLU pass over gate/up's f32 output | −5 / −21 / −166 | D4 | bitwise |
| 7 | f32 combine and the f32 down input's q8_1 (FreeToken keeps bf16) | −4 / −14 / −112 | not taken (D7) | bounded |
| — | Holon ahead (attention, the id helper) | +1 / +24 / +437 | — | — |

**Holon and FreeToken share the GEMM.** FreeToken's prefill GEMM is ggml's mmq with ids as well:
- `layers/quantization/moe/gguf.py:92-127` → `ggml.mul_mat_id`, its vendored llama.cpp `8c2134a8` plus a patch;
- that patch adds J against the mean tokens per expert, a flat map of occupied tiles, no stream-k, and bf16 out.

**llama.cpp never falls back to dequant + cuBLAS** for k-quants on Turing+ GPUs: `ggml_cuda_should_use_mmq` →
`turing_mma_available`, and `ggml_cuda_mul_mat_id` runs one mmq call with ids above 8 tokens. Per call at 4,096 tokens,
Holon's grouped mmq is 1.11–1.30× FreeToken's on the same stacks and routing; FreeToken's is 1.20–1.43× the
dense-mmq bound.

## 3. Decisions

- **D1 — Resident stores take large expert passes.**
  - `Resident.pass_tokens` gives passes of `max(CHUNK, min(PASS, max_step_tokens) ÷ CHUNK × CHUNK)` tokens (up to
    4,096) on steps of at least `CHUNK` tokens. Rounding to whole `CHUNK`s keeps every router chunk where it starts
    today, including on verify steps longer than `max_step_tokens` (`executor.py:139`). The rule
    is simpler than the offload store's `large_passes`, which also needs a device-owned policy and the ring. It
    applies only when both stacks' kernels declare row invariance (below); otherwise it stays `CHUNK`.
  - kernels.ggml_cuda's `LinearExperts` takes `max(CHUNK, min(PASS_ROWS, ID_PAIRS ÷ top_k, max_step_tokens) ÷ CHUNK
    × CHUNK)` rows per mmq call on every stack, not only arenas: whole `CHUNK`s, so no tail of ≤ 8 rows ever switches
    to mmvq. `max_step_tokens` is the build context's option, so a 2,048-token lock sizes its workspaces for 2,048.
  - **Row invariance is declared.** A provider declares it in its `KernelSpec.config` (`"row_invariant": True`, as
    `config["act"]` is read today; no core prop). The CPU reference and `testing`'s CPU provider multiply per expert
    batch, which is not row-invariant (`docs/v1-offload.md` §14, step 3b), so they do not declare it and keep 512-token
    passes.
  - *Why:* this is the gap's largest row (−449 ms of 1,176 at 4K). It is bitwise: mmq with ids always runs one block
    per tile (no stream-k, X's D4 patch), so a row gets the same bits at any rows per call. L0 checks that for arenas
    on Q4_K and Q6_K (`test_an_arena_call_of_many_chunks_gives_the_chunks_rows`) and is parametrized over resident
    stacks and every lock-class type.
  - The router and top-k stay per 512-token chunk; a step's last partial chunk stays a pass of its own (`MoE._forward`
    as today).
- **D2 — A compact grouped launch in kernels.ggml_cuda's mmq with ids** (our own `vendor.patch` code; FreeToken's
  patch was studied, not copied).
  - **J** is chosen against the mean rows per used slot (T·k ÷ min(slots, T·k)), not against T.
  - All three id helpers (`mmid.cu`) also write a **tile map**: the (compact z entry, column tile) pairs that hold
    rows. The helpers, as `mmid.cuh` dispatches them:
    - the per-slot one, when n_z equals the expert count and experts × pairs ≤ 1M;
    - otherwise the compact block one, for any call of ≤ 16,384 pairs (small arena calls, and every prefill call of
      a 2,048-token lock at k = 8);
    - the device-wide one past that. They write it inside their own launches, with no host sync,
    so the map is graph-safe.
  - The grid is nty × the map's bound, ceil(T·k ÷ J) + min(slots, T·k). The bound is fixed per T, so a captured
    bucket replays any routing; a block past the map's count exits.
  - Every NVIDIA config is a stream-k config, so the map is decoded in the stream-k kernel's tiling branch (one block
    per tile, `gridDim.x` = nty × bound). K is never split and no fixup runs: X's forced tiling for ids, kept.
  - *Why:* this is the second row (−85 ms at 4K, −694 at 32K), and the bits do not move:
    - J does not change a row's bits (step 0's `--j-check`; L0's
      `test_expert_rows_do_not_depend_on_the_slots_or_the_tile_width` sweeps J from 8 to 128 for five types today, extended to every lock-class type);
    - the map changes only which block computes a tile;
    - with ids every tile is one block (no stream-k) since X's D4 patch.
  - The compact z entries (X's "compact id helper") stay, so placement invariance holds as is.
  - It also changes batched decode above 8 rows (mmq with ids in graphs; J falls from 64 to 8 at 64 rows): bits kept,
    speed measured (acceptance 4).
- **D3 — `moe_router@1` serves prompt chunks** (kernels.ggml_cuda, bounded).
  - Above 16 rows its instance runs one launch per chunk: a tiled f32 GEMM reading x in bf16 directly, so there is no
    f32 copy; the products are exact, accumulated in f32 in an order the kernel fixes and that does not depend on the
    row count. Then `moe_topk@1`'s selection.
  - It replaces the cast + cuBLAS SGEMM + split-K reduce + top-k (4 launches, 22 ms at 4K on the A5000).
  - The contract already allows a row-count-dependent order, and the f32 router stays f32: FreeToken's bf16 router is
    not Holon's numerics.
  - *Kept only* by M0.7's rule: ≥ max(0.5%, 3 sd) on some prefill row of a lock class; else reverted and recorded.
- **D4 — SwiGLU in gate/up's mmq store** (bitwise; **conditional**: built only if step 4 leaves a 4K row short).
  - **No repack.** The offload store refuses a kernel whose `prepare` repacks (`std/holon_std/moe/offload.py:164-168`:
    its host tiers hold the checkpoint's layout), so D4 keeps the stored layout: a tile of I/2 output rows loads gate
    rows i… and up rows inter + i… from two base offsets.
  - The write-back computes `silu(g)·u` with `swiglu_f32`'s exact f32 expression, after each half's per-expert scale
    (NVFP4's `scale2`, gate's then up's, as `_swiglu` applies them today), and writes [T, k, inter]: half the f32
    bytes.
  - Arenas, F5b's CPU reader and mmvq's fused gate are untouched.
- **D5 — Dense `qlinear` mmq calls of up to the step's rows** (bounded).
  - 512-row chunks become calls of up to `PASS_ROWS`.
  - A row's bits move because stream-k's K split points depend on the call's tile count (as they already do between
    a 100- and a 512-row call).
  - Measured −25 ms at 4K, −188 ms at 32K on the A5000. Kept by M0.7's rule.
  - Its workspaces grow with the rows per call (q8_1 at `in` × rows: ≈ +66 MB on the 8B locks), sized by
    `max_step_tokens` (§6).
- **D6 — 8,192-token passes** (**conditional**: only if step 4's 32K row misses).
  - `ID_PAIRS` and the vendored `GGML_CUDA_MM_IDS_MAX_PAIRS` are raised statically to 65,536 (the device-wide
    helper path, its L0 extended). The pass and the rows per call then follow the build context's `max_step_tokens`
    (D1's formula), a lock setting tune chooses per class; no new plumbing.
  - FreeToken's 8K chunks are 10% cheaper per token than 4K calls (row 3).
  - Its transient memory doubles (§6), which on an offload lock comes out of the expert cache: tune weighs the 32K
    prefill gain against decode.
- **D7 — Not taken (recorded):**
  - **Dequant to bf16 + a tensor-core grouped GEMM** (cuBLAS / Triton / CUTLASS). On the A5000 bf16 runs at 80–90
    TFLOPS against dense mmq's 48–61 TOPS, but dequantizing a layer costs 2.5 ms per pass, more than the 1.7 ms that
    bf16 saves at 4,096 tokens, and needs 1.2 GB of scratch per layer. On the GeForce lock classes, bf16 with f32
    accumulation runs at a quarter of the int8 rate (NVIDIA's figures; step 0 checks on a 5080), so mmq wins outright.
  - **Marlin-style W4A16**: in-register dequant, no scratch, but the same rate argument.
  - **bf16 expert intermediates** (FreeToken's; row 7, ≈ 2% at 4K): bounded, and against the design's f32 expert path
    (m02 D3). Left to the owner (§9 Q4).

## 4. Contracts and plugins (no core)

| Where | Today | Change | Class |
|---|---|---|---|
| `std.moe` `Resident.pass_tokens` (`units.py:187`) | `CHUNK` (512) | `max(CHUNK, min(PASS, max_step_tokens) ÷ CHUNK × CHUNK)` on steps of ≥ `CHUNK` tokens, when both stacks' kernels declare `row_invariant` | bitwise (D1) |
| `std.moe.units` `bind_experts`, `linear_experts@2` contract text | silent on rows per call | a provider may declare `config["row_invariant"]`: a row's result is the same at any rows per call (and, for arena providers, any placement: plan §10's pending invariance term). Declared, it is checked by that provider's L0; the CPU providers do not declare it | — |
| kernels.ggml_cuda `LinearExperts.chunk` (`plugin.py:416`) | 4,096 for arenas, 512 otherwise | `max(CHUNK, min(PASS_ROWS, ID_PAIRS ÷ top_k, max_step_tokens) ÷ CHUNK × CHUNK)` for every stack; declares `row_invariant` | bitwise (D1) |
| kernels.ggml_cuda `vendor.patch` (`mmq.cu` J_best, `mmid.cu` / `mmid.cuh` helpers, `mmq.cuh` launch and tiling branch) | J against T; grid nty × ceil(T/J) × n_z | J against the mean; a tile map from all three helpers; grid nty × the map's bound | bitwise (D2) |
| kernels.ggml_cuda `MoeRouter` (`plugin.py` `MoeRouter.__call__`) | > 16 rows: cast + f32 linear + `moe_topk` | its own tiled f32 router GEMM + selection | bounded (D3) |
| kernels.ggml_cuda `QLinear.__call__` and its workspace | mmq per 512 rows | per `min(PASS_ROWS, max_step_tokens)` rows | bounded (D5) |
| kernels.ggml_cuda `ID_PAIRS`, vendored `GGML_CUDA_MM_IDS_MAX_PAIRS`; `PASS_ROWS`, `std.moe` `PASS` | 32,768; 4,096 | 65,536; the pass and rows per call follow `max_step_tokens` up to 8,192 | bitwise per row (D6, conditional) |

**Generality:** nothing names the model.
- Pass and call sizes are generic constants.
- The tile map serves every ggml type with ids.
- The router kernel serves `moe_router@1`'s facts (any E ≤ 512, k ≤ 32, hidden % 512).

`kernels.triton_experts` (dense HF stacks) runs fixed-config tiles and gives row-invariant results, so D1's larger
pass is bitwise there too: its L0 adds the rows-per-call case.

## 5. Numerics and judge (predictions fixed now)

| Change | Class | Check | Predicted |
|---|---|---|---|
| D1 | bitwise | L0: resident `LinearExperts` at 4,096 rows per call gives each row a 512-row call's bits (the arena test, parametrized over bindings and every lock-class type); snapshot ×3 | byte-identical to main |
| D2 | bitwise | L0: rows bitwise across J (8…128), slots, all three helpers and the tile map against the full grid, every lock-class type; graph replay of a captured bucket on new routing (`test_graph_replay_follows_the_routing`); sm_86 here, sm_89 / sm_120 on cluster nodes, compute-sanitizer memcheck (`memcheck_ggml_cuda_moe.py`); snapshot ×3; graphs-vs-eager at batch 64 | identical; L0's mutations fail: "the map omits each slot's last partial tile" (rows unwritten), "the tiling branch keeps stream-k's `kbc` split" (K split, a fixup, bits move) |
| D3 | bounded | L0: logits within the f32 budget of the f64 product; selection bitwise `moe_topk` on its logits; row-invariant across 17- to 4,096-row calls. L2: Tier B prefill against the mmq floor | passes; KL p99 at or under the floor's (the decode router's precedent: 0.037 against 0.040). Plant `router_prefill_short_k` (the prefill path sums hidden less its last 256 values): L2 fails |
| D5 | bounded | L2 (Tier B prefill); `holon_judge bounded --chunked` | passes (stream-k split order only). Plant `qlinear_large_no_fixup` (calls above 512 rows skip the stream-k fixup): L2 fails |
| D4 | bitwise | L0: fused output bitwise `swiglu_f32` of the unfused mmq output, every glu type; snapshot | identical |
| D6 | the step composition changes (attention per step), so a lock-setting change | L2 at the lock's settings; snapshot run-identical ×3 | passes |
| plant `pass_eats_tail` (D1) | the step's last partial chunk merged into the previous pass, so a tail of ≤ 8 rows takes mmq instead of mmvq | the snapshot trace, with a pinned 1,027-token prompt (a 3-row tail inside one kernel chunk) | differs from main (the tail rows' path changes) |

The offload store's exactness (X's acceptance 2) needs nothing new: D1 does not touch arenas, and D2's tile map works
on the compact z entries the placement-invariance L0 already covers.

Locks: D1, D2 and D4 move no snapshot. Step 1 still re-propagates every class that binds kernels.ggml_cuda
(snapshot-identical: the dense 8B GGUF and NVFP4 locks included), because the plugin's source hash changes. The MoE
classes also record `config["row_invariant"]` in each `linear_experts` site's binding. D3 moves the 7 MoE lock classes; D5 moves every lock whose `qlinear` runs
mmq on prompt chunks (up to all 23 classes). They are propagated once at the close, with the L2 verdicts above;
approval is delegated for passing verdicts. Plants live in `judge/holon_judge/planted.py`, except L0's own mutations.

## 6. Memory

What a pass at T tokens holds (k = 8, inter 768, hidden 2,048): the f32 activations are transient; the id
workspaces (the q8_1 rows, T·k per stack, and the id maps) persist per `in`. Every row is in the startup profile
(`state_memory` runs the worst-case step), so the KV pool shrinks to match.

| Buffer | 512 (today) | 2,048 (5080 resident lock) | 4,096 (D1) | 8,192 (D6) |
|---|---|---|---|---|
| gate/up output [T, k, 2·inter] f32 | 25 MB | 101 MB | 201 MB | 403 MB |
| SwiGLU output [T, k, inter] f32 | 13 MB | 50 MB | 101 MB | 201 MB |
| down output [T, k, hidden] f32 | 34 MB | 134 MB | 268 MB | 537 MB |
| id workspaces (q8_1 of gate/up in = 2,048 and down in = 768, id maps) | ≈ 13 MB | ≈ 52 MB | ≈ 104 MB | ≈ 208 MB |
| **Total** | **≈ 84 MB** | **≈ 337 MB** | **≈ 674 MB** | **≈ 1.35 GB** |

- D4 removes the gate/up row.
- **D1 on resident locks:**
  - a resident 30B at 4,096 tokens per step (the 3090 / 4090 / 5090 locks) gives up ≈ 590 MB, ≈ 6.0K KV tokens at
    96 KiB each;
  - the 16 GB RTX 5080 resident IQ3_XXS lock (`max_step_tokens` 2,048) gives up ≈ 253 MB, ≈ 2.6K KV tokens;
  - acceptance 5 records both, measured.
- **D5** grows `qlinear`'s q8_1 workspaces from 512 rows to the step's: ≈ +25 MB on the 30B, ≈ +66 MB on the 8B locks
  at 4,096.
- **Offload locks** already budget 4,096-token passes (D8), so D1 changes nothing on the 5080's natural-fit lock.
  D6 would take ≈ 0.68 GB from its expert cache (≈ 236 of ≈ 3,240 cached experts, ≈ 7%); tune decides.

## 7. Steps

| Step | Work | Checks | Days |
|---|---|---|---|
| 0 | **Done on the A5000** (spike). On the bar cards (one persistent container per desktop): `prefill.py` on the 5090 resident (shipped, probes, FreeToken), the 5080 natural-fit offload rows with nsys, and `gemm.py` (incl. the bf16 against int8 rate) on both | predictions of §1 re-stated per class before code; any row predicted under its bar goes to the owner with the step-4 / D6 plan | 1 |
| 1 | D1 (std + kernels.ggml_cuda) and the `row_invariant` declaration | L0 parametrized (bindings × lock-class types); plant `pass_eats_tail`; snapshot ×3 identical to main (A5000); A5000 rows; KV tokens per resident lock | 1 |
| 2 | D2 (`vendor.patch`, `vendor.sh --check`, provenance unchanged: our code) | L0 + its two mutations on sm_86 / 89 / 120, over all three id helpers (`HELPERS` gains the device-wide path), memcheck; `gemm.py` per call ≤ FreeToken's at 512–4,096 tokens on the A5000; snapshot ×3; batched decode B16–64 (`holon_tg`) | 2–3 |
| 3 | D3, D5 (bounded), each measured alone | keep-or-revert (M0.7 rule); L2; bounded `--chunked` | 2 |
| 4 | The bars: 5080 natural-fit offload (x4_gate's method, 3 rounds), 5090 resident, A5000 recorded | acceptance 1, 2, 4, 5 | 1 |
| 5 | Only if 32K misses: D6, tune's `max_step_tokens` per class | L0 at 65,536 pairs; L2 at the lock's settings; acceptance 1–2 again | 1–2 |
| 6 | Only if 4K misses: D4 | L0; snapshot; acceptance again | 2–3 |
| 7 | Close: one propagation (bounded items), `pytest` on a GPU, `scripts/ci/run_all.sh`, panel code review, plan §10 status | acceptance 3, 6 | 1 |

The panel (three lenses, as X's) challenges this design before step 1, and the code at step 4 and at the close.

## 8. Risks and deferred

- **The bars have no margin at 32K.**
  - On the 5080 the offload prefill is compute-bound only while compute exceeds the ring's stream. Holon streams the
    non-resident ≈ 47–53% (≈ 8.9 GB, ≈ 190 ms at 47 GB/s per 4K pass) behind ≈ 330 ms of predicted compute, so it stays
    compute-bound.
  - At 32K, attention (52% of the A5000's prefill) dominates. Holon's is already 6% faster than FreeToken's, and the
    expert rows close only to ≈ 1.0×. D6 is the planned lever. Beyond it is an attention item (FlashInfer's FA2 at
    4K-token steps), not in this milestone.
- **J-invariance on other types and arches.**
  - Step 0 checked Q4_K and Q6_K on sm_86; L0 sweeps J per type on every arch it runs.
  - A type or arch whose bits move with J keeps J against T. That is a per-type switch in the provider, recorded,
    never silent.
- **FreeToken is a moving target.** The bars are against `21d9578`, the pin X's gate used.
- **The A5000 is a professional card.** Its bf16 / int8 balance differs from the lock classes; step 0 on the bar
  cards decides D7's rejection there.
- **CPU tests.** The CPU reference multiplies per expert batch, which is not row-invariant (`docs/v1-offload.md`
  §14, step 3b). It does not declare `row_invariant`, so CPU engines keep 512-token passes and their tests keep their
  bits; only providers whose L0 proves the invariance get large passes.
- **Deferred:** bf16 expert intermediates (D7); an attention item for 32K+; a W4A16 path for professional cards (A5000,
  RTX PRO 6000) if a datacenter or workstation class ever gets locks.

## 9. Open questions for the owner

1. **Bars.** Are acceptance 1 (5080 offload, ≥ 1.0× FreeToken at 4K / 32K) and acceptance 2 (5090 resident, ≥ 1.0× at
   4K / 32K, ≥ 0.95× at 1K) the milestone's bars?
2. **Bounded items.** D3 and D5 move locks: D5 up to all 23 classes, D3 the 7 MoE ones. Is one propagation at the
   close acceptable, or should the milestone stay bitwise-only? Bitwise-only, D1 + D2 are predicted at ≈ 0.93× /
   0.97× at 4K / 32K on the A5000, and ≈ 0.97× / 0.98× with D4: under the bar.
3. **D6.** May 8,192-token steps become a lock setting on MoE classes if 32K misses, at ≈ 0.68 GB more VRAM per
   pass (on the 5080 offload lock, ≈ 7% of the cached experts)?
4. **Row 7.** Should the expert path stay f32 (m02 D3), or are bf16 intermediates (FreeToken's, ≈ 2% at 4K) open as a
   bounded option?

## 10. Review log

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
