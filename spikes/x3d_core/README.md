# x3d_core — step 3d: the step loop's per-step index and copy kernels

Records: `evidence/x3d_core_rtx-a5000_2026-10-17.json`, `evidence/x3d_core_rtx-5090_2026-10-17.json`. Design:
`docs/dev/v1-offload.md` §14 "Step 3d as built". The gap these close: step 3c's "Not done" (`spikes/x3c_kernels/README.md`):
≈ 30 per-step index and copy kernels issued by `core/holon/core/executor.py`.

## What changed (one commit each; no ABI change, `check_abi --base main` adds nothing to the branch's 1.3)

| Change | Commit | Class | What |
|---|---|---|---|
| prep | `b60328f` | bitwise | Without a drafter no row verifies, so every row's start is the host's (design §7.3). The executor stages a step's positions, page lengths and page write indices (from the host block tables, `StateManager.pages(…, positions)`) with its other inputs, and the store's slots and lengths per logits row. The device prep keeps the SeqTable start write and moves only what the device alone holds: the last tokens into the inputs, by one gather and one index write (rows whose first input is the host's write a scratch element past the step's tokens). The head reads the trunk's output as is when every token is a logits row (decode, verify). Engines with a drafter derive on the device as before (`Executor._derive`). |
| upload | `5452833` | bitwise | Steps of at most a decode / verify step's tokens (every graphed one) stage into an arena of that size (`Executor.small_uploader`), and the uploader lays its fields out smallest first: such a step's written ranges merge into one host-to-device copy. Prefill steps keep the arena of `max_step_tokens`. |
| telemetry | measured as a commit on top of `5452833`, then **dropped** | bitwise | The shared objects' telemetry gathered as bytes on the device (one `cat`) and copied to the host once, instead of one copy per buffer (the offload store's 11). It does not pay (below); the commit was removed from the branch. |

Also on the branch, separately: `cdd4785`, two std nits from step 3c's panel (the ffn unit interface's optional
member `add_residual`; a prepared handle's `.x` in the `linear_experts@2` / `linear_experts_arena@1` contract text).

## Method

- **Vehicles:** Qwen3-30B-A3B Q4_K_M (resident, and the offload store at `moe.cache=*=0.47` under F11's `lfu`) and
  Qwen3-8B Q4_K_M (dense: the changes are the core's, not MoE's), batch 1, graphed decode.
- **Inventory (`profile.py`, `inventory.py`):** nsys with `--cuda-graph-trace=node`, 64 graphed decode steps after a
  1,024-token prompt. Four eager steps run first with an NVTX range around each step-loop phase (`prepare`, each
  planner's `plan_bounds` and `prepare_device`, `trunk`, `logits`, `sample`, `store`, the uploader's copy, `observe`,
  `writes`); a graph replays the eager step's launches in order, so each graphed kernel takes the phase of the eager
  kernel at its index (names checked). Durations are profiled: node tracing inflates small kernels, so a phase's µs
  over-states what removing it saves (compare the A/B below).
- **`ab.py`:** M0.7's keep-or-revert method, as step 3c's: one engine per deployment, the changes switched in process
  (`base` = bdd8422's executor methods and uploader layout, `base.py`; each switch drops the graphs), the per-step
  wall over 256 graphed decode steps after a prompt of 1,024 / 4,096 random tokens, and one eager prompt of 512 /
  4,096 tokens; a warm-up round, then 8 rounds in ABBA order. A change's saving is against the previous arm, its sd
  that of the per-round differences. **Kept if ≥ max(0.5%, 3 sd) on some row.**
- **End to end:** `bench/holon_tg.py` (tg128 at depth 1K / 4K, 3 repetitions), base `bdd8422` against the final
  commit, one process per run, 4 rounds per arm in ABBA order.
- **Hardware:** RTX A5000 (sm_86); RTX 5090 desktop (sm_120a), because the A5000 cannot resolve 0.5% (its rows drift
  ≈ 2% over a run as the card warms, sd 0.13–0.5% per round). The first 5090 node's GPU fell off the bus (Xid 79)
  during the 8B run; that run was repeated on a second node.

## Results

**The per-step inventory (A5000, Qwen3-30B-A3B resident, batch-1 graphed decode; profiled µs incl. the gaps before
each op).** Qwen3-8B has the same rows and counts; the offload store adds `observe` and `writes` (A5000, at `*=0.47`).

| Phase (issuer) | Before: what, per step | µs | After | µs |
|---|---|---|---|---|
| upload (core: `Uploader.end`) | 6 H2D copies (8, 136, 8, 8, 8, 193 B) | 12.8 | 1 H2D copy (1,104 B) | 1.6 |
| prepare (core: `Executor._prepare`) | 19 kernels + 2 D2D copies: SeqTable start gather / `where` / write / re-gather; positions (gather, add, copy); lengths (add, cast); first inputs (2 gathers, `where`, write); slot per token (gather); write index (div, 2-D table gather, cast, mul, mod, add, copy) | 52.3 | 3 kernels: last-token gather, its write into the inputs, the SeqTable start write | 11.9 |
| logits (core: `Executor._logits`) | the logits-row gather `h[logits_index]` | 3.9 | none | — |
| store (core: `Executor._store`) | 5 kernels: slots of logits rows (gather), `last_token` write, lengths (gather, cast), `num_computed` write | 11.1 | 2 kernels: the two writes | 4.8 |
| launch (core) | the tokens' D2H copy (8 B) | 5.8 | same | 6.3 |
| observe (core, offload only) | 11 D2H copies + 1 copy kernel (the store's telemetry) | 45.4 | same (change dropped) | 45.0 |
| writes (core, offload only) | 1 H2D copy (the store's `before_launch`) | 2.1 | same | 2.2 |
| plan_bounds (FlashInfer's plan, outside the graph) | 1 copy kernel | 3.1 | same | 8.1 |
| prepare_device (std gqa planner) | write unit / offset (2 kernels) + FlashInfer's in-graph `prepare` (4 B D2D) | 4.9 | same | 6.8 |
| sample (std sampler, greedy) | parameter-row gather (unused when greedy) + memset + argmax | 14.8 | same | 14.4 |

Core-issued ops per decode step: **25 kernels + 2 D2D + 6 H2D + 1 D2H → 5 kernels + 1 H2D + 1 D2H** (offload: plus
the unchanged 12 `observe` and 1 `writes` ops). Qwen3-8B: the same counts (prepare 56.3 → 13.5 µs, store 12.6 → 5.5,
upload 13.2 → 1.4). The before / after rows of each model are in the records (`inventory`).

**Per change, decode ms per step saved (% of the step ± sd), 1K / 4K:**

| Change | RTX 5090 30B resident | RTX 5090 30B offload 47% | RTX 5090 8B | A5000 30B resident | A5000 30B offload 47% | A5000 8B | Verdict |
|---|---|---|---|---|---|---|---|
| prep | 0.022 / 0.023 (0.91 ± 0.03 / 0.86 ± 0.03%) | 0.019 / 0.018 (0.68 ± 0.06 / 0.61 ± 0.04%) | 0.022 / 0.022 (0.58 ± 0.05 / 0.54 ± 0.09%) | 0.022 / 0.020 (0.42 ± 0.14 / 0.34 ± 0.13%) | 0.029 / 0.019 (0.50 ± 0.53 / 0.31 ± 0.28%) | 0.024 / 0.019 (0.27 ± 0.19 / 0.20 ± 0.17%) | **kept** (every 5090 row) |
| upload | 0.013 / 0.014 (0.55 ± 0.04 / 0.53 ± 0.03%) | 0.021 / 0.021 (0.78 ± 0.04 / 0.72 ± 0.01%) | 0.013 / 0.013 (0.36 ± 0.05 / 0.33 ± 0.03%) | 0.003 / 0.002 (0.06 / 0.03%, sd 0.2) | 0.008 / 0.007 (0.14 / 0.12%, sd 0.4) | 0.008 / 0.008 (0.09 / 0.08%, sd 0.15) | **kept** (5090 30B rows) |
| telemetry | — (no telemetry) | 0.001 / 0.002 (0.04 ± 0.02 / 0.05 ± 0.05%) | — | — | 0.002 / 0.010 (0.04 ± 0.31 / 0.17 ± 0.48%) | — | **dropped** |
| **prep + upload** | **0.036 / 0.037** (2.44 → 2.40 ms at 1K) | **0.040 / 0.039** | **0.035 / 0.035** (3.74 → 3.70 ms at 1K) | 0.025 / 0.022 | 0.036 / 0.026 | 0.031 / 0.027 | |

Prefill (512 / 4,096 tokens, eager) is unchanged by either kept change within its noise (largest mean −0.39% ±
0.91 on the 5090 8B; the host now computes a prompt's positions and write indices with numpy instead of the device).

The A/B's savings are well under the profiled µs of the inventory: prep removes 22 ops for ≈ 22 µs, about 1 µs per
op on both cards (node tracing inflates small kernels). The dropped telemetry change cut the offload step's `observe`
from 12 ops to 3 (45.4 → 26.9 µs profiled, A5000) but saved 1–2 µs of wall on the 5090.

**End to end (`holon_tg`, tok/s, median of 4 rounds; ms per token saved ± sd):**

| | 1K | 4K |
|---|---|---|
| RTX 5090 30B resident | 419.2 → **424.6** (×1.013; −0.030 ± 0.001 ms) | 387.2 → **392.0** (×1.013; −0.030 ± 0.003 ms) |
| A5000 30B resident | 190.0 → 190.7 (−0.014 ± 0.013 ms) | 176.4 → 177.4 (−0.027 ± 0.008 ms) |
| A5000 30B offload 47% | 133.3 → 136.4 (−0.13 ± 0.16 ms) | 145.4 → 147.5 (−0.02 ± 0.21 ms) |
| A5000 8B | 115.9 → 116.5 (−0.029 ± 0.037 ms) | 108.4 → 108.6 (−0.006 ± 0.037 ms) |

On the 5090 `holon_tg` sees 0.030 ms of `ab.py`'s 0.036. The A5000's end-to-end rows do not resolve a 0.025 ms change
(its offload rows' sd is 0.16–0.21 ms); the 5090 desktop was used for that.

## Judge

- **Bitwise:** snapshots (`spikes/x2_store/exact.py`, the judge's snapshot trace over `v1.jsonl`, A5000) at `5452833`
  are identical to bdd8422's for Qwen3-30B-A3B resident and at `*=0.47`, and for Qwen3-8B; the offload store is
  identical to all-resident. (With the dropped telemetry commit on top: identical too.)
- **Tests:** `test_host_staged_inputs_equal_the_device_derived` (plugins/tests/test_engine.py, tiny CPU engine, roomy
  and preempting): every step's input ids, positions, lengths, write indices and the SeqTable after it agree between
  the host-staged and the device-derived paths (a write index off by one fails it). Every other path is covered by the
  existing engine tests (graphs, spec decoding's device path, logits processors, preemption, prefix cache).
- **Suite:** `pytest core std plugins judge testing tools` on one A5000 at `5452833`: 1,411 passed, 66 skipped;
  `scripts/ci/run_all.sh main` green (core 4,602 code lines; `check_abi --base main`: the branch's 1.3 only).
- **Review:** one reviewer on the core diff (correctness against the data flow, CPU checks incl. mixed steps, n > 1,
  prompt logprobs and steps alternating between the two arenas): no blocker or major; its nit on the small arena's
  repeated token-independent fields is recorded in the design (§14, step 3d).

## Not done, recorded

- **The telemetry copy** (dropped above).
- **std's per-step kernels** (not the core's; each under the rule alone): the greedy sampler's unused parameter-row
  gather (≈ 1 µs), the gqa planner's write unit / offset (2 kernels; the core could stage them only through a new
  `Pages` field, an ABI change), FlashInfer's plan copy and in-graph `prepare`.
- **The tokens' D2H copy** stays outside the graph: a captured copy would need one graph per output buffer.
- **The SeqTable start write** stays: plugins may read `num_computed` during a step (`SeqTable` contract).

## Reproduce

```bash
nsys profile -t cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop -o <rep> \
    python spikes/x3d_core/profile.py --model <gguf> --arm base|upload [--engine-args '<flags>']
nsys export --type sqlite -o <rep>.sqlite <rep>.nsys-rep && python spikes/x3d_core/inventory.py <rep>.sqlite --json <inv.json>
python spikes/x3d_core/ab.py --model <gguf> --rounds 8 --prefill 512 4096 --json <ab.json>   # + --engine-args for offload
python bench/holon_tg.py --model <gguf> -n 128 -r 3 --depths 1024 4096 --engine-args "--max-seqs 1"  # per commit, ABBA
python spikes/x3d_core/record.py --gpu <class> --date 2026-10-17 --commit 5452833 --out <record.json> \
    --ab resident=<ab.json> ... --tg resident=<tg dir> ... --inventory resident=<before.json>,<after.json> ...
```
