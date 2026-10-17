# Milestone X, step 3a: placement policies on the development vehicle

## Question

`docs/v1-offload.md` §9 step 3 (without the ring, step 3b): with adaptive placement on — `placement.lfu@1` (F11, the
default), `placement.lru@1` (F4), `placement.profile_decay@1` (F2–F3, two-phase swaps) — is the offload store still
bitwise `std.moe@1` all-resident and run-identical, with verify rows too? Does F11's residency, step by step, equal
the oracle extended from M6's replay (the L1 policy-parity check)? Do §6's plants fire as predicted, and only where
they should?

## Setup

| | |
|---|---|
| GPU | RTX A5000 (sm_86, 24 GB) |
| model | Qwen3-30B-A3B Q4_K_M (unsloth GGUF): 48 layers × 128 experts top-8; classes gate/up Q4_K, down Q4_K, down Q6_K |
| engine | bf16, graphs on, overlap on, `max_seqs` 64, `max_step_tokens` 4096, `max_model_len` 8192, `paged_min_tokens` 16384, `host_memory` 20 GiB |
| traces | the judge's snapshot trace (64 requests; seeded sampling, logprobs, prompt logprobs, a prefix-cache hit); a decode-heavy parity trace (the first 8 prompts of `v1.jsonl`, 256 greedy tokens each) |
| commit | `52e1bed` (evidence: `evidence/x3_policies_rtx-a5000_2026-10-17.json`) |

```bash
python spikes/x3_policies/run.py <gguf> <dir> <runs> "<name>|<cache>|<policy>|<plant>|<spec>[|decode]" ...
python spikes/x3_policies/record.py <dir> <json> <commit>          # snapshots vs all-resident, run-identical
python spikes/x3_policies/mutate.py <dir>/<name>_routes.npz        # how sharp the parity check is on a trace
python spikes/x3_policies/bench_resolve.py                         # resolve variants' cost per layer call
```

## Results

**Exactness, adaptive placement on** (`holon_judge snapshot.diff` against all-resident on the same trace):

| Configuration | Cached (gate/up of 6,144) | Adaptivity | vs all-resident | Run-identical |
|---|---|---|---|---|
| F11 `*=0.25` (×3) | 1,536 | 2,115 in-graph admissions per run | identical | ×3 |
| F2–F3 `*=0.25` (×3) | 1,536 | 16 swap rounds per run | identical | ×3 |
| F4 `*=0.25` | 1,536 | 9,280 admissions | identical | — |
| F11 `*=0.47` (the 5080's natural fit) | 2,888 | 3,123 admissions | identical | — |
| F11 fill (free VRAM above the KV floor) | 6,081 | 139 admissions | identical | — |
| F1 static `*=0.47`, `*=0.25` | 2,888, 1,536 | — | identical | — |
| F11 `*=0.25`, n-gram drafter (3 drafts; verify rows) | 1,536 | 1,394 admissions | identical (to all-resident with the drafter) | — |
| F11 `*=0.25`, decode trace | 1,536 | 23,163 admissions | identical | — |

Every placement equals all-resident, so the natural (`*=0.47`), capped (`*=0.25`) and adaptive snapshots are
byte-identical to each other (cross-placement identity).

**L1 policy parity** (`judge/holon_judge/placement.py`): the device's residency read from the delta log equals
`placement.lfu`'s `replay` at every step — 67 steps (64 with decode rows, 2,115 admissions) of the snapshot trace at
25% in each of the 3 runs, 64 at 47%, 64 at the fill, 63 with the drafter, and 258 steps (255 decode) of the decode
trace at 25% (23,163 admissions) and at 10% (11,301); and once every step has committed the device's map is the host
mirror in every run. How sharp it is on these traces (`mutate.py`): a decay of 0.9 instead of 0.915 differs from step
6–9, candidate ties broken the other way from step 2–4, victim ties from step 3–4 (decode traces), increments before
the decay from step 65 (decode trace at 10%); one add of the step's count, f64 decay, ≥ instead of >, and no stop at
the first failing pair never change a pick on these traces (the last cannot: with candidates sorted descending and
victims ascending, a failing pair fails every later one). The
rounding rules are L0's: the kernel's scores are compared bit for bit with the oracle's, and an L0 case checks the
decay-then-count order by hand.

**Cost of a variant's resolve** (corrected after the panel's review, which found the first version's claim
double-counted: §13's 0.25–0.35 ms band was the whole offload path's, not F11's alone). `bench_resolve.py`, per layer
call, each call from the same map and state so that it pairs (restoring copies 5.0 µs, in every arm, subtracted here;
2 identical rounds): `moe_resolve@1` 3.8 / 4.4 / 5.6 µs at 1 / 8 / 64 decode rows, `moe_resolve.lfu@1` 6.3 / 6.7 /
6.9, `moe_resolve.lru@1` 6.1 / 6.4 / 6.7 — F11 adds 2.5 / 2.3 / 1.3 µs per layer with candidates to pair (it was 4.7–
8.8 µs before the review: no early exit, an all-pairs ranking over every expert), ≈ 0.7 µs when a call has none (it
exits before ranking). End to end, decode tg128 at 1K context, batch 1, every expert cached (`bench/holon_tg.py`, 2–3
interleaved rounds): `std.moe@1` 5.343 ms per token, the offload store under `static` 5.663 (+0.32: step 2's path),
under F11 5.703 (+0.04 over `static`), under F4 5.675 — the all-cached figure, where no call has candidates. With
candidates (corrected after the step-3 panel): at batch 1 and 1K context 29.5% of decode layer calls have them at
47% cached and 62.3% at 25% (route dumps, 128 tokens), so F11 adds ≈ 0.06 ms per token at the natural fit and
≈ 0.09 ms at 25% (2.5 µs per call with candidates, 0.7 without, × 48 layers). Snapshot run times are not compared:
route dumps synchronize eager steps and two runs shared the host.

**Plants** (`judge/holon_judge/planted.py`; predicted in design §6 before running):

| Plant | Target run | Predicted | Observed | Fired | Other configuration |
|---|---|---|---|---|---|
| `resolve-off-by-one` | F11 `*=0.25` | snapshot differs | differs from step 0 | 2,496 eager + 1,440 captured resolves, 67 steps with misses | F11 `*=1.0`: identical (no misses) |
| `map-before-landing` | F2–F3 `*=0.25` ×3 | fails run-identical or the snapshot | both: all 3 differ from all-resident (run 0 at step 16, runs 1–2 at step 10) and run 0 ≠ run 1 | 2,836 / 2,866 / 2,794 copies | F11 `*=0.25`: identical (no host copies) |
| `map-stale-slot` | F2–F3 `*=0.25` | snapshot differs | differs at step 10 | 2,752 rotated maps | F11 `*=0.25`: identical |
| `owner-race` | F11 `*=0.25` | the precondition flag stops the engine | stopped at step 6 (`edit_errors`) | 3 edits | F1 `*=0.25`: identical (nothing to fire on) |
| `owner-race-unchecked` | F11 `*=0.25` ×3 | run-identical fails | the snapshot differs at step 7 in all 3 runs, which are identical to each other (the overwrite lands before the victim's next read); the parity check also differs, at step 5 | 64 edits per run | — |
| `lfu-victim-needed` | F11 `*=0.25` | snapshot differs (revised: on an mmq step the id-error word stops the engine; on decode-only steps the snapshot differs and parity fails) | the engine stops at step 3: the arena kernel's id-error word (4: a slot repeated in a row); on the decode trace the snapshot differs at step 11, parity at step 6 | 648 of 1,153 admissions took a needed victim (decode trace: 959 of 23,913) | F2–F3 `*=0.25`: identical |
| `lfu-shared-victim` | F11 `*=0.25` | as above | the engine stops at step 3, as above; on the decode trace the snapshot and parity differ at step 4 | 48 shared victims of 521 admissions (decode trace: 3,623 of 14,963) | F2–F3 `*=0.25`: identical |

## What it decides

- Step 3's exactness check passes for every policy shipped: adaptive placement is on in the snapshots, F11 with
  thousands of in-graph admissions, F2–F3 with 16 rounds of two-phase swaps through the copy stream.
- F11's device behaviour is its specification, step for step (L1 parity), on decode traces; prompts in mixed steps
  and device lending are step 3b's (the parity check takes each step's routed experts from its eager resolves).
- Every plant fires and is caught; two of the predictions were too weak (the id-error word stops the kernel plants
  before a wrong byte is emitted; the unchecked owner race is deterministic) — recorded in design §14.

# Step 3b: D8's prefill ring, device lending and its return, row kinds, the route transforms' call site

Same vehicle and engine as above; design §14 "Step 3b as built" says what was built. Commits: core row kinds
`7e4a49b` (ABI 1.3); the panel's 3a findings — item 1 (resolve cost) `5210c52` + `5716d22`, item 2 (f32 hand case
through both providers) `e5110da`, items 3 and 7 (plant verdicts, D3) `5f5451e`, item 4 (ring-return and prompt-time
admission replay, `replay5/`) `8883e8d`, items 5 and 6 (delta-log capacity, public store surface) `5210c52`, item 8
`7e4a49b`; step 3b `9cf9132`, `2bf880c`, `2af99f7`, `29f8979`, `dd1e930`, review fixes `16fbf27`, `f072463`.

```bash
python spikes/x3_policies/run.py <gguf> <dir> <runs> "<name>|<cache>|<policy>|<plant>|<spec>|[decode]|[seqs=,chunk=,step=,ring=]" ...
python spikes/x3_policies/prefill.py <gguf> <json> <rounds> 1024 4096 -- "<name>|<cache>|<ring_tokens>" ...
```

**Exactness and parity** (runs at `16fbf27`; F11 and F2–F3 ×3, and the mixed, fill and no-ring runs again at
`f072463`, with the same verdicts and counts). Every run is snapshot-identical to all-resident on its trace (the
mixed trace's reference is all-resident with the same `max_seqs` 8, `chunk_with_decode` 256, `max_step_tokens` 512),
and every F11 run equals `replay` at every step with the device's map the host mirror at the end:

| Configuration | Trace | Admissions | Borrowing steps (ring / stage) | Parity steps (decode) | Run-identical |
|---|---|---|---|---|---|
| F11 `*=0.25` ×3 | snapshot | 2,368 | 1 / 0 (2 `lend` calls) | 67 (64) | ×3 |
| F11 `*=0.47` | snapshot | 3,650 | 1 / 0 | 67 (64) | — |
| F11 fill | snapshot | 0 (every expert cached) | 1 / 0 | 67 (64) | — |
| F11 `*=0.25`, `ring_tokens` 0 (staging only) | snapshot | 2,115 | 0 / 0 | 67 (64) | — |
| F11 `*=0.25`, n-gram drafter | snapshot | 1,630 | 1 / 0 | 66 (63) | — |
| F11 `*=0.25` / `*=0.10`, `max_seqs` 8 | decode | 23,300 / 11,419 | 3 / 0 | 258 (255) | — |
| F11 `*=0.25` / `*=0.10`, mixed steps | decode, chunked with decode | 19,823 / 9,867 | 3 / 35 | 293 (290) | — |
| F2–F3 `*=0.25` ×3 | snapshot | 16 swap rounds | never borrows | — | ×3 |
| F4 `*=0.25`; F1 static `*=0.47` | snapshot | 9,640; — | 1 / 0; never | — | — |

The fill run caches every expert at `f072463` and 6,044 of 6,144 at `16fbf27`: the latter was the 16th engine
built in one process (the free memory read at startup is lower there); the startup profile's peak is 0.48 GB with
512- or 4,096-token passes alike (a probe on a fresh process).

**Prefill throughput** (`prefill.py` at `2af99f7`, one request of N prompt tokens, one chunk, median of 9 runs over 3
interleaved rounds; tok/s):

| Arm | 1,024 | 4,096 |
|---|---|---|
| all-resident (`std.moe@1`) | 3,669 | 3,595 |
| `*=0.25`, as built before D8 (512-token passes, misses staged) | 542 | 554 |
| `*=0.25`, D8 (lent ring, 4,096-token passes, 2,048-row mmq calls) | 896 | 3,340 |
| `*=0.47`, as built before D8 | 672 | 684 |
| `*=0.47`, D8 | 1,110 | 4,049 |

D8 is 6.0× the staged path at 4K and 25% cached, 1.65× at 1K (one layer's misses stream behind one layer's compute:
at 1K the pass is short). At 47% D8's 4K prefill is 13% faster than all-resident, whose expert passes stay at 512
tokens (`std.moe@1`'s binding; its larger calls are not this step's). 32K was not run.

**Plants** (at `29f8979` and `2af99f7`; predicted in design §6 / the panel's item 3):

| Plant | Target run | Predicted | Observed | Fired | Other configuration |
|---|---|---|---|---|---|
| `lent-slot-still-mapped` (a lent slot returned while its expert is still mapped) | F11 `*=0.25`, decode, `max_seqs` 8 | the snapshot and parity differ | parity differs at step 1 and the device's map is not the mirror; the snapshot differs at step 6 | 1,216 slots | F2–F3 `*=0.25`: identical (never lends; nothing fired) |
| same | F11 `*=0.25`, snapshot | as above (on an mmq step the id-error word stops it) | the engine stops at step 11 (`kernel_errors` 4: a slot repeated in a row) | 1,216 slots | — |
| `owner-race` under borrowing | F11 `*=0.25`, decode, `max_seqs` 8 | the precondition flag stops the engine | stopped at step 6 (`edit_errors`) | 2 edits | — |

**The ring's return and prompt-time admission** (`replay5/report.md`, the M6 replay; panel item 4): the ring returns
empty (R0) — keeping the streamed experts by decode score (R1) or the pass's routing (R2) gains ≤ 0.3% on whole
traces and costs paid copies or a different ring layout. Counting prefill rows at a routing-proportional weight (Bw12)
gains 2.1% / 0.7% (agent / mixed, 47%), never loses in a held-out cell; it changes D4 and goes to the panel unbuilt.

# Step 3's panel fixes: device lending, 4,096-row mmq calls, D8's crossover measured

The step-3 panel's findings (A: padding rows' kind; B: the store's row-kind guess; C: 1K prefill, 4,096-row calls,
lending's host syncs, F11's cost), each in its own commit — `6cebfbb` (RowKind.PADDING), `ce10352` (no guess),
`d070b40` (mmq's id helper past 16,384 pairs), `7846044` + `4773e5f` + `b191782` (lending on the device), `4283bbf`
(F11's cost) — and design §14 "Step 3's panel fixes". The A5000 here copies host to device at 26 GB/s
(measured); step 3b's prefill table above came from one that copies at 12.4 GB/s.

```bash
python spikes/x3_policies/prefill.py <gguf> <json> 2 512 1024 2048 4096 -- "resident|resident|0" "stage25|*=0.25|0" \
    "ring25|*=0.25|512" "stage47|*=0.47|0" "ring47|*=0.47|512"
python spikes/x3_policies/tpot.py <gguf> <json> 2 "*=0.47" on off
python spikes/x3_policies/candidates.py <gguf> "*=0.47" "*=0.25"
```

**Exactness and parity with lending on the device** (`7846044`; the later commits change a booking edge case no
trace here reaches and the copies' launch shape): every run snapshot-identical to all-resident, F11 equal to `replay`
at every step with the device's map the host mirror:

| Configuration | Trace | Admissions | Borrowing steps (ring / stage) | Run-identical |
|---|---|---|---|---|
| F11 `*=0.25` ×3 | snapshot | 2,252 | 1 / 0 | ×3 |
| F11 `*=0.47`; fill | snapshot | 3,255; 280 | 1 / 0 | — |
| F11 `*=0.25`, n-gram drafter | snapshot | 1,535 | 1 / 0 | — |
| F11 `*=0.25` / `*=0.10`, `max_seqs` 8 | decode | 23,249 / 11,409 | 3 / 0 | — |
| F11 `*=0.25` / `*=0.10`, mixed steps | decode | 19,773 / 9,866 | 3 / 35 | — |
| F4 `*=0.25`; F2–F3 `*=0.25`; F1 `*=0.47` | snapshot | 9,503; 16 rounds; — | 1 / 0; never; never | — |

Plants: `lent-slot-still-mapped` (384 lent experts kept mapped) — on the decode trace the parity check and the
snapshot both differ at step 4 and the device's map is not the mirror; on the snapshot trace the id-error word stops
the engine at step 1. `owner-race` under borrowing: stopped by its precondition flag at step 5.

**1K prefill, nsys** (`*=0.25`, one 1,024-token step): before, 600 ms, copies busy 573 ms (14.8 GB at 25.9 GB/s) —
PCIe-bound, its bytes 12% over the layers' misses: lending took the first 18 layers' every cached expert (576 experts;
ties by layer); after, lending spreads (420 experts, 8–9 per layer: ties by expert id, a need per class), 618 ms, the
SM copies busy 603 ms — still PCIe-bound (the misses alone take ≈ 0.51 s at this link's rate).

**Stage against ring, and 4,096-row calls** (tok/s, one request of N tokens in one step, median of 6 over 2
interleaved rounds; `b191782`; before: `ce10352`, the DMA ring with layer-ordered lending and 2,048-row calls; DMA
4,096: `d070b40`):

| Arm | 512 | 1K | 2K | 4K |
|---|---|---|---|---|
| all-resident (`std.moe@1`, 512-row calls) | 3,489 | 3,585 | 3,550 | 3,482 |
| `*=0.25` staged (lent staging, 512-token passes) | 853 | 973 | 957 | 978 |
| `*=0.25` ring (D8) | 852 | 1,686 | 3,307 | 4,680 |
| `*=0.25` ring, before / DMA 4,096 | 810 / — | 1,604 / 1,542 | 2,645 / 2,819 | 4,888 / 5,127 |
| `*=0.47` staged | 1,079 | 1,228 | 1,204 | 1,226 |
| `*=0.47` ring (D8) | 1,186 | 2,337 | 4,446 | 4,777 |
| `*=0.47` ring, before / DMA 4,096 | 1,005 / — | 1,744 / 1,747 | 3,380 / 3,239 | 4,819 / 5,181 |

D8's crossover on the A5000 is ≈ 512 tokens (the ring equals staging at 25%, is 10% ahead at 47%), as
`moe.ring_tokens`' default. 4,096-row calls gain 5–7% at 4K on the DMA ring; the ring's SM copies (no host
addresses without a sync) cost ≈ 8–9% at 4K, where the pass is compute-bound (one copy CTA per class: 4,716 tok/s at
25%; eight: 3,968), and nothing where it is PCIe-bound.

**Decode latency on the agent trace** (`tpot.py`, 47% cached, `max_seqs` 8, `chunk_with_decode` 512; ms):

| | TPOT p99, borrowing on | off | max gap p99, on | off |
|---|---|---|---|---|
| before (`ce10352`: host lending, 2–3 syncs per borrowing step) | 131.3 / 128.8 | 133.7 / 131.9 | 669 / 735 | 609 / 550 |
| after (`b191782`: device lending) | 126.0 / 121.5 | 132.0 / 124.0 | 536 / 523 | 629 / 631 |

**F11's cost with candidates** (`candidates.py`): 29.5% of batch-1 decode layer calls have candidates at 47%
cached, 62.3% at 25% → ≈ 0.06 / 0.09 ms per token (the +0.04 above is all-cached).
