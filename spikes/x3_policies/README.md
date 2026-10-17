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

**Cost of a variant's resolve** (`bench_resolve.py`, per layer call, 3 identical rounds): `moe_resolve@1` 4.1 / 4.2 /
5.5 µs at 1 / 8 / 64 decode rows, `moe_resolve.lfu@1` 8.8 / 9.1 / 10.3, `moe_resolve.lru@1` 8.0 / 8.2 / 9.0 — F11
adds ≈ 4.7 µs per layer, ≈ 0.23 ms per decode step over 48 layers (§13's band for F11's pick and decay: 0.25–0.35
ms). Snapshot run times are not compared: route dumps synchronize eager steps and two runs shared the host.

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
