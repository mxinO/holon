# Milestone X, step 7: the held-out plugins (acceptance 1)

## Question

`docs/v1-offload.md` acceptance 1 and §9 step 7: after the std offload API's tag (`v1-offload-std-api`), can a new
strategy land as one plugin, with no core or std diff? Three held out: **F10** promote-on-fetch (a policy with its own
resolve contract, oracle, CPU and CUDA providers), **A1** route tail skip (`approx`), and **a tier the design does not
design** (the page-cache tier). Each must also pass its own checks: F10 exact (snapshot byte-identical to all-resident,
run-identical, residency parity with the M6 replay), A1 judged as `approx` (L2 reported, human-gated), the page-cache
tier exact in a RAM-capped run.

## Setup

| | |
|---|---|
| GPU | RTX A5000 (sm_86, 24 GB, PCIe 4.0) on a shared host |
| model | Qwen3-30B-A3B Q4_K_M (unsloth GGUF): 48 layers × 128 experts top-8 |
| checkpoint | on a network filesystem (no local NVMe): file-tier rates here are not a desktop's |

```bash
M=<Qwen3-30B-A3B-Q4_K_M.gguf>
P="moe.store=std.moe.offload@1;moe.cache=*=0.25;moe.placement=placement.promote@1"
python spikes/x7_heldout/run.py $M <dir> "resident|1|snapshot|" "promote25|3|snapshot|$P" \
    "resident-parity|1|parity|" "promote25-parity|1|parity|$P"
T="moe.store=std.moe.offload@1;moe.cache=*=0.25;moe.ram=*=0.5;host_memory=12884901888"
python spikes/x7_heldout/run.py $M <dir> "pagecache-f11|1|snapshot|$T;moe.tiers=std.tiered.pinned@1,tiers.pagecache@1" \
    "pagecache-f1|1|snapshot|$T;moe.tiers=std.tiered.pinned@1,tiers.pagecache@1;moe.placement=std.tiered.static@1"
python spikes/x7_heldout/l2.py $M cand.safetensors \
    "moe.store=std.moe.offload@1;moe.cache=*=0.25;moe.placement=std.tiered.static@1;moe.route_transform=tail_skip"
python -m holon_judge compare --golden judge/goldens/qwen3-30b-a3b-q4_k_m/v1-fp32.safetensors \
    --candidate cand.safetensors --floor judge/goldens/qwen3-30b-a3b-q4_k_m/v1-bfloat16-mmvq-floor.safetensors
B="moe.store=std.moe.offload@1;moe.cache=*=0.25"
python spikes/x6_cpu/decode.py $M a1.json "f11|$B" "f11+a1|$B;moe.route_transform=tail_skip" "f11|$B" \
    "f11+a1|$B;moe.route_transform=tail_skip" --batch 1 --tokens 256 --depth 512 --rounds 3 \
    --engine-args "--host-memory 21474836480 --max-model-len 4096"
python spikes/x7_heldout/record.py <snapshot dir> <a1 dir> <page-cache dir> evidence/x7_heldout_rtx-a5000_<date>.json <commit>
```

Snapshot runs use the step-3 engine settings (`max_seqs` 64, graphs and overlap on); the parity trace (the first 8
prompts, 256 greedy tokens each) runs eager and synchronous so the routes can be read at the resolves. Wrap heavy
commands in a memory-capped scope on a shared host. Record: `evidence/x7_heldout_rtx-a5000_2026-10-17.json`.

## The proof: plugin-only diffs

Each strategy is its own commit; every diff against the tag touches only its plugin directories, its test file and
registration lines (an entry point in `plugins/pyproject.toml`, the generated contract listing `plugins/contracts.txt`);
`scripts/ci/check_core_diff.sh v1-offload-std-api` reports the core diff empty and `git diff --name-only` lists no
`core/` or `std/` path. Each passed ruff, its tests, `scripts/ci/run_all.sh v1-offload-std-api` and `check_private`.

| commit | plugin | files outside the plugin dirs |
|---|---|---|
| `8ce3f2e` (+ review fixes `9226253`) | F10: `placement/promote/`, `kernels/promote/` | `plugins/tests/test_placement_promote.py`; 2 entry points, 2 listing lines |
| `53f4a74` (+ `c217918`) | A1: `transforms/tail_skip/` | `plugins/tests/test_transform_tail_skip.py`; 1 entry point, 1 listing line |
| `642a020` (+ `11ff061`) | page-cache tier: `tiers/pagecache/` | `plugins/tests/test_tier_pagecache.py`; 1 entry point, 1 listing line |

(`transforms/__init__.py` and `tiers/__init__.py` are the new kinds' empty parent packages.)

## Results

**F10** (`placement.promote@1`, CUDA provider `kernels.promote`; runs at `9226253`), 25% cached:

| check | result |
|---|---|
| snapshot trace vs `std.moe@1` all-resident | **byte-identical**, 3 of 3 fresh engines (so run-identical ×3); 8,474 logged promotions per run, one ring step |
| decode trace (8 × 256 tokens) vs all-resident | **byte-identical** |
| L1 parity with `placement.promote`'s `replay` (the M6 replay's `promote_on_fetch` in the device's terms) | **equal at every one of 258 steps** (255 decode, 3 borrowing; 32,798 logged promotions); the host mirror is the device's map but for the spares' owners, mapped at their spares |
| L0 (CPU provider and CUDA vs the oracle, exact) | 56 resolve cases and 3 `lend` cases, incl. set-up from a seed, owners taking spares back, empty slots, ring pieces, the CPU's share, overflow, a fetch list past its width; `lend` without a host sync |

The first run (at `8ce3f2e`, before the review fixed the owner's needless refetch) gave the same verdicts.

**A1** (`transforms.tail_skip`, rank 7; L2 at `53f4a74`). Candidate: 25% cached at a frozen placement (static; a
teacher-forced token is a prompt row, so no device policy would admit on it either), one token per step (mmvq), against
the mmvq floor — 32,783 positions:

| | top-1 | KL p50 | KL p99 | ΔPPL vs floor |
|---|---|---|---|---|
| floor vs golden | 98.536% | 1.18e-4 | 3.98e-2 | |
| A1 candidate | 97.154% | 5.10e-4 | 1.43e-1 | +1.005% ± 0.218% |
| exact class's budget (for reference only) | ≥ 98.286% | ≤ 2.35e-4 | ≤ 7.95e-2 | ≤ 0.629% |

Outside the exact budgets, as §6 expected: A1 is `approx`, for the human gate (L2 plus a task check), never in a
default lock. By set: chat ΔPPL +0.32%, code +0.69%, recall +1.48% (top-1 96.2%), agent +1.29%. A frozen 25% placement
misses more than F11's adaptive one, so these drops are an upper bound for decode under F11.

**A1's decode** (batch 1, 512-token natural prompts, 256 tokens, F11 at 25% cached, A B A B, 3 rounds each):
46.45 tok/s → **51.67 tok/s (+11.2%)** (rounds 46.3–46.6 against 51.0–52.8), inside Strata's +10–24%.

**Page-cache tier** (`tiers.pagecache@1` as the file tier, `host_memory` 12 GiB — below the experts' 17.6 GB —, RAM
holding half the pieces, 25% cached; at `642a020`): the snapshot trace is **byte-identical** to all-resident under
F11 (2,108 admissions, RAM exclusive) and under F1. Decode against `std.tiered.gguf_direct@1` (same settings, A B A B):
16.5–16.6 tok/s for the page-cache tier; `gguf_direct` 5.0 in its first (cold) arm and 13.7–17.0 in its second. The
file lives on a network filesystem that serves both kinds through its own client cache, so no rate is claimed for
either tier here; a desktop NVMe decides.

## Decided while building

- **F10's spares are the device's own, end to end.** D4 has the host refill the spares in the apply that unmaps the
  victim; the std API has no host edit that hands a slot to a device-owned region (a device-owned policy may not edit
  at all, and `ResidencyMap.commit`'s records have no "slot kept by the device"), so the refill runs in the graph at the
  layer call's end — the M6 replay's boundary, with no lag. The policy's state holds the spares (owner, slots); a spare
  is unmapped without a record, and the record comes when another expert takes it (`(group, a, owner)`), so the host
  mirror keeps mapping a spare's owner at its slots until then. An owner promoted back takes its own spare unlogged
  and unfetched (nothing writes a spare's slots).
- **The spares start absent** and the first decode call's boundary sets them up by the same LRU rule (ties by expert
  id); M6 set them up at the seed from the profile's coldest. A set-up at the seed needs device state the policy can
  initialize (the store zeroes it); a first version that set them up from the highest seeded slots at the first decode
  call left layers spare-less when a borrowing step had lent their residents.
- **F10's CUDA provider composes**: two one-CTA kernels around the store's own `moe_resolve@1` launch (whichever
  provider serves it), so the variant reuses @1's staging, CPU-share and slot rules instead of copying them; the CPU
  share is taken from the misses left after promotion.
- **A1's best rank is over the pass's rows**, graph padding included (the transform sees no row kinds), and other
  requests' rows: a request's drops depend on its batch (Strata turns the skip off with parallel slots). Its CUDA
  provider is one Triton program (atomic minima, then the drops).
- **The page-cache tier's storage decisions**: every piece, no pinned or device memory of its own, files opened
  buffered with random-access advice; residency is the kernel's page cache.

## Not done

- **ARC placement** (the bonus): not built.
- **F10 under exclusive tiers**: refused at startup (the store asks a device policy for a heat to rank RAM by; F10's
  unlogged evictions would also leave the RAM tier's placement unaware of its victims).
- **The judge's route dump** (`holon_judge.placement`) reads the host mirror, so under F10 it counts spare owners as
  resident and `mirrored` is false; F10's parity uses the spike's own route reader (`run.py`, and the test's).
