# v1 — GLM on any host from its floor up: the RAM tier sized from the host budget

Status: **design note, rev 2.1 (2026-10-31): rev 2 cleared by the panel (A, B, C), with their rev-2 items taken (§8).**
C's replay of the step-7 recording (the panel's notes) places the grid: speed is monotonic in RAM, steep from 56 to
≈ 74 GB, flat from ≈ 84 GB; one f is within 0.09 points of heat-greedy. Steps 0 (the D1 form) and 1 are built (§9); the
GPU grid, R and the knee are the lead's runs.

**The owner's request (2026-10-31):** "we should not assume the 64g is the only one we need support, we should be able
to adapte to different mem size (with 64g as a lower limit maybe)".

**Goal.** On any host from GLM's floor up, `holon serve <model>` serves GLM-5.3-Flash at its best configuration for the
RAM it has, with no flags:
- the RAM tier is sized from auto-config's host budget, not mode (ii)'s fixed `moe.ram=*=0.6238`;
- speed rises with RAM, from the floor to the lock (mode (i)) once every expert fits;
- below the floor, the start is refused at build, naming each term of the measured minimum;
- below the performance knee, it serves with a warning.

The mechanism is in std, so every offload lock gets it. The model plugin supplies only its measured tuning, keyed by
card. Core gains two additive members.

## 1. What is fixed today, and why it is not enough

- **`glm5_next`'s spill returns `MODE2`.** Its RAM share, `moe.ram=*=0.6238`, fits a 64 GB host whatever the host has.
  - A 48 GB host is refused.
  - A 96 GB host pins about 50 GiB where about 80 fit, so it runs at 64 GB speed.
- **The spill hook sees neither the budget nor the card.** `MODE2` is gated on `moe.lend_peak = require` as a proxy for
  the RTX 5090 lock (B's s1 at the close).
- **Std's generic spill only names a manual `moe.ram='*=0.5'`.**
- **Spills merge last-setter-wins** (`cli.py` `fit_lock`), so a later owner overrides the model's measured values (B-R2).

## 2. The design

### D1. `fit`: one owner of the host remainder, sized once at build

**The value.** `moe.ram=fit`, the whole value only (no per-class `fit`), makes the store solve one expert fraction f
(D2) from the host budget. The rules:
- **When.** The store resolves `fit` once, at its first `before_materialize`, after every fixed host pool is requested
  (its own, `std.tiered.rows`', `std.moe.cpu`'s and the like; A's R2-m2), and before the startup profile. It caches
  f, and passes `*=f` to the tier. Every later `holds()`, the profile's minimum layout included, uses the cached f. A
  host pool is never re-pinned or re-seeded (B-R1, A-RAM-M1.2). Pools requested after it are named in the reserve.
- **Against what:** `BuildCtx.host_left()`, a new optional API member:
  `min(cfg.host_memory − ledger.host_bytes(), live available − headroom)`. Core reads the live RAM, since std cannot
  import `holon.core` (B-R6).
  - It is **not gated on `clamp_context`**, so `exact.py`, the bench and the tests have a budget whenever
    `host_memory` is set (B-R1). A placement-only choice is not a lock degradation.
  - **Without a host budget, `fit` refuses** ("moe.ram=fit needs a host budget: `holon serve` derives one;
    elsewhere, --host-memory").
- **Taken off before solving:**
  - the store's own host pools: the read-ahead landing rows and their on-demand halves, the I/O and run tables;
  - every host pool already requested: the embeddings and the PLE rows of `std.tiered.rows`;
  - **a generic reserve:** the I/O worker's bounce windows (≤ 512 MiB), the pinned buffers outside the ledger
    (`_readback`, `_move_host`, D8g's lists; A-RAM-M1.4) and the process's non-pinned growth R (D4), as std's
    constant, no model's (m4).
- **Re-checked before pinning.** Just before the tier's pool is requested, the store asks `host_left()` again. If the
  live reading is lower than the one f was solved from, the start refuses with both readings rather than re-solving
  (B-R6; A-RAM-M1.3).
- **One owner (A-RAM-M1, m6, R2-m3).** One rule: `fit` refuses when a state tier is configured. Nothing then races
  the store for the host remainder.
- **Limits.** A pool requested after the store's comes out of the reserve. A CPU share (`moe.cpu_plan`: its job
  table and rows, up to GBs) would, so `fit` refuses it (the code review's A m3). No lock combines them.
- **Quantized.** f is rounded down to 1/256, so placement does not drift with a few MB of free RAM from start to start
  (A-RAM-m1).

**Step 0 prices B-R4's zero-core form against this,** and picks it only if its restructure is small. That form is one
host pool across classes, granted between min and pref by the existing request, with f derived from the grant. It
needs:
- `exclusive` decided by the option;
- the statics (`owner`, `n_ram`, `_ram`) sized by pieces;
- `Pinned` as one pool with per-class offsets.

Otherwise it is `host_left()` at build. **Core: +5 to +6 lines in all** (`host_left()` with its live reading, the
spill's `card`), ABI 1.6 → 1.7 minor (B-R4, m7). Both forms keep the single owner.

### D2. One fraction of whole experts; capped at saturation

- **The rule.** An expert serves from RAM only if all its pieces are there: gate/up and down are in different
  classes, and GLM's down classes cover disjoint layer groups. So `fit` solves one fraction:

  `f = (host_left − pools − reserve) / Σ_classes(pieces × unit_bytes)`, rounded down to 1/256

  with `held = max(1, floor(f × pieces))` in each class. The store's RAM heat (`moe.ram_heat`) picks which experts.
- **No cap at saturation (B-R7, A's R2-m1).** Rev 2's `f_sat` is dropped: the VRAM cache's size and the tail's lend
  count are not known when f is fixed. Past saturation (f ≈ 0.82–0.84, a ≈ 84 GB host) more units buy nothing, but the
  lock is tried first, so the waste is only the **band from ≈ 84 GB to the lock's need (≈ 96 GB with little other
  use)**, where the spill pins RAM a smaller f would serve as fast.
- **One f against heat-per-byte (A-RAM-m2, C §3): kept.** C's replay: one f is within 0.09 points of heat-greedy whole
  experts.
- **The MTP head under `f < 1` (B-R9):** stays refused as today. The store refuses an auxiliary group's experts under
  exclusive tiers, so a Flash-Next spill with `f < 1` serves without MTP, with the existing warning. GLM's head is
  unused.
- **`f ≥ 1`:** every piece is held, the tier is not exclusive, and the store runs as the lock does. It is tested
  identical to the lock.

### D3. Where everything fits: the lock

- `holon serve`'s history tries the lock first, and the spill only after the lock's start is refused for host memory.
  So a host that holds every expert gets mode (i).
- C's model predicts the curve:
  - a rise from the floor to a knee at about 72–76 GB;
  - a plateau from about 84 GB, past saturation (f ≈ 0.82–0.84);
  - then a step up to the lock, about +20% decode, at the size where it starts (about 96 GB with little other use).
- The plateau's speed is reported against the lock, not gated to it.

### D4. The floor and the knee

- **The working floor** (A-RAM-M3, refused at build, A-RAM-m3) is the sum of:
  - auto-config's headroom (outside the budget);
  - **R**, the non-pinned process. It is measured in step 0 at a 32K step's peak, at the lock and at mode (ii), not at
    idle.
  - the ledger's actual fixed host requests at build: the landing rows, about 1.9–2.3 GiB at `read_ahead` 135, and
    the embeddings, the tables and the reserve;
  - a minimal tier: at least one unit per class, and the units a call's keeps and a step's parking need (tens of
    experts).
- **Below the floor the start is refused** with each term in bytes. It is not a constant: the refusal computes it.
- **The knee.** The 32K tail's lend (969 experts, 6.4 GiB) and C's knee at about 72–76 GB are performance terms. Bits
  do not change below them (`_park` falls back to RAM's coldest, then the SSD).
  - **Below the knee the start serves and warns** ("below N GB, decode under X tok/s"). The model plugin's spill note
    gives N and X, measured, and so does the README.
  - The owner's "64 GB as a lower limit maybe" goes to the owner with this choice: refuse at the floor and warn at the
    knee (Q1).
- **Memory-failure retries (A-RAM-M2, m5).** Under `fit`, a start killed for memory would re-derive the same f from a
  fresh snapshot. Restart admission already lowers an entry's host cap after a memory failure: `fit` solves from that
  lowered `host_memory`, so each retry gets a smaller f with no new channel, within admission's bounded count, then
  the start refuses with the measured shortfall.

### D5. Spills: the card, first-setter-wins, and the generic spill

- **The card.** `spill(options, checkpoint, *, card)`, keyword-only. Core passes `card=` only to a hook whose
  signature names it (one `inspect.signature` check), so two-argument hooks keep working. The new signature is written
  into `core/abi.txt` by hand with the 1.7 bump, since `check_abi.py` compares names only (B-R5).
- **First-setter-wins (B-R2).** `fit_lock` merges `spill = got[0] | spill`, with the model plugin asked first, so the
  model's measured set refines std's generic one. The line count is unchanged.
- **Std's generic spill becomes a real one** for every offload lock: only `moe.tiers=std.tiered.pinned@1,std.tiered.gguf_direct@1`
  and `moe.ram=fit`. No F10: `placement.promote@1` is a first-party plugin measured on GLM only (B-R3). It is not
  gated by card: an exactness failure under it is a std bug to fix (B-R8).
- **`glm5_next`'s spill becomes `MODE2[card]` with `moe.ram=fit`.**
  - **RTX 5090:** the measured set (8,192-row passes, the loan options, `read_ahead`, `stream_tokens` 65,536, F10).
  - **RTX 5080:** per D7.
  - **Other cards:** GLM's spill returns nothing and the generic spill applies (lfu, std's tiers). Its tuned set is
    never applied to a card it was not measured on; where it applies, it refines the generic one (first-setter-wins).
- **`check_locks` refuses `fit` in a lock** (a CI check). A lock's placement is fixed; `fit` lives only in spills.

### D6. Which 64 GB values move with RAM (C §2)

| Value | Prediction | Measured |
|---|---|---|
| `pass_rows` 8,192, the loan options, `stream_tokens` 65,536, `ram_heat` 0.999 | hold (VRAM-side, or matter less as SSD-only shrinks) | no |
| `read_ahead` 135 **with D8x's rows** | coupled. D8x parks a short prompt's ≈ 225 lent experts in 2 × `read_ahead` rows a role, 128 at 64 GB, so ≈ 100 park over RAM again | 1K prefill and D8x's `parked_in_rows`, at 64 GB and 80 GB: `read_ahead` 135 against D8x's rows decoupled from it |
| F10's budget (1 a layer) | likely at the plateau, where promotions only shuffle VRAM and RAM | one arm at 88 GB (on the plateau, C): budget 1 against 0 against `placement.lfu`, 32K decode |
| Parking (D8l, D8x) | self-adjusting: idle past saturation | counters only |

### D7. The RTX 5080

- Its lock serves 8K steps without a loan (`moe.lend_peak` off). Mode (ii) has never run there.
- **Step 3:** a smoke and `exact.py` at its lock's flags with the generic spill, under a 64 GiB cap. An exactness
  failure is fixed at its cause, not gated (B-R8).
- Then one capped bench at 64 GB against the lock, with `read_ahead` and F10 as in D6. If GLM's tuned set gains there,
  `MODE2["rtx-5080"]` ships with its own reported numbers (Maya on 16 GB is not a program bar); otherwise the 5080
  serves the generic spill.

## 3. Acceptance

1. **The capped RTX 5090** (a cgroup-enforced job step, the node's NVMe, D11's method):
   - 56, 64 and 72 GB at 3 rounds;
   - 48, 80 and 88 (or 96) GB at 1 round each, recording at the largest which of the lock and the spill started;
   - the uncapped lock.

   C's replay rows read as ±30% at 40–56 GB, and its ≥ 72 GB prefill rows as upper bounds.

   At each size: 1K and 32K (4K only at 64 GB and the lock), SSD MB a decode token over tokens 1–32, and D8x's
   `parked_in_rows`. Speed rises with RAM. At 64 GB it is at or above the close's mode (ii) within its round spread.
   The plateau is reported against the lock. The grid is placed by C's replay (step 0) before the GPU runs, and moved
   if the curve bends elsewhere.
2. **Zero OOM kills** at every size (`memory.events`). Below the floor the refusal comes at build, with its terms.
3. **`exact.py`:**
   - the RTX 5090 at 80 GB, the spill as `holon serve` gives it, byte-identical to the lock's first run;
   - the generic spill on Qwen3-30B-A3B (the RTX 5080 offload lock) and on Flash-Next (one card, MTP off: B-R9), each
     capped.
4. **Tests,** under the deployment's terms (the `mode2` fixture with `moe.ram=fit`):
   - `fit`'s arithmetic, engine-free: quantization, the floor's refusal message, the lower second reading's refusal,
     the refusal with a state tier, `f ≥ 1`;
   - the bitwise `ssd` cases at a `fit` share, and at the floor's f;
   - `f ≥ 1` identical to the lock;
   - first-setter-wins;
   - `check_locks` refusing `fit`;
   - the CPU default suite with no GPU visible (`CUDA_VISIBLE_DEVICES=`).
5. **The RTX 5080,** per D7.
6. **The real-model smoke** at 72 GB before reporting built.

## 4. Locks

- **Core** (`BuildCtx.host_left`, the spill's `card`) outdates every lock: 27, on four card classes.
- `std.tiered`, `std.moe`, `models.glm5_next` and `models.qwen4_exp` change in the same round.
- **Predicted:** every trace byte-identical, since no lock sets `moe.ram` and `fit` lives only in spills.
- **One round at the close** (release §22). A moved trace stops for review.

## 5. Steps

| Step | Work | Check |
|---|---|---|
| 0 | R at a 32K step's peak (lock, mode (ii)); D1's form priced (B-R4); one f against heat-greedy offline; C's replay of the step-7 recording at RAM sizes (C writes it); the retry count's source | R; the form; the gap; the grid's sizes |
| 1 | `fit` (D1, D2), the floor's refusal and the knee's warning (D4), the retries, the card and the merge (D5), `MODE2[card]`, the generic spill, `check_locks` | acceptance 4 |
| 2 | Capped RTX 5090 grid (acceptance 1), with D6's arms at 64 GB and 80 GB | acceptance 1, 2 |
| 3 | `exact.py` (acceptance 3); the RTX 5080 (D7) | acceptance 3, 5 |
| 4 | The lock round; README (GLM on any host from its floor; the knee; the table by RAM size) | acceptance 6; the panel's delta review |

## 6. Questions for the owner

- **Q1, the floor:** refuse at the computed working floor and warn below the measured knee (this design), or refuse at
  the knee (about 72–76 GB by C's model, near the "64 GB" guess)?

## 7. The panel's review of rev 1, and how rev 2 answers it

| Finding | Answer |
|---|---|
| B-R1 `fit` resolves once, against a defined `host_left()` | **Taken** (D1): once at build, cached, `*=f` to the tier; `host_left()` not gated on `clamp_context`; refuses without a budget; whole value only |
| B-R2 spills merge last-wins | **Taken** (D5): first-setter-wins, model first |
| B-R3 no F10 in the generic spill | **Taken** (D5): tiers and `fit` only; GLM's tuned set only on measured cards |
| B-R4 the fill pool is a restructure | **Taken:** `host_left()` at build (+3 core, ABI 1.7); the zero-core form priced in step 0 |
| B-R5 keyword-only `card`, `abi.txt` by hand | **Taken** (D5) |
| A-RAM-M1 one owner, sized once, re-checked before pinning, pinned buffers counted | **Taken** (D1): the store is the only reader (rev 2.1: `fit` refuses with a state tier); before the profile; live re-read; reserve |
| A-RAM-M2 a memory-failure restart loops | **Taken** (D4): the reserve doubles per retry, at most 3, then refuse |
| A-RAM-M3 the floor from real requests; the 6.4 GiB tail is a knee term | **Taken** (D4): the ledger's requests + R at a 32K peak + ≥ 1 unit a class + the working tier; the tail a knee term |
| A-RAM-m1 quantize f | **Taken** (D1): 1/256, down |
| A-RAM-m2 / C §3 one f against heat-greedy | **Taken** (D2): checked offline in step 0; keep one f under 1% |
| A-RAM-m3 refuse at build | **Taken** (D4) |
| A-RAM-m4 the spill only where tested | **Taken** (D5) |
| C §1 cap at saturation; the knee ≈ 72–76 GB; plateau; +20% step | **Taken** (D3, D4); the cap dropped in rev 2.1 (§8) |
| C §1 replay first | **Taken** (step 0; C writes it) |
| C §2 `read_ahead` coupled with D8x's rows; F10 at the plateau | **Taken** (D6) |
| C §4 the grid | **Taken** (acceptance 1) |
| Other: `check_locks` refuses `fit`; the MTP head under `fit`; generic-spill exactness on the 30B and Flash-Next; `f ≥ 1` identical | **Taken** (D5, D2, acceptance 3, 4) |

## 8. The panel's review of rev 2, and rev 2.1

| Finding | Answer |
|---|---|
| B-R6 std cannot import core for the live read | **Taken** (D1): `host_left()` returns min(remaining budget, live available − headroom); a lower second reading refuses |
| B-R7, A R2-m1 `f_sat`'s inputs are unknown when f is fixed | **Taken** (D2): dropped; the wasted band ≈ 84 GB to the lock's need noted |
| B-R8 the generic spill on every offload lock; failures are bugs | **Taken** (D5, D7) |
| B-R9 the MTP head under `f < 1` | **Taken** (D2): refused as today, served without MTP with the warning; Flash-Next's check MTP off |
| B m4 R in the generic reserve | **Taken** (D1) |
| B m5 retries through admission's lowered cap | **Taken** (D4) |
| B m6, A R2-m3 one state-tier rule | **Taken** (D1): `fit` refuses with a state tier configured |
| B m7 core lines once | **Taken** (D1): +5 to +6 |
| A R2-m2 resolve after every fixed pool | **Taken** (D1): at the first `before_materialize` |
| C the grid; F10 at 88 GB; replay error bars | **Taken** (acceptance 1, D6) |
| Q1 | With the owner; built as designed (refuse at the floor, warn at the knee); "refuse under 64" would be one constant |

## 9. Step 0 and 1 (build)

### 9.1 The D1 form: `host_left()` at build, with B-R10

**Chosen: `BuildCtx.host_left()`** (D1's default). The zero-core form was priced against B-R10 and is not small:
- B-R10 already brings half of it: under `fit` the store binds its exclusive kernels and I/O worker in the build, and
  makes its tables at `before_materialize`.
- The other half stays a restructure:
  - **No reserve without a reading.** The ledger grants a host request between min and pref against `host_memory`
    alone. A grant-derived f cannot leave the reserve (the store would need what is left to set pref), nor see the
    live RAM (B-R6). Either needs core.
  - **One pool across classes.** `Pinned` as one pool with per-class offsets changes every `SlotClass.copy` MemOp (its
    host pool's name and unit indices), the seed's reads and D8x's moves.

`host_left()` is one core member, so the zero-core form saves no lines that matter.

**How B-R10 is built** (`std/holon_std/tiered/store.py`, `fit.py`):
- **In the build, under `fit`:** `resolver()` binds `moe_tiers@2`, the I/O worker (`cpu_task@1`, an engine shared
  object, which cannot be made after the build) and D8g's copy engines. It binds each group's I/O slots and makes no
  tables. An auxiliary model's groups are added later; the "pieces added after the device tables" refusal no longer
  applies to them.
- **At the first `before_materialize`:**
  1. **The first reading decides exclusivity.** f ≥ 1 when every piece fits beside the reserve
     (`left − reserve ≥ Σ pieces × unit`). The inclusive store needs no I/O pools, so none are counted.
  2. **The tables come next** (the statics are allowed there, as the inclusive store's always were). RAM's units
     (`owner`, the stamps) come after f, sized by the share as under a manual `moe.ram` (the code review's A m2).
  3. **The store's own host pools are requested before the RAM tier's:** landing rows, request tables, runs. So the
     second reading has them off already: the ledger subtracts them, with no duplicate arithmetic of their sizes.
  4. **f is solved from the second reading.** The RAM tier is made again with its units.
  5. **The third reading** is taken just before the tier's pool (`fit.recheck`).
- **At f ≥ 1** the store has the lock's bytes and layout, its I/O worker started idle:
  - inclusive, with no landing, I/O or run pools;
  - `moe.ram_heat` unused;
  - D8g's gathers off: the copy engines bound in the build are never used. D8g is gated on exclusive tiers (the code
    review's A-M1: before, a staging step of 8K+ prompt tokens ran its passes in 2 sparse parts, a path the lock never
    takes);
  - each I/O slot bound in the build gets an idle copy task (zero roles), never signalled. The worker starts, pinned
    to a core, and dozes (50 µs naps after 100 ms idle). Its error word is zeroed at the first launch (a device
    synchronize) and read at each commit.
  - So `fit` needs the `cpu_task` prebuild and io_uring or Linux AIO even where it reads nothing.
  - "Identical to the lock" is tested as the same arenas (class, units, offset, size), seed and map, and the same
    outputs bit for bit. The other differences are the build-time kernels (two more lock sites) and the worker's
    error word. On a VRAM-bound fill arena those few bytes could cost one unit at a boundary; the tests' arenas hold
    every piece, so they do not show it.

**Two departures from the rev 2.1 text:**
- **The tier gets per-class counts, not `*=f`:** `<class>=⌊f × pieces⌋`, at least 1. `Pinned` rounds a fraction
  (`round`), which can pin half a unit a class more than f was solved for.
- **The re-check refuses when the reading no longer holds the tier's bytes and the reserve** (B-R6's "lower than the
  cached f allows"), not on any lower reading. Free RAM moves by MBs between two reads, and the 1/256 quantum leaves
  the slack.

### 9.2 `fit`'s numbers (`std/holon_std/tiered/fit.py`)

- **f** = ⌊(left − reserve) × 256 / Σ pieces × unit⌋ / 256, at most 255/256 under exclusive tiers. Each class
  holds `max(1, ⌊f × pieces⌋)` units.
- **The reserve** (std's constant, no model's) is 3.56 GiB in all:
  - the seed reads' bounce windows (`files`: 8 × 64 MiB);
  - 64 MiB of pinned buffers outside the ledger (D8g's read-back lists, D8x's move table, the ring's copy list:
    KBs each);
  - **R = 3 GiB** (`fit.GROWTH`). Step 0 measured it on an RTX 5090 at a 32K step's peak, as peak RSS less the
    ledger's host pools: 2.52 GiB at the lock, 2.62 GiB at mode (ii) on a 59 GB host. 3 GiB is that, plus margin.
- **The working floor** (D4) is the least multiple of 1/256 giving every class a call's keeps: `min(pieces, staging)`
  units, where staging is the graphed steps' misses (top-k × decode rows), the most one call keeps.
  - Below it the start is refused for memory (`OutOfUnits`, so restart admission retries with the lowered cap). The
    refusal names each term in GiB: the store's own pools by name, the reserve's three parts, the minimal tier with
    its f, what was left, and the shortfall.
  - On the tiny models the floor is a quarter (8 of 32 pieces a class).
  - On GLM's RTX 5090 lock it is f = 1/256, about 0.34 GB of experts: one decode row, top-8, so 8 of 2,304 pieces
    in the smallest class (the IQ3_XXS downs, 8 layers × 288). The floor is far below the knee: a host there
    serves, slowly, with the knee's warning.
- **Refusals without solving:** without a host budget (`ValueError`); with a state tier configured (the engine's
  `tier` slot, now in `BuildCtx.options` as `slots`); a value other than the whole `fit` (`*=fit`); `fit` without a
  file tier in `moe.tiers`.
- **The knee** is GLM's spill note: `KNEE["rtx-5090"] = (72, 28)`, "below ≈ 72 GB of RAM 32K decode falls under
  ≈ 28 tok/s". **This is a PLACEHOLDER** from C's replay (72 GB: 27.9 tok/s 32K decode) until step 2's grid.
- **Every start logs f:** `holon: moe: moe.ram=fit: 0.6211 of the experts in RAM (… GiB of … GiB left)`, or "every
  expert in RAM".
- **Retries (D4)** need no new code. Restart admission lowers an entry's `host_memory` after a memory failure, and
  `host_left()`'s budget term carries it into f.

### 9.3 Spills (D5) and m8

- **`card`:** `fit_lock` passes the lock's fingerprint class (`found[0].plan.fingerprint_class`,
  `<gpu>[+<cpu tier>]/<toolchain>`) to a hook naming it (`inspect.signature`). `glm5_next` keys `MODE2` and `KNEE` by
  its GPU part. A borrowed lock passes its own card: the spill measured with the lock it refines.
- **The merge is first-setter-wins** (`spill = got[0] | spill`), with the model plugin asked first. A spill adding
  no option is dropped with its note (A m5, B m1): GLM's set on the RTX 5090 prints GLM's note alone.
- **Std's generic spill** sets `moe.tiers` (std's two tiers) and `moe.ram=fit` on every offload lock, unless the lock
  sets `moe.ram`.
- **`glm5_next`'s spill** is `MODE2["rtx-5090"]`: the measured set with `moe.ram=fit`. It returns None on any other
  card, so the generic spill applies there (the RTX 5080's lock today).
- **`qwen4_exp`:** its PLE spill and the generic one merge into one entry. Its note now gives m8's cost: where RAM then
  holds only a share of the experts, it serves without the MTP head, with decode 24–37% slower.
- **m8, the mechanism: a second spill entry without speculation, in core's history.**
  - A drafting lock's spill entry is preceded by the same entry with `max_draft_tokens` 0. Both are memory-only.
  - Under `fit` with f < 1 the store refuses an auxiliary model's experts (an explicit check, B-R9) as `OutOfUnits`.
    The supervisor then rolls back to the entry without speculation.
  - The refusal is the warning. It names the cause and the cost: "the host holds a share of the experts only
    (exclusive tiers), which take no auxiliary model's experts (a drafter's head …): `holon serve` serves without
    speculation next (--max-draft-tokens 0), its decode slower by the drafter's gain (the model's spill note gives
    it)".
  - The number is in the model's note, not the store's message: it is a model fact, and std holds none (24–37% is
    Flash-Next's, measured on an RTX 5080 and 5090). The rollback line prints the note.
  - A Flash-Next whose PLE spill alone fits (f ≥ 1) keeps MTP.
  - A spill returning `max_draft_tokens` 0 could not do this: the hook runs before the budget exists, so it cannot
    know f. It would drop MTP on every RAM-short Flash-Next host (24–37% of decode).
  - The cost is one failed start, built up to `before_materialize`, on the hosts that need it.
- **`check_locks` refuses `moe.ram=fit` in a lock.**

### 9.4 Core, ABI 1.7

**Core: +12 code lines.** `check_core_loc.py` reads 5,278 / 6,000 on top of `mxinO/fail-stop` (5,266 at 382fbf8e).

| Where | Lines | What |
|---|---|---|
| `build.py` | +4 | `host_left()`, and `ChildContext` forwarding it (0) |
| `api/model.py` | +1 | the member |
| `api/hooks.py` | +1 | `Spills.spill`'s `*, card: str = ""`, wrapped |
| `cli.py` | +2 | `card` (`import inspect`, the call) |
| `cli.py` | +2 | m8's entry |
| `memory.py` | +2 | a pin the OS refuses is `OutOfUnits`, a memory failure (the code review's A m4) |
| `cli.py` (merge, notes, the spill entries by identity), `engine.py` (`"slots": cfg.slots`) | 0 | |

The design's +5 to +6 had no m8 entry, no wrapped signature and no pin refusal.

**`check_abi.py` renders signatures; it does not compare names only (B-R5's premise).**
- A plugin-implemented hook's changed line counted as a major change.
- The rule now treats a keyword-only parameter with a default, added to an **optional** Protocol that plugins
  implement, as a minor addition. It already did so for core-implemented Protocols. Such a parameter is
  backward-compatible for callers and implementers: the core passes it only to an implementation naming it.
- The rule is stated in the script's docstring and in `core/abi.txt`'s header (`##` lines, which are not compared).
- `abi.txt` is regenerated (`--write`): `abi 1.7`, `BuildCtx.host_left`, `Spills.spill`'s `card`, and the `slots`
  options key.
- Manifests that need 1.7 say so: `std.tiered` (`host_left`) and `models.glm5_next` (`card`).

### 9.5 What the tests cover

- **`fit`'s arithmetic, engine-free** (`test_fit_solves_one_share_once_and_refuses_below_its_floor`):
  - f and `held`, with the 1/256 quantum (8 MiB on 2 GiB) on both sides;
  - the cap below 1, `whole`, and the floor;
  - the floor's refusal message with every term;
  - the lower third reading's refusal with both readings;
  - no budget, and the state tier.
- **The bitwise GPU test** (`test_the_offload_store_on_the_gpu_is_bitwise_the_resident_store`) under `fit`:
  - `ssd` at the floor's f;
  - `ssdfit` at half;
  - `fit` (f = 1, the inclusive store, its I/O worker started and never signalled), under GLM's staging terms
    (`ring_tokens` 32, a large `stream_tokens`, `moe.split` on) with D8g's threshold at 8 tokens: no pass runs in
    parts. It fails without the idle tasks, and without A-M1's gate.
- **f ≥ 1 identical to the lock on the CPU** (`test_fit_where_every_expert_fits_is_the_locks_store`):
  - the layout, the outputs, and no I/O pools;
  - m8's refusal (the last layer taken as a head's);
  - `*=fit` refused, and `moe.cpu_plan` beside `fit`.
- **Under the deployment's terms:** `_mode2` now sets `moe.ram=fit`, with `_fit_at(share)` telling the store the host
  memory left. The `mode2` cases run under it: the loan's parking, its plants, and D8x at a quarter.
- **Spills:**
  - first-setter-wins, `card` to 2- and 3-argument hooks, and the no-speculation entry
    (`core/tests/test_auto_config.py`);
  - GLM's spill per card through the real merge with std's (`test_glm_oracles.py`);
  - `host_left()`'s two terms and None; a pin the OS refuses is `OutOfUnits`.
- **`check_locks`** refusing `fit` (`core/tests/test_api.py`).

### 9.6 Limits and open items

- **A CPU share (F5b, `moe.cpu_plan`) beside `fit` is refused** (D1, "Limits").
- **Approved as built** by the coordinator: `check_abi`'s rule for optional hooks with `card`'s default, `slots` in
  `BuildCtx.options`, core's +10 (+12 with A m4), and m8's entry without speculation.
- **The code review** (A, B on `382fbf8e..f28f53c2`): A-M1, B-M1 (the README's Host RAM table) and the minors are
  taken in this round. Two docstrings now say how `BuildCtx.host_left` and `TierCtx.host_left` differ (B m3). The
  knee's note says "est." (B m2). Plan §10's follow-up is marked delivered (B m7). B m5 (a CPU-tier suffix on
  `card`) is latent and left.
- **Open:**
  - GLM's knee (`KNEE`) is a placeholder until step 2;
  - the RTX 5080 has no `MODE2` entry (D7).


## 10. Step 2-3: the grid and the exactness runs (2026-11-01, 59482cdb)

**The RTX 5090 grid** (one desktop; each cell its own job step, its cgroup cap enforced and logged; zero OOM kills in
every cell). The grid's log labels the 72, 80 and 88 GB steps "NO-CAP": the check script's threshold
(≤ 64 GiB, written for mode (ii)), not a missing cap. Each step's `memory.max` read 72, 80 and 88 GiB exactly, and f
scales with it; the script now labels any set cap. The arm is the lock's flags + `MODE2["rtx-5090"]` (`moe.ram=fit`) + `--auto`: what `holon serve`
gives at that RAM. A cell is the median over rounds of each round's median over the context's 8 requests; tok/s; SSD
MB a decode token over tokens 1-32 and D8x's parked rows at 1K beside it.

| Cap | Rounds | f (logged) | 1K prefill / decode | 4K | 32K prefill / decode | SSD 1-32, 1K / 32K | parked (1K) |
|---|---|---|---|---|---|---|---|
| 48 GB | 1 | 0.4297 | 109 / 11.7 | | 2,195 / 9.3 | 224 / 325 | 250 |
| 56 GB | 3 | 0.5312 | 161 / 17.0 | | 2,660 / 13.4 | 133 / 236 | 235 |
| 64 GB | 3 | 0.6289 | 292 / 24.9 | 771 / 22.0 | 3,079 / 19.5 | 60 / 141 | 227 |
| 72 GB | 3 | 0.7305 | 606 / 33.1 | | 3,196 / 24.2 | 11 / 76 | 151 |
| 80 GB | 1 | 0.8281 | 647 / 35.2 | | 3,268 / 30.2 | — / 19 | — |
| 88 GB, `fit` | 1 | 0.9258 | 647 / 35.0 | | 3,257 / 32.4 | — / 1 | — |
| 88 GB, the lock (it starts: budget 81.7 GiB) | 1 | — | 710 / 41.7 | | 3,454 / 36.8 | | |
| uncapped, the lock | 1 | — | 710 / 41.4 | 2,121 / 37.2 | 3,450 / 36.5 | | |

- **Monotonic in RAM** at every context, flat from 80 to 88 GB (within one round's noise), with the step up to the
  lock where it starts (88 GB here: `holon serve` takes the lock there). C's replay placed the bend right: steep from
  56 to ≈ 72 GB, flat past ≈ 80 GB.
- **At 64 GB, at or above the close's mode (ii)** (288 / 774 / 3,030 prefill, 24.4 / 21.5 / 19.3 decode).
- **The knee: 72 GB** (`KNEE`): below it decode falls fastest (32K decode 24.2 tok/s at 72, 19.5 at 64; 1K prefill
  606 against 292). The spill's note warns under it.
- **D6:** `read_ahead` 256 against 135 loses at 64 GB (1K prefill −15%, decode −7% to −9%) and at 80 GB (32K decode
  −5%), with D8x's parked rows unchanged (227 → 230): 135 stays. **F10 at 88 GB** against `placement.lfu`: lfu decodes
  at 24.7 / 22.1 against 35.0 / 32.4 tok/s: F10 stays (no budget-0 arm: `BUDGET` is a constant).

**Exactness** (each spill under its cap against its lock's uncapped snapshot, `exact.py --pools own`, state pools
equal, zero OOM kills):

| Lock | Cap | f (logged) | Steps | Result |
|---|---|---|---|---|
| GLM, RTX 5090, `MODE2` | 80 GB | 0.8320 | 2,157 | byte-identical |
| GLM, RTX 5080, std's generic spill | 64 GB | 0.6562 | 2,157 | byte-identical |
| Qwen3-30B-A3B Q4_K_M, RTX 5080 offload lock, generic spill | 16 GB | 0.3281 | 2,158 | byte-identical |
| Flash-Next IQ2_XS, RTX 5080, MTP off, PLE + generic spill | 36 GB | 0.7539 | 2,158 | byte-identical |
| the same | 48 GB | ≥ 1 ("every expert in RAM": the inclusive `fit`, A-M1's path) | 2,158 | byte-identical |

**Open:** m8 end to end (`holon serve` Flash-Next with MTP at 36 GB: the lock refused, the spill refusing the head,
the no-drafting entry serving) needs a current lock, so it runs after the close's lock round; the RTX 5080 has the
generic spill, its own bench (D7) not run.
