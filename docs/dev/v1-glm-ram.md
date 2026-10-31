# v1 — GLM on any host from its floor up: the RAM tier sized from the host budget

Status: **design note, rev 2 (2026-10-31), answering the panel's review of rev 1** (A correctness, B architecture, C
performance; all three: buildable after rev 2). The answers are in §7. Nothing is built.

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
- **When.** The store resolves `fit` once, at its first `holds()` question (the first `resolver()`, at build), before
  the startup profile. It caches f, and passes `*=f` to the tier. Every later `holds()`, the profile's minimum layout
  included, uses the cached f. A host pool is never re-pinned or re-seeded (B-R1, A-RAM-M1.2).
- **Against what:** `BuildCtx.host_left()`, a new optional API member: `cfg.host_memory − ledger.host_bytes()`.
  - It is **not gated on `clamp_context`**, so `exact.py`, the bench and the tests have a budget whenever
    `host_memory` is set (B-R1). A placement-only choice is not a lock degradation.
  - **Without a host budget, `fit` refuses** ("moe.ram=fit needs a host budget: `holon serve` derives one;
    elsewhere, --host-memory").
- **Taken off before solving:**
  - the store's own host pools: the read-ahead landing rows and their on-demand halves, the I/O and run tables;
  - every host pool already requested: the embeddings and the PLE rows of `std.tiered.rows`;
  - **the state tier's level as a fixed request,** its configured bytes, if a state tier is set and has not built
    yet;
  - **a constant reserve:** the I/O worker's bounce windows (≤ 512 MiB), the pinned buffers outside the ledger
    (`_readback`, `_move_host`, D8g's lists; A-RAM-M1.4), and R (D4).
- **Re-checked before pinning.** Just before the tier's pool is requested, the store re-reads `platform.host_ram()`.
  It takes `f = min(f, f(live available − headroom − R))`, so the serve-start snapshot cannot over-commit by what
  other processes took meanwhile (A-RAM-M1.3).
- **One owner (A-RAM-M1).** Under `fit` the store is the only reader of the host remainder. A state tier set beside it
  takes its configured level as a fixed request, and does not clamp to `host_left()`. Its build checks `fit` among the
  options and fails rather than race.
- **Quantized.** f is rounded down to 1/256, so placement does not drift with a few MB of free RAM from start to start
  (A-RAM-m1).

**Step 0 prices B-R4's zero-core form against this,** and picks it only if its restructure is small. That form is one
host pool across classes, granted between min and pref by the existing request, with f derived from the grant. It
needs:
- `exclusive` decided by the option;
- the statics (`owner`, `n_ram`, `_ram`) sized by pieces;
- `Pinned` as one pool with per-class offsets.

Otherwise it is `host_left()` at build: about +3 core lines, ABI 1.6 → 1.7 minor (B-R4). Both forms keep the single
owner.

### D2. One fraction of whole experts; capped at saturation

- **The rule.** An expert serves from RAM only if all its pieces are there: gate/up and down are in different
  classes, and GLM's down classes cover disjoint layer groups. So `fit` solves one fraction:

  `f = min(f_sat, (host_left − pools − reserve) / Σ_classes(pieces × unit_bytes))`

  with `held = max(1, floor(f × pieces))` in each class. The store's RAM heat (`moe.ram_heat`) picks which experts.
- **The cap at saturation (C §1).** `f_sat = (pieces not in the VRAM cache + the largest step's lend count) / pieces`.
  - Past it there are no SSD-only experts left and nothing to park, so more units are wasted pinned RAM.
  - It is about 0.82–0.84 at the RTX 5090's GLM deployment.
  - The store knows both terms at its layout: the cache's slots by class, and the tail's lend count.
- **One f against heat-per-byte (A-RAM-m2, C §3).** Step 0 checks offline, from the routing profile's per-expert heat,
  the route mass captured by one f against heat-greedy whole experts at f = 0.4, 0.62 and 0.74.
  - If the gap is under 1%, keep one f.
  - If more, `fit` computes the per-class counts from the profile instead. That is the same output shape.
- **The MTP head's experts** (an auxiliary group, Flash-Next's `aux.*` classes) count among the pieces and take the
  same f. GLM's head is unused.
- **`f ≥ 1`:** every piece is held, the tier is not exclusive, and the store runs as the lock does. It is tested
  identical to the lock.

### D3. Where everything fits: the lock

- `holon serve`'s history tries the lock first, and the spill only after the lock's start is refused for host memory.
  So a host that holds every expert gets mode (i).
- C's model predicts the curve:
  - a rise from the floor to a knee at about 72–76 GB;
  - a plateau from about 84 GB, where f is at f_sat;
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
- **Memory-failure retries (A-RAM-M2).** Under `fit`, a start killed for memory re-derives f from a fresh snapshot, so
  a systematic undercount would loop.
  - Each memory-failure retry of a `fit` entry doubles the reserve, at most 3 times, said in the log. The supervisor
    passes the retry count; step 0 checks whether the existing restart admission's lowered cap already gives it with no
    new line.
  - Then the start refuses with the measured shortfall.

### D5. Spills: the card, first-setter-wins, and the generic spill

- **The card.** `spill(options, checkpoint, *, card)`, keyword-only. Core passes `card=` only to a hook whose
  signature names it (one `inspect.signature` check), so two-argument hooks keep working. The new signature is written
  into `core/abi.txt` by hand with the 1.7 bump, since `check_abi.py` compares names only (B-R5).
- **First-setter-wins (B-R2).** `fit_lock` merges `spill = got[0] | spill`, with the model plugin asked first, so the
  model's measured set refines std's generic one. The line count is unchanged.
- **Std's generic spill becomes a real one** for any offload lock: only `moe.tiers=std.tiered.pinned@1,std.tiered.gguf_direct@1`
  and `moe.ram=fit`. No F10: `placement.promote@1` is a first-party plugin measured on GLM only (B-R3). It applies on
  the cards where the exclusive-tier tests ran: the bitwise `ssd` cases and `exact.py` (A-RAM-m4).
- **`glm5_next`'s spill becomes `MODE2[card]` with `moe.ram=fit`.**
  - **RTX 5090:** the measured set (8,192-row passes, the loan options, `read_ahead`, `stream_tokens` 65,536, F10).
  - **RTX 5080:** per D7.
  - **Other cards:** GLM's spill returns nothing and the generic spill applies (lfu, std's tiers). Its tuned set is
    never applied to a card it was not measured on.
- **`check_locks` refuses `fit` in a lock** (a CI check). A lock's placement is fixed; `fit` lives only in spills.

### D6. Which 64 GB values move with RAM (C §2)

| Value | Prediction | Measured |
|---|---|---|
| `pass_rows` 8,192, the loan options, `stream_tokens` 65,536, `ram_heat` 0.999 | hold (VRAM-side, or matter less as SSD-only shrinks) | no |
| `read_ahead` 135 **with D8x's rows** | coupled. D8x parks a short prompt's ≈ 225 lent experts in 2 × `read_ahead` rows a role, 128 at 64 GB, so ≈ 100 park over RAM again | 1K prefill and D8x's `parked_in_rows`, at 64 GB and 80 GB: `read_ahead` 135 against D8x's rows decoupled from it |
| F10's budget (1 a layer) | likely at the plateau, where promotions only shuffle VRAM and RAM | one arm at 80 GB: budget 1 against 0 against `placement.lfu`, 32K decode |
| Parking (D8l, D8x) | self-adjusting: idle past f_sat | counters only |

### D7. The RTX 5080

- Its lock serves 8K steps without a loan (`moe.lend_peak` off). Mode (ii) has never run there.
- **Step 3:** a smoke and `exact.py` at its lock's flags with the generic spill, under a 64 GiB cap.
- **If exact:** one capped bench at 64 GB against the lock, with `read_ahead` and F10 as in D6. `MODE2["rtx-5080"]`
  ships with its own reported numbers (Maya on 16 GB is not a program bar).
- **If not,** the 5080's GLM spill stays None (the generic spill too), with the reason recorded.

## 3. Acceptance

1. **The capped RTX 5090** (a cgroup-enforced job step, the node's NVMe, D11's method):
   - 64, 72 and 80 GB at 3 rounds;
   - 96 GB at 1 round, recording which of the lock and the spill started;
   - 48 and 40 GB at 1 round each, for the knee;
   - the uncapped lock.

   At each size: 1K and 32K (4K only at 64 GB and the lock), SSD MB a decode token over tokens 1–32, and D8x's
   `parked_in_rows`. Speed rises with RAM. At 64 GB it is at or above the close's mode (ii) within its round spread.
   The plateau is reported against the lock. The grid is placed by C's replay (step 0) before the GPU runs, and moved
   if the curve bends elsewhere.
2. **Zero OOM kills** at every size (`memory.events`). Below the floor the refusal comes at build, with its terms.
3. **`exact.py`:**
   - the RTX 5090 at 80 GB, the spill as `holon serve` gives it, byte-identical to the lock's first run;
   - the generic spill on Qwen3-30B-A3B (the RTX 5080 offload lock) and on Flash-Next (one card), each capped.
4. **Tests,** under the deployment's terms (the `mode2` fixture with `moe.ram=fit`):
   - `fit`'s arithmetic, engine-free: quantization, f_sat, the floor's refusal message, `f ≥ 1`;
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
| A-RAM-M1 one owner, sized once, re-checked before pinning, pinned buffers counted | **Taken** (D1): the store is the only reader; the state tier a fixed request; before the profile; live re-read; reserve |
| A-RAM-M2 a memory-failure restart loops | **Taken** (D4): the reserve doubles per retry, at most 3, then refuse |
| A-RAM-M3 the floor from real requests; the 6.4 GiB tail is a knee term | **Taken** (D4): the ledger's requests + R at a 32K peak + ≥ 1 unit a class + the working tier; the tail a knee term |
| A-RAM-m1 quantize f | **Taken** (D1): 1/256, down |
| A-RAM-m2 / C §3 one f against heat-greedy | **Taken** (D2): checked offline in step 0; keep one f under 1% |
| A-RAM-m3 refuse at build | **Taken** (D4) |
| A-RAM-m4 the spill only where tested | **Taken** (D5) |
| C §1 cap at saturation; the knee ≈ 72–76 GB; plateau; +20% step | **Taken** (D2's f_sat, D3, D4) |
| C §1 replay first | **Taken** (step 0; C writes it) |
| C §2 `read_ahead` coupled with D8x's rows; F10 at the plateau | **Taken** (D6) |
| C §4 the grid | **Taken** (acceptance 1) |
| Other: `check_locks` refuses `fit`; the MTP head under `fit`; generic-spill exactness on the 30B and Flash-Next; `f ≥ 1` identical | **Taken** (D5, D2, acceptance 3, 4) |
