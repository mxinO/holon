# v1 — GLM on any host from its floor up: the RAM tier sized from the host budget

Status: **design note, rev 1 (2026-10-31), for the panel (A correctness, B architecture, C performance).** Nothing is
built.

**The owner's request (2026-10-31):** "we should not assume the 64g is the only one we need support, we should be able
to adapte to different mem size (with 64g as a lower limit maybe)".

**Goal.** On any host from GLM's floor up, `holon serve <model>` serves GLM-5.3-Flash at its best configuration for the
RAM it has, with no flags:
- the RAM tier is sized from auto-config's host budget, not mode (ii)'s fixed `moe.ram=*=0.6238`;
- speed rises with RAM, from mode (ii) at the floor to the lock (mode (i)) once every expert fits;
- below the floor, the start is refused, naming the measured minimum.

The mechanism lives in std, so every offload lock gets it (Flash-Next and Qwen3-30B-A3B as well as GLM). The model plugin
supplies only its measured tuning. There is no model logic in core.

## 1. What is fixed today, and why it is not enough

- `glm5_next`'s spill (`plugins/holon_plugins/models/glm5_next/deploy.py`) returns `MODE2`. Its RAM share,
  `moe.ram=*=0.6238`, is the share that fits a 64 GB host, whatever the host has.
  - **On a 48 GB host** the spill's start is refused too. The refusal's hint is std's generic one (B's s2 at the
    close).
  - **On a 96 GB host** it pins about 50 GiB where about 80 fit, so it runs at 64 GB speed. Mode (ii)'s decode at 4K
    and 32K is 0.98 and 0.92 of Maya's, against 1.11 and 1.05 in mode (i).
- The spill hook is called before the budget is read (`cli.py`: `fit_lock` comes before `host_budget`). It is given
  neither the budget nor the lock's card class. `MODE2` is therefore gated on `moe.lend_peak = require` as a proxy for
  "the RTX 5090 lock" (B's s1).
- **Std's generic spill** for other offload locks only *names* a manual `moe.ram='*=0.5'`. It offers nothing it can try.

## 2. The design

### D1. The RAM share is sized where the pools are known: in std, at the store's layout (recommended)

- **The new option value.** `moe.ram=fit` makes the pinned tier hold, per class, the expert share that fits the host
  budget:
  - **The budget:** `EngineConfig.host_memory`, which `holon serve` derives as auto-config's budget.
  - **Taken off first:** the store's own non-tier host pools (the staging and landing rows, the parking units) and every
    other host pool already requested. Among those are the token embeddings and PLE rows of `std.tiered.rows`.
- **How the store learns those bytes.** The store asks a build-time `host_left()` on its context. That is the same
  question the state tier asks today (`TierCtx.host_left`, `core/holon/api/tier.py`). The engine already knows the
  answer (`cfg.host_memory − ledger.host_bytes()`).
- **What core gives.** It exposes that answer on the plugin context at the store's build, as an addition. Exposing
  `host_left()` where the store builds is the open item for step 0.
  - **If the store's layout can take its RAM units after the ledger sizes them** (the RAM tier as a host *fill* pool,
    min = the floor, pref = every piece, as the VRAM cache fills free VRAM), that is preferred. It needs no new seam.
    The ledger does the arithmetic it already does for device pools, and the pools' order of requests cannot skew it.
  - **Otherwise,** the store reads `host_left()` at `resolver()` time, after reserving its own non-tier pools first.
  - Step 0 reads `TieredStore._tables` / `holds` against the ledger and picks one. Either way it is core +≤ 5 lines,
    additive (ABI 1.6 → 1.7 minor).
- **Why not in the spill hook.** The hook runs before the engine exists. It would have to re-derive the other pools'
  bytes (rows, staging, landing, parking) that the store and the ledger compute exactly at build. That arithmetic would
  be duplicated, and it would drift with every store change.

### D2. Per class: one fraction of experts, the same for every class

- **An expert is resident in RAM only when all its pieces are.** That is its gate/up and down pieces, in different
  classes (GLM: `gate_up.iq2_xxs`, `down.iq3_xxs` on 8 layers, `down.iq2_s` on 34).
- **A per-byte split would be wasted.** Splitting the budget by bytes, or by class heat, holds gate/up pieces of experts
  whose down pieces are on the SSD. Each such miss still reads the SSD, so the bytes buy nothing.
- **So `fit` solves one fraction `f` per expert:**

  `f = (host_left − reserve) / Σ_classes(pieces × unit_bytes)`

  - `held = floor(f × pieces)` in every class.
  - The store's RAM heat (`moe.ram_heat`) picks *which* experts. It already ranks per expert across classes.
- **This is what `*=0.6238` already does** (one fraction for all classes), so mode (ii) at 64 GB is the special case.
  Expect `f ≈ 0.62` there, up to the reserve.

### D3. At the boundary where everything fits: the lock, not a 100% exclusive tier

- `holon serve`'s history already tries the lock first and the spill only after the lock's start is refused for host
  memory (`serve_history`: `[defaults?, lock + spill, lock]`, tried last first). So a host that holds every expert
  gets mode (i) with no exclusive-tier overhead.
- Between the two, `fit` yields `f < 1`. At `f ≥ 1` (a lock refused for another host pool, not the tier), `fit`
  holds every piece, and the store is then non-exclusive, as the lock is.
- **Expected step at the boundary:** the lock's decode is about 1.5× mode (ii)'s at 64 GB. Part of that is the tier
  bookkeeping, which mode (ii) at `f → 1` still pays.
  - So speed rises monotonically with RAM, and jumps up at the boundary rather than down.
  - The panel may prefer the spill to stop where its predicted speed reaches the lock's. Nothing measured says it
    does.

### D4. The floor, derived, then measured

- **What decides the floor.** The floor is where the store cannot work, not where it is slow. Below the measured knee
  (D4.2), the start serves with a warning naming the knee and its speed.
- **The minimum the store refuses below** (computed at its layout, the refusal naming each term in bytes):
  - **headroom:** auto-config's kept-free, max(6 GiB, 5% of RAM), outside the budget;
  - **+ the non-pinned process:** Python, torch, the CUDA context, the GGUF's metadata, the scheduler. This is RSS
    the budget does not count. It is unmeasured for GLM, so step 0 measures it.
  - **+ the other pinned pools:** GLM's token embeddings (0.48 GiB, `rows.place=token_embd:host`), the store's staging,
    landing and parking units, and the I/O worker's bounce windows (≤ 512 MiB);
  - **+ a minimal RAM tier:** the experts a 32K prompt step lends and parks (D8b, D8l).
    - GLM's 32K tail lends 969 whole experts, 6.42 GiB at the RTX 5090 deployment (`docs/dev/v1-glm.md` §13 step 7).
    - Below that, lent experts go to the SSD and are reread in decode, which is D8l's 48% of decode-time SSD reads
      before D8l.
- **The prediction:** headroom 6 + process R + pools ≈ 1.5 + tier 6.5 ≈ **14 GiB + R**. If R ≈ 8 GiB as for
  Flash-Next, that is about 22 GiB, far below 64 GB.
- **D4.2, the knee.** At such a floor the store works but decode is SSD-bound: about 1 GB a token at f ≈ 0.08 against
  the drive's ≈ 6 GB/s, so about 5 tok/s.
  - The owner's "64 GB" is a performance floor in effect. It needs the knee, which is a measurement.
  - **Measured:** 48 GB and 40 GB on the capped RTX 5090 at 1K / 32K decode (one round each).
  - **Stated in the README:** the refusal floor (computed, logged), and the knee as "below N GB, decode under X tok/s".
  - The panel may instead set the refusal at the knee. My recommendation is to serve and warn: a slow answer beats
    none, and a cap is the user's to choose.

### D5. The model plugin's part: measured tuning, keyed by the card

- **The spill hook gains the lock's card class.** It does not need the budget: the store sizes from it (D1).
  - **Signature:** `spill(options, checkpoint, card)`.
  - **Compatibility:** core calls a spill with the old two-argument signature as today, and passes the class to one
    that takes three (one `inspect.signature` check), so the change is additive.
  - **Callers:** the three in-tree spills (std.moe, qwen4_exp, glm5_next) take it.
- **`glm5_next`'s spill becomes `MODE2[card]` with `moe.ram=fit`.**
  - **RTX 5090:** the measured set (8,192-row passes, the three loan options, `read_ahead` 135, `stream_tokens`
    65,536, F10).
  - **RTX 5080:** per D7.
  - **Other cards:** None. B's s1: an unmeasured card gets no unmeasured set.
- **Std's generic spill becomes a real one** for an offload lock without a model spill: `moe.tiers=…pinned,gguf_direct`
  `moe.ram=fit` plus F10.
  - Today it only names a manual 0.5 share. It is tried after the model's.
  - Qwen3-30B-A3B and Flash-Next's experts then degrade the same way. For Flash-Next this comes after its PLE spill,
    which it combines with.
  - **Gate:** the bitwise tests under exclusive tiers (`test_moe_engine_gpu.py`'s `ssd` cases) and `exact.py` on
    Flash-Next at one capped size.

### D6. Do mode (ii)'s 64 GB values hold at 80, 96 and 112 GB?

The prediction is from step 7's replays and probes. Only what is uncertain is measured.

| Value | Depends on | Prediction | Measure? |
|---|---|---|---|
| `ggml_cuda.pass_rows` 8,192 | VRAM (the loan beside a pass's buffers) | holds: RAM does not enter | no |
| loan options (1.12, 8192,32768, 0.55, 1,024 MiB) | VRAM, the profile's peak | holds | no |
| `moe.stream_tokens` 65,536 | prompt streaming vs the tail | holds | no |
| F10 (`placement.promote@1`, D8t's budget) | the VRAM cache's churn | holds; it promotes from RAM or SSD alike | no |
| `moe.ram_heat` 0.999 | which experts RAM keeps | holds; it matters less as f → 1 | no |
| `moe.read_ahead` 135 | the SSD-only experts a prompt step reads ahead | **uncertain:** with fewer SSD-only experts at f ≈ 0.95, its landing rows may cost more VRAM than they save | **yes:** 135 against 64 at 96 GB, prefill 1K / 4K |

### D7. The RTX 5080

- **Its lock serves 8K steps without a loan** (`moe.lend_peak` off, §7.2). Mode (ii) has never run there.
- **The rule extends** if the exclusive tiers start and are exact at its deployment (staging lent from the cache, no
  step-peak loan) with a measured set.
- **Measure:**
  - a smoke and `exact.py` at the 5080 lock's flags + `moe.ram=fit` under a 64 GiB cap;
  - then one capped bench at 64 GB against the lock, `read_ahead` {135, 64}.
- **If it starts and is exact,** `MODE2["rtx-5080"]` ships with its own reported numbers, not gated against Maya. Maya
  on 16 GB in mode (ii) is not a program bar.
- **If not,** the 5080's spill stays None, with the reason recorded.

## 3. Acceptance

1. **Capped RTX 5090** (cgroup-enforced job step, the node's NVMe, the bench's D11 method, 3 rounds after a warm-up)
   at 64, 80 and 96 GB, plus the uncapped lock:
   - decode and prefill at 1K / 4K / 32K rise monotonically with RAM;
   - at 64 GB they are at or above today's mode (ii) (the close's record), within its round spread;
   - where every expert fits, they match the lock.
   - Plus 48 and 40 GB for the knee (D4.2), one round each, reported.
2. **Zero OOM kills** at every size (`memory.events`), and the refusal below the floor names it.
3. **`exact.py` at 80 GB:** the spill as `holon serve` gives it, byte-identical to the lock's first run, on the RTX
   5090.
4. **The RTX 5080** per D7.
5. **Tests** under the deployment's terms (the `mode2` fixture with `moe.ram=fit`):
   - `fit`'s arithmetic engine-free (fractions, the floor's refusal message, `f ≥ 1`);
   - the bitwise `ssd` cases at a `fit` share;
   - the CPU default suite with no GPU visible (`CUDA_VISIBLE_DEVICES=`), as CPU CI runs it.
6. **The real-model smoke** at 80 GB before reporting built.

## 4. Locks

- **The core addition** (`host_left` at build, and the 3-argument spill call) outdates every lock: 27, on four card
  classes.
- **Std** (`std.tiered`, `std.moe`) and `models.glm5_next` / `models.qwen4_exp` change too. They are covered by the same
  round.
- **Predicted:** every trace byte-identical, because no lock sets `moe.ram` and `fit` is only in spills.
- **One round at the close,** as release §22. A moved trace stops for review.

## 5. Steps

| Step | Work | Check |
|---|---|---|
| 0 | Measure GLM's non-pinned RSS (R) at the lock and at mode (ii); read the store against the ledger and pick D1's form | R in bytes; the panel told which form |
| 1 | `fit` in `std.tiered` (D1, D2); the refusal (D4); the spill's card (D5); `glm5_next`'s `MODE2[card]`; std's real spill | engine-free tests; `ssd` bitwise at `fit`; CPU suite without a GPU |
| 2 | Capped RTX 5090: 64 / 80 / 96 GB + the lock; `read_ahead` at 96 GB (D6); 48 / 40 GB knee | acceptance 1, 2 |
| 3 | `exact.py` at 80 GB; the RTX 5080 (D7) | acceptance 3, 4 |
| 4 | The lock round; README (GLM: any host from the floor; the knee; the table by RAM size) | acceptance 5, 6; the panel's delta review |
