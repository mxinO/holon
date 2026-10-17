# v1 — expert offload: MoE models larger than VRAM on one consumer GPU

Status: **rev 4.2, panel-clean** (2026-10-17: all three reviewers report nothing remaining after four rounds); its
parameters freeze when step 0's evidence lands. Rev 4.3 applies M6's evidence (§13: the default policy is `lfu`); rev 4.4 its panel round (10 + 7 findings), panel-clean
again on 2026-10-17; rev 4.5 sets the parameters from the desktop benches (§13) — the design is frozen. Rev 1 drew 52 findings (A 20, B 15, C 17), rev 2
27 (A 11, B 8, C 8), rev 3 20 (A 8, B 7, C 5); §12 logs each and where it is answered. No code yet. Evidence base: the local research note
`docs/brainstorm/v1-offload-research.md` (not published; "R §n"), which cites every reference engine by `file:line`
in its local clone; Project Maya is R §3.1b.

Scope: plan §10 (v1, "Expert caching/offload"), with the owner's direction of 2026-10-16/17:
- support **the most popular strategies of today**, with **Strata** as the primary reference, FreeToken second, and
  llama.cpp's static split as the baseline and fallback only;
- **a new strategy must be easy to add as a plugin**: host-side policies between steps in Python, in-step policies
  as kernel plugins;
- **target models: Qwen3.8-Flash-Next** (compared with Strata) and **GLM-5.3-Flash** (compared with Project Maya, the
  Strata-derived GLM engine). Qwen3.5-122B-A10B leaves the plan.

**The v1 offload program is four milestones** (D2), so that each closes on its own evidence and one branch changes
the core at a time:

| Milestone | Doc | Closes on | Core |
|---|---|---|---|
| **X — expert offload** (this doc) | `docs/v1-offload.md` | the development gate: Qwen3-30B-A3B on the RTX 5080, at its natural fit and at a capped ≈ 25% fit | S0–S5 (§4) |
| K — KV and checkpoint tiers | own doc, after X | resume from RAM / SSD, exact; agent-trace TTFT and concurrency | prefix-index tiers (§11) |
| FN — Qwen3.8-Flash-Next | own doc, in parallel with X | the model on Holon, then offload vs Strata (bars §1.4) | hyper-connection residual; listed now (§4.7) |
| GLM — GLM-5.3-Flash | own doc, after FN's shared units | the model on Holon, then offload vs Project Maya (bars §1.4) | none expected beyond FN's |

The VMM ledger and live rebalance (plan §10's third offload line) is X's second wave or its own milestone (D13).

## 0. The references in one table

| | Strata (primary) | Project Maya (GLM) | FreeToken | llama.cpp |
|---|---|---|---|---|
| Models | Qwen3.8-Flash-Next only | GLM-5.3-Flash only | many | many |
| Expert tiers | VRAM cache over all experts pinned in RAM (or mmap of the files when RAM is short) | exclusive VRAM / pinned RAM / NVMe: each expert in exactly one; RAM sized from free RAM less 6 GB | VRAM cache over pinned RAM | static per-layer CPU or GPU; new `--moe-cache-mib` LRU |
| Residency | seeded from a routing profile; every 4 windows up to 96 swaps by decayed routing counts (×0.7, gain ≥ 1.5); the victim is unmapped before its slot is overwritten | profile warm start; the device promotes a RAM or disk miss into one of 3 spare slots per layer; every step refills the spares by evicting the layer's LRU expert, demoted to RAM if hotter than RAM's victim, else disk-only | in-graph timestamp LRU, fetch on miss | fixed (LRU in the new cache) |
| Decode miss | a share over PCIe to the GPU, the rest computed by the CPU in RAM, through in-graph spin waits; share `L/(L+1.85R)` refit from live timings | RAM: the CPU takes the coldest by count (a per-miss-count table, re-timed live), the GPU gathers the rest zero-copy; disk: the GPU waits while a host thread reads the expert from the GGUF with `O_DIRECT` | offload / CPU / hybrid by a bandwidth benchmark | CPU computes |
| Prefill | non-resident experts stream through a ring, next layer during this layer's attention; a CPU share for small chunks | chunks up to 32K, buffers borrowed from the expert pool's tail; resident experts in place, routed misses staged, likely RAM experts prestaged | every expert of the layer through a double buffer, evicting cached slots | selected experts uploaded for batches ≥ 32 |
| Output-changing defaults | route tail skip (missed low-rank experts dropped, +10–24% decode); timing-driven splits; CPU/GPU rounding moves with placement | none approximate; runs differ by timing (17 of 70 stream pairs identical by default, 70 of 70 with its fixed-placement recipe); MTP off on one GPU | — | — |

Both leaders are single-model engines with large switch sets (Strata: 529 `STRATA_*` variables). Holon's offer is
the same strategy set as plugins over any MoE model plugin, **exact by default, with reproducible placement**.

## 1. Goal and acceptance

Run MoE models whose experts do not fit in VRAM on one consumer GPU, as fast as the strongest single-model engine
for that model, with exact defaults and reproducible runs.

1. **Strategies are plugins** (D3, D16).
   - Shipped in X: placement policies `lfu` (F11, the default by §8's rule), `static` (F1), `profile_decay` (F2–F3,
     Strata parity) and `lru` (F4); miss paths PCIe fetch
     (F5a) and CPU compute (F5b) with a per-miss-count split; the SSD tier (F9); prefill streaming (F7).
   - **Held-out proof:** the std offload API is tagged at the end of step 6, once every std change of X has landed
     (steps 5–6 still change std: the Tier protocol's storage, F9, F5b; panel B-s2-1). After the tag, three strategies land as
     commits whose diff touches only their own plugin directory and tests (no core, no std diff; checked with the
     existing `check_core_diff.sh` and `git diff --stat`): **F10** promote-on-fetch (host and in-graph parts, the
     hardest case: its `moe_resolve` variant is a contract the plugin owns, with its own `reference.py` and CPU
     provider, D3), **A1** route tail skip (`approx`), and **a tier this document does not design** — the
     page-cache low-RAM tier the research note ranks first among the extras ("mmap", in effect): a `pread` tier with
     O_DIRECT off, its own storage decisions in `prepare` / `holds` (ARC as a bonus).
   - Every core seam (§4) is proven first with a toy plugin in `core/tests`.
2. **Exactness** (D4, §6):
   - **every placement is exact on the GPU path:** with the CPU share off (the default unless D4's rule picks the
     split), outputs are bitwise independent of which experts are resident, of the arena size and slot order, and of
     the policy — snapshot traces byte-identical to `std.moe@1` all-resident, with adaptive placement **on** in the
     snapshot;
   - with the CPU split on: run-identical over 3 fresh engines on the snapshot trace, and bounded (L2 within budget)
     against a path-aware floor (§6).
3. **Development gate** (milestone X's close), Qwen3-30B-A3B Q4_K_M on the RTX 5080, same file and host, no drafting,
   medians of interleaved rounds, contexts ≤ 32K (the model's native limit is 40K), at the natural fit (≈ 47% of
   experts in VRAM at 32K) **and** capped to ≈ 25%. Every row is predicted from step 0's roofline before it is
   measured. The natural-fit rows are judged at step 4 with the F5a lock; the capped-fit rows and the F1 row with
   the lock tune picks after step 6 (F5b may be what makes them pass: the vehicle routes ≈ 1.1 GB of experts per
   token, so at 25% cached decode is miss-bound, panel C-r2-1):

   | Against | Decode | Prefill |
   |---|---|---|
   | FreeToken (`auto` and `offload`) | ≥ 1.0× | ≥ 1.0× |
   | llama.cpp static split (`--fit`, tuned `--n-cpu-moe`) | ≥ 1.5× | ≥ 1.0× |
   | llama.cpp `--moe-cache-mib` | ≥ 1.1× | ≥ 1.0× |
   | Holon F1 (static) vs llama.cpp's static split | ≥ 0.95× | — |

4. **Program bars** (acceptance of FN and GLM, recorded here so the offload design answers to them; proposed, owner
   to confirm). Like-for-like only: Holon's exact defaults against each reference's no-approximation settings, and
   Holon's opt-in `approx` against the reference's defaults:

   | Against | Model, card | Decode | Prefill |
   |---|---|---|---|
   | Strata, no approximation (`STRATA_ROUTE_TAIL_SKIP=0 STRATA_IQ_MT_MIN=1`) | Flash-Next IQ2_XS on RTX 5080; IQ3_S on RTX 5090 | ≥ 1.0× | ≥ 0.9× |
   | Strata defaults vs Holon with A1 | same | ≥ 1.0× | ≥ 0.9× |
   | Project Maya defaults (no approximation by default) | GLM-5.3-Flash Maya-S on RTX 5090 | ≥ 1.0× | ≥ 0.9× |

   Holon-exact against Strata's defaults is reported, not gated. Plan §9's kill line (< 0.8× → rework the offload
   path before adding features) applies to the like-for-like rows.
5. **Memory:** pinned host memory is allocated exactly (no power-of-two rounding) and accounted against
   `host_memory`; VRAM and host splits are pinned by the lock; startup refuses with a per-pool report when a pinned
   split does not fit.
6. **Core:** only the seams of §4, each an additive ABI minor (ABI 1.2), toy-proven; ≈ 480 new code lines
   (estimates in §4), core ≤ 6000 (4248 today). Suite time grows only by the new plugins' own tests.
7. **Locks:** offload deployments get locks on the classes they run on; the existing locks are propagated once, at
   X's close (D15).

Non-goals of X: KV and checkpoint tiers (milestone K); the two model plugins (FN, GLM); multi-GPU layer split and
peer-GPU tiers; dense (HF bf16) expert offload — v1 offloads quantized experts adopted as `QTensor`s; EXL3;
predictive prefetch (D11); a CPU kernel matching the GPU's rounding (D7); expert classes with per-expert side
tensors or fp4 activations (NVFP4 `scale2`, MXFP4) — X offloads GGUF integer-quant classes only.

## 2. Decisions

**D1 — Strategy set: Strata's and Maya's, exact by default.** Classes: **H** host policy (Python, between steps),
**D** in-step device kernel (kernel plugin), **C** CPU kernel, **std** library.

| # | Strategy | Reference | Class | X default |
|---|---|---|---|---|
| F1 | Static split: a frozen residency map | llama.cpp `--n-cpu-moe` | H | fallback without a profile |
| F2 | Profile-seeded expert cache: VRAM slots filled from a ranked (layer, expert) profile — 4 slots per layer first, then the global ranking | Strata X1, Maya G6 | H | on (the warm start of every policy) |
| F3 | Adaptive swaps by decayed routing counts on a fixed cadence of committed steps | Strata X2 | H | Strata-parity arm (its swap copies cost PCIe that Strata's CPU-heavy split leaves idle; 22–52% slower than F11 with F5a, §13) |
| F4 | LRU fetch-on-miss, logical-step timestamps, victims chosen in the graph | FreeToken, llama.cpp cache | D + H | comparison arm (within 7% of F11) |
| F11 | **Decayed-count admission on fetch** (`lfu`): per-(layer, expert) counts, +1 per routed decode entry, ×0.915 per decode step; a miss is gathered straight into the coldest resident slot the step does not need when its count is higher, else into staging; the cache is device-owned, the host only seeds it | M6 (§13) | D + H | **on** |
| F5a | Misses fetched over PCIe by an in-graph zero-copy gather into staging slots | Strata X5 mode 2, FreeToken | D | on |
| F5b | Misses computed by the CPU in RAM through the in-graph handshake; which misses go to the CPU is the provider's rule (Strata: last in routing order; Maya: coldest by count) | Strata X4–X6, Maya G2, G9 | D + C | by D4's rule |
| F7 | Prefill: non-resident experts stream through a ring, next layer during this layer, when D8's time rule says so; else routed-only staging | Strata X8, Maya | D + std | on |
| F8 | Prefill CPU share for small chunks | Strata X9 | C + H | only if step 0 shows ≥ 15% |
| F9 | SSD tier: experts outside the RAM tier read from the GGUF's byte ranges with `O_DIRECT` by dedicated I/O workers into pinned staging | Maya G4–G5 | std + C | on when RAM is short |
| F10 | Promote-on-fetch into device-owned spare slots per layer; each step boundary refills the spares | Maya G1–G3 | D + H | held-out plugin (acceptance 1) |
| A1 | Route tail skip; Maya's miss / disk skips | Strata A1 (default there), Maya AG1–AG3 | D + H | off; `approx` |

**D2 — Targets, the development vehicle, and four milestones.**
- **FN: Qwen3.8-Flash-Next** (48 layers, 512 experts top-10 + shared, QSA + indexer alternating with GDN,
  hyper-connections, the PLE n-gram table on SSD, MTP), compared with Strata on its own files.
- **GLM: GLM-5.3-Flash** (45 layers: 11 MLA + DSA indexer (top-2048), 34 KDA; 288 experts top-8 + 1 shared, the
  first 3 dense; mHC; sigmoid `noaux_tc` routing with a SwiGLU clamp of 10), compared with Project Maya on Maya-S.
  Every Maya quant and Unsloth's UD-IQ4_XS use upstream ggml types only, all in `kernels.ggml_cuda` (IQ1_M aside).
  GLM brings DSA forward from plan v2; its router and clamp are new kernel facts (`moe_topk` variant, a
  `linear_experts` fact), not core.
- **Development vehicle: Qwen3-30B-A3B Q4_K_M** (18.6 GB; 48 layers × 128 experts top-8) on the RTX 5080, where it
  does not fit: its plugin (`models.qwen3_moe`) and its RTX 4090 locks exist, so offload starts now and the 4090 runs
  the all-resident reference of the same file. FreeToken and llama.cpp run it too. (Rev 2 named Qwen3.5-35B-A3B and
  said its plugin exists; it does not — `models.qwen3_5` has no MoE. Qwen3.6-35B-A3B, GDN + MoE with a shared expert,
  is FN's natural first step and lands there.) The lead chose the vehicle (the owner was asked; reversible).
- **Why four milestones** (panel B12): X must not wait on two model plugins; KV tiers share only S2 with experts and
  carry their own index invariants (§11); the model milestones need core of their own (§4.7), and one branch changes
  the core at a time. X's seams are designed with the model milestones' needs listed, so no seam is redesigned later.

**D3 — Mechanism vs policy; what a strategy plugin is.** In-step work (resolve, fetch, expert GEMM over slots, CPU
compute, row patch, combine) is native, selected through op contracts. Its parameters — the residency map, the split
table, quotas — live in device or mapped buffers owned by the offload store. Between steps, pure-Python policies read
the step's routing counts and return **edits** (D4). Plugin kinds, all chosen by **open plugin options holding a
`Ref`** (no new core slots; the core never calls these, release §15.1's reason for cutting the slots stands):

| Kind | Option (example) | Contract | Instances in X |
|---|---|---|---|
| expert store | `moe.store = "std.moe.offload@1"` (absent: resident) | a std unit `ctx.unit(Ref)` binds | resident, offload |
| placement policy (host) | `moe.placement = "plugins.placement.profile_decay@1"` | `PlacementPolicy` protocol in `std.tiered`: `seed(profile, budget) → edits`, `after_commit(counts, view) → edits`, `device_state() → {name: spec}`, `resolve_variant: str | None` (the contract name; None = `moe_resolve@1`) | lfu (default), static, profile_decay, lru |
| miss path (device + CPU) | kernel selection through facts | `moe_resolve@1` (or a variant contract a policy plugin owns), `expert_fetch@1`, `cpu_experts@1`, `moe_patch_rows@1` (§5) | fetch, CPU |
| expert tier (host) | `moe.tiers = ["std.tiered.pinned@1", "std.tiered.gguf_direct@1"]` | `Tier` protocol in `std.tiered` (the tier owns its storage; settings, e.g. the RAM tier's `ram`, forwarded by the store): `device_readable: bool`, `holds(class, pieces) → how many it holds`, `prepare(ctx, class, unit_bytes, extents, staging, granule)` (takes its storage: a device-readable tier its mapped host pool, a file tier its descriptors), `units(class)` (a device-readable tier's pool), `read(class, piece) → descriptor` (`host_slot` for a device-readable tier holding every piece; for any other tier only `pread` — extents (fd, O_DIRECT on or off, offset, nbytes) —, which the store hands to a generic copy task of `cpu_task@1` into pinned staging and refuses at prepare in any other kind) — so tiers stay pure Python | pinned RAM, SSD |
| route transform (`approx`) | `moe.route_transform = "<name>"` — explicit option only, never by kernel priority; the name selects the contract `op.moe_route_transform.<name>@1` | std owns only the signature `(ids, weights, map) → ids, weights`; each transform is a contract its plugin owns (`op.moe_route_transform.<name>@1`, own oracle and CPU provider); a negative id means **dropped** (the rank's null slot, no fetch, no CPU job, no count), defined in `moe_resolve@1` before the tag | none in X before the tag; A1 after |

A new policy, tier or transform is a plugin naming its own Ref. A policy that needs device state declares it
(`device_state`) and names its resolve contract (`resolve_variant`). A variant whose semantics the std oracle does
not know (F4's in-graph LRU, F10's promotion) is **a contract the policy plugin owns** (`op.moe_resolve.<variant>@1`:
`moe_resolve@1`'s signature plus the declared state, so the store calls it unchanged; its own `reference.py` and CPU
provider in the plugin's directory; L0 runs any plugin-supplied oracle), so it lands without a std or `testing` diff
(panel B-r2-1). After the std offload API's tag (step 6), a resolve variant ships its own CUDA provider in its own
kernel plugin: `kernels.expert_io` takes no new policy codes then (it serves `@1`, `lfu` and `lru`, built with the
API). Route transforms follow the same rule. What one plugin cannot do: a
miss path that changes the graph shape (a new node between resolve and combine) is a std minor of the offload store
— recorded, not hidden.

**D4 — Exact by default; reproducible placement.**
- **Placement-invariant GPU path.** An expert's output must not depend on how its GEMM is launched. The vendored
  expert-id mmq switches to stream-k when tile efficiency is < 90%, splitting K at points that depend on the slot
  count (`mmq.cuh:1471-1478, 1093-1094`; on the RTX 5080 it engages at 16–85 slots). X adds a vendor patch forcing
  tiled mmq on the expert-id path, and a second one rewriting the id helper to build a compact list of active slots
  in O(T·k) (a stable sort of (slot, t·k + j) pairs) and launch mmq over min(S, T·k) entries instead of S (class
  arenas would otherwise multiply both the grid and the helper's scan by S/E: 24× at the vehicle's natural fit,
  panel A-r2-2; each tile is still one CTA in fixed K order, so invariance is unchanged).
  The two mmvq kernels (one row: generic; more: MoE) and mmq reduce in different orders, so **the path is the
  reference's**: `std.moe@1` cuts a step into 512-token chunks and the kernel picks mmvq or mmq from the chunk's row
  count, and every offload launch takes the path of the chunk its rows belong to, never its own (compacted) row
  count. **Decode, verify and mixed steps are never compacted** (one launch per class over all their rows, whatever
  the row count). In prefill, the mmvq limit is per ggml type (`use_mmvq_id`: 8 for Q4_K / Q6_K, 7 for IQ2_S and
  IQ3_XXS, 6 for IQ3_S on Turing and newer), so the rule is **per class**: a class's launches are compacted into ring
  and staging groups only when the chunk's row count exceeds that class's own limit, and a group with fewer rows
  than that limit + 1 is **padded with null-slot rows** up to it, so the kernel's own row count picks mmq, as the
  reference chunk does — no contract change (adding a parameter to a signature kernel plugins implement would be a
  major, design §4.5; panel B-r4-1, A-r4-4, A-r4.2-1). Padding rows carry zero activations, read null slots and are
  discarded. L0's tile-width range covers the padded launch, and L0 includes a chunk whose row count falls between
  two classes' limits. L0 checks, on every lock class including sm_120: an expert's
  rows bitwise equal across slot counts {E, E+1, 16, 74}, slot permutations, compacted launches (T' ≤ the mmvq limit
  out of a step with T above it) and mmq tile widths 8–128.
- **The default lock computes every expert on the GPU** (F5a only). Then every placement policy is exact, and
  adaptive placement runs in the snapshot (refines design §7.10's "adaptive policies frozen", which assumed a
  placement-dependent consumer). `holon_tools tune` measures the best F5b split against F5a per (model, class); the
  lock takes the split only if it gains ≥ 5% decode, and is then `bounded` (Maya measured its CPU lane at 4–10% on a
  5090, R §3.1b).
- **Decisions use committed counts, take effect at fixed steps.** A policy's edits computed after commit c are
  applied at launch c+2 (both overlap paths: at depth 1 commit c follows launch c+1; on the synchronous paths it
  precedes it — the fixed +2 makes them agree). An edit is **two-phase** (Strata unmaps first, `generate.cpp:8895`):
  at c+2 the device apply kernel **unmaps** the victim (its routes become misses), the copy into the freed slot is an
  async MemOp ordered after that step's last read of the slot, and at c+2+d (d fixed, default 2) the apply kernel
  **maps** the new expert after the stream waits on the copy's event — a stall if it has not landed, never a
  timing-dependent result. Two-phase host edits serve the host-edit policies (`static` re-seeds, `profile_decay`)
  and their ring lending; under the device-owned default (F11, F4) the cache has no host edits at all (below).
- **One writer per map region.** The residency map is one device buffer written only by the apply kernel (host
  edits, stream-ordered) and by in-graph resolvers for the regions the policy declares device-owned (the whole cache
  under F11 and F4; F10's spares). Host edits are **unconditional within host-owned regions**; a slot moves device → host only through
  an explicit release the device writes into its per-step delta log and the host acts on at a fixed lag, so a host
  copy never targets a slot the device may still map; host → device (F10's spare refill) happens in the same apply
  as the victim's unmap, with no copy. An edit whose precondition fails raises a device error flag (never a device
  trap, whose error is sticky), checked at the commit of step c+2 before that step's tokens are emitted: the engine
  stops with an error and discards the steps launched after it (panel A-r2-1, A-r3-5). The delta log returns with the
  counts, so the host's view is the device's, lagged by a fixed number of steps. Counts live in one device buffer
  zeroed in-graph each step and copied out through S1's per-launch pinned double buffer (a captured graph bakes in
  device addresses, so the device side is single-buffered).
- **Split decisions are deterministic.** The F5b split is a per-miss-count table `plan[n]` = the number of a layer's
  n distinct misses the CPU takes, frozen by tune (C4); ties by best routing rank, then expert id. It is
  deployment-level (a device buffer), never per request: misses are grouped across rows (D12).
- **Pinned sizes.** The lock pins the VRAM expert-slot counts and the RAM-tier slot counts through the store's own
  plugin option (slots per class; with it set the store requests fixed pools, min = pref; startup refuses if they do
  not fit), so placement never follows free memory at start (Maya's top non-determinism, R §3.1b).
- **Placement invariance on the bar card.** Besides the bitwise references on a 24 GB card (§6), the RTX 5080's
  snapshots at the natural fit, the capped fit and with adaptive placement must be byte-identical to each other.
- **Row kinds.** `moe_resolve@1` takes a per-row kind code — padding, decode / verify, prefill — so in-graph policies
  can tell them apart in mixed steps: padding rows are masked (no fetch, no CPU job, no counts); prefill rows never
  count or admit (M6 and panel C: counting prompts costs 4–6%); decode and verify rows count every routed entry at
  resolve time, rejected drafts included (acceptance is known only after the step; deterministic either way).
- **F11's rule, as M6 replayed it** (`spikes/x0_offload/m6/replay.py`), specified in its contract and oracle, in this
  order per step: decay (×0.915 in f32, on every step with at least one decode / verify row, graphed or eager;
  prefill-only steps do not decay), then increments (+1.0 per counted entry, order-independent), then pairing: per
  layer, the step's distinct misses (candidates, by count
  descending) pair with the layer's residents the step does not need (victims, by count ascending), ties by expert
  id; a candidate is admitted only if its count is strictly greater, the pairing stops at the first pair that fails,
  and each layer keeps its seeded quota. An admitted miss is gathered straight into its victim's slots (gate / up and
  down: the same layer, so the same classes); the rest go to staging. An expert needed by both a decode and a prefill
  row in one step has one slot: its admitted slot if admitted, else its staging or ring slot, read by both.
- **Lending under the device-owned default.** A ring-path step's slots are lent by the device: at the step's launch
  the policy's own resolve contract picks **whole experts** through its `lend(n) → experts` device op until every
  class has at least n lent slots (an expert's gate / up and down slots go together, so no expert is half-resident;
  F11: the coldest by count, ties by layer, then expert id; F4: the oldest; a later policy its own rule, with its own
  oracle), and the std apply kernel only unmaps and logs them (counts unchanged, panel B-r4.4-1, A-r4.4-2). A lent
  slot stays owned by its layer: on return it is that layer's first victim, ranked below every resident and filled
  unconditionally by the next admission, so the cache never shrinks (an L1 test: cache size unchanged after a lend
  and its return, panel A-r4.4-1). No host edit,
  no refill copy (panel C's replay: ≤ 1.3% even with random picks, against ≈ 12% for a standing 0.5 GB reserve).
  Host-edit policies keep two-phase lending. "Prefill rows never change residency" means they never admit; lending
  evicts by rule.

**D5 — The offload store, one per engine.** `std.moe@1` keeps its interface and binds its store through
`ctx.unit(Ref)` from the `moe.store` option. Every MoE layer (the MTP layer's included) gets the **same** store
object through `BuildCtx.shared` (S0). The store owns:
- **arenas per slot class** — one per (role, ggml type, shape), global across layers (a GGUF mixes types across
  layers and roles: Q4_K_M has Q6_K `ffn_down`; Maya-S has IQ2_XXS gate/up and IQ2_S / IQ3_XXS down). A class's
  arena is `[S, …]` with the contiguous expert stride `linear_experts@2` already assumes; its top k slots are **null
  slots** (zero weights), one per routing rank, so ids stay unique per token (ggml's id helper keeps one entry per
  (token, expert): a shared null slot corrupts rows on the mmq path, panel A2); then staging slots for graphed
  steps, min(E, k × the largest graphed **token** count, verify widths included) per class, reused layer after layer
  — 64 per class for the vehicle at `max_seqs` 8 without drafting, ≈ 0.28 GB with the null slots across its three
  classes (Q4_K gate_up 1.77 MB, Q4_K down 0.885 MB, Q6_K down 1.29 MB; §7), and E = 128 for the n-gram-drafter lock;
  "distinct misses of a layer ≤ staging" is an invariant with an L1 test, since F5a has no fallback. Every step that
  carries a prefill chunk, mixed steps included, takes the ring-borrow path (slots lent from the arena tail, D8).
  A launch there that is not compacted (a mixed step, or a chunk at or below a class's limit) needs all of the
  layer's misses resident at once, so the store lends at least min(E, k × the step's tokens) slots per class for such
  a step (the whole layer, 128 per class, for a mixed step on the vehicle), lent and returned under D4's lending rule; the
  L1 "distinct misses ≤ slots" invariant covers ring-path steps too (panel A-r4.3-1); then cache slots;
- the **residency map** `[L, E, roles] → slot | −1−tier` (tier codes say where a miss lives: RAM, SSD);
- the host tiers, the policy object, the split table, the per-step count and delta buffers (D4).

The store is built once per engine by its factory on the **root** build context, so an aux model's units (the MTP
layer) get the same object and its pools are not namespaced (panel A-r2-11). It learns its classes from the units
that bind to it: each expert stack adopts its `QTensor` without copying (the GGUF source maps the file) and hands the
store its layout and S4 extents; under the offload store the gate / up merge records per-expert extent pairs (gate
rows, then up rows) and never concatenates or moves the stacks — the merge is a member of the store, so the
resident store concatenates as today and `MoE.finalize_weights` has no branch (today's merge would read ≈ 12 GB of the
vehicle's experts and copy them to the device at startup, panel A-r4-3). At S0's `before_materialize` — after every model, aux models included, is built and
loaded, before the first materialize — the store forms its classes (role, ggml type, shape), requests its VRAM fill
pools and host pools, and fills the host pools from the extents with `O_DIRECT` reads (panel A-r3-2, B-r3-2: no
architecture keys in std, and per-tensor types are known only from the tensor table). The startup memory profile materializes minimum pools and runs a worst step, then the
ledger re-materializes: `after_materialize` runs after **every** materialize, resetting device state and re-seeding,
and the arena's minimum holds its null and staging slots (panel A-r2-3).

Per MoE layer the graph runs:

```
router → moe_topk → [moe_route_transform] → moe_resolve → [publish CPU jobs; signal] → expert_fetch
       → linear_experts(gate_up, class arena) → linear_experts(down, class arena) → [wait] → moe_patch_rows
       → moe_combine
```

Brackets are absent in the default lock. Shared experts stay resident. The plan is computed on the device
(FreeToken style); M5 measures a host-planned variant (Strata's flag A), which would be another `moe_resolve` provider.

**D6 — The CPU handshake is a kernel plugin, not core** (panel A12, B2; drift from design §7.4). S6's protocol — one
word per layer in mapped pinned memory, spin-kernel wait (memop wait on displays with a watchdog), a persistent
native worker, zero-copy payloads — ships as `kernels.cpu_task` (`[build]`, prebuilt like the other native plugins)
providing `cpu_task@1` signal / wait ops and a worker service, shared through S0. The core keeps only the lifecycle:
shared objects are shut down after `drain`, and the engine can pin its loop to a reserved core (`loop_core`). The
core needs no abort hook because `cpu_task@1` is fail-open by contract: a task error answers DONE with an error
flag; the spin-wait kernel is time-bounded (Maya bounds its wait at about 60 s) and on timeout raises the flag and
proceeds, which covers a dead or hung worker; the memop flavour, which cannot time out, has a watchdog thread in the
plugin that answers for a worker it finds dead. Once the error flag is set every later wait returns at once (else
a hung worker would cost the full bound at each of 48 layers). The store reports the flag at the step's commit (an error, the
engine stops), so a graph never spins forever and `drain` always returns (panel B-r2-7, B-r3-7, A-r3-5). Worker counts are tune's (S6 saturated DRAM with 2–4 threads); F9's I/O workers are separate
from compute workers, so a blocking read never stalls a layer's CPU experts.

**D7 — CPU kernels: ggml-cpu, bitwise per token and per thread count.** `kernels.ggml_cpu` vendors ggml-cpu (MIT) at
`kernels.ggml_cuda`'s pin and runs `vec_dot` over GGUF blocks in place. `cpu_experts@1` requires a token's result to
be bitwise the same alone or grouped (2–8 tokens) and at 1 or N threads (rows split across threads, never K); L0
checks both. Faster rows (Strata's multi-token i-quant kernels, MIT, design §5.6) are admitted only if bitwise equal
to ggml-cpu's per-token dot (Maya checks this at first use). ISA tiers are kernel constraints (`Constraint.cpu_isa`);
ggml-cpu's reductions differ by ISA, so a lock that uses F5b has the CPU's ISA tier in its class (S5). Dropped from
rev 1: a CPU kernel matching the GPU's rounding — it would need the GPU's SwiGLU and q8_1 quantization rebuilt
without fast math and per-path reduction orders (panel C2); F5a-by-default makes it unnecessary for exactness.

**D8 — Prefill streams what is missing, when it pays in time.** Per chunk and layer, F7 streams layer l+1's
non-resident experts through a ring while layer l runs if their bytes over the measured PCIe rate fit under layer l's
compute at that chunk size, or beat routed-only staging's exposed transfer time; else routed-only staging (C6:
coverage passes 50% at ≈ 64 tokens for top-8-of-256, so a coverage rule would stream almost always). Resident
experts are never re-sent (FreeToken's flaw). Ring slots are lent from the store's own arena under D4's lending rule
(by the device under the default policy, two-phase under host-edit policies) — plugin-internal. Streaming layer l+1 during layer l needs a second stream inside one step: it
belongs to a kernel-plugin op, `expert_prefetch@1` (provided with `expert_fetch@1`), which owns the copy stream and
its per-layer events (invariant 6 allows `torch.cuda` in kernel plugins only). Maya's prestage of likely RAM experts
is an F7 option measured in step 4. Prefill stays eager.

**D9 — VRAM: the expert arena fills what is left above a paged floor.** A plugin pool may declare `fill=True` with a
priority: after weights, workspaces, graphs, slot state and fixed plugin pools, a deployment `paged_min_tokens`
floor goes to paged state, then fill pools by priority up to their `pref_units`, then paged state takes any
remainder. The floor defaults to `max_model_len` for one sequence; benchmark locks name their context. Experts take
everything above the floor (C15: returning VRAM to idle KV loses to Strata and Maya on single-user traces); the floor
is the knob, set by L4. The lock pins the resulting slot counts (D4).

**D10 — Host tiers: exact allocation, exclusive when RAM is short.** Host pools are allocated once, before the first
materialize and outside the device profile cycle, filled from the weight extents, with `cudaHostAlloc` or `mmap` + `cudaHostRegister` (torch's pinned
allocator rounds to a power of two: 86 GB would pin 128 GiB, panel A6), and accounted against `host_memory`. The
RAM tier holds all experts when they fit; otherwise it is **exclusive of VRAM** (an expert lives in VRAM, RAM or SSD,
moved by D2H demotion as Maya does): with an inclusive tier, GLM on a 64 GB host sends ≈ 7.5% of routes to SSD
instead of ≈ 2.0% (C8). F9 reads the GGUF's byte ranges with `O_DIRECT`; weights load the same way, so the page
cache never double-counts the file. It needs each expert slice's file extent from the weight source (S4). The
exclusive tier's victim window (panel A-r2-6): between a victim's unmap and its D2H demotion landing, its tier code
says SSD (the file is always valid), never RAM; the demotion precedes the incoming H2D on the same copy stream; a RAM
slot freed by a promotion is reused only after the map lands at c+2+d. Under F11 an in-graph admission overwrites
its victim in the same layer, with no time for a demotion: the victim becomes SSD-only (the tier code written in the
graph, Maya's "else disk-only"), the admitted expert's RAM slot is released to the host through the delta log and
reused only after the fixed lag, and promoting hot SSD-only experts back into RAM is a host policy acting at the lag.
That churns: F11 admits ≈ 55 experts per decode token at 25% cached (panel C's replay), each sending a warm expert to
SSD-only — ≈ 157 MB per token of SSD re-reads on the vehicle, more than its whole modelled step. So **F11 is the
default only where RAM holds every expert** (X's development gate). Before step 5 the replay gains an exclusive RAM
tier with an SSD rate and compares F11 with SSD-only victims, F11 with deferred admission (the miss staged this
step, its victim demoted D2H, the map flipped at a fixed later step) and F10 with spares (Maya's structure); the
RAM-short default is the least modelled time, confirmed by step 5's RAM-capped run (panel C-r4.4-1).
**Replayed (2026-10-17, `spikes/x0_offload/m6/README_exclusive.md`, `evidence/x5_exclusive_tier_replay_2026-10-17.json`):
the RAM-short default is F11 with deferred admission.** Over M6's held-out pairs at 10 / 25 / 47% VRAM and RAM
holding 25 / 50 / 75% of the rest (SSD at 2.6 GB/s): SSD-only victims lose everywhere (1.2–3.7× the best; ≈ 235 MB
per token of SSD re-reads at 25% cached, worse than D10's estimate because RAM drains); deferred admission is least
in 5 of 9 cells and F10's spares in the other 4 by ≤ 2.7% with the flip at +2, and deferred admission wins all 45
pair-cells with the flip at +1. Rules for step 5: the flip lands at the first launch where the demotion and staged
copy are stream-ordered before it (+1 where the copy runs on the step stream, else +2 — an in-graph admission needs
no host lag); a victim goes to a free RAM slot if one exists, else displaces RAM's coldest expert if hotter, else
becomes SSD-only; a demand SSD read that is not admitted to VRAM stays in RAM (it is already in host memory; −2 to
−28%). RAM-short is a fit mode, not a fast path: 1.9–8.3× slower than RAM holding every expert on these drives.

**D11 — No predictive prefetch.** Strata's Foresight measured 0.91–1.00× on every box (R §3.1 E1) and Maya's
read-ahead lost too (R §3.1b EG1). Re-open only if M6 shows next-layer recall ≥ 85% and M1 ≥ 30% idle PCIe time in
decode.

**D12 — Batching is first-class.** Misses of all rows in a step (decode, verify windows, mixed steps) are grouped per
expert before the fetch and CPU paths, so a miss is paid once per step. Maya sends each request's routes through the
tiers one after another; Strata turns batching off by default.

**D13 — VMM elastic pools are the second wave** (Strata X3: KV grows into the expert arena and back without
recapture). After F1–F9, in X if time allows, else its own milestone; `quiesce` arrives with it (not before: X's map
changes are stream-ordered).

**D14 — Comparison method.** Same card, file, context and KV dtype, same host; drafting matched. Drafting is
placement-aware: on one offloaded GPU MTP loses on GLM (consecutive tokens share ≈ 1 of 8 experts per layer; Maya
turns it off), while Strata gains 1.6–1.8× from it on Flash-Next; the lock decides per (model, class). Every reference
runs its defaults and its no-approximation settings; each comparison pins the reference's commit. Metrics follow
Strata's tables: decode tok/s at 1K / 4K / 32K / 128K (256 tokens, greedy; the development vehicle stops at 32K), prefill at 1K / 4K / 32K, the agent trace
(TTFT / TPOT p50 and p99, wall time), expert hit rate and fetch / CPU / SSD shares, peak VRAM and RAM, L2 quality,
run-to-run identity. Reference engines build in `bench/.envs/`.

**D15 — One core round, one propagation.** X's seams land on this branch; the 22 existing locks are propagated
once, at X's close. FN's core seam (§4.7) follows on its own branch after X merges.

**D16 — The held-out proof replaces "designed-together" plugins** (panel B6): see acceptance 1.

## 3. What Holon has today

| Piece | State |
|---|---|
| MoE | `std.moe@1` (`std/holon_std/moe/units.py`): f32 router, `moe_topk@1` (softmax, unique ids), gate/up merged with SwiGLU and down as `linear_experts@2` over `[E, …]` stacks in VRAM, `moe_combine@1` in f32; named in M0.2 (`docs/m02-moe.md` D2) as the offload seam |
| Expert kernels | `kernels.ggml_cuda`: mmvq up to llama.cpp's `use_mmvq_id` row limit, else mmq; GGUF types incl. IQ1_S / IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S / IQ4_NL / Q2_K–Q6_K / Q8_0 |
| CPU handshake | spike S6 only (`spikes/s6_cpu_handshake/README.md`): 1.9–2.8 µs per spin handshake on the desktops, captured in FULL graphs, drain-first stop |
| Memory | `Ledger` with fixed device pools, sized after weights load and re-materialized around the profile; `BuildCtx.pool(name, unit_bytes, segments, min_units, pref_units)`; a second request of one pool name raises; `pinned_empty` uses torch's pinned allocator |
| MemOps | `MemOp(src_pool, src_units, dst_pool, dst_units, phase)` between equal-shaped device pools, eager, on the step stream |
| Sharing between units | none: `BuildCtx.unit` builds a new instance per call; `static(name)` shares tensors by name |
| Lifecycle | `ShutsDown` on logits processors, called by `Engine.close` |
| CLI | refuses a checkpoint larger than VRAM before fetching (`core/holon/core/cli.py`) |
| Fingerprint | GPU SKU, SM, VRAM, toolchain; no CPU facts |
| Cut from ABI 1.0 | slots `eviction_policy`, `placement_policy`, `tier`, `mover`, `connector`, `codec`, `checkpoint_policy`; `HitLen.cost`, `Payload.tier`; `Pool.base`, `capacity_dev`, `device`, `segments`; `Constraint.cpu_isa`; prop `batch_invariant` (`docs/release.md` §15.1) |

## 4. Core seams (additive ABI minors → ABI 1.2)

None names experts, KV or formats. Each is proven first with a toy plugin.

| # | Seam | Today → change | Toy proof | Lines |
|---|---|---|---|---|
| S0 | **Shared objects and their lifecycle** | none → `BuildCtx.shared(key, factory)`: one object per engine per key, engine-global across child (aux-model) contexts, the factory run with the root context; keys by convention the provider's resource name (`cpu_task.worker`); hooks it may implement: `before_materialize(ctx)` (once, after every model and aux model is built and loaded, before the first materialize: request pools), `after_materialize(ctx)` (after **every** materialize, before graph capture: reset and seed), `shutdown()` (after `drain`; generalizes `ShutsDown`); EngineConfig `loop_core` | two toy units, one in an aux context, share one object; it requests a pool in `before_materialize`, re-seeds it after each materialize and stops after drain | 80 |
| S1 | **Step hooks** | planner `prepare_device` only → a shared object may register `before_launch(step) → (writes, mem_ops)` (device writes and MemOps enqueued before the replay, stream-ordered) and `after_commit(step, telemetry)` (declared device telemetry buffers, copied D2H with the step's tokens into a per-launch pinned double buffer) | a toy counter and a map edit applied at launch c+2 under depth 1 and on the synchronous paths alike | 80 |
| S2 | **Host pools and async cross-device MemOps** | device pools only; eager equal-shape MemOps → `BuildCtx.pool(…, device="host", mapped=True)`, requested at the latest in S0's `before_materialize` and allocated once, outside the device profile cycle, with `cudaHostAlloc` or `mmap` + `cudaHostRegister`, exact bytes against EngineConfig `host_memory`, a device alias; a MemOp may cross host ↔ device and carry `after` (an event of a step) and `lands_by` (a later step whose launch waits on it); the core runs it on a copy stream | a toy unit demotes a unit to host and back, bitwise; a late copy stalls the `lands_by` step, never skips it | 200 |
| S3 | **Fill pools** | fixed plugin pools → `BuildCtx.pool(…, fill=True, priority=p)`, filled after EngineConfig `paged_min_tokens`, capped at `pref_units`; a lock pins sizes through the owning plugin's own option (D4), so the core records nothing new | a toy fill pool and paged state split a budget | 40 |
| S4 | **Weight extents** | `QTensor` parts carry tensors only → an optional per-part `extent(path, offset, nbytes)` for unmodified file slices (absent when a kernel's `prepare` repacked them) | the toy weight source reports extents that read back bitwise | 20 |
| S5 | **CPU facts** | `Fingerprint` has none → `Fingerprint.cpu_isa` (a tier = a prebuilt compiler flag set such as `x86-avx2`, `x86-avx512`, never `-march=native` or a CPU model) and `Constraint.cpu_isa`. A lock that runs CPU kernels carries the tier inside its class's first segment (`rtx-5080+x86-avx2/cu130-torch2.11`, keeping the two-segment layout lock discovery globs); for a host of tier t the lookup order is `gpu+t/…`, then `gpu/…`, then another tier's lock unverified; the **store** forces F5a there (tune records the split's tier in a store option, and the store zeroes `plan[n]` when it differs from `ctx.fingerprint.cpu_isa`) | a toy CPU kernel selected by tier; class ranking with and without a tier | 40 |
| — | **CLI size check** | refuses checkpoints larger than VRAM → refuses only when it is larger than VRAM **and** EngineConfig `host_memory` is 0 (the default, so today's behaviour stands); with `host_memory` > 0 the deployment opted in and the build's per-pool report judges fit (generic, no store named, invariant 1; GLM's 93 GB file on a 32 GB card with ≈ 58 GB of host budget must not be refused) | — | 10 |

≈ 480 lines. Not core: the store, `TieredStore`, tiers, residency maps, policies, calibration, route traces and
profiles, the ring (design §6.5). No slot is re-added; `quiesce`, layer-wise loading and per-layer ready events stay
planned (D13, milestone K).

### 4.7 Core needs of the model milestones (listed now, built there)

- **FN, GLM: hyper-connection residuals.** `Model.trunk` and `tap` pass `[T, H]` hidden states; hyper-connections
  carry `[T, n, H]` streams (n = 4) through every layer. FN's doc designs the shape change (a residual-width fact on
  the trunk); it is FN's core seam.
- **FN: the PLE n-gram table** (28.8 GB, a few rows per token) reads through S2's host pools and the SSD tier's I/O
  workers; no further seam expected.
- **FN, GLM: sparse attention (QSA, DSA)** uses `page_group` and the planner path that exist; KV streaming (Strata
  C4) belongs to milestone K.
- **FN's shared expert** (Qwen3.6-35B-A3B, Flash-Next) composes beside `std.moe@1` at block level, not inside it, so X
  and FN do not both edit `std.moe`.
- **FN / GLM and step 3c's prepared q8_1** (panel C, 2026-10-18): the router writes the routed gate/up input's q8_1
  into `qlinear`'s shared per-width buffer; a shared expert's gate/up of the same width running in between makes the
  handle stale and silently loses the 1.4–1.6% saving. Run the shared expert after the routed gate/up, or give the
  router's handle its own buffer; check it with an L0 or nsys test on FN's layer.

## 5. std library and kernel contracts

**`std.tiered`** (std): `TieredStore` over tiers, `ResidencyMap` (device buffer + conditional edits + delta log), the
`PlacementPolicy` and `Tier` protocols, `DeviceResolver`; profile and trace tooling (`route_dump` from the
`after_commit` counts; `holon_tools profile` ranks (layer, expert) pairs; profiles ship in model bundles).

Kernel contracts (owned by `std.moe`, except the policy- and transform-owned variants of D3; each with `reference.py`
and a CPU provider, in `testing` for std's):

| Contract | Signature (sketch) | Notes |
|---|---|---|
| `moe_resolve@1` | `(ids [T,k], weights, map, layer, plan table, state…, row kinds) → slot_ids [T,k], fetch list, CPU jobs, counts, delta` | deterministic; misses → the rank's null slot; a negative id is dropped (no fetch, no job, no count); padding rows masked; variants with other semantics are contracts their policy plugins own (D3), each with its `lend(n)` op for device-owned policies |
| `moe_route_transform@1` | `(ids, weights, map) → ids, weights` | signature only; each transform's contract is its plugin's (D3); `approx` only, bound by explicit option |
| `expert_fetch@1` | `(fetch list, host aliases, staging slots)` | in-graph zero-copy gather (dynamic-address DMA is not graph-safe); a DMA variant with a host-planned resolve if M1 shows the gather < 0.9× DMA |
| `expert_prefetch@1` | `start(next layer's non-resident list, ring slots)`, `wait(layer)` | F7's in-step copy stream and its events stay inside the kernel plugin (D8) |
| `linear_experts@2` | unchanged; arenas use `linear_experts_arena@1` (below) | — |
| `linear_experts_arena@1` | @2 over a class arena `[S, …]`: ids are arena slots, at most E + top_k distinct per call; facts @2's for the layer (E) plus `slots`; an `errors` word; @2's oracle. Compacted prefill groups are padded with null-slot rows so the kernel's own row count picks the reference chunk's path (D4) | `kernels.ggml_cuda` (its @2 kernel, with the forced-tiling and compact-active-slot patches) and `testing.ref_kernels` register it |
| `cpu_experts@1` (CPU) | `(jobs grouped by expert, activations, host blocks, mapped f32 rows)` | bitwise per token across group sizes and thread counts; ISA constraint |
| `moe_patch_rows@1` | `(y [T,k,H], CPU rows, job map)` | before `moe_combine@1` |
| `cpu_task@1` | `signal(slot)`, `wait(slot)` graph ops + worker service | `kernels.cpu_task` (D6); fail-open: errors answer DONE and raise a flag |

Streams and events never appear in std (invariant 6): the store's between-step copies are S2 MemOps run by the core;
in-step gathers and F7's copy stream are kernel-plugin ops.

## 6. Numerics and judge

| Configuration | Class | Check |
|---|---|---|
| offload store, any placement, any policy, F5a only (the default) | exact | snapshot byte-identical to `std.moe@1` all-resident, with adaptive placement on; L0 placement-invariance (D4) |
| F5b split on | bounded | L2, with placement frozen (a static map, so the floor knows the CPU set), against a **path-aware floor**: the Tier B floor emulates q8_1 for GPU experts and Q8_K for exactly the experts the frozen split sends to the CPU (a `Blocking(256, f32)` mode in the judge's q8_1 emulation plus the split rule); KL ratio predicted on the CPU before the GPU run (Q8_K's error is 1.3–2.7× q8_1's, panel C10); run-identical ×3 with adaptive placement on |
| A1 tail skip | approx | human gate: L2 KL, top-1, task check; never in a default lock |

Floors and references: the file fits a 24 GB card, so the bitwise checks run on one card — the A5000 locally and the
RTX 4090 for the class check — comparing `std.moe@1` all-resident against the offload store at full, natural-5080 and
capped fits, and with adaptive placement on. On the RTX 5080 itself the snapshots at natural fit, capped fit and
with adaptive placement are byte-identical to each other, and run-identical ×3. One extra snapshot runs the
vehicle with the model-agnostic n-gram drafter, so verify windows (D12) and the c+2 lag under verify steps are
exercised in X (panel B-r2-8). The A5000
runs the bf16 floor as today. FN and GLM take their engine floors the same way (no
RTX PRO 6000 engine runs needed).

Plants (`judge/holon_judge/planted.py`), predicted verdicts fixed now; each reports how often it fired. The
exactness snapshots run twice — with F11 (≥ 64 in-graph admissions and ≥ 2 device lendings at the capped fit) and
with `profile_decay` (≥ 8 swap rounds, where the two-phase plants bite). Because every placement is exact, a policy
bug shows only as lost speed, so an L1 **policy-parity check** compares F11's per-step residency, read from the
delta log, with an oracle extended from the M6 replay's `lfu` to D4's whole spec (f32, the device's tie-breaks and
step order, row kinds, the one-slot-per-expert rule in mixed steps, whole-expert device lending and its return,
per-role maps) — exactly — on a captured decode-only trace and on one with prompts and mixed steps:

| Plant | Breaks | Predicted |
|---|---|---|
| `resolve_off_by_one` | a miss maps to the neighbouring slot | snapshot differs; L2 fails |
| `map_before_landing` | the map step skips the wait, the copy drops its `after` ordering, and an unseeded 0–2 ms delay is put on the copy stream before it, so reads at the map step race the copy | fails run-identical (race) or the snapshot identity against all-resident (stale bytes); which one is recorded |
| `map_stale_slot` | a deterministic map to a slot already reassigned | snapshot differs; L2 fails |
| `cpu_wait_skipped` | on 1% of layers the graph reads the CPU rows without waiting (F5b) | fails run-identical (a race) or L2 (stale rows read deterministically); which one is recorded |
| `owner_race` | a host lending edit targets a slot of F11's device-owned cache (step 3) | the precondition flag fires and the engine stops; with the check removed, run-identical fails |
| `lfu_victim_needed` | F11's victim mask ignores the step's needs, so a row reads a slot overwritten in the same step | snapshot against all-resident differs; L2 fails |
| `lfu_shared_victim` | two misses of one layer are gathered into one victim slot | snapshot against all-resident differs; L2 fails |
| `cpu_gate_up_order` | the CPU reads the merged gate / up halves swapped (F5b) | L2 fails; F5a configurations clean |

Each plant changes only the configurations that use its path. Batch-invariance of `cpu_experts` is L0's own
mutation test, not a plant.

## 7. Memory fit (estimates; step 0 measures)

Budget = (VRAM free after weights and the CUDA context) × `memory_fraction` − the worst step's peak. 0.9 leaves
≈ 1.4 GB idle on the 5080; offload locks set the fraction from tune's measured peak (Maya keeps a 700 MiB reserve).

| Model, file, context | Card | Non-expert weights | Expert bytes | Expert cache | Host tiers |
|---|---|---|---|---|---|
| Qwen3-30B-A3B Q4_K_M, 32K | RTX 5080 16 GB | ≈ 1.2 GB | ≈ 17.6 GB (6,144 × ≈ 2.86 MB) | ≈ 8.2 GB ≈ 47% incl. ≈ 0.28 GB of null and staging slots (capped run ≈ 25%) | RAM ≈ 18 GB |
| Qwen3.8-Flash-Next IQ2_XS, 32K, MTP on | RTX 5080 16 GB | ≈ 3.7 GB + the MTP layer | ≈ 36 GB (24,576 × ≈ 1.46 MB) | ≈ 3.8–7 GB ≈ 11–20% (MTP-layer experts through the store or resident) | RAM ≈ 36 GB; PLE 28.8 GB on SSD |
| Qwen3.8-Flash-Next IQ3_S, 32K | RTX 5090 32 GB | same | ≈ 50 GB | ≈ 18.7 GB ≈ 37% | RAM ≈ 50 GB |
| GLM-5.3-Flash Maya-S, 32K, MTP off | RTX 5090 32 GB | ≈ 6.9 GB | ≈ 86 GB (12,096 × ≈ 7.1 MB) | ≈ 18 GB ≈ 21% | RAM exclusive + SSD on a 64 GB host |

The KV floor grows with context (30B-A3B: 96 KiB per token in bf16, ≈ 3 GB at 32K), so each benchmark lock names its context. GLM
routes ≈ 2.4 GB of experts per token on Maya-S; above ≈ 85% hits its decode is GPU-bound (≈ 4.7 ms of GPU reads
against ≈ 3.4 ms of misses at 90%), below it miss-bound. Host RAM per desktop class is checked before scheduling
(not recorded here).

As built (step 3b, §14): staging is min(E, k × a graphed step's tokens) per class — k × `max_seqs` × (1 + drafts) —,
since prefill steps borrow their staging and ring slots from the cache (D4's device lending) and give them back; a
layout whose cache cannot lend a prefill pass's (min(E, k × 512) per class) keeps that much staging instead. The
vehicle at the gate (`max_seqs` 1, no drafts) keeps 8 staging slots per class instead of 128 (≈ 0.47 GB more cache
on the 5080). The startup profile runs a streaming step's 4,096-token expert passes; on the A5000 vehicle its peak is
0.48 GB either way (512- or 4,096-token passes: the trunk's other activations set it).

## 8. Step 0 — measure before freezing

On the RTX 5080 and 5090 desktops (the bar cards), one heavy job at a time under `MemoryMax`, results in `evidence/`.

| # | Measure | Decides |
|---|---|---|
| M1 | H2D: pinned copies, `cudaMemcpyBatchAsync` of expert-sized blobs, the zero-copy SM gather (8–128 CTAs); the gather's best rate as a ratio to DMA; per-layer prefill expert compute vs chunk size (64–4096 tokens) on the bar cards | the fetch path's rate — gather ≥ 0.9× DMA, else the DMA `expert_fetch` variant lands before step 4; ring depth; D8's rule |
| M2 | Contention: CPU expert GEMV concurrent with the gather; per-expert cost per path under contention | `plan[n]` (D4); whether F5b can gain ≥ 5% at all |
| M3 | CPU expert throughput: ggml-cpu `vec_dot` on Q4_K / IQ2_XXS / IQ2_S / IQ2_XS / IQ3_XXS / IQ3_S experts, 1–8 tokens, AVX2 vs AVX-512, 1–N threads; Strata's kernels beside them | F5b's worth; borrowing multi-token kernels; worker counts |
| M5 | Handshake with a real payload: k ≤ 10 jobs per layer, rows 1–8, device- vs host-planned; a prototype of host-issued copy-engine fetches while a graph kernel spins (Strata's DMA mode hung a driver lock, #31) | D5's plan location; `cpu_task@1`; whether the DMA `expert_fetch` variant is viable |
| M6 | Routing traces of the development vehicle (A5000) on the agent and mixed traces; offline replay of static, profile_decay (0.7 / 4 / 96 / 1.5), lru, promote-on-fetch, LFU, Belady at 10–60% cached, reporting **bytes per token per path** (hit, fetch, admission, CPU, SSD) converted to time with M1–M3's rates; profile transfer between traces; distinct experts per step and window; next-layer recall | the default policy and its knobs; D8's rule (with M1's compute); D11 |
| M7 | NVMe `O_DIRECT` reads of expert-sized ranges at queue depths 1–32 with 1–8 I/O workers | F9's reader |

Step 0 closes with a roofline (M1–M3 rates, M6's bytes per path) predicting every row of acceptance 3, written
before step 4 measures them. A natural-fit row the roofline predicts to miss its bar moves after step 6 (judged with
tune's lock), or goes back to the owner before step 4 if F5b cannot plausibly close it.

Decision rules (fixed now): the default policy is the one with the least modelled time per token on M6's held-out
trace, confirmed by L4 on the development gate; F5b enters a lock only at ≥ 5% decode gain (D4); F7 by D8's time
rule; F8 only at ≥ 15% on < 1K-token chunks with a fixed share; A1 never in a default lock.

## 9. Steps (milestone X)

| Step | Work | Checks | Days |
|---|---|---|---|
| 0 | M1–M3, M5–M7; host RAM per class | evidence; the panel reviews the parameters they set | 3 |
| 1 | S0–S5 and the CLI change with toy proofs (ABI 1.2); the forced-tiling and compact-helper vendor patches | core tests; `check_abi --base` additive; LOC; L0 placement-invariance | 5 |
| 2 | `std.tiered`, the offload store and options, class arenas with null and staging slots, `moe_resolve` / `expert_fetch` (CUDA + oracle + CPU provider), the pinned tier, `static` (F1), F5a | all-resident bitwise to `std.moe@1`; L0 | 5 |
| 3 | `lfu` (F11) and `lru` (F4), each with its own resolve contract, `profile_decay` (F2–F3) with two-phase edits, route tap and profile tool, prefill streaming (F7), the route-transform call site, scheduler row kinds | snapshots with adaptive placement on, ×3, and the n-gram drafter run; cross-placement identity; plants 1–3 and `owner_race` | 5 |
| 4 | Development gate, natural-fit rows, on the RTX 5080 vs FreeToken and llama.cpp (F5a lock); tune | acceptance 3 natural-fit rows vs the roofline; L4 | 3 |
| 5 | The replay's exclusive-tier comparison (D10), then F9 (SSD tier with I/O workers, exclusive RAM tier) and the RAM-short default | exactness; a RAM-capped run | 3 |
| 6 | `kernels.cpu_task`, `kernels.ggml_cpu`, F5b with `plan[n]`, the path-aware floor; capped-fit rows and the F1 row with tune's lock | L0 incl. bitwise per token and per thread count; bounded L2; plants 4–5; tune's ≥ 5% rule decides the dev lock; acceptance 3 capped-fit and F1 rows; **tag the std offload API** | 5 |
| 7 | Held-out plugins after the tag: F10, A1, the mmap / page-cache tier (acceptance 1) | plugin-only diffs; A1 judged as `approx` | 3 |
| 8 | Close: full suite, panel code review, one propagation of the existing locks; D13 if time allows | acceptance 5–7 | 2 |

**Order as run (2026-10-18):** step 3c (MoE glue kernels, §13) runs before step 4; step 6 runs before step 5, because an
in-graph SSD miss is a `cpu_task@1` copy task, so F9 needs `kernels.cpu_task` first; 3c and 6 → 5 run in parallel
branches. F5b comes after the development gate on purpose: F5a-only must already be competitive, and tune then shows what the
CPU adds.

## 10. Risks and deferred

- **F5a-only is miss-bound at low fits, Gen5 included.** On the vehicle at 25% cached, F5a passes the
  `--moe-cache-mib` row only if the gather reaches ≈ 40 GB/s (panel C-r2-1's roofline: 30 GB/s gives 0.78×, below the
  kill line). Hence M1's gather-vs-DMA rule, the roofline before step 4, and the capped rows judged after F5b.
- **The model milestones are the critical path to the program bars.** X closes without them.
- **Strata and Maya move fast;** each comparison pins the reference commit, and bars are re-checked against the
  newest release only at FN's and GLM's close.
- **Exactness costs speed against Strata's defaults** (tail skip +10–24%); the like-for-like bars and A1 opt-in make
  the trade visible instead of hidden.
- **Host memory:** S2 accounts every pinned byte; development runs are capped with `MemoryMax`.
- **Deferred:** D13 (VMM); predictive prefetch (D11); a GPU-matching CPU kernel (D7); peer GPUs and layer split;
  dense expert offload; EXL3.

## 11. Milestone K outline (KV and checkpoint tiers), with the requirements rev 1's review found

K1 checkpoints and K2 KV pages demoted to pinned RAM instead of dropped (Strata C1–C2, llama.cpp `--cache-ram`,
vLLM's native offload, SGLang HiCache); K3 an SSD tier surviving restarts; K4 KV streaming for sparse-attention
mixers (with FN / GLM); the end-of-answer checkpoint (plan §10). Its own doc must answer:
- **Eviction invariants** (panel A10): demoted children must not keep a VRAM parent unevictable while `could_fit`
  counts it as freeable (preemption would then fail liveness); VRAM entries stay prefix-closed and the host tier
  evicts leaf-first; a property test that `could_fit` and `evict` agree.
- **Cancellation must be write-safe:** a load lands in host staging and reaches device units only through a MemOp of
  the resuming step, or released units wait on the load's event.
- **Deterministic resume** (panel C9b): a `BLOCKED` request rejoins at a fixed step offset, waiting if its copy has
  not landed, so batch composition never follows timing.
- **Acceptance that bites** (panel C14): RAM-hit TTFT ≤ VRAM-hit TTFT + 1.5 × bytes / measured rate; concurrent
  sessions' TPOT p99 during a load ≤ 1.2×; multi-session throughput against Maya's measured ratio on the same trace.
- Its core seam (prefix-index tiers: `Payload.tier`, `HitLen.cost`, demote, cancellable `BLOCKED(future)`, an
  `on_finish` publish) is designed there; the name stays `tier`, not `kv_tier` (it also stores checkpoints).

## 13. Step 0 results and the decisions they set

**M6 — routing traces and policy replay** (`spikes/x0_offload/m6/README.md`;
`evidence/x0_m6_routing_qwen3-30b-a3b_rtx-a5000_2026-10-17.json`). Qwen3-30B-A3B Q4_K_M on one RTX A5000, eager;
244,291 routed rows (61,979 decode) over four traces (agent, mixed, chat, code); decode-only replay with the same
profile warm start for every policy, scored on held-out traces, time = (dense + hit + fetch) / 900 GB/s +
(fetch + admission) / PCIe at placeholder rates 25 and 50 GB/s (M1 replaces them).

| Held-out agent, cached | static | profile_decay | lru | promote_on_fetch | lfu | belady |
|---|---|---|---|---|---|---|
| 10%: hit % / ms per token (PCIe 25) | 12.7 / 38.4 | 36.9 / 29.5 | 43.4 / 25.7 | 33.4 / 29.9 | 43.0 / 25.8 | 60.9 / 18.3 |
| 25% | 32.5 / 30.0 | 63.0 / 18.8 | 70.2 / 14.3 | 64.9 / 16.5 | 70.6 / 14.1 | 84.4 / 8.3 |
| 47% | 55.5 / 20.3 | 85.4 / 8.8 | 89.6 / 6.1 | 88.0 / 6.8 | 90.3 / 5.8 | 95.6 / 3.5 |

Decisions (by §8's rules, fixed before measuring):
- **Default policy: `lfu` (F11).** Least modelled time in 26 of 30 held-out cells (`lru` in 4, at 10% cached, by
  ≤ 0.6%). `profile_decay` loses on admission bytes, not counts: each swap copies the expert a second time (391 + 36
  MB per token against `lfu`'s 311 at 25%); removing D4's lag recovers only ≈ 6%. Knobs moved held-out results by
  ≤ 4% and changed no ranking; decay 0.915 per step (0.9 equal, 0.99 17% worse). L4 on the development gate checks
  it against `lru` in ABBA rounds (≥ 4 per arm): `lfu` stays the default unless `lru` beats it by more than 2 sd
  (the margin, ≤ 7%, is near L4's resolution, panel C). Its win assumes F5a — admission is free only because every
  miss reaches the GPU — so where tune considers F5b, it chooses policy and split together, re-running the replay
  model with M2 / M3's CPU rates. Consequence for D4 / D5: under F11 and F4 the whole cache is a device-owned region — a miss is gathered
  straight into its victim slot in the graph, so there is no second copy and no host edit; the host seeds the cache
  (`after_materialize`) and reads the delta log. Prefill rows never count or admit (they stage or stream, D8);
  ring lending is done by the device (D4). D4's two-phase host edits remain for `static` re-seeds and `profile_decay`.
- **Profiles transfer poorly** (routing is flat: the top 10% of experts take 23–36% of routes): the profile is a warm
  start only (F2); no policy depends on it for steady state.
- **D11 stays: no prefetch in X.** Next-layer recall is 84–87% at top-8 and 94.5–96% at top-12, so the recall half
  of the trigger is borderline; but 40–46% of prefetched non-resident experts would go unused, and an idealized
  overlap model loses everywhere except at 50 GB/s with 47% cached (+9%). Re-open after M1 measures idle PCIe time.
- **D8 inputs:** a 64-token chunk touches ≈ 51% of a layer's experts, 1024 tokens ≈ 84%, 4096 tokens 94–97%. On the
  A5000 routed-only staging wins below ≈ 256 tokens and streaming from ≈ 1K; the crossover is re-derived with M1's
  compute on the bar cards.
- **Verify windows share experts:** consecutive tokens share ≈ 3.5 of 8 experts per layer (2 / 3 / 4-token windows
  touch 12.5 / 16.2 / 19.3); rows of different sequences in one step share almost none — D12's grouping pays within
  a sequence's window, not across a batch.

**M1, M2, M3, M5, M7 and prefill compute on the RTX 5080 and 5090 desktops** (`spikes/x0_offload/bench/README.md`;
`evidence/x0_bench_rtx-5080_2026-10-17.json`, `evidence/x0_bench_rtx-5090_2026-10-17.json`). Both cards behave alike.

| Measure | Result | Decision |
|---|---|---|
| M1 fetch | pinned copies 46.6–48.4 GB/s; `cudaMemcpyBatchAsync` 46.7–48.4; the in-graph SM gather 46.6–48.3 (47.5–48.3 at 2.8 MB), **0.97–1.02× DMA** at N ≥ 8, saturating with 2–8 CTAs of 256 threads; inside a CUDA graph 46–48 | `expert_fetch@1` is the in-graph gather (8 CTAs × 256 threads, unroll 8); the DMA variant and the host-planned resolve are not needed |
| M3 CPU experts (ggml-cpu, µs per expert) | Q4_K_M experts 128–148 at 1 thread, 59–69 at 4 threads (≈ 45 GB/s, memory-bound from 2–4 threads); i-quants 254–417 at 1 thread, 39–70 at 7 threads (compute-bound); AVX-512 and AVX2 builds equally fast; **0 bitwise mismatches** (alone vs groups of 2–8, 1 vs 7 threads, one job vs a batch) and **AVX-512 vs AVX2 bitwise equal for all eight classes** | D7's contract holds for ggml-cpu as is; D6: compute threads from tune — 2–4 for Q4_K decode, all non-loop cores for i-quants and multi-token jobs; a work-stealing pool (a static split let one slowed core stall every batch 1.6×); S5's tier stays in F5b lock classes, but a lock is split by tier only where the tiers' outputs differ |
| M2 contention (6 threads) | CPU expert 63–65 µs alone, 112–114 under the gather (1.8×), 148 under DMA (2.3×); the gather 47.5 → 27.5 GB/s, DMA 46.6 → 34.1 GB/s beside the CPU. CPU and PCIe share host memory: together ≈ 55 GB/s, so the split adds ≈ 15% miss throughput, not a second link | `plan[n]` ≈ n / 2 for Q4_K_M (5080: 1 1 2 2 3 3 4 5 5 6; 5090: 1 1 2 2 3 3 4 4 5 6 for n = 1…10; flat within ±1), ≈ 0.4 n for IQ2_XS; F5b's modelled decode gain ≈ 2% at 1 miss per layer, 5–6% at 2, 8% at 4, 10% at 6 — it clears D4's 5% bar only from ≈ 2 misses per layer (hit ≤ ≈ 75%): likely at the capped fit, marginal at the natural fit. With F5b on, between-step copies use the gather, never DMA beside the CPU lane |
| M5 handshake (4 threads, 48 layers) | 2.1–2.9 µs per layer at rows 1–2, ≤ 3.8 at rows 4, 2.6–6.5 at rows 8 (the CPU rows' read-back); device- vs host-planned within 0.4 µs; ≥ 96.6% of CPU time hidden behind equal GPU work; all in-graph CPU results bitwise equal to ggml-cpu | D5's device-planned resolve stands; `cpu_task@1` as designed |
| M7 local NVMe (`O_DIRECT`) | 1.7–2.4 GB/s at queue depth 1 (0.74–0.80 ms per 1.46 MB, 1.3 ms per 2.8 MB); 2.5–2.7 GB/s from depth 2 or two `pread` workers | F9: 2 I/O workers at depth 2; a disk miss ≈ 20× a RAM miss on these drives; re-measured on a Gen4 / Gen5 SSD before GLM's RAM-short runs. The bar desktops hold every expert of the vehicle, Flash-Next and GLM's Maya-S in RAM, so F9 matters for smaller hosts |
| Prefill: one layer's expert GEMMs (mmq) vs streaming its non-resident experts | 5080: 0.78 / 1.27 / 2.72 / 10.97 ms at 64 / 256 / 1024 / 4096 tokens; 5090: 0.42 / 0.71 / 1.50 / 6.00 ms; streaming 4.3 ms at 47% resident, 6.1 ms at 25% | D8, with M6's real coverage (the bench's uniform routing touched 127–128 of 128 experts from 64 tokens; real routing touches 51% at 64, 68–71% at 256, 83–85% at 1024 — panel C-r4.5-1): on the 5080 routed-only staging wins below ≈ 512 tokens (e.g. 3.5 vs 5.8 ms per layer at 64 tokens, 25% cached) and streaming above (7.4 vs 5.8 ms at 1024); the crossover is re-derived per class from these numbers. Streaming hides behind compute only from ≈ 2K-token chunks on the 5080 (3–4K on the 5090), so offload locks take larger prefill chunks for long prompts (tune picks; Strata reads 8K) |

**Roofline for acceptance 3** (written before step 4 measures; RTX 5080, Qwen3-30B-A3B, **batch 1**, 32K), as bands:
all-resident GPU work ≈ 4.5 ms per token (estimated from existing locks); Holon's own offload-path overhead (resolve,
gather launch, F11's pick and decay, telemetry) 0.25–0.5 ms, measured at step 2 as F5a fully resident against
`std.moe@1`; fetch at 47.5 GB/s; CPU experts ≈ 45 GB/s alone; M6's batch-1 bytes (chat and code, `lfu`: ≈ 50–70 MB
per token at 47%, ≈ 198–227 MB at 25%; `lru` ≈ 10–30% more); llama.cpp's cache with 10–50 µs of host sync per
layer; FreeToken's gather bounded at M1's rate (the same kind of kernel; panel C-r4.5-2…4):

| Row | Natural fit (≈ 47%) | Capped (≈ 25%) |
|---|---|---|
| Holon F5a, ms per token | ≈ 5.8–6.5 | ≈ 9.1–9.7 (F5b ≈ 3–5% at 1.4–1.7 misses per layer; not counted) |
| vs llama.cpp static split (bar 1.5×) | ≈ 2.7–3× | ≈ 2.4× |
| vs llama.cpp `--moe-cache-mib` (bar 1.1×) | ≈ 0.92–1.47×, central ≈ 1.18× (after step 2: measured overhead 0.20 ms + F11 ≈ 0.25–0.35 ms; Holon's all-resident 30B runs 0.93× llama.cpp on sm_120, `docs/release.md`) — the band's width is llama.cpp's unknown per-layer sync, measured before step 4 | ≈ 1.05–1.4× (straddles) |
| vs FreeToken (bar 1.0×) | ≈ 1.0–1.1× (thin) | ≈ 1.0–1.1× (thin) |
| F1 (static) vs llama.cpp static split (bar 0.95×) | ≈ 0.95–1.0× with F5a | — |
| Prefill vs llama.cpp (bar 1.0×) | staging below ≈ 512 tokens, streaming above with larger chunks; ≥ 1.0× predicted | same |

The static-split rows pass with margin; the cache rows (`--moe-cache-mib`, FreeToken) straddle or touch their bars —
like-for-like cache engines on the same PCIe link differ mainly by overhead and policy. By §8's rule: a natural-fit
row measured below its bar goes **to the owner**, not to step 6 (F5b adds ≈ 1.5–2% at 0.7 misses per layer and
cannot close it); the capped rows and the F1 row are judged after step 6 with tune's lock, as §1 says. M1's gather
rate is re-measured at step 2 on the store's own allocation (`mmap` + `cudaHostRegister`, with and without huge
pages), since M1 used `cudaHostAlloc` (panel C-r4.5-5).

**Reference engines on the RTX 5080** (`spikes/x4_refs/README.md`, `evidence/x4_refs_rtx-5080_2026-10-17.json`;
llama.cpp f74e2fa with `--moe-cache-mib`, FreeToken 21d9578; 3 interleaved rounds). Decode tok/s at 1K / 4K / 32K,
natural fit (≈ 47%) | capped (≈ 25%):

| Arm | Natural | Capped |
|---|---|---|
| llama.cpp `--fit` default | 78.2 / 76.3 / 61.5 | 46.6 / 45.7 / 39.9 |
| llama.cpp tuned `--n-cpu-moe` | 88.4 / 83.1 / 67.2 (66% resident) | 47.4 / 45.5 / 40.6 |
| llama.cpp `--moe-cache-mib`, default (mmap) | 159.1 / 109.2 / 81.7 | 54.6 / 46.9 / 42.5 |
| llama.cpp `--moe-cache-mib`, pinned (`--load-mode none`) | 188.8 / 147.2 / 102.0 | 100.3 / 87.0 / 70.5 |
| FreeToken (`auto` = `offload`) | 213.8 / 175.3 / 111.4 | 115.6 / 109.5 / 76.6 |

llama.cpp's host sync is ≈ 19.5 µs per MoE layer (0.94 ms of a 5.4 ms step); its GPU work is 3.93 ms per token
(the roofline assumed 4.5); its cache hits 95.2% at 47%. FreeToken prefills 2.7K / 10.7K / 5.4K tok/s (it streams
every expert once per chunk of up to 8K). **Re-banded:** the natural-fit row against the pinned cache is
1.02–1.11× / 1.00–1.07× / 0.98–1.02× at 1K / 4K / 32K (bar 1.1×; 1.09–1.17× even if Holon's kernels matched
llama.cpp's), against FreeToken 0.84–1.04× (bar 1.0×), against llama.cpp's default cache 1.22–1.53×; the static-split
rows clear 1.5× at 1K / 4K and straddle at 32K. By §8's rule the cache rows go to the owner before step 4.

**Where the gap is** (`spikes/x4_gap/README.md`, `evidence/x4_gap_rtx-5090_2026-10-17.json`,
`evidence/x4_gap_rtx-5080_2026-10-17.json`): it is placement policy and prefill design, not kernels.
- All-resident decode on the RTX 5090: Holon 389 / 362 tok/s at 1K / 4K against llama-server 340 / 294 and FreeToken
  341 / 322 — §13's 0.93× came from llama-bench, which has no server loop; Holon's remaining 0.41 ms per token
  against FreeToken is glue, router GEMV, the stacked gate/up GEMV and q8_1 quantization. Offload overhead fully
  cached: Holon +0.22 ms, FreeToken +0.53–0.60 ms.
- Offload decode at 47% on the RTX 5080: Holon's step-2 static policy misses 592 MB per token against FreeToken's
  52 MB (gather rates ≈ 41 vs 43 GB/s); replayed `lfu` on Holon's own routing predicts 0.97–0.99× FreeToken and
  1.12–1.15× llama.cpp's pinned cache (1.03–1.05× / 1.19–1.22× at FreeToken's miss bytes). The last ≈ 0.4 ms is
  FreeToken admitting the prompt's experts during prefill — step 3b re-checks D4's "prefill never admits" with
  M6's replay before changing it.
- Prefill at 4K on the 5080: FreeToken 10.7K tok/s (all experts streamed once, overlapped); Holon as built 2.0K
  (1.38 s of 2.0 s refetching per 512-token pass); measured probes 4.9K with 4,096-token expert passes and 6.6K with
  2,048-row mmq calls; modelled D8 with overlap ≈ 10.0K (2,048-row mmq) to ≈ 11.9K (4,096-row, needs the helper's
  `ID_PAIRS` above 16,384). Step 3b builds D8 with large passes, copy-engine overlap and ≥ 2,048-row mmq calls.
So the cache rows' prediction moves back toward the bars, with the FreeToken decode row the tightest; the owner
decides the bars with step 4's measurements.

**Re-banded after step 3 (panel C, 2026-10-18)**, from x4-gap's `lfu` replay plus F11's measured cost with candidates
(≈ 0.08 ms per token at 47%, ≈ 0.11 ms at 25%; the +0.04 ms was the all-cached figure): vs FreeToken ≈ 0.97× / 0.95× at
1K / 4K natural (bar 1.0×: miss) and ≈ 1.05× / 0.94× capped; vs llama.cpp's pinned cache ≈ 1.10× / 1.13× natural
(bar 1.1×: on the bar at 1K); the default-cache and static-split rows clear. Levers, by size: x4-gap's kernel rows
(glue, router GEMV, q8_1 quantization; ≈ 0.25 ms, +4–5% on every row) — taken as **step 3c** before the gate, since
the FreeToken row needs it; then prompt-time admission (+1.5–2% on whole traces, +9% / +6% on single long prompts at
47%), **deferred**: built only if step 4's FreeToken decode row lands within ≈ 2% of its bar, and then as its own
policy plugin (`placement.lfu_prompt@1`, which needs a std "adopt a lent slot" record before the tag), not as a
change to D4 or F11 (panel A's four determinism conditions apply). **32K** (`spikes/x4_gap/README_32k.md`, measured
Holon bases on the 5080 plus the replayed `lfu` bytes): vs FreeToken 0.99× natural [0.97–1.01] / 1.02× capped
[0.97–1.06]; vs the pinned cache 1.08× [1.06–1.11] / 1.10× [1.06–1.15]; the default-cache and static-split rows pass
(≥ 1.35× / ≥ 1.64×). With step 3c's ≈ 0.25 ms the 32K cache rows reach ≈ 1.02–1.04× FreeToken and ≈ 1.11–1.12× the
pinned cache: on or just over their bars, with no margin.

**Contradictions with the research note, recorded:** CPU compute is not a second link beside PCIe (they share host
memory); ggml-cpu's AVX2 and AVX-512 builds agree bitwise; "≤ 4 threads saturate DRAM" (S6) holds only for Q4_K
single-token jobs; these desktops' NVMe reads ≈ 2.6 GB/s, not 7–14. The vendored ggml-cpu for `kernels.ggml_cpu`
builds from the vendored llama.cpp's pin (51ce9c1) unless step 6 aligns it with `kernels.ggml_cuda`'s.

## 12. Review log

### Rev 1 → rev 2 (panel: A correctness, B architecture, C numerics / performance)

| Finding | Resolution |
|---|---|
| A1 [blocker] victim still mapped while its slot is overwritten | D4 two-phase edits (unmap, copy after the last read, map after the copy's event), fixed lag c+2 |
| A2 shared null slot breaks ggml's id helper on mmq | D5: one null slot per routing rank at the top of each class arena; L0 duplicate-id fuzz |
| A3 per-layer views vs the contiguous expert stride; staging per layer | D5: class arenas global across layers; staging reused layer after layer |
| A4, B1 [blocker] no per-engine shared object | S0 `BuildCtx.shared` |
| A5, B8 host pools vs weight loading and materialize; no seeding point | S2 allocates host pools before weights load; S0 `after_materialize` |
| A6 torch pinned allocator rounds to a power of two | S2 / D10 exact `cudaHostAlloc` or `mmap` + register |
| A7, C9a sizes re-measured at start change placement | D4 / D9 / S3: the lock pins VRAM and RAM-tier slot counts |
| A8, B7 CPU ISA not in the lock class | S5 `cpu_isa` tier for CPU-kernel deployments; D7 |
| A9 padding rows routed | D4 / `moe_resolve` row mask |
| A10, C9b prefix-tier invariants, cancellation, resume timing | moved to milestone K with these requirements (§11) |
| A11 device-written maps vs host flips | D4 one writer per region, conditional edits, delta log |
| A12, B2 native code in core | D6: `kernels.cpu_task`; core keeps lifecycle (S0) |
| A13, C1 mmq stream-k breaks placement invariance | D4 forced-tiling patch + L0 across slot counts, permutations, groupings |
| A14 telemetry overwritten by the next launch | S1 per-launch double buffer |
| A15 lag differs between paths | D4 fixed c+2 on every path |
| A16 per-request split share | D4 deployment-level table only |
| A17, C16 worker counts; I/O stalls compute | D6 worker counts from tune; separate I/O workers |
| A18 snapshot freezes adaptive policies; plant definitions | D4 refines §7.10 (placement-invariant path); plants redefined (§6) |
| A19 `torch.cuda` banned in std | S2 MemOps run the copies; §5 note |
| A20 nits (mmvq limit, `prepare_device` is a planner hook, CLI size refusal, memory_fraction, null units) | §3, §4 (CLI row), §7 budget line; null slots are the store's own |
| B3 re-added slots unused by core | D3: open options holding Refs; no slot re-added |
| B4 hidden core work around S5 (eviction, end-of-answer publish) | moved to milestone K (§11) |
| B5 contracts not open to F4 / F9 / F10 / A1 / new tiers | D3 protocols, `resolve_variant`, device state, tier codes, `moe_route_transform@1`, `Tier` protocol |
| B6 acceptance 1 not a real proof | D16 / acceptance 1: held-out plugins after the API tag |
| B9 file extents are core ABI | S4 |
| B10 fill shape underspecified; KV naming | S3 priority + cap, `paged_min_tokens`, config additions listed |
| B11 `kv_tier` naming; S2 / S5 overlap | milestone K, named `tier` |
| B12 scope too large | four milestones (D2), model core needs listed (§4.7) |
| B13 closed `moe.store` choice | D3 open Ref |
| B14 drift list, line budget | §4 line estimates; drifts below |
| B15 `quiesce` without a user | dropped until D13 |
| C2 D7 stretch mis-scoped | dropped (D7); F5a default gives exactness |
| C3 exact default via F5a | D4 |
| C4 split formula unsound at small miss counts | D4 `plan[n]` table |
| C5 default policy by hit rate | §8 by modelled time from bytes per path, F10 included, L4-confirmed |
| C6 coverage rule streams almost always | D8 time rule |
| C7 memory fit off (MTP layer, GLM trunk, KV floor) | §7 rewritten; contexts named |
| C8 inclusive RAM tier | D10 exclusive when RAM is short |
| C10 path-blind floor | §6 path-aware floor |
| C11 plant predictions; missing S1 bug class | §6 plants redefined, firing counts, ≥ 8 swap rounds |
| C12 bar consistency; llama.cpp arms | acceptance 3–4: like-for-like rows, per-arm llama.cpp bars, F1 ≥ 0.95× |
| C13 high-hit dev gate; "exact switches" label | capped ≈ 25% run; "no approximation" |
| C14 acceptance 4 trivially met | moved to milestone K with sharper bars (§11) |
| C15 split rule units; GLM GPU-bound claim | D9 experts take all above the floor; §7 corrected |
| C17 resource waste | 4090 dropped from step 0; M7 trimmed; no RTX PRO 6000 engine floors; L2 rows merged; batch invariance in L0 |

### Rev 2 → rev 3

| Finding | Resolution |
|---|---|
| A-r2-1 [major] conditional host edits let a copy overwrite a mapped slot | D4: unconditional edits in host-owned regions; ownership moves only by a device release in the delta log; a failed precondition asserts; plant `owner_race` |
| A-r2-2 [major] class arenas multiply the mmq grid and the id-helper scan by S/E | D4: a compact-active-slot vendor patch (launch over min(S, T·k)); invariance unchanged |
| A-r2-3 the startup profile runs the store before seeding | D5: `after_materialize` after every materialize, resets and re-seeds; arena minimum holds null and staging slots |
| A-r2-4, C-r2-6 staging size | D5: min(E, k × max rows) per class, eager steps included, an L1 invariant; §7 counts it |
| A-r2-5 F7's in-step second stream | D8, §5: `expert_prefetch@1` in the kernel plugin |
| A-r2-6 exclusive tier's victim window | D10: SSD code during the window, demotion before the incoming H2D, RAM slot reuse after the map lands |
| A-r2-7, B-r2-4 CPU tier and lock classes | S5: tier inside the first class segment, no cross-tier borrowing, unverified fallback to F5a; tiers are prebuilt flag sets |
| A-r2-8, C-r2-3 `map_before_landing` cannot fire | §6: the plant maps in the unmap step (d = 0); `cpu_wait_skipped` predicts run-identical or L2 |
| A-r2-9 per-launch device buffers vs captured addresses | D4: one device count buffer zeroed in-graph; S1's pinned double buffer |
| A-r2-10 plan drift | plan §10 updated (four milestones, F5a default, KV in milestone K) |
| A-r2-11, B-r2-6 shared objects under aux contexts; MTP sizing | S0 root context and key convention; D5 sizes from the weight source's config |
| B-r2-1 [major] F10's oracle needs std / testing edits; A1 cannot express a drop | D3: policy-owned variant contracts with their own oracles; negative id = dropped in `moe_resolve@1`; option value is the selection fact |
| B-r2-2 [major] the CLI row names the store in core | CLI row: VRAM + `host_memory`, generic |
| B-r2-3 the unnamed held-out strategy is too easy | acceptance 1: a held-out tier (mmap / page cache); graph-shape-changing miss paths named as a std minor |
| B-r2-5 how the lock pins units | D4 / S3: the store's own option, fixed pools |
| B-r2-7 no abort hook | D6: `cpu_task@1` fail-open by contract |
| B-r2-8 the vehicle never drafts | §6: an n-gram-drafter snapshot run in step 3 |
| B-r2 side notes | S2 ≈ 200 lines; FN's shared expert composes beside `std.moe` (§4.7) |
| C-r2-1 [major] the dev gate may fail at the capped fit before F5b | acceptance 3: roofline prediction; natural rows at step 4, capped and F1 rows after step 6; M1 gather-vs-DMA rule and a DMA variant; §10 |
| C-r2-2 [major] L0 wording; missing launch condition; no check on the 5080 | D4: path from the step's T; L0 covers compacted launches and tile widths on every class; cross-placement identity on the 5080 |
| C-r2-4 NVFP4 / fp4 side tensors | non-goal of X (GGUF integer-quant classes) |
| C-r2-5 floor cannot know the CPU set under adaptive placement | §6: F5b's L2 runs with placement frozen |
| C-r2-7 contexts | acceptance 3 and D14: ≤ 32K on the vehicle |
| C-r2-8 prefill compute measured on the wrong card | §8: in M1 on the bar cards |

### Rev 3 → rev 4

| Finding | Resolution |
|---|---|
| A-r3-1 [major] "path from the step's T" differs from `std.moe@1`'s 512-token chunks; `linear_experts@2` cannot be told the path | D4: the path is the reference's (the chunk's row count); decode / verify never compacted; compacted prefill groups padded with null-slot rows (rev 4.1, B-r4-1) |
| A-r3-2, B-r3-2 [major] arenas cannot be sized from the config; std would learn architecture keys | S0 `before_materialize`; D5: classes from adopted layouts and extents; host pools filled from extents |
| A-r3-3 the id helper still scans S × T·k | D4: helper rewritten to an O(T·k) compact list |
| A-r3-4, C-r3-1 [major] `map_before_landing` still cannot fail | §6: drops `after`, adds an unseeded delay on the copy stream; prediction covers race or stale bytes |
| A-r3-5, B-r3-7 fail-open has no mechanism for a dead worker; assertion mechanism | D6: time-bounded spin, watchdog for memops; D4: device error flag checked at commit, later steps discarded |
| A-r3-6 host → device direction | D4: in the same apply as the unmap, no copy |
| A-r3-7 S5 borrowing contradiction | S5: lookup order `gpu+t`, `gpu`, other tier unverified; the store forces F5a |
| A-r3-8, C-r3-2 staging undercounted, sized for prefill | D5: decode / verify sizing, ≈ 0.28 GB; prefill borrows ring slots; §7 |
| B-r3-1 [major] A1's transform needs std / testing edits | D3 / §5: transforms are plugin-owned contracts; std owns the signature and the drop |
| B-r3-3 the CLI check refuses GLM | CLI row: refuse only when larger than VRAM and `host_memory` = 0 |
| B-r3-4 `resolve_variant` described two ways | D3: it is the contract name, None = `moe_resolve@1`; §5 header |
| B-r3-5 `Tier` protocol unspecified | D3: members sketched; non-device-readable reads go through a generic `cpu_task@1` copy task, tiers stay pure Python |
| B-r3-6 `linear_experts@2` "unchanged"; `expert_prefetch@1` returns an event | §5: `path_rows` minor; `start` / `wait(layer)` ops |
| B-r3-7 line totals; S5 fallback owner | §4 ≈ 480 lines; the store owns the fallback |
| C-r3-3 which mmvq kernel runs on a compacted launch | D4: decode / verify launches are never compacted |
| C-r3-4 DMA variant mechanism; roofline-miss rule | M5 prototypes host-issued DMA beside a spinning graph; §8 rule: the row moves after step 6 or goes to the owner |
| C-r3-5 `owner_race` before F10 exists | §6: planted on F4's device-owned region at step 3; prediction corrected |
| B-r4-1 (rev 4.1) `path_rows` changes a signature providers implement (a major) | D4 / §5: no new argument; compacted prefill groups are padded with null-slot rows past the mmvq limit |
| A-r4-1 staging sized by rows, not tokens; mixed steps | D5: sized by graphed tokens incl. verify widths; steps with a prefill chunk take the ring path |
| A-r4-2 later waits after a timeout | D6: every wait returns at once once the error flag is set |
| A-r4-3 the gate / up merge reads and moves offloaded stacks | D5 / S4: extent pairs, no concatenation or move under the offload store |
| A-r4-4 short prefill chunks take mmvq in the reference | D4: chunks at or below the mmvq limit are never compacted |
| A-r4.2-1 the mmvq limit is per type | D4: the compaction rule and padding target are per class; decode / verify / mixed never compacted |
| A-r4.3-1 uncompacted ring-path steps need all of a layer's misses at once | D5: lend ≥ min(E, k × tokens) per class for such steps; L1 invariant extended |
| B-r4.2-1 S4's sequence form has no producer | S4 keeps one extent per part; gate / up pairs stay inside the store (its `merge` member: the resident store concatenates, the offload store records pairs) |

### Rev 4.3 → rev 4.4 (the panel on M6's decisions)

| Finding | Resolution |
|---|---|
| A-r4.3d-1, B-r4.3-1, C-r4.3-1 [major] ring lending is a host edit in a device-owned cache | D4: under device-owned policies the device lends (coldest n per class at the step's launch, returned empty); two-phase lending stays for host-edit policies; `owner_race` planted on lending |
| C-r4.3-2 [major] no plant or check reaches the default path | §6: snapshots with F11 (admissions, lendings) and `profile_decay`; L1 policy-parity check against the M6 replay; plants `lfu_victim_needed`, `lfu_shared_victim` |
| A-r4.3d-2 F11's victim rule unspecified | D4: per-layer pairing as M6 replayed it, ties by expert id, in the contract and oracle |
| A-r4.3d-3 counts for verify, mixed and decay steps | D4: row kinds; verify counts at resolve time; prefill never counts; decay on steps with decode / verify rows; one slot per expert per step |
| A-r4.3d-4 F11 with the exclusive RAM tier | D10: the in-graph victim becomes SSD-only; RAM slot released via the delta log |
| B-r4.3-2 the row mask only marks padding | D4 / §5: a row-kind code in `moe_resolve@1` |
| B-r4.3-3 D3 / D4 text not updated for F11 | D3 instances, D4 regions and two-phase scope updated |
| C-r4.5-1 [major] D8's always-stream rests on uniform routing | §13: M6's coverage; staging below ≈ 512 tokens on the 5080, streaming above |
| C-r4.5-2 [major] roofline lacks Holon's overhead; the tight row's fallback cannot work | §13: overhead term measured at step 2; bands; a natural-fit miss goes to the owner |
| C-r4.5-3 agent-trace bytes for a batch-1 gate | §13: batch-1 bytes; capped rows pass without F5b |
| C-r4.5-4 FreeToken at 31 GB/s | §13: bounded at M1's rate; the row is thin |
| C-r4.5-5 M1 measured another allocation | §13 / step 2: the gather re-measured on `mmap` + `cudaHostRegister`, ± huge pages |
| A-r4.4-1 lent slots are never refilled | D4: a returned slot is its layer's first victim, filled unconditionally; L1 cache-size test |
| A-r4.4-2 per-class lending leaves experts half-resident | D4: lending picks whole experts until every class has n slots |
| A-r4.4-3 parity cannot be exact as written | D4: step order decay → increments → pairing; parity on decode-only traces and on a replay extended with lending |
| C-r4.4-1 [major] F11 churns victims onto SSD under an exclusive RAM tier | D10: F11 default only where RAM holds every expert; replay comparison of three variants before step 5 |
| C-r4.4-2 per-class lending halves experts | as A-r4.4-2: whole-expert lending |
| C-r4.4-3 the parity oracle covers decode only | §6: oracle extended to D4's whole spec; parity on a trace with prompts and mixed steps |
| B-r4.4-1 the lending pick is a policy rule | D4: a `lend(n)` op in the policy-owned resolve contract; the apply kernel only unmaps and logs |
| C-r4.3-3 F11's win assumes F5a | §13: tune chooses policy and split together for F5b candidates; ABBA rule for lfu vs lru |

**Drifts from `docs/design.md`, recorded:** §7.4 (the CPU handshake lives in a kernel plugin, not
`holon.core.platform`); §6.5 and §4.2 (placement policies and tiers are std protocols bound by open options, not
re-added slots; Movers are S2 MemOps); §6.1 (the fill step and pinned split in the budget order); §6.4 and D8
(per-layer ready events and layer-wise loading stay planned; prefix tiers move to milestone K); §7.10 (adaptive
placement may run in snapshots when the consumer is placement-invariant); §9's expert-cache row (no VMM in the first
wave). Design is updated with X's code, as M0.7 did.

## 14. Step 2 as built (2026-10-17)

`std.tiered` (TieredStore, ResidencyMap, the PlacementPolicy and Tier protocols, DeviceResolver, the pinned RAM tier
`std.tiered.pinned@1`, the static policy `std.tiered.static@1`, the `O_DIRECT` extent reader), `std.moe.offload@1`
bound by `moe.store` (absent: `std.moe.resident@1`, today's path), the contracts `moe_resolve@1` and
`expert_fetch@1` (oracles in `std.moe.reference`, CPU providers in `testing.ref_kernels`, CUDA in the new
`kernels.expert_io`), F1 and F5a. Evidence: `spikes/x2_store/README.md`, `evidence/x2_store_rtx-a5000_2026-10-17.json`
— on the A5000 every fit (all, fill, 47%, 25%) is snapshot-identical to `std.moe@1` all-resident, run-identical ×3 at
25%, with the unpatched ggml kernels (stream-k never engages at the vehicle's arena sizes, ≥ 136 slots, on any card:
for the vehicle the D4 patches cut the arenas' mmq launch cost, which grows with the slots, so no unpatched prefill
timing feeds the roofline; a small-arena GPU test does take stream-k and waits for them); the offload path costs
+0.20 ms per decode token with every expert cached; with the D4 patches merged the 100%, 47% and 25% fits stay
identical (`evidence/x2_store_patched_rtx-a5000_2026-10-17.json`), and the full-fit trace runs as fast as
all-resident (4.1 s against 4.0 s; 11.8 s unpatched); `mmap` + `cudaHostRegister` gathers at `cudaHostAlloc`'s rate (0.94–1.01×,
noise between processes), huge pages or not.

Decided while building, against the text above:
- **Staging is min(E, k × a pass's tokens), a pass being `std.moe@1`'s 512-token chunk** — E for any real config
  (128 per class on the vehicle, ≈ 0.54 GB with the null slots, not D5's ≈ 0.28 GB): prefill uses routed-only
  staging until step 3's ring borrowing; then staging shrinks to D5's decode-sized count.
- **One VRAM pool holds every class's arena.** The policy seeds whole experts within its bytes (and each class's
  cap), then each class's arena — null, staging, the seeded cache slots — is laid out in it; per-class fill pools
  grow one after another and starved the later classes. `moe.cache` (`<class>=<slots or fraction>`, `*` for
  the rest) therefore **caps** a class's cache slots and pins the pool's size; the seed decides each arena's (with
  `static`, deterministically). Kernels over an arena are bound through `linear_experts_arena@1` (§5), not
  `linear_experts@2`, whose input domain (a stack of E experts) stays as it is: a contract of its own, with the
  layer's facts (`experts` = E) plus the class's largest arena (`slots`), ids that are arena slots and at most
  E + top_k of them distinct per call (the patched mmq's compact launch bound), and a declared `errors` word; resident
  MoE keeps @2, so resident locks' kernel sites do not move.
- **Errors reach the commit.** `moe_resolve@1` takes each role's arena size and flags a resident slot outside the
  cache slots (RANGE; the entry reads a null slot), and the store watches the kernels' error words (kernels.ggml_cuda's
  id-error word); the step's first resolve zeroes them, a commit that finds one set stops the engine.
- **Row kinds come from the step's plan in `before_launch`:** a mixed step's leading rows and rows whose every query
  token has logits are decode / verify, the rest prefill, tokens past them graph padding — so a one-token last
  prompt chunk counts as decode outside mixed steps (D4 says prefill never counts; step 3 settles it with F11's
  parity oracle). The step's first resolve zeroes its counts and applies its map edits in the forward; host edits
  reach the device as their own entries (a scatter), never as the host's copy of the map, so device-owned regions
  keep what the device wrote. `misses` is a running maximum since the last materialize.
- **Not yet:** the delta log (its producer is step 3's device-owned resolvers), edits after the seed (two-phase
  edits land with `profile_decay`; the store refuses them), an auxiliary model's experts through the store (an MTP
  layer needs its sub-steps' row kinds; refused at build), `lfu` as the default (step 3; `static` until then).
- The `Tier` protocol's members take the class name (a tier holds pieces of every class); the profile
  (`moe.placement.profile`) is a JSON file `{"layers": {"<layer>": [count per expert]}}` named relative to the
  checkpoint's directory, so a plan records no host path. §5 says profiles ship in model bundles: until
  `holon_tools profile` (step 3) writes them there, a profile lies beside the checkpoint; a bundle-shipped profile
  then needs the store to resolve the model plugin's bundle, decided with the tool. The store's options share the `moe.` prefix: `moe.store`,
  `moe.placement`, `moe.placement.profile`, `moe.tiers`, `moe.cache`.

### Step 3a as built (2026-10-17): the placement policies

Built: `placement.lfu@1` (F11, the offload store's default: the only tier, pinned RAM, holds every expert),
`placement.lru@1` (F4) and `placement.profile_decay@1` (F2–F3), each its own plugin under
`plugins/holon_plugins/placement/`; the device-owned two own their resolve contracts `moe_resolve.lfu@1` /
`moe_resolve.lru@1` (oracle `reference.py` and CPU provider in the plugin, CUDA provider in `kernels.expert_io`), each
with its `lend(n)`; F2's warm start for every policy (`std.tiered.static@1` included); std completions in `std.tiered`
(policies built with `items` / `roles`; per-group device state `[groups, *shape]`, zeroed at each seed and passed to
the policy's variant by name; the delta log in the telemetry; host-edit preconditions; two-phase swaps as S2 MemOps;
route taps; `TieredStore.lend`); `holon_judge.placement` (`route_dump`, the L1 policy-parity check); six plants;
`holon_tools profile`. Evidence: `spikes/x3_policies/README.md`, `evidence/x3_policies_rtx-a5000_2026-10-17.json` — on
the A5000 F11 (2,115 in-graph admissions per run at 25%), F2–F3 (16 swap rounds) and F4 snapshots are byte-identical
to `std.moe@1` all-resident and run-identical ×3 (F11, F2–F3), the n-gram-drafter run too; natural, capped and
adaptive placements are identical to each other; F11's per-step residency equals its replay oracle exactly on every
trace recorded (up to 23,163 admissions over 255 decode steps), and the device's map ends equal to the host mirror;
every plant fires and is caught (table in the README). All at `52e1bed`, after the review fixes below.

Decided while building, against the text above:
- **Scheduler row kinds (B-s2-6) not built: it is an ABI major** (built additively in step 3b: `BatchPlan.row_kind`). Filling `BatchPlan.n_decode` on every step changes
  the field's meaning ("a mixed step's leading decode rows; 0: unmixed", `core/holon/api/batch.py:127`), which
  readers rely on: `std.mixers.gqa`'s planner splits any step with `n_decode` > 0 into a decode and a prefill call
  (`std/holon_std/mixers/gqa/units.py:51-55`), `std.mixers.gdn` plans such a step per row kind and never as a decode or
  verify step (`std/holon_std/mixers/gdn/units.py:74-77`), and the core runs it eagerly (`core/holon/core/executor.py:320`,
  counted as mixed at `core/holon/core/engine.py:455`) — every decode step would lose its graph. Under §4.5 that is a
  change, not an addition. The additive way is a new `BatchPlan` field with a default (e.g. the leading decode and
  verify rows of every step, or a per-row kind array), filled by the scheduler — a minor, for the owner to decide.
  Until then the store narrows step 2's heuristic (a fix, review finding 1): a row whose every query token has
  logits counts as decode only in a step all of whose rows do (or among a mixed step's leading rows), so a one-token
  last prompt chunk beside other prompt rows is prefill — before, it counted, and one past the step's first 512-token
  expert pass made F11 decay twice in a step and pick victims a pass could not see (`test_only_decode_steps_count_and_
  decay` fails without the fix). What remains of step 2's deviation: a step made only of one-token last chunks counts
  as decode (its rows, at most `max_seqs`, lie in one pass). The parity check records what the device counted and each
  step's routed experts from its eager resolves.
- **The fetch list names each miss's slot.** An admitted miss is gathered straight into its victim's slots (D4), so
  `moe_resolve@1` returns `fetch [R, 2, staging]` (host unit, then the slot) and `expert_fetch@1` takes those pairs
  (its `first` argument is gone); `moe_resolve@1` also takes the delta log (`delta`, `n_delta`) and the layer's index,
  which @1 leaves alone and the variants append to. Both contracts changed in place: they are not on main, and the std
  offload API is tagged at step 6.
- **The delta log** is `[groups × items, 3]` int32 records, appended in a deterministic order (layer by layer, pairing
  order within one), with an `OVERFLOW` bit past its end: `(layer, a, b)` — a took b's slots in every role, b went to
  the RAM tier (−1) —, `(layer, −1, b)` — b lent. The host mirror is brought to each commit c (host edits launched by
  c, then step c's records), the same on the overlapped and synchronous paths, and the policy reads it.
- **Host edits carry a precondition** (the value the entry must hold at the apply); one that fails is not applied and
  sets the store's `edit_errors` word, so the commit stops the engine (D4's device error flag; never a trap).
- **Two-phase swaps.** A host policy returns swaps (as many items in as out per layer); the store unmaps the outgoing
  item at launch c + 2, hands in a MemOp copying the incoming item's host units into its slots with `after` = c + 2
  and `lands_by` = c + 2 + d (d = 2), and maps it at c + 2 + d. MemOps copy whole pool units of one shape, so the
  arena pool and the host pools share one unit, the classes' greatest common divisor (36,864 bytes on the vehicle;
  arenas aligned to its least common multiple with 4,096). `profile_decay` counts steps with decode rows (as the
  replay counts decode steps) and skips a round while its last swaps are not in the host's view.
- **Decay and counts once per step**: a variant decays (or ticks) in each call that holds a routed entry of a
  `DECODE` row, so under a policy with device state the store requires a step's decode and verify rows to lie in one
  expert pass (`max_seqs` × (1 + `max_draft_tokens`) ≤ `std.moe@1`'s 512-token chunk; refused otherwise), and the row
  kinds above keep them in the first. F11's arithmetic is exact f32 (× 0.915 rounded, then one rounded
  + 1.0 per entry, no FMA); its victims are experts resident in every role routed by no live entry of the call;
  candidates missing in every role, routed by a decode or verify row.
- **`lend(n)`** returns the picks as a `[layers, experts]` mask; the CUDA provider is eager torch ops (ring steps are
  never graphed), the CPU provider the oracle; `TieredStore.lend` unmaps and logs them and must run in the step's
  forward after its first resolve (which zeroes the log and the error words) — the ring that calls it, and D4's
  return of a lent slot as its layer's first victim (until then a lent slot is not refilled), are step 3b's.
- **The parity check also checks the device's map against the host mirror** once every launched step has committed
  (a kernel that changes the map without logging it would otherwise pass).
- **Cost** (corrected after the panel, C: the first figure was compared with §13's band as if that band were F11's
  alone; it is the whole offload path's): a variant's resolve adds 2.5 / 2.3 / 1.3 µs per layer call with candidates
  to pair at 1 / 8 / 64 decode rows on the A5000, ≈ 0.7 µs when a call has none (it exits before ranking; the
  ranking runs over compacted lists of packed keys, a victim ranked past the candidates stops counting); end to end
  with every expert cached, 5.663 ms per token under `static` against 5.703 under F11 (+0.04; `std.moe@1` 5.343).
  That +0.04 ms is the all-cached figure, where no call has candidates (corrected after the step-3 panel, C): at
  batch 1 and 1K context 29.5% of decode layer calls have candidates at 47% cached and 62.3% at 25% (measured), so
  F11 adds ≈ 0.06 ms per token at the natural fit and ≈ 0.09 ms at 25% (panel C's estimate: 0.08 / 0.11).
- **Plants:** `lfu-victim-needed` and `lfu-shared-victim` set plant bits the variant kernel reads (`MoeResolve.plant`,
  zero otherwise; judge-only). A needed victim's rows then read staging slot k (the victim was no miss), not its
  overwritten slot, as §6 describes. Their verdict (panel C, recorded): on an mmq step the patched mmq's id-error word
  (a slot repeated in a row) stops the engine before a wrong byte is emitted — the snapshot trace, step 3; on
  decode-only steps (mmvq, which checks no ids) the snapshot differs and the parity check fails — the decode trace:
  `lfu-victim-needed` differs at step 11 (parity at step 6; 959 needed victims taken), `lfu-shared-victim` at step 4
  (parity at step 4, and the device's map no longer the mirror; 3,623 shared victims). The other weak prediction:
  `owner-race` with its
  check removed is deterministic — the snapshot differs (and the parity check with it) in every run, run-identical
  holds. `map-before-landing` failed both checks (all three runs differ from all-resident; run 0 differs from runs
  1–2). L2 was not run for the plants (§6 predicts it fails too); the exact checks above are their verdicts.
- **Profiles** are written beside the checkpoint by `holon_tools profile` (`--name`, default `profile.json`) — for a
  hub-cache checkpoint that is the snapshot directory; shipping them in model bundles stays open (§5).
- **What the parity check leaves to step 3b:** traces with prompts in mixed steps, device lending and its return
  (cache size unchanged), and the one-slot-per-expert rule across a step's row kinds; `route_dump` already takes a
  step's routed experts from its eager resolves for them.


### Step 3b as built (2026-10-17): D8's prefill ring, device lending and its return, row kinds, the route transforms' call site

Built (commits from `7e4a49b`; the panel's 3a findings in their own commits, listed in the README):
- **Row kinds in the core** (`7e4a49b`, ABI 1.3, additive): `BatchPlan.row_kind`, a per-row `RowKind` (prefill,
  decode, verify; None: not known), filled by the scheduler on every step it plans (mixed steps and drafter
  alternation included; core/tests/test_sched.py); `n_decode` keeps its meaning. The store reads it, so a one-token last
  prompt chunk is a prefill row in every step (§14 step 2's remaining deviation is gone; failing-without test).
- **D8** (`std.tiered.ring.Lent`): a step of at least `moe.ring_tokens` prompt tokens (default 512, §13's 5080
  crossover) borrows, at its first resolve, every empty slot and then the policy's `lend(n)` (whole experts, the
  coldest) until a double buffer holds two layers' misses per class; each layer's missing pieces stream into their
  half on a copy engine (`expert_prefetch@1`, kernels.expert_io: DMA on its own stream after the work enqueued, a wait
  before the layer's resolve), layer l + 1's while layer l computes, and its resolve reads them in place (the
  resolve's `ring` table: no fetch); the step's expert passes are one per chunk of up to 4,096 tokens (the router and
  top-k still per 512-token chunk, the step's last partial chunk a pass of its own), and kernels.ggml_cuda's arena
  binding takes 2,048 rows per mmq call (`ID_PAIRS` ÷ top-8; each row bitwise a 512-row call's, L0; 4,096 since the step-3 panel's fixes, below). A smaller prompt
  step borrows its misses' staging the same way when its pass could exceed the staging slots. At the step's end every
  borrowed slot returns to its layer as an empty slot — the layer's first victim, which the policy's next admissions
  fill unconditionally (`moe_resolve@1`'s `empty` stack; records `(layer, a, −2 − j)`): the cache never shrinks (CPU
  test, per class).
- **Staging sized for graphed steps** (§7), the per-step miss check against that step's staging width.
- **The route transforms' call site**: `moe.route_transform = <name>` binds a plugin's
  `moe_route_transform.<name>@1` between the top-k and the resolve, with the map the resolve reads (after the step's
  lending, ring pieces present); offload store only (a resident store refuses it at build); a negative id is dropped
  end to end (null slot, no fetch, no count, zero contribution). A toy fixture (`tests.drop_last`) proves it; no
  transform plugin ships (A1 is held out to step 7).
- **Parity extended**: the route dump records each step's victims' scope (the resolve marks what the call holding
  the decode rows routed), its lending (`lend(n)` calls) and the layers' empty slots; `placement.lfu`'s `replay` models
  borrowing, lending, the return and empty slots as first victims, and carried scores; the check also compares the
  device's map with the host mirror at the end.

Evidence (A5000, `spikes/x3_policies/README.md` "Step 3b", `evidence/x3_policies_rtx-a5000_2026-10-17.json`
`step_3b`): every placement snapshot-identical to all-resident with borrowing on — F11 at 25% ×3, 47%, the fill, 10%;
the n-gram drafter; the decode trace and mixed steps (`chunk_with_decode`), 3 ring and 35 staging borrowings —, F2–F3
×3 run-identical; F11 equal to `replay` at every step of every run (up to 23,300 admissions) with the device's map
the host mirror. Prefill at 25% cached: 554 → 3,340 tok/s at 4K (6.0×), 542 → 896 at 1K; at 47% 4,049 tok/s at 4K
against 3,595 all-resident (whose passes stay at 512 tokens). The plant `lent-slot-still-mapped` fails parity at step
1 and the snapshot at step 6 (decode trace) or stops the engine by the id-error word (snapshot trace); `owner-race`
under borrowing is stopped by its precondition flag.

Decided while building, against the text above:
- **The ring returns empty (R0), as D4 says** — not the replay's marginally better R1. The M6 replay of three returns
  (`spikes/x3_policies/replay5/`: R0 empty, R1 refilled from the layer's streamed experts by decode score, R2 by the
  pass's routing) puts R1 ≤ 0.3% ahead on whole traces and 1.0–1.4% on single long prompts; keeping R1's experts needs
  either paid copies at the return (which the replay shows erase the gain) or a ring layout where a layer's keepers are
  never reused by later layers, which the double buffer does not have.
- **Prompt-time admission is not built** (deferred since, §13: its own policy plugin if step 4 needs it; the coordinator's item 5, folded into the panel's item 4): the same replay
  finds that counting prefill rows at a routing-proportional weight (`12 × the expert's prefill entries ÷ the step's
  prefill rows`) and letting their misses compete as candidates — admitted only where they land free (staged or
  streamed into the victim's slot) — beats D4's "prefill never admits" by 2.1% / 0.7% (agent / mixed, whole traces) and
  9% / 6% on single long prompts at 47% cached, never worse in any held-out cell; counting +1 per entry (variant B)
  loses 4–7%. That changes D4: it goes to the panel before it is built.
- **Lending reads the map back** (superseded by the step-3 panel's fixes, below: lending is the device's): DMA needs host addresses, so a borrowing step synchronizes at its first resolve (the
  map, the empty stacks, then each `lend` call's picks: two or three syncs per borrowing step, none after); empty slots
  are borrowed first (the replay's assumption too), so consecutive prompt steps do not drain the cache.
- **Ring sizing** (superseded below): two halves of the largest per-class miss count over the layers (lending makes misses, so the
  sizing iterates, at most four `lend` calls); one half if fewer slots are lent, routed staging if not even that.
- **Only device-owned policies borrow** (F11, F4); host policies (`static`, `profile_decay`) keep a pass's staging and
  never stream (D4's two-phase lending for host policies is not built). So does a layout whose cache cannot lend a
  pass's staging (decided at each materialize). A step's pass size is not the layout's: it is 4,096 tokens whenever
  the policy is device-owned, the step has at least `ring_tokens` tokens and a pass's misses fit the prefill staging
  (every expert of a layer), so the startup profile — on minimum pools, where a fill cache holds nothing to lend —
  measures the large pass the served steps run (review fix, failing-without test).
- **`moe.ring_tokens` stays at 512** (measured on the A5000 since, below), §13's crossover on the 5080: deriving it per class from M6's coverage and the
  class's compute needs the bar card's compute (step 4). On the A5000 streaming wins at 1K already (below).
- **4,096-row mmq calls are not built** (built since, below): the id helper sorts 16,384 pairs per call (1,024 threads × 16); 4,096 rows
  at top-8 need a larger instance in the vendored helper (a vendor-patch extension with its own L0). The 2,048-row
  calls are what §13's gap diagnosis modelled at ≈ 10.0K tok/s on the 5080 (≈ 11.9K with 4,096-row calls).
- **Victims are per call, the parity oracle's per step**: the resolve marks its victims' scope (`routed` |= 2 by the
  call holding the decode rows; every call ORs its own mark, so a later pass keeps it), and the route dump hands that
  to the oracle, so a mixed step spanning passes is modelled as the device runs it.
- **The CPU reference's expert GEMM rounds a row by its expert's batch**: dropping one route changes other rows' bits
  on the CPU (`linear_experts`' oracle multiplies per expert); the GPU kernels are row-invariant (D4's L0), so the
  transform's CPU test compares within rounding.
- **The delta log** is a region of 2 × items records per layer (panel A: admissions ≤ misses, lends ≤ residents,
  returns ≤ lends), ≈ 147 KB copied to the host per step on the vehicle.

### Step 3's panel fixes (2026-10-18)

Panel A, B and C's findings on step 3, in their own commits (README "Step 3's panel fixes"); the panel's decision on
prompt-time admission (deferred, §13) is recorded there.
- **Graph padding is `RowKind.PADDING`** (`6cebfbb`, additive, ABI 1.3's listing): `padded()` gave padding rows
  the bucket's decode or verify kind, so a unit reading the padded plan's kinds would have counted padding.
- **No row-kind guess** (`ce10352`): the store refuses a launched plan without `row_kind`; the startup profile's and
  capture's plans never reach `before_launch` (the executor calls it only in `launch`).
- **4,096-row mmq calls** (`d070b40`, vendor patch): mmq's id helper sorts up to 32,768 pairs per call — past one
  block's 16,384, a device-wide path (cub's device radix sort on the compact path's keys and bits, then its rows,
  bounds and checks in four more kernels, scratch from the call's workspace); the per-slot and compact paths and the
  locks are unchanged. L0: a 4,096-row call's rows are bitwise the 512-row calls', and the new path flags its three
  preconditions as the others. Only eager prefill passes reach it (`linear_experts_arena@1` takes a whole D8 pass).
- **Lending is the device's** (`7846044`, `4773e5f`, `b191782`): a borrowing step's lent slots, ring and stage tables
  are written by its own kernels, with no host sync (a GPU test runs `Lent` under sync-debug errors).
  - `lend(need per class)` (each variant's contract; CUDA: sorts and scans): whole experts in the policy's order,
    ties by expert id then layer — so lending spreads over the layers (8–9 experts per layer for a 1K prompt at 25%,
    420 in all) instead of emptying the first layers (576 before: 18 layers' every cached expert) —, for each class
    the first `need` of them holding its slots.
  - The ring's halves are sized after lending: `ROUNDS` = 4 rounds of twice the most misses of a layer, less what
    is held, the last asking for the worst case; a ring still short (a cache that cannot lend more) stages instead,
    the device's choice (`ring_ok`, the stage width in telemetry; no single-buffer ring any more: small caches stage
    where 3b streamed). Each class's lent slots are pooled (empty slots first); a layer's role takes its half's slots
    in expert order; a step's misses past the ring go to the graphed staging slots.
  - The copies are `expert_prefetch@1` from device tables: the fetch kernel (zero-copy reads of the mapped host
    pool) on its own stream, one small CTA per class — DMA needs host addresses, which only a sync gives; the SM copy
    costs the pass compute (A5000, 4K at 25%: 4,716 tok/s with one CTA, 3,968 with the staging fetch's eight, DMA
    5,127).
  - The return is the device's too, in the lend records' order (call, then expert id), which the host mirror
    replays at the commit (`ResidencyMap.commit(…, returned=True)`); a step is booked as borrowing when its forward
    borrows (a graphed step of one-token last chunks never does).
- **Both ring-copy paths, one option** (`1e3bd15`, the coordinator's decision on the trade-off above):
  `moe.ring_copy = sm | dma`. `sm` is the device lending above, its copies on a small kernel, no host sync; `dma`
  keeps the same device lending and tables (spread lending, the need per class, the staging fallback, the mirror's
  replay) and reads the copy list back once per streaming step — its one host sync (prefill steps are eager) — to copy
  on copy engines (`cudaMemcpyBatchAsync` from the pinned host pool; `expert_prefetch@1` takes the list on the device
  or on the host). Default `sm` until step 4 measures both on the 5080; the lock records the choice (a plugin
  option). `tune` today pins decode kernels from a decode benchmark only: choosing `ring_copy` per card class needs a
  prefill rate in it, step 4's. A5000, 25% / 47% cached (tok/s, median of 6): 1K 1,682 / 2,352 (sm) against
  1,748 / 2,365 (dma), 4K 4,731 / 4,819 against 5,304 / 5,493 (dma +12–14% where the pass is compute-bound); TPOT p99
  on the agent trace 127.4 / 125.3 ms (sm), 136.1 / 119.8 (dma), 121.8 / 120.9 (off) — within the rounds' spread.
  Both paths snapshot-identical to all-resident at 25% and F11 equal to `replay` at every step (dma: the snapshot
  trace and mixed steps; sm above); the GPU test runs `sm`'s lending under sync-debug errors and counts `dma`'s one
  explicit sync per streaming step.
- **F11's cost** (`4283bbf`): the +0.04 ms per token was all-cached; with candidates ≈ 0.06 ms at 47% and ≈ 0.09 ms
  at 25% (step 3a's bullet above).

Measured (an A5000 copying host to device at 26 GB/s; step 3b's prefill figures came from one copying at 12.4 GB/s —
that is why its 1K prefill sat at its own bound, 896 tok/s, not ≈ 2.5× below):
- **1K prefill, nsys** (25% cached): before, 600 ms per step, copies busy 573 ms (14.8 GB, 25.9 GB/s): PCIe-bound, the
  bytes 12% over the layers' misses because lending emptied whole layers; after (lending spread, one copy CTA per
  class), 618 ms, the copies busy 603 ms — still PCIe-bound, ≈ 83% of the bound the misses alone set (≈ 0.51 s; the
  SM copies run a little under the DMA's rate).
- **Stage against ring** (tok/s, one request of N tokens, median of 6; `prefill.py`):

| N | 512 | 1K | 2K | 4K |
|---|---|---|---|---|
| all-resident | 3,489 | 3,585 | 3,550 | 3,482 |
| 25%: stage / ring | 853 / 852 | 973 / 1,686 | 957 / 3,307 | 978 / 4,680 |
| 47%: stage / ring | 1,079 / 1,186 | 1,228 / 2,337 | 1,204 / 4,446 | 1,226 / 4,777 |

  D8's crossover on the A5000 is ≈ 512 tokens (equal at 25%, the ring 10% ahead at 47%): `moe.ring_tokens` stays at
  512, measured. Against the DMA ring with 2,048-row calls (3b's code on this card): 1K +5% / +34%, 2K +25% / +32%
  (lending spread), 4K −4% / −1% (4,096-row calls gained ≈ 5–7%, the SM copies cost ≈ 8–9%).
- **TPOT p99 on the agent trace** (`tpot.py`: mixed steps, `chunk_with_decode` 512, `max_seqs` 8, 47% cached; two
  rounds per arm): before (lending on the host, two or three syncs per borrowing step) 131.3 / 128.8 ms with
  borrowing on, 133.7 / 131.9 off; after 126.0 / 121.5 on, 132.0 / 124.0 off — borrowing never cost TPOT p99 here,
  and without its syncs it is ≈ 3% under the arm without borrowing (rounds vary by ≈ 5%). The per-request maximum
  gap's p99: before 669 / 735 ms on against 609 / 550 off; after 536 / 523 on against 629 / 631 off.

### Step 3c as built (2026-10-17): the MoE decode glue kernels

Taken before the gate because the FreeToken decode row needs it (§13, "Re-banded after step 3"). Commits `8e606ec`
(residual), `5473c4a` (router), `3015a3c` (q8_1), `7ea4b34` (the review's fixes; one reviewer, clean on re-review).
Evidence: `spikes/x3c_kernels/README.md`, `evidence/x3c_kernels_rtx-5090_2026-10-17.json`,
`evidence/x3c_kernels_rtx-a5000_2026-10-17.json`. No core change. Each change was kept by M0.7's rule (≥ max(0.5%,
3 sd) on some row) and classified as M0.7's D10 does.

| Change | Class | Saving per decode token, RTX 5090 resident 1K / 4K | Verdict |
|---|---|---|---|
| The residual add in the combine (`moe_combine@1`'s optional `add`; `std.moe`'s `add`, `std.blocks.prenorm` calling an ffn's `add`) | bitwise | 0.030 / 0.034 ms (1.16 / 1.21% ± 0.07 / 0.03) | kept (under 3 sd on the A5000) |
| `moe_router@1`: router GEMV from bf16 + softmax top-k in one launch, decode / verify calls ≤ 16 rows | bounded | 0.137 / 0.141 ms (5.3 / 5.0%) | kept, every row |
| The router's launch writes the gate/up input's q8_1 (`route_prepare`, `Prepared` handles of M0.7's D6) | bitwise | 0.039 / 0.038 ms (1.6 / 1.4%) | kept |

- **Totals, decode per token:**
  - RTX 5090 resident: −0.21 ms (2.62 → 2.42 ms at 1K). `holon_tg` 383 → 413 tok/s at 1K (×1.078) and 357 → 383 at
    4K (×1.074).
  - RTX 5090 offload store at 47%: −0.21 ms.
  - A5000: −0.22 ms resident, −0.19 / −0.21 ms at 47%.
  - Prefill is unchanged: prompt chunks above 16 rows compose the f32 linear and `moe_topk@1`, as before.
  - The ≈ 0.25 ms the re-band assumed comes out at 0.21 on the 5090. So §13's FreeToken decode rows move up by about
    4% (5080: ≈ 0.21 / 5.0 ms), less than the band's 4–5% upper end.
- **Judge:**
  - L0 on sm_86 and sm_120a.
  - The router's L2 (1-token steps against the mmvq floor, A5000) passes: top-1 98.57% (≥ 98.29%), KL p99 0.037
    (floor 0.040), ΔPPL vs floor +0.14% ± 0.11%.
  - Snapshots: the bitwise commits are identical to their predecessors, and the offload store is identical to
    all-resident at every commit (§1 acceptance 2 holds).

Decided while building:
- **The router's fused path stops at 16 rows.**
  - A single-kernel GEMV loses to cuBLAS's GEMM at prompt sizes without an SGEMM-style tiling. So a row's logits
    follow its call's path (as `qlinear`'s mmvq / mmq do), and the contract says so: deterministic, not row-invariant
    across paths.
  - Inside the fused path a row's logits are the same at any row count (one order per hidden).
  - The selection is `topk_moe_cuda`'s arithmetic with register-only arrays (the vendored kernel is untouched). It is
    bitwise `moe_topk` on the kernel's own logits, so the change is bounded only through the logits' f32 order.
- **The tile's last block selects:** a counter per tile, reset by that block, so graph replays need no memset. It
  reads the logits through L2. There are no spin waits, so co-scheduling with the store's side-stream copies cannot
  deadlock. The cost is ≈ 0.75 µs of fences and the atomic, plus ≈ 2.8 µs of selection on the A5000. The GEMV alone
  takes 2.8 µs, against the 8.9 µs of the cast + cuBLAS + top-k it replaces.
- **Where the q8_1 is written:** the prepared gate/up input is written by the router kernel, not by a separate
  quantize. It shares the `ggml_cuda.q8_1[in=…]` buffer and generation counter with `qlinear`'s `prepare_input`, so
  any later write into that buffer makes the handle stale, and the consumer quantizes from x.
- **Found on the way:** x4_gap's "stacked gate/up (+0.07 ms)" and part of its "q8_1 (+0.07 ms)" rows were measurement
  effects. Under PDL on sm_120 a kernel's measured duration includes its wait on the previous kernel: the 5090's
  expert GEMV row fell by 71 µs with no change to it once its quantize was gone. On the A5000, without PDL, the
  resident and arena GEMVs are within 2%.
- **Not built** (README "Not done"):
  - ≈ 30 per-step index and copy kernels in the core executor (≈ 40 µs per token on the 5090): a core change, so out
    of scope here and reported to the coordinator.
  - The offload store's per-step edit application (≈ 0.5%, under the rule).
  - PDL early-launch triggering in the router.
  - A faster selection tail.
- **Locks:** decode routing is no longer cuBLAS's bits, and MoE sites name `moe_router@1` (not `moe_topk@1`). The
  existing locks are propagated once, at X's close, with the router's L2 above as its bounded verdict.

### Step 3d as built (2026-10-17): the step loop's per-step index and copy kernels

A small core round, no ABI change (`check_abi --base main` adds nothing to the branch's 1.3), on step 3c's "not
built" item: the core issued 25 kernels, 2 device copies and 6 host-to-device copies per batch-1 decode step (plus the
tokens' copy back). Commits `b60328f` (prep), `5452833` (upload); `cdd4785` holds two std nits from 3c's panel.
Evidence: `spikes/x3d_core/README.md` (the per-step inventory before and after, by phase and issuer),
`evidence/x3d_core_rtx-5090_2026-10-17.json`, `evidence/x3d_core_rtx-a5000_2026-10-17.json`. Core 4,602 code lines.

| Change | Class | Saving per decode step, RTX 5090: 30B resident / 30B offload 47% / 8B, at 1K | Verdict |
|---|---|---|---|
| prep: without a drafter every start is the host's (§7.3), so positions, page lengths, write indices and the store's slots and lengths are staged; the device moves only its last tokens and writes the SeqTable; no logits gather when every token is a logits row | bitwise | 0.022 / 0.019 / 0.022 ms (0.91 / 0.68 / 0.58%, sd ≤ 0.06) | kept |
| upload: steps of at most a decode / verify step's tokens stage into an arena of that size, fields smallest first: one copy | bitwise | 0.013 / 0.021 / 0.013 ms (0.55 / 0.78 / 0.36%, sd ≤ 0.05) | kept (30B rows) |
| telemetry: the shared objects' telemetry gathered on the device and copied to the host once | bitwise | — / 0.001 ms (0.04%) / — | dropped |

- **Per step, core-issued:** 25 kernels + 2 D2D + 6 H2D + 1 D2H → 5 kernels + 1 H2D + 1 D2H (Qwen3-30B-A3B and
  Qwen3-8B alike). What is left: the last tokens' gather and write into the inputs, the SeqTable start write, the
  store's two writes, one upload, the tokens' copy back.
- **Totals, decode:** −0.036 ms per step on the 5090 30B resident (2.44 → 2.40 ms, 1.5%), −0.040 at 47%, −0.035 on
  the 8B; −0.025 ms on the A5000. `holon_tg` on the 5090, 30B resident: 419.2 → 424.6 tok/s at 1K and 387.2 → 392.0 at
  4K (×1.013, −0.030 ms per token); the A5000 does not resolve it end to end (README).
- **Judge:** snapshots identical to bdd8422 for the 30B (resident and `*=0.47`) and the 8B on the A5000; the offload
  store identical to all-resident. A tiny-engine test checks the host-staged inputs against the device-derived ones at
  every step.

Decided while building:
- **The condition is static: no drafter.** Only verify rows make a start inexact, and only a drafter makes verify
  rows; the executor raises if a step without one still brings an inexact row. Engines with a drafter keep the device
  derivation unchanged (`Executor._derive`), so graphs need no new key. The SeqTable start write stays in both paths:
  a plugin may read `num_computed` during a step.
- **The first inputs by a scratch element:** each row's device-held token is gathered from `last_token` and written at
  its first input, or, for a row whose first input is the host's, at one element past the step's tokens. Two kernels
  instead of four, the same in both paths.
- **The head reads `h` itself** when every token is a logits row (graphed steps always): `rmsnorm@1` returns a new
  tensor and, as its contract now states (the panel's finding; L0 checks both providers), never writes `x`, so a
  drafter or prompt logprobs reading `h` after it see it unchanged.
- **Two arenas, not one compact layout:** the small arena has fixed offsets like the large one (graphs bake its
  addresses); packing each step's fields by size would have made the addresses depend on what a step writes. It
  repeats the fields that do not scale with tokens (block-table rows, masks): ≈ 8 MB per page group at 256 sequences
  of 128K tokens, once on the device and twice pinned, allocated before `state_memory` reads the free memory.
- **Found on the way:** the A/B saves about 1 µs per kernel removed, far less than the inventory's profiled µs
  (node tracing inflates small kernels); the A5000's rows drift ≈ 2% over a run as the card warms, so it resolves
  none of these changes and the verdicts come from a 5090 desktop.
- **Not built** (README "Not done"): the telemetry copy (dropped); std's per-step kernels — the greedy sampler's unused
  parameter gather, the gqa planner's write unit / offset (a `Pages` field would be an ABI change), FlashInfer's plan
  copy and in-graph `prepare`; the tokens' copy inside the graph (one graph per output buffer).
- **Also (std, `cdd4785`):** the ffn unit interface's optional member is `add_residual(x, residual)` (was `add`;
  `moe_combine@1`'s instance `add` is unchanged), and the `linear_experts@2` / `linear_experts_arena@1` contract text
  says a prepared handle exposes `.x`, the tensor it stands for, for consumers that do not know it; the panel found
  the in-tree experts providers did not honour it (ggml's took only its own `Prepared`, the reference a tensor), so
  both now read any other handle's `.x` (`7305168`, tested on both). `_Prep.host_staged` names the host-staged path
  explicitly (`d73a2a2`).
### Step 6 as built (2026-10-17): F5b — kernels.cpu_task, kernels.ggml_cpu, the CPU's share, the path-aware floor

Order as run: step 6 before step 5 (§9). Built:
- **`kernels.cpu_task`** (`[build]`, D6): S6's protocol — a word per slot (one per MoE layer) in a mapped host pool the
  worker requests in `before_materialize`, spin (1-thread kernels) or memop waits (`cpu_task.wait`), a persistent
  coordinator plus a spinning pool per kind of work (`compute`; `io`, step 5's), pinned to `cpu_task.cores`
  (`cpu_task.io_cores`; a list is a pool, its first `threads` taken; default one per physical core past the first),
  and the task ABI (`csrc/holon_cpu_task.h`: a task runs on every thread, meeting at the pool's barrier). Fail-open: a
  task error answers FAILED; the spin wait gives up after `cpu_task.timeout` (60 s); the memop flavour's watchdog
  abandons an unanswered request; the CPU's and the GPU's flag words (one writer each) are sticky, every later wait
  returns at once and the coordinator answers without running; any answer but DONE ORs bit 1 into the error word the
  store checks at commit (bit 2: a timeout). A stop after the drain leaves a coordinator still in a task past the bound
  behind (its state and buffers leaked on purpose), so `close` always ends.
- **`kernels.ggml_cpu`** (`[build]`, D7): ggml-cpu's quantizers and dot products vendored unmodified at
  **kernels.ggml_cuda's pin** (cbe28bb3: §13's "unless step 6 aligns it" — aligned), a shim for ggml-base's few symbols;
  one module per CPU tier (`x86-avx2`, `x86-avx512`; `Constraint.cpu_isa`, so F5b locks carry the tier), only the
  vendored C taking the tier's flags (the loader and the f32 SwiGLU are baseline x86-64: a module loads on any host and
  rounds its SwiGLU alike on every tier). `cpu_experts@1`: a layer's jobs over GGUF blocks in place, rows split across
  threads (work stealing), never K; the Q8_K- and Q8_0-dot types. L0 (A5000, `x86-avx2`): within the oracle's budget
  (the oracle emulates Q8_K / Q8_0 exactly; a SwiGLU value at a rounding tie may flip, so 9 rows in 10 within 1e-4 of
  the row RMS, every row within one flip, 2e-2), a token bitwise alone vs grouped at 1, 7 and 40 threads on synthetic
  Q8_0 / Q4_0 and the vehicle's Q4_K / Q6_K, Q4_K / Q4_K and UD-IQ3_XXS classes; the K-split mutation caught.
- **`moe_resolve@1`** takes `plan` and `jobs` (one change; the variants keep its terms): of a call's n CPU candidates
  (missing in every role, in the read tier) the CPU takes min(plan[n], n), **the lowest-ranked** (best routing rank,
  then id: Strata's rule), never fetched nor admitted (they count), `routed |= 4`, their entries on null slots; the job
  table in mapped memory; `cpu [T, k]` out; a share past the table flags JOBS. **`moe_patch_rows@1`**: `publish` the CPU
  tokens' inputs, then the CPU's rows into the experts' output before the combine. CUDA (kernels.expert_io), oracles and
  CPU providers agree exactly.
- **`std.moe.cpu.CpuShare`** (the shared object of F5b; options `moe.cpu_plan`, `moe.cpu_isa`): per pass, resolve →
  publish; signal → fetch → GEMMs → wait → patch → combine (the signal before the fetch: the CPU's jobs overlap the
  PCIe reads). The table spells plan[1…N]; past N the last entry's share.
- **Path-aware floor** (§6): `q8_1.py`'s `cpu` path (ggml-cpu's Q8_K / Q8_0 activations, exactly, no stored-sum
  correction), `gguf.q8_1_experts(split)` applying F5b's rule per token to the frozen placement
  (`placement.cpu_split`, which requires one token per step and records it), `golden --cpu-split`, the split's digest
  in the candidate's recording and compare's check that floor and candidate share it.
- **`holon_tools tune --cpu-split [--no-lock] [--json]`**: F5a's decode rate, then the worker's threads (2, 4, 6) at
  half the misses, then the share (0.25 … 1.0) at the best count; the options (`moe.cpu_plan` spelled to k × the
  graphed tokens, `moe.cpu_isa`, `cpu_task.threads`) enter the lock only at a ≥ 5% gain (D4), else F5a stays. On the
  RTX 5080 step 4 runs the same command.
- Plants `cpu-wait-skipped` (the first layer's patch skips its wait: 1 of 48 layers, § 6's ≈ 1%) and
  `cpu-gate-up-order` (kernels.ggml_cpu's plant bit).

Evidence (`spikes/x6_cpu/README.md`, `evidence/x6_cpu_rtx-a5000_2026-10-17.json`), RTX A5000, the vehicle:
- **F5b's gain** (tune, batch 1, 25% cached, F11, natural prompts, 6 workers on one NUMA node): F5a 68.1 tok/s; at half
  the misses 83.7 / 93.5 / 93.6 with 2 / 4 / 6 threads; shares 0.25 / 0.4 / 0.5 / 0.6 / 0.75 / 1.0: 76.8 / 79.6 /
  93.6 / 94.9 / 105.3 / 52.4 — **+55% at 0.75** (`1,2,2,3,4,5,5,6`), F5b taken (a first run without the 1.0 arm: 67.8
  → 103.9, +53%, the same pick). The A5000's PCIe 4.0 gather (≈ 26 GB/s) is half the
  5080's, so the CPU takes more here than M2's n / 2. A separate A/B (`decode.py`, share 0.5, 4 threads): 78.2 → 94.8
  tok/s.
- **Exactness of F5a unchanged**: the snapshot trace's F11 at 25% and 47%, and F1 at 25%, byte-identical to
  `std.moe@1` all-resident, F11's parity with its replay at every step.
- **F5b, adaptive placement on** (F11 at 25%, the tuned split): run-identical ×3 (6,048 handshakes, 92,134 CPU
  experts per run, flag 0), F11's residency its replay's with the CPU's misses excluded.
- **F5b bounded L2** (F1 frozen at 25%, the tuned split, one token per step, 1.58 M handshakes, 32,783 positions):
  **PASS** against the path-aware floor — top-1 98.54% (≥ 98.14%), KL p50 1.26e-4 (≤ 2.49e-4), p99 3.92e-2 (≤ 8.60e-2),
  ΔPPL −0.08% ± 0.09% vs the floor. The floor's KL, computed on the CPU before the GPU run: p50 1.25e-4 / p99 4.30e-2
  against the plain mmvq floor's 1.18e-4 / 3.98e-2 (1.06× / 1.08×: Q8_K's larger error reaches the CPU's share only).
- **Plants**: `cpu-wait-skipped` fails run-identical (2 runs differ from each other from step 3; fired 36 eager + 30
  captured patches per run) — the race, as §6 predicted; `cpu-gate-up-order` fails L2 (top-1 62.7%, KL p99 12.8,
  ΔPPL +473%; fired in all 48 layers' CPU expert instances), and F5a configurations build none (it cannot fire there).

Decided while building, against the text above:
- **The split table is a share of the misses** (plan[n] = round(share · n)) measured as one parameter by end-to-end
  decode, not each entry by M2's microbenchmark: M2 found the optimum flat within ±1 (1–3%), and the decode rate is
  the quantity the 5% rule judges.
- **The share is the pass's size, not its row kind**: a pass of at most a graphed step's tokens takes it — a decode
  step of any batch, and a prompt chunk as short (which is how the teacher-forced L2 at one token per step measures the
  split; deployment-level, D4). F8 (a share for larger prefill chunks) is not built.
- **The CPU candidates are the misses held in RAM** (tier −1): with step 5's SSD tier, plan[n] counts those (D4's "n
  distinct misses" read as the CPU-eligible ones).
- **Tier binaries**: the AVX-512 module could not be run on this host (an AVX2 CPU); its L0 runs on an AVX-512 host.

### Step 5 as built (2026-10-17): F9, the exclusive RAM tier, F11 with deferred admission

Built after step 6 (§9's order; an SSD miss is a `cpu_task@1` copy task):
- **The Tier protocol owns its storage** (panel B): `holds(cls, pieces)` (known at the bind), `prepare(ctx, cls,
  unit_bytes, extents, staging, granule)` — a device-readable tier requests its mapped host pool (units of the store's
  granule, so MemOps still copy whole units) of `holds` pieces plus `staging` units where slower reads land, and fills
  it when it holds every piece —, `units(cls)` (that pool), `read(cls, piece) → descriptor` (`("host_slot", unit)`, or
  `("pread", ((fd, direct, offset, nbytes), …))`). `std.tiered.pinned@1` ported (`moe.ram` caps it per class, pinned
  as `moe.cache` is: fewer than every piece makes RAM exclusive); **`std.tiered.gguf_direct@1`** (F9): every piece
  where the checkpoint holds it, `O_DIRECT` descriptors (buffered where the filesystem has none).
- **`cpu_task@1`'s copy task**: per role the requested pieces' extents read into their units by the I/O worker
  (`cpu_task.io_threads`, 2 by M7: two blocking reads in flight), an `O_DIRECT` extent through its aligned superset and
  a per-thread bounce buffer; a bad request or a short read fails the task (fail-open, the commit stops the engine).
- **`moe_tiers@1`** (std.moe; CUDA in kernels.expert_io, oracle, CPU provider; L0 exact): one CTA after a layer
  call's resolve keeps RAM exclusive, D10's replayed rules in order — each admission (F11's pairing, unchanged) is
  **staged**, its victim **demoted** after the gather (into the admitted expert's RAM units when it came from RAM, else
  a free unit of each class, else the units of RAM's coldest expert of its classes when strictly colder — displaced to
  the SSD —, else the SSD), then the staged copy **moved** into the victim's slots; a decode SSD miss not admitted is
  **kept** in RAM the same way; every SSD read is listed (its RAM unit if kept, else a staging unit). Heat is the
  policy's declared state (`heat`: F11's score). **The flip lands in the same call**: staged → demoted → moved, all
  stream-ordered before any later reader (the next pass or step), so D10's "+1 where the copy runs on the step stream"
  holds with no host lag. A unit an admission frees is reusable from the next call (the gather still reads it).
- **`moe_resolve@1`'s arguments are unchanged by F9** (panel B asked for one change): the tiers' work is its own op
  after the resolve; the resolve only names a piece no device-readable tier holds by its fetch source −2 − e. The
  variants (`lfu`, `lru`, a held-out F10) keep their contracts.
- **The store under exclusive tiers**: per call resolve → [the CPU's share] → `moe_tiers` → the reads (signal, wait on
  the I/O worker) → fetch → demote → move; the seed reads VRAM's pieces from the file through the staging units and
  fills RAM by F2's order past VRAM (units reloaded where the device's owner table says they hold something else, so
  the startup profile's re-materialize reloads nothing it need not); F5b's CPU candidates are the RAM-held misses.

Evidence (`spikes/x6_cpu/README.md`), RTX A5000, the vehicle at 25% cached, the checkpoint on a network filesystem
(not a local NVMe):
- **Exact, RAM-capped**: `host_memory` 12 GiB (below the experts' 17.6 GB), RAM holding half the pieces: the snapshot
  trace byte-identical to `std.moe@1` all-resident under F11 (2,115 admissions, the replay's residency at every step,
  the device's map the mirror's) and under F1. The CPU and GPU engine tests run lfu and static at a quarter in RAM,
  bitwise resident, and F5b with exclusive tiers (both workers clean, bounded).
- **Decode** (batch 1, 512 tokens, the prompt set's first prompt, ×2 interleaved arms of 3 rounds): RAM holding every
  expert 70.9 / 71.2 tok/s; RAM half (12 GiB) 18.6 / 17.5 (0.25×); RAM 37.5% (10 GiB) 11.3 / 11.6 (0.16×) — RAM-short
  is a fit mode (D10's 1.9–8.3×), here bound by the file's network reads. **F11 stays the default where RAM holds every
  expert** (the default `moe.tiers` is the pinned tier alone).

Decided while building, against the text above:
- **No lending and no ring under exclusive tiers**: a lent expert would need a demotion first (not built); the store
  keeps a pass's staging (the vehicle: 128 slots per class, ≈ 0.5 GB of cache) and prefill stages its misses, SSD ones
  through the reads. RAM-short prefill is slow by design.
- **Policies**: exclusive tiers take a device-owned policy naming its heat (F11) or a frozen one (`frozen`: F1); F4
  (no heat) and `profile_decay` (host swaps would copy from RAM units the graph moves) are refused at startup.
- **Heat across layers**: RAM's coldest is compared on F11's scores of every layer, which within a decode step are one
  decay apart before and after the calling layer (deterministic; a small skew, recorded).
- **The SSD tier's evidence is a network filesystem's**: M7's local NVMe rates (2.6 GB/s) are not re-measured here.
- **One descriptor kind** (panel B, after the step): a file tier's `read` answers `pread` extents only, each with its
  own O_DIRECT flag (the copy task reads that and nothing else; the store refuses another kind at prepare), so a
  page-cache tier is a plugin-only Tier (O_DIRECT off). The RAM tier's per-class units are the store's setting `ram`,
  forwarded from std.moe's `moe.ram` (std.tiered names no MoE option).

After merging steps 3c and 3d (the router-prepared q8_1 input and the fused residual add), the snapshot trace was
re-run on the A5000: F11 at 25% and the RAM-capped F11 (half the pieces in RAM, 12 GiB) byte-identical to the new
all-resident, F5b run-identical ×2 (the full suite: 1,445 passed).
